"""High-level orchestration to feed gesture predictions to web clients."""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, Iterator, List, Optional

import cv2
import mediapipe as mp
import numpy as np

from video_gesture_model import config as model_config
from video_gesture_model.extract_landmarks import normalise_landmarks
from video_gesture_model.realtime_inference import (
    build_predict_fn,
    load_label_map,
)

from . import config


class GestureInferenceService:
    """Run the existing TensorFlow model and broadcast gestures to observers."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        camera_index: int = config.DEFAULT_CAMERA_INDEX,
        confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        sequence_length: Optional[int] = config.DEFAULT_SEQUENCE_LENGTH,
        prediction_cooldown_s: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.sequence_length = sequence_length or model_config.SEQUENCE_LENGTH
        self.prediction_cooldown_s = prediction_cooldown_s

        self._listeners: List[Queue] = []
        self._listeners_lock = threading.Lock()
        self._current_prediction: Optional[Dict[str, float]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        self._label_map: Optional[Dict[int, str]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the capture thread if it is not running."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._run, name="gesture-inference", daemon=True)
        self._thread.start()
        # Wait until the capture loop finishes initialisation to avoid race conditions
        self._ready_event.wait(timeout=10)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def label_map(self) -> Dict[int, str]:
        if self._label_map is None:
            raise RuntimeError("Model not initialised yet; call start() first to load labels.")
        return self._label_map

    def latest_prediction(self) -> Optional[Dict[str, float]]:
        return self._current_prediction

    def subscribe(self, max_queue: int = 32) -> Queue:
        """Register a listener queue that will receive prediction dictionaries."""
        q: Queue = Queue(maxsize=max_queue)
        with self._listeners_lock:
            self._listeners.append(q)
        return q

    def unsubscribe(self, q: Queue) -> None:
        with self._listeners_lock:
            if q in self._listeners:
                self._listeners.remove(q)

    def iter_predictions(self) -> Iterator[Dict[str, float]]:
        """Convenience generator yielding predictions for the caller."""
        queue = self.subscribe()
        try:
            while True:
                try:
                    item = queue.get(timeout=1.0)
                except Empty:
                    if not self.running() and self._stop_event.is_set():
                        break
                else:
                    yield item
        finally:
            self.unsubscribe(queue)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_model_and_labels(self) -> None:
        model_dir_or_file = self._resolve_model_path()
        self._predict_fn = build_predict_fn(model_dir_or_file)

        labels_path = self.labels_path or (model_dir_or_file / "labels.json")
        if not labels_path.exists():
            raise FileNotFoundError(
                f"No se encontró labels.json en {labels_path}. Ajusta el parámetro labels_path."
            )
        self._label_map = load_label_map(labels_path)

    def _resolve_model_path(self) -> Path:
        if self.model_path is not None:
            return self.model_path

        # Auto-discover most recent SavedModel directory
        candidates = sorted(config.DEFAULT_MODELS_DIR.glob("gesture_model_*/"))
        if not candidates:
            raise FileNotFoundError(
                "No se encontraron modelos entrenados en la carpeta por defecto. "
                "Proporciona model_path explícitamente."
            )
        return candidates[-1]

    def _notify_listeners(self, payload: Dict[str, float]) -> None:
        with self._listeners_lock:
            for queue in list(self._listeners):
                try:
                    queue.put_nowait(payload)
                except Exception:
                    # Ignore full queues from slow consumers
                    pass

    def _run(self) -> None:
        try:
            self._load_model_and_labels()
        except Exception as exc:  # pragma: no cover - initialisation errors bubble up
            self._ready_event.set()
            raise exc

        predict_fn = self._predict_fn
        idx_to_label = self._label_map

        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=model_config.MAX_HANDS,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, model_config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, model_config.FRAME_HEIGHT)

        buffer = deque(maxlen=self.sequence_length)

        self._ready_event.set()

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                frame_features = np.zeros(
                    (
                        model_config.MAX_HANDS,
                        model_config.NUM_HAND_LANDMARKS,
                        model_config.LANDMARK_DIM,
                    ),
                    dtype=np.float32,
                )

                if results.multi_hand_landmarks and results.multi_handedness:
                    ordering = {"Left": 0, "Right": 1}
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks, results.multi_handedness
                    ):
                        label = handedness.classification[0].label
                        idx = ordering.get(label, 0)
                        coords = np.array(
                            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                            dtype=np.float32,
                        )
                        frame_features[idx] = coords

                buffer.append(normalise_landmarks(frame_features.flatten()))

                if len(buffer) < self.sequence_length:
                    continue

                input_tensor = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)
                probabilities = predict_fn(input_tensor)[0]
                pred_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_idx])

                if confidence < self.confidence_threshold:
                    continue

                label = idx_to_label.get(pred_idx, "?")
                payload = {
                    "label": label,
                    "confidence": confidence,
                    "timestamp": time.time(),
                    "index": pred_idx,
                }

                # Debounce identical predictions within the cooldown window
                if self._current_prediction and self._current_prediction["label"] == label:
                    elapsed = payload["timestamp"] - self._current_prediction.get("timestamp", 0)
                    if elapsed < self.prediction_cooldown_s:
                        continue

                self._current_prediction = payload
                self._notify_listeners(payload)
        finally:
            cap.release()
            hands.close()

    # For introspection in JSON
    def snapshot(self) -> Dict[str, object]:
        return {
            "running": self.running(),
            "model_path": str(self.model_path) if self.model_path else None,
            "sequence_length": self.sequence_length,
            "confidence_threshold": self.confidence_threshold,
            "latest_prediction": self._current_prediction,
            "gestures": list(self._label_map.values()) if self._label_map else [],
        }
