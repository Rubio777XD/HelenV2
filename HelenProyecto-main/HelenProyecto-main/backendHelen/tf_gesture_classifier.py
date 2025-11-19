"""TensorFlow sequence gesture classifier compatible with HELEN's contract.

This wrapper adapts the LSTM-based video gesture model to the existing
``predict(features) -> Prediction`` interface used by the backend. It keeps a
fixed-length buffer of frames and only triggers the TensorFlow model when the
sequence is complete.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from Hellen_model_RN.helpers import labels_dict
from Hellen_model_RN.simple_classifier import Prediction


class TensorFlowSequenceGestureClassifier:
    """Wrap the TensorFlow LSTM model maintaining HELEN's predict contract.

    The model consumes sequences shaped as ``(sequence_length, feature_dim)``
    with ``sequence_length=96`` and ``feature_dim=126`` (21 landmarks × 3
    coordinates × 2 hands). HELEN currently emits 42 features per frame (x/y
    only, single hand). To bridge both worlds we expand the 42-D vector to 126
    dimensions by adding ``z=0`` for every landmark and duplicating the hand
    coordinates into the second hand slot. This is a temporary compatibility
    shim until the capture pipeline emits full 3D landmarks for both hands.
    """

    source = "tensorflow_sequence"

    def __init__(self, model_path: Path, *, sequence_length: int = 96, feature_dim: int = 126) -> None:
        try:
            import numpy as np  # type: ignore
            import tensorflow as tf  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("TensorFlow y NumPy son requeridos para el clasificador LSTM") from exc

        self._np = np
        self._tf = tf
        self.sequence_length = int(sequence_length)
        self.feature_dim = int(feature_dim)
        self._buffer: deque = deque(maxlen=self.sequence_length)
        self._lock = threading.Lock()

        self._predict_fn = self._build_predict_fn(Path(model_path))
        self._labels = self._load_labels(Path(model_path))

    # ------------------------------------------------------------------
    def _build_predict_fn(self, model_path: Path) -> Callable[[List[List[float]]], "np.ndarray"]:
        """Create a unified predict function for SavedModel or Keras files."""

        tf = self._tf

        if model_path.is_dir() and (model_path / "saved_model.pb").exists():
            saved = tf.saved_model.load(str(model_path))
            signature = saved.signatures.get("serve") or saved.signatures.get("serving_default")
            if signature is None:
                raise RuntimeError("El SavedModel no expone una firma de inferencia compatible")

            input_name = next(iter(signature.structured_input_signature[1].keys()))
            output_name = next(iter(signature.structured_outputs.keys()))

            def predict(batch: List[List[float]]):
                outputs = signature(**{input_name: tf.constant(batch)})[output_name]
                return outputs.numpy()

            return predict

        # Fallback to keras/weights file loading.
        model = tf.keras.models.load_model(str(model_path))

        def predict(batch: List[List[float]]):
            return model.predict(batch, verbose=0)

        return predict

    # ------------------------------------------------------------------
    def _load_labels(self, model_path: Path) -> Dict[int, str]:
        """Load idx→label mapping from labels.json, fallback to legacy labels."""

        candidate: Path
        if model_path.is_dir():
            candidate = model_path / "labels.json"
        else:
            candidate = model_path.parent / "labels.json"

        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as fp:
                raw = json.load(fp)
            # Training artifacts store gesture→idx; invert to idx→gesture.
            return {int(idx): label for label, idx in raw.items()}

        # Fallback: reuse legacy labels_dict to keep the pipeline usable even if
        # labels.json is missing or corrupted.
        return {int(idx): value for idx, value in labels_dict.items()}

    # ------------------------------------------------------------------
    def _convert_helen_features_to_model_frame(self, features_42: Iterable[float]):
        """Expand HELEN's 42-D frame into the 126-D format expected by the LSTM.

        Strategy (compatibility shim): assume a single hand is present, inject
        ``z=0`` for every landmark and duplicate the hand into the second hand
        slot. This preserves the spatial layout while the capture pipeline is
        upgraded to emit full 3D coordinates for both hands.
        """

        np = self._np
        values = np.asarray(list(features_42), dtype=np.float32)
        if values.size != 42:
            raise ValueError(f"Se esperaban 42 features por frame, recibido {values.size}")

        frame = np.zeros(self.feature_dim, dtype=np.float32)
        per_hand = self.feature_dim // 2  # 63 values per hand

        for landmark_idx in range(21):
            x = values[2 * landmark_idx]
            y = values[2 * landmark_idx + 1]

            base_left = landmark_idx * 3
            frame[base_left] = x
            frame[base_left + 1] = y
            frame[base_left + 2] = 0.0  # z placeholder

            base_right = per_hand + landmark_idx * 3
            frame[base_right] = x
            frame[base_right + 1] = y
            frame[base_right + 2] = 0.0  # z placeholder

        return frame

    # ------------------------------------------------------------------
    def predict(self, features: Iterable[float]) -> Prediction:
        frame = self._convert_helen_features_to_model_frame(features)
        self._buffer.append(frame)

        # The LSTM requires a full sequence. Emit a neutral prediction while the
        # buffer is filling up; the DecisionEngine will handle stability.
        if len(self._buffer) < self.sequence_length:
            neutral_label = self._labels.get(0, "Start")
            return Prediction(label=str(neutral_label), score=0.0)

        np = self._np
        sequence = np.array(self._buffer, dtype=np.float32)
        if sequence.shape != (self.sequence_length, self.feature_dim):
            raise ValueError(
                f"Secuencia con forma inesperada {sequence.shape}, se esperaba"
                f" ({self.sequence_length}, {self.feature_dim})"
            )

        batch = np.expand_dims(sequence, axis=0)

        with self._lock:
            probabilities = self._predict_fn(batch)

        # Normalise output shape to (num_classes,)
        if probabilities.ndim >= 2:
            probs = probabilities[0]
        else:
            probs = probabilities

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx]) if probs.size else 0.0
        label = self._labels.get(best_idx, str(best_idx))
        return Prediction(label=str(label), score=confidence)


__all__ = ["TensorFlowSequenceGestureClassifier"]
