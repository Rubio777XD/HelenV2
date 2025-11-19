"""TensorFlow sequence gesture classifier compatible con el contrato de HELEN.

Este envoltorio adapta el modelo LSTM de video al contrato existente
``predict(features) -> Prediction`` usado por el backend. Mantiene una ventana
deslizante de 96 fotogramas con 126 características (21 landmarks × 3 coords ×
2 manos) y solo dispara el modelo cuando la secuencia está completa. Las
primeras predicciones retornan una clase neutra para que la ``DecisionEngine``
conserve la misma lógica de consenso que el backend histórico.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Iterable, List


LOGGER = logging.getLogger("helen.backend.tf")


class Prediction(tuple):
    """Minimal prediction tuple used por el backend."""

    __slots__ = ()
    _fields = ("label", "score")

    def __new__(cls, label: str, score: float):  # type: ignore[override]
        return super().__new__(cls, (label, float(score)))

    @property
    def label(self) -> str:
        return self[0]

    @property
    def score(self) -> float:
        return float(self[1])


class TensorFlowSequenceGestureClassifier:
    """Wrap the TensorFlow LSTM model maintaining HELEN's predict contract.

    The model consumes sequences shaped as ``(sequence_length, feature_dim)``
    with ``sequence_length=96`` and ``feature_dim=126`` (21 landmarks × 3
    coordinates × 2 hands). The internal buffer keeps a sliding window of the
    last ``sequence_length`` frames and triggers inference only when the window
    is full.

    HELEN currently emits 42 features per frame (x/y only, single hand). To
    bridge both worlds we expand the 42-D vector to 126 dimensions by adding
    ``z=0`` for every landmark and duplicating the hand coordinates into the
    second hand slot. If the capture pipeline already provides 126 features the
    frame is passed through untouched. This makes the migration to real dual
    hand 3D landmarks explicit and documented.
    """

    source = "tensorflow_sequence"
    # Mantener sincronizado con ``MODEL_LABEL_ALIASES`` en ``backendHelen/server.py``
    # para que las predicciones del LSTM se alineen con la DecisionEngine.
    _LABEL_NORMALIZATION = {
        "activar": "Start",
        "start": "Start",
        "wake": "Start",
        "clima": "Clima",
        "weather": "Clima",
        "reloj": "Reloj",
        "clock": "Reloj",
        "home": "Inicio",
        "inicio": "Inicio",
        "configuracion": "Ajustes",
        "ajustes": "Ajustes",
        "dispositivos": "Dispositivos",
        "devices": "Dispositivos",
        "tutorial": "Tutorial",
        "alarma": "Alarma",
        "foco": "Foco",
    }

    _DEFAULT_LABELS = {
        0: "Start",
        1: "Clima",
        2: "Reloj",
        3: "Inicio",
        4: "Ajustes",
        5: "Dispositivos",
        6: "Tutorial",
        7: "Alarma",
        8: "Foco",
    }

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
            try:
                with candidate.open("r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                return {int(idx): self._canonical_label(label) for label, idx in raw.items()}
            except Exception as error:
                LOGGER.warning("labels.json corrupto en %s: %s", candidate, error)

        LOGGER.warning("labels.json no encontrado en %s, usando etiquetas por defecto", candidate.parent)
        return dict(self._DEFAULT_LABELS)

    # ------------------------------------------------------------------
    def _canonical_label(self, label: str) -> str:
        text = str(label or "").strip()
        lowered = text.lower()
        normalized = self._LABEL_NORMALIZATION.get(lowered)
        if normalized:
            return normalized

        return text.title() if text else text

    # ------------------------------------------------------------------
    def _convert_helen_features_to_model_frame(self, features: Iterable[float]):
        """Map HELEN frames into the 126-D format expected by the LSTM."""

        np = self._np
        values = np.asarray(list(features), dtype=np.float32)
        if values.size == self.feature_dim:
            return values

        if values.size != 42:
            raise ValueError(f"Se esperaban 42 o {self.feature_dim} features por frame, recibido {values.size}")

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

        try:
            with self._lock:
                probabilities = self._predict_fn(batch)
        except Exception as error:  # pragma: no cover - runtime safety net
            LOGGER.error("Fallo en inferencia TensorFlow: %s", error)
            return Prediction(label=self._labels.get(0, "Start"), score=0.0)

        # Normalise output shape to (num_classes,)
        if probabilities.ndim >= 2:
            probs = probabilities[0]
        else:
            probs = probabilities

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx]) if probs.size else 0.0
        label = self._labels.get(best_idx, str(best_idx))
        return Prediction(label=str(label), score=confidence)


class DummyGestureClassifier:
    """Clasificador neutral para mantener vivo el backend sin modelo real."""

    source = "dummy"

    def __init__(self, neutral_label: str = "none") -> None:
        self._neutral_label = neutral_label

    def predict(self, features: Iterable[float]) -> Prediction:  # noqa: ARG002 - interfaz establecida
        return Prediction(label=self._neutral_label, score=0.0)


__all__ = [
    "TensorFlowSequenceGestureClassifier",
    "Prediction",
    "DummyGestureClassifier",
]
