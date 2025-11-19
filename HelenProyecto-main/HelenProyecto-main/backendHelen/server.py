"""HELEN backend exposing Server-Sent Events for gesture navigation.

The original repository relied on several disparate scripts to run the camera
pipeline, invoke the machine-learning model and broadcast results to the
frontend.  This module consolidates that behaviour in a single HTTP server that
is production-ready:

* Usa únicamente el clasificador LSTM de video basado en TensorFlow.
* Evita depender del modelo XGBoost legacy y recurre a un clasificador dummy
  solo si el SavedModel está ausente o corrupto.
* Serves the static frontend from the ``helen/`` directory so the packaged
  application works out of the box.
* Exposes a comprehensive ``/health`` endpoint that reports the state of the
  model, camera, pipeline and SSE clients.

The module is intentionally self-contained so it can be bundled with
PyInstaller and launched both from source and from frozen executables.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import platform
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import statistics
from xml.sax.saxutils import escape

try:  # pragma: no cover - optional dependency in CI
    import cv2  # type: ignore
except Exception:  # pragma: no cover - handled gracefully at runtime
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency in CI
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover - handled gracefully at runtime
    mp = None  # type: ignore

from .tf_gesture_classifier import DummyGestureClassifier, Prediction, TensorFlowSequenceGestureClassifier
from . import camera_probe

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from .camera_probe import CameraSelection

# Mapa de etiquetas legacy conservado vacío para evitar depender del paquete
# Hellen_model_RN. Las clases XGBoost están obsoletas y no se instancian en
# runtime.
labels_dict: Dict[int, str] = {}


LOGGER = logging.getLogger("helen.backend")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

DECISION_DEBUG = bool(int(os.environ.get("HELEN_DEBUG_DECISION", "0") or 0))

with contextlib.suppress(Exception):
    import absl.logging as absl_logging  # type: ignore

    absl_logging.set_verbosity(absl_logging.WARNING)
    handler = absl_logging.get_absl_handler()
    handler.setLevel(logging.WARNING)


def _read_os_release() -> str:
    path = Path("/etc/os-release")
    if not path.exists():
        return ""
    content = {}
    with contextlib.suppress(OSError, UnicodeDecodeError):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            content[key] = value.strip().strip('"')
    name = content.get("PRETTY_NAME") or content.get("NAME") or ""
    version = content.get("VERSION_ID") or content.get("VERSION") or ""
    return f"{name} {version}".strip()


def _log_vision_runtime_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "arch": platform.machine(),
        "os_release": _read_os_release(),
        "mediapipe": {"status": "missing"},
        "opencv": {"status": "missing"},
        "numpy": {"status": "missing"},
        "notes": [],
    }

    if mp is None:
        snapshot["mediapipe"] = {
            "status": "error",
            "message": "ImportError",
            "suggestion": "Reinstala mediapipe==0.10.18 dentro del entorno .venv.",
        }
    else:
        version = getattr(mp, "__version__", "unknown")
        snapshot["mediapipe"] = {
            "status": "ok",
            "version": version,
        }
        with contextlib.suppress(Exception):  # pragma: no cover - smoke import already executed in setup
            with mp.solutions.hands.Hands():
                snapshot["mediapipe"]["hands"] = "initialised"

    if cv2 is None:
        snapshot["opencv"] = {
            "status": "error",
            "message": "ImportError",
            "suggestion": "Instala opencv-python==4.9.0.80 dentro del entorno .venv.",
        }
    else:
        build_info = ""
        with contextlib.suppress(Exception):
            build_info = cv2.getBuildInformation()
        snapshot["opencv"] = {
            "status": "ok",
            "version": getattr(cv2, "__version__", "unknown"),
            "gstreamer": "YES" if "GStreamer:                   YES" in build_info else "NO",
            "v4l2": "YES" if "V4L/V4L2:                  YES" in build_info else "NO",
        }

    with contextlib.suppress(Exception):  # pragma: no cover - optional dependency
        import numpy  # type: ignore

        snapshot["numpy"] = {
            "status": "ok",
            "version": numpy.__version__,
        }

    camera_probe.LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = camera_probe.LOG_DIR / f"vision-runtime-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with contextlib.suppress(OSError):
        path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    if snapshot["mediapipe"].get("status") != "ok" or snapshot["opencv"].get("status") != "ok":
        LOGGER.error("Stack de visión incompleto: %s", snapshot)
    else:
        LOGGER.info(
            "Stack MediaPipe/OpenCV disponible (mediapipe=%s, opencv=%s)",
            snapshot["mediapipe"].get("version"),
            snapshot["opencv"].get("version"),
        )

    return snapshot


def _debug_decision(message: str, *args: Any) -> None:
    if DECISION_DEBUG:
        LOGGER.debug(message, *args)


VISION_RUNTIME_SNAPSHOT = _log_vision_runtime_snapshot()


def _resolve_repo_root() -> Path:
    """Return the runtime root, compatible with PyInstaller bundles."""

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
FRONTEND_ROOT = REPO_ROOT / "helen"
TF_MODEL_BASE_DIR = REPO_ROOT / "Hellen_model_TF" / "video_gesture_model" / "data" / "models"

# port: int = 5000  # Referencia para pruebas de integración (mantener sincronizado con run()).


_MISSING_DATASET_NOTIFIED: set[Path] = set()


class RuntimeModelConfig:
    """Centralise model backend selection for the whole application.

    The repository now prioritises the TensorFlow/LSTM video classifier. Any
    request to use the legacy XGBoost backend is logged and automatically
    redirected to LSTM to keep the runtime stable.
    """

    ENV_VAR = "HELEN_MODEL_BACKEND"
    DEFAULT_BACKEND = "lstm"

    def __init__(self) -> None:
        requested = os.environ.get(self.ENV_VAR, self.DEFAULT_BACKEND)
        self.requested_backend = str(requested or "").strip().lower() or self.DEFAULT_BACKEND
        self.effective_backend = self._resolve_backend(self.requested_backend)

    def _resolve_backend(self, requested: str) -> str:
        if requested == "lstm":
            return "lstm"

        LOGGER.warning(
            "El backend %s está obsoleto; backend efectivo=lstm", requested or "<desconocido>"
        )
        return "lstm"

    def tf_model_dir(self, override: Optional[Path]) -> Path:
        if override:
            return Path(override)
        return _default_tf_model_dir()

    def announce(self) -> None:
        if self.requested_backend != self.effective_backend:
            LOGGER.warning(
                "HELEN_MODEL_BACKEND=%s fue solicitado pero se usará %s",
                self.requested_backend,
                self.effective_backend,
            )
        else:
            LOGGER.info("Backend de modelo seleccionado: %s", self.effective_backend)


RUNTIME_MODEL_CONFIG = RuntimeModelConfig()
LSTM_BACKEND_ACTIVE = RUNTIME_MODEL_CONFIG.effective_backend == "lstm"


def _default_tf_model_dir() -> Path:
    """Pick the newest TensorFlow SavedModel shipped with the repository."""

    if not TF_MODEL_BASE_DIR.exists():
        return TF_MODEL_BASE_DIR

    candidates = [
        path
        for path in TF_MODEL_BASE_DIR.iterdir()
        if path.is_dir() and (path / "saved_model.pb").exists()
    ]

    if not candidates:
        return TF_MODEL_BASE_DIR

    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


TRACKED_GESTURES = {"Start", "Clima", "Reloj", "Inicio"}
# Tabla de alias entre las etiquetas de labels.json (ej. "activar", "clima") y
# las etiquetas canónicas que consume la DecisionEngine (Start, Clima, Reloj,
# Inicio). Mantener sincronizada con ``_LABEL_NORMALIZATION`` en
# ``tf_gesture_classifier.py`` para que las predicciones LSTM y los reportes
# usen los mismos nombres.
MODEL_LABEL_ALIASES: Dict[str, str] = {
    "activar": "Start",
    "start": "Start",
    "wake": "Start",
    "clima": "Clima",
    "weather": "Clima",
    "reloj": "Reloj",
    "clock": "Reloj",
    "home": "Inicio",
    "inicio": "Inicio",
}
GESTURE_ALIASES = {"Start": "H", "Clima": "C", "Reloj": "R", "Inicio": "I"}

MODE_STORAGE_PATH = REPO_ROOT / "backendHelen" / "runtime_mode.json"


def _normalize_display_mode(value: Optional[str]) -> str:
    candidate = str(value or "").strip().lower()
    return "raspberry" if candidate == "raspberry" else "windows"


class DisplayModeStore:
    """Persist and retrieve the display mode selected by the operator."""

    def __init__(self, path: Path, default_mode: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._default_mode = _normalize_display_mode(default_mode)
        self._cached: Optional[str] = None

    def load(self) -> str:
        with self._lock:
            if self._cached:
                return self._cached

            try:
                text = self._path.read_text(encoding="utf-8")
            except FileNotFoundError:
                self._cached = self._default_mode
                return self._cached
            except Exception as error:
                LOGGER.warning("No se pudo leer el modo persistido (%s): %s", self._path, error)
                self._cached = self._default_mode
                return self._cached

            try:
                data = json.loads(text)
            except json.JSONDecodeError as error:
                LOGGER.warning("Archivo de modo inválido (%s): %s", self._path, error)
                self._cached = self._default_mode
                return self._cached

            mode = _normalize_display_mode(data.get("mode"))
            self._cached = mode
            return mode

    def save(self, mode: str) -> str:
        normalized = _normalize_display_mode(mode)
        payload = {"mode": normalized, "updated_at": time.time()}

        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            self._cached = normalized

        return normalized

    def cached(self) -> str:
        cached = self._cached
        if cached is not None:
            return cached
        return self.load()


def _profile_for_mode(mode: str) -> Optional[PiCameraProfile]:
    normalized = _normalize_display_mode(mode)
    return RASPBERRY_MODE_PROFILE if normalized == "raspberry" else None


@dataclass(frozen=True)
class ClassThreshold:
    enter: float
    release: float


LEGACY_CLIMA_THRESHOLD = ClassThreshold(enter=0.8, release=0.7)
UPDATED_CLIMA_THRESHOLD = ClassThreshold(enter=0.66, release=0.56)

DEFAULT_CLASS_THRESHOLDS: Dict[str, ClassThreshold] = {
    "Start": ClassThreshold(enter=0.45, release=0.35),
    "Clima": ClassThreshold(enter=0.40, release=0.30),
    "Reloj": ClassThreshold(enter=0.45, release=0.35),
    "Inicio": ClassThreshold(enter=0.45, release=0.35),
}

GLOBAL_MIN_SCORE = 0.3
DEFAULT_POLL_INTERVAL_S = 0.12


@dataclass(frozen=True)
class PiCameraProfile:
    model: str
    width: int
    height: int
    fps: int
    poll_interval: float
    process_every_n: int


@dataclass(frozen=True)
class PlatformRuntimeDefaults:
    detection_confidence: float
    tracking_confidence: float
    poll_interval: float
    frame_stride: int


@dataclass(frozen=True)
class ConsensusConfig:
    window_size: int = 3
    required_votes: int = 2


DEFAULT_CONSENSUS_CONFIG = ConsensusConfig()

CLIMA_CONSENSUS_OVERRIDE = ConsensusConfig(window_size=2, required_votes=1)

CLIMA_POST_START_DELAY = 0.4

ACTIVATION_DELAY = 0.8

SMOOTHING_WINDOW_SIZE = 4
COOLDOWN_SECONDS = ACTIVATION_DELAY
LISTENING_WINDOW_SECONDS = 4.0
COMMAND_DEBOUNCE_SECONDS = 0.75

QUALITY_MIN_LANDMARKS = 21
QUALITY_MIN_HAND_SCORE = 0.55
QUALITY_MIN_BBOX_AREA = 0.012
QUALITY_MIN_BBOX_SIDE = 0.09
QUALITY_BLUR_THRESHOLD = 35.0
QUALITY_EDGE_MARGIN = 0.015


class ConsensusVote(NamedTuple):
    label: str
    score: float
    timestamp: float


class ConsensusResult(NamedTuple):
    votes: int
    total: int
    average: float
    span_ms: float


class DecisionOutcome(NamedTuple):
    emit: bool
    label: str
    score: float
    payload: Dict[str, Any]
    reason: str
    state: str
    hint_label: Optional[str]
    support: int
    window_ms: float


@dataclass
class SampleRecord:
    timestamp: float
    label: str
    score: float
    accepted: bool
    reason: str
    state: str
    hint_label: Optional[str] = None
    support: int = 0
    window_ms: float = 0.0


@dataclass
class ThresholdSuggestion:
    label: str
    current: float
    recommended: float
    delta: float
    reason: str
    expected_f1: float


class FeatureNormalizer:
    """Apply the same normalisation used during model training when available."""

    def __init__(self, dataset_path: Optional[Path] = None) -> None:
        self._dataset_path = dataset_path
        self._transformer: Optional[Any] = None
        self._mean: Optional[List[float]] = None
        self._scale: Optional[List[float]] = None
        self._lock = threading.Lock()
        self._loaded = False
        self.reload_if_available()

    # ------------------------------------------------------------------
    def reload_if_available(self) -> None:
        path = self._dataset_path
        if path is None or not path.exists():
            return

        try:
            import pickle
        except ModuleNotFoundError:
            LOGGER.debug("pickle no disponible: no se puede cargar el normalizador")
            return

        try:
            with path.open("rb") as handle:
                data = pickle.load(handle)
        except Exception as exc:  # pragma: no cover - archivo corrupto
            LOGGER.warning("No se pudo cargar metadatos de %s: %s", path, exc)
            return

        transformer = data.get("normalizer") or data.get("scaler")
        mean = data.get("feature_mean") or data.get("mean_")
        scale = data.get("feature_std") or data.get("scale_") or data.get("feature_scale")

        # Liberar listas pesadas para evitar retener en memoria la copia completa del dataset.
        data.pop("data", None)
        data.pop("labels", None)

        with self._lock:
            self._transformer = transformer if hasattr(transformer, "transform") else None
            if self._transformer is None:
                self._mean = [float(v) for v in mean] if mean else None
                if scale:
                    self._scale = [float(v) if abs(float(v)) > 1e-6 else 1.0 for v in scale]
                else:
                    self._scale = None
            else:
                self._mean = None
                self._scale = None

            self._loaded = bool(self._transformer or (self._mean and self._scale))

    # ------------------------------------------------------------------
    def transform(self, features: Iterable[float]) -> List[float]:
        vector = [float(value) for value in features]

        with self._lock:
            transformer = self._transformer
            mean = self._mean
            scale = self._scale

        if transformer is not None:
            try:
                transformed = transformer.transform([vector])  # type: ignore[call-arg]
                return list(transformed[0])
            except Exception as exc:  # pragma: no cover - depende del artefacto
                LOGGER.warning("Normalizador entrenado falló, usando vector original: %s", exc)

        if mean and scale and len(mean) == len(vector) and len(scale) == len(vector):
            return [(value - mean[idx]) / (scale[idx] or 1.0) for idx, value in enumerate(vector)]

        return vector

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "dataset": str(self._dataset_path) if self._dataset_path else "<desactivado>",
                "loaded": self._loaded,
                "uses_transformer": bool(self._transformer),
                "uses_stats": bool(self._mean and self._scale),
            }


class ConsensusTracker:
    """Maintain a rolling window of predictions for temporal consensus."""

    def __init__(self, config: ConsensusConfig) -> None:
        self._config = config
        self._votes: Deque[ConsensusVote] = deque(maxlen=config.window_size)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self._votes.clear()

    # ------------------------------------------------------------------
    def add(self, label: str, score: float, timestamp: float) -> None:
        with self._lock:
            self._votes.append(ConsensusVote(label=label, score=score, timestamp=timestamp))

    # ------------------------------------------------------------------
    def evaluate(
        self,
        label: str,
        threshold: float,
        *,
        window_size: Optional[int] = None,
    ) -> ConsensusResult:
        with self._lock:
            votes = list(self._votes)

        if window_size is not None and window_size > 0:
            limit = max(1, int(window_size))
            votes = votes[-limit:]

        matching = [vote for vote in votes if vote.label == label]
        passing = [vote for vote in matching if vote.score >= threshold]

        average = float(sum(vote.score for vote in matching) / len(matching)) if matching else 0.0
        span_ms = (votes[-1].timestamp - votes[0].timestamp) * 1000.0 if len(votes) >= 2 else 0.0

        return ConsensusResult(votes=len(passing), total=len(votes), average=average, span_ms=span_ms)

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            votes = list(self._votes)

        return {
            "window_size": self._config.window_size,
            "required_votes": self._config.required_votes,
            "votes": [vote._asdict() for vote in votes],
        }


class GestureMetrics:
    """Aggregate per-session metrics for calibration and reporting."""

    def __init__(self) -> None:
        self._samples: List[SampleRecord] = []
        self._accepted_scores: Dict[str, List[float]] = defaultdict(list)
        self._rejected_scores: Dict[str, List[float]] = defaultdict(list)
        self._quality_checks = 0
        self._quality_rejections: Counter[str] = Counter()
        self._reason_counts: Counter[str] = Counter()
        self._reason_by_label: Dict[str, Counter[str]] = defaultdict(Counter)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(label: Optional[str]) -> str:
        return str(label or "").strip().lower()

    # ------------------------------------------------------------------
    @staticmethod
    def _canonical(label: Optional[str]) -> str:
        if not label:
            return ""
        text = str(label).strip()
        lowered = text.lower()
        alias = MODEL_LABEL_ALIASES.get(lowered)
        if alias:
            return alias
        for candidate in TRACKED_GESTURES:
            if lowered == candidate.lower():
                return candidate
        return text

    # ------------------------------------------------------------------
    @staticmethod
    def _score_stats(values: Iterable[float]) -> Dict[str, Optional[float]]:
        values = [float(v) for v in values if math.isfinite(float(v))]
        if not values:
            return {"min": None, "max": None, "mean": None, "median": None}

        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values) if hasattr(statistics, "fmean") else sum(values) / len(values),
            "median": statistics.median(values),
        }

    # ------------------------------------------------------------------
    def register_quality_check(self, valid: bool, reason: Optional[str]) -> None:
        with self._lock:
            self._quality_checks += 1
            if not valid and reason:
                self._quality_rejections[reason] += 1

    # ------------------------------------------------------------------
    def record_sample(self, record: SampleRecord) -> None:
        canonical_label = self._canonical(record.label)
        record.label = canonical_label
        canonical_hint = self._canonical(record.hint_label)
        record.hint_label = canonical_hint or None

        with self._lock:
            self._samples.append(record)
            target = canonical_label or "__unlabelled__"
            if record.accepted:
                self._accepted_scores[target].append(record.score)
            else:
                self._rejected_scores[target].append(record.score)

            self._reason_counts[record.reason] += 1
            if canonical_label:
                self._reason_by_label[canonical_label][record.reason] += 1

    # ------------------------------------------------------------------
    def _f1_counts(self, samples: List[SampleRecord], label: str) -> Tuple[int, int, int]:
        label_norm = self._normalise(label)
        tp = fp = fn = 0

        for record in samples:
            predicted = self._normalise(record.label)
            actual = self._normalise(record.hint_label)

            if actual == label_norm and actual:
                if record.accepted and predicted == label_norm:
                    tp += 1
                else:
                    fn += 1
            elif record.accepted and predicted == label_norm and actual:
                fp += 1

        return tp, fp, fn

    # ------------------------------------------------------------------
    @staticmethod
    def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    # ------------------------------------------------------------------
    def _simulate_raise(self, samples: List[SampleRecord], label: str, new_threshold: float) -> Tuple[int, int, int]:
        base_tp, base_fp, base_fn = self._f1_counts(samples, label)
        label_norm = self._normalise(label)

        for record in samples:
            if not record.accepted:
                continue
            if self._normalise(record.label) != label_norm:
                continue
            if record.score >= new_threshold:
                continue

            actual = self._normalise(record.hint_label)
            if actual == label_norm and actual:
                base_tp = max(0, base_tp - 1)
                base_fn += 1
            elif actual:
                base_fp = max(0, base_fp - 1)

        return base_tp, base_fp, base_fn

    # ------------------------------------------------------------------
    def _simulate_lower(self, samples: List[SampleRecord], label: str, new_threshold: float) -> Tuple[int, int, int]:
        base_tp, base_fp, base_fn = self._f1_counts(samples, label)
        label_norm = self._normalise(label)

        for record in samples:
            if record.accepted:
                continue
            if self._normalise(record.label) != label_norm:
                continue
            if record.score < new_threshold:
                continue
            if record.reason not in {"score_below_threshold", "score_below_global"}:
                continue

            actual = self._normalise(record.hint_label)
            if actual == label_norm and actual:
                base_tp += 1
                base_fn = max(0, base_fn - 1)
            elif actual:
                base_fp += 1

        return base_tp, base_fp, base_fn

    # ------------------------------------------------------------------
    def threshold_suggestions(self, thresholds: Dict[str, ClassThreshold]) -> List[ThresholdSuggestion]:
        with self._lock:
            samples = list(self._samples)

        suggestions: List[ThresholdSuggestion] = []

        if not samples:
            return suggestions

        for label, threshold in thresholds.items():
            label_samples = [record for record in samples if self._canonical(record.label) == label]
            if not label_samples:
                continue

            base_tp, base_fp, base_fn = self._f1_counts(samples, label)
            _, _, base_f1 = self._precision_recall_f1(base_tp, base_fp, base_fn)
            if base_f1 is None:
                continue

            candidate: Optional[ThresholdSuggestion] = None

            # Try increasing the threshold first to reduce false positives.
            for delta in (0.02, 0.03, 0.05):
                new_threshold = min(0.99, threshold.enter + delta)
                tp, fp, fn = self._simulate_raise(label_samples, label, new_threshold)
                _, _, candidate_f1 = self._precision_recall_f1(tp, fp, fn)
                if candidate_f1 is None:
                    continue
                if candidate_f1 >= base_f1 * 1.05 and fp <= base_fp:
                    candidate = ThresholdSuggestion(
                        label=label,
                        current=threshold.enter,
                        recommended=new_threshold,
                        delta=new_threshold - threshold.enter,
                        reason="Reducir falsos positivos manteniendo precisión",
                        expected_f1=candidate_f1,
                    )
                    break

            if candidate is None:
                for delta in (-0.05, -0.03, -0.02):
                    new_threshold = max(0.4, threshold.enter + delta)
                    tp, fp, fn = self._simulate_lower(label_samples, label, new_threshold)
                    _, _, candidate_f1 = self._precision_recall_f1(tp, fp, fn)
                    if candidate_f1 is None:
                        continue
                    if candidate_f1 >= base_f1 * 1.05 and fp <= base_fp:
                        candidate = ThresholdSuggestion(
                            label=label,
                            current=threshold.enter,
                            recommended=new_threshold,
                            delta=new_threshold - threshold.enter,
                            reason="Aumentar recall sin penalizar falsos positivos",
                            expected_f1=candidate_f1,
                        )
                        break

            if candidate:
                suggestions.append(candidate)

        return suggestions

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
            quality_checks = self._quality_checks
            quality_rejections = dict(self._quality_rejections)
            reason_counts = dict(self._reason_counts)
            reason_by_label = {label: dict(counter) for label, counter in self._reason_by_label.items()}
            accepted_scores = {label: list(values) for label, values in self._accepted_scores.items()}
            rejected_scores = {label: list(values) for label, values in self._rejected_scores.items()}

        return {
            "samples": [record.__dict__ for record in samples],
            "quality_checks": quality_checks,
            "quality_rejections": quality_rejections,
            "reason_counts": reason_counts,
            "reason_by_label": reason_by_label,
            "accepted_scores": accepted_scores,
            "rejected_scores": rejected_scores,
        }

    # ------------------------------------------------------------------
    def generate_report(
        self,
        *,
        thresholds: Dict[str, ClassThreshold],
        consensus: ConsensusConfig,
        dataset_info: Dict[str, Any],
        latency_stats: Dict[str, float],
        label_consensus: Optional[Dict[str, ConsensusConfig]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
            quality_checks = self._quality_checks
            quality_rejections = dict(self._quality_rejections)
            reason_counts = dict(self._reason_counts)
            reason_by_label = {label: dict(counter) for label, counter in self._reason_by_label.items()}
            accepted_scores = {label: list(values) for label, values in self._accepted_scores.items()}
            rejected_scores = {label: list(values) for label, values in self._rejected_scores.items()}

        total_rejections = sum(quality_rejections.values())
        quality_ratio = (total_rejections / quality_checks) if quality_checks else 0.0

        per_label: Dict[str, Any] = {}
        for label in TRACKED_GESTURES:
            tp, fp, fn = self._f1_counts(samples, label)
            precision, recall, f1 = self._precision_recall_f1(tp, fp, fn)
            accepted = accepted_scores.get(label, [])
            rejected = rejected_scores.get(label, [])
            per_label[label] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accepted_scores": self._score_stats(accepted),
                "rejected_scores": self._score_stats(rejected),
                "rejections": reason_by_label.get(label, {}),
            }

        confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for record in samples:
            actual = self._canonical(record.hint_label)
            if not actual:
                continue
            predicted = self._canonical(record.label) if record.accepted else "None"
            confusion[actual][predicted] += 1

        suggestions = [suggestion.__dict__ for suggestion in self.threshold_suggestions(thresholds)]

        consensus_block: Dict[str, Any] = {
            "window_size": consensus.window_size,
            "required_votes": consensus.required_votes,
        }
        if label_consensus:
            consensus_block["overrides"] = {
                label: {"window_size": cfg.window_size, "required_votes": cfg.required_votes}
                for label, cfg in label_consensus.items()
            }

        return {
            "thresholds": {label: {"enter": th.enter, "release": th.release} for label, th in thresholds.items()},
            "global_min_score": GLOBAL_MIN_SCORE,
            "consensus": consensus_block,
            "durations_s": {
                "cooldown": COOLDOWN_SECONDS,
                "listening_window": LISTENING_WINDOW_SECONDS,
                "command_debounce": COMMAND_DEBOUNCE_SECONDS,
            },
            "dataset": dataset_info,
            "latency": latency_stats,
            "samples": len(samples),
            "quality_checks": quality_checks,
            "quality_rejections": quality_rejections,
            "quality_rejection_rate": quality_ratio,
            "threshold_rejections": reason_counts,
            "classes": per_label,
            "confusion_matrix": {actual: dict(preds) for actual, preds in confusion.items()},
            "suggested_thresholds": suggestions,
        }

    # ------------------------------------------------------------------
    def to_markdown(self, report: Dict[str, Any]) -> str:
        lines = ["# Informe de sesión de gestos", ""]
        lines.append("## Configuración activa")
        lines.append("- Ventana de consenso: {window} frames (requiere {votes})".format(
            window=report["consensus"]["window_size"], votes=report["consensus"]["required_votes"]
        ))
        overrides = report.get("consensus", {}).get("overrides")
        if overrides:
            lines.append("  - Overrides por clase:")
            for label, data in sorted(overrides.items()):
                lines.append(
                    "    * {label}: {votes} votos en {window} frames".format(
                        label=label,
                        votes=data.get("required_votes"),
                        window=data.get("window_size"),
                    )
                )
        lines.append("- Cooldown tras 'Start': {0:.0f} ms".format(COOLDOWN_SECONDS * 1000))
        lines.append("- Ventana de escucha C/R/I: {0:.1f} s".format(LISTENING_WINDOW_SECONDS))
        lines.append("- Debounce de comandos: {0:.0f} ms".format(COMMAND_DEBOUNCE_SECONDS * 1000))
        lines.append("- Umbral global mínimo: {0:.2f}".format(report["global_min_score"]))
        lines.append("")
        lines.append("### Umbrales por clase")
        lines.append("| Clase | Entrada | Liberación |")
        lines.append("|-------|---------|------------|")
        for label, data in report["thresholds"].items():
            lines.append(f"| {label} | {data['enter']:.2f} | {data['release']:.2f} |")

        lines.append("")
        lines.append("## Métricas de sesión")
        lines.append(f"- Frames procesados tras filtros: {report['samples']}")
        lines.append(f"- Revisiones de calidad: {report['quality_checks']}")
        lines.append(
            "- Descartes por calidad: {0} ({1:.1%})".format(
                sum(report["quality_rejections"].values()), report["quality_rejection_rate"]
            )
        )
        lines.append("")

        lines.append("### Rendimiento por clase")
        lines.append("| Clase | Precision | Recall | F1 | TP | FP | FN |")
        lines.append("|-------|-----------|--------|----|----|----|----|")
        for label in TRACKED_GESTURES:
            stats = report["classes"][label]
            precision = stats["precision"] if stats["precision"] is not None else float("nan")
            recall = stats["recall"] if stats["recall"] is not None else float("nan")
            f1 = stats["f1"] if stats["f1"] is not None else float("nan")
            lines.append(
                "| {label} | {p:.2f} | {r:.2f} | {f:.2f} | {tp} | {fp} | {fn} |".format(
                    label=label,
                    p=precision if math.isfinite(precision) else float("nan"),
                    r=recall if math.isfinite(recall) else float("nan"),
                    f=f1 if math.isfinite(f1) else float("nan"),
                    tp=stats["tp"],
                    fp=stats["fp"],
                    fn=stats["fn"],
                )
            )

        lines.append("")
        lines.append("### Matriz de confusión")
        if report["confusion_matrix"]:
            all_labels = sorted({pred for preds in report["confusion_matrix"].values() for pred in preds})
            header = "| Actual | " + " | ".join(all_labels) + " |"
            separator = "| " + " | ".join(["---"] * (len(all_labels) + 1)) + " |"
            lines.append(header)
            lines.append(separator)
            for actual, preds in sorted(report["confusion_matrix"].items()):
                row = [actual]
                for pred in all_labels:
                    row.append(str(preds.get(pred, 0)))
                lines.append("| " + " | ".join(row) + " |")
        else:
            lines.append("Sin datos de validación etiquetados.")

        lines.append("")
        if report["suggested_thresholds"]:
            lines.append("### Sugerencias de ajuste")
            lines.append("| Clase | Actual | Recomendado | Δ | Motivo | F1 estimado |")
            lines.append("|-------|--------|-------------|---|--------|-------------|")
            for suggestion in report["suggested_thresholds"]:
                lines.append(
                    "| {label} | {current:.2f} | {recommended:.2f} | {delta:+.2f} | {reason} | {f1:.2f} |".format(
                        label=suggestion["label"],
                        current=suggestion["current"],
                        recommended=suggestion["recommended"],
                        delta=suggestion["delta"],
                        reason=suggestion["reason"],
                        f1=suggestion["expected_f1"],
                    )
                )

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    def dump_report(
        self,
        *,
        markdown_path: Path,
        thresholds: Dict[str, ClassThreshold],
        consensus: ConsensusConfig,
        dataset_info: Dict[str, Any],
        latency_stats: Dict[str, float],
        label_consensus: Optional[Dict[str, ConsensusConfig]] = None,
    ) -> None:
        report = self.generate_report(
            thresholds=thresholds,
            consensus=consensus,
            dataset_info=dataset_info,
            latency_stats=latency_stats,
            label_consensus=label_consensus,
        )
        markdown = self.to_markdown(report)

        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")

        json_path = markdown_path.with_suffix(".json")
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


LandmarkPoint = Tuple[float, float, float]


class LandmarkGeometryVerifier:
    """Apply geometric heuristics to validate model predictions."""

    _FINGER_LANDMARKS = {
        "thumb": (1, 2, 3, 4),
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }

    def __init__(self) -> None:
        self._last_warning: Optional[str] = None

    @staticmethod
    def _vector(a: LandmarkPoint, b: LandmarkPoint) -> LandmarkPoint:
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    @staticmethod
    def _norm(vector: LandmarkPoint) -> float:
        return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)

    @classmethod
    def _angle(cls, a: LandmarkPoint, b: LandmarkPoint, c: LandmarkPoint) -> float:
        ba = cls._vector(a, b)
        bc = cls._vector(c, b)
        denom = cls._norm(ba) * cls._norm(bc)
        if denom <= 1e-6:
            return 0.0
        cosine = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]) / denom))
        return math.degrees(math.acos(cosine))

    @classmethod
    def _finger_curl(cls, landmarks: Sequence[LandmarkPoint], indices: Tuple[int, int, int, int]) -> float:
        mcp, pip, dip, tip = indices
        pip_angle = cls._angle(landmarks[mcp], landmarks[pip], landmarks[dip])
        dip_angle = cls._angle(landmarks[pip], landmarks[dip], landmarks[tip])
        return (pip_angle + dip_angle) / 2.0

    @staticmethod
    def _distance(a: LandmarkPoint, b: LandmarkPoint) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _finger_states(self, landmarks: Sequence[LandmarkPoint]) -> Dict[str, Dict[str, float]]:
        data: Dict[str, Dict[str, float]] = {}
        for name, indices in self._FINGER_LANDMARKS.items():
            curl = self._finger_curl(landmarks, indices)
            data[name] = {
                "curl": curl,
                "extended": float(curl >= 150.0),
            }
        return data

    def _log_once(self, message: str) -> None:
        if message != self._last_warning:
            LOGGER.debug("Filtro geométrico: %s", message)
            self._last_warning = message

    def verify(self, label: str, landmarks: Optional[Sequence[LandmarkPoint]]) -> Tuple[bool, Optional[str]]:
        if not label or not landmarks:
            return True, None

        canonical = GestureMetrics._canonical(label)
        if canonical not in TRACKED_GESTURES:
            return True, None

        points = list(landmarks)
        if len(points) < 21:
            if canonical == "Clima" and len(points) >= 18:
                missing = 21 - len(points)
                last_point = points[-1]
                points.extend([last_point] * missing)
                self._log_once(f"landmarks_clima_padded_{missing}")
            else:
                self._log_once("landmarks_insuficientes")
                return False, "geometry_incomplete"

        finger_states = self._finger_states(points)
        wrist = points[0]
        thumb_tip = points[4]
        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]

        thumb_index_distance = self._distance(thumb_tip, index_tip)
        index_middle_distance = self._distance(index_tip, middle_tip)
        palm_spread = statistics.fmean(
            [self._distance(wrist, tip) for tip in (index_tip, middle_tip, ring_tip, pinky_tip)]
        )

        def is_extended(name: str, threshold: float = 1.0) -> bool:
            return finger_states.get(name, {}).get("extended", 0.0) >= threshold

        def is_folded(name: str) -> bool:
            return finger_states.get(name, {}).get("curl", 0.0) <= 135.0

        if canonical == "Start":
            if LSTM_BACKEND_ACTIVE:
                _debug_decision("Geometría Start omitida (backend LSTM activo)")
                return True, None
            if is_extended("index") and is_extended("middle") and is_folded("ring") and is_folded("pinky"):
                if index_middle_distance >= 0.035:
                    return True, None
                return False, "geometry_start_spacing"
            return False, "geometry_start_pattern"

        if canonical == "Clima":
            finger_names = ("index", "middle", "ring", "pinky")
            strong_extended = sum(1 for finger in finger_names if is_extended(finger, 0.8))
            relaxed_extended = sum(
                1 for finger in finger_names if finger_states.get(finger, {}).get("curl", 0.0) >= 120.0
            )
            arc_span = self._distance(index_tip, pinky_tip)
            finger_spans = {
                finger: self._distance(
                    points[self._FINGER_LANDMARKS[finger][0]], points[self._FINGER_LANDMARKS[finger][-1]]
                )
                for finger in finger_names
            }
            occluded_fingers = sum(1 for span in finger_spans.values() if span <= 0.028)
            effective_relaxed = relaxed_extended + min(occluded_fingers, 3)

            curvature_ok = (
                thumb_index_distance >= 0.038
                and palm_spread >= 0.152
                and arc_span >= 0.138
                and index_middle_distance >= 0.03
            )

            if curvature_ok and (
                strong_extended >= 3 or (strong_extended >= 2 and effective_relaxed >= 3)
            ):
                return True, None

            if curvature_ok and effective_relaxed >= 3:
                curls = [finger_states.get(finger, {}).get("curl", 0.0) for finger in finger_names]
                average_curl = statistics.fmean(curls) if curls else 0.0
                if average_curl >= 130.0:
                    return True, None

            if (
                thumb_index_distance >= 0.036
                and palm_spread >= 0.148
                and arc_span >= 0.132
                and effective_relaxed >= 3
            ):
                return True, None

            return False, "geometry_clima_pattern"

        if canonical == "Reloj":
            if is_extended("index") and is_extended("middle") and is_folded("ring") and is_folded("pinky"):
                vertical_alignment = abs(index_tip[1] - middle_tip[1]) <= 0.05
                if index_middle_distance <= 0.07 and vertical_alignment:
                    return True, None
                return False, "geometry_reloj_spacing"
            return False, "geometry_reloj_pattern"

        if canonical == "Inicio":
            if is_extended("pinky") and is_folded("index") and is_folded("middle") and is_folded("ring"):
                if thumb_index_distance <= 0.09:
                    return True, None
                return False, "geometry_inicio_thumb"
            return False, "geometry_inicio_pattern"

        return True, None


class GestureDecisionEngine:
    """Stateful filter applying consensus, hysteresis and cooldown rules."""

    def __init__(
        self,
        *,
        metrics: GestureMetrics,
        thresholds: Optional[Dict[str, ClassThreshold]] = None,
        consensus: ConsensusConfig = DEFAULT_CONSENSUS_CONFIG,
        global_min_score: float = GLOBAL_MIN_SCORE,
        geometry_verifier: Optional[LandmarkGeometryVerifier] = None,
        per_label_consensus: Optional[Dict[str, ConsensusConfig]] = None,
    ) -> None:
        self._metrics = metrics
        base_thresholds = dict(DEFAULT_CLASS_THRESHOLDS)
        if thresholds:
            base_thresholds.update(thresholds)
        self._thresholds = base_thresholds
        self._consensus_config = consensus
        self._consensus = ConsensusTracker(consensus)
        self._global_min_score = float(global_min_score)
        self._geometry_verifier = geometry_verifier
        overrides: Dict[str, ConsensusConfig] = {}
        if per_label_consensus:
            for label, override in per_label_consensus.items():
                if not override:
                    continue
                canonical = GestureMetrics._canonical(label)
                if canonical:
                    overrides[canonical] = override
        self._per_label_consensus = overrides

        self._state = "idle"
        self._cooldown_until = 0.0
        self._listen_until = 0.0
        self._command_debounce_until = 0.0
        self._listening_duration = LISTENING_WINDOW_SECONDS
        self._dominant_label: Optional[str] = None
        self._last_state_change = time.time()
        self._last_activation_at = 0.0
        self._lock = threading.Lock()
        self._last_clima_warning: Optional[str] = None
        self._last_clima_accept_signature: Optional[Tuple[int, int, int, int]] = None

    # ------------------------------------------------------------------
    def _debug_outcome(
        self,
        *,
        label: str,
        score: float,
        reason: str,
        state: str,
        support: int,
        required_votes: int,
    ) -> None:
        _debug_decision(
            "DecisionEngine: label=%s score=%.3f reason=%s state=%s votos=%d/%d",
            label,
            score,
            reason,
            state,
            support,
            required_votes,
        )

    # ------------------------------------------------------------------
    def _reset_consensus(self) -> None:
        self._consensus.reset()
        self._dominant_label = None

    # ------------------------------------------------------------------
    def _consensus_override(self, label: Optional[str]) -> Optional[ConsensusConfig]:
        if not label:
            return None
        return self._per_label_consensus.get(label)

    # ------------------------------------------------------------------
    def _update_state(self, timestamp: float) -> None:
        if self._state == "cooldown" and timestamp >= self._cooldown_until:
            self._state = "listening"
            self._listen_until = timestamp + self._listening_duration
            self._last_state_change = timestamp
            self._reset_consensus()

        if self._state == "command_debounce" and timestamp >= self._command_debounce_until:
            self._state = "idle"
            self._last_state_change = timestamp
            self._reset_consensus()

        if self._state == "listening" and timestamp >= self._listen_until:
            self._state = "idle"
            self._last_state_change = timestamp
            self._reset_consensus()

    # ------------------------------------------------------------------
    def _current_threshold(self, label: str) -> Optional[ClassThreshold]:
        return self._thresholds.get(label)

    # ------------------------------------------------------------------
    def _record(
        self,
        *,
        label: str,
        score: float,
        accepted: bool,
        reason: str,
        state: str,
        hint_label: Optional[str],
        support: int,
        window_ms: float,
        timestamp: float,
    ) -> None:
        record = SampleRecord(
            timestamp=timestamp,
            label=label,
            score=score,
            accepted=accepted,
            reason=reason,
            state=state,
            hint_label=hint_label,
            support=support,
            window_ms=window_ms,
        )
        self._metrics.record_sample(record)

    # ------------------------------------------------------------------
    def _log_clima_decision(
        self,
        *,
        accepted: bool,
        reason: str,
        score: float,
        support: int,
        required_votes: int,
        window_ms: float,
        total_votes: int,
    ) -> None:
        frames_used = max(int(total_votes), support, 0)
        window_display = max(window_ms, 0.0)
        if accepted:
            signature = (int(round(score * 1000)), support, required_votes, frames_used)
            if signature != self._last_clima_accept_signature:
                LOGGER.info(
                    "Seña 'C' aceptada (score=%.3f, votos=%d/%d, frames=%d, ventana=%.0f ms)",
                    score,
                    support,
                    required_votes,
                    frames_used,
                    window_display,
                )
                self._last_clima_accept_signature = signature
                self._last_clima_warning = None
            return

        warning_key = f"{reason}:{support}:{required_votes}:{frames_used}"
        if warning_key != self._last_clima_warning:
            LOGGER.warning(
                "Seña 'C' descartada por %s (score=%.3f, votos=%d/%d, frames=%d, ventana=%.0f ms)",
                reason,
                score,
                support,
                required_votes,
                frames_used,
                window_display,
            )
            self._last_clima_warning = warning_key

    # ------------------------------------------------------------------
    def _finalize_decision(
        self,
        emit: bool,
        canonical_label: str,
        score: float,
        payload: Dict[str, Any],
        reason: str,
        state: str,
        canonical_hint: Optional[str],
        support: int,
        window_ms: float,
        required_votes: int,
        total_votes: int,
    ) -> DecisionOutcome:
        if canonical_label == "Clima":
            self._log_clima_decision(
                accepted=emit,
                reason=reason,
                score=score,
                support=support,
                required_votes=required_votes,
                window_ms=window_ms,
                total_votes=total_votes,
            )
        self._debug_outcome(
            label=canonical_label,
            score=score,
            reason=reason,
            state=state,
            support=support,
            required_votes=required_votes,
        )
        return DecisionOutcome(emit, canonical_label, score, payload, reason, state, canonical_hint, support, window_ms)

    # ------------------------------------------------------------------
    def _apply_hysteresis(self, label: str) -> Optional[str]:
        if not self._dominant_label:
            return None
        if self._dominant_label == label:
            return None

        thresholds = self._current_threshold(self._dominant_label)
        if thresholds is None:
            self._dominant_label = None
            return None

        override = self._consensus_override(self._dominant_label)
        result = self._consensus.evaluate(
            self._dominant_label,
            thresholds.release,
            window_size=override.window_size if override else None,
        )
        if result.average < thresholds.release:
            self._dominant_label = None
            return None
        return self._dominant_label

    # ------------------------------------------------------------------
    def process(
        self,
        prediction: Prediction,
        *,
        timestamp: float,
        hint_label: Optional[str] = None,
        latency_ms: float = 0.0,
        landmarks: Optional[Sequence[LandmarkPoint]] = None,
    ) -> DecisionOutcome:
        with self._lock:
            self._update_state(timestamp)

            canonical_label = GestureMetrics._canonical(prediction.label)
            canonical_hint = GestureMetrics._canonical(hint_label)
            score = float(prediction.score)
            state = self._state
            consensus_override = self._consensus_override(canonical_label)

            payload: Dict[str, Any] = {
                "latency_ms": latency_ms,
                "state": state,
            }
            geometry_checked = False

            required_votes = (
                consensus_override.required_votes if consensus_override else self._consensus_config.required_votes
            )
            consensus_window = (
                consensus_override.window_size if consensus_override else self._consensus_config.window_size
            )

            if self._geometry_verifier is not None and landmarks is not None:
                geometry_ok, geometry_reason = self._geometry_verifier.verify(canonical_label, landmarks)
                geometry_checked = True
                if not geometry_ok:
                    reason = geometry_reason or "geometry_rejected"
                    self._record(
                        label=canonical_label,
                        score=score,
                        accepted=False,
                        reason=reason,
                        state=state,
                        hint_label=canonical_hint,
                        support=0,
                        window_ms=0.0,
                        timestamp=timestamp,
                    )
                    payload["decision_reason"] = reason
                    payload["geometry_checked"] = True
                    payload["geometry_reason"] = reason
                    return self._finalize_decision(
                        False,
                        canonical_label,
                        score,
                        payload,
                        reason,
                        state,
                        canonical_hint,
                        0,
                        0.0,
                        required_votes,
                        0,
                    )
            if self._geometry_verifier is not None:
                payload["geometry_checked"] = geometry_checked

            thresholds = self._current_threshold(canonical_label)
            self._consensus.add(canonical_label, score, timestamp)
            result = self._consensus.evaluate(
                canonical_label,
                thresholds.enter if thresholds else self._global_min_score,
                window_size=consensus_window,
            )

            support = result.votes
            window_ms = result.span_ms
            total_votes = result.total

            if thresholds and result.average >= thresholds.enter:
                self._dominant_label = canonical_label

            locked_label = self._apply_hysteresis(canonical_label)
            state = self._state

            payload.update(
                {
                    "consensus_support": support,
                    "consensus_total": result.total,
                    "consensus_span_ms": window_ms,
                    "votes_required": required_votes,
                    "consensus_window": consensus_window,
                }
            )

            if locked_label and locked_label != canonical_label:
                reason = "hysteresis_locked"
                self._record(
                    label=canonical_label,
                    score=score,
                    accepted=False,
                    reason=reason,
                    state=state,
                    hint_label=canonical_hint,
                    support=support,
                    window_ms=window_ms,
                    timestamp=timestamp,
                )
                payload["locked_label"] = locked_label
                payload["decision_reason"] = reason
                return self._finalize_decision(
                    False,
                    canonical_label,
                    score,
                    payload,
                    reason,
                    state,
                    canonical_hint,
                    support,
                    window_ms,
                    required_votes,
                    total_votes,
                )

            if score < self._global_min_score:
                reason = "score_below_global"
                self._record(
                    label=canonical_label,
                    score=score,
                    accepted=False,
                    reason=reason,
                    state=state,
                    hint_label=canonical_hint,
                    support=support,
                    window_ms=window_ms,
                    timestamp=timestamp,
                )
                payload["decision_reason"] = reason
                return self._finalize_decision(
                    False,
                    canonical_label,
                    score,
                    payload,
                    reason,
                    state,
                    canonical_hint,
                    support,
                    window_ms,
                    required_votes,
                    total_votes,
                )

            if thresholds is None:
                reason = "not_tracked"
                self._record(
                    label=canonical_label,
                    score=score,
                    accepted=False,
                    reason=reason,
                    state=state,
                    hint_label=canonical_hint,
                    support=support,
                    window_ms=window_ms,
                    timestamp=timestamp,
                )
                payload["decision_reason"] = reason
                return self._finalize_decision(
                    False,
                    canonical_label,
                    score,
                    payload,
                    reason,
                    state,
                    canonical_hint,
                    support,
                    window_ms,
                    required_votes,
                    total_votes,
                )

            if (
                canonical_label == "Clima"
                and state == "listening"
                and self._last_activation_at
                and (timestamp - self._last_activation_at) < CLIMA_POST_START_DELAY
            ):
                reason = "post_start_delay"
                self._record(
                    label=canonical_label,
                    score=score,
                    accepted=False,
                    reason=reason,
                    state=state,
                    hint_label=canonical_hint,
                    support=support,
                    window_ms=window_ms,
                    timestamp=timestamp,
                )
                payload["decision_reason"] = reason
                return self._finalize_decision(
                    False,
                    canonical_label,
                    score,
                    payload,
                    reason,
                    state,
                    canonical_hint,
                    support,
                    window_ms,
                    required_votes,
                    total_votes,
                )

            if state == "command_debounce" and timestamp < self._command_debounce_until:
                reason = "command_debounce_active"
                self._record(
                    label=canonical_label,
                    score=score,
                    accepted=False,
                    reason=reason,
                    state=state,
                    hint_label=canonical_hint,
                    support=support,
                    window_ms=window_ms,
                    timestamp=timestamp,
                )
                payload["decision_reason"] = reason
                return self._finalize_decision(
                    False,
                    canonical_label,
                    score,
                    payload,
                    reason,
                    state,
                    canonical_hint,
                    support,
                    window_ms,
                    required_votes,
                    total_votes,
                )

            if state == "cooldown" and timestamp < self._cooldown_until:
                reason = "cooldown_active"
                self._record(
                    label=canonical_label,
                    score=score,
                    accepted=False,
                    reason=reason,
                    state=state,
                    hint_label=canonical_hint,
                    support=support,
                    window_ms=window_ms,
                    timestamp=timestamp,
                )
                payload["decision_reason"] = reason
                return self._finalize_decision(
                    False,
                    canonical_label,
                    score,
                    payload,
                    reason,
                    state,
                    canonical_hint,
                    support,
                    window_ms,
                    required_votes,
                    total_votes,
                )

            if state == "idle" and canonical_label != "Start":
                reason = "awaiting_activation"
                self._record(
                    label=canonical_label,
                    score=score,
                    accepted=False,
                    reason=reason,
                    state=state,
                    hint_label=canonical_hint,
                    support=support,
                    window_ms=window_ms,
                    timestamp=timestamp,
                )
                payload["decision_reason"] = reason
                return self._finalize_decision(
                    False,
                    canonical_label,
                    score,
                    payload,
                    reason,
                    state,
                    canonical_hint,
                    support,
                    window_ms,
                    required_votes,
                    total_votes,
                )

            if state == "listening" and canonical_label == "Start":
                reason = "awaiting_command"
                self._record(
                    label=canonical_label,
                    score=score,
                    accepted=False,
                    reason=reason,
                    state=state,
                    hint_label=canonical_hint,
                    support=support,
                    window_ms=window_ms,
                    timestamp=timestamp,
                )
                payload["decision_reason"] = reason
                return self._finalize_decision(
                    False,
                    canonical_label,
                    score,
                    payload,
                    reason,
                    state,
                    canonical_hint,
                    support,
                    window_ms,
                    required_votes,
                    total_votes,
                )

            if state == "listening" and canonical_label not in {"Clima", "Reloj", "Inicio"}:
                reason = "unsupported_command"
                self._record(
                    label=canonical_label,
                    score=score,
                    accepted=False,
                    reason=reason,
                    state=state,
                    hint_label=canonical_hint,
                    support=support,
                    window_ms=window_ms,
                    timestamp=timestamp,
                )
                payload["decision_reason"] = reason
                return self._finalize_decision(
                    False,
                    canonical_label,
                    score,
                    payload,
                    reason,
                    state,
                    canonical_hint,
                    support,
                    window_ms,
                    required_votes,
                    total_votes,
                )

            if score < thresholds.enter:
                reason = "score_below_threshold"
                self._record(
                    label=canonical_label,
                    score=score,
                    accepted=False,
                    reason=reason,
                    state=state,
                    hint_label=canonical_hint,
                    support=support,
                    window_ms=window_ms,
                    timestamp=timestamp,
                )
                payload["threshold_enter"] = thresholds.enter
                payload["decision_reason"] = reason
                return self._finalize_decision(
                    False,
                    canonical_label,
                    score,
                    payload,
                    reason,
                    state,
                    canonical_hint,
                    support,
                    window_ms,
                    required_votes,
                    total_votes,
                )

            passes_votes = support >= required_votes
            passes_average = result.average >= thresholds.enter

            if not passes_votes and not passes_average:
                reason = "consensus_short"
                self._record(
                    label=canonical_label,
                    score=score,
                    accepted=False,
                    reason=reason,
                    state=state,
                    hint_label=canonical_hint,
                    support=support,
                    window_ms=window_ms,
                    timestamp=timestamp,
                )
                payload["decision_reason"] = reason
                return self._finalize_decision(
                    False,
                    canonical_label,
                    score,
                    payload,
                    reason,
                    state,
                    canonical_hint,
                    support,
                    window_ms,
                    required_votes,
                    total_votes,
                )

            reason = "accepted"
            self._record(
                label=canonical_label,
                score=score,
                accepted=True,
                reason=reason,
                state=state,
                hint_label=canonical_hint,
                support=support,
                window_ms=window_ms,
                timestamp=timestamp,
            )

            payload.update(
                {
                    "consensus_average": result.average,
                    "votes_required": required_votes,
                }
            )
            payload["decision_reason"] = reason

            if canonical_label == "Start":
                self._state = "cooldown"
                self._cooldown_until = timestamp + COOLDOWN_SECONDS
                self._last_activation_at = timestamp
                payload["next_state"] = "cooldown"
                self._reset_consensus()
            else:
                self._state = "command_debounce"
                self._command_debounce_until = timestamp + COMMAND_DEBOUNCE_SECONDS
                self._listen_until = timestamp
                payload["next_state"] = "command_debounce"
                self._reset_consensus()

            self._last_state_change = timestamp

            return self._finalize_decision(
                True,
                canonical_label,
                score,
                payload,
                reason,
                self._state,
                canonical_hint,
                support,
                window_ms,
                required_votes,
                total_votes,
            )

    # ------------------------------------------------------------------
    def thresholds(self) -> Dict[str, ClassThreshold]:
        return dict(self._thresholds)

    # ------------------------------------------------------------------
    @property
    def consensus_config(self) -> ConsensusConfig:
        return self._consensus_config

    # ------------------------------------------------------------------
    def consensus_overrides(self) -> Dict[str, ConsensusConfig]:
        return dict(self._per_label_consensus)

ACTIVATION_ALIASES = {
    # Mantener sincronizado con ``ACTIVATION_ALIASES`` en
    # ``helen/jsSignHandler/actions.js``.
    "start",
    "activar",
    "heyhelen",
    "holahelen",
    "oyehelen",
    "wake",
}

HEALTH_ENDPOINTS = {"/health", "/healthz"}


def _iso_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


PLATFORM = platform.system().lower()
IS_WINDOWS = PLATFORM.startswith("win")
IS_LINUX = PLATFORM.startswith("linux")


def _read_device_model() -> str:
    if not IS_LINUX:
        return ""

    model_path = Path("/proc/device-tree/model")
    with contextlib.suppress(OSError):
        text = model_path.read_text(encoding="utf-8")
        return text.replace("\x00", "").strip()
    return ""


def _detect_pi_camera_profile() -> Optional[PiCameraProfile]:
    model = _read_device_model().lower()
    if not model:
        return None

    if "raspberry pi 5" in model:
        return PiCameraProfile("raspberry-pi-5", 1280, 720, 25, 0.04, 3)
    if "raspberry pi 4" in model or "compute module 4" in model:
        return PiCameraProfile("raspberry-pi-4", 640, 360, 24, 0.05, 4)
    return None


PI_CAMERA_PROFILE = _detect_pi_camera_profile()
if PI_CAMERA_PROFILE:
    LOGGER.info(
        "Perfil Raspberry Pi detectado (%s): %sx%s @ %s FPS; poll_interval=%.3f; process_every_n=%s",
        PI_CAMERA_PROFILE.model,
        PI_CAMERA_PROFILE.width,
        PI_CAMERA_PROFILE.height,
        PI_CAMERA_PROFILE.fps,
        PI_CAMERA_PROFILE.poll_interval,
        PI_CAMERA_PROFILE.process_every_n,
    )


DEFAULT_DISPLAY_MODE = "raspberry" if PI_CAMERA_PROFILE else "windows"
DISPLAY_MODE_STORE = DisplayModeStore(MODE_STORAGE_PATH, DEFAULT_DISPLAY_MODE)
RASPBERRY_MODE_PROFILE = PiCameraProfile(
    PI_CAMERA_PROFILE.model if PI_CAMERA_PROFILE else "raspberry-mode",
    960,
    540,
    24,
    0.042,
    2,
)


WINDOWS_RUNTIME_DEFAULTS = PlatformRuntimeDefaults(0.72, 0.62, 0.08, 2)
PI5_RUNTIME_DEFAULTS = PlatformRuntimeDefaults(0.58, 0.55, 0.04, 3)
PI4_RUNTIME_DEFAULTS = PlatformRuntimeDefaults(0.6, 0.55, 0.05, 4)
GENERIC_RUNTIME_DEFAULTS = PlatformRuntimeDefaults(0.68, 0.6, 0.1, 3)


def _resolve_runtime_defaults(profile: Optional[PiCameraProfile]) -> PlatformRuntimeDefaults:
    model = ""
    if PI_CAMERA_PROFILE:
        model = PI_CAMERA_PROFILE.model
    elif profile:
        model = profile.model

    if model.startswith("raspberry-pi-5"):
        base = PI5_RUNTIME_DEFAULTS
    elif model.startswith("raspberry-pi-4"):
        base = PI4_RUNTIME_DEFAULTS
    elif IS_WINDOWS:
        base = WINDOWS_RUNTIME_DEFAULTS
    else:
        base = GENERIC_RUNTIME_DEFAULTS

    poll_interval = profile.poll_interval if profile else base.poll_interval
    frame_stride = profile.process_every_n if profile else base.frame_stride

    return PlatformRuntimeDefaults(
        detection_confidence=base.detection_confidence,
        tracking_confidence=base.tracking_confidence,
        poll_interval=poll_interval,
        frame_stride=max(1, int(frame_stride or base.frame_stride)),
    )


def _command_exists(command: str) -> bool:
    return bool(shutil.which(command))


def _run_command(args: Iterable[str], *, timeout: float = 10.0) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(f"Comando no disponible: {args!r}") from exc
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"El comando '{args!r}' excedió el tiempo límite") from exc


def check_online_status(timeout: float = 3.0) -> Dict[str, Any]:
    url = "http://clients3.google.com/generate_204"
    result: Dict[str, Any] = {"online": False, "checked_at": time.time()}
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            result["online"] = response.status == 204
            result["http_status"] = response.status
    except Exception as exc:  # pragma: no cover - network dependent
        result["online"] = False
        result["message"] = str(exc)
    return result


def _parse_percent(value: str) -> Optional[int]:
    candidate = value.rstrip("% ")
    try:
        return int(candidate)
    except (TypeError, ValueError):
        return None


def _scan_windows_networks() -> List[Dict[str, Any]]:
    if not _command_exists("netsh"):
        raise RuntimeError("La herramienta 'netsh' no está disponible en este sistema.")

    result = _run_command(["netsh", "wlan", "show", "networks", "mode=bssid"], timeout=15.0)
    if result.returncode != 0:
        message = result.stderr.strip() or "netsh devolvió un error al escanear redes"
        raise RuntimeError(message)

    networks: Dict[str, Dict[str, Any]] = {}
    current_ssid: Optional[str] = None

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("SSID"):
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            ssid = parts[1].strip()
            if not ssid or ssid.lower() == "<desconocido>":
                current_ssid = None
                continue
            current_ssid = ssid
            networks.setdefault(ssid, {"ssid": ssid, "signal": None, "security": ""})
            continue
        if not current_ssid:
            continue

        entry = networks[current_ssid]
        lowered = line.lower()
        if lowered.startswith("signal"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                value = _parse_percent(parts[1].strip())
                if value is not None:
                    previous = entry.get("signal")
                    if previous is None or value > previous:
                        entry["signal"] = value
        elif lowered.startswith("authentication"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                entry["security"] = parts[1].strip()

    return list(networks.values())


def _scan_nmcli_networks() -> List[Dict[str, Any]]:
    if not _command_exists("nmcli"):
        raise RuntimeError("nmcli no está disponible en este sistema.")

    result = _run_command(["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi"], timeout=15.0)
    if result.returncode != 0:
        message = result.stderr.strip() or "nmcli devolvió un error al escanear redes"
        raise RuntimeError(message)

    networks: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}

    for raw_line in result.stdout.splitlines():
        parts = raw_line.split(":", 2)
        if len(parts) < 3:
            continue
        ssid = parts[0].strip()
        if not ssid:
            continue
        signal = _parse_percent(parts[1].strip())
        security = parts[2].strip()
        entry = seen.setdefault(ssid, {"ssid": ssid, "signal": signal, "security": security})
        if signal is not None:
            existing = entry.get("signal")
            if existing is None or signal > existing:
                entry["signal"] = signal
        if security:
            entry["security"] = security

    networks.extend(seen.values())
    return networks


def scan_wifi_networks() -> List[Dict[str, Any]]:
    if IS_WINDOWS:
        return _scan_windows_networks()
    if _command_exists("nmcli"):
        return _scan_nmcli_networks()
    raise RuntimeError("Escaneo Wi-Fi no soportado en esta plataforma.")


def _windows_wifi_status() -> Dict[str, Any]:
    if not _command_exists("netsh"):
        return {}

    result = _run_command(["netsh", "wlan", "show", "interfaces"], timeout=10.0)
    if result.returncode != 0:
        return {}

    status: Dict[str, Any] = {}
    current: Dict[str, Any] = {}

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            if current.get("state", "").lower().startswith("connected"):
                status = current
                break
            current = {}
            continue

        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

        if key == "name":
            current["iface"] = value
        elif key == "state":
            current["state"] = value
        elif key == "ssid" and value:
            current["connected_ssid"] = value
        elif key == "signal":
            parsed = _parse_percent(value)
            if parsed is not None:
                current["signal"] = parsed
        elif key == "authentication":
            current["security"] = value
        elif key.startswith("ipv4") and value:
            current["ipv4"] = value.split("(")[0].strip()

    if not status and current.get("state", "").lower().startswith("connected"):
        status = current

    return {k: v for k, v in status.items() if k in {"iface", "state", "connected_ssid", "signal", "security", "ipv4"}}


def _nmcli_wifi_status() -> Dict[str, Any]:
    if not _command_exists("nmcli"):
        return {}

    status: Dict[str, Any] = {}

    result = _run_command(["nmcli", "-t", "-f", "DEVICE,STATE,CONNECTION", "dev", "status"], timeout=8.0)
    if result.returncode != 0:
        return {}

    active_device: Optional[str] = None
    for raw_line in result.stdout.splitlines():
        parts = raw_line.split(":", 2)
        if len(parts) < 3:
            continue
        device, state, connection = parts
        if state == "connected":
            active_device = device
            status["connected_ssid"] = connection
            status["iface"] = device
            break

    if not active_device:
        return status

    result_signal = _run_command(["nmcli", "-t", "-f", "ACTIVE,SSID,SIGNAL", "dev", "wifi"], timeout=8.0)
    if result_signal.returncode == 0:
        for raw_line in result_signal.stdout.splitlines():
            parts = raw_line.split(":", 2)
            if len(parts) < 3:
                continue
            active, ssid, signal = parts
            if active == "yes":
                if ssid:
                    status["connected_ssid"] = ssid
                parsed = _parse_percent(signal.strip())
                if parsed is not None:
                    status["signal"] = parsed
                break

    result_ip = _run_command(["nmcli", "-t", "-f", "IP4.ADDRESS", "dev", "show", active_device], timeout=8.0)
    if result_ip.returncode == 0:
        for raw_line in result_ip.stdout.splitlines():
            if ":" not in raw_line:
                continue
            _, value = raw_line.split(":", 1)
            value = value.strip()
            if value:
                status["ipv4"] = value.split("/")[0]
                break

    return status


def current_wifi_status() -> Dict[str, Any]:
    if IS_WINDOWS:
        return _windows_wifi_status()
    if _command_exists("nmcli"):
        return _nmcli_wifi_status()
    return {}


def _build_windows_profile(ssid: str, password: str) -> str:
    ssid_text = escape(ssid)
    ssid_hex = ssid.encode("utf-8").hex()
    if password:
        auth_block = """
                <authentication>WPA2PSK</authentication>
                <encryption>AES</encryption>
                <useOneX>false</useOneX>
        """
        key_block = f"""
            <sharedKey>
                <keyType>passPhrase</keyType>
                <protected>false</protected>
                <keyMaterial>{escape(password)}</keyMaterial>
            </sharedKey>
        """
    else:
        auth_block = """
                <authentication>open</authentication>
                <encryption>none</encryption>
                <useOneX>false</useOneX>
        """
        key_block = ""

    return f"""<?xml version=\"1.0\"?>
<WLANProfile xmlns=\"http://www.microsoft.com/networking/WLAN/profile/v1\">
    <name>{ssid_text}</name>
    <SSIDConfig>
        <SSID>
            <hex>{ssid_hex}</hex>
            <name>{ssid_text}</name>
        </SSID>
    </SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM>
        <security>
            <authEncryption>
{auth_block}
            </authEncryption>
{key_block}
        </security>
    </MSM>
</WLANProfile>
"""


def _connect_wifi_windows(ssid: str, password: str) -> Tuple[bool, str]:
    if not _command_exists("netsh"):
        return False, "La herramienta 'netsh' no está disponible."

    profile_xml = _build_windows_profile(ssid, password)
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as handle:
            handle.write(profile_xml)
            profile_path = handle.name
    except OSError as exc:  # pragma: no cover - filesystem dependent
        raise RuntimeError(f"No se pudo crear un perfil temporal de Wi-Fi: {exc}") from exc

    try:
        _run_command(["netsh", "wlan", "delete", "profile", f"name={ssid}"], timeout=6.0)
        added = _run_command(["netsh", "wlan", "add", "profile", f"filename={profile_path}", "user=all"], timeout=10.0)
        if added.returncode != 0:
            message = added.stderr.strip() or "No se pudo registrar el perfil Wi-Fi."
            return False, message
        connected = _run_command(["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}"], timeout=15.0)
        if connected.returncode != 0:
            message = connected.stderr.strip() or "No se pudo iniciar la conexión Wi-Fi."
            return False, message
        return True, ""
    finally:
        with contextlib.suppress(OSError):
            Path(profile_path).unlink(missing_ok=True)


def _connect_wifi_nmcli(ssid: str, password: str) -> Tuple[bool, str]:
    if not _command_exists("nmcli"):
        return False, "nmcli no está disponible."

    command = ["nmcli", "dev", "wifi", "connect", ssid]
    if password:
        command.extend(["password", password])

    result = _run_command(command, timeout=25.0)
    if result.returncode != 0:
        message = result.stderr.strip() or "No se pudo establecer la conexión Wi-Fi."
        return False, message
    return True, ""


def connect_wifi(ssid: str, password: str) -> Tuple[bool, str]:
    if not ssid:
        raise RuntimeError("SSID requerido para conectar.")
    if IS_WINDOWS:
        return _connect_wifi_windows(ssid, password)
    if _command_exists("nmcli"):
        return _connect_wifi_nmcli(ssid, password)
    raise RuntimeError("Conexión Wi-Fi no soportada en esta plataforma.")


@dataclass
class RuntimeConfig:
    camera_index: Optional[Union[int, str]] = None
    camera_device: Optional[str] = None
    camera_backend: Optional[str] = None
    camera_width: Optional[int] = None
    camera_height: Optional[int] = None
    detection_confidence: Optional[float] = None
    tracking_confidence: Optional[float] = None
    poll_interval_s: Optional[float] = None
    enable_camera: bool = True
    fallback_to_synthetic: bool = True
    model_backend: str = field(default_factory=lambda: RUNTIME_MODEL_CONFIG.effective_backend)
    tf_model_dir: Optional[Path] = None
    process_every_n: Optional[int] = None
    display_mode: str = DEFAULT_DISPLAY_MODE
    camera_profile: Optional[PiCameraProfile] = None


@dataclass
class HealthSnapshot:
    status: str
    model_loaded: bool
    model_source: str
    session_id: str
    pipeline_running: bool
    stream_source: str
    clients: int
    uptime_s: float
    last_prediction: Optional[Dict[str, Any]]
    last_prediction_at: Optional[str]
    avg_latency_ms: float
    camera_ok: bool
    camera_index: Optional[Union[int, str]]
    camera_device: Optional[str]
    camera_backend: Optional[str]
    camera_resolution: Optional[str]
    camera_fps: Optional[float]
    camera_pixel_format: Optional[str]
    camera_probe_latency_ms: Optional[float]
    camera_last_capture: Optional[str]
    camera_last_error: Optional[str]
    last_error: Optional[str] = None


class ProductionGestureClassifier:
    """Thin wrapper around the trained XGBoost model stored in ``model.p``."""

    source = "production"

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {model_path!s}")

        try:
            import pickle
        except ModuleNotFoundError as exc:  # pragma: no cover - stdlib always available
            raise RuntimeError("El módulo pickle no está disponible") from exc

        try:
            import numpy as np  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("NumPy es requerido para el modelo de producción") from exc

        try:
            model_dict = pickle.load(model_path.open("rb"))
        except Exception as exc:  # pragma: no cover - file corruption
            raise RuntimeError(f"No se pudo cargar {model_path!s}: {exc}") from exc

        model = model_dict.get("model")
        if model is None:
            raise RuntimeError("El archivo model.p no contiene la clave 'model'")

        self._model = model
        self._encoder = model_dict.get("encoder") or model_dict.get("label_encoder")
        self._classes = list(getattr(model, "classes_", []))
        self._lock = threading.Lock()
        self._labels_map = {int(idx): value for idx, value in labels_dict.items()}
        self._numpy = np

    # ------------------------------------------------------------------
    def predict(self, features: Iterable[float]) -> Prediction:
        np = self._numpy
        array = np.asarray(list(features), dtype=float).reshape(1, -1)

        with self._lock:
            label_value: Any
            score: float = 1.0

            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(array)
                if proba.size:
                    if self._classes:
                        best_index = int(proba[0].argmax())
                        label_value = self._classes[best_index]
                        score = float(proba[0][best_index])
                    else:
                        best_index = int(proba[0].argmax())
                        label_value = best_index
                        score = float(proba[0][best_index])
                else:
                    label_value = self._model.predict(array)[0]
            else:
                label_value = self._model.predict(array)[0]

        label = self._to_label(label_value)
        return Prediction(label=label, score=float(score))

    # ------------------------------------------------------------------
    def _to_label(self, raw_value: Any) -> str:
        if self._encoder is not None:
            with contextlib.suppress(Exception):
                decoded = self._encoder.inverse_transform([raw_value])[0]
                return str(decoded)

        try:
            numeric = int(raw_value)
        except (TypeError, ValueError):
            return str(raw_value)

        return self._labels_map.get(numeric, str(raw_value))


class SyntheticStreamAdapter:
    """Wrap ``SyntheticGestureStream`` adding runtime diagnostics."""

    source = "synthetic"

    def __init__(self, dataset_path: Path) -> None:
        self._stream = SyntheticGestureStream(dataset_path)
        self._last_capture: Optional[float] = None

    def next(self, timeout: float = 0.0) -> Tuple[List[float], Optional[str]]:  # noqa: ARG002 - signature parity
        features, label = self._stream.next()
        self._last_capture = time.time()
        return features, label

    def status(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "camera_index": None,
            "last_capture": self._last_capture,
            "last_error": None,
            "frames_without_hand": 0,
        }

    def last_landmarks(self) -> Optional[List[LandmarkPoint]]:  # pragma: no cover - synthetic stream
        return None

    def close(self) -> None:  # pragma: no cover - nothing to clean up
        return None


class CameraGestureStream:
    """Capture MediaPipe hand landmarks from a physical camera."""

    source = "camera"

    def __init__(
        self,
        *,
        camera_index: Optional[Union[int, str]] = None,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
        metrics: Optional[GestureMetrics] = None,
        profile: Optional[PiCameraProfile] = None,
        selection: Optional[CameraSelection] = None,
        forced_backend: Optional[str] = None,
        width_override: Optional[int] = None,
        height_override: Optional[int] = None,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV no está instalado. Ejecuta `pip install opencv-python`.")
        if mp is None:
            raise RuntimeError("MediaPipe no está instalado. Ejecuta `pip install mediapipe`.")

        self._selection = selection
        self._forced_backend = camera_probe.normalize_backend_name(forced_backend)
        self._width_override = int(width_override) if width_override else None
        self._height_override = int(height_override) if height_override else None
        resolved_index: Optional[Union[int, str]] = camera_index
        if selection:
            if selection.index is not None:
                resolved_index = selection.index
            elif selection.device:
                resolved_index = selection.device
        self._camera_index: Optional[Union[int, str]] = resolved_index
        self._detection_confidence = detection_confidence
        self._tracking_confidence = tracking_confidence
        self._metrics = metrics
        if selection and selection.device:
            self._device_path: Optional[str] = selection.device
        elif isinstance(camera_index, str):
            self._device_path = camera_index
        else:
            self._device_path = None
        backend_candidate = self._forced_backend or (selection.backend if selection else None)
        backend_candidate = camera_probe.normalize_backend_name(backend_candidate)
        if not backend_candidate:
            backend_candidate = "directshow" if camera_probe.IS_WINDOWS else "v4l2"
        self._preferred_backend = backend_candidate
        self._selection_dict = selection.to_dict() if selection else None
        self._probe_latency_ms = selection.latency_ms if selection else None
        self._orientation_hint = selection.orientation if selection else None
        self._pixel_format = selection.pixel_format if selection else None

        self._cap: Optional[Any] = None
        self._hands: Optional[Any] = None
        self._opened = False
        self._profile: Optional[PiCameraProfile] = profile
        self._capture_backend: Optional[str] = None
        self._gstreamer_pipeline: Optional[str] = None

        self._last_capture: Optional[float] = None
        self._last_error: Optional[str] = None
        self._healthy = False
        self._frames_without_hand = 0
        self._landmark_buffer: Deque[List[LandmarkPoint]] = deque(maxlen=SMOOTHING_WINDOW_SIZE)
        self._quality_rejections: Counter[str] = Counter()
        self._last_landmarks: Optional[List[LandmarkPoint]] = None
        self._last_frame_shape: Optional[Tuple[int, int]] = None
        self._last_roi: Optional[Dict[str, Any]] = None
        self._lenient_quality = LSTM_BACKEND_ACTIVE

    # ------------------------------------------------------------------
    def _desired_dimensions(self) -> Tuple[int, int, float]:
        if self._width_override and self._height_override:
            return int(self._width_override), int(self._height_override), 0.0
        selection = self._selection
        if selection:
            width = int(getattr(selection, "width", 0) or 0)
            height = int(getattr(selection, "height", 0) or 0)
            fps = float(getattr(selection, "fps", 0.0) or 0.0)
            if width > 0 and height > 0:
                return width, height, fps
        profile = self._profile
        if profile:
            return profile.width, profile.height, float(profile.fps)
        return 0, 0, 0.0

    # ------------------------------------------------------------------
    def _configure_capture(self, cap: Any) -> None:
        width, height, fps = self._desired_dimensions()
        if self._pixel_format:
            with contextlib.suppress(Exception):
                camera_probe._apply_pixel_format(cap, self._pixel_format)
        if width > 0 and height > 0:
            with contextlib.suppress(Exception):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            with contextlib.suppress(Exception):
                cap.set(cv2.CAP_PROP_FPS, fps)

    # ------------------------------------------------------------------
    def _attempt_opencv(self, backend_name: str) -> Tuple[Optional[Any], Optional[str]]:
        target: Any = self._device_path if self._device_path is not None else self._camera_index
        if target is None:
            return None, "No hay cámara seleccionada"

        backend_flag = camera_probe.resolve_backend_flag(backend_name) or camera_probe.DEFAULT_CAPTURE_FLAG
        try:
            backend_flag_int = int(backend_flag)
        except Exception:
            backend_flag_int = 0
        try:
            cap = cv2.VideoCapture(target, backend_flag_int) if backend_flag_int else cv2.VideoCapture(target)
        except Exception as error:  # pragma: no cover - depende del entorno
            return None, str(error)
        if not cap or not cap.isOpened():
            if cap:
                with contextlib.suppress(Exception):
                    cap.release()
            return None, f"{backend_name} no se pudo abrir en {target}"

        self._configure_capture(cap)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        LOGGER.info(
            "Ruta de cámara inicializada: %s (target=%s, %sx%s @ %.2f fps, formato=%s)",
            backend_name,
            target,
            actual_width,
            actual_height,
            actual_fps,
            self._pixel_format or "<driver>",
        )
        self._capture_backend = backend_name
        self._gstreamer_pipeline = None
        return cap, None

    # ------------------------------------------------------------------
    def _build_gstreamer_pipeline(self) -> str:
        if self._selection_dict and self._selection_dict.get("pipeline"):
            return str(self._selection_dict["pipeline"])
        width, height, fps = self._desired_dimensions()
        if width <= 0:
            width = 1280
        if height <= 0:
            height = 720
        if fps <= 0:
            fps = 30
        return (
            "libcamerasrc ! video/x-raw,width="
            f"{width},height={height},framerate={fps}/1,format=RGB ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2"
        )

    # ------------------------------------------------------------------
    def _attempt_gstreamer(self) -> Tuple[Optional[Any], Optional[str]]:
        pipeline = self._build_gstreamer_pipeline()
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap or not cap.isOpened():
            if cap:
                with contextlib.suppress(Exception):
                    cap.release()
            return None, "GStreamer/libcamerasrc no se pudo inicializar"

        LOGGER.info("Ruta de cámara inicializada: gstreamer (pipeline=%s)", pipeline)
        self._capture_backend = "gstreamer"
        self._gstreamer_pipeline = pipeline
        self._pixel_format = "BGR"
        return cap, None

    # ------------------------------------------------------------------
    def _initialise_capture(self) -> Any:
        errors: List[str] = []
        preferred_order = list(camera_probe.preferred_backend_order(self._preferred_backend))

        cap: Optional[Any] = None
        error: Optional[str] = None
        for backend in preferred_order:
            if backend == "gstreamer":
                cap, error = self._attempt_gstreamer()
            else:
                cap, error = self._attempt_opencv(backend)
            if cap is not None:
                break
            if error:
                errors.append(error)

        if cap is None:
            target = self._device_path if self._device_path else self._camera_index
            details = "; ".join(errors) if errors else f"No se pudo abrir la cámara {target}"
            suggestion = (
                "Sugerencias: prueba índices 0/1/2, cambia el backend con --camera-backend directshow|v4l2 "
                "o reduce la resolución (--camera-width 960 --camera-height 720)."
            )
            LOGGER.error("%s %s", details, suggestion)
            self._last_error = f"{details}. {suggestion}"
            raise RuntimeError(self._last_error)

        return cap

    # ------------------------------------------------------------------
    def _switch_to_gstreamer(self) -> bool:
        cap, error = self._attempt_gstreamer()
        if cap is None:
            if error:
                LOGGER.error("No se pudo iniciar pipeline GStreamer tras un fallo de cámara: %s", error)
            return False

        if self._cap is not None:
            with contextlib.suppress(Exception):
                self._cap.release()

        self._cap = cap
        self._healthy = True
        return True

    # ------------------------------------------------------------------
    def open(self) -> None:
        if self._opened:
            return

        cap = self._initialise_capture()

        self._cap = cap
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self._detection_confidence,
            min_tracking_confidence=self._tracking_confidence,
        )
        self._opened = True
        self._healthy = True
        self._last_error = None

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._cap is not None:
            with contextlib.suppress(Exception):
                self._cap.release()
        if self._hands is not None:
            with contextlib.suppress(Exception):
                self._hands.close()
        self._opened = False
        self._capture_backend = None
        self._gstreamer_pipeline = None
        self._pixel_format = self._selection.pixel_format if self._selection else None

    # ------------------------------------------------------------------
    def next(self, timeout: float = 2.0) -> Tuple[List[float], Optional[str]]:
        if not self._opened:
            self.open()

        assert self._cap is not None
        assert self._hands is not None

        start = time.time()
        while True:
            if timeout and (time.time() - start) > timeout:
                self._last_error = "Tiempo de espera agotado sin detectar mano"
                self._healthy = False
                raise TimeoutError(self._last_error)

            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._last_error = "No se pudo leer un frame de la cámara"
                self._healthy = False
                if self._capture_backend == "v4l2" and self._switch_to_gstreamer():
                    LOGGER.warning("Lectura fallida con V4L2; cambiando a pipeline GStreamer")
                    time.sleep(0.1)
                    continue
                time.sleep(0.05)
                continue

            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                self._last_error = "Dimensiones de imagen no válidas"
                self._healthy = False
                self._register_quality_check(False, "invalid_dimensions")
                time.sleep(0.05)
                continue

            self._last_frame_shape = (int(height), int(width))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if hasattr(image, "flags"):
                image.flags.writeable = False
            try:
                results = self._hands.process(image)
            finally:
                if hasattr(image, "flags"):
                    image.flags.writeable = True

            if not results.multi_hand_landmarks:
                self._frames_without_hand += 1
                if self._frames_without_hand > 2:
                    self._landmark_buffer.clear()
                    self._last_landmarks = None
                    self._last_roi = None
                time.sleep(0.02)
                continue

            self._frames_without_hand = 0
            landmarks = results.multi_hand_landmarks[0]
            if not self._validate_landmarks(frame, results, landmarks, width, height):
                self._last_landmarks = None
                self._last_roi = None
                continue

            coords = [
                (
                    self._clamp_normalized(float(lm.x)),
                    self._clamp_normalized(float(lm.y)),
                    float(getattr(lm, "z", 0.0)),
                )
                for lm in landmarks.landmark
            ]
            roi_snapshot = self._snapshot_roi(coords, width, height)
            if roi_snapshot is None:
                self._register_quality_check(False, "roi_projection")
                self._landmark_buffer.clear()
                self._last_landmarks = None
                self._last_roi = None
                continue

            self._landmark_buffer.append(coords)
            smoothed = self._smooth_landmarks()
            self._last_landmarks = [tuple(point) for point in smoothed]
            self._last_roi = roi_snapshot
            features = self._extract_features(smoothed)
            self._register_quality_check(True, None)
            self._last_capture = time.time()
            self._last_error = None
            self._healthy = True
            return features, None

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "healthy": self._healthy,
            "camera_index": self._camera_index,
            "last_capture": self._last_capture,
            "last_error": self._last_error,
            "frames_without_hand": self._frames_without_hand,
            "quality_rejections": dict(self._quality_rejections),
            "frame_shape": self._last_frame_shape,
            "roi_snapshot": self._last_roi,
            "capture_backend": self._capture_backend,
            "gstreamer_pipeline": self._gstreamer_pipeline,
            "device_path": self._device_path,
            "preferred_backend": self._preferred_backend,
            "probe_latency_ms": self._probe_latency_ms,
            "orientation_hint": self._orientation_hint,
            "pixel_format": self._pixel_format,
        }

        if self._selection_dict:
            status["selection"] = dict(self._selection_dict)

        return status

    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_normalized(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return min(1.0, max(0.0, float(value)))

    # ------------------------------------------------------------------
    @staticmethod
    def _normalised_to_pixel(value: float, size: int) -> int:
        size = max(1, int(size))
        return max(0, min(size - 1, int(round(value * (size - 1)))))

    # ------------------------------------------------------------------
    def _snapshot_roi(self, coords: Sequence[LandmarkPoint], width: int, height: int) -> Optional[Dict[str, Any]]:
        if not coords or width <= 0 or height <= 0:
            return None

        x_values = [point[0] for point in coords]
        y_values = [point[1] for point in coords]
        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)

        if max_x <= min_x or max_y <= min_y:
            return None

        pixel_coords = [
            (
                self._normalised_to_pixel(point[0], width),
                self._normalised_to_pixel(point[1], height),
            )
            for point in coords
        ]

        x_pixels = [coord[0] for coord in pixel_coords]
        y_pixels = [coord[1] for coord in pixel_coords]

        pixel_width = max(x_pixels) - min(x_pixels)
        pixel_height = max(y_pixels) - min(y_pixels)
        if pixel_width <= 0 or pixel_height <= 0:
            return None

        coverage = float((max_x - min_x) * (max_y - min_y))
        pixel_coverage = float((pixel_width / width) * (pixel_height / height))

        return {
            "x1": int(min(x_pixels)),
            "y1": int(min(y_pixels)),
            "x2": int(max(x_pixels)),
            "y2": int(max(y_pixels)),
            "width": int(width),
            "height": int(height),
            "normalized_area": coverage,
            "pixel_area": float(pixel_width * pixel_height),
            "pixel_coverage": pixel_coverage,
        }

    # ------------------------------------------------------------------
    def expand_last_roi_for_clima(self) -> None:
        if not self._last_roi:
            return

        roi = dict(self._last_roi)
        if roi.get("expanded_for_clima"):
            return

        width = int(roi.get("width") or 0)
        height = int(roi.get("height") or 0)
        if width <= 0 or height <= 0:
            return

        margin_x = max(1, int(round(width * 0.09)))
        margin_top = max(1, int(round(height * 0.08)))
        margin_bottom = max(1, int(round(height * 0.04)))

        roi["x1"] = max(0, int(roi["x1"]) - margin_x)
        roi["x2"] = min(width - 1, int(roi["x2"]) + margin_x)
        roi["y1"] = max(0, int(roi["y1"]) - margin_top)
        roi["y2"] = min(height - 1, int(roi["y2"]) + margin_bottom)

        pixel_width = max(0, int(roi["x2"]) - int(roi["x1"]))
        pixel_height = max(0, int(roi["y2"]) - int(roi["y1"]))
        total_pixels = float(width * height) if width and height else 0.0
        if pixel_width and pixel_height:
            pixel_area = float(pixel_width * pixel_height)
            roi["pixel_area"] = pixel_area
            roi["pixel_coverage"] = (pixel_area / total_pixels) if total_pixels else 0.0
            roi["normalized_area"] = roi["pixel_coverage"]

        roi["expanded_for_clima"] = True
        self._last_roi = roi

    # ------------------------------------------------------------------
    def _register_quality_check(self, valid: bool, reason: Optional[str]) -> None:
        if self._metrics:
            self._metrics.register_quality_check(valid, reason)
        if not valid and reason:
            self._quality_rejections[reason] += 1

    # ------------------------------------------------------------------
    def _validate_landmarks(
        self,
        frame: Any,
        results: Any,
        landmarks: Any,
        image_width: int,
        image_height: int,
    ) -> bool:
        hand_score = 1.0
        try:
            classifications = results.multi_handedness[0].classification
            if classifications:
                hand_score = float(classifications[0].score)
        except (AttributeError, IndexError, TypeError):
            hand_score = 1.0

        if hand_score < QUALITY_MIN_HAND_SCORE:
            self._register_quality_check(False, "low_confidence")
            return False

        points = getattr(landmarks, "landmark", [])
        if len(points) < QUALITY_MIN_LANDMARKS:
            self._register_quality_check(False, "incomplete_landmarks")
            return False

        if image_width <= 0 or image_height <= 0:
            self._register_quality_check(False, "invalid_dimensions")
            return False

        x_coords = [float(lm.x) for lm in points]
        y_coords = [float(lm.y) for lm in points]
        if not x_coords or not y_coords:
            self._register_quality_check(False, "empty_landmarks")
            return False

        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        out_of_bounds = (
            min_x < -0.05 or min_y < -0.05 or max_x > 1.05 or max_y > 1.05
        )
        extreme_bounds = (
            min_x < -0.2 or min_y < -0.2 or max_x > 1.2 or max_y > 1.2
        )
        if out_of_bounds:
            if self._lenient_quality and not extreme_bounds:
                _debug_decision(
                    "ROI fuera de límites pero permitido (min=(%.3f, %.3f), max=(%.3f, %.3f))",
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                )
                self._register_quality_check(True, "roi_out_of_bounds_warning")
            else:
                self._register_quality_check(False, "roi_out_of_bounds")
                return False

        if (
            min_x <= QUALITY_EDGE_MARGIN
            or min_y <= QUALITY_EDGE_MARGIN
            or max_x >= (1.0 - QUALITY_EDGE_MARGIN)
            or max_y >= (1.0 - QUALITY_EDGE_MARGIN)
        ):
            if self._lenient_quality:
                _debug_decision(
                    "Mano cerca del borde, marcando warning pero sin descartar"
                )
                self._register_quality_check(True, "hand_near_edge_warning")
            else:
                self._register_quality_check(False, "hand_near_edge")
                return False

        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        if width < QUALITY_MIN_BBOX_SIDE or height < QUALITY_MIN_BBOX_SIDE or area < QUALITY_MIN_BBOX_AREA:
            self._register_quality_check(False, "small_bbox")
            return False

        pixel_width = self._normalised_to_pixel(max_x, image_width) - self._normalised_to_pixel(min_x, image_width)
        pixel_height = self._normalised_to_pixel(max_y, image_height) - self._normalised_to_pixel(min_y, image_height)
        if pixel_width <= 6 or pixel_height <= 6:
            self._register_quality_check(False, "roi_too_small")
            return False

        if QUALITY_BLUR_THRESHOLD:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if variance < QUALITY_BLUR_THRESHOLD:
                    self._register_quality_check(False, "blur")
                    return False
            except Exception:
                # If blur detection fails we do not discard the frame.
                pass

        return True

    # ------------------------------------------------------------------
    def _smooth_landmarks(self) -> List[LandmarkPoint]:
        if not self._landmark_buffer:
            return []

        buffer = list(self._landmark_buffer)
        count = len(buffer)
        length = len(buffer[0])
        smoothed: List[LandmarkPoint] = []
        for index in range(length):
            avg_x = sum(item[index][0] for item in buffer) / count
            avg_y = sum(item[index][1] for item in buffer) / count
            avg_z = sum(item[index][2] for item in buffer) / count
            smoothed.append((avg_x, avg_y, avg_z))
        return smoothed

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_features(coords: Iterable[LandmarkPoint]) -> List[float]:
        coords = list(coords)
        if not coords:
            return []

        x_coords = [point[0] for point in coords]
        y_coords = [point[1] for point in coords]
        min_x = min(x_coords)
        min_y = min(y_coords)

        data_aux: List[float] = []
        for point in coords:
            x, y = point[0], point[1]
            data_aux.append(x - min_x)
            data_aux.append(y - min_y)

        return data_aux

    # ------------------------------------------------------------------
    def last_landmarks(self) -> Optional[List[LandmarkPoint]]:
        if not self._last_landmarks:
            return None
        return [tuple(point) for point in self._last_landmarks]


class EventStream:
    """Minimal Server-Sent Events (SSE) broadcaster."""

    def __init__(self) -> None:
        self._clients: Dict[int, "_SSEClient"] = {}
        self._lock = threading.Lock()
        self._sequence = 0

    def register(self, handler: "HelenRequestHandler") -> int:
        with self._lock:
            self._sequence += 1
            client_id = self._sequence
            self._clients[client_id] = _SSEClient(client_id, handler)
            LOGGER.info("SSE client %s connected from %s", client_id, handler.client_address)
            return client_id

    def unregister(self, client_id: int) -> None:
        with self._lock:
            client = self._clients.pop(client_id, None)
        if client is not None:
            LOGGER.info("SSE client %s disconnected", client_id)
            client.close()

    def broadcast(self, payload: Dict[str, Any]) -> None:
        message = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        frame = b"data: " + message + b"\n\n"

        dead: List[int] = []
        with self._lock:
            for client_id, client in self._clients.items():
                try:
                    client.write(frame)
                except ConnectionError:
                    dead.append(client_id)

        for client_id in dead:
            self.unregister(client_id)

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)


class _SSEClient:
    def __init__(self, client_id: int, handler: "HelenRequestHandler") -> None:
        self.client_id = client_id
        self._handler = handler
        self._lock = threading.Lock()
        self._closed = False

    def write(self, data: bytes) -> None:
        if self._closed:
            raise ConnectionError("SSE connection already closed")

        with self._lock:
            try:
                self._handler.wfile.write(data)
                self._handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError) as exc:  # pragma: no cover
                self._closed = True
                raise ConnectionError("client disconnected") from exc

    def close(self) -> None:
        with self._lock:
            self._closed = True
            with contextlib.suppress(Exception):
                self._handler.wfile.flush()


class GesturePipeline:
    """Background thread that feeds predictions to the runtime."""

    def __init__(
        self,
        runtime: "HelenRuntime",
        interval_s: float = 0.12,
        *,
        frame_stride: int = 1,
    ) -> None:
        self._runtime = runtime
        self._interval = max(0.01, float(interval_s))
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._sequence = 0
        self._frame_stride = max(1, int(frame_stride))
        self._stride_cursor = 0

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def _run(self) -> None:
        LOGGER.info("Gesture pipeline started")
        while self._running.is_set():
            self._runtime.register_heartbeat()
            try:
                features, source_label = self._runtime.stream.next(timeout=max(1.0, self._interval * 6))
            except TimeoutError as timeout_error:
                LOGGER.debug("Pipeline timeout waiting for hand landmarks: %s", timeout_error)
                continue
            except Exception as error:  # pragma: no cover - unexpected runtime failure
                self._runtime.report_error(f"stream_error: {error}")
                time.sleep(0.5)
                continue

            if self._frame_stride > 1:
                self._stride_cursor = (self._stride_cursor + 1) % self._frame_stride
                if self._stride_cursor != 1:
                    time.sleep(self._interval)
                    continue

            try:
                transformed = self._runtime.feature_normalizer.transform(features)
            except Exception as error:  # pragma: no cover - unexpected normalization failure
                LOGGER.warning("No se pudo normalizar el frame: %s", error)
                transformed = list(features)

            try:
                start = time.perf_counter()
                prediction: Prediction = self._runtime.classifier.predict(transformed)
                latency_ms = (time.perf_counter() - start) * 1000.0
            except Exception as error:  # pragma: no cover - classifier failure
                self._runtime.report_error(f"classifier_error: {error}")
                time.sleep(0.5)
                continue

            timestamp = time.time()
            landmarks: Optional[Sequence[LandmarkPoint]] = None
            last_landmarks_getter = getattr(self._runtime.stream, "last_landmarks", None)
            if callable(last_landmarks_getter):
                with contextlib.suppress(Exception):
                    landmarks_candidate = last_landmarks_getter()
                    if landmarks_candidate:
                        landmarks = list(landmarks_candidate)
            _debug_decision(
                "Predicción LSTM: label=%s score=%.3f hint=%s latency=%.1f ms",
                prediction.label,
                prediction.score,
                source_label or "",
                latency_ms,
            )
            decision = self._runtime.decision_engine.process(
                prediction,
                timestamp=timestamp,
                hint_label=source_label,
                latency_ms=latency_ms,
                landmarks=landmarks,
            )

            self._runtime.clear_error()

            if decision.emit:
                event = self._runtime.build_event(
                    label=decision.label,
                    score=decision.score,
                    latency_ms=latency_ms,
                    timestamp=timestamp,
                    sequence=self._sequence,
                    origin="pipeline",
                    hint_label=decision.hint_label,
                    payload=decision.payload,
                )
                self._runtime.push_prediction(event)
                self._sequence += 1
            time.sleep(self._interval)

        LOGGER.info("Gesture pipeline stopped")


class HelenRuntime:
    """Holds application state shared across HTTP handlers."""

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig()

        provided_mode = getattr(self.config, "display_mode", None)
        if config is None or provided_mode is None:
            active_mode = DISPLAY_MODE_STORE.cached()
        else:
            active_mode = DISPLAY_MODE_STORE.save(provided_mode)
        active_mode = _normalize_display_mode(active_mode)
        self.config.display_mode = active_mode

        profile_override = getattr(self.config, "camera_profile", None)
        profile = profile_override if profile_override is not None else _profile_for_mode(active_mode)
        self.config.camera_profile = profile

        self._apply_runtime_defaults(profile)
        self._configure_mode_runtime(active_mode, profile)
        self.session_id = uuid.uuid4().hex
        self.started_at = time.time()
        self.event_stream = EventStream()
        self.metrics = GestureMetrics()
        self.vision_snapshot = VISION_RUNTIME_SNAPSHOT
        self._camera_selection: Optional[CameraSelection] = None
        if self.config.enable_camera:
            self._camera_selection = self._ensure_camera_selection(force=False)

        self.dataset_info = {"path": None, "primary_available": False, "using_fallback": False, "exists": False}

        self.feature_normalizer = FeatureNormalizer()
        self.geometry_verifier = self._create_geometry_verifier()
        self.decision_engine = GestureDecisionEngine(
            metrics=self.metrics,
            geometry_verifier=self.geometry_verifier,
            per_label_consensus={"Clima": CLIMA_CONSENSUS_OVERRIDE},
        )
        self._log_clima_tuning()

        classifier, classifier_meta = self._create_classifier()
        self.classifier = classifier
        self.model_source = classifier_meta.get("source", "")
        self.model_loaded = classifier_meta.get("loaded", False)
        self.dataset_info = {
            "path": classifier_meta.get("path"),
            "primary_available": classifier_meta.get("loaded", False),
            "using_fallback": False,
            "exists": classifier_meta.get("loaded", False),
        }

        stream, stream_meta = self._create_stream()
        self.stream = stream
        self.stream_source = stream_meta["source"]

        self.pipeline = GesturePipeline(
            self,
            interval_s=self.config.poll_interval_s,
            frame_stride=self.config.process_every_n,
        )
        self.lock = threading.Lock()
        self.latency_history: Deque[float] = deque(maxlen=240)
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.last_prediction_at: Optional[float] = None
        self.last_heartbeat = 0.0
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    def _apply_runtime_defaults(self, profile: Optional[PiCameraProfile]) -> None:
        defaults = _resolve_runtime_defaults(profile)

        detection = self.config.detection_confidence
        if detection is None:
            detection = defaults.detection_confidence
        self.config.detection_confidence = max(0.2, min(float(detection), 0.99))

        tracking = self.config.tracking_confidence
        if tracking is None:
            tracking = defaults.tracking_confidence
        self.config.tracking_confidence = max(0.2, min(float(tracking), 0.99))

        poll_interval = self.config.poll_interval_s
        if poll_interval is None:
            poll_interval = defaults.poll_interval
        self.config.poll_interval_s = max(0.02, float(poll_interval))

        stride = self.config.process_every_n
        if stride is None:
            stride = defaults.frame_stride
        self.config.process_every_n = max(1, int(stride))

    # ------------------------------------------------------------------
    def _configure_mode_runtime(self, mode: str, profile: Optional[PiCameraProfile]) -> None:
        if profile:
            tuned_interval = float(profile.poll_interval)
            self.config.poll_interval_s = tuned_interval
            stride = max(1, int(profile.process_every_n))
            self.config.process_every_n = stride
            LOGGER.info(
                "Modo %s activo (%s): %sx%s @ %s FPS; poll_interval=%.3f s; frame_stride=%s",
                mode,
                profile.model,
                profile.width,
                profile.height,
                profile.fps,
                tuned_interval,
                stride,
            )
        else:
            stride = max(1, int(getattr(self.config, "process_every_n", 3) or 3))
            self.config.process_every_n = stride
            LOGGER.info(
                "Modo %s activo (perfil estándar); poll_interval=%.3f s; frame_stride=%s",
                mode,
                float(self.config.poll_interval_s),
                stride,
            )

    # ------------------------------------------------------------------
    def _log_clima_tuning(self) -> None:
        try:
            thresholds = self.decision_engine.thresholds().get("Clima") or UPDATED_CLIMA_THRESHOLD
            override = self.decision_engine.consensus_overrides().get("Clima")
            override_votes = (
                override.required_votes if override else DEFAULT_CONSENSUS_CONFIG.required_votes
            )
            override_window = (
                override.window_size if override else DEFAULT_CONSENSUS_CONFIG.window_size
            )
            LOGGER.info(
                "Ajuste 'C': threshold enter %.2f→%.2f; release %.2f→%.2f; consenso %s/%s→%s/%s",
                LEGACY_CLIMA_THRESHOLD.enter,
                thresholds.enter,
                LEGACY_CLIMA_THRESHOLD.release,
                thresholds.release,
                DEFAULT_CONSENSUS_CONFIG.required_votes,
                DEFAULT_CONSENSUS_CONFIG.window_size,
                override_votes,
                override_window,
            )
        except Exception as error:
            LOGGER.debug("No se pudo registrar los ajustes de Clima: %s", error)

    # ------------------------------------------------------------------
    def _ensure_camera_selection(self, *, force: bool) -> Optional[CameraSelection]:
        preferred: Optional[Union[int, str]] = None
        camera_spec = getattr(self.config, "camera_device", None)
        if camera_spec is None:
            camera_spec = getattr(self.config, "camera_index", None)
        if isinstance(camera_spec, (int, str)) and camera_spec != "":
            preferred = camera_spec

        try:
            selection = camera_probe.ensure_camera_selection(
                force=force,
                preferred=preferred,
                logger=LOGGER,
                forced_backend=getattr(self.config, "camera_backend", None),
            )
        except Exception as error:  # pragma: no cover - depends on environment
            LOGGER.warning("Auto-probe de cámara falló: %s", error)
            return None

        if selection:
            if selection.device:
                self.config.camera_device = selection.device
            if selection.index is not None:
                self.config.camera_index = selection.index
            elif selection.device:
                self.config.camera_index = selection.device
        return selection

    # ------------------------------------------------------------------
    def _create_geometry_verifier(self) -> Optional[LandmarkGeometryVerifier]:
        return LandmarkGeometryVerifier()

    # ------------------------------------------------------------------
    def _create_classifier(self) -> Tuple[Any, Dict[str, Any]]:
        RUNTIME_MODEL_CONFIG.announce()
        backend = RUNTIME_MODEL_CONFIG.effective_backend

        model_dir = RUNTIME_MODEL_CONFIG.tf_model_dir(getattr(self.config, "tf_model_dir", None))
        model_dir = Path(model_dir)
        try:
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"No se encontró el modelo TensorFlow en {model_dir!s}. "
                    "Configura tf_model_dir o coloca el SavedModel entrenado."
                )

            classifier = TensorFlowSequenceGestureClassifier(model_dir)
            LOGGER.info(
                "Modelo LSTM cargado desde %s (backend efectivo=%s)",
                model_dir,
                backend,
            )
            return classifier, {"source": TensorFlowSequenceGestureClassifier.source, "loaded": True, "path": str(model_dir)}
        except Exception as error:
            LOGGER.error("No se pudo cargar el modelo LSTM (%s). Se usará clasificador dummy: %s", backend, error)
            fallback = DummyGestureClassifier()
            return fallback, {"source": fallback.source, "loaded": False, "path": str(model_dir)}

    # ------------------------------------------------------------------
    def _create_stream(self) -> Tuple[Any, Dict[str, Any]]:
        if self.config.enable_camera:
            selection = getattr(self, "_camera_selection", None)
            try:
                stream = CameraGestureStream(
                    camera_index=self.config.camera_index,
                    detection_confidence=self.config.detection_confidence,
                    tracking_confidence=self.config.tracking_confidence,
                    metrics=self.metrics,
                    profile=self.config.camera_profile,
                    selection=selection,
                    forced_backend=self.config.camera_backend,
                    width_override=self.config.camera_width,
                    height_override=self.config.camera_height,
                )
                target = selection.device if selection and selection.device else self.config.camera_index
                LOGGER.info("Usando cámara física en %s", target)
                return stream, {"source": CameraGestureStream.source}
            except Exception as error:
                LOGGER.warning("No se pudo inicializar la cámara: %s", error)
                if not self.config.fallback_to_synthetic:
                    raise
                refreshed = self._ensure_camera_selection(force=True)
                if refreshed:
                    self._camera_selection = refreshed
                    try:
                        stream = CameraGestureStream(
                            camera_index=self.config.camera_index,
                            detection_confidence=self.config.detection_confidence,
                            tracking_confidence=self.config.tracking_confidence,
                            metrics=self.metrics,
                            profile=self.config.camera_profile,
                            selection=refreshed,
                            forced_backend=self.config.camera_backend,
                            width_override=self.config.camera_width,
                            height_override=self.config.camera_height,
                        )
                        target = refreshed.device if refreshed.device else refreshed.index
                        LOGGER.info("Cámara reprovisionada automáticamente en %s", target)
                        return stream, {"source": CameraGestureStream.source}
                    except Exception as retry_error:
                        LOGGER.warning("Reintento de cámara fallido: %s", retry_error)
        raise RuntimeError("No se pudo inicializar la cámara y no hay flujo alternativo habilitado")

    # ------------------------------------------------------------------
    def start(self) -> None:
        self.pipeline.start()

    # ------------------------------------------------------------------
    def stop(self, *, export_report: bool = True) -> None:
        self.pipeline.stop()
        close_stream = getattr(self.stream, "close", None)
        if callable(close_stream):
            close_stream()
        if export_report:
            self._export_session_report()

    # ------------------------------------------------------------------
    def register_heartbeat(self) -> None:
        with self.lock:
            self.last_heartbeat = time.time()

    # ------------------------------------------------------------------
    def clear_error(self) -> None:
        with self.lock:
            self.last_error = None

    # ------------------------------------------------------------------
    def mode_snapshot(self, persisted_mode: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            active = self.config.display_mode
            poll_interval = float(self.config.poll_interval_s)
            stride = int(self.config.process_every_n)
            profile = self.config.camera_profile
            stream_source = self.stream_source

        snapshot: Dict[str, Any] = {
            "active": active,
            "persisted": _normalize_display_mode(persisted_mode or DISPLAY_MODE_STORE.cached()),
            "poll_interval_s": poll_interval,
            "process_every_n": stride,
            "stream_source": stream_source,
            "camera_profile": None,
        }

        if profile:
            snapshot["camera_profile"] = {
                "model": profile.model,
                "width": profile.width,
                "height": profile.height,
                "fps": profile.fps,
                "poll_interval": profile.poll_interval,
                "process_every_n": profile.process_every_n,
            }

        return snapshot

    # ------------------------------------------------------------------
    def apply_display_mode(self, mode: str) -> Dict[str, Any]:
        normalized = _normalize_display_mode(mode)
        persisted = DISPLAY_MODE_STORE.save(normalized)

        with self.lock:
            current_mode = self.config.display_mode
        if normalized == current_mode:
            return self.mode_snapshot(persisted_mode=persisted)

        LOGGER.info("Cambiando modo de visualización: %s → %s", current_mode, normalized)

        self.stop(export_report=False)

        with self.lock:
            self.config.display_mode = normalized
            self.config.camera_profile = _profile_for_mode(normalized)
            profile = self.config.camera_profile
            self._configure_mode_runtime(normalized, profile)
            stream, stream_meta = self._create_stream()
            self.stream = stream
            self.stream_source = stream_meta.get("source", "")
            self.pipeline = GesturePipeline(
                self,
                interval_s=self.config.poll_interval_s,
                frame_stride=self.config.process_every_n,
            )

        self.pipeline.start()
        return self.mode_snapshot(persisted_mode=persisted)

    # ------------------------------------------------------------------
    def engine_status(self) -> Dict[str, Any]:
        thresholds = {
            label: {"enter": round(cfg.enter, 3), "release": round(cfg.release, 3)}
            for label, cfg in self.decision_engine.thresholds().items()
        }

        consensus_overrides = {
            label: {
                "window_size": override.window_size,
                "required_votes": override.required_votes,
            }
            for label, override in self.decision_engine.consensus_overrides().items()
        }

        consensus_payload = {
            "default": {
                "window_size": self.decision_engine.consensus_config.window_size,
                "required_votes": self.decision_engine.consensus_config.required_votes,
            },
            "overrides": consensus_overrides,
        }

        stream_status = getattr(self.stream, "status", lambda: {})()
        mode_snapshot = self.mode_snapshot()

        payload = {
            "mode": mode_snapshot,
            "ui_mode": mode_snapshot.get("active", DEFAULT_DISPLAY_MODE),
            "thresholds": thresholds,
            "consensus": consensus_payload,
            "pipeline": {
                "poll_interval_s": float(self.config.poll_interval_s),
                "process_every_n": int(self.config.process_every_n),
                "running": self.pipeline.is_running(),
            },
            "stream": stream_status,
            "vision": self.vision_snapshot,
        }

        if self._camera_selection:
            payload["camera_selection"] = self._camera_selection.to_dict()

        return payload

    # ------------------------------------------------------------------
    def report_error(self, message: str) -> None:
        LOGGER.error("%s", message)
        with self.lock:
            self.last_error = message

    # ------------------------------------------------------------------
    def build_event(
        self,
        *,
        label: str,
        score: float,
        latency_ms: float,
        timestamp: float,
        sequence: int,
        origin: str,
        hint_label: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        collapsed = label.strip().lower()
        is_activation = collapsed in ACTIVATION_ALIASES

        base_event: Dict[str, Any] = {
            "session_id": self.session_id,
            "sequence": sequence,
            "timestamp": _iso_timestamp(timestamp),
            "character": label,
            "gesture": label,
            "score": round(float(score), 4),
            "latency_ms": round(float(latency_ms), 3),
            "source": origin,
            "numeric": collapsed.isdigit(),
        }

        if is_activation:
            base_event["active"] = True
            base_event.setdefault("state", label)

        if hint_label and hint_label != label:
            base_event["label_hint"] = hint_label

        if payload:
            base_event.update(payload)

        return base_event

    # ------------------------------------------------------------------
    def push_prediction(self, event: Dict[str, Any]) -> None:
        label = str(event.get("gesture") or event.get("character") or "")
        if label == "Clima":
            expand_roi = getattr(self.stream, "expand_last_roi_for_clima", None)
            if callable(expand_roi):
                try:
                    expand_roi()
                except Exception as error:
                    LOGGER.debug("No se pudo expandir el ROI para Clima: %s", error)

        previous_event: Optional[Dict[str, Any]]
        with self.lock:
            previous_event = self.last_prediction
            self.last_prediction = event
            self.last_prediction_at = time.time()
            self.last_heartbeat = time.time()
            self.latency_history.append(float(event.get("latency_ms", 0.0)))

        LOGGER.debug("Broadcasting event: %s", event)
        self.event_stream.broadcast(event)

        previous_label = ""
        if previous_event:
            previous_label = str(previous_event.get("gesture") or previous_event.get("character") or "")

        if label == "Clima" and previous_label == "Start":
            poll_interval = float(getattr(self.config, "poll_interval_s", DEFAULT_POLL_INTERVAL_S) or DEFAULT_POLL_INTERVAL_S)
            approx_fps = 1.0 / max(poll_interval, 1e-3)
            LOGGER.info(
                "Secuencia H→C confirmada (latencia %.1f ms, ~%.1f fps)",
                float(event.get("latency_ms", 0.0)),
                approx_fps,
            )

    # ------------------------------------------------------------------
    def receive_external_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = time.time()
        sequence = int(payload.get("sequence", 0))
        label = str(payload.get("gesture") or payload.get("character") or "")
        if not label:
            raise ValueError("Payload must include a gesture label")

        score = float(payload.get("score", 0.0))
        latency_ms = float(payload.get("latency_ms", 0.0))

        event = self.build_event(
            label=label,
            score=score,
            latency_ms=latency_ms,
            timestamp=timestamp,
            sequence=sequence,
            origin="http",
            payload={"raw": payload},
        )

        self.push_prediction(event)
        return event

    # ------------------------------------------------------------------
    def _latency_snapshot(self) -> Dict[str, float]:
        with self.lock:
            samples = list(self.latency_history)

        if samples:
            average = statistics.fmean(samples) if hasattr(statistics, "fmean") else sum(samples) / len(samples)
            sorted_samples = sorted(samples)
            index = min(len(sorted_samples) - 1, int(len(sorted_samples) * 0.95))
            p95 = sorted_samples[index]
            maximum = max(sorted_samples)
        else:
            average = 0.0
            p95 = 0.0
            maximum = 0.0

        return {"avg_ms": average, "p95_ms": p95, "max_ms": maximum, "count": len(samples)}

    # ------------------------------------------------------------------
    def _export_session_report(self) -> None:
        try:
            latency_stats = self._latency_snapshot()
            dataset_info = dict(self.dataset_info)
            dataset_info["normalizer"] = self.feature_normalizer.snapshot()
            report_path = REPO_ROOT / "reports" / "gesture_session_report.md"
            self.metrics.dump_report(
                markdown_path=report_path,
                thresholds=self.decision_engine.thresholds(),
                consensus=self.decision_engine.consensus_config,
                dataset_info=dataset_info,
                latency_stats=latency_stats,
                label_consensus=self.decision_engine.consensus_overrides(),
            )
            LOGGER.info("Reporte de métricas actualizado en %s", report_path)
        except Exception as error:  # pragma: no cover - escritura opcional
            LOGGER.warning("No se pudo escribir el reporte de métricas: %s", error)

    # ------------------------------------------------------------------
    def health(self) -> HealthSnapshot:
        with self.lock:
            last_prediction = self.last_prediction
            last_prediction_at = self.last_prediction_at
            avg_latency = (
                sum(self.latency_history) / len(self.latency_history)
                if self.latency_history
                else 0.0
            )
            last_error = self.last_error
            heartbeat_age = time.time() - self.last_heartbeat if self.last_heartbeat else None

        pipeline_running = self.pipeline.is_running()
        stream_status = getattr(self.stream, "status", lambda: {})()
        camera_ok = bool(stream_status.get("healthy")) if self.stream_source == "camera" else True

        selection_info = stream_status.get("selection") or {}
        device_path = stream_status.get("device_path") or selection_info.get("device")
        camera_index = stream_status.get("camera_index")
        if camera_index is None and selection_info.get("index") is not None:
            camera_index = selection_info.get("index")
        resolution = None
        width = selection_info.get("width")
        height = selection_info.get("height")
        if width and height:
            resolution = f"{int(width)}x{int(height)}"
        fps_value = selection_info.get("fps")
        pixel_format = stream_status.get("pixel_format") or selection_info.get("pixel_format")
        probe_latency = stream_status.get("probe_latency_ms")

        status = "HEALTHY"
        if last_error:
            status = "ERROR"
        elif not pipeline_running or (heartbeat_age is not None and heartbeat_age > 5.0) or not camera_ok or not self.model_loaded:
            status = "DEGRADED"

        return HealthSnapshot(
            status=status,
            model_loaded=self.model_loaded,
            model_source=self.model_source,
            session_id=self.session_id,
            pipeline_running=pipeline_running,
            stream_source=self.stream_source,
            clients=self.event_stream.client_count(),
            uptime_s=time.time() - self.started_at,
            last_prediction=last_prediction,
            last_prediction_at=_iso_timestamp(last_prediction_at) if last_prediction_at else None,
            avg_latency_ms=round(avg_latency, 3),
            camera_ok=camera_ok,
            camera_index=camera_index,
            camera_device=device_path,
            camera_backend=stream_status.get("capture_backend"),
            camera_resolution=resolution,
            camera_fps=fps_value,
            camera_pixel_format=pixel_format,
            camera_probe_latency_ms=probe_latency,
            camera_last_capture=(
                _iso_timestamp(stream_status["last_capture"])
                if stream_status.get("last_capture")
                else None
            ),
            camera_last_error=stream_status.get("last_error"),
            last_error=last_error,
        )


class HelenRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving the SPA and the SSE endpoints."""

    server_version = "HelenHTTP/1.0"
    runtime: HelenRuntime  # populated at server construction time

    def __init__(self, *args: Any, runtime: HelenRuntime, **kwargs: Any) -> None:
        self.runtime = runtime
        super().__init__(*args, directory=str(FRONTEND_ROOT), **kwargs)

    def _write_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover - forwarded to logging
        LOGGER.info("HTTP %s - %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: D401 - inherited API
        path = self.path.split("?", 1)[0]

        if path in {"", "/"}:
            self.path = "/index.html"
        else:
            self.path = path

        if path in HEALTH_ENDPOINTS:
            snapshot = self.runtime.health()
            self._write_json(snapshot.__dict__)
            return

        if path == "/engine/status":
            payload = self.runtime.engine_status()
            self._write_json(payload)
            return

        if path == "/net/online":
            payload = check_online_status()
            status_info = current_wifi_status()
            if status_info.get("iface") and not payload.get("iface"):
                payload["iface"] = status_info.get("iface")
            if status_info.get("connected_ssid") and not payload.get("connected_ssid"):
                payload["connected_ssid"] = status_info.get("connected_ssid")
            self._write_json(payload)
            return

        if path == "/net/scan":
            try:
                networks = scan_wifi_networks()
            except RuntimeError as error:
                self._write_json({"networks": [], "error": str(error)}, status=HTTPStatus.BAD_GATEWAY)
                return
            self._write_json({"networks": networks, "timestamp": time.time()})
            return

        if path == "/net/status":
            self._write_json(current_wifi_status())
            return

        if path == "/mode/get":
            snapshot = self.runtime.mode_snapshot()
            self._write_json(snapshot)
            return

        if path.startswith("/events"):
            self._handle_sse()
            return

        super().do_GET()

    # ------------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: D401 - inherited API
        path = self.path.split("?", 1)[0]

        if path == "/net/connect":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._write_json({"connected": False, "reason": "JSON inválido"}, status=HTTPStatus.BAD_REQUEST)
                return

            ssid = str(data.get("ssid", "")).strip()
            password = data.get("password", "")
            if not ssid:
                self._write_json({"connected": False, "reason": "SSID requerido"}, status=HTTPStatus.BAD_REQUEST)
                return

            try:
                success, reason = connect_wifi(ssid, str(password or ""))
            except RuntimeError as error:
                self._write_json({"connected": False, "reason": str(error)}, status=HTTPStatus.BAD_GATEWAY)
                return

            status_info = current_wifi_status()
            connected_ssid = status_info.get("connected_ssid", "")
            is_connected = bool(success and connected_ssid and connected_ssid.lower() == ssid.lower())
            payload = {
                "connected": is_connected,
                "reason": reason or "",
                "status": status_info,
            }
            self._write_json(payload)
            return

        if path == "/mode/set":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._write_json({"ok": False, "error": "JSON inválido"}, status=HTTPStatus.BAD_REQUEST)
                return

            mode_value = data.get("mode", "")
            try:
                snapshot = self.runtime.apply_display_mode(str(mode_value))
            except Exception as error:  # pragma: no cover - runtime dependent
                LOGGER.error("No se pudo aplicar el modo de visualización: %s", error)
                self._write_json({"ok": False, "error": str(error)}, status=HTTPStatus.BAD_GATEWAY)
                return

            self._write_json({"mode": snapshot["active"], "snapshot": snapshot})
            return

        if path == "/gestures/gesture-key":
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON payload")
                return

            try:
                event = self.runtime.receive_external_payload(data)
            except ValueError as error:
                self.send_error(HTTPStatus.BAD_REQUEST, str(error))
                return

            body = json.dumps({"status": "ok", "event": event}).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Ruta no encontrada")

    # ------------------------------------------------------------------
    def _handle_sse(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        client_id = self.runtime.event_stream.register(self)

        warmup = {
            "session_id": self.runtime.session_id,
            "sequence": -1,
            "timestamp": _iso_timestamp(time.time()),
            "message": "connected",
            "source": "sse",
        }
        self.runtime.event_stream.broadcast(warmup)

        try:
            while True:
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - network race
            pass
        finally:
            self.runtime.event_stream.unregister(client_id)


def run(host: str = "0.0.0.0", port: int = 5000, *, config: Optional[RuntimeConfig] = None) -> None:
    runtime = HelenRuntime(config=config)
    runtime.start()

    handler_factory = partial(HelenRequestHandler, runtime=runtime)
    with ThreadingHTTPServer((host, port), handler_factory) as httpd:
        LOGGER.info("HELEN backend serving from %s:%s", host, port)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:  # pragma: no cover - manual shutdown
            LOGGER.info("Shutting down backend")
        finally:
            runtime.stop()


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


__all__ = ["HelenRuntime", "HelenRequestHandler", "RuntimeConfig", "run", "main"]


def _parse_camera_spec(value: Optional[str]) -> Optional[Union[int, str]]:
    if value is None:
        return None
    candidate = value.strip()
    if not candidate or candidate.lower() == "auto":
        return None
    if candidate.isdigit():
        return int(candidate)
    try:
        return int(candidate)
    except ValueError:
        return candidate


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HELEN backend server")
    parser.add_argument("--host", default="0.0.0.0", help="Dirección de enlace del servidor HTTP")
    parser.add_argument("--port", type=int, default=5000, help="Puerto del servidor HTTP")
    parser.add_argument(
        "--camera",
        "--camera-index",
        dest="camera_index",
        default=None,
        help="Índice numérico (0,1,2) o ruta /dev/videoX a utilizar. Usa 'auto' para autodetección",
        metavar="INDEX|PATH",
    )
    parser.add_argument(
        "--camera-backend",
        dest="camera_backend",
        default=None,
        help="Backend de captura a forzar (directshow, dshow, v4l2)",
        metavar="NAME",
    )
    parser.add_argument(
        "--camera-width",
        dest="camera_width",
        type=int,
        default=None,
        help="Ancho deseado de captura en píxeles",
    )
    parser.add_argument(
        "--camera-height",
        dest="camera_height",
        type=int,
        default=None,
        help="Alto deseado de captura en píxeles",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=None,
        help="Umbral de detección de MediaPipe (se infiere según plataforma si se omite)",
    )
    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=None,
        help="Umbral de seguimiento de MediaPipe (se infiere según plataforma si se omite)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Intervalo entre inferencias en segundos (se ajusta automáticamente por plataforma)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="Procesa un frame de cada N muestras para reducir carga (>=1)",
    )
    parser.add_argument("--no-camera", action="store_true", help="Desactiva el uso de cámara física")
    parser.add_argument(
        "--no-synthetic-fallback",
        action="store_true",
        help="Falla si la cámara no está disponible en lugar de usar el dataset sintético",
    )

    args = parser.parse_args(argv)
    camera_spec = _parse_camera_spec(args.camera_index)
    camera_device = camera_spec if isinstance(camera_spec, str) else None

    backend_preference = camera_probe.normalize_backend_name(args.camera_backend)

    frame_stride = args.frame_stride if args.frame_stride is not None else None
    if frame_stride is not None:
        frame_stride = max(1, frame_stride)

    width_override = args.camera_width if args.camera_width and args.camera_width > 0 else None
    height_override = args.camera_height if args.camera_height and args.camera_height > 0 else None

    config = RuntimeConfig(
        camera_index=camera_spec,
        camera_device=camera_device,
        camera_backend=backend_preference,
        camera_width=width_override,
        camera_height=height_override,
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence,
        poll_interval_s=args.poll_interval,
        enable_camera=not args.no_camera,
        fallback_to_synthetic=not args.no_synthetic_fallback,
        process_every_n=frame_stride,
    )

    run(args.host, args.port, config=config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
