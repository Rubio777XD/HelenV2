"""HELEN backend exposing Server-Sent Events for gesture navigation.

The original repository relied on several disparate scripts to run the camera
pipeline, invoke the machine-learning model and broadcast results to the
frontend.  This module consolidates that behaviour in a single HTTP server that
is production-ready:

* Uses the real ``model.p`` XGBoost model when available.
* Falls back to the synthetic centroid classifier only when the production
  model or the camera pipeline cannot be initialised.
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
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple
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

from Hellen_model_RN.helpers import labels_dict
from Hellen_model_RN.simple_classifier import (
    Prediction,
    SimpleGestureClassifier,
    SyntheticGestureStream,
)


LOGGER = logging.getLogger("helen.backend")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

with contextlib.suppress(Exception):
    import absl.logging as absl_logging  # type: ignore

    absl_logging.set_verbosity(absl_logging.WARNING)
    handler = absl_logging.get_absl_handler()
    handler.setLevel(logging.WARNING)


def _resolve_repo_root() -> Path:
    """Return the runtime root, compatible with PyInstaller bundles."""

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
FRONTEND_ROOT = REPO_ROOT / "helen"
MODEL_DIR = REPO_ROOT / "Hellen_model_RN"
MODEL_PATH = MODEL_DIR / "model.p"

PRIMARY_DATASET_NAME = "data.pickle"
LEGACY_DATASET_NAME = "data1.pickle"

# port: int = 5000  # Referencia para pruebas de integración (mantener sincronizado con run()).


_MISSING_DATASET_NOTIFIED: set[Path] = set()


def _default_dataset_path() -> Path:
    """Pick the best available dataset shipped with the build.

    ``data.pickle`` is the preferred bundle. When it is not present (for example
    in source checkouts where the large artifact is omitted) we fall back to the
    legacy ``data1.pickle`` file so development environments can keep working.
    The returned path may not exist – call sites are responsible for logging the
    appropriate warning to the operator.
    """

    candidate = MODEL_DIR / PRIMARY_DATASET_NAME
    if candidate.exists():
        LOGGER.debug("Dataset principal detectado: %s", candidate)
        return candidate

    _notify_missing_dataset(candidate)

    legacy = MODEL_DIR / LEGACY_DATASET_NAME
    if legacy.exists():
        LOGGER.info("Se utilizará el dataset legado %s", legacy)
        return legacy

    return candidate


def _notify_missing_dataset(dataset_path: Path) -> None:
    resolved = dataset_path.resolve()
    if resolved in _MISSING_DATASET_NOTIFIED:
        return

    _MISSING_DATASET_NOTIFIED.add(resolved)

    if dataset_path.name.lower() == PRIMARY_DATASET_NAME:
        LOGGER.warning(
            "data.pickle no encontrado (archivo grande omitido del repo, ver documentación de build). "
            "Fallback activo; la calibración puede degradar accuracy."
        )
    else:
        LOGGER.error("Dataset no encontrado en %s", dataset_path)


DATASET_PATH = _default_dataset_path()

TRACKED_GESTURES = {"Start", "Clima", "Reloj", "Inicio"}
GESTURE_ALIASES = {"Start": "H", "Clima": "C", "Reloj": "R", "Inicio": "I"}


@dataclass(frozen=True)
class ClassThreshold:
    enter: float
    release: float


DEFAULT_CLASS_THRESHOLDS: Dict[str, ClassThreshold] = {
    "Start": ClassThreshold(enter=0.62, release=0.52),
    "Clima": ClassThreshold(enter=0.65, release=0.55),
    "Reloj": ClassThreshold(enter=0.63, release=0.53),
    "Inicio": ClassThreshold(enter=0.62, release=0.52),
}


@dataclass(frozen=True)
class QualityProfile:
    blur_laplacian_min: float
    roi_min_coverage: float
    hand_range_px: Tuple[float, float]


@dataclass(frozen=True)
class TemporalProfile:
    consensus_n: int
    consensus_m: int
    cooldown_s: float
    listen_window_s: float
    min_pos_stability_var: float
    activation_delay_s: float = 0.45


@dataclass(frozen=True)
class ClassProfile:
    score_min: float
    angle_tol_deg: float
    norm_dev_max: float
    curvature_min: Optional[float] = None
    gap_ratio_range: Optional[Tuple[float, float]] = None
    missing_distal_allowance: int = 0
    missing_distal_strict_curvature: Optional[float] = None
    curvature_consistency_boost: float = 0.0
    curvature_consistency_window: int = 0
    curvature_consistency_min_frames: int = 0
    curvature_consistency_tolerance: float = 0.0
    curvature_consistency_min_curvature: Optional[float] = None


@dataclass(frozen=True)
class HysteresisProfile:
    on_offset: float
    off_delta: float


@dataclass(frozen=True)
class RateLimitProfile:
    frameskip_strict: int
    frameskip_balanced: int
    frameskip_relaxed: int
    fps_threshold: float


@dataclass(frozen=True)
class SensitivityProfile:
    mode: str
    quality: QualityProfile
    temporal: TemporalProfile
    classes: Dict[str, ClassProfile]


SENSITIVITY_CONFIG_PATH = REPO_ROOT / "config" / "thresholds.json"
SENSITIVITY_MODES = {"STRICT", "BALANCED", "RELAXED"}


@dataclass(frozen=True)
class ConsensusConfig:
    window_size: int = 4
    required_votes: int = 2


DEFAULT_CONSENSUS_CONFIG = ConsensusConfig()

ACTIVATION_DELAY = 0.45
SMOOTHING_WINDOW_SIZE = 4
COOLDOWN_SECONDS = 0.7
LISTENING_WINDOW_SECONDS = 6.0
COMMAND_DEBOUNCE_SECONDS = 0.75


QUALITY_BLUR_THRESHOLD = 28.0


def _default_sensitivity() -> Tuple[Dict[str, SensitivityProfile], HysteresisProfile, RateLimitProfile, int]:
    quality = QualityProfile(blur_laplacian_min=QUALITY_BLUR_THRESHOLD, roi_min_coverage=0.62, hand_range_px=(70, 560))
    temporal = TemporalProfile(
        consensus_n=DEFAULT_CONSENSUS_CONFIG.required_votes,
        consensus_m=DEFAULT_CONSENSUS_CONFIG.window_size,
        cooldown_s=COOLDOWN_SECONDS,
        listen_window_s=LISTENING_WINDOW_SECONDS,
        min_pos_stability_var=12.0,
        activation_delay_s=ACTIVATION_DELAY,
    )
    classes: Dict[str, ClassProfile] = {}
    for canonical, threshold in DEFAULT_CLASS_THRESHOLDS.items():
        classes[canonical] = ClassProfile(score_min=threshold.enter, angle_tol_deg=18.0, norm_dev_max=0.2)
    profiles = {
        mode: SensitivityProfile(mode=mode, quality=quality, temporal=temporal, classes=dict(classes))
        for mode in SENSITIVITY_MODES
    }
    hysteresis = HysteresisProfile(on_offset=0.0, off_delta=0.12)
    rate_limit = RateLimitProfile(frameskip_strict=4, frameskip_balanced=2, frameskip_relaxed=2, fps_threshold=25.0)
    return profiles, hysteresis, rate_limit, 0


def _load_sensitivity_profiles() -> Tuple[Dict[str, SensitivityProfile], HysteresisProfile, RateLimitProfile, int]:
    if not SENSITIVITY_CONFIG_PATH.exists():
        LOGGER.warning(
            "Archivo de configuración de sensibilidad no encontrado en %s; usando valores por defecto", SENSITIVITY_CONFIG_PATH
        )
        return _default_sensitivity()

    try:
        data = json.loads(SENSITIVITY_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as error:  # pragma: no cover - lectura de disco
        LOGGER.error("No se pudo leer thresholds.json: %s", error)
        return _default_sensitivity()

    version = int(data.get("version", 0) or 0)
    hysteresis_data = data.get("hysteresis", {})
    rate_data = data.get("rate_limit", {})
    hysteresis = HysteresisProfile(
        on_offset=float(hysteresis_data.get("on_offset", 0.0)),
        off_delta=float(hysteresis_data.get("off_delta", 0.12)),
    )
    rate_limit = RateLimitProfile(
        frameskip_strict=int(rate_data.get("frameskip_strict", 4) or 4),
        frameskip_balanced=int(rate_data.get("frameskip_balanced", 3) or 3),
        frameskip_relaxed=int(rate_data.get("frameskip_relaxed", 3) or 3),
        fps_threshold=float(rate_data.get("fps_threshold", 25.0) or 25.0),
    )

    modes_payload = data.get("modes", {})
    inverse_alias = {alias: gesture for gesture, alias in GESTURE_ALIASES.items()}
    profiles: Dict[str, SensitivityProfile] = {}

    for mode_name, payload in modes_payload.items():
        upper_mode = str(mode_name).strip().upper()
        if upper_mode not in SENSITIVITY_MODES:
            continue

        quality_data = payload.get("global", {}).get("quality", {})
        temporal_data = payload.get("global", {}).get("temporal", {})
        quality = QualityProfile(
            blur_laplacian_min=float(quality_data.get("blur_laplacian_min", QUALITY_BLUR_THRESHOLD)),
            roi_min_coverage=float(quality_data.get("roi_min_coverage", 0.62)),
            hand_range_px=tuple(float(v) for v in quality_data.get("hand_range_px", (70, 560)))[:2],
        )
        if len(quality.hand_range_px) != 2:
            quality = QualityProfile(
                blur_laplacian_min=quality.blur_laplacian_min,
                roi_min_coverage=quality.roi_min_coverage,
                hand_range_px=(70.0, 560.0),
            )

        temporal = TemporalProfile(
            consensus_n=int(temporal_data.get("consensus_N", DEFAULT_CONSENSUS_CONFIG.required_votes) or 1),
            consensus_m=int(temporal_data.get("consensus_M", DEFAULT_CONSENSUS_CONFIG.window_size) or 1),
            cooldown_s=float(temporal_data.get("cooldown_s", COOLDOWN_SECONDS)),
            listen_window_s=float(temporal_data.get("listen_window_s", LISTENING_WINDOW_SECONDS)),
            min_pos_stability_var=float(temporal_data.get("min_pos_stability_var", 12.0)),
            activation_delay_s=float(temporal_data.get("activation_delay_s", ACTIVATION_DELAY)),
        )

        class_profiles: Dict[str, ClassProfile] = {}
        for alias, class_payload in payload.get("classes", {}).items():
            alias_upper = str(alias).strip().upper()
            gesture = inverse_alias.get(alias_upper)
            if not gesture:
                continue
            score_min = float(class_payload.get("score_min", DEFAULT_CLASS_THRESHOLDS.get(gesture, ClassThreshold(0.6, 0.52)).enter))
            angle_tol = float(class_payload.get("angle_tol_deg", 18.0))
            norm_dev = float(class_payload.get("norm_dev_max", 0.2))
            curvature = class_payload.get("curvature_min")
            curvature_min = float(curvature) if curvature is not None else None
            gap_range = class_payload.get("gap_ratio_range")
            gap_tuple: Optional[Tuple[float, float]] = None
            if isinstance(gap_range, (list, tuple)) and len(gap_range) >= 2:
                gap_tuple = (float(gap_range[0]), float(gap_range[1]))

            missing_allowance = int(class_payload.get("missing_distal_allowance", 0) or 0)
            missing_strict_curvature = class_payload.get("missing_distal_strict_curvature")
            missing_strict_value = (
                float(missing_strict_curvature)
                if missing_strict_curvature is not None
                else None
            )
            consistency_boost = float(class_payload.get("curvature_consistency_boost", 0.0) or 0.0)
            consistency_window = int(class_payload.get("curvature_consistency_window", 0) or 0)
            consistency_min_frames = int(class_payload.get("curvature_consistency_min_frames", 0) or 0)
            consistency_tolerance = float(class_payload.get("curvature_consistency_tolerance", 0.0) or 0.0)
            consistency_min_curvature_raw = class_payload.get("curvature_consistency_min_curvature")
            consistency_min_curvature = (
                float(consistency_min_curvature_raw)
                if consistency_min_curvature_raw is not None
                else None
            )

            class_profiles[gesture] = ClassProfile(
                score_min=score_min,
                angle_tol_deg=angle_tol,
                norm_dev_max=norm_dev,
                curvature_min=curvature_min,
                gap_ratio_range=gap_tuple,
                missing_distal_allowance=missing_allowance,
                missing_distal_strict_curvature=missing_strict_value,
                curvature_consistency_boost=consistency_boost,
                curvature_consistency_window=consistency_window,
                curvature_consistency_min_frames=consistency_min_frames,
                curvature_consistency_tolerance=consistency_tolerance,
                curvature_consistency_min_curvature=consistency_min_curvature,
            )

        if not class_profiles:
            continue

        profiles[upper_mode] = SensitivityProfile(
            mode=upper_mode,
            quality=quality,
            temporal=temporal,
            classes=class_profiles,
        )

    if not profiles:
        return _default_sensitivity()

    return profiles, hysteresis, rate_limit, version


SENSITIVITY_PROFILES, HYSTERESIS_SETTINGS, RATE_LIMIT_SETTINGS, SENSITIVITY_PROFILE_VERSION = _load_sensitivity_profiles()

DEFAULT_SENS_MODE = os.getenv("HELEN_SENS_MODE", "BALANCED").strip().upper() or "BALANCED"
if DEFAULT_SENS_MODE not in SENSITIVITY_PROFILES:
    DEFAULT_SENS_MODE = "BALANCED" if "BALANCED" in SENSITIVITY_PROFILES else next(iter(SENSITIVITY_PROFILES))


def _class_thresholds_from_profile(profile: SensitivityProfile, hysteresis: HysteresisProfile) -> Dict[str, ClassThreshold]:
    thresholds: Dict[str, ClassThreshold] = {}
    for canonical, class_profile in profile.classes.items():
        enter = float(class_profile.score_min + hysteresis.on_offset)
        release = max(0.0, enter - hysteresis.off_delta)
        thresholds[canonical] = ClassThreshold(enter=enter, release=release)
    return thresholds

GLOBAL_MIN_SCORE = 0.46
DEFAULT_POLL_INTERVAL_S = 0.12

AUTO_THRESHOLD_WINDOW_S = 60.0
AUTO_THRESHOLD_STEP = 0.02
AUTO_THRESHOLD_MAX_DELTA = 0.08
AUTO_THRESHOLD_NEAR_MARGIN = 0.06
AUTO_THRESHOLD_MIN_RATIO = 0.7
REARM_START_MIN_DELAY = 1.2


@dataclass(frozen=True)
class PiCameraProfile:
    model: str
    width: int
    height: int
    fps: int
    poll_interval: float
    process_every_n: int


QUALITY_MIN_LANDMARKS = 21
QUALITY_MIN_HAND_SCORE = 0.48
QUALITY_MIN_BBOX_AREA = 0.008
QUALITY_MIN_BBOX_SIDE = 0.075
QUALITY_EDGE_MARGIN = 0.01


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

    def __init__(self, dataset_path: Path) -> None:
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
        if not path.exists():
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
                "dataset": str(self._dataset_path),
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
    def evaluate(self, label: str, threshold: float) -> ConsensusResult:
        with self._lock:
            votes = list(self._votes)

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
        self._context_mode: str = DEFAULT_SENS_MODE
        self._context_version: int = SENSITIVITY_PROFILE_VERSION

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
    def configure_context(self, mode: str, version: int) -> None:
        with self._lock:
            self._context_mode = str(mode)
            self._context_version = int(version)

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
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            total_samples = len(self._samples)
            accepted = sum(1 for record in self._samples if record.accepted)
            quality_checks = self._quality_checks
            quality_rejections = dict(self._quality_rejections)
            reason_counts = dict(self._reason_counts)

        rejected = total_samples - accepted
        rejection_rate = (
            sum(quality_rejections.values()) / quality_checks if quality_checks else 0.0
        )

        return {
            "samples": total_samples,
            "accepted": accepted,
            "rejected": rejected,
            "quality_checks": quality_checks,
            "quality_rejections": quality_rejections,
            "quality_rejection_rate": rejection_rate,
            "reason_counts": reason_counts,
        }

    # ------------------------------------------------------------------
    def generate_report(
        self,
        *,
        thresholds: Dict[str, ClassThreshold],
        consensus: ConsensusConfig,
        dataset_info: Dict[str, Any],
        latency_stats: Dict[str, float],
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

        precision_by_class = {label: per_label[label]["precision"] for label in TRACKED_GESTURES}
        recall_by_class = {label: per_label[label]["recall"] for label in TRACKED_GESTURES}

        latency_windows = [record.window_ms for record in samples if record.window_ms > 0.0]
        if latency_windows:
            sorted_latencies = sorted(latency_windows)
            mid_index = len(sorted_latencies) // 2
            p50_latency = sorted_latencies[mid_index]
            p95_latency = sorted_latencies[min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95))]
        else:
            p50_latency = 0.0
            p95_latency = 0.0

        none_events = [record for record in samples if not self._canonical(record.hint_label)]
        none_false_positives = sum(1 for record in none_events if record.accepted)
        fp_rate_none = (none_false_positives / len(none_events)) if none_events else 0.0

        confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for record in samples:
            actual = self._canonical(record.hint_label)
            if not actual:
                continue
            predicted = self._canonical(record.label) if record.accepted else "None"
            confusion[actual][predicted] += 1

        suggestions = [suggestion.__dict__ for suggestion in self.threshold_suggestions(thresholds)]

        temporal_info = dataset_info.get("temporal", {}) if isinstance(dataset_info, dict) else {}
        durations = {
            "cooldown": float(temporal_info.get("cooldown_s", COOLDOWN_SECONDS)),
            "listening_window": float(temporal_info.get("listen_window_s", LISTENING_WINDOW_SECONDS)),
            "command_debounce": COMMAND_DEBOUNCE_SECONDS,
        }

        return {
            "thresholds": {label: {"enter": th.enter, "release": th.release} for label, th in thresholds.items()},
            "global_min_score": min((th.enter for th in thresholds.values()), default=GLOBAL_MIN_SCORE),
            "consensus": {
                "window_size": consensus.window_size,
                "required_votes": consensus.required_votes,
            },
            "durations_s": durations,
            "dataset": dataset_info,
            "latency": latency_stats,
            "samples": len(samples),
            "quality_checks": quality_checks,
            "quality_rejections": quality_rejections,
            "quality_rejection_rate": quality_ratio,
            "threshold_rejections": reason_counts,
            "classes": per_label,
            "precision_by_class": precision_by_class,
            "recall_by_class": recall_by_class,
            "fp_rate_none": fp_rate_none,
            "consensus_latency_ms": {"p50": p50_latency, "p95": p95_latency},
            "mode": self._context_mode,
            "profile_version": self._context_version,
            "frameskip_used": dataset_info.get("frameskip_used"),
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
        lines.append(
            "- Cooldown tras 'Start': {0:.0f} ms".format(float(report["durations_s"]["cooldown"]) * 1000)
        )
        lines.append(
            "- Ventana de escucha C/R/I: {0:.1f} s".format(float(report["durations_s"]["listening_window"]))
        )
        lines.append("- Debounce de comandos: {0:.0f} ms".format(COMMAND_DEBOUNCE_SECONDS * 1000))
        lines.append("- Umbral global mínimo: {0:.2f}".format(report["global_min_score"]))
        lines.append("- Modo: {0} (perfil v{1})".format(report.get("mode", "UNKNOWN"), report.get("profile_version", "?")))
        consensus_latency = report.get("consensus_latency_ms", {})
        lines.append(
            "- Latencia consenso p50/p95: {0:.1f} / {1:.1f} ms".format(
                float(consensus_latency.get("p50", 0.0)), float(consensus_latency.get("p95", 0.0))
            )
        )
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
    ) -> None:
        report = self.generate_report(
            thresholds=thresholds, consensus=consensus, dataset_info=dataset_info, latency_stats=latency_stats
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
        self._class_profiles: Dict[str, ClassProfile] = {}
        self._last_metrics: Dict[str, Dict[str, Any]] = {}
        self._curvature_history: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=6))

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

    def _average_curvature(self, landmarks: Sequence[LandmarkPoint]) -> float:
        curvatures: List[float] = []
        for name in ("index", "middle", "ring", "pinky"):
            indices = self._FINGER_LANDMARKS[name]
            if any(self._is_missing(landmarks[idx]) for idx in indices):
                continue
            curl = self._finger_curl(landmarks, indices)
            curvature = max(0.0, 180.0 - min(180.0, curl)) / 180.0
            curvatures.append(curvature)
        if not curvatures:
            return 0.0
        return statistics.fmean(curvatures) if hasattr(statistics, "fmean") else sum(curvatures) / len(curvatures)

    def _gap_ratio(self, landmarks: Sequence[LandmarkPoint]) -> float:
        if any(self._is_missing(landmarks[idx]) for idx in (8, 20, 5, 17)):
            return 0.0
        arc = self._distance(landmarks[8], landmarks[20])
        palm = self._distance(landmarks[5], landmarks[17]) or 1e-3
        return arc / palm

    @staticmethod
    def _is_missing(point: LandmarkPoint) -> bool:
        return not all(math.isfinite(coord) for coord in point) or (abs(point[0]) < 1e-4 and abs(point[1]) < 1e-4)

    def configure(self, profile: SensitivityProfile) -> None:
        self._class_profiles = dict(profile.classes)
        self._last_metrics.clear()
        self._curvature_history.clear()

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

    def metrics_for(self, label: str) -> Optional[Dict[str, Any]]:
        canonical = GestureMetrics._canonical(label)
        metrics = self._last_metrics.get(canonical)
        if not metrics:
            return None
        data = dict(metrics)
        history = self._curvature_history.get(canonical)
        if history:
            data["history"] = [dict(entry) for entry in history]
        else:
            data["history"] = []
        return data

    def class_profile(self, label: str) -> Optional[ClassProfile]:
        canonical = GestureMetrics._canonical(label)
        return self._class_profiles.get(canonical)

    def _record_clima_metrics(
        self,
        *,
        avg_curvature: float,
        curvature_min: Optional[float],
        strict_curvature: Optional[float],
        missing_distal: int,
        allowance: int,
        accepted: bool,
        reason: Optional[str],
    ) -> None:
        loss_rate = missing_distal / 4.0
        metrics = {
            "avg_curvature": avg_curvature,
            "curvature_min": curvature_min,
            "curvature_delta": avg_curvature - (curvature_min or 0.0),
            "strict_curvature": strict_curvature,
            "missing_distal": missing_distal,
            "missing_allowance": allowance,
            "loss_rate": loss_rate,
            "accepted": accepted,
            "reason": reason,
        }
        self._last_metrics["Clima"] = metrics
        history_entry = {
            "curvature": avg_curvature,
            "missing_distal": missing_distal,
            "accepted": accepted,
            "timestamp": time.time(),
        }
        self._curvature_history["Clima"].append(history_entry)
        if accepted:
            LOGGER.debug(
                "geometry_clima accepted curvature=%.3f delta=%.3f missing=%d loss_rate=%.2f",
                avg_curvature,
                metrics["curvature_delta"],
                missing_distal,
                loss_rate,
            )

    def verify(self, label: str, landmarks: Optional[Sequence[LandmarkPoint]]) -> Tuple[bool, Optional[str]]:
        if not label or not landmarks:
            return True, None

        canonical = GestureMetrics._canonical(label)
        if canonical not in TRACKED_GESTURES:
            return True, None

        points = list(landmarks)
        if len(points) < 21:
            self._log_once("landmarks_insuficientes")
            return False, "geometry_incomplete"

        class_profile = self._class_profiles.get(canonical)
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
            if is_extended("index") and is_extended("middle") and is_folded("ring") and is_folded("pinky"):
                if index_middle_distance >= 0.027:
                    return True, None
                return False, "geometry_start_spacing"
            return False, "geometry_start_pattern"

        if canonical == "Clima":
            if class_profile:
                avg_curvature = self._average_curvature(points)
                gap_ratio = self._gap_ratio(points)
                missing_distal = sum(1 for idx in (8, 12, 16, 20) if self._is_missing(points[idx]))
                allowance = max(0, int(getattr(class_profile, "missing_distal_allowance", 0)))
                effective_allowance = max(allowance, 2)
                strict_curvature = getattr(class_profile, "missing_distal_strict_curvature", None)

                def reject(reason: str) -> Tuple[bool, str]:
                    self._record_clima_metrics(
                        avg_curvature=avg_curvature,
                        curvature_min=class_profile.curvature_min,
                        strict_curvature=strict_curvature,
                        missing_distal=missing_distal,
                        allowance=effective_allowance,
                        accepted=False,
                        reason=reason,
                    )
                    return False, reason

                curvature_floor = None
                if class_profile.curvature_min is not None:
                    curvature_floor = class_profile.curvature_min * 0.82
                if curvature_floor is not None and avg_curvature < curvature_floor:
                    return reject("geometry_clima_curvature")

                if class_profile.gap_ratio_range:
                    low, high = class_profile.gap_ratio_range
                    low *= 0.85
                    high *= 1.15
                    if not (low <= gap_ratio <= high):
                        return reject("geometry_clima_gap")

                angles: List[float] = []
                lengths: List[float] = []
                for base, tip in ((5, 8), (9, 12), (13, 16), (17, 20)):
                    if self._is_missing(points[tip]) or self._is_missing(points[base]):
                        continue
                    vector_angle = math.degrees(
                        math.atan2(points[tip][1] - points[base][1], points[tip][0] - points[base][0])
                    )
                    angles.append(vector_angle)
                    lengths.append(self._distance(wrist, points[tip]))

                if angles and class_profile.angle_tol_deg:
                    avg_angle = sum(angles) / len(angles)
                    max_delta = max(abs(angle - avg_angle) for angle in angles)
                    allowed_delta = class_profile.angle_tol_deg * 1.35
                    if max_delta > allowed_delta:
                        return reject("geometry_clima_angle")

                if lengths and class_profile.norm_dev_max:
                    max_length = max(lengths) or 1.0
                    normalised = [length / max_length for length in lengths]
                    deviation = statistics.pstdev(normalised) if len(normalised) > 1 else 0.0
                    allowed_deviation = class_profile.norm_dev_max * 1.28
                    if deviation > allowed_deviation:
                        return reject("geometry_clima_norm")

                if missing_distal > 0:
                    if missing_distal > effective_allowance:
                        return reject("geometry_clima_missing_distal")
                    if strict_curvature is not None:
                        strict_threshold = strict_curvature * 0.85
                        if avg_curvature < strict_threshold:
                            return reject("geometry_clima_missing_curvature")

                self._record_clima_metrics(
                    avg_curvature=avg_curvature,
                    curvature_min=class_profile.curvature_min,
                    strict_curvature=strict_curvature,
                    missing_distal=missing_distal,
                    allowance=effective_allowance,
                    accepted=True,
                    reason=None,
                )
                return True, None

            extended_count = sum(1 for finger in ("index", "middle", "ring", "pinky") if is_extended(finger, 0.9))
            if extended_count >= 3 and thumb_index_distance >= 0.04 and palm_spread >= 0.14:
                return True, None
            return False, "geometry_clima_pattern"

        if canonical == "Reloj":
            if is_extended("index") and is_extended("middle") and is_folded("ring") and is_folded("pinky"):
                vertical_alignment = abs(index_tip[1] - middle_tip[1]) <= 0.075
                if index_middle_distance <= 0.085 and vertical_alignment:
                    return True, None
                return False, "geometry_reloj_spacing"
            return False, "geometry_reloj_pattern"

        if canonical == "Inicio":
            if is_extended("pinky") and is_folded("index") and is_folded("middle") and is_folded("ring"):
                if thumb_index_distance <= 0.105:
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
        temporal_profile: Optional[TemporalProfile] = None,
        sensitivity_mode: str = DEFAULT_SENS_MODE,
        profile_version: int = SENSITIVITY_PROFILE_VERSION,
        session_started_at: Optional[float] = None,
    ) -> None:
        self._metrics = metrics
        base_thresholds = dict(DEFAULT_CLASS_THRESHOLDS)
        if thresholds:
            base_thresholds.update(thresholds)
        self._thresholds = base_thresholds
        self._baseline_thresholds = {label: ClassThreshold(value.enter, value.release) for label, value in base_thresholds.items()}
        self._consensus_config = consensus
        self._consensus = ConsensusTracker(consensus)
        self._global_min_score = float(global_min_score)
        self._geometry_verifier = geometry_verifier
        self._temporal_profile = temporal_profile
        self._mode = sensitivity_mode
        self._profile_version = profile_version
        self._session_started_at = float(session_started_at or time.time())
        self._activation_delay = (
            float(temporal_profile.activation_delay_s)
            if temporal_profile and getattr(temporal_profile, "activation_delay_s", None) is not None
            else ACTIVATION_DELAY
        )
        self._threshold_adjustments: Dict[str, float] = defaultdict(float)
        self._recent_rejections: Dict[str, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=12))

        self._state = "idle"
        self._cooldown_until = 0.0
        self._listen_until = 0.0
        self._command_debounce_until = 0.0
        self._listening_duration = (
            float(temporal_profile.listen_window_s) if temporal_profile else LISTENING_WINDOW_SECONDS
        )
        self._cooldown_duration = float(temporal_profile.cooldown_s) if temporal_profile else COOLDOWN_SECONDS
        self._min_position_variance = (
            float(temporal_profile.min_pos_stability_var) if temporal_profile else 12.0
        )
        self._dominant_label: Optional[str] = None
        self._last_state_change = time.time()
        self._last_activation_at = 0.0
        self._lock = threading.RLock()
        self._position_history: Deque[Tuple[float, float]] = deque(maxlen=max(5, consensus.window_size * 2))
        self._rearm_block_until = 0.0

    # ------------------------------------------------------------------
    def _reset_consensus(self) -> None:
        self._consensus.reset()
        self._dominant_label = None
        self._position_history.clear()

    # ------------------------------------------------------------------
    def _set_rearm_block(self, target: float) -> None:
        if target <= 0.0:
            return
        self._rearm_block_until = max(self._rearm_block_until, target)

    # ------------------------------------------------------------------
    def defer_rearm(self, *, duration: float = 0.0, until: Optional[float] = None) -> None:
        if until is None:
            if duration <= 0.0:
                return
            target = time.time() + float(duration)
        else:
            target = float(until)

        with self._lock:
            self._set_rearm_block(target)

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
    def _apply_hysteresis(self, label: str) -> Optional[str]:
        if not self._dominant_label:
            return None
        if self._dominant_label == label:
            return None

        thresholds = self._current_threshold(self._dominant_label)
        if thresholds is None:
            self._dominant_label = None
            return None

        result = self._consensus.evaluate(self._dominant_label, thresholds.release)
        if result.average < thresholds.release:
            self._dominant_label = None
            return None
        return self._dominant_label

    # ------------------------------------------------------------------
    def _record_position(self, roi: Optional[Dict[str, Any]]) -> None:
        if not roi:
            return
        try:
            cx = (float(roi.get("x1", 0.0)) + float(roi.get("x2", 0.0))) / 2.0
            cy = (float(roi.get("y1", 0.0)) + float(roi.get("y2", 0.0))) / 2.0
        except (TypeError, ValueError):
            return
        self._position_history.append((cx, cy))

    # ------------------------------------------------------------------
    def _position_variance(self) -> float:
        if len(self._position_history) < 3:
            return 0.0
        xs = [point[0] for point in self._position_history]
        ys = [point[1] for point in self._position_history]
        var_x = statistics.pvariance(xs) if len(xs) > 1 else 0.0
        var_y = statistics.pvariance(ys) if len(ys) > 1 else 0.0
        return var_x + var_y

    # ------------------------------------------------------------------
    def _apply_dynamic_boost(
        self,
        label: str,
        score: float,
        metrics: Optional[Dict[str, Any]],
    ) -> Tuple[float, float]:
        if not self._geometry_verifier:
            return score, 0.0

        profile = self._geometry_verifier.class_profile(label)
        if not profile or profile.curvature_consistency_boost <= 0.0:
            return score, 0.0

        if metrics is None:
            metrics = self._geometry_verifier.metrics_for(label)
        if not metrics:
            return score, 0.0

        history = metrics.get("history") or []
        if not history:
            return score, 0.0

        window = max(0, int(profile.curvature_consistency_window or 0))
        if window <= 0 or window > len(history):
            window = len(history)
        if window <= 0:
            return score, 0.0

        tail = history[-window:]
        min_frames = max(1, int(profile.curvature_consistency_min_frames or 0))

        required_curvature = profile.curvature_consistency_min_curvature
        if required_curvature is None:
            strict_curvature = metrics.get("strict_curvature")
            if isinstance(strict_curvature, (int, float)):
                required_curvature = float(strict_curvature)
            elif profile.curvature_min is not None:
                required_curvature = float(profile.curvature_min)
            else:
                required_curvature = 0.0

        tolerance = max(0.0, float(profile.curvature_consistency_tolerance or 0.0))

        qualifying: List[Dict[str, Any]] = [
            entry
            for entry in tail
            if entry.get("accepted")
            and float(entry.get("curvature", 0.0)) >= required_curvature
            and int(entry.get("missing_distal", 0)) <= max(0, profile.missing_distal_allowance)
        ]

        if len(qualifying) < min_frames:
            return score, 0.0

        curvatures = [float(entry.get("curvature", 0.0)) for entry in qualifying[-min_frames:]]
        if not curvatures:
            return score, 0.0

        if max(curvatures) - min(curvatures) > tolerance:
            return score, 0.0

        boost = float(profile.curvature_consistency_boost or 0.0)
        if boost <= 0.0:
            return score, 0.0

        boosted = min(1.0, score + boost)
        return boosted, boosted - score

    # ------------------------------------------------------------------
    def _consider_auto_adjust(
        self,
        label: str,
        score: float,
        threshold: ClassThreshold,
        timestamp: float,
    ) -> ClassThreshold:
        if (timestamp - self._session_started_at) > AUTO_THRESHOLD_WINDOW_S:
            return threshold

        history = self._recent_rejections[label]
        history.append((timestamp, score))
        near_threshold = [value for _, value in history if value >= threshold.enter - AUTO_THRESHOLD_NEAR_MARGIN]

        if len(history) < max(4, self._consensus_config.window_size):
            return threshold
        if not near_threshold or (len(near_threshold) / len(history)) < AUTO_THRESHOLD_MIN_RATIO:
            return threshold

        applied = self._threshold_adjustments[label]
        if applied >= AUTO_THRESHOLD_MAX_DELTA:
            return threshold

        new_enter = max(0.4, threshold.enter - AUTO_THRESHOLD_STEP)
        if new_enter >= threshold.enter:
            return threshold

        release_gap = max(0.01, threshold.enter - threshold.release)
        new_release = max(0.0, new_enter - release_gap)
        self._thresholds[label] = ClassThreshold(enter=new_enter, release=new_release)
        self._threshold_adjustments[label] = applied + (threshold.enter - new_enter)
        history.clear()

        LOGGER.info(
            "auto_threshold_adjust label=%s enter=%.3f->%.3f release=%.3f->%.3f delta=%.3f",
            label,
            threshold.enter,
            new_enter,
            threshold.release,
            new_release,
            threshold.enter - new_enter,
        )
        return self._thresholds[label]

    # ------------------------------------------------------------------
    def baseline_thresholds(self) -> Dict[str, ClassThreshold]:
        return dict(self._baseline_thresholds)

    # ------------------------------------------------------------------
    def threshold_adjustments(self) -> Dict[str, float]:
        adjustments: Dict[str, float] = {}
        for label, base in self._baseline_thresholds.items():
            current = self._thresholds.get(label, base)
            delta = current.enter - base.enter
            if not math.isclose(delta, 0.0, abs_tol=1e-6):
                adjustments[label] = delta
        return adjustments

    # ------------------------------------------------------------------
    def process(
        self,
        prediction: Prediction,
        *,
        timestamp: float,
        hint_label: Optional[str] = None,
        latency_ms: float = 0.0,
        landmarks: Optional[Sequence[LandmarkPoint]] = None,
        roi: Optional[Dict[str, Any]] = None,
    ) -> DecisionOutcome:
        with self._lock:
            self._update_state(timestamp)

            canonical_label = GestureMetrics._canonical(prediction.label)
            canonical_hint = GestureMetrics._canonical(hint_label)
            score = float(prediction.score)
            state = self._state

            payload: Dict[str, Any] = {
                "latency_ms": latency_ms,
                "state": state,
            }
            geometry_checked = False
            curvature_metrics: Optional[Dict[str, Any]] = None

            self._record_position(roi)
            payload["mode"] = self._mode
            payload["profile_version"] = self._profile_version

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
                    return DecisionOutcome(
                        False,
                        canonical_label,
                        score,
                        payload,
                        reason,
                        state,
                        canonical_hint,
                        0,
                        0.0,
                    )
                curvature_metrics = self._geometry_verifier.metrics_for(canonical_label)
            elif self._geometry_verifier is not None:
                curvature_metrics = self._geometry_verifier.metrics_for(canonical_label)
            if self._geometry_verifier is not None:
                payload["geometry_checked"] = geometry_checked
            if curvature_metrics:
                payload["geometry_curvature"] = curvature_metrics.get("avg_curvature")
                payload["geometry_curvature_delta"] = curvature_metrics.get("curvature_delta")
                payload["geometry_loss_rate"] = curvature_metrics.get("loss_rate")
                payload["geometry_missing_distal"] = curvature_metrics.get("missing_distal")

            if self._geometry_verifier is not None:
                boosted_score, boost_delta = self._apply_dynamic_boost(
                    canonical_label,
                    score,
                    curvature_metrics,
                )
                if boost_delta > 0.0:
                    payload["score_boost"] = boost_delta
                    score = boosted_score

            thresholds = self._current_threshold(canonical_label)
            self._consensus.add(canonical_label, score, timestamp)
            result = self._consensus.evaluate(
                canonical_label,
                thresholds.enter if thresholds else self._global_min_score,
            )

            support = result.votes
            window_ms = result.span_ms

            variance = self._position_variance()
            payload["position_variance"] = variance

            if thresholds and result.average >= thresholds.enter:
                self._dominant_label = canonical_label

            locked_label = self._apply_hysteresis(canonical_label)
            state = self._state

            payload.update(
                {
                    "consensus_support": support,
                    "consensus_total": result.total,
                    "consensus_span_ms": window_ms,
                }
            )

            if (
                self._temporal_profile
                and len(self._position_history) >= 3
                and variance > self._min_position_variance
            ):
                reason = "position_unstable"
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
                return DecisionOutcome(
                    False,
                    canonical_label,
                    score,
                    payload,
                    reason,
                    state,
                    canonical_hint,
                    support,
                    window_ms,
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
                return DecisionOutcome(False, canonical_label, score, payload, reason, state, canonical_hint, support, window_ms)

            if (
                state == "listening"
                and canonical_label in {"Clima", "Reloj", "Inicio"}
                and self._last_activation_at > 0.0
                and (timestamp - self._last_activation_at) < self._activation_delay
            ):
                reason = "activation_delay_active"
                remaining = max(0.0, self._activation_delay - (timestamp - self._last_activation_at))
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
                payload["activation_delay_remaining_ms"] = round(remaining * 1000.0, 3)
                return DecisionOutcome(
                    False,
                    canonical_label,
                    score,
                    payload,
                    reason,
                    state,
                    canonical_hint,
                    support,
                    window_ms,
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
                return DecisionOutcome(False, canonical_label, score, payload, reason, state, canonical_hint, support, window_ms)

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
                return DecisionOutcome(False, canonical_label, score, payload, reason, state, canonical_hint, support, window_ms)

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
                return DecisionOutcome(False, canonical_label, score, payload, reason, state, canonical_hint, support, window_ms)

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
                return DecisionOutcome(False, canonical_label, score, payload, reason, state, canonical_hint, support, window_ms)

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
                return DecisionOutcome(False, canonical_label, score, payload, reason, state, canonical_hint, support, window_ms)

            if state == "listening" and canonical_label == "Start":
                rearm_threshold = max(REARM_START_MIN_DELAY, self._activation_delay + (self._cooldown_duration * 0.5))
                min_ready_at = 0.0
                if self._last_activation_at > 0.0:
                    min_ready_at = self._last_activation_at + rearm_threshold

                block_until = max(min_ready_at, self._rearm_block_until)
                if block_until > 0.0 and timestamp < block_until:
                    if timestamp < self._rearm_block_until:
                        reason = "start_rearm_deferred"
                        remaining = max(0.0, self._rearm_block_until - timestamp)
                        payload["rearm_block_remaining_ms"] = round(remaining * 1000.0, 3)
                    else:
                        reason = "awaiting_command"
                        remaining = max(0.0, min_ready_at - timestamp)
                        payload["rearm_min_remaining_ms"] = round(remaining * 1000.0, 3)

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
                    return DecisionOutcome(
                        False,
                        canonical_label,
                        score,
                        payload,
                        reason,
                        state,
                        canonical_hint,
                        support,
                        window_ms,
                    )

                self._state = "idle"
                state = self._state
                payload["state"] = state
                self._reset_consensus()
                self._last_activation_at = 0.0
                self._rearm_block_until = 0.0

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
                return DecisionOutcome(False, canonical_label, score, payload, reason, state, canonical_hint, support, window_ms)

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
                updated_threshold = self._consider_auto_adjust(canonical_label, score, thresholds, timestamp)
                if updated_threshold.enter != thresholds.enter:
                    payload["threshold_enter_new"] = updated_threshold.enter
                    payload["threshold_release_new"] = updated_threshold.release
                    payload.setdefault("auto_threshold_adjusted", True)
                payload["decision_reason"] = reason
                return DecisionOutcome(False, canonical_label, score, payload, reason, state, canonical_hint, support, window_ms)

            passes_votes = support >= self._consensus_config.required_votes
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
                return DecisionOutcome(False, canonical_label, score, payload, reason, state, canonical_hint, support, window_ms)

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
                    "votes_required": self._consensus_config.required_votes,
                }
            )
            payload["decision_reason"] = reason

            if canonical_label == "Start":
                self._state = "cooldown"
                self._cooldown_until = timestamp + self._cooldown_duration
                self._last_activation_at = timestamp
                rearm_ready = max(
                    REARM_START_MIN_DELAY,
                    self._activation_delay + (self._cooldown_duration * 0.5),
                )
                self._set_rearm_block(timestamp + rearm_ready)
                payload["next_state"] = "cooldown"
                self._reset_consensus()
            else:
                self._state = "command_debounce"
                self._command_debounce_until = timestamp + COMMAND_DEBOUNCE_SECONDS
                self._listen_until = timestamp
                payload["next_state"] = "command_debounce"
                self._reset_consensus()

            self._last_state_change = timestamp

            return DecisionOutcome(True, canonical_label, score, payload, reason, self._state, canonical_hint, support, window_ms)

    # ------------------------------------------------------------------
    def thresholds(self) -> Dict[str, ClassThreshold]:
        return dict(self._thresholds)

    # ------------------------------------------------------------------
    @property
    def consensus_config(self) -> ConsensusConfig:
        return self._consensus_config

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

HEALTH_ENDPOINTS = {"/health", "/healthz", "/engine/status"}


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
    camera_index: int = 0
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.6
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
    enable_camera: bool = True
    fallback_to_synthetic: bool = True
    model_path: Path = MODEL_PATH
    dataset_path: Path = field(default_factory=_default_dataset_path)
    process_every_n: int = 3
    sensitivity_mode: Optional[str] = None


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
    camera_index: Optional[int]
    camera_backend: Optional[str]
    camera_last_capture: Optional[str]
    camera_last_error: Optional[str]
    latency_p95_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_count: int = 0
    frames_total: int = 0
    frames_with_hand: int = 0
    frames_valid: int = 0
    frames_returned: int = 0
    fps: float = 0.0
    quality_rejections: Dict[str, int] = field(default_factory=dict)
    reason_counts: Dict[str, int] = field(default_factory=dict)
    thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    threshold_adjustments: Dict[str, float] = field(default_factory=dict)
    model_path: str = ""
    engine_mode: str = ""
    sensitivity_version: int = 0
    last_quality_reason: Optional[str] = None
    dataset_info: Dict[str, Any] = field(default_factory=dict)
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
        camera_index: int = 0,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
        metrics: Optional[GestureMetrics] = None,
        quality_profile: Optional[QualityProfile] = None,
        sensitivity_mode: str = DEFAULT_SENS_MODE,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV no está instalado. Ejecuta `pip install opencv-python`.")
        if mp is None:
            raise RuntimeError("MediaPipe no está instalado. Ejecuta `pip install mediapipe`.")

        self._camera_index = camera_index
        self._detection_confidence = detection_confidence
        self._tracking_confidence = tracking_confidence
        self._metrics = metrics
        quality = quality_profile or QualityProfile(
            blur_laplacian_min=QUALITY_BLUR_THRESHOLD,
            roi_min_coverage=0.62,
            hand_range_px=(70.0, 580.0),
        )
        hand_range = tuple(sorted((float(quality.hand_range_px[0]), float(quality.hand_range_px[1]))))
        self._quality_profile = QualityProfile(
            blur_laplacian_min=float(quality.blur_laplacian_min),
            roi_min_coverage=float(quality.roi_min_coverage),
            hand_range_px=hand_range,
        )
        self._sensitivity_mode = sensitivity_mode

        self._cap: Optional[Any] = None
        self._hands: Optional[Any] = None
        self._opened = False
        self._profile: Optional[PiCameraProfile] = PI_CAMERA_PROFILE
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
        self._last_quality_reason: Optional[str] = None
        self._last_frame_ts: Optional[float] = None
        self._fps_history: Deque[float] = deque(maxlen=90)
        self._frames_total = 0
        self._frames_with_hand = 0
        self._frames_valid = 0
        self._frames_returned = 0
        self._weak_frame_budget = 2
        self._weak_allow_reasons = {
            "low_confidence",
            "hand_near_edge",
            "small_bbox",
            "roi_too_small",
            "roi_coverage",
            "hand_range",
            "blur",
        }

    # ------------------------------------------------------------------
    def _configure_capture(self, cap: Any) -> None:
        profile = self._profile
        if not profile:
            return

        with contextlib.suppress(Exception):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, profile.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, profile.height)
            cap.set(cv2.CAP_PROP_FPS, profile.fps)

    # ------------------------------------------------------------------
    def _attempt_v4l2(self) -> Tuple[Optional[Any], Optional[str]]:
        args: Tuple[Any, ...]
        if IS_LINUX and hasattr(cv2, "CAP_V4L2"):
            args = (self._camera_index, cv2.CAP_V4L2)
        else:
            args = (self._camera_index,)

        cap = cv2.VideoCapture(*args)
        if not cap or not cap.isOpened():
            if cap:
                with contextlib.suppress(Exception):
                    cap.release()
            return None, f"V4L2 no se pudo abrir en el índice {self._camera_index}"

        self._configure_capture(cap)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        LOGGER.info(
            "Ruta de cámara inicializada: v4l2 (index=%s, %sx%s @ %.2f fps)",
            self._camera_index,
            actual_width,
            actual_height,
            actual_fps,
        )
        self._capture_backend = "v4l2"
        self._gstreamer_pipeline = None
        return cap, None

    # ------------------------------------------------------------------
    def _build_gstreamer_pipeline(self) -> str:
        profile = self._profile
        width = profile.width if profile else 1280
        height = profile.height if profile else 720
        fps = profile.fps if profile else 30
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
        return cap, None

    # ------------------------------------------------------------------
    def _initialise_capture(self) -> Any:
        errors: List[str] = []

        cap, error = self._attempt_v4l2()
        if cap is None:
            if error:
                errors.append(error)
            cap, error = self._attempt_gstreamer()

        if cap is None:
            if error:
                errors.append(error)
            message = "; ".join(errors) if errors else f"No se pudo abrir la cámara {self._camera_index}"
            self._last_error = message
            raise RuntimeError(message)

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

            self._frames_total += 1

            now_ts = time.time()
            if self._last_frame_ts:
                delta = now_ts - self._last_frame_ts
                if delta > 0:
                    fps = min(120.0, 1.0 / delta)
                    self._fps_history.append(fps)
            self._last_frame_ts = now_ts

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
            self._frames_with_hand += 1
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
            self._last_quality_reason = None
            self._last_capture = time.time()
            self._last_error = None
            self._healthy = True
            self._frames_valid += 1
            self._frames_returned += 1
            return features, None

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "healthy": self._healthy,
            "camera_index": self._camera_index,
            "last_capture": self._last_capture,
            "last_error": self._last_error,
            "frames_without_hand": self._frames_without_hand,
            "frames_total": self._frames_total,
            "frames_with_hand": self._frames_with_hand,
            "frames_valid": self._frames_valid,
            "frames_returned": self._frames_returned,
            "quality_rejections": dict(self._quality_rejections),
            "frame_shape": self._last_frame_shape,
            "roi_snapshot": self._last_roi,
            "capture_backend": self._capture_backend,
            "gstreamer_pipeline": self._gstreamer_pipeline,
            "measured_fps": round(self.measured_fps(), 2),
            "last_quality_reason": self._last_quality_reason,
            "weak_frame_budget": self._weak_frame_budget,
        }

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
    def _register_quality_check(self, valid: bool, reason: Optional[str]) -> None:
        if self._metrics:
            self._metrics.register_quality_check(valid, reason)
        if valid:
            self._weak_frame_budget = 2
        elif reason:
            self._quality_rejections[reason] += 1

    # ------------------------------------------------------------------
    def _log_quality_rejection(self, reason: str) -> None:
        if reason != self._last_quality_reason:
            LOGGER.debug("Descartado por calidad (%s) en modo %s", reason, self._sensitivity_mode)
            self._last_quality_reason = reason

    # ------------------------------------------------------------------
    def _allow_weak_frame(self, reason: str) -> bool:
        if reason not in self._weak_allow_reasons:
            return False
        if self._weak_frame_budget <= 0:
            return False
        self._weak_frame_budget -= 1
        LOGGER.debug(
            "Frame degradado permitido (%s); presupuesto restante=%d",
            reason,
            self._weak_frame_budget,
        )
        return True

    # ------------------------------------------------------------------
    def _fail_or_allow(self, reason: str, *, allow_weak: bool = False) -> bool:
        self._log_quality_rejection(reason)
        self._register_quality_check(False, reason)
        if allow_weak and self._allow_weak_frame(reason):
            return False
        return True

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
            if self._fail_or_allow("low_confidence", allow_weak=True):
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

        if (
            min_x < -0.08
            or min_y < -0.08
            or max_x > 1.08
            or max_y > 1.08
        ):
            self._register_quality_check(False, "roi_out_of_bounds")
            return False

        if (
            min_x <= QUALITY_EDGE_MARGIN
            or min_y <= QUALITY_EDGE_MARGIN
            or max_x >= (1.0 - QUALITY_EDGE_MARGIN)
            or max_y >= (1.0 - QUALITY_EDGE_MARGIN)
        ):
            if self._fail_or_allow("hand_near_edge", allow_weak=True):
                return False

        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        if width < QUALITY_MIN_BBOX_SIDE or height < QUALITY_MIN_BBOX_SIDE or area < QUALITY_MIN_BBOX_AREA:
            if self._fail_or_allow("small_bbox", allow_weak=True):
                return False

        pixel_width = self._normalised_to_pixel(max_x, image_width) - self._normalised_to_pixel(min_x, image_width)
        pixel_height = self._normalised_to_pixel(max_y, image_height) - self._normalised_to_pixel(min_y, image_height)
        if pixel_width <= 4 or pixel_height <= 4:
            if self._fail_or_allow("roi_too_small", allow_weak=True):
                return False

        quality = self._quality_profile
        pixel_coverage = float((pixel_width / image_width) * (pixel_height / image_height))
        if pixel_coverage < quality.roi_min_coverage:
            if self._fail_or_allow("roi_coverage", allow_weak=True):
                return False

        hand_size = math.sqrt(float(pixel_width) ** 2 + float(pixel_height) ** 2)
        min_range, max_range = quality.hand_range_px
        if hand_size < min_range or hand_size > max_range:
            if self._fail_or_allow("hand_range", allow_weak=True):
                return False

        blur_threshold = float(quality.blur_laplacian_min or 0.0)
        if (
            blur_threshold > 0.0
            and self._sensitivity_mode == "BALANCED"
            and pixel_coverage >= 0.68
        ):
            blur_threshold *= 0.85
        if blur_threshold > 0.0:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if variance < blur_threshold:
                    if self._fail_or_allow("blur", allow_weak=True):
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

    # ------------------------------------------------------------------
    def last_roi(self) -> Optional[Dict[str, Any]]:
        if not self._last_roi:
            return None
        return dict(self._last_roi)

    # ------------------------------------------------------------------
    def measured_fps(self) -> float:
        samples = list(self._fps_history)
        if not samples:
            return 0.0
        return statistics.fmean(samples) if hasattr(statistics, "fmean") else sum(samples) / len(samples)


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
        self._stride_lock = threading.Lock()

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
    def set_frame_stride(self, stride: int) -> None:
        stride = max(1, int(stride))
        with self._stride_lock:
            if stride == self._frame_stride:
                return
            self._frame_stride = stride
            self._stride_cursor = 0

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

            measured_fps = 0.0
            fps_getter = getattr(self._runtime.stream, "measured_fps", None)
            if callable(fps_getter):
                with contextlib.suppress(Exception):
                    measured_fps = float(fps_getter())
            self._runtime.update_frameskip(measured_fps)

            with self._stride_lock:
                frame_stride = self._frame_stride
                if frame_stride > 1:
                    self._stride_cursor = (self._stride_cursor + 1) % frame_stride
                    skip_frame = self._stride_cursor != 1
                else:
                    skip_frame = False
            if skip_frame:
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

            roi_snapshot: Optional[Dict[str, Any]] = None
            last_roi_getter = getattr(self._runtime.stream, "last_roi", None)
            if callable(last_roi_getter):
                with contextlib.suppress(Exception):
                    roi_candidate = last_roi_getter()
                    if roi_candidate:
                        roi_snapshot = dict(roi_candidate)

            fallback_used = False
            fallback_score = 0.0
            fallback_label: Optional[str] = None
            thresholds = self._runtime.class_thresholds
            fallback_classifier = getattr(self._runtime, "fallback_classifier", None)
            canonical_label = GestureMetrics._canonical(prediction.label)
            threshold = thresholds.get(canonical_label)
            if (
                fallback_classifier
                and threshold
                and threshold.enter - 0.04 <= prediction.score < threshold.enter
            ):
                with contextlib.suppress(Exception):
                    fallback_prediction = fallback_classifier.predict(transformed)
                    if (
                        GestureMetrics._canonical(fallback_prediction.label) == canonical_label
                        and fallback_prediction.score >= threshold.enter
                    ):
                        fallback_used = True
                        fallback_score = float(fallback_prediction.score)
                        fallback_label = fallback_prediction.label
                        prediction = Prediction(label=prediction.label, score=max(prediction.score, fallback_score))
                        LOGGER.info(
                            "fallback_confirmed %s score=%.3f", canonical_label, fallback_score
                        )  # // SENS_MODE

            decision = self._runtime.decision_engine.process(
                prediction,
                timestamp=timestamp,
                hint_label=source_label,
                latency_ms=latency_ms,
                landmarks=landmarks,
                roi=roi_snapshot,
            )

            self._runtime.clear_error()

            if fallback_used:
                decision.payload.setdefault("fallback_confirmed", True)
                decision.payload.setdefault("fallback_score", round(fallback_score, 4))
                decision.payload.setdefault("fallback_label", fallback_label)

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

        requested_mode = (self.config.sensitivity_mode or DEFAULT_SENS_MODE).strip().upper() or DEFAULT_SENS_MODE
        if requested_mode not in SENSITIVITY_PROFILES:
            LOGGER.warning(
                "Modo de sensibilidad %s no disponible, se utilizará %s", requested_mode, DEFAULT_SENS_MODE
            )
            requested_mode = DEFAULT_SENS_MODE

        self.sensitivity_mode = requested_mode
        self.sensitivity_profile = SENSITIVITY_PROFILES[self.sensitivity_mode]
        self.hysteresis_profile = HYSTERESIS_SETTINGS
        self.rate_limit_profile = RATE_LIMIT_SETTINGS
        LOGGER.info("mode=%s profile=version:%s", self.sensitivity_mode, SENSITIVITY_PROFILE_VERSION)  # // SENS_MODE

        if (
            PI_CAMERA_PROFILE
            and math.isclose(
                float(self.config.poll_interval_s),
                DEFAULT_POLL_INTERVAL_S,
                rel_tol=1e-3,
                abs_tol=1e-3,
            )
        ):
            tuned_interval = PI_CAMERA_PROFILE.poll_interval
            self.config.poll_interval_s = tuned_interval
            LOGGER.info(
                "Intervalo de inferencia ajustado automáticamente a %.3f s para %s",
                tuned_interval,
                PI_CAMERA_PROFILE.model,
            )

        base_frameskip_map = {
            "STRICT": max(1, int(self.rate_limit_profile.frameskip_strict or 1)),
            "BALANCED": max(1, int(self.rate_limit_profile.frameskip_balanced or 1)),
            "RELAXED": max(1, int(self.rate_limit_profile.frameskip_relaxed or 1)),
        }
        self.base_frameskip = base_frameskip_map.get(self.sensitivity_mode, 3)
        stride = int(getattr(self.config, "process_every_n", 0) or 0)
        if stride <= 0:
            stride = self.base_frameskip
        if PI_CAMERA_PROFILE and stride == self.base_frameskip == 3:
            stride = max(1, int(getattr(PI_CAMERA_PROFILE, "process_every_n", stride) or stride))
        self.config.process_every_n = max(1, stride)
        self.active_frameskip = self.config.process_every_n

        self.session_id = uuid.uuid4().hex
        self.started_at = time.time()
        self.event_stream = EventStream()
        self.metrics = GestureMetrics()

        dataset_path = self.config.dataset_path
        primary_exists = (MODEL_DIR / PRIMARY_DATASET_NAME).exists()
        using_fallback = dataset_path.exists() and dataset_path.name != PRIMARY_DATASET_NAME
        self.dataset_info = {
            "path": str(dataset_path),
            "primary_available": primary_exists,
            "using_fallback": using_fallback,
            "exists": dataset_path.exists(),
            "sensitivity_mode": self.sensitivity_mode,
            "profile_version": SENSITIVITY_PROFILE_VERSION,
            "temporal": {
                "cooldown_s": self.sensitivity_profile.temporal.cooldown_s,
                "listen_window_s": self.sensitivity_profile.temporal.listen_window_s,
                "consensus_n": self.sensitivity_profile.temporal.consensus_n,
                "consensus_m": self.sensitivity_profile.temporal.consensus_m,
                "min_pos_stability_var": self.sensitivity_profile.temporal.min_pos_stability_var,
                "activation_delay_s": self.sensitivity_profile.temporal.activation_delay_s,
            },
            "frameskip_base": self.base_frameskip,
        }

        self.feature_normalizer = FeatureNormalizer(dataset_path)
        self.geometry_verifier = self._create_geometry_verifier()
        if self.geometry_verifier:
            self.geometry_verifier.configure(self.sensitivity_profile)

        dynamic_thresholds = _class_thresholds_from_profile(self.sensitivity_profile, self.hysteresis_profile)
        thresholds = dict(DEFAULT_CLASS_THRESHOLDS)
        thresholds.update({label: value for label, value in dynamic_thresholds.items() if label in TRACKED_GESTURES})
        self.class_thresholds = thresholds
        self.global_min_score = min((threshold.enter for threshold in thresholds.values()), default=GLOBAL_MIN_SCORE)
        self.consensus_config = ConsensusConfig(
            window_size=max(1, int(self.sensitivity_profile.temporal.consensus_m or 1)),
            required_votes=max(1, int(self.sensitivity_profile.temporal.consensus_n or 1)),
        )
        if self.consensus_config.required_votes > self.consensus_config.window_size:
            self.consensus_config = ConsensusConfig(
                window_size=self.consensus_config.window_size,
                required_votes=self.consensus_config.window_size,
            )

        self.metrics.configure_context(self.sensitivity_mode, SENSITIVITY_PROFILE_VERSION)

        classifier, classifier_meta = self._create_classifier()
        self.classifier = classifier
        self.model_source = classifier_meta["source"]
        self.model_loaded = classifier_meta["loaded"]
        self.fallback_classifier: Optional[SimpleGestureClassifier] = None
        if not isinstance(self.classifier, SimpleGestureClassifier):
            self.fallback_classifier = self._load_secondary_classifier(dataset_path)
        self.dataset_info["fallback_secondary"] = bool(self.fallback_classifier)

        threshold_log = ", ".join(
            f"{label}={value.enter:.2f}/{value.release:.2f}" for label, value in sorted(self.class_thresholds.items())
        )
        LOGGER.info(
            "Modelo activo source=%s loaded=%s path=%s dataset=%s exists=%s fallback=%s",
            self.model_source,
            self.model_loaded,
            self.config.model_path,
            dataset_path,
            dataset_path.exists(),
            using_fallback,
        )
        LOGGER.info(
            "Umbrales aplicados modo=%s perfil_v%s => %s",
            self.sensitivity_mode,
            SENSITIVITY_PROFILE_VERSION,
            threshold_log,
        )

        stream, stream_meta = self._create_stream()
        self.stream = stream
        self.stream_source = stream_meta["source"]
        self.pipeline = GesturePipeline(
            self,
            interval_s=self.config.poll_interval_s,
            frame_stride=self.config.process_every_n,
        )
        self.decision_engine = GestureDecisionEngine(
            metrics=self.metrics,
            thresholds=self.class_thresholds,
            consensus=self.consensus_config,
            global_min_score=self.global_min_score,
            geometry_verifier=self.geometry_verifier,
            temporal_profile=self.sensitivity_profile.temporal,
            sensitivity_mode=self.sensitivity_mode,
            profile_version=SENSITIVITY_PROFILE_VERSION,
            session_started_at=self.started_at,
        )
        self.lock = threading.Lock()
        self.latency_history: Deque[float] = deque(maxlen=240)
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.last_prediction_at: Optional[float] = None
        self.last_heartbeat = 0.0
        self.last_error: Optional[str] = None

        initial_fps = 0.0
        fps_getter = getattr(self.stream, "measured_fps", None)
        if callable(fps_getter):
            with contextlib.suppress(Exception):
                initial_fps = float(fps_getter())
        self.update_frameskip(initial_fps)

    # ------------------------------------------------------------------
    def _create_geometry_verifier(self) -> Optional[LandmarkGeometryVerifier]:
        primary_dataset = MODEL_DIR / PRIMARY_DATASET_NAME
        if primary_dataset.exists():
            LOGGER.info("Verificación geométrica activada con %s", primary_dataset.name)
            return LandmarkGeometryVerifier()

        LOGGER.warning(
            "Verificación geométrica deshabilitada: %s no está presente. Se utilizarán solo las predicciones del modelo.",
            primary_dataset.name,
        )
        return None

    # ------------------------------------------------------------------
    def _create_classifier(self) -> Tuple[Any, Dict[str, Any]]:
        try:
            classifier = ProductionGestureClassifier(self.config.model_path)
            LOGGER.info("Modelo de producción cargado desde %s", self.config.model_path)
            return classifier, {"source": ProductionGestureClassifier.source, "loaded": True}
        except Exception as error:
            LOGGER.warning("No se pudo cargar el modelo de producción: %s", error)
            dataset_path = self.config.dataset_path
            if not dataset_path.exists():
                _notify_missing_dataset(dataset_path)
                raise RuntimeError("No hay dataset disponible para el clasificador de respaldo") from error
            fallback = SimpleGestureClassifier(dataset_path)
            return fallback, {"source": "synthetic", "loaded": True}

    # ------------------------------------------------------------------
    def _load_secondary_classifier(self, dataset_path: Path) -> Optional[SimpleGestureClassifier]:
        if not dataset_path.exists():
            LOGGER.debug("Dataset de respaldo no disponible para clasificador secundario")
            return None
        try:
            fallback = SimpleGestureClassifier(dataset_path)
            LOGGER.info("Clasificador secundario preparado desde %s", dataset_path)  # // SENS_MODE
            return fallback
        except Exception as error:
            LOGGER.warning("No se pudo inicializar el clasificador secundario: %s", error)
            return None

    # ------------------------------------------------------------------
    def _create_stream(self) -> Tuple[Any, Dict[str, Any]]:
        if self.config.enable_camera:
            try:
                stream = CameraGestureStream(
                    camera_index=self.config.camera_index,
                    detection_confidence=self.config.detection_confidence,
                    tracking_confidence=self.config.tracking_confidence,
                    metrics=self.metrics,
                    quality_profile=self.sensitivity_profile.quality,
                    sensitivity_mode=self.sensitivity_mode,
                )
                LOGGER.info("Usando cámara física en el índice %s", self.config.camera_index)
                return stream, {"source": CameraGestureStream.source}
            except Exception as error:
                LOGGER.warning("No se pudo inicializar la cámara: %s", error)
                if not self.config.fallback_to_synthetic:
                    raise

        dataset_path = self.config.dataset_path
        if not dataset_path.exists():
            _notify_missing_dataset(dataset_path)
            raise RuntimeError("No se puede iniciar el flujo sintético: falta el dataset")

        LOGGER.info("Usando flujo sintético de gestos desde %s", dataset_path)
        return SyntheticStreamAdapter(dataset_path), {"source": "synthetic"}

    # ------------------------------------------------------------------
    def start(self) -> None:
        self.pipeline.start()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self.pipeline.stop()
        close_stream = getattr(self.stream, "close", None)
        if callable(close_stream):
            close_stream()
        self._export_session_report()
        self._export_tuning_summary()

    # ------------------------------------------------------------------
    def update_frameskip(self, measured_fps: float) -> int:
        target = self.base_frameskip
        if self.sensitivity_mode == "BALANCED":
            if measured_fps > self.rate_limit_profile.fps_threshold:
                target = max(1, int(self.rate_limit_profile.frameskip_balanced or target))
            else:
                target = 1
        elif self.sensitivity_mode == "RELAXED":
            target = max(1, int(self.rate_limit_profile.frameskip_relaxed or target))
        else:
            target = max(1, int(self.rate_limit_profile.frameskip_strict or target))

        if target != self.active_frameskip:
            self.active_frameskip = target
            if hasattr(self, "pipeline") and self.pipeline:
                self.pipeline.set_frame_stride(target)
        return self.active_frameskip  # // SENS_MODE

    # ------------------------------------------------------------------
    def register_heartbeat(self) -> None:
        with self.lock:
            self.last_heartbeat = time.time()

    # ------------------------------------------------------------------
    def clear_error(self) -> None:
        with self.lock:
            self.last_error = None

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
            "mode": self.sensitivity_mode,
            "profile_version": SENSITIVITY_PROFILE_VERSION,
            "frameskip_used": int(self.active_frameskip),
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
        with self.lock:
            self.last_prediction = event
            self.last_prediction_at = time.time()
            self.last_heartbeat = time.time()
            self.latency_history.append(float(event.get("latency_ms", 0.0)))

        LOGGER.debug("Broadcasting event: %s", event)
        self.event_stream.broadcast(event)

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
            dataset_info["frameskip_used"] = int(self.active_frameskip)
            dataset_info["baseline_thresholds"] = {
                label: {"enter": value.enter, "release": value.release}
                for label, value in self.decision_engine.baseline_thresholds().items()
            }
            dataset_info["current_thresholds"] = {
                label: {"enter": value.enter, "release": value.release}
                for label, value in self.decision_engine.thresholds().items()
            }
            dataset_info["threshold_adjustments"] = self.decision_engine.threshold_adjustments()
            report_path = REPO_ROOT / "reports" / "gesture_session_report.md"
            report = self.metrics.generate_report(
                thresholds=self.decision_engine.thresholds(),
                consensus=self.decision_engine.consensus_config,
                dataset_info=dataset_info,
                latency_stats=latency_stats,
            )
            markdown = self.metrics.to_markdown(report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(markdown, encoding="utf-8")
            json_path = report_path.with_suffix(".json")
            json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            logs_path = REPO_ROOT / "logs" / f"metrics_{self.sensitivity_mode}.json"
            logs_path.parent.mkdir(parents=True, exist_ok=True)
            logs_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            LOGGER.info("Reporte de métricas actualizado en %s", report_path)
        except Exception as error:  # pragma: no cover - escritura opcional
            LOGGER.warning("No se pudo escribir el reporte de métricas: %s", error)

    # ------------------------------------------------------------------
    def _export_tuning_summary(self) -> None:
        try:
            before = self.decision_engine.baseline_thresholds()
            after = self.decision_engine.thresholds()
            adjustments = self.decision_engine.threshold_adjustments()
            consensus = self.decision_engine.consensus_config
            temporal = self.sensitivity_profile.temporal

            lines = [
                "# Ajustes de umbrales",
                "",
                f"- Sesión: {self.session_id}",
                f"- Modo: {self.sensitivity_mode} (perfil v{SENSITIVITY_PROFILE_VERSION})",
                f"- Modelo: {self.model_source} ({self.config.model_path})",
                "- Consenso requerido: "
                f"{consensus.required_votes}/{consensus.window_size}",
                f"- Ventana de comandos: {temporal.listen_window_s:.2f} s",
                f"- Cooldown tras H: {temporal.cooldown_s:.2f} s",
                f"- Retardo de activación: {temporal.activation_delay_s:.2f} s",
                "",
                "| Gesto | Enter inicial | Enter final | Δ | Release final |",
                "|-------|--------------:|------------:|----:|--------------:|",
            ]

            for label in sorted(set(before) | set(after)):
                base = before.get(label)
                current = after.get(label, base)
                if base is None or current is None:
                    continue
                delta = current.enter - base.enter
                lines.append(
                    "| {label} | {enter_before:.3f} | {enter_after:.3f} | {delta:+.3f} | {release_after:.3f} |".format(
                        label=label,
                        enter_before=base.enter,
                        enter_after=current.enter,
                        delta=delta,
                        release_after=current.release,
                    )
                )

            lines.append("")
            if adjustments:
                lines.append("## Ajustes aplicados")
                for label, delta in sorted(adjustments.items()):
                    lines.append(f"- {label}: {delta:+.3f}")
            else:
                lines.append("No se aplicaron ajustes automáticos de umbral.")

            summary_path = REPO_ROOT / "reports" / "tuning_summary.md"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            health_snapshot = self.health()
            thresholds_log = ", ".join(
                f"{label}={values['enter']:.2f}/{values['release']:.2f}"
                for label, values in health_snapshot.thresholds.items()
            )
            LOGGER.info(
                "diagnostic session=%s mode=%s model=%s frames_total=%d frames_valid=%d fps=%.2f latency_avg=%.2fms "
                "latency_p95=%.2fms thresholds={%s} reasons=%s",
                self.session_id,
                self.sensitivity_mode,
                health_snapshot.model_path,
                health_snapshot.frames_total,
                health_snapshot.frames_valid,
                health_snapshot.fps,
                health_snapshot.avg_latency_ms,
                health_snapshot.latency_p95_ms,
                thresholds_log,
                health_snapshot.reason_counts,
            )
        except Exception as error:  # pragma: no cover - escritura opcional
            LOGGER.warning("No se pudo generar el resumen de ajustes: %s", error)

    # ------------------------------------------------------------------
    def health(self) -> HealthSnapshot:
        with self.lock:
            last_prediction = self.last_prediction
            last_prediction_at = self.last_prediction_at
            last_error = self.last_error
            heartbeat_age = time.time() - self.last_heartbeat if self.last_heartbeat else None

        latency_stats = self._latency_snapshot()
        pipeline_running = self.pipeline.is_running()
        stream_status = getattr(self.stream, "status", lambda: {})()
        camera_ok = bool(stream_status.get("healthy")) if self.stream_source == "camera" else True

        metrics_summary = self.metrics.summary()
        thresholds_dict = {
            label: {"enter": value.enter, "release": value.release}
            for label, value in self.decision_engine.thresholds().items()
        }
        threshold_adjustments = self.decision_engine.threshold_adjustments()

        dataset_snapshot = {
            **self.dataset_info,
            "temporal": dict(self.dataset_info.get("temporal", {})),
            "metrics_summary": metrics_summary,
        }

        status = "HEALTHY"
        if last_error:
            status = "ERROR"
        elif not pipeline_running or (heartbeat_age is not None and heartbeat_age > 5.0) or not camera_ok or not self.model_loaded:
            status = "DEGRADED"

        fps_value = float(stream_status.get("measured_fps", 0.0) or 0.0)

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
            avg_latency_ms=round(latency_stats.get("avg_ms", 0.0), 3),
            latency_p95_ms=round(latency_stats.get("p95_ms", 0.0), 3),
            latency_max_ms=round(latency_stats.get("max_ms", 0.0), 3),
            latency_count=int(latency_stats.get("count", 0)),
            camera_ok=camera_ok,
            camera_index=stream_status.get("camera_index"),
            camera_backend=stream_status.get("capture_backend"),
            camera_last_capture=(
                _iso_timestamp(stream_status["last_capture"])
                if stream_status.get("last_capture")
                else None
            ),
            camera_last_error=stream_status.get("last_error"),
            frames_total=int(stream_status.get("frames_total") or 0),
            frames_with_hand=int(stream_status.get("frames_with_hand") or 0),
            frames_valid=int(stream_status.get("frames_valid") or 0),
            frames_returned=int(stream_status.get("frames_returned") or 0),
            fps=round(fps_value, 2),
            quality_rejections=metrics_summary.get("quality_rejections", {}),
            reason_counts=metrics_summary.get("reason_counts", {}),
            thresholds=thresholds_dict,
            threshold_adjustments=threshold_adjustments,
            model_path=str(self.config.model_path),
            engine_mode=self.sensitivity_mode,
            sensitivity_version=SENSITIVITY_PROFILE_VERSION,
            last_quality_reason=stream_status.get("last_quality_reason"),
            dataset_info=dataset_snapshot,
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

        decision_engine = getattr(self.runtime, "decision_engine", None)
        if decision_engine is not None:
            decision_engine.defer_rearm(duration=REARM_START_MIN_DELAY)

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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HELEN backend server")
    parser.add_argument("--host", default="0.0.0.0", help="Dirección de enlace del servidor HTTP")
    parser.add_argument("--port", type=int, default=5000, help="Puerto del servidor HTTP")
    parser.add_argument("--camera-index", type=int, default=0, help="Índice de la cámara de video a utilizar")
    parser.add_argument("--detection-confidence", type=float, default=0.7, help="Umbral de detección de MediaPipe")
    parser.add_argument("--tracking-confidence", type=float, default=0.6, help="Umbral de seguimiento de MediaPipe")
    parser.add_argument("--poll-interval", type=float, default=0.12, help="Intervalo entre inferencias en segundos")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=3,
        help="Procesa un frame de cada N muestras para reducir carga (>=1)",
    )
    parser.add_argument("--no-camera", action="store_true", help="Desactiva el uso de cámara física")
    parser.add_argument(
        "--no-synthetic-fallback",
        action="store_true",
        help="Falla si la cámara no está disponible en lugar de usar el dataset sintético",
    )
    parser.add_argument(
        "--sensitivity-mode",
        choices=sorted(SENSITIVITY_MODES),
        default=None,
        help="Override del modo de sensibilidad (STRICT, BALANCED, RELAXED)",
    )

    args = parser.parse_args(argv)
    config = RuntimeConfig(
        camera_index=args.camera_index,
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence,
        poll_interval_s=args.poll_interval,
        enable_camera=not args.no_camera,
        fallback_to_synthetic=not args.no_synthetic_fallback,
        process_every_n=max(1, args.frame_stride),
        sensitivity_mode=args.sensitivity_mode,
    )

    run(args.host, args.port, config=config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
