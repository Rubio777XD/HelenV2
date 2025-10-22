"""Static decision parameters for the simplified gesture engine.

The previous implementation supported configurable runtime tuning loaded from
JSON files. That flexibility introduced a large amount of complexity and
ultimately caused regressions in the gesture classifier. The new backend
purposely keeps a single set of permissive parameters baked into the source code
so the behaviour is predictable across platforms and deployments.

These values were tuned to favour recall while keeping false positives under
control.  Thresholds were reduced by roughly ten percent compared to the
legacy settings and the geometric checks allow slightly wider tolerances so
natural variations of the hand do not prevent detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ClassThreshold:
    """Score thresholds to enter and release a gesture."""

    enter: float
    release: float


@dataclass(frozen=True)
class GeometrySettings:
    """Heuristics applied to the MediaPipe landmarks."""

    angle_tolerance_deg: float
    norm_deviation_max: float
    gap_ratio_range: Optional[Tuple[float, float]]
    curvature_min: Optional[float]
    missing_landmarks_allowed: int
    strict_curvature_min: Optional[float]
    curvature_boost: float
    boost_window: int
    boost_min_frames: int
    boost_tolerance: float


@dataclass(frozen=True)
class QualitySettings:
    """Image quality filters applied before classification."""

    blur_laplacian_min: float
    roi_min_coverage: float
    hand_range_px: Tuple[float, float]
    roi_margin: float


@dataclass(frozen=True)
class TemporalSettings:
    """Temporal smoothing and state machine parameters."""

    consensus_required: int
    consensus_window: int
    cooldown_s: float
    listen_window_s: float
    activation_delay_s: float
    command_debounce_s: float
    cooldown_start_min_delay: float
    min_position_variance: float


@dataclass(frozen=True)
class RateControl:
    """Fixed inference cadence."""

    process_every_n: int
    poll_interval_s: float


@dataclass(frozen=True)
class DecisionSettings:
    """Container bundling all constants used by the backend."""

    thresholds: Dict[str, ClassThreshold]
    global_min_score: float
    quality: QualitySettings
    temporal: TemporalSettings
    geometry: Dict[str, GeometrySettings]
    rate: RateControl


DECISION_SETTINGS = DecisionSettings(
    thresholds={
        "Start": ClassThreshold(enter=0.369, release=0.259),
        "Clima": ClassThreshold(enter=0.342, release=0.242),
        "Reloj": ClassThreshold(enter=0.351, release=0.251),
        "Inicio": ClassThreshold(enter=0.351, release=0.251),
    },
    global_min_score=0.32,
    quality=QualitySettings(
        blur_laplacian_min=24.0,
        roi_min_coverage=0.52,
        hand_range_px=(55.0, 640.0),
        roi_margin=0.05,
    ),
    temporal=TemporalSettings(
        consensus_required=2,
        consensus_window=4,
        cooldown_s=0.7,
        listen_window_s=5.5,
        activation_delay_s=0.45,
        command_debounce_s=0.75,
        cooldown_start_min_delay=1.2,
        min_position_variance=18.0,
    ),
    geometry={
        "Start": GeometrySettings(
            angle_tolerance_deg=38.0,
            norm_deviation_max=0.45,
            gap_ratio_range=None,
            curvature_min=None,
            missing_landmarks_allowed=2,
            strict_curvature_min=None,
            curvature_boost=0.0,
            boost_window=0,
            boost_min_frames=0,
            boost_tolerance=0.0,
        ),
        "Clima": GeometrySettings(
            angle_tolerance_deg=48.0,
            norm_deviation_max=0.49,
            gap_ratio_range=(0.17, 0.74),
            curvature_min=0.18,
            missing_landmarks_allowed=2,
            strict_curvature_min=0.22,
            curvature_boost=0.03,
            boost_window=3,
            boost_min_frames=2,
            boost_tolerance=0.10,
        ),
        "Reloj": GeometrySettings(
            angle_tolerance_deg=38.0,
            norm_deviation_max=0.42,
            gap_ratio_range=None,
            curvature_min=None,
            missing_landmarks_allowed=1,
            strict_curvature_min=None,
            curvature_boost=0.0,
            boost_window=0,
            boost_min_frames=0,
            boost_tolerance=0.0,
        ),
        "Inicio": GeometrySettings(
            angle_tolerance_deg=38.0,
            norm_deviation_max=0.42,
            gap_ratio_range=None,
            curvature_min=None,
            missing_landmarks_allowed=1,
            strict_curvature_min=None,
            curvature_boost=0.0,
            boost_window=0,
            boost_min_frames=0,
            boost_tolerance=0.0,
        ),
    },
    rate=RateControl(process_every_n=3, poll_interval_s=0.12),
)


__all__ = [
    "ClassThreshold",
    "DecisionSettings",
    "GeometrySettings",
    "QualitySettings",
    "RateControl",
    "TemporalSettings",
    "DECISION_SETTINGS",
]

