"""Camera probing utilities for Raspberry Pi deployments.

This module enumerates V4L2 and libcamera sources, validates them with
OpenCV, and persists the most reliable option so the backend can start
without interactive configuration.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - optional dependency in some CI environments
    import cv2  # type: ignore
except Exception:  # pragma: no cover - handled gracefully at runtime
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency in some CI environments
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - handled gracefully at runtime
    np = None  # type: ignore

LOGGER = logging.getLogger("helen.camera_probe")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
LOG_DIR = REPORTS_DIR / "logs" / "pi"
CONFIG_DIR = REPORTS_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "camera_selection.json"

BLACK_FRAME_MEAN_THRESHOLD = 5.0
BLACK_FRAME_STD_THRESHOLD = 3.5
PROBE_TIMEOUT_S = 4.0
FRAME_SAMPLE_LIMIT = 18
LATENCY_FALLBACK_MS = 9999.0


@dataclass(frozen=True)
class CameraMode:
    width: int
    height: int
    fps: int

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}@{self.fps}"


PREFERRED_MODES: Tuple[CameraMode, ...] = (
    CameraMode(640, 480, 30),
    CameraMode(640, 360, 24),
    CameraMode(320, 240, 24),
)


@dataclass
class CameraCandidate:
    identifier: str
    label: str
    kind: str  # usb | csi | unknown
    backend_hint: str  # v4l2 | gstreamer
    path: Optional[str] = None
    index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        return {
            "id": self.identifier,
            "label": self.label,
            "kind": self.kind,
            "backend": self.backend_hint,
            "path": self.path,
            "index": self.index,
            "metadata": dict(self.metadata),
        }


@dataclass
class ProbeResult:
    candidate: CameraCandidate
    backend: str
    mode: CameraMode
    success: bool
    reason: Optional[str] = None
    latency_ms: float = LATENCY_FALLBACK_MS
    resolution: Tuple[int, int] = (0, 0)
    fps: float = 0.0
    orientation: Optional[str] = None
    frames_sampled: int = 0
    pipeline: Optional[str] = None

    def score(self) -> float:
        base = 0.0
        if not self.success:
            return -1000.0
        if self.candidate.kind == "csi":
            base += 200.0
        elif self.candidate.kind == "usb":
            base += 150.0
        else:
            base += 100.0

        base += min(self.fps, 45.0)
        base += min(self.resolution[0], self.resolution[1]) / 40.0
        base -= min(self.latency_ms, 800.0) / 25.0
        return base


@dataclass
class CameraSelection:
    backend: str
    device: Optional[str]
    index: Optional[int]
    pipeline: Optional[str]
    width: int
    height: int
    fps: float
    latency_ms: float
    orientation: Optional[str]
    kind: str
    mode_name: str
    hardware_signature: str
    probed_at: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_read_text(path: Path) -> str:
    with contextlib.suppress(OSError, UnicodeDecodeError):
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def _extract_index(identifier: str) -> Optional[int]:
    digits = re.findall(r"(\d+)", identifier)
    if not digits:
        return None
    try:
        return int(digits[-1])
    except ValueError:
        return None


def _list_v4l2_devices() -> List[CameraCandidate]:
    devices: List[CameraCandidate] = []
    video_root = Path("/dev")
    if not video_root.exists():
        return devices

    for entry in sorted(video_root.glob("video*")):
        if not entry.is_char_device():
            continue
        identifier = entry.name
        index = _extract_index(identifier)
        sysfs = Path("/sys/class/video4linux") / identifier
        label = _safe_read_text(sysfs / "name") or identifier
        kind = "usb"
        lower_label = label.lower()
        if "unicam" in lower_label or "csi" in lower_label or "imx" in lower_label:
            kind = "csi"
        candidate = CameraCandidate(
            identifier=f"v4l2:{identifier}",
            label=label,
            kind=kind,
            backend_hint="v4l2",
            path=str(entry),
            index=index,
        )
        devices.append(candidate)
    return devices


def _run_command(args: Sequence[str], timeout: float = 3.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_libcamera_output(text: str) -> List[CameraCandidate]:
    candidates: List[CameraCandidate] = []
    for line in text.splitlines():
        match = re.match(r"\s*(\d+)\s*[:\-]\s*(.+)", line)
        if not match:
            continue
        index = int(match.group(1))
        description = match.group(2).strip()
        label = description
        identifier = f"libcamera:{index}"
        metadata: Dict[str, Any] = {"description": description}
        resolution_match = re.search(r"(\d+)x(\d+)", description)
        if resolution_match:
            metadata["resolution_hint"] = (
                int(resolution_match.group(1)),
                int(resolution_match.group(2)),
            )
        candidate = CameraCandidate(
            identifier=identifier,
            label=label,
            kind="csi",
            backend_hint="gstreamer",
            index=index,
            metadata=metadata,
        )
        candidates.append(candidate)
    return candidates


def _list_libcamera_devices() -> List[CameraCandidate]:
    commands = [
        ["libcamera-hello", "--list-cameras"],
        ["rpicam-hello", "--list-cameras"],
    ]
    for cmd in commands:
        try:
            result = _run_command(cmd, timeout=4.0)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0:
            continue
        parsed = _parse_libcamera_output(result.stdout)
        if parsed:
            return parsed
    return []


def list_sources() -> Dict[str, List[Dict[str, Any]]]:
    """Return discovered V4L2 and libcamera sources for diagnostic UIs."""

    v4l2 = [candidate.describe() for candidate in _list_v4l2_devices()]
    libcamera = [candidate.describe() for candidate in _list_libcamera_devices()]
    return {"v4l2": v4l2, "libcamera": libcamera}


# ---------------------------------------------------------------------------
# Probe utilities
# ---------------------------------------------------------------------------


def _frame_is_valid(frame: Any) -> bool:
    if frame is None:
        return False
    try:
        height, width = frame.shape[:2]
    except Exception:  # pragma: no cover - depends on OpenCV dtype
        return False
    if height <= 0 or width <= 0:
        return False

    if np is not None:
        mean_value = float(np.mean(frame))
        std_dev = float(np.std(frame))
    else:  # pragma: no cover - numpy should be available but guard just in case
        mean_tuple = cv2.mean(frame) if cv2 is not None else (0.0,)
        mean_value = float(sum(mean_tuple)) / max(len(mean_tuple), 1)
        std_dev = 0.0

    return mean_value > BLACK_FRAME_MEAN_THRESHOLD or std_dev > BLACK_FRAME_STD_THRESHOLD


def _configure_capture(cap: Any, mode: CameraMode) -> None:
    with contextlib.suppress(Exception):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode.width)
    with contextlib.suppress(Exception):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode.height)
    with contextlib.suppress(Exception):
        cap.set(cv2.CAP_PROP_FPS, mode.fps)


def _read_frames(cap: Any, *, timeout: float = PROBE_TIMEOUT_S) -> Tuple[bool, float, Tuple[int, int], float, int]:
    start = time.time()
    first_valid: Optional[float] = None
    resolution = (0, 0)
    fps = 0.0
    frames = 0

    while time.time() - start < timeout and frames < FRAME_SAMPLE_LIMIT:
        ok, frame = cap.read()
        frames += 1
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        height, width = frame.shape[:2]
        resolution = (int(width), int(height))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if _frame_is_valid(frame):
            first_valid = time.time()
            break
        time.sleep(0.05)

    success = first_valid is not None
    latency = (first_valid - start) * 1000.0 if first_valid else LATENCY_FALLBACK_MS
    return success, latency, resolution, fps, frames


def _probe_with_v4l2(candidate: CameraCandidate, mode: CameraMode) -> ProbeResult:
    if cv2 is None:
        return ProbeResult(candidate=candidate, backend="v4l2", mode=mode, success=False, reason="opencv-missing")

    target: Union[str, int, None] = candidate.path if candidate.path else candidate.index
    if target is None:
        return ProbeResult(candidate=candidate, backend="v4l2", mode=mode, success=False, reason="no-target")

    backend_flag = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
    try:
        cap = cv2.VideoCapture(target, backend_flag) if backend_flag else cv2.VideoCapture(target)
    except Exception as error:  # pragma: no cover - depends on runtime
        return ProbeResult(candidate=candidate, backend="v4l2", mode=mode, success=False, reason=str(error))

    if not cap or not cap.isOpened():
        if cap:
            with contextlib.suppress(Exception):
                cap.release()
        return ProbeResult(candidate=candidate, backend="v4l2", mode=mode, success=False, reason="open-failed")

    _configure_capture(cap, mode)
    success, latency, resolution, fps, frames = _read_frames(cap)

    with contextlib.suppress(Exception):
        cap.release()

    orientation = None
    if resolution[0] and resolution[1]:
        orientation = "portrait" if resolution[1] > resolution[0] else "landscape"

    return ProbeResult(
        candidate=candidate,
        backend="v4l2",
        mode=mode,
        success=success,
        reason=None if success else "no-valid-frame",
        latency_ms=latency,
        resolution=resolution,
        fps=fps,
        orientation=orientation,
        frames_sampled=frames,
    )


def _build_gstreamer_pipeline(candidate: CameraCandidate, mode: CameraMode) -> str:
    if candidate.kind == "usb" and candidate.path:
        return (
            f"v4l2src device={candidate.path} ! video/x-raw,width={mode.width},height={mode.height},"
            f"framerate={mode.fps}/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2"
        )
    return (
        "libcamerasrc ! video/x-raw,width="
        f"{mode.width},height={mode.height},framerate={mode.fps}/1,format=RGB ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2"
    )


def _probe_with_gstreamer(candidate: CameraCandidate, mode: CameraMode) -> ProbeResult:
    if cv2 is None:
        return ProbeResult(candidate=candidate, backend="gstreamer", mode=mode, success=False, reason="opencv-missing")
    pipeline = _build_gstreamer_pipeline(candidate, mode)
    try:
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    except Exception as error:  # pragma: no cover - depends on runtime
        return ProbeResult(candidate=candidate, backend="gstreamer", mode=mode, success=False, reason=str(error))

    if not cap or not cap.isOpened():
        if cap:
            with contextlib.suppress(Exception):
                cap.release()
        return ProbeResult(candidate=candidate, backend="gstreamer", mode=mode, success=False, reason="open-failed", pipeline=pipeline)

    success, latency, resolution, fps, frames = _read_frames(cap)
    with contextlib.suppress(Exception):
        cap.release()

    orientation = None
    if resolution[0] and resolution[1]:
        orientation = "portrait" if resolution[1] > resolution[0] else "landscape"

    return ProbeResult(
        candidate=candidate,
        backend="gstreamer",
        mode=mode,
        success=success,
        reason=None if success else "no-valid-frame",
        latency_ms=latency,
        resolution=resolution,
        fps=fps,
        orientation=orientation,
        frames_sampled=frames,
        pipeline=pipeline,
    )


def _probe_candidate(candidate: CameraCandidate) -> Optional[ProbeResult]:
    attempts: List[ProbeResult] = []
    preferred_backends: Sequence[str]
    if candidate.backend_hint == "gstreamer":
        preferred_backends = ("gstreamer", "v4l2")
    else:
        preferred_backends = ("v4l2", "gstreamer")

    for mode in PREFERRED_MODES:
        for backend in preferred_backends:
            if backend == "v4l2":
                result = _probe_with_v4l2(candidate, mode)
            else:
                result = _probe_with_gstreamer(candidate, mode)
            attempts.append(result)
            if result.success:
                return result
    if attempts:
        return max(attempts, key=lambda item: item.score())
    return None


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _hardware_signature() -> str:
    signature_parts: List[str] = []
    for candidate in _list_v4l2_devices():
        signature_parts.append(f"{candidate.identifier}:{candidate.label}")
    libcamera_output = []
    for candidate in _list_libcamera_devices():
        libcamera_output.append(f"{candidate.identifier}:{candidate.label}")
    signature_parts.extend(libcamera_output)
    signature_parts.append(platform.machine())
    signature = "|".join(signature_parts)
    return hashlib.sha256(signature.encode("utf-8", errors="ignore")).hexdigest()


def _load_cached_selection() -> Optional[CameraSelection]:
    if not CONFIG_PATH.exists():
        return None
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - depends on filesystem
        return None
    return CameraSelection(**payload)


def _save_selection(selection: CameraSelection) -> None:
    _ensure_dirs()
    CONFIG_PATH.write_text(json.dumps(selection.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"camera-probe-{timestamp}.json"
    log_payload = selection.to_dict()
    log_path.write_text(json.dumps(log_payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_camera_selection(
    *,
    force: bool = False,
    preferred: Optional[Union[str, int]] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[CameraSelection]:
    """Return the cached camera selection or probe the available devices."""

    if cv2 is None:
        if logger:
            logger.warning("OpenCV no está disponible; se omite la auto-detección de cámara.")
        return None

    current_signature = _hardware_signature()
    cached = _load_cached_selection()
    if not force and cached and cached.hardware_signature == current_signature:
        if logger:
            logger.info(
                "Selección de cámara reutilizada: backend=%s device=%s pipeline=%s",
                cached.backend,
                cached.device or cached.index,
                cached.pipeline,
            )
        return cached

    candidates = _list_libcamera_devices() + _list_v4l2_devices()
    if not candidates:
        if logger:
            logger.warning("No se detectaron cámaras V4L2/libcamera disponibles.")
        return None

    preferred_str: Optional[str] = None
    if preferred is not None:
        preferred_str = str(preferred)

    best_result: Optional[ProbeResult] = None
    for candidate in candidates:
        if logger:
            logger.info("Probing %s (%s)", candidate.identifier, candidate.label)
        result = _probe_candidate(candidate)
        if not result:
            continue
        if preferred_str and (preferred_str == str(candidate.index) or preferred_str in (candidate.path or "")):
            # Boost preferred candidate if it succeeded
            if result.success:
                best_result = result
                break
        if not best_result or result.score() > best_result.score():
            best_result = result

    if not best_result or not best_result.success:
        if logger:
            reason = best_result.reason if best_result else "unknown"
            logger.warning("No se pudo validar ninguna cámara física (%s)", reason)
        return None

    mode = best_result.mode
    selection = CameraSelection(
        backend=best_result.backend,
        device=best_result.candidate.path,
        index=best_result.candidate.index,
        pipeline=best_result.pipeline,
        width=best_result.resolution[0] or mode.width,
        height=best_result.resolution[1] or mode.height,
        fps=best_result.fps or float(mode.fps),
        latency_ms=best_result.latency_ms,
        orientation=best_result.orientation,
        kind=best_result.candidate.kind,
        mode_name=mode.label,
        hardware_signature=current_signature,
        probed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    _save_selection(selection)
    if logger:
        logger.info(
            "Cámara seleccionada: backend=%s device=%s pipeline=%s (%sx%s @ %.2f fps, %.1f ms)",
            selection.backend,
            selection.device or selection.index,
            selection.pipeline or "<v4l2>",
            selection.width,
            selection.height,
            selection.fps,
            selection.latency_ms,
        )
    return selection


def probe_specific_device(
    identifier: Union[str, int],
    *,
    width: int,
    height: int,
    fps: int,
) -> ProbeResult:
    """Probe a specific device or pipeline for CLI diagnostics."""

    if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
        index = int(identifier)
        candidate = CameraCandidate(
            identifier=f"manual:{index}",
            label=f"/dev/video{index}",
            kind="usb",
            backend_hint="v4l2",
            path=f"/dev/video{index}",
            index=index,
        )
    else:
        candidate = CameraCandidate(
            identifier=f"manual:{identifier}",
            label=str(identifier),
            kind="usb",
            backend_hint="v4l2",
            path=str(identifier),
        )
    mode = CameraMode(width=width, height=height, fps=fps)
    primary = _probe_with_v4l2(candidate, mode)
    if primary.success:
        return primary
    secondary = _probe_with_gstreamer(candidate, mode)
    if secondary.success:
        return secondary
    return secondary if secondary.reason else primary


def parse_resolution(value: str) -> Tuple[int, int]:
    match = re.match(r"^(\d+)[xX](\d+)$", value.strip())
    if not match:
        raise ValueError(f"Resolución inválida: {value}")
    return int(match.group(1)), int(match.group(2))


__all__ = [
    "CameraSelection",
    "list_sources",
    "ensure_camera_selection",
    "probe_specific_device",
    "parse_resolution",
]
