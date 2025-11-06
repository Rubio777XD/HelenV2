"""Camera probing utilities for HELEN deployments.

This module enumerates available capture sources (V4L2, libcamera and
DirectShow), validates them with OpenCV, and persists the most reliable
option so the backend can start without interactive configuration.
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
import shutil
import subprocess
import sys
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

SYSTEM_NAME = platform.system()
PLATFORM = SYSTEM_NAME.lower()
IS_WINDOWS = sys.platform.startswith("win") or SYSTEM_NAME == "Windows"
IS_LINUX = sys.platform.startswith("linux") or SYSTEM_NAME == "Linux"
IS_MAC = sys.platform.startswith("darwin") or SYSTEM_NAME == "Darwin"

LOGGER = logging.getLogger("helen.camera_probe")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


def _default_capture_flag() -> int:
    if cv2 is None:
        return 0
    if IS_WINDOWS and hasattr(cv2, "CAP_DSHOW"):
        return int(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_V4L2"):
        return int(cv2.CAP_V4L2)
    if hasattr(cv2, "CAP_ANY"):
        return int(getattr(cv2, "CAP_ANY"))
    return 0


DEFAULT_CAPTURE_FLAG = _default_capture_flag()


def normalize_backend_name(value: Optional[str]) -> Optional[str]:
    """Return a normalised backend identifier or ``None``."""

    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    mapping = {
        "dshow": "directshow",
        "direct_show": "directshow",
        "direct-show": "directshow",
        "directshow": "directshow",
        "msmf": "directshow",
        "video4linux": "v4l2",
        "v4l": "v4l2",
        "libcamera": "gstreamer",
        "gst": "gstreamer",
        "gstreamer": "gstreamer",
        "any": None,
        "auto": None,
    }
    return mapping.get(normalized, normalized)


def resolve_backend_flag(name: Optional[str]) -> Optional[int]:
    """Map a backend name to the OpenCV capture flag."""

    if cv2 is None:
        return None

    normalized = normalize_backend_name(name)
    if normalized is None:
        if DEFAULT_CAPTURE_FLAG:
            return int(DEFAULT_CAPTURE_FLAG)
        return int(getattr(cv2, "CAP_ANY", 0)) if hasattr(cv2, "CAP_ANY") else None

    if normalized == "directshow":
        if hasattr(cv2, "CAP_DSHOW"):
            return int(cv2.CAP_DSHOW)
        if DEFAULT_CAPTURE_FLAG:
            return int(DEFAULT_CAPTURE_FLAG)
        return int(getattr(cv2, "CAP_ANY", 0)) if hasattr(cv2, "CAP_ANY") else None

    if normalized == "v4l2":
        if hasattr(cv2, "CAP_V4L2"):
            return int(cv2.CAP_V4L2)
        return int(getattr(cv2, "CAP_ANY", 0)) if hasattr(cv2, "CAP_ANY") else None

    if normalized == "gstreamer":
        return int(getattr(cv2, "CAP_GSTREAMER", DEFAULT_CAPTURE_FLAG or 0))

    return int(DEFAULT_CAPTURE_FLAG) if DEFAULT_CAPTURE_FLAG else int(getattr(cv2, "CAP_ANY", 0)) if hasattr(cv2, "CAP_ANY") else None


def preferred_backend_order(preferred: Optional[str] = None) -> Tuple[str, ...]:
    """Return a tuple with the preferred backend order for the platform."""

    order: List[str] = []
    normalized = normalize_backend_name(preferred)
    if normalized:
        order.append(normalized)
    if IS_WINDOWS:
        order.append("directshow")
    order.append("v4l2")
    order.append("gstreamer")

    seen: set[str] = set()
    deduped: List[str] = []
    for backend in order:
        backend_normalized = normalize_backend_name(backend)
        if not backend_normalized:
            continue
        if backend_normalized not in seen:
            seen.add(backend_normalized)
            deduped.append(backend_normalized)
    return tuple(deduped)


def _log(logger: Optional[logging.Logger], level: str, message: str, *args: Any) -> None:
    target = logger if logger is not None else LOGGER
    if not target:
        return
    handler = getattr(target, level, None)
    if callable(handler):
        handler(message, *args)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"

LOG_DIR = REPORTS_DIR / "logs" / ("win" if IS_WINDOWS else "pi")
CONFIG_DIR = REPORTS_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "camera_selection.json"

CAMERA_NOT_FOUND_MESSAGE = "❌ Cámara no detectada. Verifique conexión o permisos."

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
    CameraMode(1280, 720, 30),
    CameraMode(960, 720, 30),
    CameraMode(640, 480, 30),
    CameraMode(640, 360, 24),
    CameraMode(320, 240, 24),
)

PIXEL_FORMAT_PREFERENCE: Tuple[str, ...] = ("MJPG", "YUYV", "NV12", "H264")


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
    pixel_format: Optional[str] = None

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
    pixel_format: Optional[str] = None

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


def _collect_v4l2ctl_metadata() -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    try:
        result = _run_command(["v4l2-ctl", "--list-devices"], timeout=4.0)
    except (OSError, subprocess.TimeoutExpired):
        return metadata
    if result.returncode != 0 or not result.stdout:
        return metadata

    current_label: Optional[str] = None
    for raw_line in result.stdout.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            current_label = None
            continue
        if not line.startswith("\t") and not line.startswith("    "):
            current_label = line.rstrip(":")
            continue
        path = line.strip()
        if not path.startswith("/dev/video"):
            continue
        kind = "usb"
        label_lower = (current_label or "").lower()
        if any(keyword in label_lower for keyword in ("unicam", "csi", "imx", "ov", "raspberry", "pi camera")):
            kind = "csi"
        metadata[path] = {
            "label": current_label or path,
            "kind": kind,
        }
    return metadata


def _list_v4l2_devices() -> List[CameraCandidate]:
    devices: List[CameraCandidate] = []
    video_root = Path("/dev")
    if not video_root.exists():
        return devices

    ctl_metadata = _collect_v4l2ctl_metadata()

    for entry in sorted(video_root.glob("video*")):
        if not entry.is_char_device():
            continue
        identifier = entry.name
        index = _extract_index(identifier)
        sysfs = Path("/sys/class/video4linux") / identifier
        label = _safe_read_text(sysfs / "name") or identifier
        kind = "usb"
        lower_label = label.lower()
        if identifier and f"/dev/{identifier}" in ctl_metadata:
            label = ctl_metadata[f"/dev/{identifier}"]["label"] or label
            kind = ctl_metadata[f"/dev/{identifier}"].get("kind", kind)
        elif "unicam" in lower_label or "csi" in lower_label or "imx" in lower_label:
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

    for path, meta in ctl_metadata.items():
        if any(candidate.path == path for candidate in devices):
            continue
        index = _extract_index(path)
        candidate = CameraCandidate(
            identifier=f"v4l2:{Path(path).name}",
            label=meta.get("label", path),
            kind=meta.get("kind", "usb"),
            backend_hint="v4l2",
            path=path,
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


def _select_camera_list_command() -> Optional[List[str]]:
    """Return the preferred command to enumerate CSI sensors."""

    for executable in ("rpicam-hello", "libcamera-hello"):
        if shutil.which(executable):
            return [executable, "--list-cameras"]
    return None


def _list_libcamera_devices() -> List[CameraCandidate]:
    command = _select_camera_list_command()
    if command is None:
        return []

    try:
        result = _run_command(command, timeout=4.0)
    except (OSError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        LOGGER.warning("%s devolvió código %s", command[0], result.returncode)
        return []

    parsed = _parse_libcamera_output(result.stdout)
    return parsed


def _fallback_candidates() -> List[CameraCandidate]:
    fallbacks: List[CameraCandidate] = []
    if IS_WINDOWS:
        for index in range(3):
            fallbacks.append(
                CameraCandidate(
                    identifier=f"fallback:win{index}",
                    label=f"DirectShow index {index}",
                    kind="usb",
                    backend_hint="directshow",
                    index=index,
                    metadata={"fallback": True, "platform": "windows"},
                )
            )
    else:
        for index in range(2):
            path = f"/dev/video{index}"
            if Path(path).exists():
                fallbacks.append(
                    CameraCandidate(
                        identifier=f"fallback:{path}",
                        label=path,
                        kind="usb",
                        backend_hint="v4l2",
                        path=path,
                        index=index,
                        metadata={"fallback": True},
                    )
                )
    if Path("/usr/bin/rpicam-hello").exists() or Path("/usr/bin/libcamera-hello").exists():
        fallbacks.append(
            CameraCandidate(
                identifier="fallback:libcamerasrc",
                label="libcamerasrc auto",
                kind="csi",
                backend_hint="gstreamer",
                metadata={"fallback": True},
            )
        )
    return fallbacks


def list_sources() -> Dict[str, List[Dict[str, Any]]]:
    """Return discovered V4L2 and libcamera sources for diagnostic UIs."""

    v4l2 = [candidate.describe() for candidate in _list_v4l2_devices()]
    libcamera = [candidate.describe() for candidate in _list_libcamera_devices()]
    fallback = _fallback_candidates()
    for candidate in fallback:
        description = candidate.describe()
        if candidate.backend_hint in {"v4l2", "directshow"}:
            if not any(entry.get("path") == description.get("path") for entry in v4l2):
                v4l2.append(description)
        else:
            if not any(entry.get("id") == description.get("id") for entry in libcamera):
                libcamera.append(description)
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


def _apply_pixel_format(cap: Any, pixel_format: Optional[str]) -> bool:
    if cv2 is None or not pixel_format:
        return False
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if not callable(fourcc_fn):
        return False
    try:
        code = fourcc_fn(*pixel_format)
    except Exception:
        return False
    with contextlib.suppress(Exception):
        return bool(cap.set(cv2.CAP_PROP_FOURCC, code))
    return False


def _configure_capture(cap: Any, mode: CameraMode, pixel_format: Optional[str] = None) -> None:
    if pixel_format:
        _apply_pixel_format(cap, pixel_format)
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


def _probe_with_opencv(
    candidate: CameraCandidate,
    mode: CameraMode,
    *,
    backend_name: str,
    backend_flag: Optional[int] = None,
    pixel_formats: Optional[Sequence[Optional[str]]] = None,
) -> ProbeResult:
    if cv2 is None:
        return ProbeResult(candidate=candidate, backend=backend_name, mode=mode, success=False, reason="opencv-missing")

    target: Union[str, int, None] = candidate.path if candidate.path else candidate.index
    if target is None:
        return ProbeResult(candidate=candidate, backend=backend_name, mode=mode, success=False, reason="no-target")

    resolved_flag: Optional[int]
    if backend_flag is not None:
        resolved_flag = int(backend_flag)
    else:
        resolved_flag = resolve_backend_flag(backend_name)

    attempts: List[ProbeResult] = []
    pixel_formats = tuple(pixel_formats) if pixel_formats is not None else tuple(PIXEL_FORMAT_PREFERENCE) + (None,)

    for pixel_format in pixel_formats:
        try:
            if resolved_flag is not None:
                cap = cv2.VideoCapture(target, resolved_flag)
            else:
                cap = cv2.VideoCapture(target)
        except Exception as error:  # pragma: no cover - depends on runtime
            attempts.append(
                ProbeResult(
                    candidate=candidate,
                    backend=backend_name,
                    mode=mode,
                    success=False,
                    reason=str(error),
                    pixel_format=pixel_format,
                )
            )
            continue

        if not cap or not cap.isOpened():
            if cap:
                with contextlib.suppress(Exception):
                    cap.release()
            attempts.append(
                ProbeResult(
                    candidate=candidate,
                    backend=backend_name,
                    mode=mode,
                    success=False,
                    reason=f"open-failed:{backend_name}",
                    pixel_format=pixel_format,
                )
            )
            continue

        _configure_capture(cap, mode, pixel_format)
        success, latency, resolution, fps, frames = _read_frames(cap)

        with contextlib.suppress(Exception):
            cap.release()

        orientation = None
        if resolution[0] and resolution[1]:
            orientation = "portrait" if resolution[1] > resolution[0] else "landscape"

        result = ProbeResult(
            candidate=candidate,
            backend=backend_name,
            mode=mode,
            success=success,
            reason=None if success else f"no-valid-frame:{pixel_format or 'default'}",
            latency_ms=latency,
            resolution=resolution,
            fps=fps,
            orientation=orientation,
            frames_sampled=frames,
            pixel_format=pixel_format if success else pixel_format,
        )
        attempts.append(result)
        if success:
            return result

    if attempts:
        return attempts[-1]

    return ProbeResult(candidate=candidate, backend=backend_name, mode=mode, success=False, reason="probe-failed")


def _probe_with_v4l2(candidate: CameraCandidate, mode: CameraMode) -> ProbeResult:
    return _probe_with_opencv(candidate, mode, backend_name="v4l2")


def _probe_with_directshow(candidate: CameraCandidate, mode: CameraMode) -> ProbeResult:
    return _probe_with_opencv(
        candidate,
        mode,
        backend_name="directshow",
        backend_flag=resolve_backend_flag("directshow"),
        pixel_formats=(None,),
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
        pixel_format="BGR" if success else None,
    )


def _probe_candidate(candidate: CameraCandidate, forced_backend: Optional[str] = None) -> Optional[ProbeResult]:
    attempts: List[ProbeResult] = []

    sequence: List[Optional[str]] = [forced_backend, candidate.backend_hint]
    sequence.extend(preferred_backend_order(None))

    ordered: List[str] = []
    seen: set[str] = set()
    for entry in sequence:
        normalized = normalize_backend_name(entry)
        if not normalized:
            continue
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)

    if not ordered:
        ordered = list(preferred_backend_order(None))

    for mode in PREFERRED_MODES:
        for backend in ordered:
            if backend == "gstreamer":
                result = _probe_with_gstreamer(candidate, mode)
            elif backend == "directshow":
                result = _probe_with_directshow(candidate, mode)
            else:
                result = _probe_with_v4l2(candidate, mode)
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


def _annotate_selection(selection: CameraSelection, *, origin: str, verified: bool) -> CameraSelection:
    annotated = dataclasses.replace(selection)
    setattr(annotated, "_origin", origin)
    setattr(annotated, "_verified", verified)
    return annotated


def _validate_cached_selection(
    selection: CameraSelection,
    *,
    logger: Optional[logging.Logger] = None,
) -> Optional[CameraSelection]:
    if cv2 is None:
        return None

    candidate = dataclasses.replace(selection)

    width = candidate.width or 640
    height = candidate.height or 480
    fps = int(round(candidate.fps or 30)) if candidate.fps else 30
    mode = CameraMode(width=width, height=height, fps=max(fps, 1))

    cap = None
    try:
        if candidate.backend == "gstreamer" and candidate.pipeline:
            cap = cv2.VideoCapture(candidate.pipeline, cv2.CAP_GSTREAMER)
        else:
            target: Union[str, int, None]
            if candidate.device:
                target = candidate.device
            else:
                target = candidate.index
            if target is None:
                return None
            backend_flag = DEFAULT_CAPTURE_FLAG
            cap = cv2.VideoCapture(target, backend_flag) if backend_flag else cv2.VideoCapture(target)
    except Exception as error:  # pragma: no cover - depende del entorno
        if logger:
            logger.debug("Validación de cámara cacheada falló (%s)", error)
        return None

    if not cap or not cap.isOpened():
        if cap:
            with contextlib.suppress(Exception):
                cap.release()
        if logger:
            logger.debug("La cámara cacheada no se pudo abrir correctamente.")
        return None

    pixel_format = getattr(candidate, "pixel_format", None)
    _configure_capture(cap, mode, pixel_format)
    success, latency, resolution, fps_value, frames = _read_frames(cap, timeout=2.5)
    with contextlib.suppress(Exception):
        cap.release()

    if not success:
        if logger:
            logger.debug("La cámara cacheada no entregó frames válidos (frames=%s)", frames)
        return None

    candidate.latency_ms = latency
    if resolution[0] and resolution[1]:
        candidate.width = resolution[0]
        candidate.height = resolution[1]
    if fps_value:
        candidate.fps = fps_value
    candidate.orientation = "portrait" if candidate.height > candidate.width else "landscape"
    candidate.mode_name = f"{candidate.width}x{candidate.height}@{int(round(candidate.fps or mode.fps))}"
    candidate.probed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    return _annotate_selection(candidate, origin="cache", verified=True)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_camera_selection(
    *,
    force: bool = False,
    preferred: Optional[Union[str, int]] = None,
    logger: Optional[logging.Logger] = None,
    forced_backend: Optional[str] = None,
) -> Optional[CameraSelection]:
    """Return the cached camera selection or probe the available devices."""

    if cv2 is None:
        _log(logger, "warning", "OpenCV no está disponible; se omite la auto-detección de cámara.")
        return None

    current_signature = _hardware_signature()
    cached = _load_cached_selection()
    if not force and cached and cached.hardware_signature == current_signature:
        validated = _validate_cached_selection(cached, logger=logger)
        if validated:
            _log(
                logger,
                "info",
                "Selección de cámara reutilizada: backend=%s device=%s pipeline=%s",
                validated.backend,
                validated.device or validated.index,
                validated.pipeline,
            )
            return validated
        _log(logger, "warning", "La cámara cacheada no respondió, se reprobará.")

    candidates = _list_libcamera_devices() + _list_v4l2_devices()
    seen_identifiers = {candidate.identifier for candidate in candidates}
    for fallback in _fallback_candidates():
        if fallback.identifier not in seen_identifiers:
            candidates.append(fallback)
            seen_identifiers.add(fallback.identifier)

    if not candidates:
        _log(logger, "error", CAMERA_NOT_FOUND_MESSAGE)
        return None

    preferred_str: Optional[str] = None
    if preferred is not None:
        preferred_str = str(preferred)

    best_result: Optional[ProbeResult] = None
    for candidate in candidates:
        _log(logger, "info", "Probing %s (%s)", candidate.identifier, candidate.label)
        result = _probe_candidate(candidate, forced_backend)
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
        if best_result:
            reason = best_result.reason if best_result else "unknown"
            _log(logger, "warning", "No se pudo validar ninguna cámara física (%s)", reason or "unknown")
        _log(logger, "error", CAMERA_NOT_FOUND_MESSAGE)
        if IS_WINDOWS:
            _log(
                logger,
                "info",
                "Sugerencias: prueba con --camera-index 0/1/2, cambia el backend con --camera-backend directshow|v4l2 o reduce la resolución con --camera-width 960 --camera-height 720.",
            )
        return None

    mode = best_result.mode
    resolved_width = best_result.resolution[0] or mode.width
    resolved_height = best_result.resolution[1] or mode.height
    resolved_fps = best_result.fps or float(mode.fps)
    orientation = best_result.orientation
    if not orientation and resolved_width and resolved_height:
        orientation = "portrait" if resolved_height > resolved_width else "landscape"

    selection = CameraSelection(
        backend=best_result.backend,
        device=best_result.candidate.path,
        index=best_result.candidate.index,
        pipeline=best_result.pipeline,
        width=resolved_width,
        height=resolved_height,
        fps=resolved_fps,
        latency_ms=best_result.latency_ms,
        orientation=orientation,
        kind=best_result.candidate.kind,
        mode_name=f"{resolved_width}x{resolved_height}@{int(round(resolved_fps or mode.fps))}",
        hardware_signature=current_signature,
        probed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        pixel_format=best_result.pixel_format,
    )

    selection = _annotate_selection(selection, origin="probe", verified=True)
    _save_selection(selection)
    _log(
        logger,
        "info",
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
    attempts: List[ProbeResult] = []
    for backend in preferred_backend_order("directshow" if IS_WINDOWS else "v4l2"):
        if backend == "gstreamer":
            result = _probe_with_gstreamer(candidate, mode)
        elif backend == "directshow":
            result = _probe_with_directshow(candidate, mode)
        else:
            result = _probe_with_v4l2(candidate, mode)
        attempts.append(result)
        if result.success:
            return result
    return attempts[-1] if attempts else ProbeResult(
        candidate=candidate,
        backend="directshow" if IS_WINDOWS else "v4l2",
        mode=mode,
        success=False,
        reason="probe-failed",
    )


def parse_resolution(value: str) -> Tuple[int, int]:
    match = re.match(r"^(\d+)[xX](\d+)$", value.strip())
    if not match:
        raise ValueError(f"Resolución inválida: {value}")
    return int(match.group(1)), int(match.group(2))


def get_cached_selection() -> Optional[CameraSelection]:
    cached = _load_cached_selection()
    if not cached:
        return None
    return _annotate_selection(cached, origin="cache", verified=False)


__all__ = [
    "CameraSelection",
    "list_sources",
    "ensure_camera_selection",
    "probe_specific_device",
    "parse_resolution",
    "get_cached_selection",
    "CAMERA_NOT_FOUND_MESSAGE",
    "normalize_backend_name",
    "resolve_backend_flag",
    "preferred_backend_order",
]
