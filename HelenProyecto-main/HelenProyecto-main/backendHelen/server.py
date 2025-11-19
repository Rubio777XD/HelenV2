"""HELEN backend powered by the TensorFlow video gesture model."""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import platform
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:  # pragma: no cover - optional dependency in CI
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency in CI
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None  # type: ignore

from Hellen_model_TF.frontend_bridge import config as bridge_config
from Hellen_model_TF.frontend_bridge.service import GestureInferenceService

from . import camera_probe

LOGGER = logging.getLogger("helen.backend")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

with contextlib.suppress(Exception):
    import absl.logging as absl_logging  # type: ignore

    absl_logging.set_verbosity(absl_logging.WARNING)
    handler = absl_logging.get_absl_handler()
    handler.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Paths and runtime metadata
# ---------------------------------------------------------------------------
def _resolve_repo_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
FRONTEND_ROOT = REPO_ROOT / "helen"
MODE_STORAGE_PATH = REPO_ROOT / "reports" / "display-mode.json"


def _read_os_release() -> str:
    path = Path("/etc/os-release")
    if not path.exists():
        return ""
    content: Dict[str, str] = {}
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
        "notes": [],
    }

    if mp is None:
        snapshot["mediapipe"] = {
            "status": "error",
            "message": "ImportError",
            "suggestion": "Instala mediapipe==0.10.18 dentro del entorno .venv.",
        }
    else:
        snapshot["mediapipe"] = {"status": "ok", "version": getattr(mp, "__version__", "unknown")}
        with contextlib.suppress(Exception):
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

    camera_probe.LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = camera_probe.LOG_DIR / f"vision-runtime-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with contextlib.suppress(OSError):
        path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    return snapshot


VISION_RUNTIME_SNAPSHOT = _log_vision_runtime_snapshot()


# ---------------------------------------------------------------------------
# Display mode helpers
# ---------------------------------------------------------------------------
DEFAULT_DISPLAY_MODE = "windows"
DISPLAY_MODE_OPTIONS = {"windows", "raspberry", "minimal"}


def _normalize_display_mode(value: Optional[str]) -> str:
    if not value:
        return DEFAULT_DISPLAY_MODE
    normalized = str(value).strip().lower()
    if normalized not in DISPLAY_MODE_OPTIONS:
        return DEFAULT_DISPLAY_MODE
    return normalized


class DisplayModeStore:
    def __init__(self, path: Path, default_mode: str) -> None:
        self._path = path
        self._default_mode = _normalize_display_mode(default_mode)
        self._lock = threading.Lock()

    def cached(self) -> str:
        with self._lock:
            if not self._path.exists():
                return self._default_mode
            with contextlib.suppress(Exception):
                data = json.loads(self._path.read_text(encoding="utf-8"))
                return _normalize_display_mode(data.get("mode"))
            return self._default_mode

    def save(self, mode: str) -> str:
        normalized = _normalize_display_mode(mode)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mode": normalized, "updated_at": datetime.utcnow().isoformat() + "Z"}
        with self._lock:
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return normalized


DISPLAY_MODE_STORE = DisplayModeStore(MODE_STORAGE_PATH, DEFAULT_DISPLAY_MODE)


# ---------------------------------------------------------------------------
# Wi-Fi helpers (copied from the legacy backend for feature parity)
# ---------------------------------------------------------------------------
PLATFORM = platform.system().lower()
IS_WINDOWS = PLATFORM.startswith("win")
IS_LINUX = PLATFORM.startswith("linux")


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


def check_online_status() -> Dict[str, Any]:
    payload = {"online": False, "latency_ms": None, "iface": None}
    target = "8.8.8.8"
    start = time.perf_counter()
    with contextlib.suppress(OSError):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            sock.connect((target, 53))
            payload["online"] = True
            payload["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
        finally:
            sock.close()
    return payload


def current_wifi_status() -> Dict[str, Any]:
    if IS_WINDOWS:
        args = ["netsh", "wlan", "show", "interfaces"]
        result = _run_command(args, timeout=5.0)
        if result.returncode != 0:
            return {"connected": False, "error": result.stderr.strip()}
        connected_ssid = ""
        iface = ""
        for line in result.stdout.splitlines():
            if "Name" in line and not iface:
                iface = line.split(":", 1)[-1].strip()
            if "SSID" in line and "BSSID" not in line:
                connected_ssid = line.split(":", 1)[-1].strip()
        return {"connected": bool(connected_ssid), "connected_ssid": connected_ssid, "iface": iface}

    if _command_exists("nmcli"):
        result = _run_command(["nmcli", "-t", "-f", "DEVICE,STATE,CONNECTION", "device"], timeout=5.0)
        if result.returncode != 0:
            return {"connected": False, "error": result.stderr.strip()}
        for line in result.stdout.splitlines():
            parts = line.split(":")
            if len(parts) < 3:
                continue
            device, state, connection = parts[:3]
            if state == "connected" and connection:
                return {"connected": True, "connected_ssid": connection, "iface": device}
        return {"connected": False}

    return {"connected": False, "error": "Sin soporte nmcli"}


def scan_wifi_networks() -> List[Dict[str, Any]]:
    networks: List[Dict[str, Any]] = []
    if IS_WINDOWS:
        args = ["netsh", "wlan", "show", "networks", "mode=Bssid"]
        result = _run_command(args, timeout=10.0)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "No se pudo escanear redes Wi-Fi")
        current: Dict[str, Any] = {}
        for line in result.stdout.splitlines():
            if line.strip().startswith("SSID"):
                if current:
                    networks.append(current)
                    current = {}
                current["ssid"] = line.split(":", 1)[-1].strip()
            if line.strip().startswith("Signal"):
                value = line.split(":", 1)[-1].strip().replace("%", "")
                current["signal"] = int(value) if value.isdigit() else None
        if current:
            networks.append(current)
        return networks

    if _command_exists("nmcli"):
        args = ["nmcli", "-t", "-f", "SSID,SIGNAL", "device", "wifi", "list"]
        result = _run_command(args, timeout=10.0)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "No se pudo escanear redes Wi-Fi")
        for line in result.stdout.splitlines():
            ssid, *rest = line.split(":")
            signal = rest[0] if rest else ""
            networks.append({"ssid": ssid or "<oculto>", "signal": int(signal) if signal.isdigit() else None})
        return networks

    raise RuntimeError("Escaneo Wi-Fi no soportado en esta plataforma")


def _build_windows_profile(ssid: str, password: str) -> str:
    security = "WPA2PSK"
    auth_block = f"<authEncryption>\n            <authentication>WPA2PSK</authentication>\n            <encryption>AES</encryption>\n            <useOneX>false</useOneX>\n        </authEncryption>"
    key_block = ""
    if password:
        key_block = (
            "            <sharedKey>\n"
            "                <keyType>passPhrase</keyType>\n"
            f"                <protected>false</protected>\n                <keyMaterial>{password}</keyMaterial>\n"
            "            </sharedKey>"
        )
    profile = f"""<?xml version=\"1.0\"?>
<WLANProfile xmlns=\"http://www.microsoft.com/networking/WLAN/profile/v1\">
    <name>{ssid}</name>
    <SSIDConfig>
        <SSID>
            <name>{ssid}</name>
        </SSID>
    </SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM>
        <security>
{auth_block}
{key_block}
        </security>
    </MSM>
</WLANProfile>
"""
    return profile


def _connect_wifi_windows(ssid: str, password: str) -> Tuple[bool, str]:
    if not _command_exists("netsh"):
        return False, "netsh no disponible"
    profile_xml = _build_windows_profile(ssid, password)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as handle:
        handle.write(profile_xml)
        profile_path = handle.name
    try:
        _run_command(["netsh", "wlan", "delete", "profile", f"name={ssid}"], timeout=6.0)
        added = _run_command(["netsh", "wlan", "add", "profile", f"filename={profile_path}", "user=all"], timeout=10.0)
        if added.returncode != 0:
            return False, added.stderr.strip() or "No se pudo registrar el perfil"
        connected = _run_command(["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}"], timeout=15.0)
        if connected.returncode != 0:
            return False, connected.stderr.strip() or "No se pudo iniciar la conexión"
        return True, ""
    finally:
        with contextlib.suppress(OSError):
            Path(profile_path).unlink(missing_ok=True)


def _connect_wifi_nmcli(ssid: str, password: str) -> Tuple[bool, str]:
    if not _command_exists("nmcli"):
        return False, "nmcli no disponible"
    command = ["nmcli", "dev", "wifi", "connect", ssid]
    if password:
        command.extend(["password", password])
    result = _run_command(command, timeout=25.0)
    if result.returncode != 0:
        return False, result.stderr.strip() or "No se pudo establecer la conexión"
    return True, ""


def connect_wifi(ssid: str, password: str) -> Tuple[bool, str]:
    if not ssid:
        raise RuntimeError("SSID requerido")
    if IS_WINDOWS:
        return _connect_wifi_windows(ssid, password)
    if IS_LINUX:
        return _connect_wifi_nmcli(ssid, password)
    raise RuntimeError("Plataforma sin soporte Wi-Fi")


# ---------------------------------------------------------------------------
# Runtime data classes
# ---------------------------------------------------------------------------
ACTIVATION_ALIASES = {"start", "activar", "heyhelen", "holahelen", "oyehelen", "wake"}
HEALTH_ENDPOINTS = {"/health", "/healthz"}


def _iso_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


@dataclass
class RuntimeConfig:
    model_path: Optional[Path] = None
    labels_path: Optional[Path] = None
    camera_index: int = bridge_config.DEFAULT_CAMERA_INDEX
    confidence_threshold: float = bridge_config.DEFAULT_CONFIDENCE_THRESHOLD
    sequence_length: Optional[int] = bridge_config.DEFAULT_SEQUENCE_LENGTH
    prediction_cooldown_s: float = 0.5
    display_mode: str = DEFAULT_DISPLAY_MODE
    poll_interval_s: float = 0.08
    process_every_n: int = 1


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
    camera_device: Optional[str]
    camera_backend: Optional[str]
    camera_resolution: Optional[str]
    camera_fps: Optional[float]
    camera_pixel_format: Optional[str]
    camera_probe_latency_ms: Optional[float]
    camera_last_capture: Optional[str]
    camera_last_error: Optional[str]
    last_error: Optional[str] = None


class GestureLabelMapper:
    def __init__(self, aliases: Optional[Dict[str, str]] = None) -> None:
        self._aliases = {k.lower(): v for k, v in (aliases or {}).items()}

    def normalize(self, label: str) -> str:
        if not label:
            return ""
        text = str(label).strip()
        canonical = self._aliases.get(text.lower())
        return canonical or text


class TensorFlowGesturePipeline:
    def __init__(
        self,
        runtime: "HelenRuntime",
        *,
        service_factory: Any = GestureInferenceService,
        model_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        camera_index: int = bridge_config.DEFAULT_CAMERA_INDEX,
        confidence_threshold: float = bridge_config.DEFAULT_CONFIDENCE_THRESHOLD,
        sequence_length: Optional[int] = None,
        prediction_cooldown_s: float = 0.5,
    ) -> None:
        self._runtime = runtime
        self._service_factory = service_factory
        self._service_kwargs = {
            "model_path": model_path,
            "labels_path": labels_path,
            "camera_index": camera_index,
            "confidence_threshold": confidence_threshold,
            "sequence_length": sequence_length,
            "prediction_cooldown_s": prediction_cooldown_s,
        }
        self._service: Optional[GestureInferenceService] = None
        self._thread: Optional[threading.Thread] = None
        self._queue: Optional[Queue] = None
        self._running = threading.Event()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self._service is None:
            self._service = self._service_factory(**self._service_kwargs)
        self._service.start()
        self._queue = self._service.subscribe()
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="tf-gesture-pipeline", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        if self._queue and self._service:
            self._service.unsubscribe(self._queue)
        self._queue = None
        if self._service:
            self._service.stop()

    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        service = self._service
        if service is None:
            return {"running": False, "gestures": []}
        return service.snapshot()

    # ------------------------------------------------------------------
    def _run(self) -> None:
        if not self._queue or not self._service:
            return
        queue = self._queue
        while self._running.is_set():
            try:
                payload = queue.get(timeout=1.0)
            except Empty:
                continue
            self._runtime.handle_prediction(payload)


class EventStream:
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


# ---------------------------------------------------------------------------
# Helen runtime
# ---------------------------------------------------------------------------
class HelenRuntime:
    def __init__(
        self,
        config: Optional[RuntimeConfig] = None,
        *,
        label_aliases: Optional[Dict[str, str]] = None,
        service_factory: Any = GestureInferenceService,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.label_mapper = GestureLabelMapper(label_aliases)
        self.session_id = uuid.uuid4().hex
        self.started_at = time.time()
        self.event_stream = EventStream()
        self.metrics = deque(maxlen=256)
        self.latency_history: deque[float] = deque(maxlen=240)
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.last_prediction_at: Optional[float] = None
        self.last_error: Optional[str] = None
        self.stream_source = "tensorflow"
        self.model_loaded = False
        self.model_source = "tf_savedmodel"
        self.lock = threading.Lock()
        self.sequence = 0
        self.vision_snapshot = VISION_RUNTIME_SNAPSHOT

        self.pipeline = TensorFlowGesturePipeline(
            self,
            service_factory=service_factory,
            model_path=self.config.model_path,
            labels_path=self.config.labels_path,
            camera_index=self.config.camera_index,
            confidence_threshold=self.config.confidence_threshold,
            sequence_length=self.config.sequence_length,
            prediction_cooldown_s=self.config.prediction_cooldown_s,
        )

    # ------------------------------------------------------------------
    def start(self) -> None:
        try:
            self.pipeline.start()
            self.model_loaded = True
        except Exception as exc:  # pragma: no cover - hardware dependent
            LOGGER.exception("No se pudo iniciar la tubería de gestos: %s", exc)
            with self.lock:
                self.last_error = str(exc)
            self.model_loaded = False
            raise

    # ------------------------------------------------------------------
    def stop(self, *, export_report: bool = False) -> None:
        self.pipeline.stop()

    # ------------------------------------------------------------------
    def handle_prediction(self, payload: Dict[str, Any]) -> None:
        label = self.label_mapper.normalize(str(payload.get("label", "")))
        confidence = float(payload.get("confidence", 0.0))
        timestamp = float(payload.get("timestamp", time.time()))
        latency_ms = max(0.0, (time.time() - timestamp) * 1000.0)
        event = self.build_event(
            label=label,
            score=confidence,
            latency_ms=latency_ms,
            timestamp=timestamp,
            sequence=self.sequence,
            origin="tf",
            payload={"raw": payload},
        )
        self.sequence += 1
        self.push_prediction(event)

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
        with self.lock:
            self.last_prediction = event
            self.last_prediction_at = time.time()
            self.latency_history.append(float(event.get("latency_ms", 0.0)))
            self.last_error = None
        self.event_stream.broadcast(event)

    # ------------------------------------------------------------------
    def receive_external_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = time.time()
        sequence = int(payload.get("sequence", self.sequence))
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
    def mode_snapshot(self, persisted_mode: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            active = self.config.display_mode
            poll_interval = float(self.config.poll_interval_s)
            stride = int(self.config.process_every_n)
        snapshot: Dict[str, Any] = {
            "active": active,
            "persisted": _normalize_display_mode(persisted_mode or DISPLAY_MODE_STORE.cached()),
            "poll_interval_s": poll_interval,
            "process_every_n": stride,
            "stream_source": self.stream_source,
            "camera_profile": None,
        }
        return snapshot

    # ------------------------------------------------------------------
    def apply_display_mode(self, mode: str) -> Dict[str, Any]:
        normalized = _normalize_display_mode(mode)
        persisted = DISPLAY_MODE_STORE.save(normalized)
        with self.lock:
            self.config.display_mode = normalized
        return self.mode_snapshot(persisted_mode=persisted)

    # ------------------------------------------------------------------
    def engine_status(self) -> Dict[str, Any]:
        pipeline_status = self.pipeline.snapshot()
        return {
            "mode": self.mode_snapshot(),
            "ui_mode": self.config.display_mode,
            "thresholds": {},
            "consensus": {},
            "pipeline": {
                "poll_interval_s": float(self.config.poll_interval_s),
                "process_every_n": int(self.config.process_every_n),
                "running": self.pipeline.is_running(),
            },
            "stream": pipeline_status,
            "vision": self.vision_snapshot,
        }

    # ------------------------------------------------------------------
    def _latency_snapshot(self) -> Dict[str, float]:
        with self.lock:
            samples = list(self.latency_history)
        if samples:
            average = sum(samples) / len(samples)
            sorted_samples = sorted(samples)
            index = min(len(sorted_samples) - 1, int(len(sorted_samples) * 0.95))
            p95 = sorted_samples[index]
            maximum = max(sorted_samples)
        else:
            average = p95 = maximum = 0.0
        return {"avg_ms": average, "p95_ms": p95, "max_ms": maximum, "count": len(samples)}

    # ------------------------------------------------------------------
    def health(self) -> HealthSnapshot:
        with self.lock:
            last_prediction = self.last_prediction
            last_prediction_at = self.last_prediction_at
            avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0.0
            last_error = self.last_error
        status = "HEALTHY"
        if last_error:
            status = "ERROR"
        elif not self.pipeline.is_running() or not self.model_loaded:
            status = "DEGRADED"
        return HealthSnapshot(
            status=status,
            model_loaded=self.model_loaded,
            model_source=self.model_source,
            session_id=self.session_id,
            pipeline_running=self.pipeline.is_running(),
            stream_source=self.stream_source,
            clients=self.event_stream.client_count(),
            uptime_s=time.time() - self.started_at,
            last_prediction=last_prediction,
            last_prediction_at=_iso_timestamp(last_prediction_at) if last_prediction_at else None,
            avg_latency_ms=round(avg_latency, 3),
            camera_ok=True,
            camera_index=self.config.camera_index,
            camera_device=str(self.config.camera_index),
            camera_backend="opencv",
            camera_resolution="",
            camera_fps=None,
            camera_pixel_format=None,
            camera_probe_latency_ms=None,
            camera_last_capture=None,
            camera_last_error=None,
            last_error=last_error,
        )


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class HelenRequestHandler(SimpleHTTPRequestHandler):
    server_version = "HelenHTTP/2.0"
    runtime: HelenRuntime

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

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover - forwarded to logging
        LOGGER.info("HTTP %s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: D401
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

    def do_POST(self) -> None:  # noqa: D401
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
            payload = {"connected": is_connected, "reason": reason or "", "status": status_info}
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
            snapshot = self.runtime.apply_display_mode(str(mode_value))
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
        except (BrokenPipeError, ConnectionResetError):  # pragma: no cover
            pass
        finally:
            self.runtime.event_stream.unregister(client_id)


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------
class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def run(host: str = "0.0.0.0", port: int = 5000, *, config: Optional[RuntimeConfig] = None) -> None:
    runtime = HelenRuntime(config=config)
    runtime.start()
    handler_factory = partial(HelenRequestHandler, runtime=runtime)
    with ThreadingHTTPServer((host, port), handler_factory) as httpd:
        LOGGER.info("HELEN backend (TF) serving from %s:%s", host, port)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:  # pragma: no cover
            LOGGER.info("Shutting down backend")
        finally:
            runtime.stop()


__all__ = ["HelenRuntime", "HelenRequestHandler", "RuntimeConfig", "run", "main", "ThreadingHTTPServer"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    return Path(value)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HELEN backend server (TensorFlow model)")
    parser.add_argument("--host", default="0.0.0.0", help="Dirección de enlace del servidor HTTP")
    parser.add_argument("--port", type=int, default=5000, help="Puerto del servidor HTTP")
    parser.add_argument("--model-path", type=_parse_path, default=None, help="Ruta al SavedModel/.keras del modelo de gestos")
    parser.add_argument("--labels", type=_parse_path, default=None, help="Ruta opcional a labels.json")
    parser.add_argument("--camera-index", type=int, default=bridge_config.DEFAULT_CAMERA_INDEX, help="Índice de cámara OpenCV")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=bridge_config.DEFAULT_CONFIDENCE_THRESHOLD,
        help="Probabilidad mínima para emitir gestos",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Número de frames acumulados antes de inferir (None usa config del modelo)",
    )
    parser.add_argument(
        "--prediction-cooldown",
        type=float,
        default=0.5,
        help="Ventana mínima entre predicciones iguales",
    )
    args = parser.parse_args(argv)
    config = RuntimeConfig(
        model_path=args.model_path,
        labels_path=args.labels,
        camera_index=args.camera_index,
        confidence_threshold=args.confidence_threshold,
        sequence_length=args.sequence_length,
        prediction_cooldown_s=args.prediction_cooldown,
    )
    run(args.host, args.port, config=config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
