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
import socketserver
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

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


def _resolve_repo_root() -> Path:
    """Return the runtime root, compatible with PyInstaller bundles."""

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
FRONTEND_ROOT = REPO_ROOT / "helen"
MODEL_DIR = REPO_ROOT / "Hellen_model_RN"
MODEL_PATH = MODEL_DIR / "model.p"
DATASET_PATH = MODEL_DIR / "data1.pickle"

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


@dataclass
class RuntimeConfig:
    camera_index: int = 0
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.6
    poll_interval_s: float = 0.12
    enable_camera: bool = True
    fallback_to_synthetic: bool = True
    model_path: Path = MODEL_PATH
    dataset_path: Path = DATASET_PATH


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
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV no está instalado. Ejecuta `pip install opencv-python`.")
        if mp is None:
            raise RuntimeError("MediaPipe no está instalado. Ejecuta `pip install mediapipe`.")

        self._camera_index = camera_index
        self._detection_confidence = detection_confidence
        self._tracking_confidence = tracking_confidence

        self._cap: Optional[Any] = None
        self._hands: Optional[Any] = None
        self._opened = False

        self._last_capture: Optional[float] = None
        self._last_error: Optional[str] = None
        self._healthy = False
        self._frames_without_hand = 0

    # ------------------------------------------------------------------
    def open(self) -> None:
        if self._opened:
            return

        cap = cv2.VideoCapture(self._camera_index)
        if not cap or not cap.isOpened():
            self._last_error = f"No se pudo abrir la cámara {self._camera_index}"
            raise RuntimeError(self._last_error)

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
            if not ok:
                self._last_error = "No se pudo leer un frame de la cámara"
                self._healthy = False
                time.sleep(0.05)
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._hands.process(image)

            if not results.multi_hand_landmarks:
                self._frames_without_hand += 1
                time.sleep(0.02)
                continue

            self._frames_without_hand = 0
            landmarks = results.multi_hand_landmarks[0]
            features = self._extract_features(landmarks)
            self._last_capture = time.time()
            self._last_error = None
            self._healthy = True
            return features, None

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "healthy": self._healthy,
            "camera_index": self._camera_index,
            "last_capture": self._last_capture,
            "last_error": self._last_error,
            "frames_without_hand": self._frames_without_hand,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_features(landmarks: Any) -> List[float]:
        x_coords = [lm.x for lm in landmarks.landmark]
        y_coords = [lm.y for lm in landmarks.landmark]
        min_x = min(x_coords)
        min_y = min(y_coords)

        data_aux: List[float] = []
        for lm in landmarks.landmark:
            data_aux.append(lm.x - min_x)
            data_aux.append(lm.y - min_y)

        return data_aux


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

    def __init__(self, runtime: "HelenRuntime", interval_s: float = 0.12) -> None:
        self._runtime = runtime
        self._interval = max(0.01, float(interval_s))
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._sequence = 0

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

            try:
                start = time.perf_counter()
                prediction: Prediction = self._runtime.classifier.predict(features)
                latency_ms = (time.perf_counter() - start) * 1000.0
            except Exception as error:  # pragma: no cover - classifier failure
                self._runtime.report_error(f"classifier_error: {error}")
                time.sleep(0.5)
                continue

            timestamp = time.time()
            event = self._runtime.build_event(
                label=prediction.label,
                score=prediction.score,
                latency_ms=latency_ms,
                timestamp=timestamp,
                sequence=self._sequence,
                origin="pipeline",
                hint_label=source_label,
            )
            self._runtime.clear_error()
            self._runtime.push_prediction(event)
            self._sequence += 1
            time.sleep(self._interval)

        LOGGER.info("Gesture pipeline stopped")


class HelenRuntime:
    """Holds application state shared across HTTP handlers."""

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig()
        self.session_id = uuid.uuid4().hex
        self.started_at = time.time()
        self.event_stream = EventStream()

        classifier, classifier_meta = self._create_classifier()
        self.classifier = classifier
        self.model_source = classifier_meta["source"]
        self.model_loaded = classifier_meta["loaded"]

        stream, stream_meta = self._create_stream()
        self.stream = stream
        self.stream_source = stream_meta["source"]

        self.pipeline = GesturePipeline(self, interval_s=self.config.poll_interval_s)
        self.lock = threading.Lock()
        self.latency_history: Deque[float] = deque(maxlen=240)
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.last_prediction_at: Optional[float] = None
        self.last_heartbeat = 0.0
        self.last_error: Optional[str] = None

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
                raise RuntimeError("No hay dataset disponible para el clasificador de respaldo") from error
            fallback = SimpleGestureClassifier(dataset_path)
            return fallback, {"source": "synthetic", "loaded": False}

    # ------------------------------------------------------------------
    def _create_stream(self) -> Tuple[Any, Dict[str, Any]]:
        if self.config.enable_camera:
            try:
                stream = CameraGestureStream(
                    camera_index=self.config.camera_index,
                    detection_confidence=self.config.detection_confidence,
                    tracking_confidence=self.config.tracking_confidence,
                )
                LOGGER.info("Usando cámara física en el índice %s", self.config.camera_index)
                return stream, {"source": CameraGestureStream.source}
            except Exception as error:
                LOGGER.warning("No se pudo inicializar la cámara: %s", error)
                if not self.config.fallback_to_synthetic:
                    raise

        dataset_path = self.config.dataset_path
        if not dataset_path.exists():
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
            camera_index=stream_status.get("camera_index"),
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

    # ------------------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover - forwarded to logging
        LOGGER.info("HTTP %s - %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: D401 - inherited API
        if self.path in {"", "/"}:
            self.path = "/index.html"

        if self.path in HEALTH_ENDPOINTS:
            snapshot = self.runtime.health()
            payload = json.dumps(snapshot.__dict__).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path.startswith("/events"):
            self._handle_sse()
            return

        super().do_GET()

    # ------------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: D401 - inherited API
        if self.path == "/gestures/gesture-key":
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HELEN backend server")
    parser.add_argument("--host", default="0.0.0.0", help="Dirección de enlace del servidor HTTP")
    parser.add_argument("--port", type=int, default=5000, help="Puerto del servidor HTTP")
    parser.add_argument("--camera-index", type=int, default=0, help="Índice de la cámara de video a utilizar")
    parser.add_argument("--detection-confidence", type=float, default=0.7, help="Umbral de detección de MediaPipe")
    parser.add_argument("--tracking-confidence", type=float, default=0.6, help="Umbral de seguimiento de MediaPipe")
    parser.add_argument("--poll-interval", type=float, default=0.12, help="Intervalo entre inferencias en segundos")
    parser.add_argument("--no-camera", action="store_true", help="Desactiva el uso de cámara física")
    parser.add_argument(
        "--no-synthetic-fallback",
        action="store_true",
        help="Falla si la cámara no está disponible en lugar de usar el dataset sintético",
    )

    args = parser.parse_args(argv)
    config = RuntimeConfig(
        camera_index=args.camera_index,
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence,
        poll_interval_s=args.poll_interval,
        enable_camera=not args.no_camera,
        fallback_to_synthetic=not args.no_synthetic_fallback,
    )

    run(args.host, args.port, config=config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
