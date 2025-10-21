"""Self-contained HELEN backend with SSE transport and synthetic inference."""

from __future__ import annotations

import json
import logging
import socketserver
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
from typing import Deque, Dict, List, Optional

from Hellen_model_RN.simple_classifier import (
    Prediction,
    SimpleGestureClassifier,
    SyntheticGestureStream,
)


LOGGER = logging.getLogger("helen.backend")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = REPO_ROOT / "helen"
MODEL_DIR = REPO_ROOT / "Hellen_model_RN"
DATASET_PATH = MODEL_DIR / "data1.pickle"

ACTIVATION_ALIASES = {
    "start",
    "activar",
    "heyhelen",
    "holahelen",
    "oyehelen",
    "wake",
    "foco",
}


def _iso_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


@dataclass
class HealthSnapshot:
    status: str
    model_loaded: bool
    session_id: str
    pipeline_running: bool
    source: str
    clients: int
    uptime_s: float
    last_prediction: Optional[Dict[str, object]]
    avg_latency_ms: float
    last_error: Optional[str] = None


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

    def broadcast(self, payload: Dict[str, object]) -> None:
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
            try:
                self._handler.wfile.flush()
            except Exception:  # pragma: no cover - best effort
                pass


class GesturePipeline:
    """Background thread that feeds predictions to the runtime."""

    def __init__(self, runtime: "HelenRuntime", interval_s: float = 0.9) -> None:
        self._runtime = runtime
        self._interval = interval_s
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._sequence = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        LOGGER.info("Gesture pipeline started")
        while self._running.is_set():
            features, source_label = self._runtime.stream.next()
            start = time.perf_counter()
            prediction: Prediction = self._runtime.classifier.predict(features)
            latency_ms = (time.perf_counter() - start) * 1000.0
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
            self._runtime.push_prediction(event)
            self._sequence += 1
            time.sleep(self._interval)

        LOGGER.info("Gesture pipeline stopped")


class HelenRuntime:
    """Holds application state shared across HTTP handlers."""

    def __init__(self) -> None:
        self.session_id = uuid.uuid4().hex
        self.started_at = time.time()
        self.event_stream = EventStream()
        self.classifier = SimpleGestureClassifier(DATASET_PATH)
        self.stream = SyntheticGestureStream(DATASET_PATH)
        self.pipeline = GesturePipeline(self)
        self.lock = threading.Lock()
        self.latency_history: Deque[float] = deque(maxlen=120)
        self.last_prediction: Optional[Dict[str, object]] = None
        self.last_error: Optional[str] = None
        self.last_heartbeat = 0.0

    # ------------------------------------------------------------------
    def start(self) -> None:
        self.pipeline.start()

    def stop(self) -> None:
        self.pipeline.stop()

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
        payload: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        collapsed = label.strip().lower()
        is_activation = collapsed in ACTIVATION_ALIASES

        base_event: Dict[str, object] = {
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
    def push_prediction(self, event: Dict[str, object]) -> None:
        with self.lock:
            self.last_prediction = event
            self.last_heartbeat = time.time()
            self.latency_history.append(float(event.get("latency_ms", 0.0)))

        LOGGER.debug("Broadcasting event: %s", event)
        self.event_stream.broadcast(event)

    # ------------------------------------------------------------------
    def receive_external_payload(self, payload: Dict[str, object]) -> Dict[str, object]:
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
            avg_latency = (
                sum(self.latency_history) / len(self.latency_history)
                if self.latency_history
                else 0.0
            )
            last_error = self.last_error
            heartbeat_age = time.time() - self.last_heartbeat if self.last_heartbeat else None

        pipeline_running = self.pipeline._thread is not None and self.pipeline._thread.is_alive()
        healthy = pipeline_running and (heartbeat_age is None or heartbeat_age < 5.0)

        status = "HEALTHY" if healthy else "DEGRADED"
        return HealthSnapshot(
            status=status,
            model_loaded=True,
            session_id=self.session_id,
            pipeline_running=pipeline_running,
            source="synthetic",
            clients=self.event_stream.client_count(),
            uptime_s=time.time() - self.started_at,
            last_prediction=last_prediction,
            avg_latency_ms=round(avg_latency, 3),
            last_error=last_error,
        )


class HelenRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving the SPA and the SSE endpoints."""

    server_version = "HelenHTTP/1.0"
    runtime: HelenRuntime  # populated at server construction time

    def __init__(self, *args, runtime: HelenRuntime, **kwargs) -> None:
        self.runtime = runtime
        super().__init__(*args, directory=str(FRONTEND_ROOT), **kwargs)

    # ------------------------------------------------------------------
    def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover - forwarded to logging
        LOGGER.info("HTTP %s - %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        if self.path in {"", "/"}:
            self.path = "/index.html"

        if self.path == "/healthz":
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
    def do_POST(self) -> None:
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

        # Send a warm-up event so the frontend can establish latency baseline.
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
                # Keep the HTTP handler alive until the client disconnects.
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - network race
            pass
        finally:
            self.runtime.event_stream.unregister(client_id)


def run(host: str = "0.0.0.0", port: int = 5000) -> None:
    runtime = HelenRuntime()
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


__all__ = ["HelenRuntime", "HelenRequestHandler", "run"]


if __name__ == "__main__":  # pragma: no cover
    run()

