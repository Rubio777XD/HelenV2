"""Flask application exposing gesture predictions to the frontend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import Namespace, SocketIO

from . import config
from .service import GestureInferenceService


class GestureNamespace(Namespace):
    """Custom namespace to broadcast predictions to connected clients."""

    def __init__(self, namespace: str, service: GestureInferenceService, socketio: SocketIO) -> None:
        super().__init__(namespace)
        self._service = service
        self._socketio = socketio
        self._background_thread = None

    def on_connect(self) -> None:  # pragma: no cover - integration hook
        if not self._service.running():
            self._service.start()
        if self._background_thread is None:
            self._background_thread = self._socketio.start_background_task(self._push_predictions)

    def on_disconnect(self) -> None:  # pragma: no cover - integration hook
        # Nothing special; the background thread keeps running to serve SSE clients too.
        pass

    def _push_predictions(self) -> None:
        for payload in self._service.iter_predictions():
            self._socketio.emit(
                config.SOCKETIO_EVENT,
                payload,
                namespace=self.namespace,
                broadcast=True,
            )


def create_app(
    model_path: Optional[Path] = None,
    labels_path: Optional[Path] = None,
    camera_index: int = config.DEFAULT_CAMERA_INDEX,
    confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    sequence_length: Optional[int] = config.DEFAULT_SEQUENCE_LENGTH,
) -> Flask:
    """Factory that configures Flask, Socket.IO and the inference service."""
    app = Flask(__name__)
    CORS(app)

    service = GestureInferenceService(
        model_path=model_path,
        labels_path=labels_path,
        camera_index=camera_index,
        confidence_threshold=confidence_threshold,
        sequence_length=sequence_length,
    )

    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    socketio.on_namespace(GestureNamespace(config.SOCKETIO_NAMESPACE, service, socketio))

    api = Blueprint("gesture_api", __name__)

    @api.route("/status")
    def status() -> Response:
        if not service.running():
            service.start()
        return jsonify(service.snapshot())

    @api.route("/gestures")
    def gestures() -> Response:
        if not service.running():
            service.start()
        return jsonify(service.label_map())

    @api.route("/inference/start", methods=["POST"])
    def start_inference() -> Response:
        service.start()
        return jsonify({"running": True})

    @api.route("/inference/stop", methods=["POST"])
    def stop_inference() -> Response:
        service.stop()
        return jsonify({"running": False})

    @api.route("/stream")
    def stream() -> Response:
        if not service.running():
            service.start()

        def event_stream() -> Any:
            queue = service.subscribe()
            try:
                while True:
                    payload = queue.get()
                    yield "retry: %d\n" % config.SSE_RETRY_MS
                    yield "data: %s\n\n" % json.dumps(payload)
            finally:
                service.unsubscribe(queue)

        headers = {"Cache-Control": "no-cache"}
        return Response(event_stream(), mimetype="text/event-stream", headers=headers)

    @api.route("/config", methods=["POST"])
    def update_config() -> Response:
        data: Dict[str, Any] = request.get_json(force=True)  # type: ignore[arg-type]
        if "confidence_threshold" in data:
            service.confidence_threshold = float(data["confidence_threshold"])
        if "sequence_length" in data:
            service.sequence_length = int(data["sequence_length"])
        if "prediction_cooldown_s" in data:
            service.prediction_cooldown_s = float(data["prediction_cooldown_s"])
        return jsonify(service.snapshot())

    app.register_blueprint(api, url_prefix="/api")
    app.socketio = socketio  # type: ignore[attr-defined]
    app.gesture_service = service  # type: ignore[attr-defined]
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge gesture model with a web frontend")
    parser.add_argument("--model-path", type=Path, default=None, help="Ruta del SavedModel o archivo .keras/.h5")
    parser.add_argument("--labels", type=Path, default=None, help="Ruta opcional a labels.json")
    parser.add_argument("--camera-index", type=int, default=config.DEFAULT_CAMERA_INDEX)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=config.DEFAULT_CONFIDENCE_THRESHOLD,
        help="Probabilidad mínima para emitir un gesto",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=config.DEFAULT_SEQUENCE_LENGTH,
        help="Número de frames acumulados antes de predecir (None usa config del modelo)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:  # pragma: no cover - entry point
    args = parse_args()
    app = create_app(
        model_path=args.model_path,
        labels_path=args.labels,
        camera_index=args.camera_index,
        confidence_threshold=args.confidence_threshold,
        sequence_length=args.sequence_length,
    )
    socketio: SocketIO = app.socketio  # type: ignore[attr-defined]
    socketio.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
