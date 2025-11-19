import threading
import time
from queue import Queue
from typing import Any, Dict, List

import pytest

from backendHelen.server import GestureLabelMapper, HelenRuntime, RuntimeConfig, TensorFlowGesturePipeline


class DummyRuntime(HelenRuntime):
    """Runtime subclass that records events for assertions."""

    def __init__(self, **kwargs: Any) -> None:
        config = kwargs.pop("config", RuntimeConfig())
        super().__init__(config=config, **kwargs)
        self.received: List[Dict[str, Any]] = []

    def handle_prediction(self, payload: Dict[str, Any]) -> None:  # type: ignore[override]
        super().handle_prediction(payload)
        self.received.append(payload)


class PredictOnceService:
    """Test double that mimics ``GestureInferenceService``."""

    def __init__(self, **_: Any) -> None:
        self._subscribers: List[Queue] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._emit_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def subscribe(self, max_queue: int = 32) -> Queue:
        queue: Queue = Queue(maxsize=max_queue)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def snapshot(self) -> Dict[str, Any]:
        return {"running": self._running, "gestures": ["Start", "Clima"]}

    def _emit_loop(self) -> None:
        while self._running:
            if not self._subscribers:
                time.sleep(0.01)
                continue
            payload = {
                "label": "Start",
                "confidence": 0.93,
                "timestamp": time.time(),
                "index": 0,
            }
            for queue in list(self._subscribers):
                queue.put(payload)
            time.sleep(0.1)
            break
        self._running = False


class EagerService(PredictOnceService):
    """Pushes two gestures sequentially for latency tests."""

    def _emit_loop(self) -> None:
        while self._running and not self._subscribers:
            time.sleep(0.01)
        gestures = [("Start", 0.95), ("Clima", 0.88)]
        for idx, (label, score) in enumerate(gestures):
            payload = {
                "label": label,
                "confidence": score,
                "timestamp": time.time(),
                "index": idx,
            }
            for queue in list(self._subscribers):
                queue.put(payload)
            time.sleep(0.05)
        self._running = False


def test_gesture_label_mapper_applies_aliases():
    mapper = GestureLabelMapper({"hola": "Start", "nube": "Clima"})
    assert mapper.normalize("HOLA") == "Start"
    assert mapper.normalize("nube") == "Clima"
    assert mapper.normalize("Reloj") == "Reloj"


def test_tensorflow_pipeline_emits_predictions():
    runtime = DummyRuntime(service_factory=lambda **kwargs: PredictOnceService(**kwargs))
    runtime.start()
    try:
        deadline = time.time() + 2.0
        while not runtime.received and time.time() < deadline:
            time.sleep(0.05)
        assert runtime.received, "La tubería no recibió predicciones"
        first = runtime.received[0]
        assert first["label"] == "Start"
        assert 0.0 < first["confidence"] <= 1.0
    finally:
        runtime.stop()


def test_runtime_builds_activation_events():
    runtime = DummyRuntime(service_factory=lambda **kwargs: EagerService(**kwargs))
    runtime.start()
    try:
        deadline = time.time() + 2.0
        while len(runtime.received) < 2 and time.time() < deadline:
            time.sleep(0.05)
        assert len(runtime.received) >= 2
        activation_event = runtime.last_prediction
        assert activation_event is not None
        assert activation_event["gesture"] in {"Start", "Clima"}
        assert "score" in activation_event
    finally:
        runtime.stop()


def test_tensorflow_pipeline_class_is_instantiable():
    runtime = DummyRuntime(service_factory=lambda **kwargs: PredictOnceService(**kwargs))
    pipeline = TensorFlowGesturePipeline(runtime, service_factory=lambda **kwargs: PredictOnceService(**kwargs))
    assert pipeline.snapshot()["gestures"] == []
    pipeline.start()
    time.sleep(0.1)
    assert pipeline.is_running() is True
    pipeline.stop()
    assert pipeline.is_running() is False
    runtime.stop()
