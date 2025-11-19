import json
import socket
import threading
import json
import socket
import threading
import time
from contextlib import closing
from http.client import HTTPConnection
from queue import Queue
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pytest

from backendHelen.server import HelenRequestHandler, HelenRuntime, ThreadingHTTPServer


class StubGestureService:
    def __init__(self, **_: Any) -> None:
        self._subscribers: List[Queue] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sequence = 0

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

    def running(self) -> bool:
        return self._running

    def subscribe(self, max_queue: int = 32) -> Queue:
        queue: Queue = Queue(maxsize=max_queue)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def snapshot(self) -> Dict[str, Any]:
        return {"running": self._running, "gestures": ["Start", "Clima", "Reloj"]}

    def _emit_loop(self) -> None:
        gestures = [
            ("Start", 0.94),
            ("Clima", 0.86),
            ("Reloj", 0.81),
        ]
        while self._running:
            label, score = gestures[self._sequence % len(gestures)]
            payload = {
                "label": label,
                "confidence": score,
                "timestamp": time.time(),
                "index": self._sequence,
            }
            for queue in list(self._subscribers):
                try:
                    queue.put_nowait(payload)
                except Exception:
                    pass
            self._sequence += 1
            time.sleep(0.15)


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture(scope='module')
def runtime():
    instance = HelenRuntime(service_factory=lambda **kwargs: StubGestureService(**kwargs))
    instance.start()
    yield instance
    instance.stop()


@pytest.fixture
def live_server(runtime):
    port = find_free_port()
    handler = lambda *args, **kwargs: HelenRequestHandler(*args, runtime=runtime, **kwargs)
    server = ThreadingHTTPServer(('127.0.0.1', port), handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f'http://127.0.0.1:{port}'

    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class SSEClient:
    def __init__(self, base_url: str):
        parts = urlparse(base_url)
        self.connection = HTTPConnection(parts.hostname, parts.port, timeout=5)
        self.connection.request('GET', '/events')
        self.response = self.connection.getresponse()
        if self.response.status != 200:
            raise AssertionError(f'Unexpected SSE status: {self.response.status}')

    def read_event(self, timeout: float = 5.0):
        sock = getattr(getattr(self.response, 'fp', None), 'raw', None)
        if sock is not None and hasattr(sock, 'settimeout'):
            sock.settimeout(timeout)
        data_lines = []
        while True:
            line = self.response.readline()
            if not line:
                return None
            if line.strip() == b'':
                if not data_lines:
                    continue
                payload = b''.join(data_lines).decode('utf-8')
                return json.loads(payload)
            if line.startswith(b'data:'):
                data_lines.append(line[len(b'data: '):])

    def close(self):
        try:
            self.response.close()
        finally:
            self.connection.close()


def test_health_endpoint_reports_status(live_server):
    parts = urlparse(live_server)
    conn = HTTPConnection(parts.hostname, parts.port, timeout=5)
    conn.request('GET', '/healthz')
    response = conn.getresponse()
    body = response.read()
    conn.close()

    assert response.status == 200
    payload = json.loads(body.decode('utf-8'))
    assert payload['model_loaded'] is True
    assert payload['status'] in {'HEALTHY', 'DEGRADED'}
    assert 'session_id' in payload


def test_pipeline_emits_events_over_sse(live_server):
    client = SSEClient(live_server)
    try:
        warmup = client.read_event(timeout=3)
        assert warmup['message'] == 'connected'

        event = client.read_event(timeout=5)
        assert event is not None
        assert event['gesture'] in {'Start', 'Clima', 'Reloj'}
        assert 0.0 <= event['score'] <= 1.0
        assert 'timestamp' in event
        assert 'session_id' not in event or isinstance(event['session_id'], str)
    finally:
        client.close()


def test_http_post_broadcasts_payload(live_server):
    client = SSEClient(live_server)
    try:
        client.read_event(timeout=3)  # Descarta warmup

        payload = {
            'gesture': 'Foco',
            'character': 'Foco',
            'score': 0.91,
            'latency_ms': 12.5,
            'sequence': 999,
        }
        parts = urlparse(live_server)
        conn = HTTPConnection(parts.hostname, parts.port, timeout=5)
        conn.request('POST', '/gestures/gesture-key', body=json.dumps(payload), headers={'Content-Type': 'application/json'})
        response = conn.getresponse()
        response.read()
        conn.close()
        assert response.status == 200

        event = client.read_event(timeout=3)
        assert event['gesture'] == 'Foco'
        assert event['raw']['sequence'] == 999
        assert event['score'] == pytest.approx(0.91)
    finally:
        client.close()


def test_foco_command_is_not_flagged_as_activation(live_server):
    client = SSEClient(live_server)
    try:
        client.read_event(timeout=3)

        payload = {
            'gesture': 'Foco',
            'character': 'Foco',
            'score': 0.77,
            'latency_ms': 9.1,
            'sequence': 321,
        }

        parts = urlparse(live_server)
        conn = HTTPConnection(parts.hostname, parts.port, timeout=5)
        conn.request('POST', '/gestures/gesture-key', body=json.dumps(payload), headers={'Content-Type': 'application/json'})
        response = conn.getresponse()
        response.read()
        conn.close()
        assert response.status == 200

        for _ in range(6):
            event = client.read_event(timeout=3)
            if not event:
                continue
            if event.get('raw', {}).get('character') == 'Foco':
                assert not event.get('active', False)
                break
        else:
            pytest.fail('No se recibió el evento de Foco enviado por HTTP')
    finally:
        client.close()
