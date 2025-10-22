import json
import socket
import threading
from contextlib import closing
from http.client import HTTPConnection
from urllib.parse import urlparse

import pytest

from backendHelen.server import (
    HelenRequestHandler,
    HelenRuntime,
    ThreadingHTTPServer,
    GestureDecisionEngine,
    GestureMetrics,
    ConsensusConfig,
)
from Hellen_model_RN.simple_classifier import Prediction


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture(scope='module')
def runtime():
    instance = HelenRuntime()
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
        assert event['gesture'] in ('Start', 'Clima', 'Foco', 'Ajustes', 'Inicio', 'Dispositivos', 'Reloj', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
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


def test_command_debounce_prevents_spam():
    metrics = GestureMetrics()
    engine = GestureDecisionEngine(
        metrics=metrics,
        consensus=ConsensusConfig(window_size=3, required_votes=1),
    )

    timestamp = 0.0
    outcome_start = engine.process(Prediction(label='Start', score=0.92), timestamp=timestamp)
    assert outcome_start.emit is True

    timestamp += 1.0  # supera el cooldown de activación
    outcome_clima = engine.process(Prediction(label='Clima', score=0.93), timestamp=timestamp)
    assert outcome_clima.emit is True

    timestamp += 0.3  # dentro del periodo de debounce (0.75s)
    outcome_repeat = engine.process(Prediction(label='Clima', score=0.94), timestamp=timestamp)
    assert outcome_repeat.emit is False
    assert outcome_repeat.reason == 'command_debounce_active'
