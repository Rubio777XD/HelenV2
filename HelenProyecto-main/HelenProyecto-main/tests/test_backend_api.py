import sys
import types
from pathlib import Path

import pytest

pytest.importorskip('flask')
pytest.importorskip('flask_socketio')

from backendHelen.server import app, socket

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / 'Hellen_model_RN'

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


@pytest.fixture
def flask_app_client():
    return app.test_client()


@pytest.fixture
def socketio_client():
    client = socket.test_client(app)
    assert client.is_connected()
    yield client
    if client.is_connected():
        client.disconnect()


def test_index_route_renders(flask_app_client):
    response = flask_app_client.get('/')
    assert response.status_code == 200
    assert b'<!DOCTYPE html>' in response.data


def test_health_endpoint_emits_socket_event(flask_app_client, socketio_client):
    socketio_client.get_received()

    response = flask_app_client.get('/health')

    assert response.status_code == 200
    assert response.json == {'status': 'ok'}

    events = socketio_client.get_received()
    assert any(event['name'] == 'message' and event['args'][0] == {'status': 'ok'} for event in events)


def test_publish_gesture_key_broadcasts_payload(flask_app_client, socketio_client):
    socketio_client.get_received()

    payload = {'gesture': 'Clima', 'character': 'Clima'}
    response = flask_app_client.post('/gestures/gesture-key', json=payload)

    assert response.status_code == 200

    events = socketio_client.get_received()
    assert any(event['name'] == 'message' and event['args'][0] == payload for event in events)


def test_unknown_route_returns_404(flask_app_client):
    response = flask_app_client.get('/ruta-inexistente')
    assert response.status_code == 404


def test_socket_reconnect_cycle():
    client = socket.test_client(app)
    assert client.is_connected()
    client.disconnect()
    assert not client.is_connected()

    reconnected = client.connect()
    assert reconnected
    assert client.is_connected()
    client.disconnect()


def test_post_gesturekey_round_trip(monkeypatch, flask_app_client):
    from Hellen_model_RN import backendConexion as model_backend

    socket_client = socket.test_client(app)
    assert socket_client.is_connected()
    socket_client.get_received()

    def fake_post(url, json):
        response = flask_app_client.post('/gestures/gesture-key', json=json)
        return types.SimpleNamespace(status_code=response.status_code)

    monkeypatch.setattr(model_backend.requests, 'post', fake_post)

    status = model_backend.post_gesturekey('Inicio')
    assert status == 200

    events = socket_client.get_received()
    assert any(event['args'][0]['character'] == 'Inicio' for event in events)

    socket_client.disconnect()
