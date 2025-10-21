from __future__ import annotations

import http.client as http_client
import json
from urllib.parse import ParseResult, urlparse

try:  # pragma: no cover - compatibilidad con ejecuciones como script
    from .helpers import GestureData, labels_dict
except ImportError:  # pragma: no cover
    from helpers import GestureData, labels_dict

server_url = 'http://127.0.0.1:5000'
post_server_url = f'{server_url}/gestures/gesture-key'

_SEQUENCE = 0
_URL_PARTS = urlparse(post_server_url)


def _build_path(parts: ParseResult) -> str:
    path = parts.path or '/'
    if parts.query:
        path = f"{path}?{parts.query}"
    return path


def post_gesturekey(prediction, *, score: float = 1.0, session_id: str | None = None) -> int:
    global _SEQUENCE
    _SEQUENCE += 1

    gesture_name = prediction if __is_gesture_name(prediction) else None
    data = GestureData(
        gesture=gesture_name,
        character=str(prediction),
        score=float(score),
        session_id=session_id,
        sequence=_SEQUENCE,
    ).to_payload()
    data.setdefault('latency_ms', 0.0)

    payload = json.dumps(data).encode('utf-8')
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    port = _URL_PARTS.port or (443 if _URL_PARTS.scheme == 'https' else 80)
    connection = http_client.HTTPConnection(_URL_PARTS.hostname, port=port, timeout=3)
    try:
        connection.request('POST', _build_path(_URL_PARTS), body=payload, headers=headers)
        response = connection.getresponse()
        response.read()
        return response.status
    except OSError as exc:
        print(f'Error Server: {exc}')
        return 500
    finally:
        try:
            connection.close()
        except Exception:
            pass


def __is_gesture_name(prediction):
    return prediction in labels_dict.values()
