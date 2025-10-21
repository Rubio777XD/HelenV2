"""Minimal Socket.IO stub compatible with the subset used in tests."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Optional


class SocketIO:
    def __init__(self, app=None, **kwargs) -> None:
        self.app = None
        self.cors_allowed_origins = kwargs.get("cors_allowed_origins")
        self._event_handlers: DefaultDict[str, List[Callable[..., Any]]] = defaultdict(list)
        self._clients: List["_SocketTestClient"] = []
        if app is not None:
            self.init_app(app, **kwargs)

    def init_app(self, app, **kwargs) -> None:
        self.app = app

    def on(self, event_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._event_handlers[event_name].append(func)
            return func

        return decorator

    def emit(self, event_name: str, data: Any, namespace: Optional[str] = None) -> None:
        namespace = namespace or "/"
        for client in list(self._clients):
            client._queue.append({"name": event_name, "args": [data], "namespace": namespace})
        # Server handlers registered via @socket.on should observe outgoing traffic for logging
        for handler in self._event_handlers.get(event_name, []):
            handler(data)

    def run(self, app, host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
        # No-op in the stub environment. Provided only for API compatibility.
        return None

    def test_client(self, app) -> "_SocketTestClient":
        client = _SocketTestClient(self)
        self._clients.append(client)
        client.connect()
        return client


class _SocketTestClient:
    def __init__(self, server: SocketIO) -> None:
        self._server = server
        self._queue: List[Dict[str, Any]] = []
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        if self._connected:
            return True
        self._connected = True
        if self not in self._server._clients:
            self._server._clients.append(self)
        for handler in self._server._event_handlers.get("connect", []):
            handler()
        return True

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        if self in self._server._clients:
            self._server._clients.remove(self)
        for handler in self._server._event_handlers.get("disconnect", []):
            handler()

    def emit(self, event_name: str, data: Any) -> None:
        for handler in self._server._event_handlers.get(event_name, []):
            handler(data)

    def get_received(self) -> List[Dict[str, Any]]:
        events = list(self._queue)
        self._queue.clear()
        return events
