"""Minimal Flask-compatible shim for offline testing environments."""
from __future__ import annotations

import contextvars
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

__all__ = [
    "Flask",
    "Response",
    "jsonify",
    "request",
    "render_template",
]


_request_ctx: contextvars.ContextVar["_RequestState"] = contextvars.ContextVar("flask_request")
_app_ctx: contextvars.ContextVar["Flask"] = contextvars.ContextVar("flask_app")


class Request:
    """Very small subset of Flask's request object."""

    def __init__(self, json: Optional[Any] = None) -> None:
        self.json = json


class _RequestState:
    def __init__(self, request: Request) -> None:
        self.request = request


class _RequestProxy:
    def __getattr__(self, item: str) -> Any:
        state = _request_ctx.get(None)
        if state is None:
            raise RuntimeError("No active request context")
        return getattr(state.request, item)


request = _RequestProxy()


class Response:
    """Simplified HTTP response object."""

    def __init__(
        self,
        response: Any = b"",
        status: int = 200,
        mimetype: str = "text/html",
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if isinstance(response, bytes):
            self.data = response
        elif response is None:
            self.data = b""
        else:
            self.data = str(response).encode("utf-8")
        self.status_code = status
        self.mimetype = mimetype
        self.headers = headers or {}

    @property
    def json(self) -> Any:
        if not self.data:
            return None
        return json.loads(self.data.decode("utf-8"))


class Flask:
    """Extremely small Flask-like application for tests."""

    def __init__(self, import_name: str) -> None:
        self.import_name = import_name
        self.config: Dict[str, Any] = {}
        self._routes: Dict[Tuple[str, str], Callable[..., Any]] = {}

        module = sys.modules.get(import_name)
        if module is not None and hasattr(module, "__file__"):
            self.root_path = Path(module.__file__).resolve().parent
        else:
            self.root_path = Path.cwd()
        self.template_folder = self.root_path / "templates"
        self.response_class = Response

    # route decorators -------------------------------------------------
    def route(self, rule: str, methods: Optional[Iterable[str]] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        allowed_methods = [method.upper() for method in (methods or ["GET"])]

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            for method in allowed_methods:
                self._routes[(method, rule)] = func
            return func

        return decorator

    def get(self, rule: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.route(rule, methods=["GET"])

    def post(self, rule: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self.route(rule, methods=["POST"])

    # request handling -------------------------------------------------
    def test_client(self) -> "_FlaskTestClient":
        return _FlaskTestClient(self)

    def make_response(self, rv: Any) -> Response:
        if isinstance(rv, Response):
            return rv
        if isinstance(rv, tuple):
            body, status = rv[0], rv[1]
            return Response(body, status=status)
        if isinstance(rv, (bytes, bytearray)):
            return Response(bytes(rv))
        if isinstance(rv, str):
            return Response(rv)
        if rv is None:
            return Response(b"", status=204)
        return Response(json.dumps(rv), mimetype="application/json")


class _FlaskTestClient:
    def __init__(self, app: Flask) -> None:
        self._app = app

    def _invoke(self, path: str, method: str, json_payload: Optional[Any] = None) -> Response:
        endpoint = self._app._routes.get((method.upper(), path))
        if endpoint is None:
            return Response(status=404)

        request_obj = Request(json=json_payload)
        request_token = _request_ctx.set(_RequestState(request_obj))
        app_token = _app_ctx.set(self._app)
        try:
            rv = endpoint()
        finally:
            _request_ctx.reset(request_token)
            _app_ctx.reset(app_token)
        return self._app.make_response(rv)

    def get(self, path: str) -> Response:
        return self._invoke(path, "GET")

    def post(self, path: str, json: Optional[Any] = None) -> Response:
        return self._invoke(path, "POST", json_payload=json)


def jsonify(*args: Any, **kwargs: Any) -> Response:
    if args and kwargs:
        raise TypeError("jsonify expected either args or kwargs, not both")
    if len(args) == 1:
        payload = args[0]
    elif args:
        payload = list(args)
    else:
        payload = kwargs
    return Response(json.dumps(payload), mimetype="application/json")


def render_template(template_name: str, **context: Any) -> str:
    app = _app_ctx.get(None)
    if app is None:
        raise RuntimeError("render_template called outside of app context")
    template_path = app.template_folder / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Template {template_name} not found in {app.template_folder}")
    return template_path.read_text(encoding="utf-8")
