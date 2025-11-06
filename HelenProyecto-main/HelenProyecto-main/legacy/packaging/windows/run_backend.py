"""Entrypoint used by PyInstaller to launch the HELEN backend."""

from importlib import import_module
from typing import Callable, Iterable, Tuple


def _resolve_entrypoint() -> Callable[[], int | None]:
    """Locate the backend launcher, supporting legacy and new layouts."""

    candidates: Iterable[Tuple[str, str]] = (
        ("backendHelen.server", "main"),
        ("backendHelen.main_backend", "main"),
        ("helen_backend.main_backend", "main"),
        ("helen_backend.server", "main"),
    )

    for module_name, attr_name in candidates:
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            continue
        entry = getattr(module, attr_name, None)
        if callable(entry):
            return entry  # type: ignore[return-value]

    tried = ", ".join(f"{mod}.{attr}" for mod, attr in candidates)
    raise RuntimeError(
        "No se encontró el punto de entrada del backend. Se intentó resolver: {0}".format(tried)
    )


_ENTRYPOINT = _resolve_entrypoint()


def main() -> int:
    result = _ENTRYPOINT()
    return 0 if result is None else int(result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
