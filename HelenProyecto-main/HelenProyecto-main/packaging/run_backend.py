"""Entrypoint used by PyInstaller to launch the HELEN backend."""

from backendHelen import server


def main() -> int:
    return server.main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
