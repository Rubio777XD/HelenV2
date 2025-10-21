"""Diagnostic utilities for validating HELEN hardware integrations."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

from . import server

LOGGER = logging.getLogger("helen.diagnostics")


def run_camera_check(
    *,
    camera_index: int = 0,
    frames: int = 30,
    detection_confidence: float = 0.7,
    tracking_confidence: float = 0.6,
    allow_missing: bool = False,
) -> int:
    """Attempt to read a handful of frames from the configured camera.

    The function prints a short status report and returns ``0`` on success.  When
    ``allow_missing`` is ``True`` any camera errors are downgraded to warnings so
    the diagnostic can run on environments without physical hardware (for
    example CI runners).
    """

    try:
        stream = server.CameraGestureStream(
            camera_index=camera_index,
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
        )
    except Exception as error:
        message = f"No se pudo inicializar la cámara {camera_index}: {error}"
        if allow_missing:
            LOGGER.warning(message)
            return 0
        LOGGER.error(message)
        return 1

    successes = 0
    failures = 0
    try:
        for _ in range(max(1, frames)):
            try:
                stream.next(timeout=2.0)
            except TimeoutError:
                failures += 1
                LOGGER.debug("No se detectó mano en el frame actual")
            except Exception as error:  # pragma: no cover - errores inesperados
                failures += 1
                LOGGER.error("Error leyendo la cámara: %s", error)
            else:
                successes += 1
                LOGGER.debug("Frame válido capturado")
            time.sleep(0.05)
    finally:
        stream.close()

    if successes:
        LOGGER.info(
            "Cámara %s operativa. Frames válidos: %s/%s",
            camera_index,
            successes,
            frames,
        )
        return 0

    if allow_missing:
        LOGGER.warning(
            "No se detectaron manos en %s frames, pero se permite continuar.",
            frames,
        )
        return 0

    LOGGER.error(
        "No se detectaron gestos en la cámara %s tras %s intentos.",
        camera_index,
        frames,
    )
    return 2


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Herramientas de diagnóstico de HELEN")
    parser.add_argument("--camera-index", type=int, default=0, help="Índice de la cámara a verificar")
    parser.add_argument("--frames", type=int, default=30, help="Cantidad de frames a evaluar")
    parser.add_argument("--detection-confidence", type=float, default=0.7, help="Umbral de detección de MediaPipe")
    parser.add_argument("--tracking-confidence", type=float, default=0.6, help="Umbral de seguimiento de MediaPipe")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="No falla si la cámara no está disponible (útil para CI)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    return run_camera_check(
        camera_index=args.camera_index,
        frames=args.frames,
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence,
        allow_missing=args.allow_missing,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
