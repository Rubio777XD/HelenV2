"""Utility for recording short gesture clips with both hands visible.

Press ``s`` to start capturing a clip and ``q`` to exit. Each clip is saved to
``data/raw_videos/<gesture>/<gesture>_<timestamp>.mp4`` so it can later be
processed into landmarks. Recording defaults are defined in :mod:`config`.

Herramienta de consola para grabar videos cortos de cada gesto manteniendo ambas
manos en cuadro; estos clips se usarán posteriormente para extraer landmarks.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

import cv2

# Permite ejecutar como paquete (python -m pkg.mod) o como script directo
try:
    from . import config
    from .cli_utils import (
        gesture_inventory,
        print_inventory_table,
        prompt_for_single_gesture,
    )
except ImportError:  # ejecución directa
    import config  # type: ignore
    from cli_utils import (  # type: ignore
        gesture_inventory,
        print_inventory_table,
        prompt_for_single_gesture,
    )


def parse_args() -> argparse.Namespace:
    """Interpretar los argumentos entregados por la línea de comandos."""
    parser = argparse.ArgumentParser(description="Record gesture clips with OpenCV")
    parser.add_argument(
        "gesture",
        nargs="?",  # opcional: si falta, se pedirá por CLI
        help="Name of the gesture being recorded (used as folder prefix).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=config.CLIP_DURATION,
        help="Length of each recorded clip in seconds.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=config.FPS,
        help="Frames per second of the recorded clip.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Camera device index passed to OpenCV (default: 0).",
    )
    return parser.parse_args()


def record_clip(cap: cv2.VideoCapture, writer: cv2.VideoWriter, duration: float) -> None:
    """Grabar un clip durante el tiempo indicado escribiendo cada frame en disco."""
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        cv2.imshow("Grabando gesto", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main() -> None:
    """Ejecutar el flujo principal de captura de video para un gesto concreto."""
    args = parse_args()

    existing = gesture_inventory()

    if args.gesture:
        gesture_name = args.gesture
    else:
        if existing:
            print("Señas registradas actualmente:")
            print_inventory_table(existing)
        else:
            print(
                "Aún no hay señas registradas. Se creará la carpeta cuando guardes el primer clip."
            )
        gesture_name = prompt_for_single_gesture(existing, show_table=False)

    if gesture_name.strip() == "":
        raise ValueError("El nombre de la seña no puede estar vacío.")

    gesture_dir = config.VIDEOS_DIR / gesture_name
    gesture_dir.mkdir(parents=True, exist_ok=True)

    existing_count = sum(1 for _ in gesture_dir.glob("*.mp4"))

    print(
        f"Seleccionada la seña '{gesture_name}'. Actualmente tiene {existing_count} clip(s)."
    )

    # Inicializamos la cámara con la resolución y FPS definidos en la configuración.
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara.")

    print(
        f"Grabando gesto '{gesture_name}'. Presiona 's' para capturar un clip, 'q' para salir."
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    session_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Vista previa", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                # Se genera un nombre único con sello de tiempo para cada grabación.
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = gesture_dir / f"{gesture_name}_{timestamp}.mp4"
                writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    args.fps,
                    (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                )
                print(f"➡️  Grabando clip en {video_path}")
                record_clip(cap, writer, args.duration)
                writer.release()
                session_count += 1
                total_count = existing_count + session_count
                print(
                    f"✅ Clip guardado. Clips en esta sesión: {session_count}. Total acumulado: {total_count}.\n"
                )

            elif key == ord("q"):
                print("Grabación finalizada por el usuario.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    