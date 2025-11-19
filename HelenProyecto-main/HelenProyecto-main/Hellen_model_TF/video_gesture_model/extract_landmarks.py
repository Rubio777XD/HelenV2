"""Extract MediaPipe landmarks from recorded gesture videos.

This script iterates through the gesture folders in ``data/raw_videos`` and
converts each clip into a fixed-length sequence of 3D landmarks for up to two
hands. The resulting tensors and labels are stored in ``data/features`` and will
be consumed by the TensorFlow training script.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:  # pragma: no cover - optional dependency en CI
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency en CI
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None  # type: ignore
import numpy as np

# Permite ejecutar como paquete (python -m pkg.mod) o como script directo
try:
    from . import config
except ImportError:  # ejecución directa
    import config  # type: ignore


@dataclass
class Sample:
    features: np.ndarray
    label: int
    gesture: str


def parse_args() -> argparse.Namespace:
    """Leer la lista de gestos y parámetros de salida proporcionados por el usuario."""
    parser = argparse.ArgumentParser(description="Extract landmarks from gesture clips")
    parser.add_argument(
        "gestures",
        nargs="*",  # opcional: si faltan, se pedirá por CLI con cli_utils
        help="Gesture folder names located under data/raw_videos",
    )
    parser.add_argument(
        "--output",
        default="gesture_dataset",
        help="Base name for the generated dataset files (without extension).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=config.SEQUENCE_LENGTH,
        help="Number of frames per sample after padding/truncation.",
    )
    return parser.parse_args()


def normalise_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalise coordinates relative to the wrist of each hand.

    Se desplaza cada coordenada para que la muñeca de cada mano actúe como origen.
    """
    if landmarks.size == 0:
        return landmarks
    reshaped = landmarks.reshape(
        config.MAX_HANDS, config.NUM_HAND_LANDMARKS, config.LANDMARK_DIM
    )
    for hand_idx in range(config.MAX_HANDS):
        hand_landmarks = reshaped[hand_idx]
        if not hand_landmarks.any():
            continue
        wrist = hand_landmarks[0].copy()
        reshaped[hand_idx] -= wrist
    return reshaped.reshape(-1)


def extract_from_video(
    video_path: Path, hands: mp.solutions.hands.Hands, sequence_length: int
) -> np.ndarray:
    """Procesar un video y devolver una secuencia con landmarks normalizados."""
    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        frame_features = np.zeros(
            (config.MAX_HANDS, config.NUM_HAND_LANDMARKS, config.LANDMARK_DIM),
            dtype=np.float32,
        )
        if results.multi_hand_landmarks and results.multi_handedness:
            ordering: Dict[str, int] = {"Left": 0, "Right": 1}
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label
                hand_idx = ordering.get(label, 0)
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )
                frame_features[hand_idx] = coords

        frames.append(normalise_landmarks(frame_features.flatten()))

    cap.release()

    if not frames:
        raise ValueError(f"El video {video_path} no contiene manos detectadas.")

    # Ajustar longitud de la secuencia.
    # Si hay más frames que los requeridos se recorta; si faltan se añade padding.
    frames_array = np.array(frames, dtype=np.float32)
    if len(frames_array) >= sequence_length:
        return frames_array[:sequence_length]

    padding = np.zeros(
        (sequence_length - len(frames_array), frames_array.shape[1]), dtype=np.float32
    )
    return np.vstack([frames_array, padding])


def main() -> None:
    """Recorrer los videos de cada gesto y generar el dataset comprimido."""
    args = parse_args()

    gestures = list(args.gestures or [])
    if not gestures:
        try:  # imports diferidos para evitar dependencias al usarse como módulo
            from .cli_utils import gesture_inventory, prompt_for_multiple_gestures
        except ImportError:  # ejecución directa
            from cli_utils import gesture_inventory, prompt_for_multiple_gestures  # type: ignore
        gestures = prompt_for_multiple_gestures(gesture_inventory())
    print(f"Procesando las señas: {', '.join(gestures)}")

    samples: List[Sample] = []
    label_map: Dict[str, int] = {gesture: idx for idx, gesture in enumerate(sorted(gestures))}

    # Configuramos MediaPipe Hands para detectar hasta dos manos por cuadro.
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=config.MAX_HANDS,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    try:
        for gesture in gestures:
            gesture_dir = config.VIDEOS_DIR / gesture
            if not gesture_dir.exists():
                raise FileNotFoundError(
                    f"No se encontró la carpeta de videos para el gesto '{gesture}'."
                )

            before = len(samples)
            for video_path in sorted(gesture_dir.glob("*.mp4")):
                # Extraemos los landmarks por frame y los asociamos a la etiqueta correspondiente.
                features = extract_from_video(video_path, hands, args.sequence_length)
                samples.append(
                    Sample(features=features, label=label_map[gesture], gesture=gesture)
                )
                print(f"✅ Procesado {video_path}")

            if len(samples) == before:
                print(f"⚠️  No se encontraron videos mp4 para la seña '{gesture}'.")
    finally:
        hands.close()

    if not samples:
        raise RuntimeError("No se generaron muestras. Asegúrate de que existan videos mp4.")

    # Se agrupan los datos en tensores listos para entrenar o validar el modelo.
    X = np.stack([sample.features for sample in samples])
    y = np.array([sample.label for sample in samples], dtype=np.int64)

    summary: Dict[str, int] = {}
    for sample in samples:
        summary[sample.gesture] = summary.get(sample.gesture, 0) + 1

    dataset_name = f"{args.output}.npz"
    dataset_path = config.FEATURES_DIR / dataset_name
    np.savez_compressed(dataset_path, X=X, y=y)
    print(f"📦 Dataset guardado en {dataset_path}")

    label_map_path = config.FEATURES_DIR / f"{args.output}_labels.json"
    with label_map_path.open("w", encoding="utf-8") as fp:
        json.dump(label_map, fp, ensure_ascii=False, indent=2)
    print(f"🗂️  Mapa de etiquetas guardado en {label_map_path}")

    print("Resumen de muestras por seña:")
    for gesture, count in sorted(summary.items()):
        print(f"   • {gesture}: {count} muestra(s)")


if __name__ == "__main__":
    main()