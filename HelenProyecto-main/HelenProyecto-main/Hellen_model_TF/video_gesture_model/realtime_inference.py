"""Real-time gesture recognition using a TensorFlow model trained on video clips.

Ejecuta la cámara en vivo, detecta ambas manos con MediaPipe y clasifica la
secuencia acumulada mediante el modelo entrenado (SavedModel o archivo Keras).
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, Optional

try:  # pragma: no cover - optional dependency en CI
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency en CI
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None  # type: ignore

import numpy as np

try:  # pragma: no cover - optional dependency en CI
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None  # type: ignore

# Imports robustos: paquete o script directo
try:
    from . import config
    from .extract_landmarks import normalise_landmarks
except Exception:
    import config  # type: ignore
    from extract_landmarks import normalise_landmarks  # type: ignore


def parse_args() -> argparse.Namespace:
    """Definir los parámetros de ejecución aceptados desde la terminal."""
    parser = argparse.ArgumentParser(description="Run real-time gesture detection")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,  # si falta, se pedirá por CLI
        help="Directory containing the SavedModel (export) OR a .keras/.h5 model file",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional path to labels.json. If omitted, the script looks inside the model directory.",
    )
    parser.add_argument("--device", type=int, default=0, help="Índice de cámara para OpenCV.")
    parser.add_argument("--confidence-threshold", type=float, default=0.8, help="Umbral de confianza para mostrar etiqueta.")
    parser.add_argument("--sequence-length", type=int, default=config.SEQUENCE_LENGTH, help="Longitud de la ventana temporal.")
    return parser.parse_args()


def load_label_map(path: Path) -> Dict[int, str]:
    """Invertir el diccionario gesto->índice para mostrar etiquetas legibles."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {idx: gesture for gesture, idx in data.items()}


def build_predict_fn(model_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    """
    Devuelve una función predict(x: np.ndarray)->np.ndarray que entrega
    probabilidades (1, num_classes). Soporta:
      - SavedModel exportado en carpeta (contiene saved_model.pb)
      - Archivo Keras (.keras / .h5)
    """
    if tf is None:
        raise RuntimeError(
            "TensorFlow no está disponible en este entorno. Instala tensorflow==2.12.0 para usar el modelo."
        )
    if model_path.is_dir() and (model_path / "saved_model.pb").exists():
        # SavedModel exportado con model.export(...)
        saved = tf.saved_model.load(str(model_path))
        # Nombre típico de la firma impreso por export: "serve"
        if "serve" in saved.signatures:
            infer = saved.signatures["serve"]
        else:
            # fallback común
            infer = saved.signatures.get("serving_default")
            if infer is None:
                raise RuntimeError("No se encontró la firma 'serve' ni 'serving_default' en el SavedModel.")
        # La firma espera argumento llamado como el InputSpec; en tu export se imprime 'landmarks'
        input_name = next(iter(infer.structured_input_signature[1].keys()))
        output_name = next(iter(infer.structured_outputs.keys()))

        def _predict(x: np.ndarray) -> np.ndarray:
            out = infer(**{input_name: tf.constant(x, dtype=tf.float32)})
            y = out[output_name].numpy()
            return y  # shape (1, num_classes)

        return _predict

    # Si es archivo .keras / .h5 cargamos con Keras
    if model_path.suffix.lower() in {".keras", ".h5"} or model_path.is_file():
        model = tf.keras.models.load_model(str(model_path))

        def _predict(x: np.ndarray) -> np.ndarray:
            return model.predict(x, verbose=0)

        return _predict

    raise ValueError(
        f"No se reconoce el formato del modelo en: {model_path}. "
        f"Usa una carpeta SavedModel (con saved_model.pb) o un archivo .keras / .h5."
    )


def main() -> None:
    """Configurar MediaPipe, cargar el modelo y realizar inferencia cuadro a cuadro."""
    args = parse_args()

    if cv2 is None or mp is None:
        raise RuntimeError(
            "OpenCV/MediaPipe no están disponibles en este entorno. Instala opencv-python y mediapipe."
        )

    model_dir_or_file: Optional[Path] = args.model_dir
    if model_dir_or_file is None:
        try:  # import diferido
            from .cli_utils import list_saved_models, prompt_for_model_dir
        except ImportError:
            from cli_utils import list_saved_models, prompt_for_model_dir  # type: ignore
        # Selección interactiva de un SavedModel reciente
        model_dir_or_file = prompt_for_model_dir(list_saved_models())

    # labels.json: si no se especifica, se busca dentro del directorio del modelo
    label_path = args.labels or (model_dir_or_file / "labels.json")
    if not label_path.exists():
        raise FileNotFoundError(
            f"No se encontró labels.json. Indica la ruta con --labels o colócalo en {model_dir_or_file}."
        )

    idx_to_label = load_label_map(label_path)
    print("Gestos reconocidos por este modelo:")
    for idx, label in sorted(idx_to_label.items()):
        print(f"   • {idx}: {label}")

    # Predictor unificado (SavedModel o .keras/.h5)
    predict = build_predict_fn(model_dir_or_file)
    print(f"Modelo cargado desde {model_dir_or_file}")

    # MediaPipe Hands detectará hasta dos manos y devolverá sus landmarks por frame.
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=config.MAX_HANDS,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    buffer: Deque[np.ndarray] = deque(maxlen=args.sequence_length)

    print(
        "Presiona 'q' en la ventana de video para salir. Umbral de confianza:",
        args.confidence_threshold,
    )

    try:
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
                ordering = {"Left": 0, "Right": 1}
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    label = handedness.classification[0].label
                    idx = ordering.get(label, 0)
                    coords = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                        dtype=np.float32,
                    )
                    frame_features[idx] = coords
                    drawing.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )

            buffer.append(normalise_landmarks(frame_features.flatten()))

            if len(buffer) == args.sequence_length:
                # Cuando el buffer está lleno se construye el tensor y se predice la etiqueta.
                input_tensor = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)
                probabilities = predict(input_tensor)[0]  # (num_classes,)
                pred_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_idx])

                if confidence >= args.confidence_threshold:
                    label = idx_to_label.get(pred_idx, "?")
                    cv2.putText(
                        frame,
                        f"{label} ({confidence:.2f})",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3,
                    )

            cv2.imshow("Detección de gestos", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()