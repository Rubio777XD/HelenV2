"""Raspberry Pi client that captures hand landmarks and queries the EC2 server."""

from __future__ import annotations

import collections
import time
from typing import Deque, Tuple

import cv2
import mediapipe as mp
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------

# TODO: Coloca aquí la IP pública o DNS de tu instancia EC2.
EC2_URL = "http://<IP_PUBLICA_EC2>:8000/predict"

SEQUENCE_LENGTH = 40  # Debe coincidir con el valor usado durante el entrenamiento.
MAX_HANDS = 2
NUM_LANDMARKS = 21
LANDMARK_DIM = 3
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

REQUEST_TIMEOUT = 2.0  # segundos


# ---------------------------------------------------------------------------
# Utilidades de landmarks
# ---------------------------------------------------------------------------

def normalise_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normaliza los landmarks restando la posición de la muñeca por mano."""
    reshaped = landmarks.reshape(MAX_HANDS, NUM_LANDMARKS, LANDMARK_DIM)
    for hand_idx in range(MAX_HANDS):
        hand_landmarks = reshaped[hand_idx]
        if not np.any(hand_landmarks):
            continue
        wrist = hand_landmarks[0].copy()
        reshaped[hand_idx] -= wrist
    return reshaped.reshape(-1)


def extract_frame_landmarks(
    frame: np.ndarray, hands: mp.solutions.hands.Hands
) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve el frame original y el vector de landmarks normalizados."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    landmarks = np.zeros((MAX_HANDS, NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        ordering = {"Left": 0, "Right": 1}
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label
            hand_idx = ordering.get(label, 0)
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
            )
            landmarks[hand_idx] = coords

    normalised = normalise_landmarks(landmarks.reshape(-1))
    return frame, normalised


# ---------------------------------------------------------------------------
# Comunicación con el servidor
# ---------------------------------------------------------------------------

def query_server(sequence: np.ndarray) -> Tuple[str, float]:
    """Envía la secuencia al servidor remoto y devuelve (label, confidence)."""
    payload = {"sequence": sequence.tolist()}
    response = requests.post(EC2_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    label = data.get("label", "?")
    confidence = float(data.get("confidence", 0.0))
    return label, confidence


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def main() -> None:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    sequence_buffer: Deque[np.ndarray] = collections.deque(maxlen=SEQUENCE_LENGTH)
    last_label = "--"
    last_confidence = 0.0
    last_error_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer de la cámara. Esperando...")
                time.sleep(0.1)
                continue

            frame, features = extract_frame_landmarks(frame, hands)
            sequence_buffer.append(features)

            display_text = f"{last_label} ({last_confidence:.2f})"

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                sequence = np.stack(sequence_buffer, axis=0)
                try:
                    label, confidence = query_server(sequence)
                    last_label = label
                    last_confidence = confidence
                    display_text = f"{label} ({confidence:.2f})"
                except requests.RequestException:
                    # Evitamos inundar la pantalla con mensajes de error.
                    current_time = time.time()
                    if current_time - last_error_time > 2.0:
                        print("⚠️  Error al comunicarse con el servidor de inferencia.")
                        last_error_time = current_time
                    display_text = "ERR conn"

            cv2.putText(
                frame,
                display_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0) if display_text != "ERR conn" else (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

            cv2.imshow("Gesture Client", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
