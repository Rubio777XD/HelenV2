"""Flask server for remote gesture inference on AWS EC2.

This script exposes a `/predict` endpoint that receives landmark sequences from
remote clients (e.g. a Raspberry Pi) and returns the gesture prediction produced
by a TensorFlow SavedModel.

Usage:
    python server_ec2_flask.py

The server listens on 0.0.0.0:8000 so it can be reached from outside the EC2
instance. Make sure the instance's security group allows inbound TCP traffic on
port 8000.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# TODO: Coloca aquí la ruta donde subirás el SavedModel en la instancia EC2.
# El directorio debe contener el archivo `saved_model.pb` y la carpeta `variables/`.
MODEL_DIR = Path("data/models/gesture_model_latest")

# TODO: Copia el archivo labels.json (generado durante la extracción/entrenamiento)
# en la misma carpeta que el modelo o ajusta la ruta en consecuencia.
LABELS_PATH = MODEL_DIR / "labels.json"

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8000

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Model loading utilities
# ---------------------------------------------------------------------------

def load_label_map(labels_path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Load the gesture label mapping from disk."""
    if not labels_path.exists():
        raise FileNotFoundError(
            f"No se encontró labels.json en {labels_path}. Asegúrate de subirlo al servidor."
        )
    with labels_path.open("r", encoding="utf-8") as fp:
        gesture_to_index: Dict[str, int] = json.load(fp)
    index_to_gesture = {idx: gesture for gesture, idx in gesture_to_index.items()}
    return index_to_gesture, gesture_to_index


def load_model(model_dir: Path) -> tf.types.experimental.GenericFunction:
    """Load the TensorFlow SavedModel from disk and return the serving function."""
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio del modelo en {model_dir}. Sube el SavedModel antes de ejecutar el servidor."
        )
    logger.info("Cargando modelo desde %s", model_dir)
    model = tf.saved_model.load(str(model_dir))
    # Preferimos la firma 'serving_default', pero si no existe usamos la primera disponible.
    if "serving_default" in model.signatures:
        return model.signatures["serving_default"]
    if model.signatures:
        first_signature = next(iter(model.signatures.values()))
        logger.warning("No se encontró 'serving_default'. Usando la primera firma disponible.")
        return first_signature
    raise ValueError("El SavedModel cargado no contiene firmas para servir.")


INDEX_TO_GESTURE, GESTURE_TO_INDEX = load_label_map(LABELS_PATH)
SERVE_FN = load_model(MODEL_DIR)


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)


def prepare_input(sequence: Any) -> np.ndarray:
    """Validate and convert the incoming JSON payload to a NumPy array."""
    if sequence is None:
        raise ValueError("El campo 'sequence' es obligatorio.")

    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            "La secuencia debe ser una lista bidimensional [sequence_length, feature_dim]."
        )
    return arr


def run_inference(sequence: np.ndarray) -> Tuple[int, float]:
    """Run the TensorFlow model and return the predicted index and confidence."""
    input_tensor = tf.convert_to_tensor(sequence, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # Añadimos dimensión batch.

    outputs = SERVE_FN(input_tensor)
    if not outputs:
        raise RuntimeError("El modelo no produjo salidas.")

    # Tomamos la primera salida disponible (probabilidades o logits).
    prediction = next(iter(outputs.values()))
    prediction_np = prediction.numpy()

    if prediction_np.ndim == 2 and prediction_np.shape[0] == 1:
        prediction_np = prediction_np[0]

    if prediction_np.ndim != 1:
        raise ValueError(
            "El modelo devolvió un tensor con dimensiones inesperadas. Se esperaba un vector 1D."
        )

    probabilities = tf.nn.softmax(prediction_np).numpy()
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    return predicted_index, confidence


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    try:
        payload = request.get_json(force=True)
    except Exception as exc:  # pragma: no cover - Flask already logs this
        logger.warning("No se pudo parsear el JSON recibido: %s", exc)
        return jsonify({"error": "JSON inválido"}), 400

    sequence = payload.get("sequence") if isinstance(payload, dict) else None

    try:
        features = prepare_input(sequence)
        predicted_index, confidence = run_inference(features)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - logging unexpected errors
        logger.exception("Error al ejecutar la inferencia")
        return jsonify({"error": "Error interno del servidor"}), 500

    label = INDEX_TO_GESTURE.get(predicted_index, str(predicted_index))
    response = {"label": label, "index": predicted_index, "confidence": round(confidence, 4)}
    return jsonify(response)


@app.route("/health", methods=["GET"])
def healthcheck() -> Any:
    """Simple endpoint to verify that the service is running."""
    return jsonify({"status": "ok"})


def main() -> None:
    logger.info("Servidor de inferencia iniciado en %s:%s", LISTEN_HOST, LISTEN_PORT)
    app.run(host=LISTEN_HOST, port=LISTEN_PORT)


if __name__ == "__main__":
    main()
