"""Configuration constants for the video-based gesture recognition pipeline.

This module centralises local paths and AWS placeholders so all scripts share
consistent defaults. Adjust the values here before running any of the scripts.

Este módulo define constantes reutilizables para que todos los scripts
compartan las mismas rutas y parámetros por defecto dentro del pipeline.
"""

from pathlib import Path

# Root directory for generated assets (videos, extracted features, models).
# Directorio base donde se generarán y almacenarán todos los artefactos.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "raw_videos"
FRAMES_DIR = DATA_DIR / "frames"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"

# Default recording options.
# Parámetros por defecto utilizados durante la captura de los videos.
FPS = 24
CLIP_DURATION = 4  # seconds per recorded sample
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# TensorFlow training defaults.
# Configuración base para definir el tamaño de las secuencias y vectores.
SEQUENCE_LENGTH = FPS * CLIP_DURATION
NUM_HAND_LANDMARKS = 21
LANDMARK_DIM = 3
MAX_HANDS = 2
FEATURE_SIZE = NUM_HAND_LANDMARKS * LANDMARK_DIM * MAX_HANDS

# AWS integration placeholders. These values are not used until credentials are
# configured, but the keys are defined here to make the connection step trivial.
# Valores de ejemplo para S3 que se reemplazarán al integrar las credenciales reales.
AWS_REGION = "us-east-1"
S3_DATASET_BUCKET = "your-gesture-dataset-bucket"
S3_MODELS_BUCKET = "your-gesture-models-bucket"
S3_DATASET_PREFIX = "datasets/gesture_videos"
S3_MODEL_PREFIX = "models/gesture_videos"

# Ensure the required directories exist.
# Se crean todas las carpetas necesarias si aún no existen en el sistema de archivos.
for path in [DATA_DIR, VIDEOS_DIR, FRAMES_DIR, FEATURES_DIR, MODELS_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
    