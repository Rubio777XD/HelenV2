"""Configuration helpers for the frontend bridge service."""
from __future__ import annotations

from pathlib import Path

# Directory where trained models are stored by default.
DEFAULT_MODELS_DIR = Path("video_gesture_model/data/models")

# Default confidence threshold required before broadcasting a gesture.
DEFAULT_CONFIDENCE_THRESHOLD = 0.75

# Number of frames that must accumulate before pushing a new prediction to
# subscribers. This mirrors ``config.SEQUENCE_LENGTH`` from the model package but
# can be overridden if needed for experimentation.
DEFAULT_SEQUENCE_LENGTH = None  # ``None`` = use model config at runtime.

# Camera index used by OpenCV.
DEFAULT_CAMERA_INDEX = 0

# Socket.IO namespace for gesture events.
SOCKETIO_NAMESPACE = "/gestures"

# Event name emitted over Socket.IO when a new gesture is detected.
SOCKETIO_EVENT = "gesture_prediction"

# Event source retry interval in milliseconds for SSE clients.
SSE_RETRY_MS = 2_000
