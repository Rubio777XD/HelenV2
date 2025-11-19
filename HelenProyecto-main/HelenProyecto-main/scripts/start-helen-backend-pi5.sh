#!/usr/bin/env bash
# Arranca el backend de HELEN en Raspberry Pi 5 usando el modelo LSTM.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/reports/logs/pi"
LOG_FILE="$LOG_DIR/backend-$(date +%Y%m%d-%H%M%S).log"

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"

if [[ -d "$ROOT_DIR/.venv" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.venv/bin/activate"
fi

export HELEN_MODEL_BACKEND=${HELEN_MODEL_BACKEND:-lstm}

echo "[HELEN] Backend TensorFlow iniciado a las $(date). Log: $LOG_FILE"
exec python -m backendHelen.server "$@" >>"$LOG_FILE" 2>&1
