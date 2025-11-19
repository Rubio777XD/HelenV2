#!/usr/bin/env bash
# Arranca HELEN con el backend LSTM en Raspberry Pi 5
# Uso: bash scripts/run_pi5_lstm.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .venv/bin/activate ]; then
  echo "No se encontró el entorno virtual en .venv. Créalo con 'python3 -m venv .venv'" >&2
  exit 1
fi

source .venv/bin/activate
export HELEN_MODEL_BACKEND=lstm
python -m backendHelen.server &
SERVER_PID=$!

# Lanzar Chromium en modo kiosk hacia el frontend
if command -v chromium-browser >/dev/null 2>&1; then
  chromium-browser --kiosk http://localhost:5000 &
elif command -v chromium >/dev/null 2>&1; then
  chromium --kiosk http://localhost:5000 &
else
  echo "Chromium no está instalado; abre manualmente http://localhost:5000" >&2
fi

wait ${SERVER_PID}
