#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
PORT="${HELEN_PORT:-5000}"

MODEL_FILE="/proc/device-tree/model"
if [[ -z "${DEVICE_MODEL:-}" && -r "${MODEL_FILE}" ]]; then
    DEVICE_MODEL="$(tr -d '\0' < "${MODEL_FILE}" 2>/dev/null | xargs)"
fi

case "${DEVICE_MODEL,,}" in
    *"raspberry pi 5"*)
        DEFAULT_POLL_INTERVAL="0.040"
        ;;
    *"raspberry pi 4"*)
        DEFAULT_POLL_INTERVAL="0.050"
        ;;
    *)
        DEFAULT_POLL_INTERVAL=""
        ;;
endcase

POLL_INTERVAL="${POLL_INTERVAL:-${DEFAULT_POLL_INTERVAL}}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "No se encontró ${PYTHON_BIN}. Ajusta la variable PYTHON antes de ejecutar este script." >&2
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/reports/logs"
BACKEND_LOG="${PROJECT_ROOT}/reports/logs/backend-pi.log"
CHROMIUM_LOG="${PROJECT_ROOT}/reports/logs/chromium.log"

echo "Iniciando backend de HELEN (logs en ${BACKEND_LOG})"
BACKEND_CMD=("${PYTHON_BIN}" -m backendHelen.server --host 0.0.0.0 --port "${PORT}")
if [[ -n "${POLL_INTERVAL}" ]]; then
    BACKEND_CMD+=(--poll-interval "${POLL_INTERVAL}")
fi
if [[ -n "${HELEN_CAMERA_INDEX:-}" ]]; then
    BACKEND_CMD+=(--camera-index "${HELEN_CAMERA_INDEX}")
fi
if [[ -n "${HELEN_BACKEND_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    BACKEND_CMD+=(${HELEN_BACKEND_EXTRA_ARGS})
fi

"${BACKEND_CMD[@]}" >>"${BACKEND_LOG}" 2>&1 &
BACKEND_PID=$!

cleanup() {
    if [[ -n "${CHROMIUM_PID:-}" ]]; then
        kill "${CHROMIUM_PID}" 2>/dev/null || true
        wait "${CHROMIUM_PID}" 2>/dev/null || true
    fi
    if [[ -n "${BACKEND_PID:-}" ]]; then
        kill "${BACKEND_PID}" 2>/dev/null || true
        wait "${BACKEND_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Espera breve para que el backend exponga /health antes de abrir Chromium.
sleep 4

if [[ -n "${HELEN_NO_UI:-}" ]]; then
    echo "Variable HELEN_NO_UI detectada; se omitirá el kiosko."
    wait "${BACKEND_PID}"
    exit 0
fi

for candidate in chromium chromium-browser; do
    if command -v "${candidate}" >/dev/null 2>&1; then
        CHROMIUM_BIN="$(command -v "${candidate}")"
        break
    fi
done

if [[ -z "${CHROMIUM_BIN:-}" ]]; then
    echo "No se encontró Chromium en el PATH. El backend seguirá activo sin kiosko." >&2
    wait "${BACKEND_PID}"
    exit 0
fi

export DISPLAY="${DISPLAY:-:0}"
if [[ -z "${XAUTHORITY:-}" && -d "/home/${USER:-pi}" ]]; then
    export XAUTHORITY="/home/${USER:-pi}/.Xauthority"
fi

if command -v xset >/dev/null 2>&1; then
    xset s off || true
    xset -dpms || true
    xset s noblank || true
fi

CHROMIUM_FLAGS=(
    --noerrdialogs
    --disable-session-crashed-bubble
    --disable-component-update
    --kiosk
    --app="http://localhost:${PORT}"
    --incognito
    --check-for-update-interval=31536000
    --no-first-run
    --disable-pinch
    --overscroll-history-navigation=0
    --autoplay-policy=no-user-gesture-required
)

echo "Lanzando Chromium en modo kiosko (logs en ${CHROMIUM_LOG})"
"${CHROMIUM_BIN}" "${CHROMIUM_FLAGS[@]}" >>"${CHROMIUM_LOG}" 2>&1 &
CHROMIUM_PID=$!

wait "${BACKEND_PID}"

