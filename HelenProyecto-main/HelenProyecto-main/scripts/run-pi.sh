#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PORT="${HELEN_PORT:-5000}"
HOST="${HELEN_HOST:-0.0.0.0}"
LAUNCH_BROWSER=${HELEN_PI_LAUNCH_BROWSER:-0}
KIOSK_MODE=${HELEN_PI_KIOSK_MODE:-0}
BROWSER_BIN="${HELEN_PI_BROWSER_BINARY:-chromium-browser}"
BROWSER_URL="${HELEN_PI_BROWSER_URL:-http://localhost:${PORT}}"
BROWSER_FLAGS_RAW="${HELEN_PI_BROWSER_FLAGS:-}"
CAMERA_WIDTH="${HELEN_CAMERA_WIDTH:-}"
CAMERA_HEIGHT="${HELEN_CAMERA_HEIGHT:-}"

usage() {
    cat <<'USAGE'
Uso: scripts/run-pi.sh [opciones]

Opciones:
  --model-path RUTA       Ruta al SavedModel (por defecto busca gesture_model_* en data/models).
  --labels-path RUTA      Ruta a labels.json (opcional si vive junto al modelo).
  --camera-index N        Índice de cámara UVC (0 = /dev/video0).
  --frame-width PIXELES   Establece HELEN_CAMERA_WIDTH para la sesión actual.
  --frame-height PIXELES  Establece HELEN_CAMERA_HEIGHT para la sesión actual.
  --launch-browser        Abre Chromium apuntando a http://localhost:5000.
  --no-browser            Evita lanzar el navegador (comportamiento por defecto).
  --kiosk                 Usa flags de kiosk cuando se lanza el navegador.
  --windowed              Fuerza modo ventana normal.
  --browser-binary RUTA   Ruta al binario de Chromium/Chrome.
  --browser-url URL       URL a abrir (por defecto http://localhost:5000).
  -h, --help              Muestra esta ayuda.

También puedes definir HELEN_BACKEND_EXTRA_ARGS para pasar flags extra
(p.ej. "--confidence-threshold 0.8 --prediction-cooldown 0.6").
USAGE
}

MODEL_PATH="${HELEN_TF_MODEL_PATH:-}"
LABELS_PATH="${HELEN_TF_LABELS_PATH:-}"
CAMERA_INDEX="${HELEN_CAMERA_INDEX:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --labels-path)
            LABELS_PATH="$2"
            shift 2
            ;;
        --camera-index)
            CAMERA_INDEX="$2"
            shift 2
            ;;
        --frame-width)
            CAMERA_WIDTH="$2"
            shift 2
            ;;
        --frame-height)
            CAMERA_HEIGHT="$2"
            shift 2
            ;;
        --launch-browser)
            LAUNCH_BROWSER=1
            shift
            ;;
        --no-browser)
            LAUNCH_BROWSER=0
            shift
            ;;
        --kiosk)
            KIOSK_MODE=1
            shift
            ;;
        --windowed)
            KIOSK_MODE=0
            shift
            ;;
        --browser-binary)
            BROWSER_BIN="$2"
            shift 2
            ;;
        --browser-url)
            BROWSER_URL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Opción desconocida: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

log() {
    printf '[HELEN][run-pi] %s\n' "$*"
}

if [[ -x "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
fi

if [[ -x "${VENV_DIR}/bin/python" ]]; then
    PYTHON_BIN="${VENV_DIR}/bin/python"
elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    log "No se encontró un intérprete de Python. Ejecuta scripts/setup-pi.sh primero."
    exit 1
fi

find_default_model() {
    HELEN_PROJECT_ROOT="${PROJECT_ROOT}" "${PYTHON_BIN}" <<'PY'
from pathlib import Path
import os
import sys
root = Path(os.environ["HELEN_PROJECT_ROOT"]) / "Hellen_model_TF" / "video_gesture_model" / "data" / "models"
candidates = []
if root.exists():
    for path in sorted(root.glob("gesture_model_*")):
        if (path / "saved_model.pb").exists():
            candidates.append(path)
if not candidates:
    sys.exit(1)
print(candidates[-1])
PY
}

if [[ -z "${MODEL_PATH}" ]]; then
    if MODEL_PATH=$(find_default_model 2>/dev/null); then
        log "Usando modelo detectado en ${MODEL_PATH}"
    else
        log "No se encontró un SavedModel en data/models. Define HELEN_TF_MODEL_PATH o usa --model-path."
        exit 1
    fi
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    log "La ruta del modelo (${MODEL_PATH}) no existe."
    exit 1
fi

if [[ -z "${LABELS_PATH}" && -f "${MODEL_PATH}/labels.json" ]]; then
    LABELS_PATH="${MODEL_PATH}/labels.json"
fi

LOG_DIR="${PROJECT_ROOT}/reports/logs/pi"
mkdir -p "${LOG_DIR}"
RUN_ID="$(date '+%Y%m%d-%H%M%S')"
BACKEND_LOG="${LOG_DIR}/backend-${RUN_ID}.log"

export HELEN_TF_MODEL_PATH="${MODEL_PATH}"
export HELEN_CAMERA_INDEX="${CAMERA_INDEX}"
if [[ -n "${LABELS_PATH}" ]]; then
    export HELEN_TF_LABELS_PATH="${LABELS_PATH}"
fi
if [[ -n "${CAMERA_WIDTH}" ]]; then
    export HELEN_CAMERA_WIDTH="${CAMERA_WIDTH}"
fi
if [[ -n "${CAMERA_HEIGHT}" ]]; then
    export HELEN_CAMERA_HEIGHT="${CAMERA_HEIGHT}"
fi

BACKEND_CMD=(
    "${PYTHON_BIN}" -m backendHelen.server
    --host "${HOST}"
    --port "${PORT}"
    --camera-index "${CAMERA_INDEX}"
    --model-path "${MODEL_PATH}"
)

if [[ -n "${LABELS_PATH}" ]]; then
    BACKEND_CMD+=(--labels "${LABELS_PATH}")
fi

if [[ -n "${HELEN_BACKEND_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( ${HELEN_BACKEND_EXTRA_ARGS} )
    BACKEND_CMD+=("${EXTRA_ARGS[@]}")
fi

log "Comando del backend: ${BACKEND_CMD[*]}"
log "Los logs se almacenarán en ${BACKEND_LOG}"

touch "${BACKEND_LOG}"

cleanup() {
    local status=$?
    if [[ -n "${BROWSER_PID:-}" ]]; then
        kill "${BROWSER_PID}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${TAIL_PID:-}" ]]; then
        kill "${TAIL_PID}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${BACKEND_PID:-}" ]]; then
        kill "${BACKEND_PID}" >/dev/null 2>&1 || true
    fi
    wait >/dev/null 2>&1 || true
    return "$status"
}
trap cleanup EXIT SIGINT SIGTERM

"${BACKEND_CMD[@]}" >>"${BACKEND_LOG}" 2>&1 &
BACKEND_PID=$!

tail -n +1 -F "${BACKEND_LOG}" &
TAIL_PID=$!

launch_browser() {
    if (( LAUNCH_BROWSER == 0 )); then
        return 0
    fi
    if ! command -v "${BROWSER_BIN}" >/dev/null 2>&1; then
        log "No se encontró ${BROWSER_BIN}; no se lanzará el navegador."
        return 0
    fi

    local -a browser_cmd=("${BROWSER_BIN}" "${BROWSER_URL}")
    if (( KIOSK_MODE == 1 )); then
        browser_cmd+=(
            --kiosk --incognito --noerrdialogs --disable-session-crashed-bubble
            --autoplay-policy=no-user-gesture-required
        )
    else
        browser_cmd+=(--new-window)
    fi
    if [[ -n "${BROWSER_FLAGS_RAW}" ]]; then
        # shellcheck disable=SC2206
        EXTRA_BROWSER_FLAGS=( ${BROWSER_FLAGS_RAW} )
        browser_cmd+=("${EXTRA_BROWSER_FLAGS[@]}")
    fi

    log "Abriendo navegador (${browser_cmd[*]})"
    "${browser_cmd[@]}" >/dev/null 2>&1 &
    BROWSER_PID=$!
}

# Espera pequeña para que el backend abra el puerto antes del navegador.
sleep 2 || true
launch_browser

wait "${BACKEND_PID}"
BACKEND_STATUS=$?
log "Backend finalizado con código ${BACKEND_STATUS}."
exit "${BACKEND_STATUS}"
