#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PORT="${HELEN_PORT:-5000}"
CAMERA_BOOTSTRAP_LOG="${PROJECT_ROOT}/reports/logs/pi/camera-bootstrap.log"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="${PYTHON}"
elif [[ -x "${DEFAULT_VENV_PYTHON}" ]]; then
    PYTHON_BIN="${DEFAULT_VENV_PYTHON}"
else
    PYTHON_BIN="python3"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "No se encontró ${PYTHON_BIN}. Ajusta la variable PYTHON o ejecuta setup_pi.sh primero." >&2
    exit 1
fi

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
esac

POLL_INTERVAL="${POLL_INTERVAL:-${DEFAULT_POLL_INTERVAL}}"

LOG_DIR="${PROJECT_ROOT}/reports/logs/pi"
mkdir -p "${LOG_DIR}"
RUN_ID="$(date '+%Y%m%d-%H%M%S')"
BACKEND_LOG="${LOG_DIR}/backend-${RUN_ID}.log"
CHROMIUM_LOG="${LOG_DIR}/chromium-${RUN_ID}.log"
touch "${CAMERA_BOOTSTRAP_LOG}"

if [[ -z "${OPENCV_VIDEOIO_PRIORITY_LIST:-}" ]]; then
    export OPENCV_VIDEOIO_PRIORITY_LIST="GSTREAMER,V4L2"
fi

GST_SEARCH_PATHS=(
    "/usr/lib/aarch64-linux-gnu/gstreamer-1.0"
    "/usr/lib/arm-linux-gnueabihf/gstreamer-1.0"
    "/usr/local/lib/gstreamer-1.0"
)

if [[ -z "${GST_PLUGIN_PATH:-}" ]]; then
    for candidate in "${GST_SEARCH_PATHS[@]}"; do
        if [[ -d "${candidate}" ]]; then
            if [[ -z "${GST_PLUGIN_PATH:-}" ]]; then
                GST_PLUGIN_PATH="${candidate}"
            else
                GST_PLUGIN_PATH+=":${candidate}"
            fi
        fi
    done
    if [[ -n "${GST_PLUGIN_PATH:-}" ]]; then
        export GST_PLUGIN_PATH
    fi
fi

if [[ -z "${GST_REGISTRY_1_0:-}" ]]; then
    export GST_REGISTRY_1_0="${PROJECT_ROOT}/.cache/gstreamer/gstreamer-registry.bin"
fi

if [[ -n "${GST_REGISTRY_1_0:-}" ]]; then
    mkdir -p "$(dirname "${GST_REGISTRY_1_0}")"
fi

if [[ -z "${LIBCAMERA_RPI_TUNING_FILE:-}" ]]; then
    for candidate in /usr/share/libcamera/ipa/rpi/*/imx*.json; do
        if [[ -r "${candidate}" ]]; then
            export LIBCAMERA_RPI_TUNING_FILE="${candidate}"
            break
        fi
    done
fi

echo "[HELEN] Los logs se guardarán en ${LOG_DIR}"

log_camera_stack() {
    local picamera_output
    {
        echo "[HELEN] ===== Verificación del stack de cámara (${RUN_ID}) ====="
        for tool in rpicam-hello rpicam-vid libcamera-hello; do
            if command -v "${tool}" >/dev/null 2>&1; then
                version_output="$("${tool}" --version 2>&1 | head -n 1 | tr -d '\r')"
                if [[ -n "${version_output}" ]]; then
                    echo "[HELEN] ${tool}: ${version_output}"
                else
                    echo "[HELEN] ${tool}: disponible"
                fi
            else
                echo "[HELEN] ${tool}: no encontrado"
            fi
        done
        if picamera_output="$(${PYTHON_BIN} - <<'PY' 2>&1)"; then
            printf '%s\n' "${picamera_output}"
        else
            echo "[HELEN] picamera2: módulo no disponible"
            if [[ -n "${picamera_output}" ]]; then
                printf '%s\n' "${picamera_output}"
            fi
        fi
PY
from importlib import util
spec = util.find_spec("picamera2")
if spec is None:
    raise ImportError("picamera2 module not found")
from picamera2 import Picamera2  # noqa: F401  # type: ignore
print("[HELEN] picamera2: módulo detectado (%s)" % (spec.origin,))
PY
    } >>"${CAMERA_BOOTSTRAP_LOG}" 2>&1
}

log_camera_stack

bootstrap_camera() {
    local result_json
    local tmp_json

    if [[ -n "${HELEN_CAMERA_INDEX:-}" ]]; then
        echo "[HELEN] HELEN_CAMERA_INDEX predefinido (${HELEN_CAMERA_INDEX}); se omite auto-detección." | tee -a "${CAMERA_BOOTSTRAP_LOG}"
        return 0
    fi

    if [[ -n "${HELEN_SKIP_CAMERA_BOOTSTRAP:-}" ]]; then
        echo "[HELEN] HELEN_SKIP_CAMERA_BOOTSTRAP activo; no se ejecutará camera_probe.ensure_camera_selection." | tee -a "${CAMERA_BOOTSTRAP_LOG}"
        return 0
    fi

    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "[HELEN] No es posible inicializar la cámara: ${PYTHON_BIN} no está disponible." | tee -a "${CAMERA_BOOTSTRAP_LOG}" >&2
        return 1
    fi

    tmp_json="${LOG_DIR}/camera-selection-${RUN_ID}.json"
    result_json="$(${PYTHON_BIN} <<'PY'
import json
import sys

try:
    from backendHelen import camera_probe
except Exception as exc:  # pragma: no cover - depende del entorno
    print(json.dumps({"status": "error", "message": f"camera_probe import failed: {exc}"}))
    sys.exit(1)

selection = camera_probe.ensure_camera_selection(force=True)
if not selection:
    print(json.dumps({"status": "error", "message": "no camera selection"}))
    sys.exit(2)

payload = selection.to_dict()
payload["status"] = "ok"
print(json.dumps(payload))
PY
)"

    if [[ "${result_json}" == *'"status": "ok"'* ]]; then
        printf '%s\n' "${result_json}" | tee "${tmp_json}" >>"${CAMERA_BOOTSTRAP_LOG}"
        echo "[HELEN] Cámara detectada (bootstrap cache generado)." | tee -a "${CAMERA_BOOTSTRAP_LOG}"
        return 0
    fi

    if [[ -z "${result_json}" ]]; then
        result_json='{"status": "error", "message": "unknown"}'
    fi
    printf '%s\n' "${result_json}" | tee -a "${CAMERA_BOOTSTRAP_LOG}" >/dev/null
    echo "[HELEN] No se pudo validar una cámara física; el backend podría activar el flujo sintético." | tee -a "${CAMERA_BOOTSTRAP_LOG}" >&2
    return 1
}

bootstrap_camera || true

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

wait_for_backend() {
    local timeout="${HELEN_BACKEND_WAIT:-90}"
    local deadline=$((SECONDS + timeout))
    local health_url="http://127.0.0.1:${PORT}/health"

    while (( SECONDS < deadline )); do
        if command -v curl >/dev/null 2>&1; then
            if curl --silent --max-time 2 --fail "${health_url}" >/dev/null; then
                return 0
            fi
        elif command -v wget >/dev/null 2>&1; then
            if wget -q -O- --timeout=2 "${health_url}" >/dev/null; then
                return 0
            fi
        else
            if HEALTH_URL="${health_url}" "${PYTHON_BIN}" - <<'PY'
import os
import sys
import urllib.request

url = os.environ.get("HEALTH_URL")
try:
    with urllib.request.urlopen(url, timeout=2):
        pass
except Exception:
    sys.exit(1)
PY
            then
                return 0
            fi
        fi

        sleep 2
    done

    return 1
}

if ! wait_for_backend; then
    echo "[HELEN] El backend no respondió en el tiempo esperado." >&2
    wait "${BACKEND_PID}"
    exit 1
fi

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

