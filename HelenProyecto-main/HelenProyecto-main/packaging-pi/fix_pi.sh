#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements-pi.txt"
DEFAULT_PIP="${PROJECT_ROOT}/.venv/bin/pip"
RUN_AFTER_SETUP=0
FORCE_SYSTEM_PIP=0

usage() {
    cat <<'USAGE'
Uso: packaging-pi/fix_pi.sh [opciones]

Opciones:
  --run           Ejecuta run_pi.sh (sin UI) tras instalar dependencias.
  --system-pip    Fuerza el uso del pip del sistema (fuera de .venv).
  -h, --help      Muestra esta ayuda.

El script instala numpy, OpenCV, MediaPipe y el resto de requirements
necesarios para Raspberry Pi. Si el entorno aplica PEP 668, se reintenta
automáticamente con --break-system-packages.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            RUN_AFTER_SETUP=1
            shift
            ;;
        --system-pip)
            FORCE_SYSTEM_PIP=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Opción desconocida: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -x "${DEFAULT_PIP}" && "${FORCE_SYSTEM_PIP}" -eq 0 ]]; then
    PIP_BIN="${DEFAULT_PIP}"
    echo "[HELEN] Usando entorno virtual ${PROJECT_ROOT}/.venv"
else
    if ! command -v pip3 >/dev/null 2>&1; then
        echo "[HELEN] No se encontró pip3 en el sistema." >&2
        exit 1
    fi
    PIP_BIN="$(command -v pip3)"
    echo "[HELEN] Usando pip del sistema (${PIP_BIN})"
fi

pip_retry() {
    local -a base=("${PIP_BIN}" install --no-cache-dir)
    local -a args=("$@")

    if "${base[@]}" "${args[@]}"; then
        return 0
    fi

    echo "[HELEN] Reintentando ${args[*]} con --break-system-packages" >&2
    "${base[@]}" --break-system-packages "${args[@]}"
}

PACKAGES=(
    "numpy==1.26.4"
    "opencv-python==4.9.0.80"
    "mediapipe==0.10.18"
)

for spec in "${PACKAGES[@]}"; do
    echo "[HELEN] Instalando ${spec}"
    pip_retry "${spec}"
done

echo "[HELEN] Instalando requirements adicionales (${REQ_FILE})"
pip_retry -r "${REQ_FILE}"

if command -v "${PIP_BIN%/pip}/python" >/dev/null 2>&1; then
    PYTHON_BIN="${PIP_BIN%/pip}/python"
else
    PYTHON_BIN="$(dirname "${PIP_BIN}")/python3"
fi

if [[ -x "${PYTHON_BIN}" ]]; then
    echo "[HELEN] Verificando importaciones de MediaPipe y OpenCV"
    "${PYTHON_BIN}" - <<'PY'
import cv2
import mediapipe
import numpy

print(f"OpenCV {cv2.__version__}")
print(f"MediaPipe {mediapipe.__version__}")
print(f"NumPy {numpy.__version__}")
PY
fi

if [[ "${RUN_AFTER_SETUP}" -eq 1 ]]; then
    echo "[HELEN] Ejecutando backend en modo headless para prueba rápida"
    HELEN_NO_UI=1 "${SCRIPT_DIR}/run_pi.sh"
fi
