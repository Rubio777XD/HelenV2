#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${PROJECT_ROOT}/.venv"

if ! command -v apt-get >/dev/null 2>&1; then
    echo "Este script requiere un sistema basado en Debian con apt-get." >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "No se encontró ${PYTHON_BIN}. Ajusta la variable de entorno PYTHON para apuntar al intérprete de Python 3.11 del sistema." >&2
    exit 1
fi

if [[ "${EUID}" -ne 0 ]]; then
    SUDO="sudo"
else
    SUDO=""
fi

timestamp() {
    date '+%Y%m%d-%H%M%S'
}

LOG_DIR="${PROJECT_ROOT}/reports/logs/pi"
mkdir -p "${LOG_DIR}"
SETUP_LOG="${LOG_DIR}/setup-$(timestamp).log"

exec > >(tee -a "${SETUP_LOG}") 2>&1

echo "[HELEN] Registrando instalación en ${SETUP_LOG}"

resolve_pkg() {
    for candidate in "$@"; do
        if apt-cache show "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

APT_PACKAGES=(
    libatlas-base-dev
    libopenblas-dev
    libportaudio2
    libjpeg-dev
    libtiff-dev
    python3-pip
    python3-venv
)

if pkg=$(resolve_pkg libcamera0.5 libcamera0); then
    APT_PACKAGES+=("${pkg}")
else
    echo "[HELEN] Advertencia: no se encontró un paquete libcamera compatible" >&2
fi

if pkg=$(resolve_pkg rpicam-apps-core libcamera-apps); then
    APT_PACKAGES+=("${pkg}")
fi

if pkg=$(resolve_pkg libavcodec-extra); then
    APT_PACKAGES+=("${pkg}")
fi
if pkg=$(resolve_pkg libavcodec-dev); then
    APT_PACKAGES+=("${pkg}")
fi
if pkg=$(resolve_pkg libavformat-dev); then
    APT_PACKAGES+=("${pkg}")
fi
if pkg=$(resolve_pkg libswscale-dev); then
    APT_PACKAGES+=("${pkg}")
fi

if pkg=$(resolve_pkg chromium-browser chromium); then
    APT_PACKAGES+=("${pkg}")
else
    echo "[HELEN] Advertencia: no se encontró un paquete de Chromium en los repositorios. Instálalo manualmente." >&2
fi

echo "[HELEN] Actualizando índices APT"
${SUDO} apt-get update

echo "[HELEN] Instalando dependencias del sistema"
${SUDO} apt-get install -y "${APT_PACKAGES[@]}"

echo "[HELEN] Preparando entorno virtual en ${VENV_DIR}"
if [[ -d "${VENV_DIR}" ]]; then
    echo "[HELEN] Limpiando entorno virtual previo"
    rm -rf "${VENV_DIR}"
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"

VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "No se pudo crear el entorno virtual en ${VENV_DIR}." >&2
    exit 1
fi

echo "[HELEN] Actualizando pip y herramientas base"
"${VENV_PIP}" install --upgrade pip wheel setuptools

echo "[HELEN] Instalando dependencias de Python para Raspberry Pi"
"${VENV_PIP}" install --no-cache-dir -r "${SCRIPT_DIR}/requirements-pi.txt"

cat <<'MSG'
Dependencias instaladas y entorno virtual configurado.
Activa el entorno con "source .venv/bin/activate" antes de ejecutar HELEN.
Recuerda habilitar la cámara con `sudo raspi-config` (Interfacing Options > Camera) y reiniciar el equipo si es la primera vez.
MSG

