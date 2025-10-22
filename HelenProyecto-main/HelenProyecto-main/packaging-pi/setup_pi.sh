#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

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
    libtiff5
    libcamera0
    libcamera-apps
    python3-pip
    python3-venv
)

if pkg=$(resolve_pkg libavformat60 libavformat59 libavformat58); then
    APT_PACKAGES+=("${pkg}")
fi
if pkg=$(resolve_pkg libavcodec60 libavcodec59 libavcodec58); then
    APT_PACKAGES+=("${pkg}")
fi
if pkg=$(resolve_pkg libswscale7 libswscale6 libswscale5); then
    APT_PACKAGES+=("${pkg}")
fi

if pkg=$(resolve_pkg chromium chromium-browser); then
    APT_PACKAGES+=("${pkg}")
else
    echo "Advertencia: no se encontró un paquete de Chromium en los repositorios. Instálalo manualmente." >&2
fi

${SUDO} apt-get update
${SUDO} apt-get install -y "${APT_PACKAGES[@]}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install --upgrade wheel setuptools
"${PYTHON_BIN}" -m pip install --no-cache-dir -r "${SCRIPT_DIR}/requirements-pi.txt"

cat <<'MSG'
Dependencias instaladas.
Recuerda habilitar la cámara con `sudo raspi-config` (Interfacing Options > Camera) y reiniciar el equipo si es la primera vez.
MSG

