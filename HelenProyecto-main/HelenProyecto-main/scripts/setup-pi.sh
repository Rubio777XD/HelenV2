#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
SKIP_APT=0

usage() {
    cat <<'USAGE'
Uso: scripts/setup-pi.sh [opciones]

Opciones:
  --skip-apt           Omite la instalación de paquetes APT (cuando ya están presentes).
  --python-bin RUTA    Python 3.10 a usar para crear el entorno virtual (por defecto python3.10).
  --venv-dir RUTA      Ubicación del entorno virtual (.venv por defecto).
  -h, --help           Muestra esta ayuda.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-apt)
            SKIP_APT=1
            shift
            ;;
        --python-bin)
            if [[ $# -lt 2 ]]; then
                echo "Falta el valor para --python-bin" >&2
                exit 1
            fi
            PYTHON_BIN="$2"
            shift 2
            ;;
        --venv-dir)
            if [[ $# -lt 2 ]]; then
                echo "Falta el valor para --venv-dir" >&2
                exit 1
            fi
            VENV_DIR="$2"
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
    printf '[HELEN][setup-pi] %s\n' "$*"
}

ensure_python() {
    if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        return 0
    fi
    log "No se encontró ${PYTHON_BIN}. Instala python3.10 y vuelve a ejecutar este script."
    return 1
}

install_apt_packages() {
    if (( SKIP_APT == 1 )); then
        log "Se omitió la instalación APT por --skip-apt."
        return 0
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        log "apt-get no está disponible; omitiendo instalación de paquetes del sistema."
        return 0
    fi

    local sudo_cmd=()
    if (( EUID != 0 )); then
        if command -v sudo >/dev/null 2>&1; then
            sudo_cmd=(sudo)
        else
            log "Debes ejecutar este script como root o tener sudo para instalar dependencias."
            return 1
        fi
    fi

    local -a base_packages=(
        python3.10 python3.10-venv python3-pip git ffmpeg v4l-utils
        libatlas-base-dev libopenblas-dev libjpeg-dev libtiff5 zlib1g-dev
        gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
        gstreamer1.0-plugins-bad libcamera-apps chromium-browser
    )

    log "Actualizando índices APT (puede tardar unos minutos en Raspberry Pi 5)..."
    "${sudo_cmd[@]}" apt-get update -y

    local -a available_packages=()
    local pkg
    for pkg in "${base_packages[@]}"; do
        if apt-cache show "$pkg" >/dev/null 2>&1; then
            available_packages+=("$pkg")
        else
            log "[AVISO] El paquete $pkg no está en los repositorios actuales; se omite."
        fi
    done

    if (( ${#available_packages[@]} == 0 )); then
        log "No hay paquetes disponibles para instalar."
        return 0
    fi

    log "Instalando dependencias del sistema: ${available_packages[*]}"
    "${sudo_cmd[@]}" apt-get install -y --no-install-recommends "${available_packages[@]}"
}

create_venv() {
    if [[ -d "${VENV_DIR}" ]]; then
        log "El entorno virtual ya existe en ${VENV_DIR}."
        return 0
    fi
    log "Creando entorno virtual en ${VENV_DIR} usando ${PYTHON_BIN}..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
}

install_python_requirements() {
    local pip_bin="${VENV_DIR}/bin/pip"
    if [[ ! -x "${pip_bin}" ]]; then
        log "pip no se encontró en ${pip_bin}. ¿Se creó correctamente el entorno virtual?"
        return 1
    fi

    log "Actualizando pip/setuptools/wheel dentro del entorno..."
    "${pip_bin}" install --upgrade pip setuptools wheel

    local -a req_files=(
        "${PROJECT_ROOT}/requirements.txt"
        "${PROJECT_ROOT}/Hellen_model_TF/video_gesture_model/requirements.txt"
    )

    local req
    for req in "${req_files[@]}"; do
        if [[ -f "${req}" ]]; then
            log "Instalando dependencias desde ${req}"
            "${pip_bin}" install -r "${req}"
        else
            log "[AVISO] No se encontró ${req}; se omite."
        fi
    done
}

main() {
    install_apt_packages
    ensure_python
    create_venv
    install_python_requirements
    log "Setup completado. Usa ${PROJECT_ROOT}/scripts/run-pi.sh para iniciar HELEN."
}

main "$@"
