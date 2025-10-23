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
    echo "No se encontró ${PYTHON_BIN}. Ajusta la variable PYTHON para apuntar al intérprete de Python 3.11 del sistema." >&2
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

readarray -t PY_INFO < <("${PYTHON_BIN}" - <<'PYCODE'
import platform
import sys
print(platform.python_version())
print(sys.version_info.major)
print(sys.version_info.minor)
print(sys.version_info.micro)
PYCODE
)

PY_VERSION="${PY_INFO[0]}"
PY_MAJOR="${PY_INFO[1]}"
PY_MINOR="${PY_INFO[2]}"
PY_PATCH="${PY_INFO[3]}"
ARCH="$(uname -m)"

echo "[HELEN] Python detectado: ${PY_VERSION} (${ARCH})"

if (( PY_MAJOR != 3 )); then
    echo "[HELEN] Se requiere Python 3.x para ejecutar HELEN en Raspberry Pi." >&2
    exit 1
fi

if (( PY_MINOR < 10 )); then
    echo "[HELEN] Python ${PY_MAJOR}.${PY_MINOR} es demasiado antiguo. Instala Python 3.11 desde los repositorios oficiales." >&2
    exit 1
fi

if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "arm64" && "${ARCH}" != "armv7l" && "${ARCH}" != "armv6l" ]]; then
    echo "[HELEN] Advertencia: arquitectura ${ARCH} no verificada. Los wheels oficiales están optimizados para ARM." >&2
fi

resolve_mediapipe_spec() {
    local major="$1"
    local minor="$2"
    local arch="$3"

    if (( major != 3 )); then
        echo "[HELEN] MediaPipe requiere Python 3.x" >&2
        return 1
    fi

    local spec=""
    local extra_index=""

    case "${arch}" in
        armv7l|armv6l)
            if (( minor >= 11 )); then
                echo "[HELEN] Error: MediaPipe para ${arch} solo publica ruedas hasta Python 3.10. Instala Raspberry Pi OS de 64 bits o Python 3.10 (32 bits)." >&2
                exit 1
            fi
            spec="mediapipe-rpi4==0.10.9"
            extra_index="https://google-coral.github.io/py-repo/"
            echo "[HELEN] Arquitectura ${arch} detectada. Se utilizará paquete especializado ${spec}." >&2
            ;;
        aarch64|arm64)
            if (( minor >= 12 )); then
                echo "[HELEN] MediaPipe no publica ruedas oficiales para Python ${major}.${minor} en ARM64. Instala Python 3.11 (sudo apt install python3.11-full) y reejecuta este script." >&2
                exit 1
            fi
            case "${minor}" in
                11)
                    spec="mediapipe==0.10.21"
                    ;;
                10)
                    echo "[HELEN] Advertencia: Python 3.10 detectado. Se forzará mediapipe==0.10.11 (última versión validada)." >&2
                    spec="mediapipe==0.10.11"
                    ;;
                *)
                    echo "[HELEN] Advertencia: Python ${major}.${minor} no está validado oficialmente. Se intentará mediapipe==0.10.21." >&2
                    spec="mediapipe==0.10.21"
                    ;;
            esac
            ;;
        *)
            echo "[HELEN] Advertencia: arquitectura ${arch} no verificada. Se intentará instalar mediapipe==0.10.21 desde PyPI." >&2
            spec="mediapipe==0.10.21"
            ;;
    esac

    printf '%s\n%s\n' "${spec}" "${extra_index}"
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
"${SUDO}" apt-get update

echo "[HELEN] Instalando dependencias del sistema"
"${SUDO}" apt-get install -y "${APT_PACKAGES[@]}"

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

NUMPY_SPEC="numpy==1.26.4"
OPENCV_SPEC="opencv-python==4.9.0.80"
readarray -t MEDIAPIPE_INFO < <(resolve_mediapipe_spec "${PY_MAJOR}" "${PY_MINOR}" "${ARCH}")
MEDIAPIPE_SPEC="${MEDIAPIPE_INFO[0]}"
MEDIAPIPE_EXTRA_INDEX="${MEDIAPIPE_INFO[1]:-}"

if [[ -z "${MEDIAPIPE_SPEC}" ]]; then
    echo "[HELEN] No se pudo determinar la versión correcta de MediaPipe para este entorno." >&2
    exit 1
fi

echo "[HELEN] Instalando stack numérico (${NUMPY_SPEC} ${OPENCV_SPEC})"
"${VENV_PIP}" install --no-cache-dir "${NUMPY_SPEC}" "${OPENCV_SPEC}"

echo "[HELEN] Instalando MediaPipe (${MEDIAPIPE_SPEC})"
if [[ -n "${MEDIAPIPE_EXTRA_INDEX}" ]]; then
    echo "[HELEN] Repositorio adicional: ${MEDIAPIPE_EXTRA_INDEX}"
fi
MEDIAPIPE_INSTALL_CMD=("${VENV_PIP}" install --no-cache-dir)
if [[ -n "${MEDIAPIPE_EXTRA_INDEX}" ]]; then
    MEDIAPIPE_INSTALL_CMD+=("--extra-index-url" "${MEDIAPIPE_EXTRA_INDEX}")
fi
MEDIAPIPE_INSTALL_CMD+=("${MEDIAPIPE_SPEC}")
"${MEDIAPIPE_INSTALL_CMD[@]}"

echo "[HELEN] Instalando dependencias de Python restantes"
"${VENV_PIP}" install --no-cache-dir -r "${SCRIPT_DIR}/requirements-pi.txt"

if ! "${VENV_PIP}" check; then
    echo "[HELEN] Advertencia: 'pip check' reportó inconsistencias. Revisa la salida anterior." >&2
fi

export HELEN_PI_LOG_DIR="${LOG_DIR}"
export HELEN_MEDIAPIPE_SPEC="${MEDIAPIPE_SPEC}"
export HELEN_MEDIAPIPE_EXTRA_INDEX="${MEDIAPIPE_EXTRA_INDEX}"
"${VENV_PYTHON}" - <<'PYCODE'
import json
import os
import platform
import sys
import time
from pathlib import Path

info = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python_version": platform.python_version(),
    "arch": platform.machine(),
    "os_release": "",
    "mediapipe": {"status": "missing"},
    "opencv": {"status": "missing"},
    "numpy": {"status": "missing"},
}

os_release_path = Path("/etc/os-release")
if os_release_path.exists():
    content = {}
    for line in os_release_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        content[key] = value.strip().strip('"')
    name = content.get("PRETTY_NAME") or content.get("NAME")
    version = content.get("VERSION_ID")
    info["os_release"] = f"{name} {version}".strip()

mediapipe_ok = False
try:
    import mediapipe as mp  # type: ignore

    version = getattr(mp, "__version__", "unknown")
    with mp.solutions.hands.Hands() as hands:  # noqa: F841 - smoke test
        pass
    location = Path(getattr(mp, "__file__", "")).resolve()
    info["mediapipe"] = {
        "status": "ok",
        "version": version,
        "location": str(location),
        "spec": os.environ.get("HELEN_MEDIAPIPE_SPEC", ""),
        "extra_index": os.environ.get("HELEN_MEDIAPIPE_EXTRA_INDEX", ""),
    }
    mediapipe_ok = True
except Exception as error:  # pragma: no cover - depende del entorno
    info["mediapipe"] = {
        "status": "error",
        "message": str(error),
    }

try:
    import cv2  # type: ignore

    version = getattr(cv2, "__version__", "unknown")
    build_info = ""
    try:
        build_info = cv2.getBuildInformation()
    except Exception as error:  # pragma: no cover - depende del entorno
        build_info = f"<no build info>: {error}"
    info["opencv"] = {
        "status": "ok",
        "version": version,
        "gstreamer": "YES" if "GStreamer:                   YES" in build_info else "NO",
        "v4l2": "YES" if "Video I/O:" in build_info and "V4L/V4L2:                  YES" in build_info else "NO",
    }
except Exception as error:  # pragma: no cover - depende del entorno
    info["opencv"] = {
        "status": "error",
        "message": str(error),
    }

try:
    import numpy  # type: ignore

    info["numpy"] = {
        "status": "ok",
        "version": numpy.__version__,
    }
except Exception as error:  # pragma: no cover - depende del entorno
    info["numpy"] = {
        "status": "error",
        "message": str(error),
    }

log_dir = Path(os.environ.get("HELEN_PI_LOG_DIR", "."))
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"vision-stack-{time.strftime('%Y%m%d-%H%M%S')}.json"
log_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[HELEN] Snapshot de dependencias guardado en {log_path}")

if mediapipe_ok:
    mp_info = info["mediapipe"]
    print(
        "[HELEN] MediaPipe validado: {version} ({location})".format(
            version=mp_info.get("version", "unknown"),
            location=mp_info.get("location", "<desconocido>"),
        )
    )
else:
    print("[HELEN] Error: MediaPipe no se pudo importar. Revisa el log anterior.", file=sys.stderr)
    sys.exit(1)
PYCODE

echo "[HELEN] Resumen de paquetes instalados"
"${VENV_PYTHON}" - <<'PYCODE'
from pathlib import Path
import mediapipe
import cv2
import numpy

print(f"[HELEN] mediapipe {mediapipe.__version__} -> {Path(mediapipe.__file__).resolve()}")
print(f"[HELEN] opencv-python {cv2.__version__}")
print(f"[HELEN] numpy {numpy.__version__}")
PYCODE

echo "Dependencias instaladas y entorno virtual configurado."
echo "Activa el entorno con 'source .venv/bin/activate' antes de ejecutar HELEN."
echo "Los detalles de la instalación se encuentran en ${SETUP_LOG}."
