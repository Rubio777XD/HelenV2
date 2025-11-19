#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# El script principal vive junto al resto de herramientas de Raspberry Pi dentro de
# legacy/ para mantener compatibilidad con instalaciones existentes.
exec "${PROJECT_ROOT}/legacy/packaging/linux-rpi/setup_pi.sh" "$@"
