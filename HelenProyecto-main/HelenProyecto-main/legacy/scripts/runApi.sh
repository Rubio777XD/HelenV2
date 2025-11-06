#!/bin/sh
set -eu

BASE_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
cd "$BASE_DIR/backendHelen"

python -m backendHelen.server "$@"