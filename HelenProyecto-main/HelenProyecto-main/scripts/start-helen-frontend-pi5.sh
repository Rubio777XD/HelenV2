#!/usr/bin/env bash
# Lanza Chromium en modo kiosco apuntando al frontend de HELEN.

set -euo pipefail
URL=${1:-http://localhost:5000}

chromium-browser \
  --kiosk \
  --incognito \
  --noerrdialogs \
  --disable-infobars \
  --disable-translate \
  --autoplay-policy=no-user-gesture-required \
  "$URL"
