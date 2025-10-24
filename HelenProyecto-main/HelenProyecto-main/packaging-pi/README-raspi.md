# HELEN en Raspberry Pi 4/5 (Raspberry Pi OS 64-bit legacy)

Esta guía documenta el flujo clásico basado en kiosko para Raspberry Pi 4 y las instalaciones existentes de Pi 5 que siguen usando el modo pantalla completa. La versión optimizada para Pi 5 con ventana normal se detalla en [README-PI.md](README-PI.md). Ambas comparten scripts y requisitos, pero esta guía mantiene los parámetros conservadores y las notas de migración.

## 1. Requisitos

### Hardware

- Raspberry Pi 4 Model B (4 GB o más) o Raspberry Pi 5.
- Fuente oficial de 5V/3A (Pi 4) u 5V/5A (Pi 5).
- Tarjeta microSD UHS-I (32 GB mínimo, 64 GB recomendado).
- Cámara UVC o módulo oficial compatible con el stack `rpicam`.
- Monitor/pantalla HDMI más teclado/ratón para la configuración inicial.

### Software

- Raspberry Pi OS 64-bit (Bookworm) actualizado (`sudo apt full-upgrade`).
- Python 3.11 provisto por el sistema.
- Repositorio clonado en `${HOME}/HelenV2/HelenProyecto-main/HelenProyecto-main`.

> **Dataset**: copia `Hellen_model_RN/data.pickle` a la ruta anterior antes de ejecutar el backend para evitar el fallback al dataset legado `data1.pickle`.

## 2. Configuración inicial

1. Actualiza el sistema y reinicia:
   ```bash
   sudo apt update
   sudo apt full-upgrade
   sudo reboot
   ```
2. Habilita la cámara y desactiva el ahorro de pantalla:
   ```bash
   sudo raspi-config
   ```
   - *Interface Options → Camera → Enable*.
   - *Display Options → Screen Blanking → Disable*.
   - *System Options → Boot / Auto Login → Desktop Autologin* (para kiosko).
3. Verifica el módulo de cámara:
   ```bash
   rpicam-hello -t 5000
   ```
   y comprueba que `gst-launch-1.0 libcamerasrc ...` también produce imagen si necesitas validar la ruta GStreamer.

## 3. Instalación automática (`dos comandos`)

Ejecuta desde la raíz del repositorio:

```bash
bash packaging-pi/setup_pi.sh
bash packaging-pi/run_pi.sh
```

El primer script:
- Instala dependencias de sistema (`libatlas-base-dev`, `libopenblas-dev`, `libportaudio2`, `libjpeg-dev`, `libtiff-dev`, `gstreamer1.0-*`, `ffmpeg`, `libavcodec-extra59` o `libavcodec59` según disponibilidad) y resuelve el conflicto `libavcodec59` vs `libavcodec-extra59` eliminando la variante incompatible.
- Detecta `chromium` o `chromium-browser`, añade `libcamera0.5`/`rpicam-apps-core` y limpia paquetes heredados de Bullseye.
- Crea `.venv/`, actualiza `pip` y provisiona las versiones fijadas en [requirements-pi.txt](requirements-pi.txt) (Flask, Socket.IO, NumPy, OpenCV, MediaPipe).
- Genera snapshots de dependencias en `reports/logs/pi/vision-stack-*.json`.

El segundo script:
- Detecta el modelo (`/proc/device-tree/model`), ajusta `--poll-interval` (0.05 s para Pi 4, 0.04 s para Pi 5) y cachea la auto detección de cámara en `reports/logs/pi/camera-selection-*.json`.
- Arranca `backendHelen.server` en segundo plano y espera a que `/health` responda antes de lanzar Chromium en kiosko.
- Guarda los logs en `reports/logs/pi/backend-*.log`, `reports/logs/pi/chromium-*.log` y `reports/logs/pi/camera-bootstrap.log`.
- Permite personalizar:
  - `HELEN_CAMERA_INDEX` (`auto`, índice numérico o `/dev/videoX`).
  - `POLL_INTERVAL` para forzar otro intervalo.
  - `HELEN_BACKEND_EXTRA_ARGS` para añadir flags (`--frame-stride`, `--detection-confidence`, etc.).
  - `HELEN_NO_UI=1` si solo quieres el backend (sin Chromium).

Detén la ejecución con `Ctrl+C`; el script cierra backend y navegador limpiamente.

## 4. Servicios systemd

Los archivos `packaging-pi/helen.service` y `packaging-pi/kiosk.service` usan rutas relativas al usuario `pi` y al nuevo layout `HelenV2/HelenProyecto-main/HelenProyecto-main`.

1. Copia y habilita los servicios:
   ```bash
   sudo cp packaging-pi/helen.service /etc/systemd/system/
   sudo cp packaging-pi/kiosk.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable helen.service kiosk.service
   ```
2. Asegúrate de que coincidan con tu ruta y binarios:
   - `WorkingDirectory=%h/HelenV2/HelenProyecto-main/HelenProyecto-main`
   - `ExecStart=%h/HelenV2/HelenProyecto-main/HelenProyecto-main/.venv/bin/python -m backendHelen.server --host 0.0.0.0 --port=5000`
   - `ExecStart` del kiosko usa `/usr/bin/env bash -lc '...'` para resolver `chromium` o `chromium-browser` automáticamente.
3. Valida manualmente:
   ```bash
   sudo systemctl start helen.service kiosk.service
   systemctl status helen.service kiosk.service
   journalctl -u helen.service -u kiosk.service -n 50
   curl http://localhost:5000/health
   ```

## 5. Validaciones funcionales

1. `/health` debe reportar `"camera_ok": true`, `"stream_source": "camera"` y la resolución/fps esperada (Pi 4 ≈640×360 @24 fps).
2. La SPA debe cargar en `http://localhost:5000` en modo kiosko y mostrar el tutorial.
3. Reproduce los gestos H, C, R, I y comprueba que los paneles reaccionan y los SSE aparecen en `http://localhost:5000/events`.
4. Revisa los logs en `reports/logs/pi/` y `journalctl` para confirmar que la cámara usa `v4l2` o `gstreamer` según disponibilidad.

## 6. Solución de problemas

| Problema | Diagnóstico | Acción |
|----------|-------------|--------|
| Cámara en negro | `reports/logs/pi/camera-bootstrap.log` muestra `status:error`. | Ejecuta `v4l2-ctl --list-formats-ext`, forza MJPG, revisa permisos (`sudo usermod -aG video $USER`). |
| Conflicto `libavcodec` | `apt` bloquea dependencias. | Repite `bash packaging-pi/setup_pi.sh`; el script elimina la variante incorrecta y reinstala la esperada. |
| Chromium no arranca | `kiosk.service` queda en `activating`. | Verifica `DISPLAY=:0`, revisa `/usr/bin/chromium` vs `chromium-browser` y agrega flags Wayland vía `Environment=HELEN_CHROMIUM_FLAGS=...`. |
| Latencia alta/FPS bajos | `/health` muestra <15 fps. | Ajusta `POLL_INTERVAL` o `--frame-stride`, reduce resolución desde `camera_probe.ensure_camera_selection`. |
| Consumo CPU >60% | `top` reporta carga alta. | Incrementa `--poll-interval`, usa `HELEN_BACKEND_EXTRA_ARGS="--frame-stride 5"`. |
| Backend sin cámara | `camera_ok:false`. | Exporta `HELEN_CAMERA_INDEX=/dev/video0` o ejecuta `python -m backendHelen.camera_probe --list`. |

## 7. Limpieza y mantenimiento

- Para reinstalar dependencias vuelve a ejecutar `packaging-pi/setup_pi.sh`.
- Para desinstalar servicios: `sudo systemctl disable --now helen.service kiosk.service` y elimina los archivos en `/etc/systemd/system/`.
- Las dependencias de entrenamiento (scikit-learn, xgboost) se mantienen en `Hellen_model_RN/requirements.txt`; no forman parte de este flujo de ejecución.

Consulta [README-PI.md](README-PI.md) para los ajustes específicos de Pi 5 sin kiosko y [CHANGELOG_DOCS.md](../CHANGELOG_DOCS.md) para el historial de modificaciones de esta guía.
