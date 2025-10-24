# FIX REPORT

## Problemas corregidos y causa raíz

- **Duplicación de carpetas (`helen`, `backendHelen`) fuera de la ruta oficial**: los scripts antiguos trabajaban desde `HelenV2/`
  y generaban copias parciales del frontend/backend. Se añadieron envoltorios en `scripts/` que siempre resuelven la raíz real del
  repo (`HelenProyecto-main/HelenProyecto-main`) y los servicios/systemd deben apuntar ahí; cualquier copia residual en la raíz
  puede eliminarse sin riesgo.
- **Conflicto de paquetes `libavcodec59` vs `libavcodec-extra59` en Raspberry Pi OS Bookworm**: `apt-get` se detenía pidiendo
  intervención manual. El nuevo `setup-pi.sh` detecta la variante instalada, desinstala la incompatible y fuerza la instalación de
  `libavcodec-extra59` (o `libavcodec59` si la versión *extra* no existe) antes de continuar.
- **`run_pi.sh` fallando con `command substitution: 1 unterminated here-document`**: los scripts `.sh` tenían finales de línea CRLF.
  Se normalizaron todos los `*.sh` a LF y se añadió una prueba automática para evitar regresiones.
- **El backend solo aceptaba índices numéricos**: ahora `--camera-index`/`--camera` aceptan rutas (`/dev/videoX`) y el backend
  almacena tanto el índice como la ruta preferida.
- **Reconocimiento inconsistente entre Windows y Raspberry Pi**: se añadieron perfiles por plataforma que ajustan resolución,
  `poll_interval`, `frame_stride` y umbrales de MediaPipe, sincronizando la precisión con las limitaciones de hardware (RPi 4/5 vs
  PCs x86).
- **Arranque manual y poco repetible**: se introdujeron `scripts/setup-*.sh|ps1` y `scripts/run-*.sh|ps1` que encapsulan la
  preparación de dependencias, creación de `.venv`, arranque del backend y apertura del frontend en dos comandos.

## Cambios destacados

- `backendHelen/camera_probe.py`
  - Autodetección más robusta (modos 1280×720→320×240, preferencia MJPG/YUYV/NV12/H264) y soporte DirectShow en Windows.
  - Guardado del formato elegido y latencia en el caché de cámara, además de rutas de logs diferenciadas (`reports/logs/pi|win`).
- `backendHelen/server.py`
  - Configuración por plataforma (`RuntimeConfig` opcional), defaults dinámicos y aceptación de índices o rutas.
  - El endpoint `/health` expone resolución, fps, formato y dispositivo seleccionado; `CameraGestureStream` aplica el FourCC
    correcto según la selección.
- `packaging-pi/setup_pi.sh`
  - Resolución automática del conflicto `libavcodec*`, limpieza de variantes obsoletas y logs cronológicos en `reports/logs/pi/`.
- Nuevos entrypoints multiplataforma:
  - `scripts/setup-pi.sh` y `scripts/run-pi.sh` (envoltorios idempotentes).
  - `scripts/setup-windows.ps1` y `scripts/run-windows.ps1` (PowerShell, logs en `reports\logs\win`).
- Documentación renovada (`README.md`) + se añade `FIX_REPORT.md` con este resumen.
- Pruebas nuevas: smoke-test del backend sin cámara, verificación del orden de sondeo de cámara y control de finales de línea.

## Auto-detección de cámara

1. Enumeración de candidatos (libcamera, `/dev/video*`, fallbacks DirectShow).
2. Por cada candidato se prueban los modos `1280x720@30`, `960x720@30`, `640x480@30`, `640x360@24`, `320x240@24`.
3. En cada modo se aplican `MJPG → YUYV → NV12 → H264 → default` antes de descartar el dispositivo.
4. Si V4L2 no entrega frames válidos se genera una tubería `libcamerasrc` vía GStreamer.
5. El resultado cacheado incluye `pixel_format`, `fps`, `latency_ms`, se escribe en `reports/logs/<pi|win>/camera-probe-*.json` y
   puede consultarse desde `/health`.

## Valores por defecto por plataforma

- **Raspberry Pi 5**: 1280×720 @ 25 fps, `poll_interval=0.040`, `frame_stride=3`, `detection_confidence=0.58`,
  `tracking_confidence=0.55`.
- **Raspberry Pi 4**: 640×360 @ 24 fps, `poll_interval=0.050`, `frame_stride=4`, `detection_confidence=0.60`,
  `tracking_confidence=0.55`.
- **Windows 10/11**: prioridad UVC HD, `poll_interval=0.080`, `frame_stride=2`, `detection_confidence=0.72`,
  `tracking_confidence=0.62`.
- **Linux genérico (x86)**: 960×540 @ 24 fps, `poll_interval=0.100`, `frame_stride=3`, `detection_confidence=0.68`,
  `tracking_confidence=0.60`.

Estos valores se aplican automáticamente salvo que el operador especifique overrides mediante los flags clásicos.

## Operación y diagnósticos

- **Endpoint de salud**: `http://localhost:5000/health` expone `camera_ok`, formato (`camera_pixel_format`), resolución y fps.
- **Logs**: `reports/logs/pi/` y `reports/logs/win/` almacenan `setup-*.log`, `backend-*.log`, `camera-bootstrap.log` y snapshots
  de dependencias.
- **Forzar cámara específica**:
  - Raspberry Pi: `export HELEN_CAMERA_INDEX=/dev/video1` (o índice numérico) antes de `run-pi.sh`.
  - Windows/Linux: `python -m backendHelen.server --camera-index /dev/video2` o equivalente; también acepta `--camera` como alias.

> Nota: ya no es necesario pasar explícitamente `/dev/video0`; si no se define preferencia, el sistema selecciona la mejor cámara
> disponible y degrada a resoluciones menores si la prueba a 1280×720 falla.
