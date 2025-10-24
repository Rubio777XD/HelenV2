# HELEN

HELEN es una experiencia de asistente doméstico controlada por gestos. El backend escrito en Flask/Socket.IO expone los eventos de cámara y la lógica de reconocimiento de gestos mientras que el frontend web sirve el tutorial, alarmas, temporizador y dispositivos conectados. Todo el proyecto vive dentro de `HelenV2/HelenProyecto-main/HelenProyecto-main`, por lo que cualquier referencia de rutas en esta documentación usa esa jerarquía como base.

## Inicio rápido ("dos comandos")

Los flujos de instalación quedaron reducidos a dos comandos por plataforma. Los scripts son idempotentes y pueden ejecutarse múltiples veces sin dejar el entorno en un estado inconsistente.

### Raspberry Pi 5 (Bookworm optimizado)
1. `bash ./scripts/setup-pi.sh`
   - Detecta Python 3.11 del sistema, instala dependencias con `apt`, resuelve automáticamente el conflicto `libavcodec59`/`libavcodec-extra59`, crea `.venv/` y provisiona NumPy/OpenCV/MediaPipe con pines compatibles para ARM64.
2. `bash ./scripts/run-pi.sh`
   - Refresca la caché de auto detección de cámara, lanza el backend en `http://0.0.0.0:5000`, registra los logs en `reports/logs/pi/` y abre Chromium en modo ventana (puedes activar kiosko con `HELEN_PI_KIOSK=1`).

### Raspberry Pi 4/5 (Bookworm con kiosko clásico)
1. `bash packaging-pi/setup_pi.sh`
2. `bash packaging-pi/run_pi.sh`
   - Equivalen a los scripts anteriores pero conservan el kiosko tradicional y parámetros conservadores para Pi 4. Usa esta guía si migras instalaciones existentes descritas en [packaging-pi/README-raspi.md](packaging-pi/README-raspi.md).

### Windows 10/11 (PowerShell 7+ recomendado)
1. `powershell -ExecutionPolicy Bypass -File .\scripts\setup-windows.ps1`
   - Crea `.venv\`, instala NumPy/OpenCV/MediaPipe y el resto de requisitos listados en [`requirements.txt`](requirements.txt), dejando un log en `reports\logs\win\setup-*.log`.
2. `powershell -ExecutionPolicy Bypass -File .\scripts\run-windows.ps1`
   - Arranca el backend desde la raíz del proyecto, espera al endpoint `/health`, abre el navegador predeterminado en `http://localhost:5000` y conserva los logs en `reports\logs\win\backend-*.log`.

Para empaquetado en Windows vía PyInstaller/Inno Setup consulta [packaging/README-build-win.md](packaging/README-build-win.md).

## Auto detección de cámara y degradación

El backend acepta tanto índices numéricos (`0`, `1`, …) como rutas (`/dev/video0`). Si no se especifica nada se elige automáticamente el mejor dispositivo disponible siguiendo esta estrategia:

1. Enumeración de cámaras UVC/CSI (`/dev/video*`) y fallbacks de DirectShow (Windows).
2. Pruebas en cascada de los modos `1280x720@30`, `960x720@30`, `640x480@30`, `640x360@24`, `320x240@24`.
3. Para cada modo se intentan los formatos `MJPG`, `YUYV`, `NV12` y `H264` antes de degradar a V4L2 sin FourCC explícito.
4. Si V4L2 no entrega frames válidos se activa automáticamente una tubería `libcamerasrc` (GStreamer).

Las selecciones exitosas se guardan en `reports/logs/pi/camera-bootstrap.log` (Linux) o `reports/logs/win/backend-*.log` (Windows) y se exponen en [`/health`](http://localhost:5000/health).

### Valores por plataforma

| Plataforma            | Resolución/FPS objetivo | `poll_interval` | `frame_stride` | `detection_confidence` | `tracking_confidence` |
|----------------------|-------------------------|-----------------|----------------|------------------------|-----------------------|
| Raspberry Pi 5       | 1280×720 @ 25 fps       | 0.040 s         | 3              | 0.58                   | 0.55                  |
| Raspberry Pi 4       | 640×360 @ 24 fps        | 0.050 s         | 4              | 0.60                   | 0.55                  |
| Windows 10/11        | Auto (prioriza UVC HD)  | 0.080 s         | 2              | 0.72                   | 0.62                  |

Si necesitas sobrescribirlos, usa los flags clásicos (`--detection-confidence`, `--tracking-confidence`, `--poll-interval`, `--frame-stride`, `--camera-index`) o las variables de entorno `HELEN_CAMERA_INDEX`/`HELEN_BACKEND_EXTRA_ARGS` documentadas en las guías de plataforma.

## Endpoint `/health`

`http://localhost:5000/health` devuelve un JSON con los campos relevantes del pipeline:

- `status`: `HEALTHY`, `DEGRADED` o `ERROR`.
- `camera_ok`: `true` cuando la captura entrega frames válidos.
- `camera_index` y `camera_device`: índice y ruta/resumen del dispositivo.
- `camera_backend`: backend seleccionado (`v4l2`, `gstreamer`, `directshow`).
- `camera_resolution`, `camera_fps`, `camera_pixel_format`, `camera_probe_latency_ms` y `camera_last_capture`.
- `model_loaded`, `model_source`, `avg_latency_ms`, `clients` (SSE conectados) y `last_prediction`.

La ruta `/healthz` es un alias pensada para balanceadores. En Linux los logs asociados viven en `reports/logs/pi/`, en Windows en `reports/logs/win/`.

## Validaciones rápidas

1. **Backend**: `python -m backendHelen.server --no-camera --host 127.0.0.1 --port 8765` (Ctrl+C para salir). La consola debe registrar el snapshot de dependencias aunque no haya cámara.
2. **Logs**: verifica `reports/logs/**` tras cada ejecución.
3. **Telemetría**: abre `http://localhost:5000/health` y comprueba `"camera_ok": true`.
4. **Servicios Pi**: `systemctl status helen.service kiosk.service` y `journalctl -u helen.service -u kiosk.service -n 50`.
5. **Frontend**: la SPA debe cargar en `http://localhost:5000` mostrando el tutorial interactivo.

## Matriz de pruebas (documentación)

| Plataforma & cámara                 | Script                | Resultado esperado |
|-------------------------------------|-----------------------|--------------------|
| Raspberry Pi 5 + OBSBOT Tiny 2 Lite | `scripts/run-pi.sh`   | `camera_ok:true`, `camera_backend:"v4l2"`, 24–25 fps, tutorial visible en ventana.
| Raspberry Pi 4 + UVC genérica       | `packaging-pi/run_pi.sh` | `camera_ok:true`, degradación a 640×360 MJPG o YUYV, kiosko activo.
| Windows 10/11 + webcam integrada    | `scripts\run-windows.ps1` | `camera_ok:true`, backend en `http://localhost:5000`, navegador predeterminado abierto.

## Solución de problemas comunes

- **Cámara en negro o frames nulos**: revisa `reports/logs/*/camera-bootstrap.log`, ejecuta `v4l2-ctl --list-formats-ext` (Linux) o usa el Administrador de dispositivos (Windows) para confirmar MJPG. Exporta `HELEN_CAMERA_INDEX=/dev/videoX` para forzar una cámara.
- **Conflictos `libavcodec59`/`libavcodec-extra59`**: `scripts/setup-pi.sh` y `packaging-pi/setup_pi.sh` detectan y eliminan la variante incompatible antes de instalar dependencias.
- **Formato no soportado / latencia alta**: el backend degradará automáticamente a YUYV/640×360. Ajusta `POLL_INTERVAL` o `HELEN_BACKEND_EXTRA_ARGS="--frame-stride 4"` para reducir carga.
- **Chromium/Wayland**: añade `HELEN_CHROMIUM_FLAGS="--ozone-platform=wayland --enable-features=UseOzonePlatform"` antes de ejecutar los scripts en escritorios Wayland.
- **Permisos V4L2**: `sudo usermod -aG video $USER && sudo reboot`.
- **Consumo de CPU**: valida con `top`/`htop` en Pi y ajusta `--poll-interval` o `--frame-stride`.
- **Diagnóstico rápido**: `ffmpeg -f v4l2 -list_formats all -i /dev/video0` y `v4l2-ctl --list-devices` ayudan a confirmar compatibilidad.

## Documentación adicional

- [Guía de build Windows](packaging/README-build-win.md)
- [Guía Pi 5 optimizada](packaging-pi/README-PI.md)
- [Guía Pi 4/5 legacy](packaging-pi/README-raspi.md)
- [CHANGELOG de documentación](CHANGELOG_DOCS.md)

Mantén estas guías sincronizadas con los scripts reales; cualquier cambio funcional debe reflejarse tanto en la documentación como en los requisitos.
