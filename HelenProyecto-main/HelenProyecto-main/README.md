# HELEN

HELEN es una experiencia de asistente doméstico controlada por gestos. El backend escrito en Flask expone los eventos de cámara y
la lógica de reconocimiento de gestos mientras que el frontend web sirve el tutorial, alarmas, temporizador y dispositivos
conectados.

## Inicio rápido

Los flujos estándar quedaron reducidos a dos comandos por plataforma. Ambos scripts son idempotentes: puedes ejecutarlos varias
veces sin dejar el entorno en un estado inconsistente.

### Raspberry Pi 4/5 (Raspberry Pi OS 64-bit)
1. `./scripts/setup-pi.sh`
   - Detecta el intérprete de Python 3.11, instala dependencias de sistema con apt, resuelve automáticamente el conflicto
     `libavcodec59`/`libavcodec-extra59`, crea `.venv/` y provisiona NumPy/OpenCV/MediaPipe.
2. `./scripts/run-pi.sh`
   - Refresca la caché de auto-detección de cámara, lanza el backend en `http://0.0.0.0:5000`, registra los logs en
     `reports/logs/pi/` y abre Chromium en modo interactivo (normal o kiosk según tu configuración).

### Windows 10/11 (PowerShell 7+ recomendado)
1. `powershell -ExecutionPolicy Bypass -File .\scripts\setup-windows.ps1`
   - Utiliza `py`/`python` para localizar Python 3.11, crea `.venv\`, instala NumPy/OpenCV/MediaPipe y el resto de dependencias,
     dejando un log en `reports\logs\win\setup-*.log`.
2. `powershell -ExecutionPolicy Bypass -File .\scripts\run-windows.ps1`
   - Arranca el backend desde la raíz del proyecto, espera al endpoint `/health`, abre el navegador predeterminado en
     `http://localhost:5000` y conserva todos los mensajes en `reports\logs\win\backend-*.log`.

## Auto-detección de cámara

El backend ahora acepta tanto índices numéricos (`0`, `1`, `2`) como rutas (`/dev/video0`). Si no se especifica nada, el sistema
elige automáticamente la mejor cámara disponible siguiendo esta estrategia:

1. Enumeración de dispositivos UVC/CSI (`/dev/video*`, libcamera y fallbacks de DirectShow en Windows).
2. Pruebas en cascada de los siguientes modos por cada candidato: `1280x720@30`, `960x720@30`, `640x480@30`, `640x360@24`,
   `320x240@24`.
3. Para cada modo se intentan los formatos `MJPG`, `YUYV`, `NV12` y `H264` antes de degradar a V4L2 sin FourCC explícito.
4. Si V4L2 no entrega frames válidos, se construye automáticamente una tubería `libcamerasrc` vía GStreamer.

La selección final incluye información de resolución, fps, formato y latencia. Está disponible en `/reports/logs/pi/` (Linux) o
`/reports/logs/win/` (Windows) y expuesta en el endpoint `/health` (`camera_pixel_format`, `camera_resolution`, `camera_fps`).

## Valores por plataforma

| Plataforma            | Resolución/FPS objetivo | `poll_interval` | `frame_stride` | `detection_conf.` | `tracking_conf.` |
|----------------------|-------------------------|-----------------|----------------|-------------------|------------------|
| Raspberry Pi 5       | 1280×720 @ 25 fps       | 0.040 s         | 3              | 0.58              | 0.55             |
| Raspberry Pi 4       | 640×360 @ 24 fps        | 0.050 s         | 4              | 0.60              | 0.55             |
| Windows 10/11        | Auto (prioriza UVC HD)  | 0.080 s         | 2              | 0.72              | 0.62             |
| Linux genérico (x86) | 960×540 @ 24 fps        | 0.100 s         | 3              | 0.68              | 0.60             |

Los valores se ajustan automáticamente en función del hardware detectado para alinear la precisión entre Windows y Raspberry Pi.
Si necesitas sobrescribirlos, los flags históricos (`--detection-confidence`, `--tracking-confidence`, `--poll-interval`,
`--frame-stride`, `--camera-index`) siguen disponibles y aceptan tanto índices como rutas.

## Supervisión y salud

- `http://localhost:5000/health` devuelve el estado del pipeline, incluyendo resolución activa, formato, fps y última captura.
- Los logs del backend se agrupan por plataforma en `reports/logs/pi/` y `reports/logs/win/`.
- Para forzar una cámara concreta exporta `HELEN_CAMERA_INDEX` (Raspberry Pi) o pasa `--camera-index`/`--camera` al backend.

## Solución de problemas rápidos

- **Cámara en negro o frames nulos**: revisa `reports/logs/*/camera-bootstrap.log`, asegura que el formato MJPG esté disponible
  (`v4l2-ctl --list-formats-ext`) y confirma permisos (`sudo usermod -aG video $USER`).
- **Conflictos de codecs (`libavcodec59` vs `libavcodec-extra59`)**: el script `setup-pi.sh` detecta y elimina la variante
  incompatible; ejecuta nuevamente el setup si el conflicto reaparece tras una actualización del sistema.
- **Formato no soportado / latencia alta**: la auto-detección degradará a YUYV/640×360 automáticamente. Puedes forzar otro modo
  ejecutando `python -m backendHelen.camera_probe --help` para validar manualmente cada dispositivo.
- **Aceleración TFLite/MediaPipe**: verifica que `mediapipe==0.10.18` se haya instalado dentro de `.venv/` (`pip show mediapipe`) y
  que el log `vision-runtime-*.json` reporte `mediapipe.status = ok`.

Para más detalle sobre los cambios recientes y diagnóstico consulta `FIX_REPORT.md`.
