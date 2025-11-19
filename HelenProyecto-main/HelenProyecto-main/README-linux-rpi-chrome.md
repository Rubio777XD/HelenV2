# HELEN – Guía completa para ejecutar en Chrome (Linux / Raspberry Pi)

Esta guía describe el flujo soportado para correr HELEN en distribuciones basadas en Debian
(Ubuntu 22.04+, Raspberry Pi OS Bookworm) con Python 3.10+ y Google Chrome/Chromium. Toda la
aplicación usa el modelo de gestos en video exportado como TensorFlow SavedModel.

## 1. Inicio rápido

### 1.1 Requisitos mínimos

| Componente                 | Detalles                                                                                                                                 |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| Sistema operativo         | Ubuntu 22.04+, Debian 12 o Raspberry Pi OS Bookworm (64 bits recomendado).                                                               |
| Python                    | Python 3.10 o 3.11 instalado con `python3-venv` y `pip`.                                                                                 |
| Cámara                    | Webcam UVC o módulo CSI compatible con V4L2/libcamera.                                                                                    |
| GPU / CPU                 | CPU con soporte AVX/NEON. En Raspberry Pi 4/5 se recomienda enfriamiento activo.                                                          |
| Navegador                 | Google Chrome 124+ o Chromium (`chromium-browser`).                                                                                       |
| Dependencias del sistema  | `ffmpeg`, pila GStreamer (`gstreamer1.0-*`), `libatlas-base-dev`, `libcamera`, `chromium`, `v4l-utils`.                                    |

### 1.2 Ejecutar todo con scripts oficiales

```bash
cd HELEN/HelenProyecto-main/HelenProyecto-main
./scripts/setup-pi.sh        # instala paquetes APT y crea .venv (requiere sudo)
./scripts/run-pi.sh          # inicia backendHelen.server y abre Chromium
```

`scripts/setup-pi.sh` y `scripts/run-pi.sh` delegan en `legacy/packaging/linux-rpi/*.sh`, por lo que registran
logs en `reports/logs/pi/`. Las variables de entorno más útiles:

- `HELEN_CAMERA_INDEX` (por defecto vacío → auto detección con `camera_probe`).
- `HELEN_TF_MODEL_PATH` (ruta absoluta al SavedModel cuando no vive en `Hellen_model_TF/video_gesture_model/`).
- `HELEN_BACKEND_EXTRA_ARGS` para pasar flags adicionales (`--confidence-threshold 0.75`, `--prediction-cooldown 0.5`).
- `HELEN_PORT` para cambiar el puerto por defecto (5000).

Tras ejecutar `run-pi.sh`, abre `http://localhost:5000` en Chrome/Chromium si el script no lanzó el navegador.
Permite el acceso a la cámara cuando aparezca el diálogo.

## 2. Flujo manual (sin scripts)

1. Instala paquetes del sistema:

   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip ffmpeg v4l-utils libatlas-base-dev \
        libopenblas-dev libjpeg-dev libtiff-dev gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad} \
        libcamera0.5 libcamera-apps chromium-browser
   ```

2. Clona el repositorio y crea el entorno virtual:

   ```bash
   git clone https://github.com/tu-organizacion/HELEN.git
   cd HELEN/HelenProyecto-main/HelenProyecto-main
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Define las variables necesarias y ejecuta el backend con el modelo TF:

   ```bash
   export HELEN_CAMERA_INDEX=0
   export HELEN_TF_MODEL_PATH="$PWD/Hellen_model_TF/video_gesture_model/saved_model"
   export HELEN_TF_LABELS_PATH="$PWD/Hellen_model_TF/video_gesture_model/labels.json"
   export HELEN_BACKEND_EXTRA_ARGS="--confidence-threshold 0.75 --prediction-cooldown 0.5"

   python -m backendHelen.server \
       --host 0.0.0.0 --port 5000 --camera-index ${HELEN_CAMERA_INDEX:-0} \
       --model-path "$HELEN_TF_MODEL_PATH" --labels "$HELEN_TF_LABELS_PATH"
   ```

4. Abre `http://localhost:5000` en Chrome/Chromium (o en otro dispositivo apuntando a la IP de la máquina).
   El frontend solicitará permisos de cámara; acéptalos para que aparezca el ring de activación.

## 3. Validaciones rápidas

1. `curl http://127.0.0.1:5000/health` → debe responder `{"status":"HEALTHY"}` y `"camera_ok": true`.
2. En la UI, realiza el gesto **Start/H** y verifica que el anillo se ilumina y aparece el evento en el log lateral.
3. Navega entre tarjetas (Clima, Reloj) usando los gestos equivalentes para confirmar que las etiquetas siguen vigentes.
4. Ejecuta las pruebas automáticas desde `.venv`:

   ```bash
   pytest tests/test_backend_api.py tests/test_model_pipeline.py
   ```

## 4. Solución de problemas

| Problema                                      | Solución sugerida                                                                                           |
|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `camera_ok:false` en `/health`                | Ejecuta `v4l2-ctl --list-devices`, ajusta `HELEN_CAMERA_INDEX` y confirma permisos en `/dev/video*`.        |
| Chromium no abre o sin permisos de cámara     | Ejecuta `chromium-browser --use-fake-ui-for-media-stream http://localhost:5000` la primera vez.             |
| TensorFlow falla al importar (`illegal instr`) | Verifica que el CPU soporte AVX (x86_64) o NEON (ARM64). En ARM32 usa `tensorflow-cpu==2.12.0` manualmente. |
| MediaPipe no instala en ARM64                 | Asegúrate de usar Python 3.10/3.11 de 64 bits e instala desde `requirements.txt` (incluye `mediapipe==0.10.18`). |
| SSE sin eventos                               | Confirma que `python -m backendHelen.server` muestra inferencias en consola y que no hay errores en `reports/logs/pi/`. |
| Cámara CSI conflictiva                         | Exporta `OPENCV_VIDEOIO_PRIORITY_LIST=GSTREAMER,V4L2` antes de ejecutar el backend para forzar GStreamer.   |

## 5. Limpieza y actualización

- Elimina `.venv` y vuelve a ejecutar `./scripts/setup-pi.sh` para reinstalar dependencias.
- Limpia logs antiguos desde `reports/logs/pi/` si necesitas liberar espacio.
- Para actualizar el modelo TF, reemplaza el directorio dentro de `Hellen_model_TF/video_gesture_model/` y reinicia el backend.

## 6. Expectativas de UX

- El gesto **Start/H** sigue siendo la señal de activación para el ring.
- Los gestos `Clima`, `Reloj`, `Temporizador` conservan sus etiquetas originales; el backend aplica `GestureLabelMapper`
  para normalizar cualquier alias proveniente del modelo.
- Eventualmente puedes desactivar el ring desde la UI, pero los eventos de gestos continuarán llegando por SSE.

Esta guía se actualiza junto con `README.md` y `README-windows-chrome.md` para garantizar paridad entre plataformas.
