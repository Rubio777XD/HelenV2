# HELEN en Raspberry Pi 5 (Raspberry Pi OS + Chromium)

Esta guía cubre **exclusivamente** el flujo oficial para ejecutar HELEN en una
Raspberry Pi 5 corriendo Raspberry Pi OS (Bookworm) de 64 bits. El backend vive en la
propia Pi, sirve el frontend web y controla la cámara Obsbot Tiny 2 (u otra cámara
UVC) mediante OpenCV + TensorFlow. Nada aquí aplica a distribuciones Linux de
escritorio: si necesitas instrucciones para Windows u otros entornos, consulta
`README.md` y `README-windows-chrome.md`.

## 1. Requisitos de sistema (Pi 5)

| Componente           | Detalles recomendados |
|---------------------|-----------------------|
| Hardware            | Raspberry Pi 5 (8 GB) con ventilación activa. |
| Sistema operativo   | Raspberry Pi OS Bookworm (64 bits) actualizado (`sudo apt full-upgrade`). |
| Python              | Python 3.10 + `python3.10-venv`. |
| Cámara              | Dispositivo UVC (ej. **Obsbot Tiny 2** conectada vía USB-C). |
| Navegador           | `chromium-browser` (preinstalado en Pi OS) o Google Chrome ARM64. |
| Dependencias extra  | `git`, `ffmpeg`, `v4l-utils`, pila GStreamer, `libatlas-base-dev`, `libopenblas-dev`, `libcamera-apps`. |

## 2. Instalación inicial paso a paso

```bash
cd /home/pi  # o tu ruta preferida
git clone https://github.com/tu-organizacion/HELEN.git
cd HELEN/HelenProyecto-main/HelenProyecto-main
```

1. **Instala dependencias APT (una sola vez).**

   ```bash
   sudo apt update
   sudo apt install -y git python3.10 python3.10-venv python3-pip ffmpeg v4l-utils \
        libatlas-base-dev libopenblas-dev libjpeg-dev libtiff5 zlib1g-dev \
        gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad} libcamera-apps chromium-browser
   ```

   > Puedes delegar toda esta sección ejecutando `./scripts/setup-pi.sh`, el cual:
   > - Valida que exista Python 3.10 instalado vía APT.
   > - Crea `.venv` (o el directorio indicado por `VENV_DIR`).
   > - Instala `requirements.txt` y los requisitos del modelo de video.
   > - Es idempotente y acepta `--skip-apt`, `--python-bin` y `--venv-dir`.

2. **Activa el entorno virtual.**

   ```bash
   source .venv/bin/activate
   ```

3. **Importa el modelo TensorFlow.** Copia la carpeta SavedModel más reciente (por
   ejemplo `gesture_model_20251031_183504`) dentro de
   `Hellen_model_TF/video_gesture_model/data/models/`. Debe contener `saved_model.pb`,
   `variables/` y `labels.json`. Si `labels.json` vive en la misma carpeta, puedes
   omitir `--labels` al arrancar el backend.

## 3. Ejecución diaria con `scripts/run-pi.sh`

Para iniciar el backend y (opcionalmente) Chromium:

```bash
./scripts/run-pi.sh \
  --camera-index 0 \
  --frame-width 1280 --frame-height 720 \
  --launch-browser --kiosk
```

El script realiza lo siguiente:

1. Activa `.venv` y usa `python -m backendHelen.server`.
2. Detecta automáticamente el SavedModel `gesture_model_*` más reciente si no
definiste `HELEN_TF_MODEL_PATH`.
3. Exporta variables de entorno útiles para procesos hijos:
   - `HELEN_TF_MODEL_PATH` – Ruta al SavedModel.
   - `HELEN_TF_LABELS_PATH` – Ruta al `labels.json` (se autodetecta si vive junto al modelo).
   - `HELEN_CAMERA_INDEX` – Índice OpenCV (0 = `/dev/video0`).
   - `HELEN_CAMERA_WIDTH` / `HELEN_CAMERA_HEIGHT` – Resuelve la captura en el
     servicio de inferencia sin tocar el código fuente.
   - `HELEN_BACKEND_EXTRA_ARGS` – Flags adicionales soportados por `backendHelen.server`
     (ej.: `"--confidence-threshold 0.8 --prediction-cooldown 0.6"`).
   - `HELEN_PORT` / `HELEN_HOST` – Personaliza el puerto/host del backend.
4. Guarda los logs en `reports/logs/pi/backend-<timestamp>.log` y los muestra en pantalla.
5. Si especificas `--launch-browser`, abre Chromium apuntando a `http://localhost:5000`
   (modo ventana o kiosk según `--kiosk` o `HELEN_PI_KIOSK_MODE=1`).

Parámetros útiles del script:

| Flag / variable                 | Descripción |
|--------------------------------|-------------|
| `--model-path` / `HELEN_TF_MODEL_PATH` | Ruta manual al SavedModel. |
| `--labels-path` / `HELEN_TF_LABELS_PATH` | Ruta a `labels.json` si no está junto al modelo. |
| `--camera-index` / `HELEN_CAMERA_INDEX` | Ajusta la cámara USB/CSI. |
| `--frame-width` / `HELEN_CAMERA_WIDTH` | Define resolución horizontal (ej. 1280). |
| `--frame-height` / `HELEN_CAMERA_HEIGHT` | Define resolución vertical (ej. 720). |
| `--launch-browser` / `HELEN_PI_LAUNCH_BROWSER=1` | Abre Chromium automáticamente. |
| `--kiosk` / `HELEN_PI_KIOSK_MODE=1` | Añade flags `--kiosk --incognito --noerrdialogs`. |
| `--browser-binary` / `HELEN_PI_BROWSER_BINARY` | Ruta al binario (`chromium-browser`, `chromium`, `google-chrome`). |
| `--browser-url` / `HELEN_PI_BROWSER_URL` | URL personalizada (útil si sirves desde otra máquina). |
| `HELEN_PI_BROWSER_FLAGS` | Flags adicionales para Chromium (ej. `"--use-fake-ui-for-media-stream"`). |

## 4. Modo kiosk y automatización

Para lanzar Chromium en kiosk manualmente:

```bash
HELEN_PI_LAUNCH_BROWSER=1 HELEN_PI_KIOSK_MODE=1 ./scripts/run-pi.sh
```

Systemd de ejemplo (incluidos en `system_scripts/`):

- `system_scripts/helen.service` → inicia el backend ejecutando `scripts/run-pi.sh`.
- `system_scripts/chromium-kiosk.service` → abre Chromium en modo kiosk apuntando a
  `http://localhost:5000`.

Instalación manual:

```bash
sudo cp system_scripts/helen.service /etc/systemd/system/
sudo cp system_scripts/chromium-kiosk.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable helen.service chromium-kiosk.service
sudo systemctl start helen.service chromium-kiosk.service
```

Asegúrate de editar `ExecStart`/`User` si usas una ruta diferente.

## 5. Notas para la cámara Obsbot Tiny 2

- Se comporta como cualquier cámara UVC → debería exponerse como `/dev/video0`.
- Verifica la detección con:

  ```bash
  lsusb | grep -i obsbot
  ls /dev/video*
  v4l2-ctl --list-devices
  ```

- Si la cámara aparece como `/dev/video2`, ajusta `HELEN_CAMERA_INDEX=2` o pasa
  `--camera-index 2` al script.
- Para balancear rendimiento y nitidez en la Pi 5, se recomienda `1280x720` usando
  `--frame-width 1280 --frame-height 720` o exportando `HELEN_CAMERA_WIDTH/HEIGHT`.
- Si el video se ve oscuro, usa la app de escritorio de Obsbot para fijar la
  exposición antes de conectarla a la Pi (los ajustes persisten a nivel firmware).

## 6. Validaciones rápidas

1. **Salud del backend:** `curl http://127.0.0.1:5000/health` → `{"status":"HEALTHY"}` y `"camera_ok": true`.
2. **Permisos de cámara:** al abrir el frontend, Chromium debe solicitar acceso.
3. **Ring de activación:** realiza el gesto Start/H y verifica que el anillo del UI
   y el panel de eventos reaccionan.
4. **Logs en vivo:** revisa `reports/logs/pi/backend-*.log` para confirmar que el
   modelo TensorFlow carga sin errores.

## 7. Solución de problemas

| Problema | Acción sugerida |
|----------|-----------------|
| `camera_ok:false` en `/health` | Ajusta `HELEN_CAMERA_INDEX`, ejecuta `v4l2-ctl --list-devices` y confirma permisos sobre `/dev/video*`. |
| Chromium sin video | Lanza el navegador con `--use-fake-ui-for-media-stream` la primera vez o elimina bloqueos en `chrome://settings/content/camera`. |
| TensorFlow no arranca en ARM64 | Asegúrate de usar Python 3.10 de 64 bits y reinstala con `./scripts/setup-pi.sh --skip-apt` para recompilar dependencias. |
| CPU/GPU saturados | Reduce la resolución (`HELEN_CAMERA_WIDTH=960 HELEN_CAMERA_HEIGHT=540`) o incrementa `--prediction-cooldown` vía `HELEN_BACKEND_EXTRA_ARGS`. |
| Chromium no debería arrancar siempre | Ejecuta `./scripts/run-pi.sh --no-browser` o evita exportar `HELEN_PI_LAUNCH_BROWSER`. |
| Sin modelo en `data/models` | Copia `gesture_model_YYYYMMDD_HHMMSS` desde tu PC y vuelve a ejecutar `run-pi.sh --model-path /ruta/al/modelo`. |

## 8. Referencias rápidas

- **Comandos clave**
  - Setup completo: `./scripts/setup-pi.sh`
  - Ejecución manual: `source .venv/bin/activate && ./scripts/run-pi.sh`
  - Modo kiosk: `HELEN_PI_LAUNCH_BROWSER=1 HELEN_PI_KIOSK_MODE=1 ./scripts/run-pi.sh`
- **Rutas útiles**
  - Backend: `backendHelen/server.py`
  - Modelo: `Hellen_model_TF/video_gesture_model/data/models/gesture_model_*`
  - Scripts Pi: `scripts/setup-pi.sh`, `scripts/run-pi.sh`
  - Systemd: `system_scripts/helen.service`, `system_scripts/chromium-kiosk.service`

Con esta guía puedes replicar el mismo flujo usado en Windows pero totalmente
contenidizado en Raspberry Pi 5, manteniendo el modelo TensorFlow de video como
única fuente de verdad y sin introducir validaciones adicionales.
