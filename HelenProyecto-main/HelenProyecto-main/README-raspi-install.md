# Instalación completa de HELEN en Raspberry Pi 5

Esta guía describe de forma **lineal** el proceso para preparar una Raspberry Pi 5
(Raspberry Pi OS Bookworm 64 bits) y dejar corriendo HELEN con la cámara Obsbot Tiny 2.
A diferencia de `README-linux-rpi-chrome.md`, aquí se lista cada comando necesario
para instalar dependencias, clonar el repositorio, provisionar el entorno virtual y
lanzar el backend + frontend desde cero.

## 1. Requisitos previos

| Elemento | Detalle |
| --- | --- |
| Hardware | Raspberry Pi 5 (recomendado 8 GB) con fuente USB-C de 27 W y ventilación activa. |
| Sistema operativo | Raspberry Pi OS Bookworm 64 bits actualizado (`sudo apt full-upgrade`). |
| Cámara | Dispositivo UVC (Obsbot Tiny 2 probado). |
| Navegador | Chromium (preinstalado) o Google Chrome ARM64. |
| Cuenta | Usuario con permisos `sudo`. |

> **Nota:** La Pi debe tener al menos 15 GB libres para TensorFlow + modelos.

## 2. Preparar el sistema operativo

1. Actualiza la imagen base y reinicia:
   ```bash
   sudo apt update
   sudo apt full-upgrade -y
   sudo reboot
   ```
2. Tras reiniciar, vuelve a iniciar sesión y continúa con la instalación.

## 3. Instalar dependencias del sistema

Ejecuta el bloque completo (o deja que `scripts/setup-pi.sh` lo haga por ti):

```bash
sudo apt install -y git python3.10 python3.10-venv python3-pip ffmpeg v4l-utils \
     libatlas-base-dev libopenblas-dev libjpeg-dev libtiff5 zlib1g-dev \
     gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad} libcamera-apps chromium-browser
```

Estos paquetes cubren Python, OpenCV, TensorFlow ARM64 y herramientas de cámara.

## 4. Clonar el repositorio de HELEN

```bash
cd /home/pi
git clone https://github.com/tu-organizacion/HELEN.git
cd HELEN/HelenProyecto-main/HelenProyecto-main
```

## 5. Configurar el entorno virtual

### 5.1 Opción automática (recomendada)

```
./scripts/setup-pi.sh
```

El script instala dependencias Python, crea `.venv`, sincroniza `pip` y valida que
exista al menos un modelo en `Hellen_model_TF/video_gesture_model/data/models/`.

Parámetros útiles:
- `--skip-apt` si ya ejecutaste la instalación de paquetes.
- `--python-bin /usr/bin/python3.10` para forzar el intérprete.
- `--venv-dir /home/pi/helen-venv` para crear el entorno fuera del repo.

### 5.2 Opción manual

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r Hellen_model_TF/video_gesture_model/requirements.txt
```

## 6. Descargar o copiar el modelo TensorFlow

El modelo estable vive como carpeta SavedModel (ej. `gesture_model_20251031_183504`).
Colócala dentro de `Hellen_model_TF/video_gesture_model/data/models/` junto a
`labels.json`. Si ya viene en el repositorio, verifica que `saved_model.pb` y el
subdirectorio `variables/` estén presentes.

## 7. Variables de entorno recomendadas

Define las variables en tu shell o agrégalas a `~/.bashrc`:

```bash
export HELEN_TF_MODEL_PATH="$(pwd)/Hellen_model_TF/video_gesture_model/data/models/gesture_model_20251031_183504"
export HELEN_CAMERA_INDEX=0            # /dev/video0 (Obsbot Tiny 2)
export HELEN_CAMERA_WIDTH=1280
export HELEN_CAMERA_HEIGHT=720
export HELEN_BACKEND_EXTRA_ARGS="--confidence-threshold 0.75 --prediction-cooldown 0.6"
```

El script `run-pi.sh` exporta estos valores automáticamente si lo deseas.

## 8. Arrancar HELEN

### 8.1 Con el script oficial

```bash
./scripts/run-pi.sh --launch-browser --kiosk
```

El script:
1. Activa `.venv`.
2. Busca el último modelo disponible si no definiste `HELEN_TF_MODEL_PATH`.
3. Ejecuta `python -m backendHelen.server --host 0.0.0.0 --port 5000 --camera-index $HELEN_CAMERA_INDEX --model-path "$HELEN_TF_MODEL_PATH"`.
4. Guarda los logs en `reports/logs/pi/`.
5. Lanza Chromium hacia `http://localhost:5000` (modo normal o kiosk según los flags).

Flags adicionales útiles:
- `--browser-binary chromium-browser`
- `--browser-url http://localhost:5000`
- `--frame-width 1280 --frame-height 720`
- `--no-browser` si solo quieres el backend.

### 8.2 Lanzamiento manual

```bash
source .venv/bin/activate
python -m backendHelen.server \
  --host 0.0.0.0 \
  --port 5000 \
  --camera-index 0 \
  --model-path "Hellen_model_TF/video_gesture_model/data/models/gesture_model_20251031_183504"
```

En otra terminal abre Chromium:

```bash
chromium-browser http://localhost:5000
```

## 9. Validar la instalación

1. Revisa la salud del backend: `curl http://127.0.0.1:5000/health` → debe mostrar `"status":"HEALTHY"`.
2. Confirma que Chromium pide acceso a la cámara y que el ring de activación responde.
3. Consulta `reports/logs/pi/backend-*.log` ante cualquier error de TensorFlow o cámara.

## 10. Opcional: Modo kiosk y systemd

- **Modo kiosk rápido:** `HELEN_PI_LAUNCH_BROWSER=1 HELEN_PI_KIOSK_MODE=1 ./scripts/run-pi.sh`
- **Unidades systemd:** copia `system_scripts/helen.service` y
  `system_scripts/chromium-kiosk.service` a `/etc/systemd/system/`, ajusta `User=` y
  habilítalas con `sudo systemctl enable --now helen chromium-kiosk`.

## 11. Solución de problemas

| Síntoma | Posible causa | Acción |
| --- | --- | --- |
| `/health` muestra `camera_ok:false` | Índice incorrecto o permisos | Ejecuta `v4l2-ctl --list-devices`, cambia `HELEN_CAMERA_INDEX` y revisa pertenencia al grupo `video`. |
| Chromium no muestra video | Bloqueo de permisos en navegador | Abre `chrome://settings/content/camera` y selecciona la Obsbot Tiny 2. |
| TensorFlow falla al importar | Dependencias faltantes en ARM64 | Repite `./scripts/setup-pi.sh --skip-apt` para recompilar ruedas. |
| FPS bajos / CPU alta | Resolución excesiva | Baja a `960x540` o incrementa `--prediction-cooldown`. |

Con esto tu Raspberry Pi 5 queda lista para operar HELEN de forma estable.
