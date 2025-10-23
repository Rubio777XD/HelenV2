# HELEN en Raspberry Pi 4/5 (Raspberry Pi OS 64-bit)

Esta guía describe el procedimiento oficial para instalar, configurar y operar HELEN en una Raspberry Pi 4 o 5 con Raspberry Pi OS 64-bit (Bookworm). El flujo instala el backend Flask/Socket.IO, habilita la cámara oficial/libcamera, lanza la interfaz web en Chromium en modo kiosko y registra servicios `systemd` para iniciar todo automáticamente al encender el dispositivo.

## 1. Requisitos

### Hardware

- Raspberry Pi 4 Model B (4 GB o más) **o** Raspberry Pi 5.
- Fuente oficial de 5V/3A (Pi 4) u 5V/5A (Pi 5).
- Tarjeta microSD UHS-I (32 GB mínimo, 64 GB recomendado).
- Cámara compatible con libcamera (Camera Module 2/3 o HQ) correctamente conectada.
- Monitor táctil o pantalla HDMI con ratón/teclado (para la instalación inicial).
- Conectividad Wi-Fi o Ethernet.

### Software

- Raspberry Pi OS 64-bit (Bookworm) actualizado (`sudo apt full-upgrade`).
- Python 3.11 provisto por el sistema.
- Acceso sudo para instalar paquetes.
- Repositorio de HELEN clonado en `${HOME}/HelenProyecto-main/HelenProyecto-main` (ajusta si usas otra ruta).

> **Nota:** HELEN asume que el dataset de producción `Hellen_model_RN/data.pickle` está presente. Copia el archivo antes de iniciar el backend para evitar que se use el dataset legado.

## 2. Preparación del sistema

1. Actualiza la distribución y reinicia:
   ```bash
   sudo apt update
   sudo apt full-upgrade
   sudo reboot
   ```
2. Habilita la cámara desde `raspi-config` y desactiva el ahorro de pantalla:
   ```bash
   sudo raspi-config
   ```
   - **Interface Options → Camera → Enable**.
   - **Display Options → Screen Blanking → Disable**.
   - Si quieres arranque al escritorio automáticamente: **System Options → Boot / Auto Login → Desktop Autologin** (requerido para el kiosko).
3. Verifica la cámara con libcamera:
   ```bash
   libcamera-hello -t 5000
   ```
   Si el comando muestra imagen durante ~5 s, la cámara está lista. Ante errores revisa el cable plano, actualiza el firmware (`sudo rpi-update`) o revisa permisos.

## 3. Instalar dependencias (apt + pip)

Desde la raíz del repositorio ejecuta:
```bash
cd ~/HelenProyecto-main/HelenProyecto-main
bash packaging-pi/setup_pi.sh
```

El script:
- Instala bibliotecas del sistema actuales (`libatlas-base-dev`, `libopenblas-dev`, `libportaudio2`, `libjpeg-dev`, `libtiff-dev`, `libcamera0.5`, `rpicam-apps-core`, `libavcodec-extra`, `libavcodec-dev`, `libavformat-dev`, `libswscale-dev` y el paquete disponible de Chromium).
- Crea un entorno virtual aislado en `.venv/` (usa `python3 -m venv`), actualiza `pip`, `setuptools` y `wheel` dentro de él e instala `packaging-pi/requirements-pi.txt` sin tocar los paquetes del sistema.
- Registra la salida completa en `reports/logs/pi/setup-*.log` para facilitar auditorías.

Si tu entorno usa un intérprete alternativo, exporta `PYTHON=/ruta/a/python3.11` antes de ejecutar el script. Después de la instalación activa el entorno con `source .venv/bin/activate` si quieres trabajar manualmente.

## 4. Ejecutar HELEN manualmente

Para validar la instalación sin servicios en segundo plano:
```bash
cd ~/HelenProyecto-main/HelenProyecto-main
bash packaging-pi/run_pi.sh
```

El script detecta si ejecutas una Pi 4 o Pi 5 y ajusta automáticamente el intervalo de inferencia (`--poll-interval`) a 0.050 s (Pi 4, ≈20 fps efectivos) o 0.040 s (Pi 5, ≈25 fps). También:
- Usa el intérprete de `.venv/bin/python` (o el indicado por `PYTHON`) para lanzar `backendHelen.server` en segundo plano y deja los logs en `reports/logs/pi/backend-*.log`.
- Espera a que `http://127.0.0.1:5000/health` responda antes de abrir Chromium y guarda los logs de navegador en `reports/logs/pi/chromium-*.log`.
- Abre Chromium en modo kiosko apuntando a `http://localhost:5000` y deshabilita temporalmente el protector de pantalla.
- Permite personalizar parámetros mediante variables de entorno:
  - `HELEN_CAMERA_INDEX` para seleccionar otra cámara.
  - `POLL_INTERVAL` o `HELEN_BACKEND_EXTRA_ARGS` para sobreescribir argumentos del backend.
  - `HELEN_NO_UI=1` si solo quieres el backend.

Detén el script con `Ctrl+C`; ambos procesos se cierran limpiamente.

## 5. Servicios systemd para arranque automático

1. Copia los archivos de unidad y ajusta rutas según tu usuario/directorio:
   ```bash
   sudo cp packaging-pi/helen.service /etc/systemd/system/
   sudo cp packaging-pi/kiosk.service /etc/systemd/system/
   ```
   Edita ambos archivos con `sudo nano` y verifica:
   - `WorkingDirectory` apunta a la carpeta raíz del repositorio.
   - `ExecStart` usa la ruta correcta de Python (`/usr/bin/python3`) y de Chromium (`/usr/bin/chromium` o `chromium-browser`).
   - `User`/`Group` coinciden con el usuario que inicia sesión (por defecto `pi`).
2. Recarga systemd y habilita los servicios:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable helen.service kiosk.service
   ```
3. Arranca manualmente para validar:
   ```bash
   sudo systemctl start helen.service kiosk.service
   ```
4. Revisa el estado y los logs:
   ```bash
   systemctl status helen.service kiosk.service
   journalctl -u helen.service -u kiosk.service -n 50
   ```
   En los logs del backend deberías ver una entrada indicando la ruta de cámara utilizada, por ejemplo:
   - `Ruta de cámara inicializada: v4l2 ...` cuando `cv2.VideoCapture(0)` funciona.
   - `Ruta de cámara inicializada: gstreamer ...` cuando se usa el pipeline `libcamerasrc`.

Tras un reinicio (`sudo reboot`) el backend y Chromium deben iniciar automáticamente en modo kiosko.

## 6. Validaciones funcionales obligatorias

Después de habilitar los servicios verifica:

1. **Cámara activa**: `journalctl -u helen.service | grep "Ruta de cámara"` debe indicar V4L2 o GStreamer. Ejecuta también `curl http://localhost:5000/health` y comprueba `"stream_source": "camera"` y `"camera_ok": true`.
2. **UI operativa**: Chromium abre la SPA sin barras ni scroll; el tutorial de onboarding aparece centrado.
3. **Gestos**: realiza los gestos H (Start), C, R, I. Confirma en la interfaz que se activan con consenso y cooldown (observa el anillo y los paneles Clima/Reloj/Inicio).
4. **Telemetría**: visita `http://localhost:5000/events` desde otra máquina de la red y comprueba que se reciben eventos SSE con timestamps crecientes.
5. **Alarmas/temporizador/dispositivos**: crea alarmas y dispositivos, recarga la página y valida que el estado persiste (se usa `localStorage`).
6. **Rendimiento**:
   - Pi 5: objetivo 720p @ 24–25 fps, CPU promedio ≤35 %, temperatura <75 °C. Ajusta `POLL_INTERVAL=0.035` si necesitas más fps.
   - Pi 4: objetivo 640×360 @ 24 fps, CPU promedio ≤35 %. Puedes reducir la resolución desde `libcamera-hello` (`--width 640 --height 360`) y HELEN la respetará.
   Monitoriza con `top`, `vcgencmd measure_temp` y `watch -n 5 vcgencmd get_throttled`.
7. **Logs limpios**: `journalctl -u helen.service` no debe mostrar errores críticos recurrentes. Las advertencias de `absl` quedan en nivel WARNING y no bloquean el flujo.

## 7. Mantenimiento y solución de problemas

- **Ruta de cámara**: el backend intenta primero V4L2 (`cv2.VideoCapture`) y, si falla, reinicia la captura con un pipeline GStreamer basado en `libcamerasrc`. Revisa `journalctl -u helen.service` para confirmar la ruta y actualiza `libcamera-apps` si persisten errores.
- **Chromium no arranca**: verifica que el usuario `pi` inicia sesión en el escritorio y que `DISPLAY=:0` está disponible. Si usas Wayland, añade `--ozone-platform=wayland --enable-features=UseOzonePlatform` al servicio.
- **Caídas por falta de memoria**: asegúrate de tener espacio de intercambio (por defecto 100 MB). Puedes incrementarlo en `sudo raspi-config` → Performance Options → Overlay File System → swap.
- **Actualizar dependencias**: vuelve a ejecutar `bash packaging-pi/setup_pi.sh`. El script reinstala solo los paquetes faltantes.
- **Desinstalar**: `sudo systemctl disable --now kiosk.service helen.service` y elimina los archivos en `/etc/systemd/system/`.

## 8. Referencia rápida

- **Scripts útiles**:
  - `packaging-pi/setup_pi.sh`: instala dependencias de sistema y Python.
  - `packaging-pi/run_pi.sh`: arranca backend + Chromium en kiosko manualmente.
- **Archivos de configuración**:
  - `packaging-pi/helen.service`: servicio systemd del backend.
  - `packaging-pi/kiosk.service`: servicio systemd para Chromium.
- **Documentación complementaria**:
  - [Guía de packaging Windows](../packaging/README-build-win.md).
  - [Reportes de métricas](../reports/gesture_session_report.md) generados por el backend.

Con estos pasos, HELEN queda lista para operar en Raspberry Pi 4/5 utilizando la cámara real, manteniendo 24–30 fps y la experiencia de usuario alineada con la versión de Windows.
