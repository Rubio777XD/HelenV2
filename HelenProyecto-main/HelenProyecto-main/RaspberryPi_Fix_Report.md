# Raspberry Pi Fix Report

## Resumen de cambios
- Se actualizaron los scripts de instalación y ejecución para Raspberry Pi OS Bookworm:
  - `packaging-pi/setup_pi.sh` instala los paquetes actuales (`libcamera0.5`, `rpicam-apps-core`, `libtiff-dev`, familia FFmpeg genérica) y crea un entorno virtual `.venv` donde se instalan las dependencias Python (incluyendo MediaPipe 0.10.18).
  - `packaging-pi/run_pi.sh` reutiliza `.venv/bin/python`, registra logs por ejecución en `reports/logs/pi/`, espera a que el backend responda en `/health` antes de abrir Chromium en modo kiosko, limpia ambos procesos al salir y ahora deja constancia del stack `rpicam`/`picamera2` detectado antes de arrancar.
- `packaging-pi/requirements-pi.txt` fija `mediapipe==0.10.18`, compatible con Python 3.11 en Raspberry Pi.
- Se documentó el nuevo flujo en `packaging-pi/README-raspi.md` (paquetes reemplazados, uso de `.venv`, ubicación de logs).
- Se rediseñó el modo Raspberry en `helen/css/globals.css` con un enfoque fit-first:
  - Límites con `clamp` para tipografías, iconos y paddings.
  - Grid 2×2 para accesos rápidos en 800×480, safe area para wifi y barra inferior, autoescalado de último recurso.
  - Ajustes específicos para Home, Clima, Reloj, Dispositivos, Configuración y Tutorial.
- Se añadió soporte de auto-escalado en `eventConnector.js` (`window.HelenRaspberryFit.refresh`) para recalcular cuando cambia el modo o el contenido.
- `devices.js` ahora pagina la lista (4 dispositivos por página) y notifica al auto-fit tras renderizar.

## Dependencias actualizadas
- APT: `libcamera0.5`, `rpicam-apps-core`, `libtiff-dev`, `libavcodec-extra`, `libavcodec-dev`, `libavformat-dev`, `libswscale-dev`, además de los paquetes existentes.
- Python (`packaging-pi/requirements-pi.txt`): `mediapipe==0.10.18`.

## Entorno virtual
- `.venv/` se crea automáticamente con `python3 -m venv` desde `setup_pi.sh`.
- El script instala `pip`, `setuptools`, `wheel` y las dependencias de `requirements-pi.txt` dentro del entorno.
- Activa el entorno con:
  ```bash
  source .venv/bin/activate
  ```
  y desactívalo con `deactivate`.

## Cómo ejecutar
1. Instala dependencias y crea la `.venv`:
   ```bash
   cd ~/HelenProyecto-main/HelenProyecto-main
   bash packaging-pi/setup_pi.sh
   ```
2. Arranca HELEN manualmente (logs en `reports/logs/pi/`):
   ```bash
   bash packaging-pi/run_pi.sh
   ```
   Usa `Ctrl+C` para detener; el script cierra backend y Chromium con limpieza.

## Validación visual (fit-first)
Capturas obtenidas sirviendo la interfaz estática en resolución 800×480 y 1024×600.

- Home 800×480: ![Home 800×480](browser:/invocations/salrkxgg/artifacts/artifacts/raspberry-home-800x480.png)
- Home 1024×600: ![Home 1024×600](browser:/invocations/smorikyo/artifacts/artifacts/raspberry-home-1024x600.png)
- Clima 800×480: ![Clima 800×480](browser:/invocations/twhnswgo/artifacts/artifacts/raspberry-weather-800x480.png)
- Reloj 800×480: ![Reloj 800×480](browser:/invocations/oxozbcmz/artifacts/artifacts/raspberry-clock-800x480.png)
- Dispositivos 800×480: ![Dispositivos 800×480](browser:/invocations/nofcoejg/artifacts/artifacts/raspberry-devices-800x480.png)
- Dispositivos 1024×600: ![Dispositivos 1024×600](browser:/invocations/tbbyylti/artifacts/artifacts/raspberry-devices-1024x600.png)
- Configuración 800×480: ![Configuración 800×480](browser:/invocations/neroqusf/artifacts/artifacts/raspberry-settings-800x480.png)
- Tutorial 800×480: ![Tutorial 800×480](browser:/invocations/yxqpzulg/artifacts/artifacts/raspberry-tutorial-800x480.png)

Cada pantalla entra completa en ambas resoluciones, sin scroll ni elementos recortados, preservando el modo Windows gracias al uso de selectores `body[data-mode="raspberry"]`.
