# HELEN en Raspberry Pi 5 (Bookworm optimizado)

Esta guía describe el flujo recomendado para Raspberry Pi 5 con Raspberry Pi OS Bookworm 64-bit. El objetivo es obtener una instalación lista en dos comandos, abrir la interfaz en una ventana normal de Chromium y mantener 24–25 fps con cámaras UVC modernas (por ejemplo, OBSBOT Tiny 2 Lite) sin pasos manuales adicionales.

> Para instalaciones heredadas en Pi 4 o despliegues que continúan usando kiosko consulta [README-raspi.md](README-raspi.md).

## 1. Requisitos

- Raspberry Pi 5 con fuente oficial de 5A.
- Python 3.11 del sistema (Bookworm lo incluye de forma predeterminada).
- Cámara UVC (recomendada) o módulo CSI compatible con `libcamera`.
- Repositorio clonado en `${HOME}/HelenV2/HelenProyecto-main/HelenProyecto-main`.
- Dataset `Hellen_model_RN/data.pickle` copiado en esa ruta (opcional pero recomendado para el modelo actual).

## 2. Instalación (`dos comandos`)

Desde la raíz del repositorio ejecuta:

```bash
bash ./scripts/setup-pi.sh
bash ./scripts/run-pi.sh
```

`scripts/setup-pi.sh` actúa como envoltura de `packaging-pi/setup_pi.sh` pero conserva los valores optimizados para Pi 5:

- Instala dependencias de sistema con `apt` resolviendo automáticamente `libavcodec59`/`libavcodec-extra59`.
- Detecta `chromium` o `chromium-browser`, instala `libcamera0.5`, `rpicam-apps-core` y limpia paquetes obsoletos.
- Crea `.venv/`, actualiza `pip`, `setuptools`, `wheel` y aplica los pines de [requirements-pi.txt](requirements-pi.txt) (Flask 3.0.3, Flask-SocketIO 5.3.6, eventlet 0.36.1, numpy 1.26.4, opencv-python 4.9.0.80, mediapipe 0.10.18, etc.).
- Guarda un snapshot de dependencias en `reports/logs/pi/vision-stack-*.json` validando que MediaPipe pueda inicializar `Hands`.

`scripts/run-pi.sh` (que invoca `packaging-pi/run_pi.sh`) aplica los defaults específicos para Pi 5:

- Ajusta `--poll-interval` a 0.04 s y `--frame-stride` a 3 salvo que definas `POLL_INTERVAL`/`HELEN_BACKEND_EXTRA_ARGS`.
- Ejecuta `backendHelen.camera_probe.ensure_camera_selection()` para elegir la mejor cámara disponible (índice o `/dev/videoX`), degradando formatos y resoluciones automáticamente y almacenando los resultados en `reports/logs/pi/camera-selection-*.json`.
- Lanza `backendHelen.server` y espera a `http://127.0.0.1:5000/health` antes de abrir Chromium en modo ventana. El kiosko se puede reactivar con `HELEN_PI_KIOSK=1`.
- Registra los logs en `reports/logs/pi/backend-*.log`, `chromium-*.log` y `camera-bootstrap.log`.

Finaliza con `Ctrl+C`; el script detendrá Chromium y el backend de forma ordenada.

## 3. Validación rápida

1. **Estado de cámara**: ejecuta `curl http://localhost:5000/health | jq '.camera_ok, .camera_resolution, .camera_pixel_format'`. Debe mostrar `true`, `"1280x720"` (o el modo seleccionado) y el formato elegido (`"MJPG"` o `"YUYV"`).
2. **Tutorial**: confirma que la SPA abre en una ventana normal (`chromium` sin `--kiosk`).
3. **Rendimiento**: `top` debería mostrar ~35–45 % de CPU total y `vcgencmd measure_temp` ≤75 °C. Ajusta `POLL_INTERVAL=0.035` si necesitas más fps o `0.045` si quieres bajar consumo.
4. **Logs**: revisa `reports/logs/pi/backend-*.log` para verificar que la cámara usa `v4l2` (preferido) o `gstreamer` si V4L2 falla.

## 4. Servicios systemd (opcional)

Si deseas arranque automático, utiliza los mismos archivos que la guía legacy pero asegúrate de actualizar `packaging-pi/helen.service` y `packaging-pi/kiosk.service` con tus rutas:

- `WorkingDirectory=%h/HelenV2/HelenProyecto-main/HelenProyecto-main`
- `ExecStart=%h/HelenV2/HelenProyecto-main/HelenProyecto-main/.venv/bin/python -m backendHelen.server --host 0.0.0.0 --port=5000`
- `kiosk.service` usa `/usr/bin/env bash -lc '...'` para elegir `chromium` o `chromium-browser`. Si prefieres mantener la ventana, añade `Environment=HELEN_PI_KIOSK=0`.

Después de copiar los servicios a `/etc/systemd/system/`:

```bash
sudo systemctl daemon-reload
sudo systemctl enable helen.service kiosk.service
sudo systemctl start helen.service kiosk.service
systemctl status helen.service kiosk.service
journalctl -u helen.service -u kiosk.service -n 50
```

Comprueba `/health` tras el arranque para confirmar `"camera_ok": true`.

## 5. Troubleshooting específico Pi 5

| Síntoma | Diagnóstico | Acción |
|---------|-------------|--------|
| `camera_ok:false` | La auto detección falló. | Revisa `reports/logs/pi/camera-bootstrap.log`, fuerza `HELEN_CAMERA_INDEX=/dev/video0` o `python -m backendHelen.camera_probe --list`. |
| FPS inestables (picos a <15 fps) | `/health` muestra degradación frecuente. | Incrementa `POLL_INTERVAL`, fuerza MJPG ejecutando `v4l2-ctl --set-fmt-video=width=1280,height=720,pixelformat=MJPG`. |
| Chromium abre en kiosko | `HELEN_PI_KIOSK` no está definido o vale `1`. | Exporta `HELEN_PI_KIOSK=0` antes de `scripts/run-pi.sh` o ajusta el servicio systemd. |
| `libavcodec` bloquea `apt` | Apto no puede instalar `libavcodec-extra59`. | Repite `scripts/setup-pi.sh`; removerá la variante conflictiva y reinstalará la correcta. |
| Falta soporte `chromium` | `command -v chromium` falla. | Instala `chromium-browser` (`sudo apt install chromium-browser`) o modifica `HELEN_CHROMIUM_BIN` antes de ejecutar el script. |

## 6. Recursos adicionales

- [README principal](../README.md)
- [Guía legacy Pi 4/5](README-raspi.md)
- [CHANGELOG de documentación](../CHANGELOG_DOCS.md)

Siguiendo estos pasos obtendrás la experiencia completa de HELEN en Raspberry Pi 5 con un flujo reproducible de instalación y ejecución de dos comandos.
