# CHANGELOG_DOCS

## 2025-10-24

### README.md
- Reescrito con la guía de "dos comandos" por plataforma (Windows y Raspberry Pi).
- Documentado el pipeline de auto detección de cámara, valores por plataforma y la semántica completa de `/health`.
- Añadida matriz de pruebas, validaciones rápidas y sección de troubleshooting con comandos específicos (`systemctl`, `journalctl`, `v4l2-ctl`, `ffmpeg`).

### packaging/README-build-win.md
- Alineado al flujo simplificado: `scripts\setup-windows.ps1` y `scripts\run-windows.ps1` como base.
- Separadas las secciones de ejecución y empaquetado (PyInstaller + Inno Setup).
- Actualizado el troubleshooting y la referencia a `requirements.txt`/`packaging/requirements-win.txt`.

### packaging-pi/README-PI.md
- Nueva guía para Raspberry Pi 5 (Bookworm) describiendo instalación en ventana normal, defaults y métricas objetivo.
- Documentadas variables de entorno (`HELEN_PI_KIOSK`, `HELEN_CHROMIUM_FLAGS`) y los pasos de validación.

### packaging-pi/README-raspi.md
- Actualizado para reflejar el flujo legacy con kiosko, conflictos `libavcodec`, logs y rutas definitivas.
- Añadidas instrucciones de systemd con `WorkingDirectory` y `ExecStart` basados en `.venv`.

### packaging-pi/run_pi.sh
- Detección automática de modo ventana/kiosko según el modelo (Pi 5 → ventana) con `HELEN_PI_KIOSK` sobrescribible.
- Soporte para `HELEN_CHROMIUM_FLAGS` y reutilización del mismo set de flags que la guía.

### packaging-pi/setup_pi.sh
- (Sin cambios funcionales) Reutiliza los nuevos requisitos mínimos.

### packaging-pi/helen.service / kiosk.service
- Actualizados `WorkingDirectory` y `ExecStart` a `%h/HelenV2/HelenProyecto-main/HelenProyecto-main` y `.venv/bin/python`.
- `kiosk.service` usa `bash -lc` para elegir `chromium`/`chromium-browser` y honrar `HELEN_CHROMIUM_FLAGS`.

### requirements.txt / packaging/requirements-win.txt / packaging-pi/requirements-pi.txt
- Reducidos a las dependencias estrictamente usadas por el runtime (Flask/Socket.IO, eventlet, numpy, opencv, mediapipe, protobuf).
- Sincronizados los pines entre plataformas para mantener paridad.

### packaging/helen_backend.spec
- Eliminadas referencias a `xgboost`, `scipy`, `sounddevice` y `sklearn` para que el bundle coincida con el runtime actual.
- Simplificados `hiddenimports` a la lista mínima necesaria (incluyendo `numpy.fft`).

## Requisitos fijados / riesgos conocidos
- `mediapipe==0.10.18`, `opencv-python==4.9.0.80`, `numpy==1.26.4`, `protobuf==4.25.3` validados en Bookworm ARM64 y Windows 10/11.
- MediaPipe requiere Python ≤3.11 en ARM64; el script de setup aborta si detecta versiones no soportadas.
- `chromium` puede variar de nombre (`chromium` vs `chromium-browser`); los scripts y servicios prueban ambos.

## Verificaciones sugeridas
1. **Line endings**: `python tools/check_line_endings.py` (o `python - <<'PY' ...` según README) para confirmar `.sh` en LF y `.ps1` en CRLF.
2. **Linter de enlaces**: `python tools/check_md_links.py` (ver sección de validación automática abajo).
3. **Smoke test backend**: `python -m backendHelen.server --no-camera --host 127.0.0.1 --port 8765` y comprobar `http://127.0.0.1:8765/health`.
4. **Pi**: ejecutar `bash ./scripts/setup-pi.sh && HELEN_NO_UI=1 bash ./scripts/run-pi.sh` y revisar `reports/logs/pi/`.
5. **Windows**: `powershell -ExecutionPolicy Bypass -File .\scripts\setup-windows.ps1` seguido de `run-windows.ps1`.

Mantén este changelog actualizado cuando se modifique documentación, scripts de empaquetado o listas de dependencias.
