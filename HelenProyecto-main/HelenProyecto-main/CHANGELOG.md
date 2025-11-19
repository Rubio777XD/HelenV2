# CHANGELOG

# 2024-05-22 – README de instalación paso a paso para Raspberry Pi 5
- Se añadió `README-raspi-install.md`, una guía lineal desde el formateo de la Pi
  hasta la ejecución en modo kiosk, cumpliendo con la solicitud de documentar el
  proceso completo de instalación.

# 2024-05-21 – Guía dedicada para Raspberry Pi 5
- `README-linux-rpi-chrome.md` ahora se enfoca exclusivamente en Raspberry Pi 5 con
  Raspberry Pi OS 64 bits, eliminando referencias a otras distribuciones Linux para
  evitar confusiones.

# 2024-05-20 – Flujo Raspberry Pi 5 y documentación actualizada
- `scripts/setup-pi.sh` y `scripts/run-pi.sh` ahora son scripts nativos (sin `legacy/`)
  que preparan `.venv`, exportan `HELEN_*` y permiten lanzar Chromium/kiosk.
- Se añadieron `HELEN_CAMERA_WIDTH/HEIGHT` para ajustar la resolución de captura vía
  variables de entorno.
- Nuevo `system_scripts/chromium-kiosk.service` y guía `README-linux-rpi-chrome.md`
  detallando la instalación en Raspberry Pi 5 con la cámara Obsbot Tiny 2.

## 2024-05-14 – Documentación multiplataforma y guías actualizadas
- Se añadió `README-linux-rpi-chrome.md` con el flujo detallado para Ubuntu/Raspberry Pi usando el modelo TF.
- Se corrigió `scripts/setup-pi.sh` para que utilice el instalador soportado en `legacy/packaging/linux-rpi/`.
- Se creó este CHANGELOG para centralizar los cambios relevantes solicitados por el equipo de HELEN.

## 2024-05-12 – Migración a la tubería de video en TensorFlow
- `backendHelen/server.py` ahora delega toda la inferencia en `TensorFlowGesturePipeline` y desecha cualquier dependencia del modelo RN.
- Se actualizaron `requirements.txt`, scripts de Windows (`scripts/run-windows.ps1`, `scripts/helen-run.ps1`) y la documentación (`README.md`, `README-windows-chrome.md`).
- Se reescribieron las pruebas (`tests/test_backend_api.py`, `tests/test_model_pipeline.py`) para validar el flujo basado en gestos de video.
