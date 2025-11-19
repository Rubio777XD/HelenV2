# CHANGELOG

## 2024-05-14 – Documentación multiplataforma y guías actualizadas
- Se añadió `README-linux-rpi-chrome.md` con el flujo detallado para Ubuntu/Raspberry Pi usando el modelo TF.
- Se corrigió `scripts/setup-pi.sh` para que utilice el instalador soportado en `legacy/packaging/linux-rpi/`.
- Se creó este CHANGELOG para centralizar los cambios relevantes solicitados por el equipo de HELEN.

## 2024-05-12 – Migración a la tubería de video en TensorFlow
- `backendHelen/server.py` ahora delega toda la inferencia en `TensorFlowGesturePipeline` y desecha cualquier dependencia del modelo RN.
- Se actualizaron `requirements.txt`, scripts de Windows (`scripts/run-windows.ps1`, `scripts/helen-run.ps1`) y la documentación (`README.md`, `README-windows-chrome.md`).
- Se reescribieron las pruebas (`tests/test_backend_api.py`, `tests/test_model_pipeline.py`) para validar el flujo basado en gestos de video.
