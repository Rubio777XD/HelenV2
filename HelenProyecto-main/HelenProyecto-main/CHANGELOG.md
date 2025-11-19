# CHANGELOG

## 2024-11-05

### Añadido
- Clasificador TensorFlow LSTM habilitado por defecto cuando `HELEN_MODEL_BACKEND` no está definido, conservando compatibilidad con la seña de activación y el contrato del `DecisionEngine`.
- Normalización de etiquetas en `TensorFlowSequenceGestureClassifier` para mapear `activar`→`Start` y mantener los alias históricos de gestos.
- Scripts nuevos para Windows (`scripts/start-helen-windows-tf.bat`, `scripts/start-frontend-chrome-windows.bat`, `scripts/start-helen-all-windows.bat`) y Raspberry Pi 5 (`scripts/start-helen-backend-pi5.sh`, `scripts/start-helen-frontend-pi5.sh`).
- Unidad de ejemplo `system_scripts/helen-pi5.service` y guía `docs/RaspberryPi5_TF.md` con pasos de instalación y arranque en modo kiosk.

### Cambiado
- `backendHelen/requirements.txt` incluye TensorFlow CPU y NumPy para soportar el backend LSTM sin dependencias externas adicionales.

## 2024-06-03

### Añadido
- **Interfaz de personalización**: se incorporó control global de tamaño de letra y un modo de rendimiento que desactiva
  animaciones al activar la clase `perf-mode`. Ambos ajustes se sincronizan vía `localStorage` y respetan
  `prefers-reduced-motion`.

### Cambiado
- **Temas dinámicos**: el selector de color ahora aplica presets completos (fondos, superficies, halos y acentos) mediante
  variables `--helen-*`, lo que garantiza que todas las pantallas adopten el nuevo tema de inmediato.
- **scripts/helen-run.ps1** y **scripts/run-windows.ps1**: se corrige el manejo de argumentos adicionales con un parser que
  respeta comillas, se generan listas para `Start-Process` y se preservan los valores por defecto incluso con rutas que
  contienen espacios.
- **scripts/run-pi.sh**: apunta al script soportado en `legacy/packaging/linux-rpi/run_pi.sh` para el flujo de kiosko.

### Documentación
- Guías de Windows y Linux actualizadas para referirse a **Configuración → Personalización** en lugar del modo Raspberry.

## 2024-05-29

### Añadido
- **`scripts/helen-run.ps1` y `scripts/helen-run.bat`**: flujo de “un comando” que detecta Python 3.11, prepara `.venv`,
  instala dependencias, define variables de entorno (`HELEN_CAMERA_INDEX`, `HELEN_BACKEND_EXTRA_ARGS`) y lanza el backend
  antes de abrir Chrome. Expone parámetros (`-Port`, `-CameraIndex`, `-ExtraArgs`, `-SkipBrowser`) documentados en la guía
  de Windows.

### Cambiado
- **`scripts/run-windows.ps1`**: corrige la redirección duplicada de stdout/stderr, escribe logs separados
  (`backend-*.out.log`/`backend-*.err.log`), aplica DirectShow + 1280x720 + `--frame-stride 2` + `--poll-interval 0.08` por
  defecto y permite omitir el navegador con `-SkipBrowser`.
- **`backendHelen/camera_probe.py`**: añade detección explícita de plataforma, backend DirectShow por defecto en Windows,
  mapeo de `--camera-backend` (`directshow`, `dshow`, `v4l2`), sugerencias cuando la cámara falla y utilidades para
  resolver flags (`normalize_backend_name`, `resolve_backend_flag`, `preferred_backend_order`).
- **`backendHelen/server.py`**: acepta `--camera-backend/--camera-width/--camera-height`, propaga los overrides al stream de
  cámara y registra sugerencias cuando no se puede abrir el dispositivo.
- **Documentación** (`README-windows-chrome.md`, `README.md`): describe el flujo de “1 comando”, los nuevos parámetros y
  la ubicación de los logs.

## 2024-05-15

### Eliminado / Archivado
- **`packaging/` y `packaging-pi/`**: movidos a `legacy/packaging/` porque el flujo oficial dejó de distribuir
  instaladores PyInstaller e Inno Setup. Usa las nuevas guías de ejecución en Chrome para preparar entornos de
  Windows y Linux/Raspberry Pi.
- **Scripts `run*.bat` y `run*.sh` en la raíz**: reubicados en `legacy/scripts/` al no representar el flujo soportado.
  Los scripts mantenidos viven en `scripts/` y se documentan en las guías actualizadas.

### Añadido
- **`README-windows-chrome.md`**: guía completa para ejecutar HELEN en Windows usando únicamente Python y Chrome.
- **`README-linux-rpi-chrome.md`**: instrucciones detalladas para Debian/Ubuntu/Raspberry Pi OS con Chromium/Chrome.
- **`legacy/README_legacy.md`**: describe el estado no soportado de los activos archivados.
- **`CHANGELOG.md`**: documento oficial para rastrear cambios estructurales y de documentación.

### Cambiado
- **`README.md`**: ahora enlaza únicamente a las guías de ejecución en Chrome y aclara qué scripts siguen bajo soporte.
- **Tema de fondo**: el selector de color en Configuración actualiza la variable CSS `--bg` tanto en Linux/Raspberry Pi
  como en Windows, conservando halos y animaciones existentes.
