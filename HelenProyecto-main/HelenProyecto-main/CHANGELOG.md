# CHANGELOG

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
