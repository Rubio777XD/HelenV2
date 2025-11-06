# CHANGELOG

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
