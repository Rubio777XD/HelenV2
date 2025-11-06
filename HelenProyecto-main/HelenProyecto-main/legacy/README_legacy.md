# HELEN Legacy Assets

Este directorio conserva scripts y archivos de empaquetado que ya no forman parte del flujo soportado de HELEN. Se mantienen
únicamente como referencia histórica para instalaciones que dependan de los empaquetados antiguos generados con PyInstaller,
Inno Setup o scripts personalizados de ejecución.

> **Importante:** Las guías oficiales para ejecutar HELEN ahora se centran exclusivamente en abrir el backend en Python y
> consumir la interfaz web desde Google Chrome o Chromium. No se dará soporte ni mantenimiento a los instaladores o
> scripts ubicados aquí.

## Contenido

- `packaging/windows/`: especificaciones de PyInstaller, configuraciones de Inno Setup y requirements antiguos para crear
  ejecutables de Windows.
- `packaging/linux-rpi/`: scripts de automatización orientados a imágenes legacy para Raspberry Pi.
- `scripts/`: wrappers históricos (`run*.bat`, `run*.sh`) que asumían estructuras de carpetas anteriores o entornos no
  compatibles con el flujo actual.

Si necesitas recuperar alguno de estos procesos, revisa el historial de cambios o considera portar la lógica a los scripts
de `scripts/` y a las guías actualizadas descritas en `README-windows-chrome.md` y `README-linux-rpi-chrome.md`.
