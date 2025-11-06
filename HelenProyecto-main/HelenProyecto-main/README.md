# HELEN

HELEN es un asistente doméstico controlado por gestos compuesto por un backend en Python (Flask + Socket.IO) y una
interfaz web optimizada para ejecutarse en Google Chrome o Chromium. A partir de esta versión se abandona cualquier
flujo de empaquetado en ejecutables: el proyecto se distribuye como código fuente y se ejecuta directamente con Python.

## Documentación principal

- [HELEN – Guía completa para ejecutar en Chrome (Windows)](README-windows-chrome.md)
- [HELEN – Guía completa para ejecutar en Chrome (Linux / Raspberry Pi)](README-linux-rpi-chrome.md)
- [CHANGELOG](CHANGELOG.md)

Cada guía cubre requisitos, instalación, comandos de ejecución, validaciones manuales y solución de problemas específicas
por plataforma. El CHANGELOG detalla cualquier eliminación o movimiento de archivos legacy relacionados con empaquetado o
scripts obsoletos.

## Arquitectura resumida

```
+----------------------+        Eventos / SSE        +-------------------------+
|  Google Chrome /     |  <----------------------->  |  backendHelen.server    |
|  Chromium (Frontend) |                            |  Flask + Socket.IO      |
+----------+-----------+                            +-----------+-------------+
           |  HTTP/WebSocket                                   |
           v                                                   v
   UI, timers, tutoriales                              MediaPipe / OpenCV, cámara
```

- **Frontend**: vive en `helen/` y se sirve directamente desde Flask. Las preferencias de UI (p. ej. color de fondo) se
guardan en `localStorage` para que persistan entre reinicios.
- **Backend**: contenido en `backendHelen/`, expone la API REST, streaming de video y diagnósticos.

## Scripts de apoyo vigentes

Los únicos scripts mantenidos para automatizar la instalación y ejecución son los que residen en `scripts/`:

- `scripts/helen-run.ps1` / `scripts/helen-run.bat`
- `scripts/setup-windows.ps1`
- `scripts/run-windows.ps1`
- `scripts/setup-pi.sh`
- `scripts/run-pi.sh`

El resto de los scripts históricos (`run*.bat`, `run*.sh`) fueron archivados en `legacy/` y no reciben soporte.

## Activos legacy

Todo el material relacionado con empaquetado (PyInstaller, Inno Setup, kioskos heredados, etc.) ahora vive en el
_directorio_ [`legacy/`](legacy/README_legacy.md). Conserva la estructura original únicamente como referencia para equipos
que aún dependan de esos artefactos, pero no forma parte del flujo oficial.

Para cualquier contribución nueva utiliza las guías actualizadas y mantén sincronizados los cambios funcionales entre el
código y la documentación.
