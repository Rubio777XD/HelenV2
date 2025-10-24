# HELEN

HELEN es una experiencia de asistente doméstico controlada por gestos. El backend escrito en Flask/Socket.IO expone los eventos de cámara y la lógica de reconocimiento de gestos (H, C, R, I) mientras que el frontend web muestra el tutorial, alarmas, temporizador y dispositivos conectados. Este repositorio contiene tanto el backend, como el modelo de detección de gestos y los recursos de la interfaz.

## Instalación

Selecciona la guía correspondiente a tu plataforma:

- [Windows 10/11 (ejecutable y instalador Inno Setup)](packaging/README-build-win.md)
- [Raspberry Pi 5 – flujo Bookworm optimizado](packaging-pi/README-PI.md)
- [Raspberry Pi 4/5 con Raspberry Pi OS 64-bit (legacy)](packaging-pi/README-raspi.md)

Cada guía incluye los requisitos, comandos y validaciones específicas para garantizar que HELEN se ejecute con cámara real. En la versión para Raspberry Pi 5 el frontend se abre en **ventana normal de Chromium** (sin modo kiosko) y el flujo completo se reduce a dos pasos:

```bash
packaging-pi/setup_pi.sh   # instalación idempotente
packaging-pi/run_pi.sh     # arranque + Chromium
```

### Parámetros sugeridos en Raspberry Pi 5

Los valores siguientes mantienen un equilibrio entre estabilidad y consumo de CPU (~35–45 %) con cámaras UVC como OBSBOT Tiny 2 Lite:

| Flag / variable         | Valor recomendado |
|-------------------------|-------------------|
| `--detection-confidence`| `0.88`            |
| `--tracking-confidence` | `0.86`            |
| `--poll-interval`       | `0.05`            |
| `--frame-stride`        | `3`               |
| `--camera-index`        | `auto` o `/dev/video0` |

El flag `--camera-index` acepta tanto números como rutas `/dev/videoX`. El backend detecta automáticamente si la cámara es UVC o CSI, fuerza MJPG cuando es posible y expone la telemetría completa en [`/health`](http://localhost:5000/health): backend elegido, formato y FPS de captura/proceso/presentación, así como el estado del clasificador y la cantidad de clientes SSE.
