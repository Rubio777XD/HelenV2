# Frontend bridge service

Este módulo proporciona una forma de conectar el modelo de gestos con una
interfaz web (HTML/CSS/JS puro) que viva en otra carpeta o en una Raspberry Pi.
La idea es reutilizar el pipeline existente **sin modificar nada del modelo** y
exponer los resultados mediante REST, Server-Sent Events (SSE) y Socket.IO.

## Componentes

- `GestureInferenceService`: levanta el modelo, captura video, ejecuta la
  inferencia y mantiene un conjunto de oyentes. Puedes reutilizarlo desde otros
  procesos (por ejemplo, un demonio controlado por `systemd`).
- `server.py`: aplica la clase anterior dentro de un servidor Flask + Socket.IO
  con rutas REST y un endpoint SSE listo para ser consumido por el frontend.

## Endpoints principales

| Método | Ruta                 | Descripción |
|--------|---------------------|-------------|
| GET    | `/api/status`       | Resumen del estado actual y último gesto. |
| GET    | `/api/gestures`     | Diccionario índice → etiqueta disponible. |
| POST   | `/api/inference/start` | Arranca el hilo de inferencia (idempotente). |
| POST   | `/api/inference/stop`  | Detiene el hilo de inferencia. |
| GET    | `/api/stream`       | Flujo SSE con predicciones confirmadas. |
| POST   | `/api/config`       | Ajusta parámetros (`confidence_threshold`, etc.). |

Además, el servidor expone un namespace Socket.IO (`/gestures`) que emite el
evento `gesture_prediction` cada vez que se detecta un gesto con suficiente
confianza. Esto permite a la UI reaccionar en tiempo real sin necesidad de
preguntas HTTP extra.

## Ejecución local

```bash
pip install -r video_gesture_model/requirements.txt
pip install flask flask-socketio flask-cors
python -m frontend_bridge.server --model-path /ruta/al/modelo
```

Si no indicas `--model-path`, el servicio buscará el SavedModel más reciente en
`video_gesture_model/data/models`. Asegúrate de copiar también el `labels.json`
correspondiente.

## Integración con Raspberry Pi

1. Activa la cámara y permisos necesarios (por ejemplo, `sudo raspi-config`).
2. Crea un servicio `systemd` que ejecute `python -m frontend_bridge.server` al
   arranque. El servidor escucha en `0.0.0.0:8000` por defecto.
3. Desde Chromium en modo kiosk, el frontend puede:
   - Suscribirse al SSE: `const evtSource = new EventSource('/api/stream');`
   - Conectarse por Socket.IO: `io('/gestures').on('gesture_prediction', cb)`.
   - Pedir el listado de gestos soportados vía `fetch('/api/gestures')` para
     mapear las acciones (hay 8 gestos previstos en el modelo actual).
4. Asocia cada gesto a la acción correspondiente dentro del JavaScript del
   frontend.

Con esta estructura, el modelo sigue encapsulado y el frontend queda totalmente
agregado sobre la capa de servicios.
