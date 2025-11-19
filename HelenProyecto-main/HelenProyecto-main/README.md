# HELEN – Asistente visual por gestos (backend LSTM)

## Descripción general
HELEN es un asistente tipo "Echo Show" para personas sordas. Captura gestos de mano con MediaPipe y OpenCV, forma secuencias compactas de **24–48 fotogramas × 126 características** (21 landmarks × 3 coords × 2 manos) y las reexpande a **96 frames** para alimentar el modelo **LSTM en TensorFlow**. El backend expone eventos SSE para que la interfaz web en `helen/` encienda el anillo de activación y navegue entre pantallas al reconocer los gestos.

- Backend: Python (Flask + SSE) en `backendHelen/`, clasificador principal `TensorFlowSequenceGestureClassifier`.
- Frontend: HTML/JS servido desde `/`, escucha `/events` y mantiene el mismo contrato histórico de eventos (`message` con `gesture`, `score`, `active`, etc.).
- Modelo: se carga automáticamente el SavedModel más reciente de `Hellen_model_TF/video_gesture_model/data/models/gesture_model_*`.
- Backend efectivo: **siempre LSTM**. La variable `HELEN_MODEL_BACKEND` se ignora salvo para registrar warnings. XGBoost queda como código legacy no ejecutado.

## Requisitos
- Python 3.10 (64 bits). TensorFlow 2.15 no es compatible con Python 3.11+ y el backend está fijado a `python3.10`/`py -3.10`.
- TensorFlow CPU `tensorflow==2.15.0` (incluido en `requirements.txt`).
- Dependencias clave: `mediapipe`, `opencv-python`, `numpy`, `Flask`, `Flask-SocketIO`.
- Cámara compatible con OpenCV/MediaPipe.
- Navegador: Google Chrome (Windows) o Chromium (Raspberry Pi) para la UI.

## Instalación en Windows (paso a paso)
1. Clona el repositorio y entra al directorio raíz.
2. Crea el entorno virtual: `py -3.10 -m venv .venv`.
3. Actívalo: `\.venv\Scripts\activate`.
4. Instala dependencias: `pip install -r requirements.txt`.
5. Verifica el modelo: `py -3.10 scripts/check_tf_model.py` (confirma SavedModel y labels).
6. Ejecuta con LSTM (valor por defecto):
   ```bat
   .\.venv\Scripts\activate
   py -3.10 -m backendHelen.server
   ```
   También puedes usar `scripts\run_windows_lstm.bat` que realiza estos pasos.
7. Abre `http://localhost:5000` en Google Chrome.

### Ejecución rápida con cámara en Windows
- Conecta la cámara y prueba: `py -3.10 -m backendHelen.server --camera 0 --detection-profile fast --poll-interval 0.03`.
- Para mayor estabilidad usa `--detection-profile normal` si hay mucho ruido visual.

## Instalación y ejecución en Raspberry Pi 5
1. Instala Python 3.10 y crea entorno virtual: `python3.10 -m venv .venv && source .venv/bin/activate`.
2. Instala dependencias (`tensorflow` CPU o wheel compatible con ARM) y librerías del requirements: `pip install -r requirements.txt`.
3. Comprueba el modelo: `python3.10 scripts/check_tf_model.py`.
4. Lanza el backend y Chromium kiosk con `bash scripts/run_pi5_lstm.sh`.
5. Para modo kiosk persistente, invoca el script desde un servicio `systemd` o un `.desktop` que ejecute Chromium apuntando a `http://localhost:5000`.

## Uso del backend LSTM
- Buffer: la primera predicción requiere 24–32 frames reales según el perfil elegido (fast o normal). El `poll_interval` por defecto es **0.03s** y el `frame_stride` es **1**, por lo que la ventana se llena en ~1s manteniendo estabilidad; el resto se rellena hasta 96 frames para mantener la forma del modelo.
- Entrada esperada del modelo: tensor `(1, 96, 126)` en `float32`, rellenado a partir de la ventana de 24/48 frames reales.
- Si MediaPipe produce 42 features (x, y de una mano), el backend rellena `z=0` y duplica la mano para simular dos manos mientras se captura la otra. La ventana compacta siempre preserva la forma final `(1, 96, 126)` tras el padding.
- Si el modelo TensorFlow no carga, el servidor usa un clasificador **dummy** (siempre `score=0.0`) para no caer en XGBoost ni detener el servicio.

## Prueba rápida (modo fast)
1. Activa el entorno: `source .venv/bin/activate` (o `\.venv\Scripts\activate` en Windows).
2. Lanza el backend en modo rápido: `HELEN_DEBUG=0 HELEN_DETECTION_PROFILE=fast python3.10 -m backendHelen.server --poll-interval 0.03 --process-every-n 1`.
3. Acerca la mano a cámara y mantén la seña ~1s; el backend usará ventanas de 24 frames, un solo voto y umbral 0.45/0.30 para evitar falsos positivos sin ser estricto.
4. Observa en consola las etiquetas emitidas; el frontend en `http://localhost:5000` reflejará los cambios casi inmediato aun con FPS bajos.

## Modo normal equilibrado
- Valor por defecto (`HELEN_DETECTION_PROFILE=normal` o sin definir).
- Ventana de 48 frames (min. 32 reales), consenso 2 votos, umbral 0.5/0.30.
- Usa `py -3.10 -m backendHelen.server --detection-profile normal` si quieres fijarlo por CLI en Windows.

## Gestos y anillo de activación
- Las etiquetas se toman de `labels.json` del modelo. La seña de activación suele mapear a `Start`.
- El endpoint `/events` mantiene el contrato SSE existente. Cuando `active=true` se enciende el anillo en el frontend; `active=false` lo apaga.
- La `DecisionEngine` usa umbrales moderados para el LSTM: `global_min_score=0.30`, umbral de activación 0.45–0.5 y liberación 0.30 según perfil, consenso de 1 voto (fast) o 2 votos (normal) sin filtros geométricos adicionales.

## Solución de problemas
- **No se enciende el anillo**: verifica la cámara, revisa logs del backend, confirma que `scripts/check_tf_model.py` carga el modelo y que `HELEN_MODEL_BACKEND` no apunta a backends legacy. Ajusta el perfil (`fast` o `normal`) si el entorno es muy ruidoso.
- **La página no carga**: asegúrate de que `py -3.10 -m backendHelen.server` esté corriendo en `http://localhost:5000` y que el navegador apunte a esa URL.
- **Warnings de TensorFlow (AVX/AVX2)**: son informativos en CPU; no bloquean la inferencia.
- **Modelo faltante**: copia un SavedModel dentro de `Hellen_model_TF/video_gesture_model/data/models/gesture_model_*`.

## Cómo depurar si las señas no se detectan (LSTM)
Activa el modo de depuración para ver por consola cada ventana de 24 frames (rellenadas a 96 antes de inferir) y las razones de descarte de la `DecisionEngine`:

```bat
cd C:\...\HelenProyecto-main\HelenProyecto-main
.\.venv\Scripts\activate
set HELEN_MODEL_BACKEND=lstm
set HELEN_DEBUG=1
set HELEN_PROFILE=debug_lstm
py -3.10 -m backendHelen.server
```

- **Qué verás en logs**: líneas `Inferencia LSTM` (score por etiqueta), `DecisionEngine` (motivo exacto de descarte) y `EMIT gesture=...` cuando se envía SSE al frontend.
- **Si el modelo nunca sube de 0.1**: revisa `reports/gesture_session_report.md/json`, sección "Máximo score observado" para confirmar si el modelo está saturado.
- **Si el modelo predice pero se descarta**: la tabla "Ejemplos recientes de decisiones" muestra `reason` (ej. `score_below_threshold`, `consensus_short`).
- **Si no hay landmarks**: el reporte contará descartes `no_hand_detected`; revisa alineación y luz.
- **Prueba offline**: ejecuta `py -3.10 scripts/debug_lstm_offline.py --profile debug_lstm` para simular secuencias sin cámara.

## Scripts disponibles
- `scripts/check_tf_model.py`: confirma carga del SavedModel y muestra etiquetas.
- `scripts/run_windows_lstm.bat`: arranca backend en Windows con `HELEN_MODEL_BACKEND=lstm`.
- `scripts/run_pi5_lstm.sh`: arranca backend en Raspberry Pi 5 y abre Chromium en modo kiosk.

## Registro de cambios
Consulta `CHANGELOG.md` para un historial resumido de modificaciones.
