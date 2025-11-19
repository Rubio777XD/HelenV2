# HELEN – Asistente visual por gestos (backend LSTM)

## Descripción general
HELEN es un asistente tipo "Echo Show" para personas sordas. Captura gestos de mano con MediaPipe y OpenCV, forma secuencias de **96 fotogramas × 126 características** (21 landmarks × 3 coords × 2 manos) y las clasifica con un modelo **LSTM en TensorFlow**. El backend expone eventos SSE para que la interfaz web en `helen/` encienda el anillo de activación y navegue entre pantallas al reconocer los gestos.

- Backend: Python (Flask + SSE) en `backendHelen/`, clasificador principal `TensorFlowSequenceGestureClassifier`.
- Frontend: HTML/JS servido desde `/`, escucha `/events` y mantiene el mismo contrato histórico de eventos (`message` con `gesture`, `score`, `active`, etc.).
- Modelo: se carga automáticamente el SavedModel más reciente de `Hellen_model_TF/video_gesture_model/data/models/gesture_model_*`.
- Backend efectivo: **siempre LSTM**. La variable `HELEN_MODEL_BACKEND` se ignora salvo para registrar warnings. XGBoost queda como código legacy no ejecutado.

## Requisitos
- Python 3.10 o superior.
- TensorFlow CPU `tensorflow==2.15.0` (incluido en `requirements.txt`).
- Dependencias clave: `mediapipe`, `opencv-python`, `numpy`, `Flask`, `Flask-SocketIO`.
- Cámara compatible con OpenCV/MediaPipe.
- Navegador: Google Chrome (Windows) o Chromium (Raspberry Pi) para la UI.

## Instalación en Windows (paso a paso)
1. Clona el repositorio y entra al directorio raíz.
2. Crea el entorno virtual: `python -m venv .venv`.
3. Actívalo: `\.venv\Scripts\activate`.
4. Instala dependencias: `pip install -r requirements.txt`.
5. Verifica el modelo: `python scripts/check_tf_model.py` (confirma SavedModel y labels).
6. Ejecuta con LSTM (valor por defecto):
   ```bat
   .\.venv\Scripts\activate
   python -m backendHelen.server
   ```
   También puedes usar `scripts\run_windows_lstm.bat` que realiza estos pasos.
7. Abre `http://localhost:5000` en Google Chrome.

## Instalación y ejecución en Raspberry Pi 5
1. Instala Python y crea entorno virtual: `python3 -m venv .venv && source .venv/bin/activate`.
2. Instala dependencias (`tensorflow` CPU o wheel compatible con ARM) y librerías del requirements: `pip install -r requirements.txt`.
3. Comprueba el modelo: `python scripts/check_tf_model.py`.
4. Lanza el backend y Chromium kiosk con `bash scripts/run_pi5_lstm.sh`.
5. Para modo kiosk persistente, invoca el script desde un servicio `systemd` o un `.desktop` que ejecute Chromium apuntando a `http://localhost:5000`.

## Uso del backend LSTM
- Buffer: la primera predicción requiere llenar 96 frames (≈3 segundos a 30 FPS).
- Entrada esperada del modelo: tensor `(1, 96, 126)` en `float32`.
- Si MediaPipe produce 42 features (x, y de una mano), el backend rellena `z=0` y duplica la mano para simular dos manos mientras se captura la otra.
- Si el modelo TensorFlow no carga, el servidor usa un clasificador **dummy** (siempre `score=0.0`) para no caer en XGBoost ni detener el servicio.

## Gestos y anillo de activación
- Las etiquetas se toman de `labels.json` del modelo. La seña de activación suele mapear a `Start`.
- El endpoint `/events` mantiene el contrato SSE existente. Cuando `active=true` se enciende el anillo en el frontend; `active=false` lo apaga.
- La `DecisionEngine` usa umbrales relajados para probar el LSTM: `global_min_score=0.30`, ventanas de consenso de `3` frames con `2` votos mínimos (Clima mantiene ventana 2/voto único) y la geometría de la seña **Start** está desactivada para este backend.

## Solución de problemas
- **No se enciende el anillo**: verifica la cámara, revisa logs del backend, confirma que `scripts/check_tf_model.py` carga el modelo y que `HELEN_MODEL_BACKEND` no apunta a backends legacy.
- **La página no carga**: asegúrate de que `python -m backendHelen.server` esté corriendo en `http://localhost:5000` y que el navegador apunte a esa URL.
- **Warnings de TensorFlow (AVX/AVX2)**: son informativos en CPU; no bloquean la inferencia.
- **Modelo faltante**: copia un SavedModel dentro de `Hellen_model_TF/video_gesture_model/data/models/gesture_model_*`.

## Scripts disponibles
- `scripts/check_tf_model.py`: confirma carga del SavedModel y muestra etiquetas.
- `scripts/run_windows_lstm.bat`: arranca backend en Windows con `HELEN_MODEL_BACKEND=lstm`.
- `scripts/run_pi5_lstm.sh`: arranca backend en Raspberry Pi 5 y abre Chromium en modo kiosk.

## Registro de cambios
Consulta `CHANGELOG.md` para un historial resumido de modificaciones.
