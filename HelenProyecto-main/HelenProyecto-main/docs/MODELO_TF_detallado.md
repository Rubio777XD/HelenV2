# Modelo de gestos en video (TensorFlow/Keras)

## 1. Arquitectura general del modelo
- **Ubicación de los artefactos entrenados**: los modelos se guardan como **SavedModel** en `video_gesture_model/data/models/gesture_model_YYYYMMDD_HHMMSS`, acompañados de `training_history.json` y `labels.json` con el mapeo gesto→índice.【F:video_gesture_model/train_model.py†L212-L235】 Los checkpoints de mejores pesos (`*.weights.h5`) se guardan en la misma carpeta base.【F:video_gesture_model/train_model.py†L171-L200】
- **Definición de la arquitectura**: el modelo es una **red LSTM para clasificación de secuencias** de landmarks 3D. La pila es `Input(sequence_length, feature_dim)` → `Masking` → `LSTM` (return_sequences) → `Dropout` → `LSTM` → `Dropout` → `Dense(relu)` → `Dense(softmax)` que produce probabilidades por clase.【F:video_gesture_model/train_model.py†L95-L114】
- **Tipo de problema**: **clasificación de secuencias de frames** (no por frame individual). Se alimenta con secuencias de landmarks de ambas manos obtenidas con MediaPipe y normalizadas respecto a la muñeca.【F:video_gesture_model/extract_landmarks.py†L59-L125】
- **Datos esperados**: vectores de landmarks 3D (x, y, z) para hasta dos manos (21 puntos por mano), no imágenes RGB crudas.【F:video_gesture_model/config.py†L29-L35】【F:video_gesture_model/extract_landmarks.py†L92-L110】

Fragmento de construcción del modelo (extraído de `video_gesture_model/train_model.py`):
```python
inputs = tf.keras.layers.Input(shape=(sequence_length, feature_dim), name="landmarks")
x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
x = tf.keras.layers.LSTM(lstm_units[0], return_sequences=True)(x)
x = tf.keras.layers.Dropout(dropout)(x)
x = tf.keras.layers.LSTM(lstm_units[1])(x)
x = tf.keras.layers.Dropout(dropout)(x)
x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```
【F:video_gesture_model/train_model.py†L95-L114】

## 2. Datos de entrada del modelo
1. **Preparación de datos en inferencia**:
   - `video_gesture_model/realtime_inference.py` usa MediaPipe Hands para obtener landmarks por frame, los normaliza con `normalise_landmarks` (resta la posición de la muñeca por mano) y los acumula en un `deque` hasta completar `sequence_length` frames.【F:video_gesture_model/realtime_inference.py†L126-L195】【F:video_gesture_model/extract_landmarks.py†L59-L125】
   - `server_ec2_flask.py` espera que los clientes envíen ya la secuencia completa de landmarks como lista bidimensional JSON (`sequence_length x feature_dim`).【F:server_ec2_flask.py†L90-L151】

2. **Forma (shape) de la entrada**:
   - `sequence_length = FPS * CLIP_DURATION = 24 * 4 = 96` frames por muestra.【F:video_gesture_model/config.py†L22-L35】
   - `feature_dim = 21 landmarks * 3 coords * 2 manos = 126` valores por frame.【F:video_gesture_model/config.py†L29-L35】
   - Tensor de entrada para predicción: `(batch_size=1, 96, 126)` en `float32` (se agrega dimensión batch antes de llamar al modelo).【F:video_gesture_model/realtime_inference.py†L175-L183】【F:server_ec2_flask.py†L103-L127】

3. **Tipo de datos y rango de valores**:
   - Los landmarks de MediaPipe vienen normalizados en `[0,1]` para x/y y valores relativos en z; después de restar la muñeca, el rango queda centrado alrededor de 0 con valores positivos y negativos (sin reescalado adicional).【F:video_gesture_model/extract_landmarks.py†L59-L75】
   - Todos los tensores se manejan como `np.float32`/`tf.float32` antes de llamar al modelo.【F:video_gesture_model/realtime_inference.py†L175-L183】【F:server_ec2_flask.py†L95-L127】

4. **Stackeo temporal**:
   - Los frames se apilan en orden de captura; si faltan frames (solo en dataset offline) se rellena con ceros, y el modelo aplica `Masking` sobre valor 0.0 para ignorarlos.【F:video_gesture_model/extract_landmarks.py†L116-L125】【F:video_gesture_model/train_model.py†L95-L114】

Ejemplo de creación del tensor justo antes de predecir (`realtime_inference.py`):
```python
buffer.append(normalise_landmarks(frame_features.flatten()))
...
if len(buffer) == args.sequence_length:
    input_tensor = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)
    probabilities = predict(input_tensor)[0]
```
【F:video_gesture_model/realtime_inference.py†L175-L182】

## 3. Salida del modelo y mapeo a clases
1. **Forma y tipo de salida**:
   - La última capa es `Dense(num_classes, activation="softmax")`, por lo que produce probabilidades por clase en un vector `(batch_size, num_classes)`.【F:video_gesture_model/train_model.py†L95-L113】

2. **Interpretación**:
   - En inferencia local, se toma `argmax` sobre el vector de probabilidades para obtener el índice predicho y su confianza.【F:video_gesture_model/realtime_inference.py†L177-L186】
   - En el servidor EC2 se aplica `tf.nn.softmax` (seguro aunque el modelo ya devuelva softmax) y luego `argmax`; se devuelve `index`, `label` y `confidence` (probabilidad).【F:server_ec2_flask.py†L103-L151】
   - El mapeo índice → etiqueta humana se carga desde `labels.json` guardado junto al modelo durante el entrenamiento.【F:video_gesture_model/train_model.py†L232-L234】【F:video_gesture_model/realtime_inference.py†L117-L124】

3. **Ejemplo de interpretación**:
```python
probabilities = predict(input_tensor)[0]  # shape (num_classes,)
pred_idx = int(np.argmax(probabilities))
confidence = float(probabilities[pred_idx])
label = idx_to_label.get(pred_idx, "?")
```
【F:video_gesture_model/realtime_inference.py†L177-L186】

## 4. Flujo de inferencia (de los datos al resultado)
1. **Script recomendado**: `python -m video_gesture_model.realtime_inference --model-dir <ruta_al_modelo>`.
2. **Pasos detallados**:
   - **Selección del modelo y etiquetas**: se leen argumentos CLI, se elige carpeta SavedModel o archivo `.keras/.h5`, y se localiza `labels.json` (explícito o junto al modelo).【F:video_gesture_model/realtime_inference.py†L30-L124】
   - **Carga del predictor**: `build_predict_fn` admite SavedModel (`tf.saved_model.load` con firma `serve/serving_default`) o archivos Keras y devuelve un closure `predict(x)` que retorna `np.ndarray` de probabilidades.【F:video_gesture_model/realtime_inference.py†L57-L98】
   - **Captura y preprocesamiento**: se inicializa MediaPipe Hands (hasta 2 manos); cada frame se convierte a RGB, se extraen landmarks, se normalizan restando la muñeca y se dibujan sobre la imagen. Los vectores planos (126) se encolan en `buffer` de longitud fija (96).【F:video_gesture_model/realtime_inference.py†L126-L176】【F:video_gesture_model/extract_landmarks.py†L59-L109】
   - **Construcción del tensor y predicción**: cuando el buffer está lleno, se arma un tensor `(1, 96, 126)` `float32`, se ejecuta `predict`, se calcula `argmax` y se obtiene la confianza.【F:video_gesture_model/realtime_inference.py†L175-L186】
   - **Postprocesado y visualización**: si la confianza supera el umbral CLI (`--confidence-threshold`), se pinta en la ventana de OpenCV la etiqueta y probabilidad; la ventana se actualiza en bucle hasta que se presiona `q`.【F:video_gesture_model/realtime_inference.py†L184-L205】

## 5. API recomendada para integrar este modelo en otro proyecto
Se sugiere un wrapper ligero inspirado en `build_predict_fn` y el preprocesamiento usado en `realtime_inference.py`:

```python
import numpy as np
import tensorflow as tf
from pathlib import Path
from video_gesture_model.extract_landmarks import normalise_landmarks

class GestureModelWrapper:
    def __init__(self, model_path: str):
        model_path = Path(model_path)
        if model_path.is_dir() and (model_path / "saved_model.pb").exists():
            saved = tf.saved_model.load(str(model_path))
            self.infer = saved.signatures.get("serve") or saved.signatures["serving_default"]
            self.input_name = next(iter(self.infer.structured_input_signature[1].keys()))
            self.output_name = next(iter(self.infer.structured_outputs.keys()))
        else:
            self.model = tf.keras.models.load_model(str(model_path))
            self.infer = None

    def predict(self, sequence: np.ndarray) -> dict:
        """sequence shape: (96, 126) float32 con landmarks ya normalizados."""
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.shape != (96, 126):
            raise ValueError("Se esperaba una secuencia (96, 126) con landmarks")
        batch = np.expand_dims(seq, axis=0)
        if self.infer is not None:
            out = self.infer(**{self.input_name: tf.constant(batch)})[self.output_name].numpy()[0]
        else:
            out = self.model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(out))
        return {"index": idx, "score": float(out[idx]), "probs": out.tolist()}
```

- **Parámetros de `predict`**: recibe un `np.ndarray` `(96, 126)` `float32` **ya normalizado** (resta de la muñeca por mano). Debe conservar el orden temporal.
- **Salidas**: diccionario con índice de clase, probabilidad asociada y el vector completo de probabilidades.
- **Preprocesamiento mínimo externo**: si la fuente son landmarks crudos de MediaPipe, aplicar `normalise_landmarks` y verificar que haya exactamente 96 frames; si sobran, recortar; si faltan, decidir si pad con ceros (el modelo está entrenado con padding=0 y `Masking`).【F:video_gesture_model/extract_landmarks.py†L59-L125】【F:video_gesture_model/train_model.py†L95-L114】

## 6. Ejemplos mínimos de uso (código)
- **Ejemplo sintético con el wrapper sugerido**:
```python
import numpy as np
from ruta_al_wrapper import GestureModelWrapper

model = GestureModelWrapper("video_gesture_model/data/models/gesture_model_20251031_183504")
dummy_sequence = np.zeros((96, 126), dtype=np.float32)  # padding/zeros aceptados por el Masking
resultado = model.predict(dummy_sequence)
print(resultado)  # {'index': 0, 'score': 0.12, 'probs': [...]}  # valores de ejemplo
```

- **Uso resumido de `realtime_inference.py`** (código real simplificado):
```python
predict = build_predict_fn(model_dir_or_file)               # carga SavedModel o .keras/.h5
buffer.append(normalise_landmarks(frame_features.flatten()))
if len(buffer) == args.sequence_length:                     # cuando hay 96 frames
    input_tensor = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)
    probabilities = predict(input_tensor)[0]
    pred_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_idx])
```
【F:video_gesture_model/realtime_inference.py†L57-L98】【F:video_gesture_model/realtime_inference.py†L175-L186】

## 7. Riesgos, limitaciones y notas
- **Tamaño fijo**: la secuencia debe tener exactamente 96 frames y cada frame 126 valores; cualquier integración debe respetar este tamaño (recortar o rellenar con ceros).【F:video_gesture_model/config.py†L22-L35】【F:video_gesture_model/extract_landmarks.py†L116-L125】
- **Dependencias pesadas**: requiere TensorFlow (>=2.x) y MediaPipe; el script de tiempo real usa OpenCV para captura/visualización.【F:video_gesture_model/realtime_inference.py†L14-L18】
- **Preprocesamiento obligatorio**: restar la posición de la muñeca por mano; si se omite, las distribuciones de entrada no coincidirán con el entrenamiento y la precisión caerá.【F:video_gesture_model/extract_landmarks.py†L59-L75】
- **Orden de manos**: se asume índice 0 = mano izquierda, 1 = derecha; invertirlo generará desalineación de features.【F:video_gesture_model/realtime_inference.py†L155-L173】
- **Clases fijas**: el archivo `labels.json` define el orden de las clases; cualquier cambio en ese archivo debe ir acompañado del modelo correspondiente.【F:video_gesture_model/train_model.py†L232-L234】【F:video_gesture_model/realtime_inference.py†L117-L124】
- **Rendimiento**: la inferencia en CPU puede ser suficiente por la ligereza del modelo (dos LSTM pequeñas), pero la extracción de landmarks con MediaPipe domina el tiempo de ejecución; optimizar la cámara/FPS puede ser más crítico que el modelo en sí.【F:video_gesture_model/realtime_inference.py†L126-L195】