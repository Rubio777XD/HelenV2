# Guía de migración del modelo de gestos de HELEN

Esta guía describe con lujo de detalle cómo funciona hoy el modelo de gestos de HELEN, cómo viajan los datos desde la cámara hasta el frontend y qué pasos seguir para reemplazar el modelo existente por uno nuevo (por ejemplo, un modelo entrenado en TensorFlow). El objetivo es que cualquier desarrollador pueda entender el flujo completo y realizar la migración sin romper el contrato actual con el resto del sistema.

## 1. Arquitectura actual del modelo en HELEN

### Dónde vive el modelo y cómo se carga
- **Ruta del artefacto del modelo**: `Hellen_model_RN/model.p` (pickle con el modelo XGBoost y metadatos adicionales). La ruta se arma en `backendHelen/server.py` como `MODEL_DIR = REPO_ROOT / "Hellen_model_RN"` y `MODEL_PATH = MODEL_DIR / "model.p"`.
- **Clase de carga**: `ProductionGestureClassifier` en `backendHelen/server.py`. Esta clase:
  - Verifica la existencia del archivo `model.p` y carga con `pickle` un diccionario que incluye `model`, `encoder`/`label_encoder` y `classes_` cuando están presentes.【F:backendHelen/server.py†L2319-L2352】
  - Convierte las características en un `numpy.asarray(...).reshape(1, -1)` para alimentar al modelo XGBoost.【F:backendHelen/server.py†L2355-L2378】
  - Si el modelo ofrece `predict_proba`, usa la probabilidad máxima como `score`; si no, usa `predict` directamente. Luego convierte el valor crudo a etiqueta humana usando el `encoder` o el mapeo `labels_dict`.【F:backendHelen/server.py†L2355-L2394】
- **Diccionario de etiquetas**: `labels_dict` en `Hellen_model_RN/helpers.py`, que mapea índices numéricos a nombres canónicos (`Start`, `Clima`, `Reloj`, etc.).【F:Hellen_model_RN/helpers.py†L8-L26】

### Dataset y normalización
- **Dataset**: `Hellen_model_RN/data.pickle` (o `data1.pickle` como fallback). La ruta se resuelve en `server.py` mediante `_default_dataset_path()` y se expone como `DATASET_PATH` para reuso.
- **Normalización**: La clase `FeatureNormalizer` carga del dataset un normalizador (`normalizer`/`scaler`) o estadísticas `feature_mean` y `feature_std` y aplica `transform()` sobre el vector de entrada antes de la inferencia.【F:backendHelen/server.py†L419-L501】 Si no hay normalizador disponible, devuelve el vector original.

### Flujo de datos hacia el modelo
1. **Captura de landmarks**: `CameraGestureStream` (en `server.py`) abre la cámara con OpenCV/MediaPipe, obtiene `results.multi_hand_landmarks[0]` y guarda 21 puntos `(x, y, z)` normalizados.【F:backendHelen/server.py†L2725-L2762】
2. **Suavizado**: Se mantiene un buffer `deque` y se promedia cada coordenada con `_smooth_landmarks()` para reducir ruido.【F:backendHelen/server.py†L2985-L2989】
3. **Extracción de features**: `_extract_features()` construye un vector plano tomando sólo `x` e `y` de cada landmark, restando el mínimo `x`/`y` para normalizar traslación. El resultado es un vector de longitud 42 (21 puntos × 2 coords).【F:backendHelen/server.py†L2992-L3009】
4. **Normalización estadística**: `FeatureNormalizer.transform()` aplica el scaler del dataset o resta media y divide por desviación estándar si existen estadísticas completas.【F:backendHelen/server.py†L472-L491】
5. **Inferencia**: `GesturePipeline` llama `self._runtime.classifier.predict(transformed)`; si `model.p` no se carga, `_create_classifier()` cae a `SimpleGestureClassifier` (clasificador de centroides) usando el mismo dataset.【F:backendHelen/server.py†L3144-L3189】【F:backendHelen/server.py†L3385-L3395】【F:Hellen_model_RN/simple_classifier.py†L1-L149】
6. **Postprocesamiento de etiquetas**: `ProductionGestureClassifier._to_label()` traduce índices a etiquetas humanas con el `encoder` o `labels_dict`. El clasificador sintético ya retorna la etiqueta canónica. Ambas rutas producen un `Prediction(label: str, score: float)`.

### Salida y traducción a acciones
- **Evento interno**: `GestureDecisionEngine` aplica umbrales, consenso temporal y validaciones geométricas; cuando acepta una predicción, `GesturePipeline` construye un `event` con `build_event()` que incluye `gesture`, `character`, `score`, `latency_ms`, etc.【F:backendHelen/server.py†L3177-L3189】【F:backendHelen/server.py†L3585-L3622】
- **Publicación**: `push_prediction()` envía el evento a todos los clientes SSE mediante `EventStream.broadcast()` y actualiza el estado interno.【F:backendHelen/server.py†L3625-L3644】
- **Frontend**: `helen/jsSignHandler/SocketIO.js` abre un `EventSource`, parsea cada mensaje SSE como JSON y emite eventos `message` a los listeners registrados.【F:helen/jsSignHandler/SocketIO.js†L1-L125】

## 2. Flujo completo: de la cámara al frontend
1. **Inicialización del runtime** (`HelenRuntime.__init__` en `server.py`): resuelve rutas, carga `FeatureNormalizer`, `GestureDecisionEngine`, crea el clasificador (producción o sintético) y el stream (cámara o sintético).【F:backendHelen/server.py†L3228-L3267】【F:backendHelen/server.py†L3383-L3395】【F:backendHelen/server.py†L3397-L3435】
2. **Captura de cámara** (`CameraGestureStream.next`): abre OpenCV/MediaPipe, verifica calidad, obtiene landmarks, suaviza y genera `features` (vector de 42 floats). Devuelve `features` y opcional `label` (solo en modo sintético se envía hint).【F:backendHelen/server.py†L2725-L2762】【F:backendHelen/server.py†L2992-L3009】
3. **Normalización de features** (`FeatureNormalizer.transform`): aplica scaler entrenado o estadísticas del dataset antes de la inferencia.【F:backendHelen/server.py†L472-L491】
4. **Inferencia del modelo** (`GesturePipeline._run`): llama `classifier.predict(transformed)`; mide latencia y captura landmarks para verificaciones posteriores.【F:backendHelen/server.py†L3144-L3157】
5. **Filtro de decisión** (`GestureDecisionEngine.process`, no mostrado en detalle): aplica umbrales por clase, consenso temporal y verificaciones geométricas específicas por gesto (ej. separación de dedos para `Start`, arco para `Clima`).【F:backendHelen/server.py†L1090-L1189】
6. **Construcción del evento** (`HelenRuntime.build_event`): empaqueta la predicción en un diccionario con `gesture/character`, `score`, `latency_ms`, `session_id`, etc.【F:backendHelen/server.py†L3585-L3622】
7. **Broadcast** (`HelenRuntime.push_prediction`): actualiza el estado y envía el evento por SSE usando `EventStream.broadcast()`.【F:backendHelen/server.py†L3625-L3644】
8. **Recepción en frontend** (`helen/jsSignHandler/SocketIO.js`): instancia `EventSource`, reintenta en caso de error y despacha el payload JSON a los listeners registrados (por ejemplo, UI de navegación por gestos).【F:helen/jsSignHandler/SocketIO.js†L1-L125】

## 3. Estrategia para cambiar el modelo por uno nuevo (TensorFlow u otro)

### Contrato actual del modelo
- Firma esperada: un objeto con método `predict(features: Iterable[float]) -> Prediction`, donde `Prediction` tiene `label: str` y `score: float`.
- Dimensión de entrada: lista de 42 floats (landmarks normalizados y centrados).【F:backendHelen/server.py†L2992-L3009】
- Flujo previo y posterior (cámara, normalizador, decisión, SSE) no depende de la implementación interna del modelo siempre que se respete la firma anterior.

### Punto de aislamiento recomendado
- Mantener una única interfaz de predicción en `backendHelen/server.py`, reemplazando la implementación de `ProductionGestureClassifier` por un wrapper equivalente para TensorFlow. Alternativa: crear una nueva clase (ej. `TensorFlowGestureClassifier`) con la misma interfaz (`predict`) y devolver `Prediction`.
- Lugar sugerido: mismo archivo `backendHelen/server.py`, cerca de la definición actual de `ProductionGestureClassifier` para minimizar cambios en el wiring.

### Qué no debe cambiar
- **Captura y preprocesamiento**: `CameraGestureStream` y `_extract_features()` deben seguir generando el vector de 42 floats.
- **Normalización**: `FeatureNormalizer` puede reutilizarse; si el nuevo modelo requiere otro escalado, actualiza el dataset/normalizador pero conserva la llamada `feature_normalizer.transform(features)`.
- **Transporte**: `GestureDecisionEngine`, `build_event`, `push_prediction` y el frontend SSE permanecen iguales.
- **Firma externa**: `predict(features) -> Prediction(label, score)` sigue siendo el contrato.

### Qué cambia con TensorFlow
- **Carga del modelo**: usar `tf.keras.models.load_model("ruta/al/modelo")` en lugar de `pickle.load`.
- **Entrada**: asegurarse de convertir la lista de 42 floats a `np.array(features, dtype=np.float32).reshape(1, 42)` (o la forma que espere el modelo). Si el modelo consume secuencias o canales adicionales, adaptar aquí pero conservar la entrada nominal.
- **Salida**: `model.predict(...)` suele devolver un vector de probabilidades; usar `np.argmax` para obtener índice y luego mapear con `labels_dict` o un `LabelEncoder` equivalente. El `score` puede ser la probabilidad máxima.
- **Wrapper ejemplo** (plantilla a adaptar):
  ```python
  import tensorflow as tf
  import numpy as np
  from Hellen_model_RN.helpers import labels_dict
  from Hellen_model_RN.simple_classifier import Prediction

  class TensorFlowGestureClassifier:
      source = "tensorflow"

      def __init__(self, model_path: Path):
          self._model = tf.keras.models.load_model(model_path)
          self._labels = {int(k): v for k, v in labels_dict.items()}

      def predict(self, features: Iterable[float]) -> Prediction:
          array = np.asarray(list(features), dtype=np.float32).reshape(1, -1)
          proba = self._model.predict(array, verbose=0)
          best_idx = int(np.argmax(proba[0]))
          label = self._labels.get(best_idx, str(best_idx))
          score = float(proba[0][best_idx]) if proba.ndim == 2 else 1.0
          return Prediction(label=label, score=score)
  ```
  > **Nota**: El código anterior es una plantilla; ajusta formas de entrada/salida según el modelo real.

## 4. Pasos concretos para reemplazar el modelo

1. **Identificar el cargador actual**
   - Archivo: `backendHelen/server.py`
   - Clase: `ProductionGestureClassifier` (inicio en la línea ~2319).【F:backendHelen/server.py†L2319-L2394】

2. **Sustituir la lógica de carga**
   - Antes (XGBoost con `pickle` y `numpy`):
     ```python
     model_dict = pickle.load(model_path.open("rb"))
     model = model_dict.get("model")
     array = np.asarray(list(features), dtype=float).reshape(1, -1)
     proba = self._model.predict_proba(array)
     ```
     【F:backendHelen/server.py†L2338-L2377】
   - Después (ejemplo TensorFlow):
     ```python
     import tensorflow as tf
     import numpy as np
     self._model = tf.keras.models.load_model(model_path)
     array = np.asarray(list(features), dtype=np.float32).reshape(1, -1)
     proba = self._model.predict(array, verbose=0)
     ```

3. **Ajustar el preprocesamiento**
   - El vector actual es de 42 floats (21 landmarks × 2). Si el nuevo modelo espera otra forma (ej. `(1, 42)` o `(1, 21, 2)`), hacer el `reshape` correspondiente dentro del nuevo clasificador.
   - Mantener la llamada a `FeatureNormalizer.transform(features)` en `GesturePipeline._run` para no romper el flujo.【F:backendHelen/server.py†L3144-L3148】

4. **Ajustar el postprocesamiento**
   - Si `model.predict` devuelve probabilidades, usar `argmax` y mapear con `labels_dict` o un `LabelEncoder` entrenado.
   - Construir `Prediction(label, score)` igual que hoy. Si el modelo entrega logits, aplicar `softmax` o la operación necesaria antes de extraer la probabilidad máxima.

5. **Mantener la interfaz externa**
   - La clase nueva debe exponer `predict(features) -> Prediction` para que `GesturePipeline` y `GestureDecisionEngine` sigan funcionando sin cambios.【F:backendHelen/server.py†L3150-L3153】
   - No modificar `build_event` ni `push_prediction`, que asumen que `label` es una cadena legible para el frontend.【F:backendHelen/server.py†L3585-L3622】

6. **Cablear el nuevo clasificador**
   - Editar `_create_classifier()` para instanciar `TensorFlowGestureClassifier` en lugar de `ProductionGestureClassifier`, o agregar una rama condicional (por ejemplo, revisar extensión `.h5` o un flag en config) y devolver el wrapper nuevo manteniendo el diccionario de metadatos `{"source": ..., "loaded": True}`.【F:backendHelen/server.py†L3383-L3395】

## 5. Riesgos y errores frecuentes al cambiar el modelo
- **Desajuste de dimensiones**: el nuevo modelo debe aceptar exactamente el vector que produce `_extract_features()` (42 floats). Cambios en la longitud romperán `reshape` y el pipeline.【F:backendHelen/server.py†L2992-L3009】
- **Tipos de datos**: TensorFlow/Keras suele esperar `float32`; si se dejan `float64` podría haber advertencias o penalizaciones de rendimiento. Castear explícitamente.
- **Orden de clases**: si el entrenamiento cambia el orden de etiquetas, actualizar el mapeo `labels_dict` o el `LabelEncoder`; de lo contrario, `label` podría no coincidir con la acción esperada.【F:Hellen_model_RN/helpers.py†L8-L26】
- **Rendimiento**: modelos pesados pueden incrementar latencia; `GesturePipeline` mide `latency_ms` y lo envía al frontend, así que monitorea impactos.【F:backendHelen/server.py†L3150-L3187】
- **Dependencias nativas**: asegurar que TensorFlow esté disponible en el entorno objetivo (ARM vs x86, GPU, etc.).
- **Uso de cámara**: el modelo nuevo no debe abrir la cámara por su cuenta; debe consumir el vector de features que ya entrega `CameraGestureStream`.

## 6. Pruebas recomendadas después de la migración
- **Pruebas unitarias**:
  - Crear un test que instancie el nuevo clasificador y llame `predict` con un vector sintético de 42 floats; verificar que devuelve `Prediction` con `label` válido y `score` en `[0,1]`.
  - Mockear `FeatureNormalizer.transform` para asegurar que el clasificador acepta la salida normalizada.
- **Pruebas de integración (offline)**:
  - Forzar el flujo sintético (`--no-camera` con dataset disponible) y verificar que el pipeline sigue emitiendo eventos SSE con etiquetas correctas.
- **Pruebas manuales**:
  1. Levantar el servidor (`python -m backendHelen.server`) con cámara real.
  2. Realizar los gestos `Start`, `Clima`, `Reloj`, `Inicio` y observar en logs la secuencia H→C y las puntuaciones.
  3. Confirmar que el frontend recibe eventos (abrir la app y revisar que la UI reacciona).【F:backendHelen/server.py†L3625-L3644】【F:helen/jsSignHandler/SocketIO.js†L1-L125】

Con esta guía deberías poder localizar el modelo actual, entender cómo se acopla al pipeline de cámara→normalización→inferencia→SSE, y reemplazarlo por un modelo TensorFlow (u otro) sin romper la interfaz que el resto de HELEN consume.
