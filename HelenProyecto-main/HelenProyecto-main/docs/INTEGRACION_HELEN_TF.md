# Integración del modelo TensorFlow en HELEN

Este documento resume cómo conviven los dos backends de clasificación de gestos disponibles en HELEN y cómo seleccionar cada uno en tiempo de ejecución.

## Backends disponibles

- **`xgboost`** (por defecto): usa `ProductionGestureClassifier` y el artefacto `Hellen_model_RN/model.p`. Consume vectores de 42 features por frame (21 landmarks × 2 coordenadas) generados por el pipeline actual.
- **`lstm`**: usa `TensorFlowSequenceGestureClassifier` y los artefactos TensorFlow guardados en `Hellen_model_TF/video_gesture_model/data/models/gesture_model_*` (formato SavedModel). Consume secuencias de `(96, 126)` (`96` frames × `126` features = 21 landmarks × 3 coords × 2 manos).

## Selección de backend

El backend se controla con la variable de entorno `HELEN_MODEL_BACKEND` (valores permitidos: `xgboost`, `lstm`). Ejemplos:

```bash
# Backend clásico (valor por defecto)
HELEN_MODEL_BACKEND=xgboost python -m backendHelen.server

# Backend LSTM basado en TensorFlow
HELEN_MODEL_BACKEND=lstm python -m backendHelen.server
```

- Para el backend `lstm`, la ruta del modelo se resuelve automáticamente al SavedModel más reciente en `Hellen_model_TF/video_gesture_model/data/models`. Si se desea forzar una ruta distinta, se puede instanciar `RuntimeConfig` con `tf_model_dir` apuntando a la carpeta del modelo.

## Forma de las features y preprocesamiento

| Backend | Forma de entrada | Preprocesamiento |
| --- | --- | --- |
| `xgboost` | `(42,)` | Normalización estadística (`FeatureNormalizer`) sobre vectores 2D centrados en min(x,y). |
| `lstm` | `(96, 126)` | Se mantiene un buffer de 96 frames. Cada frame expande las 42 features actuales añadiendo `z=0` y duplicando la mano en el slot de la segunda mano (hack temporal hasta emitir landmarks 3D de ambas manos). Los valores se convierten a `float32` y se apilan con dimensión batch `(1, 96, 126)`. |

## Carga del modelo y labels

- El backend `lstm` carga automáticamente un SavedModel (firma `serve`/`serving_default`) o un archivo Keras. En ambos casos se genera un `predict_fn` que devuelve probabilidades por clase.
- Se busca `labels.json` junto al modelo para mapear índice → etiqueta humana. Si el archivo no está disponible, se usa `labels_dict` del modelo clásico como respaldo.

## Limitaciones y notas

- El puente 42→126 asume una sola mano y `z=0`; esta simplificación puede impactar la precisión del LSTM hasta que el pipeline capture landmarks 3D de ambas manos.
- La inferencia secuencial solo ocurre cuando el buffer llega a 96 frames; antes de eso se devuelve una predicción neutra con `score=0.0` y la DecisionEngine gestiona la estabilidad.
- Requiere que TensorFlow esté instalado (por ejemplo, `tensorflow` o `tensorflow-cpu` en `requirements.txt`).
- El flujo general (cámara → normalización → classifier → DecisionEngine → SSE) permanece intacto; solo cambia la implementación interna del clasificador.
