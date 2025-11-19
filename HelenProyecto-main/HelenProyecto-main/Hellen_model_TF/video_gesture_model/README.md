# TensorFlow video gesture model

Esta carpeta contiene una versión preparada del pipeline para capturar gestos en
video, extraer landmarks 3D con MediaPipe, entrenar un modelo con TensorFlow y
ejecutar inferencia en tiempo real. También incluye utilidades para cargar los
datasets y modelos resultantes a AWS S3 una vez que se configuren las
credenciales.

## Flujo de trabajo

1. **Captura de clips:**
   ```bash
   python -m Hellen_model_RN.video_gesture_model.capture_videos [nombre_gesto]
   ```
   Si omites el nombre, el script mostrará una tabla con las señas disponibles y
   podrás elegir una existente o escribir una nueva. Presiona `s` para grabar
   clips de `config.CLIP_DURATION` segundos y `q` para finalizar. Los videos se
   almacenan en `data/raw_videos/<nombre_gesto>` y la terminal te indicará cuántos
   clips llevas en la sesión y el total acumulado para esa seña.

2. **Extracción de landmarks:**
   ```bash
   python -m Hellen_model_RN.video_gesture_model.extract_landmarks [gesto1 gesto2 ...]
   ```
   Si no especificas nada, aparecerá un menú para seleccionar una o varias señas
   detectadas automáticamente. El script genera un archivo `.npz` con las
   secuencias de landmarks para ambas manos y un archivo `*_labels.json` con el
   mapa gesto→índice, además de un resumen de muestras por seña.
=======
   python -m Hellen_model_RN.video_gesture_model.capture_videos <nombre_gesto>
   ```
   Presiona `s` para grabar clips de `config.CLIP_DURATION` segundos y `q` para
   finalizar. Los videos se almacenan en `data/raw_videos/<nombre_gesto>`.

2. **Extracción de landmarks:**
   ```bash
   python -m Hellen_model_RN.video_gesture_model.extract_landmarks gesto1 gesto2 ...
   ```
   Genera un archivo `.npz` con las secuencias de landmarks para ambas manos y un
   archivo `*_labels.json` que mapea cada gesto con su índice.


3. **Entrenamiento con TensorFlow:**
   ```bash
   python -m Hellen_model_RN.video_gesture_model.train_model --dataset path/al/dataset.npz --labels path/a/labels.json
   ```
   Puedes ajustar las épocas, tamaño de batch, tasa de aprendizaje, unidades LSTM,
   dropout y tamaño de la capa densa mediante argumentos opcionales. Antes de
   entrenar, la terminal mostrará un resumen de las muestras por clase y de los
   hiperparámetros seleccionados. El resultado es un `SavedModel` listo para
   conectarse posteriormente con el frontend.

4. **Inferencia en tiempo real:**
   ```bash
   python -m Hellen_model_RN.video_gesture_model.realtime_inference [--model-dir data/models/gesture_model_YYYYMMDD_HHMMSS]
   ```
   Si no indicas la carpeta del modelo, se listarán los modelos entrenados para
   elegir uno desde la terminal. El script carga el mapa de etiquetas, muestra las
   clases disponibles y utiliza un búfer de `sequence_length` frames para predecir
   el gesto actual y reflejarlo en pantalla.
=======
   Guarda un `SavedModel` listo para conectarse posteriormente con el frontend.

4. **Inferencia en tiempo real:**
   ```bash
   python -m Hellen_model_RN.video_gesture_model.realtime_inference --model-dir data/models/gesture_model_YYYYMMDD_HHMMSS
   ```
   Utiliza un búfer de `sequence_length` frames para predecir el gesto actual y
   mostrarlo en pantalla.


5. **Carga opcional a AWS:**
   ```bash
   python -m Hellen_model_RN.video_gesture_model.aws_utils dataset --name gesture_dataset
   python -m Hellen_model_RN.video_gesture_model.aws_utils model data/models/gesture_model_YYYYMMDD_HHMMSS
   ```
   Asegúrate de actualizar `config.py` con los nombres de buckets y de haber
   configurado previamente `aws configure` o las variables de entorno.

## Dependencias

- Python 3.9+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- TensorFlow (`tensorflow`)
- Boto3 (`boto3`) para la integración con AWS

Puedes instalar las dependencias mínimas con:

```bash
pip install -r requirements.txt
```

## requirements.txt sugerido

```
mediapipe>=0.10
opencv-python
tensorflow>=2.12
numpy
boto3
```

Guarda este archivo en la raíz del proyecto o en un entorno virtual dedicado
antes de ejecutar los scripts.
