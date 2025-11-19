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
- **Modelo de gestos**: el SavedModel de TensorFlow vive en `Hellen_model_TF/video_gesture_model/` y se carga desde
  `backendHelen.server`. Ese módulo es ahora el **punto único de verdad** para la inferencia y para la emisión de eventos
  hacia el ring de activación del frontend.

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

## Cómo correr HELEN en Windows (Python 3.10 + Google Chrome)

La siguiente guía describe el flujo manual recomendado cuando quieres ejecutar HELEN con el modelo de gestos por video
exportado en TensorFlow. Todos los pasos toman como referencia la raíz del repositorio (`HelenProyecto-main/`) y los
mismos componentes que usa la automatización oficial (`scripts/helen-run.ps1`).

### 1. Componentes a tener presentes

- **Backend principal** (`backendHelen/server.py`): expone la API HTTP/SSE, sirve el frontend estático y orquesta el
  pipeline de gestos dentro del mismo proceso en el puerto `5000` por defecto.
- **Servicio del modelo de video** (`Hellen_model_TF/frontend_bridge/server.py`): permite probar el SavedModel de forma
  aislada en el puerto `8000` y reutiliza la clase `GestureInferenceService` que también consume el backend.
- **Modelo y labels** (`Hellen_model_TF/video_gesture_model/`): aquí se guardan los `SavedModel` dentro de
  `data/models/gesture_model_*` junto con el `labels.json` necesario para mapear índices → gestos. Si no indicas
  `--model-path`, el servicio busca el modelo más reciente en esa carpeta.
- **Frontend** (`helen/`): HTML/CSS/JS servido por el backend mediante `SimpleHTTPRequestHandler`, por lo que basta con
  abrir `http://127.0.0.1:5000` en Chrome para ver la interfaz.

### 2. Requisitos previos

1. **Windows 10/11 de 64 bits** con permisos de administrador para instalar dependencias.
2. **Python 3.10 (x64)** agregado al `PATH`. TensorFlow 2.12 (la versión fijada en `requirements.txt`) sólo publica
   ruedas estables para 3.10 en Windows. Verifica con `py -3.10 --version`.
3. **Git 2.40+** (opcional si ya recibiste el repositorio). Descarga desde https://git-scm.com y acepta agregarlo al
   `PATH` para clonar vía `git clone`.
4. **Google Chrome 124 o superior** (o Microsoft Edge basado en Chromium). El frontend está optimizado para Chrome y
   requiere acceso a cámara.
5. **Cámara UVC** visible para Windows (prueba con la app “Cámara”). Cierra cualquier aplicación que la use antes de
   ejecutar HELEN.
6. **Opcional**: Visual Studio Code / Windows Terminal facilitan abrir PowerShell en la carpeta del proyecto.

### 3. Clonar o ubicar el repositorio

En PowerShell:

```powershell
cd $HOME\Documents
git clone https://github.com/tu-organizacion/HELEN.git
cd HELEN\HelenProyecto-main\HelenProyecto-main
```

Si ya tienes el repositorio sincronizado, únicamente navega hasta la carpeta `HelenProyecto-main\HelenProyecto-main`.

### 4. Crear y activar el entorno virtual (Python 3.10)

Todos los comandos siguientes se ejecutan desde la raíz del proyecto.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

Para sesiones nuevas sólo necesitas volver a ejecutar `.\.venv\Scripts\activate` antes de lanzar los servicios.

### 5. Instalar dependencias

La guía instala tres grupos de requisitos para cubrir backend + pipeline de visión:

```powershell
pip install -r requirements.txt                               # Backend, Flask y TensorFlow 2.12
pip install -r Hellen_model_TF\video_gesture_model\requirements.txt   # MediaPipe + utilidades de captura/entrenamiento
pip install -r Hellen_model_TF\frontend_bridge\requirements.txt        # Flask-CORS para el bridge opcional
```

Los archivos listan explícitamente las librerías requeridas (Flask, Flask-SocketIO, TensorFlow, MediaPipe, OpenCV y
Flask-CORS).

### 6. Preparar el modelo SavedModel y labels

- Copia la carpeta del modelo exportado a `Hellen_model_TF\video_gesture_model\data\models\gesture_model_YYYYMMDD_HHMMSS`.
- Dentro de esa carpeta asegúrate de incluir `saved_model.pb`, la subcarpeta `variables/` y el archivo `labels.json`
  generado durante `train_model.py`. Si `labels.json` vive junto al modelo no es necesario pasar `--labels` en los
  comandos.
- Si guardaste el modelo en otra ruta, toma nota para pasarla como argumento explícito (`--model-path`).

La clase `GestureInferenceService` usa la carpeta más reciente en `data/models` cuando no le indicas una ruta manual, por
lo que basta con mantener únicamente los modelos vigentes.

### 7. Arrancar el servicio del modelo TensorFlow (opcional pero recomendado)

Usa este servicio para verificar la cámara y las predicciones antes de abrir el backend completo. En una terminal con el
entorno activado:

```powershell
cd Hellen_model_TF
python -m frontend_bridge.server `
  --model-path video_gesture_model\data\models\gesture_model_20240601_120000 `
  --camera-index 0 `
  --confidence-threshold 0.75 `
  --port 8000
```

Puntos clave:

- El servidor expone `http://127.0.0.1:8000/api/status`, `/api/gestures`, `/api/stream` y un namespace Socket.IO para
  consumir predicciones en tiempo real.
- Ajusta `--camera-index` si tienes varias webcams.
- Finaliza con `Ctrl+C` para liberar la cámara antes de iniciar el backend.

### 8. Arrancar el backend principal de HELEN

En una segunda terminal PowerShell (con `.venv` activado y posicionada en la raíz del repo):

```powershell
python -m backendHelen.server `
  --host 0.0.0.0 `
  --port 5000 `
  --camera-index 0 `
  --model-path Hellen_model_TF\video_gesture_model\data\models\gesture_model_20240601_120000 `
  --confidence-threshold 0.75 `
  --prediction-cooldown 0.5
```

Notas importantes:

- `backendHelen.server` levanta un `ThreadingHTTPServer` que sirve el frontend desde `helen/`, publica `/health` y abre
  un stream SSE en `/events`.
- Puedes omitir `--model-path` si quieres que reutilice el mismo mecanismo de auto-descubrimiento del servicio del
  modelo. El backend también acepta `--labels` y `--sequence-length` si necesitas rutas personalizadas.
- Los registros del backend quedan en la consola. Para diagnósticos adicionales existe `python -m
  backendHelen.diagnostics --frames 100`.

### 9. Abrir HELEN en Google Chrome

1. Con ambos procesos corriendo, abre `http://127.0.0.1:5000` en Google Chrome.
2. Cuando el navegador pregunte, concede permisos permanentes para la cámara.
3. Comprueba que el anillo de activación aparece alrededor del avatar y que el panel **Wi-Fi** puede listar redes (usa el
   menú Configuración).
4. Si Chrome no muestra video, asegúrate de que el backend siga mostrando capturas en la terminal; de lo contrario,
   reinicia el backend tras cerrar cualquier app que use la cámara.

### 10. Prueba rápida de la seña de activación y navegación

1. Mira a la cámara y realiza la seña asociada a la etiqueta `start`. El backend considera equivalentes los aliases
   `start`, `activar`, `heyhelen`, `holahelen`, `oyehelen` y `wake`, que son los mismos que consume el frontend al recibir
   eventos de Socket.IO.
2. Cuando el backend confirme el gesto, el anillo se ilumina y se mantiene activo unos segundos (`prediction_cooldown`)
   mientras decides el comando siguiente.
3. Pronuncia o repite el gesto correspondiente a una pestaña soportada (`clima`, `foco`, `ajustes`, `inicio`,
   `dispositivos`, `reloj`). Estas etiquetas aparecen como `gestureActions` en el frontend y están cubiertas por las
   pruebas automatizadas.
4. Verifica que la UI cambie de sección sin errores y que la consola del backend registre el gesto detectado.

Si la UI no reacciona, revisa `/health` en el navegador para confirmar que `status` sea `HEALTHY` y que `pipeline_running`
sea `true`.

### 11. Nota breve para Linux/Raspberry Pi

Los pasos equivalentes (incluyendo scripts `setup-pi.sh` y `run-pi.sh`) ya están documentados en
[`README-linux-rpi-chrome.md`](README-linux-rpi-chrome.md). Utiliza esa guía cuando necesites automatizar la ejecución en
Debian/Raspberry Pi; la configuración de modelos y labels es exactamente la misma descrita anteriormente.
