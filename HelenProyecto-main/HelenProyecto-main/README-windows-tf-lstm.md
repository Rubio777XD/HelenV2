# HELEN – TensorFlow LSTM en Windows (Chrome/Chromium)

Esta guía explica, paso a paso y con detalle técnico, cómo instalar, configurar y ejecutar HELEN en Windows 10/11 (x64) utilizando el backend **TensorFlow LSTM** con dependencias **TensorFlow CPU** y **NumPy**. Se centra en la ejecución en Google Chrome (o Chromium) y en la señal de activación **Start** normalizada por el clasificador.

## 1. Introducción técnica

| Concepto | Detalle |
| --- | --- |
| ¿Qué es HELEN? | Asistente por gestos con backend Flask/Socket.IO (`backendHelen/`) y frontend web servido desde el mismo backend (`helen/`). |
| Backend LSTM | Usa `TensorFlowSequenceGestureClassifier` para consumir secuencias de landmarks y producir gestos normalizados. La implementación vive en `backendHelen/tf_gesture_classifier.py`. |
| Dependencias TensorFlow CPU / NumPy | El clasificador LSTM importa **TensorFlow** y **NumPy** en su constructor; si faltan, el backend lanza un `RuntimeError` y no arranca. |
| Normalización de la señal **Start** | El clasificador traduce sinónimos como `activar`, `start` o `wake` al label **Start** antes de devolver predicciones, garantizando que el disparador de activación sea consistente. |
| Tensores `(1, 96, 126) float32` | El modelo espera un *batch* de 1 secuencia, con 96 frames y 126 features por frame. HELEN expande sus 42 features por frame a 126 (duplica mano y agrega `z=0`) y luego añade la dimensión de batch para formar `(1, 96, 126)` en `np.float32`. |

## 2. Requisitos del sistema

| Requisito | Detalle |
| --- | --- |
| Sistema operativo | Windows 10/11 de 64 bits. |
| Python soportado | Python 3.11 (64 bits) agregado al `PATH` con el *launcher* `py`. |
| Dependencias obligatorias | TensorFlow (CPU), NumPy, OpenCV, MediaPipe, Flask, Flask-SocketIO y dependencias de `requirements.txt`. |
| Navegador | Google Chrome o Chromium instalado localmente. |
| Verificar Chrome en `PATH` | En `CMD`, ejecuta `where chrome` (Chrome) o `where chromium` (Chromium). Si no devuelve ruta, añade la carpeta de instalación al `PATH`. |
| Runtimes para TensorFlow | En Windows suelen requerirse los **Microsoft Visual C++ Redistributable 2015-2022 x64**. Instálalos y reinicia si TensorFlow lanza errores de DLL. |

## 3. Instalación completa (paso a paso)

### 3.1 Instalar Python 3.11

1. Descarga el instalador de Python 3.11 (64 bits) desde python.org.
2. Durante la instalación, marca **Add python.exe to PATH**.
3. Reinicia la sesión de Windows tras instalar.

### 3.2 Crear entorno virtual

```cmd
cd %USERPROFILE%\Documents
:: Clona el repo o copia el código fuente
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
```

### 3.3 Instalar dependencias (TensorFlow CPU, NumPy, etc.)

```cmd
cd RUTA\AL\REPOSITORIO\HelenProyecto-main\HelenProyecto-main
pip install --upgrade pip
pip install -r requirements.txt
```

- `tensorflow` (CPU) y `numpy` son obligatorios para cargar el clasificador LSTM.
- Si usas GPU, omite paquetes de CUDA: esta guía asume **TensorFlow CPU**.

### 3.4 Instalar HELEN y scripts

El backend se ejecuta directamente desde el código fuente; no hay instalador. Asegúrate de que estás en la raíz del proyecto (`HelenProyecto-main\HelenProyecto-main`).

### 3.5 Copiar o colocar modelos (SavedModel)

- Ruta predeterminada de modelos TensorFlow: `Hellen_model_TF/video_gesture_model/data/models/`.
- Coloca allí la carpeta del **SavedModel** (`saved_model.pb`, `variables/`, `labels.json`).
- El backend elige automáticamente el SavedModel más reciente dentro de esa carpeta.

### 3.6 Variables de entorno

Define antes de ejecutar:

```cmd
set HELEN_MODEL_BACKEND=lstm
set HELEN_ACTIVATION_SIGNAL=Start
```

- `HELEN_MODEL_BACKEND=lstm`: obliga al backend a usar el clasificador LSTM (si falta, el valor por defecto también es `lstm`).
- `HELEN_ACTIVATION_SIGNAL=Start`: refuerza la señal de activación estándar para los gestos de encendido/atención. Mantén la “S” mayúscula.

## 4. Ejecutar HELEN en Windows (CLI + modo Chrome)

### 4.1 Comando básico para iniciar HELEN

```cmd
cd RUTA\AL\REPOSITORIO\HelenProyecto-main\HelenProyecto-main
call .venv\Scripts\activate
set HELEN_MODEL_BACKEND=lstm
set HELEN_ACTIVATION_SIGNAL=Start
python -m backendHelen.server --host 0.0.0.0 --port 3000
```

- El puerto `3000` se usa para emparejarse con el modo kiosco solicitado; cambia a `5000` si prefieres el valor histórico.

### 4.2 Abrir Chrome automáticamente

Después de lanzar el backend, abre la interfaz:

```cmd
start "" chrome --app=http://localhost:3000
```

### 4.3 Modo kiosco (pantalla completa)

```cmd
chrome.exe --kiosk --app=http://localhost:3000
```

### 4.4 Scripts `.bat` de ejemplo

Copia estos archivos en `scripts/` (ya están incluidos en esta guía):

- `scripts\start_helen.bat` — Inicia backend con LSTM.
- `scripts\start_kiosk.bat` — Abre Chrome en modo kiosco.
- `scripts\start_all.bat` — Lanza backend y Chrome juntos.

Ejecuta con doble clic o desde `CMD`.

## 5. Cómo correr HELEN en Chromium en Windows

| Aspecto | Chrome | Chromium |
| --- | --- | --- |
| Ruta típica | `C:\Program Files\Google\Chrome\Application\chrome.exe` | `C:\Program Files\Chromium\Application\chromium.exe` o `%LOCALAPPDATA%\Chromium\Application\chromium.exe` |
| Verificar en PATH | `where chrome` | `where chromium` |
| Modo kiosco | `chrome.exe --kiosk --app=http://localhost:3000` | `chromium.exe --kiosk --app=http://localhost:3000` |
| Evitar popups iniciales | Agrega `--disable-features=TranslateUI --disable-infobars --no-first-run --disable-notifications` | Igual que Chrome |

## 6. Pruebas y verificación

1. **Verificar TensorFlow LSTM activo**: en la consola debe aparecer `Modelo LSTM cargado desde ...` al arrancar el backend.
2. **Confirmar backend cargado**: visita `http://localhost:3000/health` y comprueba `"status":"HEALTHY"`.
3. **Probar señal de activación “Start”**: realiza el gesto asociado; el event log mostrará `gesture: "Start"` aunque uses un sinónimo (p. ej., `activar`).
4. **Errores GPU/CPU o librerías faltantes**: si ves `DLL load failed` o mensajes sobre CUDA, reinstala TensorFlow CPU y el VC++ Redistributable x64; elimina variables de entorno de CUDA si no usas GPU.

## 7. Problemas comunes y soluciones

| Problema | Causa probable | Solución |
| --- | --- | --- |
| `TensorFlow DLL missing` | Falta VC++ Redistributable o TensorFlow corrupto. | Reinstala VC++ 2015-2022 x64 y vuelve a ejecutar `pip install --force-reinstall tensorflow`. |
| Error en NumPy / Runtime | Mismatch de versiones al actualizar solo TensorFlow. | Ejecuta `pip install --upgrade --force-reinstall numpy tensorflow`. |
| Chrome no abre | `chrome` no está en `PATH`. | Usa ruta completa del ejecutable o añade la carpeta a `PATH`. |
| Puerto 3000 ocupado | Otro servicio usa el puerto. | Ejecuta `python -m backendHelen.server --port 5050` y abre `http://localhost:5050`. |
| Modelos corruptos | `saved_model.pb` o `variables/` dañados. | Reemplaza el SavedModel en `Hellen_model_TF/video_gesture_model/data/models/` por una copia intacta. |
| Regenerar SavedModel | Solo si tienes el proyecto de entrenamiento. Vuelve a exportar el modelo y copia la carpeta completa a la ruta anterior. |
| Cambiar backend manualmente | Soporta `cnn`, `lstm` o fallback sintético. | `set HELEN_MODEL_BACKEND=cnn` (o `lstm`) antes de ejecutar; si falla carga, cae al clasificador sintético con el dataset local. |

## 8. Notas finales y recursos

- **Actualizar HELEN**: ejecuta `git pull` y después `pip install -r requirements.txt` dentro de `.venv`.
- **Cambiar de backend sin romper compatibilidad**: define `HELEN_MODEL_BACKEND` antes de arrancar; el sistema mantiene la forma `(1,96,126)` para LSTM y conserva la señal **Start** normalizada.
- **Usar modelos antiguos sin reentrenarlos**: coloca el SavedModel legado en la carpeta de modelos; el backend elige el más reciente, pero puedes dejar solo el modelo que quieras usar.
- **Rendimiento en Windows**: usa `--frame-stride 2` y `--poll-interval 0.08` si necesitas bajar carga de CPU; ejecuta Chrome con `--disable-gpu` si hay drivers inestables.

---

Para copiar este README a GitHub o a un archivo local, duplica el contenido en un archivo `README.md` dentro de tu repositorio o súbelo directamente mediante tu cliente Git. Luego confirma el commit y haz push a tu rama de trabajo.
