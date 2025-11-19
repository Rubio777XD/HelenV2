# HELEN – Guía completa para ejecutar en Chrome (Windows)

Esta guía cubre el flujo oficial para poner en marcha HELEN en Windows 10/11 sin instaladores. A partir de esta
versión, todo el proceso puede automatizarse con un solo comando que prepara el entorno virtual, instala
dependencias, configura la cámara y abre la interfaz web en Google Chrome.

## 1. Inicio rápido (1 comando)

### 1.1 Pre-requisitos mínimos

| Requisito                          | Detalles                                                                 |
|-----------------------------------|--------------------------------------------------------------------------|
| Python                            | Python 3.10 (x64) instalado y agregado al `PATH` (incluye el launcher `py`). |
| PowerShell                        | PowerShell 7.0 o superior (ejecuta `pwsh -v` para verificar).            |
| Microsoft VC++ Redistributable    | Paquete 2015-2022 x64 (`vc_redist.x64.exe`).                             |
| TensorFlow                        | CPU compatible con AVX/AVX2 (requerido por TensorFlow 2.12 en Windows).  |
| Navegador                         | Google Chrome 124+ (o Microsoft Edge basado en Chromium).                |
| Hardware                          | Webcam UVC con permisos para el usuario actual.                          |

> Reinicia el equipo después de instalar Python y el VC++ Redistributable para garantizar que `PATH` quede
> actualizado.

### 1.2 Ejecutar HELEN

```powershell
# Desde la raíz del repositorio
powershell -ExecutionPolicy Bypass -File .\scripts\helen-run.ps1
```

También puedes usar el *wrapper* para equipos sin PowerShell 7:

```cmd
:: Equivalente en CMD
scripts\helen-run.bat
```

#### ¿Qué hace `helen-run.ps1`?

1. Detecta Python 3.10 disponible (`py -3.10`, `python`, etc.).
2. Crea o actualiza `.venv` con Python 3.10 y ejecuta `pip install --upgrade pip` + `pip install -r requirements.txt`
   (incluye TensorFlow 2.12 para CPU).
3. Honra las variables opcionales `HELEN_TF_MODEL_PATH` y `HELEN_TF_LABELS_PATH` para pasar `--model-path` y
   `--labels` al backend (útil cuando el SavedModel vive fuera del repositorio).
4. Exporta `HELEN_CAMERA_INDEX` (por defecto `0`) y `HELEN_BACKEND_EXTRA_ARGS` con los flags recomendados del nuevo
   modelo (`--confidence-threshold 0.75 --prediction-cooldown 0.5`).
5. Lanza `scripts/run-windows.ps1`, que inicia `python -m backendHelen.server`, espera a que `/health` responda y
   abre `http://localhost:5000` en Chrome/Edge.
6. Deja los logs en `reports\logs\win\backend-*.out.log` y `backend-*.err.log`.

#### Parámetros útiles de `helen-run.ps1`

| Parámetro      | Descripción                                                                          | Ejemplo                                                       |
|----------------|--------------------------------------------------------------------------------------|---------------------------------------------------------------|
| `-Port`        | Puerto del backend (por defecto 5000).                                               | `-Port 5050`                                                  |
| `-CameraIndex` | Índice numérico de la cámara (0,1,2).                                                | `-CameraIndex 1`                                              |
| `-ExtraArgs`   | Flags adicionales concatenados en `HELEN_BACKEND_EXTRA_ARGS`.                        | `-ExtraArgs "--confidence-threshold 0.8 --prediction-cooldown 0.4"` |
| `-SkipBrowser` | Evita abrir el navegador automáticamente (útil en sesiones remotas o headless).     | `-SkipBrowser`                                                |

> `scripts/run-windows.ps1` acepta los mismos parámetros y respeta `HELEN_TF_MODEL_PATH` / `HELEN_TF_LABELS_PATH` si los defines antes de ejecutarlo.

## 2. Flujo manual (cuando prefieras pasos individuales)

Sigue este camino si deseas comprender o personalizar cada etapa.

### 2.1 Clonar el repositorio y crear el entorno virtual

```powershell
cd $HOME\Documents
git clone https://github.com/tu-organizacion/HELEN.git
cd HELEN\HelenProyecto-main\HelenProyecto-main
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.2 Configurar variables y lanzar el backend

```powershell
$env:HELEN_CAMERA_INDEX = 0
$env:HELEN_TF_MODEL_PATH = "C:\\ruta\\al\\gesture_model_2024"
$env:HELEN_TF_LABELS_PATH = "C:\\ruta\\al\\gesture_model_2024\\labels.json"  # opcional si ya vive junto al modelo
$env:HELEN_BACKEND_EXTRA_ARGS = "--confidence-threshold 0.75 --prediction-cooldown 0.5"
.\.venv\Scripts\python.exe -m backendHelen.server --host 0.0.0.0 --port 5000 --camera-index $env:HELEN_CAMERA_INDEX --model-path $env:HELEN_TF_MODEL_PATH --labels $env:HELEN_TF_LABELS_PATH
```

- `--model-path` debe apuntar a la carpeta SavedModel exportada (`gesture_model_*/saved_model.pb`) o a un `.keras/.h5`.
- `--labels` puede omitirse si `labels.json` vive dentro de la carpeta del modelo.
- `--confidence-threshold` y `--prediction-cooldown` se pueden ajustar desde `HELEN_BACKEND_EXTRA_ARGS`.

Puedes sobrescribirlos en la línea de comandos:

```powershell
.\.venv\Scripts\python.exe -m backendHelen.server --host 0.0.0.0 --port 5000 --camera-index 1 --model-path D:\\gestures\\gesture_model_2024 --confidence-threshold 0.8 --prediction-cooldown 0.4
```

`backendHelen.server` expone `/health`, SSE y sirve la aplicación web desde la misma ruta.

### 2.3 Abrir la interfaz web

1. Visita `http://localhost:5000` en Chrome.
2. Concede permisos de cámara cuando aparezca el diálogo.
3. Comprueba que `/health` devuelve `{"status":"HEALTHY","camera_ok":true,...}`.
4. Cambia el color de fondo desde **Configuración → Personalización** para verificar que la UI
   responde inmediatamente.

## 3. Arquitectura y componentes

```text
+----------------------------+      HTTP / Socket.IO      +------------------------------+
|  Google Chrome (Frontend)  |  <-----------------------> |  backendHelen.server (Flask)  |
|  Reloj, temporizador, UI   |                            |  MediaPipe + OpenCV + SSE     |
+-------------+--------------+                            +---------------+--------------+
              |                                                          |
              v                                                          v
        Eventos de usuario                                    Cámara / pipeline de visión
```

- **Backend** (`backendHelen/`): Flask + Socket.IO, captura la cámara con OpenCV/MediaPipe y expone APIs REST/SSE.
- **Frontend** (`helen/`): aplicación web servida por Flask, guarda preferencias en `localStorage` (modo oscuro, color de
  fondo, etc.).

## 4. Checklist de validación rápida

1. **Backend activo**: la consola muestra `Running on http://0.0.0.0:5000` sin errores.
2. **Endpoint de salud**: `curl http://127.0.0.1:5000/health` devuelve `status=HEALTHY` y `camera_ok=true`.
3. **Video en vivo**: los módulos de cámara muestran imagen y landmarks en tiempo real.
4. **Temporizador**: inicia, pausa y reinicia sin saltos.
5. **Color de fondo**: cambiar el color aplica el tema inmediatamente y persiste tras recargar.
6. **Logs**: existen `reports\logs\win\backend-*.out.log` y `backend-*.err.log` después de ejecutar `run-windows.ps1` o
   `helen-run.ps1`.
7. **Cierre limpio**: al presionar `Ctrl+C` el backend se detiene sin `Traceback` inesperados.

## 5. Diagnósticos y herramientas

- **Endpoint de salud**:

  ```powershell
  curl http://127.0.0.1:5000/health
  ```

- **Diagnóstico de cámara (100 frames)**:

  ```powershell
  .\.venv\Scripts\python.exe -m backendHelen.diagnostics --frames 100
  ```

- **Logs**: revisa `reports\logs\win\backend-*.out.log` (stdout) y `backend-*.err.log` (stderr). Cada ejecución crea un par
  nuevo.

## 6. Solución de problemas frecuentes

1. **Cámara en negro / `camera_ok:false`**
   - Ejecuta `Get-PnpDevice -Class Camera` para listar webcams.
   - Cambia `-CameraIndex` o `HELEN_CAMERA_INDEX` a 1/2.
   - Revisa **Configuración → Privacidad y seguridad → Cámara** y habilita el acceso para PowerShell/`cmd` y Chrome.
2. **Permisos de cámara bloqueados en Chrome**
   - Abre `chrome://settings/content/camera` y permite `http://localhost:5000`.
   - Restablece permisos desde el candado en la barra de direcciones.
3. **Puerto 5000 ocupado**
   - Ejecuta `Get-NetTCPConnection -LocalPort 5000` para identificar el proceso y liberarlo si es seguro.
   - Lanza HELEN con `-Port 5050` y visita `http://localhost:5050`.
4. **`ImportError: DLL load failed` en OpenCV**
   - Reinstala VC++ Redistributable x64.
   - Forza la reinstalación: `pip install --force-reinstall opencv-python==4.9.0.80` dentro de `.venv`.
5. **Errores de TensorFlow/MediaPipe al inicializar**
   - Verifica `pip show tensorflow` (debe ser 2.12.x). Si falta instala `pip install tensorflow==2.12.0` dentro de `.venv`.
   - En CPUs sin AVX usa `pip install tensorflow-cpu==2.12.0` para evitar fallos de carga.
6. **Cámara IR/ToF seleccionada por error**
   - Cambia `-CameraIndex 1` o `2`.
   - Usa `Get-CimInstance Win32_PnPEntity | Where-Object {$_.Service -eq 'usbvideo'}` para identificar dispositivos.
7. **Chrome no solicita permisos**
   - Elimina permisos previos desde el candado → **Restablecer permisos**.
8. **CPU alta o lag**
   - Baja la ventana del frontend (Chrome) y deja solo la tarjeta principal visible para reducir renderizado.
   - Usa `-ExtraArgs "--sequence-length 12"` o incrementa `--prediction-cooldown` para espaciar inferencias.
9. **`ModuleNotFoundError` al iniciar**
   - Asegúrate de ejecutar desde `.venv` (`.\.venv\Scripts\activate`).
10. **Logs vacíos o truncados**
    - Si usas `helen-run.ps1`, revisa tanto `.out.log` (stdout) como `.err.log` (stderr). Cada script genera archivos
      separados para evitar colisiones de redirección en PowerShell.

## 7. Apéndices

### 7.1 Dependencias clave

| Paquete        | Versión fijada |
|----------------|----------------|
| Flask          | 3.0.3          |
| Flask-SocketIO | 5.3.6          |
| eventlet       | 0.36.1         |
| numpy          | 1.26.4         |
| opencv-python  | 4.9.0.80       |
| mediapipe      | 0.10.18        |

### 7.2 Flags soportados por `backendHelen.server`

| Flag                   | Descripción                                                                    | Ejemplo                                               |
|------------------------|--------------------------------------------------------------------------------|-------------------------------------------------------|
| `--camera-index`       | Índice numérico de la cámara utilizado por OpenCV.                            | `--camera-index 1`                                    |
| `--model-path`         | Carpeta SavedModel o archivo `.keras/.h5` del clasificador de video.          | `--model-path C:\models\gesture_model_2024`      |
| `--labels`             | Ruta a `labels.json` (opcional si vive junto al modelo).                      | `--labels C:\models\gesture_model_2024\labels.json` |
| `--confidence-threshold` | Probabilidad mínima para emitir un gesto al frontend.                        | `--confidence-threshold 0.8`                          |
| `--sequence-length`    | Número de frames consecutivos acumulados antes de inferir.                   | `--sequence-length 16`                                |
| `--prediction-cooldown` | En segundos, evita repetir la misma etiqueta durante la ventana indicada.    | `--prediction-cooldown 0.6`                           |
| `--host` / `--port`    | Dirección y puerto donde se expone el backend HTTP/SSE.                       | `--host 0.0.0.0 --port 5050`                          |

### 7.3 Preguntas frecuentes

- **¿Puedo usar Edge en lugar de Chrome?** Sí, siempre que sea la versión basada en Chromium.
- **¿Necesito reinstalar dependencias en cada ejecución?** No. `helen-run.ps1` verifica `.venv` y sólo reinstala si falta
  algo. Puedes forzar parámetros adicionales con `-ExtraArgs`.
- **¿Qué pasa si tengo varias cámaras USB?** Usa `-CameraIndex` para seleccionar la correcta o prueba `0/1/2`. HELEN utiliza
  OpenCV (CAP_DSHOW) por defecto y documenta cualquier error en `reports\\logs\\win`.
- **¿Dónde se guardan los logs?** En `reports\logs\win\backend-YYYYMMDD-HHMMSS.out.log` (stdout) y `.err.log` (stderr).

Mantén sincronizadas estas instrucciones cada vez que cambien los scripts o flags soportados para que HELEN siga siendo un
proyecto “sin sorpresas” al desplegarse en nuevos equipos Windows.
