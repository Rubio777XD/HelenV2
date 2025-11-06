# HELEN – Guía completa para ejecutar en Chrome (Windows)

Esta guía cubre el flujo oficial para poner en marcha HELEN en Windows 10/11 sin instaladores. A partir de esta
versión, todo el proceso puede automatizarse con un solo comando que prepara el entorno virtual, instala
dependencias, configura la cámara y abre la interfaz web en Google Chrome.

## 1. Inicio rápido (1 comando)

### 1.1 Pre-requisitos mínimos

| Requisito                          | Detalles                                                                 |
|-----------------------------------|--------------------------------------------------------------------------|
| Python                            | Python 3.11 instalado y agregado al `PATH` (incluye el *launcher* `py`). |
| PowerShell                        | PowerShell 7.0 o superior (ejecuta `pwsh -v` para verificar).            |
| Microsoft VC++ Redistributable    | Paquete 2015-2022 x64 (`vc_redist.x64.exe`).                             |
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

1. Detecta Python 3.11 disponible (`py -3.11`, `python`, etc.).
2. Crea o actualiza `.venv` con Python 3.11 (reemplaza entornos con versiones distintas).
3. Ejecuta `pip install --upgrade pip` y `pip install -r requirements.txt`.
4. Exporta `HELEN_CAMERA_INDEX` (por defecto `0`) y `HELEN_BACKEND_EXTRA_ARGS` con los flags recomendados
   (`--camera-backend directshow --camera-width 1280 --camera-height 720 --frame-stride 2 --poll-interval 0.08`).
5. Lanza `scripts/run-windows.ps1`, que inicia `python -m backendHelen.server`, espera a que `/health` responda y
   abre `http://localhost:5000` en Chrome/Edge.
6. Deja los logs en `reports\logs\win\backend-*.out.log` y `backend-*.err.log`.

#### Parámetros útiles de `helen-run.ps1`

| Parámetro        | Descripción                                                                                                   | Ejemplo                                         |
|------------------|---------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| `-Port`          | Puerto del backend (por defecto 5000).                                                                        | `-Port 5050`                                    |
| `-CameraIndex`   | Índice numérico de la cámara (0,1,2).                                                                         | `-CameraIndex 1`                                |
| `-ExtraArgs`     | Flags adicionales para el backend (se concatenan a `HELEN_BACKEND_EXTRA_ARGS`).                               | `-ExtraArgs "--camera-backend v4l2 --poll-interval 0.1"` |
| `-SkipBrowser`   | Evita abrir el navegador automáticamente (útil en sesiones remotas).                                          | `-SkipBrowser`                                  |

> `scripts/run-windows.ps1` acepta los mismos puertos y ahora incluye `-SkipBrowser`. Además se asegura de usar
> DirectShow, resolución 1280x720, `--frame-stride 2` y `--poll-interval 0.08` por defecto.

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
$env:HELEN_BACKEND_EXTRA_ARGS = "--camera-backend directshow --camera-width 1280 --camera-height 720 --frame-stride 2 --poll-interval 0.08"
.\.venv\Scripts\python.exe -m backendHelen.server --host 0.0.0.0 --port 5000
```

- `--camera-backend directshow` fuerza el uso de DirectShow en Windows (OpenCV `CAP_DSHOW`).
- `--camera-width/--camera-height` solicitan 1280x720.
- `--frame-stride` y `--poll-interval` reducen la carga de CPU.

Puedes sobrescribirlos en la línea de comandos:

```powershell
.\.venv\Scripts\python.exe -m backendHelen.server --host 0.0.0.0 --port 5000 --camera-backend dshow --camera-index 1 --camera-width 960 --camera-height 720 --frame-stride 3
```

`backendHelen.server` expone `/health`, SSE y sirve la aplicación web desde la misma ruta.

### 2.3 Abrir la interfaz web

1. Visita `http://localhost:5000` en Chrome.
2. Concede permisos de cámara cuando aparezca el diálogo.
3. Comprueba que `/health` devuelve `{"status":"HEALTHY","camera_ok":true,...}`.
4. Cambia el color de fondo desde **Configuración → Raspberry Pi → Color de fondo de HELEN** para verificar que la UI
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
   - Intenta `-ExtraArgs "--camera-backend v4l2"` o baja resolución `--camera-width 960 --camera-height 720`.
2. **Permisos de cámara bloqueados en Chrome**
   - Abre `chrome://settings/content/camera` y permite `http://localhost:5000`.
   - Restablece permisos desde el candado en la barra de direcciones.
3. **Puerto 5000 ocupado**
   - Ejecuta `Get-NetTCPConnection -LocalPort 5000` para identificar el proceso y liberarlo si es seguro.
   - Lanza HELEN con `-Port 5050` y visita `http://localhost:5050`.
4. **`ImportError: DLL load failed` en OpenCV**
   - Reinstala VC++ Redistributable x64.
   - Forza la reinstalación: `pip install --force-reinstall opencv-python==4.9.0.80` dentro de `.venv`.
5. **`mediapipe` reporta errores de GPU**
   - Añade `-ExtraArgs "--no-gpu"` o `--no-gpu` en `HELEN_BACKEND_EXTRA_ARGS` y ejecuta `helen-run.ps1` de nuevo.
6. **Cámara IR/ToF seleccionada por error**
   - Cambia `-CameraIndex 1` o `2`.
   - Usa `Get-CimInstance Win32_PnPEntity | Where-Object {$_.Service -eq 'usbvideo'}` para identificar dispositivos.
7. **Chrome no solicita permisos**
   - Elimina permisos previos desde el candado → **Restablecer permisos**.
8. **CPU alta o lag**
   - Incrementa `--frame-stride` (por ejemplo `-ExtraArgs "--frame-stride 3"`).
   - Reduce resolución: `-ExtraArgs "--camera-width 960 --camera-height 720"`.
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

| Flag                     | Descripción                                                   | Ejemplo                                         |
|--------------------------|---------------------------------------------------------------|-------------------------------------------------|
| `--camera-index`         | Índice numérico o ruta DirectShow (string).                   | `--camera-index 1`                              |
| `--camera-backend`       | Backend (`directshow`, `dshow`, `v4l2`).                      | `--camera-backend directshow`                   |
| `--camera-width/height`  | Resolución objetivo de captura.                              | `--camera-width 1280 --camera-height 720`       |
| `--frame-stride`         | Procesa 1 de cada *n* frames para reducir carga.             | `--frame-stride 3`                              |
| `--poll-interval`        | Intervalo entre lecturas de cámara en segundos.              | `--poll-interval 0.12`                          |
| `--no-camera`            | Desactiva la cámara física (usa stream sintético).           | `--no-camera`                                   |
| `--detection-confidence` | Ajusta umbral de detección de MediaPipe.                     | `--detection-confidence 0.6`                    |
| `--tracking-confidence`  | Ajusta umbral de tracking de MediaPipe.                      | `--tracking-confidence 0.5`                     |

### 7.3 Preguntas frecuentes

- **¿Puedo usar Edge en lugar de Chrome?** Sí, siempre que sea la versión basada en Chromium.
- **¿Necesito reinstalar dependencias en cada ejecución?** No. `helen-run.ps1` verifica `.venv` y sólo reinstala si falta
  algo. Puedes forzar parámetros adicionales con `-ExtraArgs`.
- **¿Qué pasa si tengo varias cámaras USB?** Usa `-CameraIndex` para seleccionar la correcta o prueba `0/1/2`. HELEN probará
  DirectShow primero y registrará sugerencias si la apertura falla.
- **¿Dónde se guardan los logs?** En `reports\logs\win\backend-YYYYMMDD-HHMMSS.out.log` (stdout) y `.err.log` (stderr).

Mantén sincronizadas estas instrucciones cada vez que cambien los scripts o flags soportados para que HELEN siga siendo un
proyecto “sin sorpresas” al desplegarse en nuevos equipos Windows.
