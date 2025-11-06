# HELEN – Guía completa para ejecutar en Chrome (Windows)

Esta guía describe paso a paso cómo clonar, configurar y ejecutar HELEN en Windows 10/11 utilizando Google Chrome
(como navegador recomendado) sin recurrir a instaladores o empaquetados. Sigue cada sección en orden para obtener un
backend funcional y una interfaz web lista para usarse con cámara.

## 1. Resumen del flujo

1. Instalar los prerrequisitos del sistema.
2. Clonar el repositorio y crear un entorno virtual de Python 3.11.
3. Instalar las dependencias del backend (Flask, MediaPipe, OpenCV, etc.).
4. Definir las variables de entorno y lanzar `backendHelen.server`.
5. Abrir `http://localhost:5000` en Google Chrome, conceder permisos de cámara y verificar el estado en `/health`.

## 2. ¿Qué es HELEN?

- **Backend** (`backendHelen/`): servicio Flask + Socket.IO que captura la cámara, procesa gestos con MediaPipe/OpenCV y
  expone endpoints REST/SSE.
- **Frontend** (`helen/`): aplicación web servida por el backend, incluye reloj, temporizador, alarmas y controles de
  accesibilidad. Las preferencias (p. ej. color de fondo) se guardan en `localStorage` y se aplican tanto en Windows como
  en Raspberry Pi sin pasos adicionales.

## 3. Arquitectura mínima

```
+----------------------------+      HTTP / Socket.IO      +------------------------------+
|  Google Chrome (Frontend)  |  <-----------------------> |  backendHelen.server (Flask)  |
|  Reloj, temporizador, UI   |                            |  MediaPipe + OpenCV + SSE     |
+-------------+--------------+                            +---------------+--------------+
              |                                                          |
              v                                                          v
        Eventos de usuario                                    Cámara / pipeline de visión
```

## 4. Requisitos del sistema

| Requisito                                   | Detalles                                                                 |
|---------------------------------------------|--------------------------------------------------------------------------|
| Sistema operativo                           | Windows 10 u 11 de 64 bits                                              |
| Python                                      | Python 3.11 instalado y agregado al `PATH`                              |
| PowerShell                                  | PowerShell 7.0 o superior                                               |
| Microsoft VC++ Redistributable              | x64 2015-2022 (`vc_redist.x64.exe`)                                      |
| Navegador                                   | Google Chrome 124+ (o Microsoft Edge basado en Chromium)                |
| Hardware                                    | Webcam UVC compatible y permisos de cámara para el usuario actual       |

> Sugerencia: reinicia el sistema después de instalar Python y el VC++ Redistributable para asegurar que el `PATH`
> quede actualizado.

## 5. Descargar el repositorio

```powershell
cd $HOME\Documents
git clone https://github.com/tu-organizacion/HELEN.git
cd HELEN\HelenProyecto-main\HelenProyecto-main
```

## 6. Crear y activar el entorno virtual

Ejecuta los comandos dentro de PowerShell 7 (no en CMD):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Problemas frecuentes al instalar dependencias

- **Error al compilar `mediapipe` u `opencv-python`**: verifica que el VC++ Redistributable esté instalado. Si el error
  persiste, instala las ruedas precompiladas más recientes con `pip install --force-reinstall mediapipe==0.10.18
  opencv-python==4.9.0.80`.
- **`pip` no reconoce el comando**: confirma que estás dentro del entorno virtual (`(venv)` en el prompt). Vuelve a
  ejecutar `.\.venv\Scripts\activate` si es necesario.

## 7. Ejecutar el backend

Inicia el servidor con las variables de entorno recomendadas para Windows:

```powershell
$env:HELEN_CAMERA_INDEX = 0
$env:HELEN_BACKEND_EXTRA_ARGS = "--frame-stride 2 --poll-interval 0.08"
.\.venv\Scripts\python.exe -m backendHelen.server --host 0.0.0.0 --port 5000
```

- `HELEN_CAMERA_INDEX`: índice de la cámara (0 = webcam integrada). Cambia a 1/2 si tienes varias cámaras.
- `HELEN_BACKEND_EXTRA_ARGS`: argumentos adicionales que se pasan al backend. Puedes añadir flags como
  `--camera-backend directshow` o `--camera-width 1280` según lo necesites.

El servidor queda escuchando en `http://0.0.0.0:5000`. Mantén la consola abierta para observar los logs.

## 8. Abrir el frontend en Chrome

1. Lanza Google Chrome.
2. Visita `http://localhost:5000`.
3. Concede permisos de cámara cuando se soliciten.
4. Verifica que el reloj muestre la hora local, el temporizador responda y el fondo pueda cambiarse desde
   **Configuración → Raspberry Pi → Color de fondo de HELEN** (el ajuste se aplica inmediatamente en Windows).

## 9. Checklist de validación manual

Marca cada paso una vez completado:

1. **Backend activo**: la consola de PowerShell muestra `Running on http://0.0.0.0:5000` sin errores.
2. **Endpoint de salud**: `curl http://127.0.0.1:5000/health` devuelve `"status":"HEALTHY"` y `"camera_ok":true`.
3. **Reloj en la página Home**: refleja la hora del sistema y se actualiza cada segundo.
4. **Temporizador**: inicia, pausa y reinicia desde la página Clock sin saltos en los segundos.
5. **Color de fondo**: al elegir un color distinto al azul se actualiza el tema inmediatamente y persiste tras recargar.
6. **Permisos de cámara**: Chrome muestra video en vivo en los módulos que lo requieren.
7. **Logs**: existe un archivo `reports\logs\win\backend-*.log` cuando utilizas `scripts\run-windows.ps1`.
8. **Cierre limpio**: al presionar `Ctrl+C` el backend se detiene sin stack traces inesperados.

## 10. Diagnóstico rápido

- **Endpoint de salud**:

  ```powershell
  curl http://127.0.0.1:5000/health
  ```

  Debe regresar `status=HEALTHY`, `camera_ok=true` y los datos de la cámara seleccionada.

- **Vista previa de landmarks (opcional)**:

  ```powershell
  .\.venv\Scripts\python.exe -m backendHelen.diagnostics --frames 100
  ```

  Genera estadísticas de captura sin iniciar la interfaz gráfica.

## 11. Solución de problemas (8+ casos)

1. **Cámara en negro / `camera_ok:false`**  
   Ejecuta `Get-PnpDevice -Class Camera` para confirmar el dispositivo. Ajusta `HELEN_CAMERA_INDEX` o exporta
   `--camera-backend directshow` en `HELEN_BACKEND_EXTRA_ARGS`.
2. **Permisos de cámara bloqueados en Chrome**  
   Abre `chrome://settings/content/camera` y habilita el acceso para `http://localhost:5000`.
3. **Backend no arranca por puerto en uso**  
   Cambia a `--port 5050` y actualiza la URL en Chrome. Libera el puerto con `Stop-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess` si es seguro hacerlo.
4. **Error `ImportError: DLL load failed` en OpenCV**  
   Reinstala el VC++ Redistributable x64 y luego `pip install --force-reinstall opencv-python==4.9.0.80`.
5. **`mediapipe` arroja errores de GPU**  
   Añade `--no-gpu` en `HELEN_BACKEND_EXTRA_ARGS` para forzar CPU y asegúrate de contar con drivers actualizados.
6. **No se detecta la cámara correcta en laptops con cámara IR**  
   Cambia `HELEN_CAMERA_INDEX` a 1 o 2. Puedes listar cámaras con `Get-CimInstance Win32_PnPEntity | Where-Object {$_.Service -eq 'usbvideo'}`.
7. **Chrome no solicita permisos**  
   Si ya los denegaste, haz clic en el candado de la barra de direcciones y restablece los permisos a "Permitir".
8. **Lag o CPU alta**  
   Incrementa `--frame-stride` a 3 o `--poll-interval` a 0.12. Reduce la resolución con
   `--camera-width 960 --camera-height 720` si es necesario.
9. **`ModuleNotFoundError` al lanzar `backendHelen.server`**  
   Asegúrate de activar el entorno virtual antes (`.\.venv\Scripts\activate`).

## 12. Rendimiento y calidad

- `--frame-stride`: procesa un frame cada *n* capturas (2 por defecto en Windows).
- `--poll-interval`: controla el tiempo entre lecturas de cámara. Aumentarlo reduce la carga de CPU.
- Usa iluminación uniforme y coloca la cámara a la altura del rostro para mejores predicciones.

## 13. Persistencia de preferencias de la UI

La interfaz almacena configuraciones como el color de fondo en `localStorage` bajo las llaves `helen:display-mode` y
`helen:background-color`. El selector de color disponible en Configuración actualiza la variable CSS `--bg`, lo que
permite que Windows refleje el mismo tema que Raspberry Pi.

## 14. Buenas prácticas y contribuciones

- Sigue el formato de commits estilo `tipo: descripción` (p. ej. `docs: actualizar guía de Windows`).
- Antes de abrir un PR, ejecuta `scripts/run-windows.ps1` para validar `/health`.
- Asegúrate de actualizar la documentación correspondiente cada vez que cambien los scripts o variables.

## 15. Apéndice A – Dependencias clave

| Paquete             | Versión fijada |
|---------------------|----------------|
| Flask               | 3.0.3          |
| Flask-SocketIO      | 5.3.6          |
| eventlet            | 0.36.1         |
| numpy               | 1.26.4         |
| opencv-python       | 4.9.0.80       |
| mediapipe           | 0.10.18        |

## 16. Apéndice B – Flags de cámara útiles

| Flag                     | Descripción                                         | Ejemplo                                       |
|--------------------------|-----------------------------------------------------|-----------------------------------------------|
| `--camera-index`         | Selecciona el índice numérico de la cámara.         | `--camera-index 1`                            |
| `--camera-backend`       | Fuerza un backend específico (`directshow`).        | `--camera-backend directshow`                 |
| `--camera-width`/`height`| Ajusta la resolución solicitada al dispositivo.     | `--camera-width 1280 --camera-height 720`     |
| `--frame-stride`         | Salta frames para reducir carga de CPU.             | `--frame-stride 3`                            |
| `--poll-interval`        | Intervalo entre lecturas de cámara en segundos.     | `--poll-interval 0.12`                        |

## 17. Apéndice C – Mini FAQ

- **¿Puedo usar Edge en lugar de Chrome?** Sí, siempre que sea la versión basada en Chromium.
- **¿Funciona con múltiples usuarios en Windows?** Cada usuario debe crear su propio `.venv` y otorgar permisos de cámara.
- **¿Cómo actualizo HELEN?** Ejecuta `git pull`, reactiva el entorno virtual y vuelve a correr `pip install -r requirements.txt`.
- **¿Hay soporte para GPU dedicada?** MediaPipe corre en CPU por defecto; el soporte GPU en Windows no está habilitado.
