# HELEN – Guía completa para ejecutar en Chrome (Linux / Raspberry Pi)

Esta guía describe el flujo oficial para correr HELEN en distribuciones Debian/Ubuntu y Raspberry Pi OS empleando
Chromium o Google Chrome. Se enfoca exclusivamente en la ejecución directa del backend Python y el consumo del frontend
web desde el navegador, sin empaquetados adicionales.

## 1. Resumen del flujo

1. Instalar dependencias del sistema (Python 3.11, bibliotecas nativas de cámara, Chromium/Chrome).
2. Clonar el repositorio y crear un entorno virtual.
3. Instalar los requisitos de Python definidos en `requirements.txt`.
4. Exportar variables de entorno recomendadas y lanzar `backendHelen.server`.
5. Abrir `http://localhost:5000` en Chromium/Chrome, conceder permisos de cámara y validar `/health`.

## 2. Componentes de HELEN

- **Backend** (`backendHelen/`): aplicación Flask + Socket.IO que procesa frames de la cámara con MediaPipe/OpenCV,
  expone endpoints REST y eventos SSE.
- **Frontend** (`helen/`): interfaz web servida por Flask, incluye reloj, temporizador y controles táctiles optimizados
  para kioskos. Las preferencias como el color de fondo se guardan en `localStorage`, compartiendo el mismo flujo que en
  Windows.

## 3. Arquitectura mínima

```
+-----------------------------+     HTTP / Socket.IO     +------------------------------+
|  Chromium / Google Chrome   | <----------------------> |  backendHelen.server (Flask)  |
|  UI táctil + accesibilidad  |                         |  MediaPipe + OpenCV + SSE     |
+--------------+--------------+                         +---------------+--------------+
               |                                                         |
               v                                                         v
        Interacción del usuario                                Cámara UVC / CSI / V4L2
```

## 4. Requisitos del sistema

| Requisito                         | Detalles                                                                          |
|-----------------------------------|-----------------------------------------------------------------------------------|
| Distribución                      | Debian 12, Ubuntu 22.04+ o Raspberry Pi OS (Bookworm)                             |
| Python                            | Python 3.11 + `python3.11-venv`                                                   |
| Librerías del sistema             | `sudo apt install python3.11 python3.11-venv python3-pip libatlas-base-dev \
                                     libopenblas-dev liblapack-dev libjpeg-dev libqt6gui6 libqt6core6 libqt6opengl6 \
                                     ffmpeg v4l-utils`                                                                  |
| Navegador                         | `chromium-browser` o `google-chrome-stable`                                      |
| Hardware                          | Cámara UVC/CSI conectada a `/dev/video*` con permisos para el usuario (`video`)  |

Tras instalar los paquetes, agrega el usuario actual al grupo `video` si aún no pertenece:

```bash
sudo usermod -aG video $USER
newgrp video
```

## 5. Descargar el repositorio

```bash
cd ~/helen
mkdir -p ~/helen
cd ~/helen
git clone https://github.com/tu-organizacion/HELEN.git
cd HELEN/HelenProyecto-main/HelenProyecto-main
```

## 6. Crear y activar el entorno virtual

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Problemas frecuentes al instalar dependencias

- **Errores de compilación en `mediapipe`**: asegúrate de contar con suficiente RAM e intercambio. En Raspberry Pi 4/5 se
  recomienda activar un swap de al menos 2 GB (`sudo dphys-swapfile swapoff && sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile && sudo dphys-swapfile setup && sudo dphys-swapfile swapon`).
- **Faltan bibliotecas Qt**: ejecuta `sudo apt install libqt6gui6 libqt6core6 libqt6opengl6`.
- **`pip` intenta usar Python 3.9**: verifica que `python3.11` está primero en el `PATH` o usa rutas absolutas (p. ej.
  `/usr/bin/python3.11 -m venv .venv`).

## 7. Ejecutar el backend

Lanza el servidor con los parámetros recomendados para Linux / Raspberry Pi:

```bash
export HELEN_CAMERA_INDEX=0
export HELEN_BACKEND_EXTRA_ARGS="--frame-stride 2 --poll-interval 0.08 --camera-backend v4l2 --camera-width 1280 --camera-height 720"
python -m backendHelen.server --host 0.0.0.0 --port 5000
```

- `HELEN_CAMERA_INDEX`: índice (`0`, `1`, etc.) o ruta (`/dev/video0`).
- `HELEN_BACKEND_EXTRA_ARGS`: flags opcionales que ajustan backend y cámara. Cambia `--camera-backend` a `gstreamer` si
  utilizas módulos CSI en Raspberry Pi 5.

Mantén la terminal abierta para revisar mensajes y estadísticas de captura.

## 8. Abrir el frontend en Chromium / Chrome

1. Ejecuta `chromium-browser --app=http://localhost:5000` o abre la URL manualmente.
2. Concede permisos de cámara cuando se te soliciten.
3. Navega a **Configuración → Raspberry Pi** y prueba el selector de color para confirmar que el fondo cambia al instante.

## 9. Checklist de validación manual

1. **Backend activo**: la terminal muestra `Running on http://0.0.0.0:5000` sin excepciones.
2. **Endpoint `/health`**: `curl http://127.0.0.1:5000/health` devuelve `"status":"HEALTHY"` y `"camera_ok":true`.
3. **Vista del reloj**: la hora coincide con `date` del sistema y se actualiza cada segundo.
4. **Temporizador**: responde a iniciar/pausar/reiniciar sin saltos al reanudar.
5. **Selector de color**: cambiar el tono base modifica la variable `--bg` sin afectar halos ni animaciones.
6. **Permisos V4L2**: `v4l2-ctl --list-devices` lista la cámara activa sin errores.
7. **Uso de CPU**: `top` muestra el proceso de Python estable (50-120% en Pi 4/5 dependiendo de la cámara).
8. **Apagado controlado**: `Ctrl+C` detiene el servidor y libera `/dev/video*` (compruébalo con `lsof /dev/video0`).

## 10. Diagnóstico rápido

- **Endpoint de salud**:

  ```bash
  curl http://127.0.0.1:5000/health
  ```

- **Diagnóstico de frames**:

  ```bash
  python -m backendHelen.diagnostics --frames 100
  ```

- **Inspección de dispositivos**:

  ```bash
  v4l2-ctl --list-devices
  ```

## 11. Solución de problemas (8+ casos)

1. **`camera_ok:false` en `/health`**  
   Cambia `HELEN_CAMERA_INDEX` a `/dev/video1` o utiliza `--camera-backend gstreamer` si usas cámara CSI.
2. **Permisos insuficientes sobre `/dev/video*`**  
   Ejecuta `sudo usermod -aG video $USER` y vuelve a iniciar sesión.
3. **Chromium muestra pantalla negra**  
   Lanza el navegador con `chromium-browser --use-gl=desktop --enable-features=VaapiVideoDecoder`.
4. **Errores de `mediapipe` sobre `GLIBCXX`**  
   Instala `sudo apt install libstdc++6` actualizado o utiliza `pip install mediapipe==0.10.18 --force-reinstall`.
5. **CPU excesiva en Raspberry Pi 4**  
   Ajusta `HELEN_BACKEND_EXTRA_ARGS` a `--frame-stride 4 --poll-interval 0.12 --camera-width 640 --camera-height 360`.
6. **El backend no libera la cámara al cerrar**  
   Si `lsof /dev/video0` sigue mostrando el proceso, mata el PID con `kill <PID>` y vuelve a iniciar.
7. **No existe `python3.11` en la distribución**  
   Añade el PPA `sudo add-apt-repository ppa:deadsnakes/ppa` (Ubuntu) o instala la versión disponible desde `pyenv`.
8. **Errores de certificados TLS al hacer `pip install`**  
   Configura `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org` temporalmente.
9. **La UI no carga assets estáticos**  
   Asegúrate de ejecutar el backend desde la raíz del proyecto; revisa que `FLASK_ENV` no esté apuntando a otra carpeta.

## 12. Rendimiento y calidad

- Incrementa `--frame-stride` y `--poll-interval` en Raspberry Pi para equilibrar latencia vs. consumo.
- Utiliza iluminación frontal y evita fondos saturados para mejorar el tracking.
- Para kioskos táctiles, combina `chromium-browser --app` con flags como `--kiosk` o `--start-fullscreen`.

## 13. Persistencia de preferencias de la UI

Los ajustes de accesibilidad (modo Raspberry, color de fondo) se almacenan en `localStorage`. El backend no necesita
reiniciarse para aplicar cambios; el selector de color modifica la variable CSS `--bg`, reutilizada en Linux y Windows.

## 14. Buenas prácticas y contribuciones

- Realiza commits atómicos describiendo cambios (`docs: detallar setup en Raspberry Pi`).
- Ejecuta `scripts/run-pi.sh` (si lo prefieres) para automatizar los pasos anteriores y confirma `/health` antes de
  enviar PRs.
- Documenta cualquier ajuste adicional en `CHANGELOG.md`.

## 15. Apéndice A – Dependencias clave

| Paquete             | Versión |
|---------------------|---------|
| Flask               | 3.0.3   |
| Flask-SocketIO      | 5.3.6   |
| eventlet            | 0.36.1  |
| numpy               | 1.26.4  |
| opencv-python       | 4.9.0.80|
| mediapipe           | 0.10.18 |

## 16. Apéndice B – Flags de cámara recomendados

| Flag                              | Descripción                                        | Ejemplo                                             |
|-----------------------------------|----------------------------------------------------|-----------------------------------------------------|
| `--camera-index` / `--camera`     | Selecciona índice o ruta del dispositivo.          | `--camera /dev/video1`                              |
| `--camera-backend`                | Fuerza backend (`v4l2`, `gstreamer`).              | `--camera-backend gstreamer`                        |
| `--camera-width` `--camera-height`| Ajusta resolución objetivo.                         | `--camera-width 960 --camera-height 720`            |
| `--frame-stride`                  | Salta frames para reducir carga.                   | `--frame-stride 4`                                  |
| `--poll-interval`                 | Intervalo entre lecturas (segundos).               | `--poll-interval 0.1`                               |
| `--no-gpu`                        | Desactiva GPU si causa problemas.                  | `--no-gpu`                                          |

## 17. Apéndice C – Mini FAQ

- **¿Puedo usar Firefox?** No se recomienda: el flujo soportado es Chrome/Chromium por su compatibilidad con WebRTC.
- **¿Cómo arranco HELEN al inicio del sistema?** Crea un servicio `systemd` que ejecute el comando descrito en la sección 7.
- **¿Chromium en modo kiosko pierde el puntero?** Añade `--simulate-outdated-no-au='Tue, 31 Dec 2099 23:59:59 GMT'` para
  evitar avisos de actualizaciones.
- **¿Dónde se guardan los logs?** En `reports/logs/pi/` cuando usas `scripts/run-pi.sh`; también puedes redirigir stdout a un
  archivo manualmente.
