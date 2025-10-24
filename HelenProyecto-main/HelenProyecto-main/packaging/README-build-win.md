# HELEN – Guía de build e instalación en Windows

Este documento cubre dos escenarios:

1. **Flujo de desarrollo/QA**: ejecutar HELEN en Windows 10/11 con exactamente dos comandos (`scripts\setup-windows.ps1` y `scripts\run-windows.ps1`).
2. **Empaquetado**: generar un ejecutable autónomo con PyInstaller y un instalador Inno Setup.

Ambos flujos comparten dependencias fijadas y la misma estructura de rutas dentro de `HelenV2\HelenProyecto-main\HelenProyecto-main`.

## 1. Requisitos de entorno

- Windows 10/11 de 64 bits (build 19045+). Funciona en equipos con GPU integrada o solo CPU.
- [Python 3.11](https://www.python.org/downloads/) instalado con la opción *Add python.exe to PATH*.
- PowerShell 7+ (o Windows Terminal) con permisos para ejecutar scripts (`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`).
- [Visual C++ Redistributable 2015-2022](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist).
- Para empaquetar: [PyInstaller 6.6.0](https://pyinstaller.org) y [Inno Setup 6](https://jrsoftware.org/isinfo.php).

## 2. Flujo "dos comandos"

Desde la raíz del repositorio:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run-windows.ps1
```

`scripts\setup-windows.ps1`:
- Ubica el intérprete (`py -3.11`, `python3`, `python`) y crea `.venv\`.
- Instala NumPy 1.26.4, OpenCV 4.9.0.80, MediaPipe 0.10.18 y el resto de dependencias listadas en [`requirements.txt`](../requirements.txt) (Flask 3.0.3, Flask-SocketIO 5.3.6, eventlet 0.36.1, etc.).
- Ejecuta `pip check` y guarda un log en `reports\logs\win\setup-*.log`.

`scripts\run-windows.ps1`:
- Arranca `backendHelen.server` desde `.venv\` con los argumentos por defecto (`--host 0.0.0.0 --port 5000`).
- Espera a que `http://127.0.0.1:5000/health` responda y abre el navegador predeterminado en `http://localhost:5000`.
- Guarda los logs del backend en `reports\logs\win\backend-*.log`.
- Permite personalizar variables antes de ejecutarlo:
  - `$env:HELEN_CAMERA_INDEX` (`auto`, índice numérico o ruta DirectShow).
  - `$env:HELEN_BACKEND_EXTRA_ARGS="--frame-stride 2 --poll-interval 0.08"` para ajustar los defaults.

Si necesitas ejecutar el backend sin UI usa `$env:HELEN_NO_UI=1`.

## 3. Endpoint `/health` y validaciones

- `Invoke-WebRequest http://127.0.0.1:5000/health | ConvertFrom-Json` debe devolver `status="HEALTHY"` y `camera_ok=true` cuando la webcam está disponible.
- Los campos `camera_backend` y `camera_resolution` reflejan la auto detección (DirectShow → UVC HD → degradaciones a 640×480 si es necesario).
- Los logs de cámara residen en `reports\logs\win\backend-*.log` junto a los eventos SSE.

## 4. Empaquetado con PyInstaller

1. (Opcional) Crea un entorno limpio:
   ```powershell
   python -m venv packaging\build-env
   .\packaging\build-env\Scripts\Activate.ps1
   python -m pip install --upgrade pip wheel setuptools
   python -m pip install -r packaging\requirements-win.txt
   ```
2. Genera el ejecutable:
   ```powershell
   python -m PyInstaller --clean --noconfirm packaging/helen_backend.spec
   ```
3. Verifica el binario:
   ```powershell
   cd dist\helen-backend
   .\helen-backend.exe --no-camera --host 127.0.0.1 --port 8765
   Start-Sleep -Seconds 5
   Invoke-WebRequest -Uri http://127.0.0.1:8765/health | Select-Object StatusCode
   Stop-Process -Name "helen-backend" -ErrorAction SilentlyContinue
   ```

El directorio `dist\helen-backend` incluye `helen-backend.exe`, los recursos del frontend (`helen\`), modelos MediaPipe y el dataset `data.pickle`/`data1.pickle` si están presentes.

### Modelos y datasets

- `Hellen_model_RN/data.pickle` no se versiona por su tamaño. Copia el archivo antes de empaquetar; de lo contrario el backend usará `data1.pickle` y registrará una advertencia.
- Las dependencias de entrenamiento (scikit-learn, xgboost) permanecen en `Hellen_model_RN/requirements.txt` y no forman parte del runtime.

## 5. Instalador con Inno Setup (opcional)

1. Abre `packaging/inno_setup.iss` en Inno Setup 6 o ejecuta:
   ```powershell
   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\inno_setup.iss
   ```
2. El instalador `dist/Setup Helen.exe` copiará los archivos a `C:\Program Files\HELEN` y creará accesos directos.
3. Prueba en una VM limpia:
   - Ejecuta el instalador.
   - Abre HELEN desde el menú Inicio.
   - Verifica `/health` y el funcionamiento de la cámara real.

## 6. Solución de problemas

| Problema | Diagnóstico | Acción |
|----------|-------------|--------|
| `python` no encontrado | `scripts\setup-windows.ps1` falla al resolver intérprete. | Instala Python 3.11 desde python.org y vuelve a ejecutar el script. |
| `mediapipe` no carga | El log reporta `ImportError`. | Reinstala con `.venv\Scripts\python.exe -m pip install mediapipe==0.10.18`. |
| Cámara negra | `/health` indica `camera_ok:false`. | Revisa permisos de la webcam, fuerza `$env:HELEN_CAMERA_INDEX=0` y reinicia el script. |
| Empaquetado sin `data.pickle` | El log muestra la advertencia correspondiente. | Copia el archivo a `Hellen_model_RN\` antes de ejecutar PyInstaller. |
| Inno Setup falla | `ISCC.exe` no se encuentra. | Verifica la ruta de instalación de Inno Setup o ajusta el comando. |

## 7. Mantenimiento

- Mantén sincronizados `requirements.txt` y `packaging/requirements-win.txt` con los imports reales del backend.
- Actualiza `packaging/helen_backend.spec` si cambias la estructura de archivos estáticos o datasets.
- Documenta cualquier cambio en `CHANGELOG_DOCS.md` junto con los pasos de verificación.

Con estos pasos puedes trabajar en Windows con un flujo simple de dos comandos y, cuando sea necesario, generar builds reproducibles listas para distribución.
