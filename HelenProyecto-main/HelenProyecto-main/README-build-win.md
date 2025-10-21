# HELEN – Guía de build e instalación en Windows

Esta guía resume el flujo oficial para generar el ejecutable de HELEN, empaquetarlo con Inno Setup y preparar un instalador que funcione en una máquina limpia (sin Python preinstalado). Incluye los pasos para manejar los artefactos de modelo —incluido el archivo grande `data.pickle`— y las validaciones mínimas antes de entregar el instalador.

## 1. Requisitos del entorno

- Windows 10/11 de 64 bits.
- [Python 3.11](https://www.python.org/downloads/) instalado y agregado al `PATH`.
- [Visual C++ Redistributable 2015-2022](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) (PyInstaller lo requiere para algunas dependencias nativas).
- [Inno Setup 6](https://jrsoftware.org/isinfo.php) (para generar el instalador `Setup Helen.exe`).
- PowerShell 5+ o Windows Terminal para ejecutar los comandos.

> **Sugerencia:** Trabaja dentro del directorio raíz del proyecto (`HelenProyecto-main/HelenProyecto-main`).

## 2. Preparar el entorno Python

1. Crea y activa un entorno virtual (opcional pero recomendado):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Instala las dependencias de ejecución enumeradas en `packaging/requirements-win.txt`:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r packaging/requirements-win.txt
   ```
3. Instala las herramientas de build:
   ```powershell
   python -m pip install pyinstaller==6.6.0 pyinstaller-hooks-contrib==2024.6 requests==2.31.0
   ```

## 3. Colocar los modelos y `data.pickle`

La carpeta `Hellen_model_RN/` debe contener estos archivos:

- `model.p` y `model.json` (modelo de producción).
- `data1.pickle` (dataset histórico incluido en el repo).
- `data.pickle` (**no se versiona** por su tamaño). Coloca el archivo manualmente en `Hellen_model_RN/` antes de ejecutar PyInstaller. Si falta, el backend registrará el mensaje:
  
  > `data.pickle no encontrado (archivo grande omitido del repo, ver documentación de build)`

Cuando `data.pickle` está presente se usa automáticamente; si no, se recurre a `data1.pickle` y se muestra el aviso anterior para que puedas incluirlo en la build final.

## 4. Generar el ejecutable con PyInstaller

Ejecuta el `spec` preparado en `packaging/helen_backend.spec`:

```powershell
python -m PyInstaller --clean --noconfirm packaging/helen_backend.spec
```

El resultado queda en `dist/helen-backend/` e incluye:

- `helen-backend.exe`: ejecutable principal.
- Assets estáticos (frontend en `helen/`, plantillas, modelos MediaPipe, etc.).
- Librerías nativas requeridas (`opencv`, `mediapipe`, `xgboost.dll`, `sounddevice`).

### Verificación rápida

Antes de empaquetar:

```powershell
cd dist\helen-backend
.\helen-backend.exe --no-camera --host 127.0.0.1 --port 8765
```

En otra terminal:

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8765/health | Select-Object StatusCode
```

Debe devolver `StatusCode : 200`. Presiona `Ctrl+C` para detener el servidor.

## 5. Generar el instalador con Inno Setup

1. Asegúrate de que Inno Setup esté instalado en `C:\Program Files (x86)\Inno Setup 6`.
2. Ejecuta el script incluido:
   ```powershell
   "C:\\Program Files (x86)\\Inno Setup 6\\ISCC.exe" packaging\inno_setup.iss
   ```
3. Se creará `dist/Setup Helen.exe`. Copia junto con `dist/helen-backend/` si necesitas depurar.

## 6. Prueba del instalador en limpio

- Ejecuta `Setup Helen.exe` en una VM o equipo sin Python.
- El instalador copia el backend y los assets a `C:\Program Files\HELEN` y crea accesos directos.
- Inicia `helen-backend.exe`. Debe abrir la cámara real si está disponible y mantener operativo el endpoint `/health`.
- Verifica que el log distinga claramente entre cámara ausente, MediaPipe no instalado o modelo faltante.

## 7. Automatización en GitHub Actions

El workflow `packaging/windows-build.yml` automatiza el proceso:

1. Usa `actions/setup-python@v5` con Python 3.11.
2. Instala dependencias (`requirements-win.txt`, PyInstaller, hooks, requests).
3. Ejecuta PyInstaller y después Inno Setup.
4. Realiza una prueba de humo: lanza `helen-backend.exe --no-camera` en el puerto `8765` y consulta `http://127.0.0.1:8765/health`.
5. Sube como artefactos `dist/helen-backend/` y `dist/Setup Helen.exe`.

> **Nota:** El workflow asume que `data.pickle` ya está presente en el repositorio privado (por ejemplo mediante artefactos o secretos). Añádelo antes de disparar la acción para evitar advertencias.

## 8. Solución de problemas comunes

| Problema | Solución |
| --- | --- |
| `data.pickle` no encontrado | Copia el archivo a `Hellen_model_RN/`. El log mostrará el aviso exacto hasta que esté disponible. |
| La cámara no responde | Ejecuta el backend con `--no-camera` para verificar el resto del flujo; revisa permisos del dispositivo en Windows. |
| Error al cargar `mediapipe` o `opencv` | Reinstala dependencias usando `packaging/requirements-win.txt` y verifica que no falten librerías nativas en `dist/helen-backend/`. |
| El instalador no arranca | Confirma que Inno Setup esté instalado y que la ruta al ejecutable sea correcta. |

## 9. Mantenimiento

- Actualiza `packaging/requirements-win.txt` solo con dependencias de runtime; las herramientas de build se instalan aparte (workflow y guía).
- Documenta cualquier cambio en la estructura del modelo para que `helen_backend.spec` pueda incluir nuevos archivos.
- Conserva esta guía sincronizada con los ajustes del pipeline de CI.

¡Listo! Con estos pasos tendrás un `.exe` funcional, un instalador preparado para usuarios finales y la automatización en CI alineada con el flujo manual.
