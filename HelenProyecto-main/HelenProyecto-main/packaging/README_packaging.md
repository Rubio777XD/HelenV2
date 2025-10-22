# Empaquetado de HELEN para Windows

Este directorio contiene todo lo necesario para generar un ejecutable y un
instalador de HELEN en Windows utilizando PyInstaller e Inno Setup.

## Estructura

- `requirements-win.txt`: dependencias específicas para compilar e instalar el
  backend (MediaPipe, OpenCV, XGBoost, etc.).
- `helen_backend.spec`: especificación de PyInstaller que empaqueta el backend,
  el modelo y los recursos del frontend.
- `inno_setup.iss`: script de Inno Setup que genera `Setup Helen.exe` a partir
  del resultado de PyInstaller.
- `build_windows.ps1`: script principal que automatiza la creación del entorno,
  la instalación de dependencias, la compilación y el diagnóstico de cámara.
- `windows-build.yml`: workflow de GitHub Actions que ejecuta el empaquetado en
  `windows-latest` y publica los artefactos.

## Requisitos previos locales

1. Windows 10/11 con Python 3.11 disponible en `PATH` (misma versión que CI).
2. Herramientas de compilación de Visual Studio (recomendado para XGBoost).
3. Inno Setup 6 instalado en `C:\Program Files (x86)\Inno Setup 6` si se desea
   generar el instalador.
4. Acceso a Internet para descargar dependencias.

## Construir el ejecutable con PyInstaller

```powershell
cd <ruta-del-repo>\packaging
# Ejecutar el script (crea venv, instala dependencias y corre PyInstaller)
./build_windows.ps1 -AllowMissingCamera
```

- Opciones útiles:
  - `-AllowMissingCamera`: evita que el diagnóstico falle en máquinas sin cámara (por ejemplo, entornos virtuales).
  - `-SkipInstaller`: omite la ejecución de Inno Setup (útil para pruebas rápidas).

- La opción `-AllowMissingCamera` evita que el diagnóstico falle en máquinas sin
  cámara (por ejemplo, entornos virtuales).
- El ejecutable se genera en `dist\helen-backend\helen-backend.exe`.

Para forzar un entorno limpio puedes borrar la carpeta `build-venv` y
`dist/` antes de ejecutar el script.

## Generar el instalador con Inno Setup

Si `ISCC.exe` está disponible el script `build_windows.ps1` ejecutará Inno Setup
al finalizar PyInstaller. Si prefieres invocarlo manualmente:

```powershell
# Después de ejecutar PyInstaller
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\inno_setup.iss
```

El instalador se generará como `dist\Setup Helen.exe`.

## Diagnóstico de cámara

El script ejecuta `python -m backendHelen.diagnostics` al terminar la
compilación. Este diagnóstico comprueba que OpenCV y MediaPipe pueden abrir la
cámara y detectar gestos básicos.

- Salida esperada con cámara disponible:
  - `Cámara <n> operativa. Frames válidos: X/Y`
  - Ningún mensaje de error.
- Si no hay cámara y se usa `-AllowMissingCamera`, se mostrará una advertencia
  pero el proceso continuará.

Para ejecutarlo de forma independiente:

```powershell
python -m backendHelen.diagnostics --frames 50 --allow-missing
```

## Ejecución del backend empaquetado

1. Ejecuta `helen-backend.exe`. Por defecto levanta el servidor en
   `http://127.0.0.1:5000`.
2. Abre un navegador apuntando a esa dirección para utilizar el frontend
   empaquetado.
3. Usa `http://127.0.0.1:5000/health` para validar el estado general.

## Workflow de GitHub Actions

El archivo `windows-build.yml` define un workflow que:

1. Se ejecuta en un runner `windows-latest` tras cada `push` o `workflow_dispatch`.
2. Instala Python, las dependencias de `requirements-win.txt` y ejecuta
   PyInstaller.
3. Instala Inno Setup mediante Chocolatey y genera `Setup Helen.exe`.
4. Ejecuta el diagnóstico de cámara en modo tolerante (`--allow-missing`).
5. Publica los artefactos `dist/helen-backend` y `dist/Setup Helen.exe`.

Para lanzar el workflow manualmente entra en **Actions → Windows Build → Run
workflow**.

## Próximos pasos sugeridos

- Firmar digitalmente el instalador antes de su distribución.
- Configurar variables de entorno sensibles (por ejemplo URLs de backend) antes
  de compilar en entornos CI.
