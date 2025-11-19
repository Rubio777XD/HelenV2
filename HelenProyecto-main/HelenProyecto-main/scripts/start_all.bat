@echo off
setlocal
REM Lanza backend HELEN y abre Chrome en modo kiosco.
pushd %~dp0\..

if not exist .venv\Scripts\python.exe (
    echo [ERROR] No se encontro el entorno .venv. Ejecuta "python -m venv .venv".
    popd
    exit /b 1
)

set HELEN_MODEL_BACKEND=lstm
set HELEN_ACTIVATION_SIGNAL=Start
call .venv\Scripts\activate

start "HELEN Backend" cmd /c "python -m backendHelen.server --host 0.0.0.0 --port 3000"

REM Espera breve para que el backend levante antes de abrir el navegador.
ping -n 5 127.0.0.1 >nul

start "" chrome.exe --kiosk --app=http://localhost:3000 --disable-features=TranslateUI --disable-infobars --no-first-run --disable-notifications
popd
