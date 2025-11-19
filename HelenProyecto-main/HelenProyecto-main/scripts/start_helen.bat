@echo off
setlocal
REM Inicia el backend HELEN con TensorFlow LSTM en el puerto 3000.
pushd %~dp0\..

if not exist .venv\Scripts\python.exe (
    echo [ERROR] No se encontro el entorno .venv. Crea uno con "python -m venv .venv".
    popd
    exit /b 1
)

call .venv\Scripts\activate
set HELEN_MODEL_BACKEND=lstm
set HELEN_ACTIVATION_SIGNAL=Start

python -m backendHelen.server --host 0.0.0.0 --port 3000
popd
