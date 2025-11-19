@echo off
REM Inicia el backend de HELEN utilizando el clasificador LSTM de TensorFlow.
REM 1) Activa el entorno virtual local si existe.
REM 2) Define HELEN_MODEL_BACKEND=lstm para que el servidor elija el modelo de secuencias.
REM 3) Ejecuta el backend en la raíz del repositorio.

setlocal
cd /d "%~dp0.."

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

set HELEN_MODEL_BACKEND=lstm

python -m backendHelen.server %*
endlocal
