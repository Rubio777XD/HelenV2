@echo off
REM Inicia HELEN con el backend LSTM de video en Windows
REM Ejecuta este script desde la raíz del proyecto

setlocal
title HELEN - backend LSTM

if not exist .\.venv\Scripts\activate.bat (
    echo No se encontró el entorno virtual en .\.venv. Crea uno con "python -m venv .venv".
    exit /b 1
)

call .\.venv\Scripts\activate.bat
set HELEN_MODEL_BACKEND=lstm
python -m backendHelen.server
