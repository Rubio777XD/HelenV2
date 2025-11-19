@echo off
REM Lanza el backend LSTM y el frontend en Chrome en ventanas separadas.
REM Útil para pruebas rápidas en Windows.

setlocal
cd /d "%~dp0.."

start "HELEN Backend" cmd /k "scripts\start-helen-windows-tf.bat"
start "HELEN Frontend" cmd /k "scripts\start-frontend-chrome-windows.bat"
endlocal
