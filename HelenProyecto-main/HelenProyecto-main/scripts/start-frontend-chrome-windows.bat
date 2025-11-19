@echo off
REM Abre Google Chrome apuntando al frontend de HELEN.
REM Ajusta la ruta de Chrome si está instalada en otra ubicación.

setlocal
set HELEN_FRONTEND_URL=http://localhost:5000
set CHROME_PATH="%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not exist %CHROME_PATH% set CHROME_PATH="%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"

start "HELEN UI" %CHROME_PATH% --new-window --disable-infobars --start-maximized --disable-extensions %HELEN_FRONTEND_URL%
endlocal
