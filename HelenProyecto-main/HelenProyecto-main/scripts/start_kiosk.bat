@echo off
setlocal
REM Abre Chrome/Chromium en modo kiosco apuntando a HELEN en localhost:3000.
set TARGET_URL=http://localhost:3000
set CHROME_BIN=chrome.exe

if not "%~1"=="" (
    set CHROME_BIN=%~1
)

start "" "%CHROME_BIN%" --kiosk --app=%TARGET_URL% --disable-features=TranslateUI --disable-infobars --no-first-run --disable-notifications
