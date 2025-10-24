param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$LogDir = Join-Path $ProjectRoot "reports\logs\win"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogPath = Join-Path $LogDir "setup-$timestamp.log"
Start-Transcript -Path $LogPath -Force | Out-Null
Write-Host "[HELEN] Registrando instalación en $LogPath"

function Resolve-Python {
    param([string]$Override)
    if ($Override) {
        return $Override
    }
    $candidates = @("py -3.11", "py -3", "python3", "python")
    foreach ($cmd in $candidates) {
        try {
            $null = & $cmd --version 2>$null
            return $cmd
        } catch {
            continue
        }
    }
    throw "No se encontró un intérprete de Python 3.11. Instala Python 3.11 y vuelve a ejecutar."
}

$pythonCmd = Resolve-Python $Python
$pythonExe = & $pythonCmd -c "import sys; print(sys.executable)"
if (-not $pythonExe) {
    throw "No se pudo resolver la ruta del intérprete de Python."
}
Write-Host "[HELEN] Usando intérprete: $pythonExe"

$VenvDir = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "[HELEN] Creando entorno virtual en $VenvDir"
    & $pythonCmd -m venv $VenvDir
} else {
    Write-Host "[HELEN] Reutilizando entorno virtual existente en $VenvDir"
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "No se encontró $VenvPython. Verifica que Python 3.11 esté instalado."
}

Write-Host "[HELEN] Actualizando pip, wheel y setuptools"
& $VenvPython -m pip install --upgrade pip wheel setuptools

Write-Host "[HELEN] Instalando dependencias críticas (NumPy / OpenCV / MediaPipe)"
& $VenvPython -m pip install numpy==1.26.4 opencv-python==4.9.0.80 mediapipe==0.10.18

Write-Host "[HELEN] Instalando requisitos del proyecto"
$requirementsPath = Join-Path $ProjectRoot "requirements.txt"
& $VenvPython -m pip install -r $requirementsPath

try {
    & $VenvPython -m pip check
} catch {
    Write-Warning "[HELEN] pip check finalizó con advertencias: $_"
}

Stop-Transcript | Out-Null
Write-Host "[HELEN] Configuración completada. Revisa $LogPath para los detalles."
