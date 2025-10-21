Param(
    [string]$PythonVersion = "3.10",
    [string]$VenvPath = "build-venv",
    [switch]$SkipInstaller,
    [switch]$AllowMissingCamera
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
Write-Host "==> Proyecto" $projectRoot

$usePyLauncher = $false
if (Get-Command py.exe -ErrorAction SilentlyContinue) {
    try {
        & py.exe -$PythonVersion -c "import sys" | Out-Null
        $usePyLauncher = $true
    } catch {
        Write-Warning "No se encontró Python $PythonVersion con py.exe. Se usará 'python'."
    }
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "==> Creando entorno virtual en" $VenvPath
    if ($usePyLauncher) {
        & py.exe -$PythonVersion -m venv $VenvPath
    } else {
        & python -m venv $VenvPath
    }
}

$venvPython = Join-Path $VenvPath "Scripts/python.exe"

Write-Host "==> Actualizando pip"
& $venvPython -m pip install --upgrade pip

Write-Host "==> Instalando dependencias de Windows"
& $venvPython -m pip install -r (Join-Path $scriptDir "requirements-win.txt")

Write-Host "==> Ejecutando PyInstaller"
$specFile = Join-Path $scriptDir "helen_backend.spec"
& $venvPython -m PyInstaller --clean --noconfirm $specFile

if (-not $SkipInstaller) {
    $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (-not (Test-Path $iscc)) {
        Write-Warning "Inno Setup no se encontró en $iscc. Instalalo o ejecuta con -SkipInstaller."
    } else {
        Write-Host "==> Generando instalador con Inno Setup"
        & $iscc (Join-Path $scriptDir "inno_setup.iss")
    }
}

Write-Host "==> Ejecutando diagnóstico de cámara"
$diagArgs = @()
if ($AllowMissingCamera) { $diagArgs += "--allow-missing" }
& $venvPython -m backendHelen.diagnostics @diagArgs

Write-Host "==> Artefactos generados en dist/"
