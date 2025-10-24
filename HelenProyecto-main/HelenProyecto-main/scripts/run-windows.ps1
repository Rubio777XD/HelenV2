param(
    [int]$Port = 5000
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "No se encontró el entorno virtual (.venv). Ejecuta scripts\setup-windows.ps1 primero."
}

$LogDir = Join-Path $ProjectRoot "reports\logs\win"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$BackendLog = Join-Path $LogDir "backend-$timestamp.log"

Write-Host "[HELEN] Iniciando backend (log: $BackendLog)"
$backendArgs = @("-m", "backendHelen.server", "--host", "0.0.0.0", "--port", $Port)
$backend = Start-Process -FilePath $VenvPython -ArgumentList $backendArgs -RedirectStandardOutput $BackendLog -RedirectStandardError $BackendLog -PassThru -WorkingDirectory $ProjectRoot

$healthUrl = "http://127.0.0.1:$Port/health"
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if ($backend.HasExited) {
        Write-Warning "[HELEN] El proceso del backend terminó prematuramente. Revisa $BackendLog"
        break
    }
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "[HELEN] Backend listo. Abriendo http://localhost:$Port"
            Start-Process "http://localhost:$Port"
            break
        }
    } catch {
        continue
    }
    if ($i -eq 29) {
        Write-Warning "[HELEN] No se pudo verificar el endpoint /health. Continúa monitoreando el log."
    }
}

try {
    Wait-Process -Id $backend.Id
} finally {
    if (-not $backend.HasExited) {
        Stop-Process -Id $backend.Id -Force
    }
    Write-Host "[HELEN] Backend detenido. Log disponible en $BackendLog"
}
