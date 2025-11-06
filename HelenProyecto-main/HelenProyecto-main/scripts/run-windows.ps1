param(
    [int]$Port = 5000,
    [switch]$SkipBrowser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function ConvertTo-ArgumentList {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return @()
    }

    $parseErrors = $null
    $tokens = [System.Management.Automation.PSParser]::Tokenize($Value, [ref]$parseErrors)
    if ($parseErrors -and $parseErrors.Count -gt 0) {
        throw "No se pudieron analizar HELEN_BACKEND_EXTRA_ARGS: $($parseErrors[0].Message)"
    }

    $args = @()
    foreach ($token in $tokens) {
        if ($token.Type -eq 'String' -or $token.Type -eq 'CommandArgument') {
            $args += $token.Content
        }
    }
    return $args
}

function Open-HelenBrowser {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url
    )

    $candidates = @(
        @{ Name = 'Google Chrome'; Command = 'chrome'; Arguments = @('--new-window', $Url) },
        @{ Name = 'Google Chrome'; Command = Join-Path $env:ProgramFiles 'Google\\Chrome\\Application\\chrome.exe'; Arguments = @('--new-window', $Url) },
        @{ Name = 'Google Chrome'; Command = Join-Path ${env:ProgramFiles(x86)} 'Google\\Chrome\\Application\\chrome.exe'; Arguments = @('--new-window', $Url) },
        @{ Name = 'Microsoft Edge'; Command = 'msedge'; Arguments = @($Url) }
    )

    foreach ($candidate in $candidates) {
        if (-not $candidate.Command) {
            continue
        }
        $resolved = Get-Command $candidate.Command -ErrorAction SilentlyContinue
        if ($resolved) {
            Write-Host "[HELEN] Abriendo $($candidate.Name) en $Url"
            Start-Process -FilePath $resolved.Source -ArgumentList $candidate.Arguments | Out-Null
            return
        }
        if (Test-Path $candidate.Command) {
            Write-Host "[HELEN] Abriendo $($candidate.Name) en $Url"
            Start-Process -FilePath $candidate.Command -ArgumentList $candidate.Arguments | Out-Null
            return
        }
    }

    Write-Host "[HELEN] No se encontró Chrome/Edge. Abriendo navegador predeterminado en $Url"
    Start-Process $Url | Out-Null
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$VenvPython = Join-Path $ProjectRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "No se encontró el entorno virtual (.venv). Ejecuta scripts\\helen-run.ps1 o scripts\\setup-windows.ps1 primero."
}

$LogDir = Join-Path $ProjectRoot "reports\\logs\\win"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$BackendStdout = Join-Path $LogDir "backend-$timestamp.out.log"
$BackendStderr = Join-Path $LogDir "backend-$timestamp.err.log"

$originalCameraIndex = $env:HELEN_CAMERA_INDEX
$originalExtraArgs = $env:HELEN_BACKEND_EXTRA_ARGS

$cameraIndex = if ([string]::IsNullOrWhiteSpace($env:HELEN_CAMERA_INDEX)) { '0' } else { $env:HELEN_CAMERA_INDEX }
$defaultExtraArgsString = "--camera-backend directshow --camera-width 1280 --camera-height 720 --frame-stride 2 --poll-interval 0.08"
$combinedExtraArgs = if ([string]::IsNullOrWhiteSpace($env:HELEN_BACKEND_EXTRA_ARGS)) {
    $defaultExtraArgsString
} else {
    "$defaultExtraArgsString $($env:HELEN_BACKEND_EXTRA_ARGS)"
}

$env:HELEN_CAMERA_INDEX = $cameraIndex
$env:HELEN_BACKEND_EXTRA_ARGS = $combinedExtraArgs.Trim()

$backendArgs = @('-m', 'backendHelen.server', '--host', '0.0.0.0', '--port', $Port, '--camera-index', $cameraIndex)
$backendArgs += ConvertTo-ArgumentList -Value $env:HELEN_BACKEND_EXTRA_ARGS

Write-Host "[HELEN] Iniciando backend con Python en $VenvPython"
Write-Host "[HELEN] HELEN_CAMERA_INDEX=$cameraIndex"
Write-Host "[HELEN] HELEN_BACKEND_EXTRA_ARGS=$($env:HELEN_BACKEND_EXTRA_ARGS)"
Write-Host "[HELEN] Logs: stdout -> $BackendStdout | stderr -> $BackendStderr"

$backend = $null
$healthUrl = "http://127.0.0.1:$Port/health"
try {
    $backend = Start-Process -FilePath $VenvPython -ArgumentList $backendArgs -RedirectStandardOutput $BackendStdout -RedirectStandardError $BackendStderr -PassThru -WorkingDirectory $ProjectRoot

    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 1
        if ($backend.HasExited) {
            Write-Warning "[HELEN] El proceso del backend terminó prematuramente. Revisa $BackendStdout y $BackendStderr"
            break
        }
        try {
            $response = Invoke-WebRequest -Uri $healthUrl -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                Write-Host "[HELEN] Backend listo en http://localhost:$Port"
                if (-not $SkipBrowser) {
                    Open-HelenBrowser -Url "http://localhost:$Port"
                }
                break
            }
        } catch {
            continue
        }
        if ($i -eq 39) {
            Write-Warning "[HELEN] No se pudo verificar /health tras 40 segundos. Consulta $BackendStdout"
        }
    }

    if ($backend) {
        Wait-Process -Id $backend.Id
    }
} finally {
    if ($backend -and -not $backend.HasExited) {
        Stop-Process -Id $backend.Id -Force
    }

    if ($null -eq $originalCameraIndex) {
        Remove-Item Env:HELEN_CAMERA_INDEX -ErrorAction SilentlyContinue
    } else {
        $env:HELEN_CAMERA_INDEX = $originalCameraIndex
    }

    if ($null -eq $originalExtraArgs) {
        Remove-Item Env:HELEN_BACKEND_EXTRA_ARGS -ErrorAction SilentlyContinue
    } else {
        $env:HELEN_BACKEND_EXTRA_ARGS = $originalExtraArgs
    }

    Write-Host "[HELEN] Backend detenido. Revisa $BackendStdout y $BackendStderr para más detalles."
}
