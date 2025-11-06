param(
    [int]$Port = 5000,
    [switch]$SkipBrowser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Split-ArgumentString {
    param(
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return @()
    }

    $builder = New-Object System.Text.StringBuilder
    $arguments = New-Object System.Collections.Generic.List[string]

    $inSingleQuote = $false
    $inDoubleQuote = $false
    $isEscaping = $false

    foreach ($char in $Value.ToCharArray()) {
        if ($isEscaping) {
            [void]$builder.Append($char)
            $isEscaping = $false
            continue
        }

        if ($char -eq '"' -and -not $inSingleQuote) {
            $inDoubleQuote = -not $inDoubleQuote
            continue
        }

        if ($char -eq "'" -and -not $inDoubleQuote) {
            $inSingleQuote = -not $inSingleQuote
            continue
        }

        if ($char -eq '`' -and -not $inSingleQuote) {
            $isEscaping = $true
            continue
        }

        if (-not $inSingleQuote -and -not $inDoubleQuote -and [char]::IsWhiteSpace($char)) {
            if ($builder.Length -gt 0) {
                $arguments.Add($builder.ToString())
                [void]$builder.Clear()
            }
            continue
        }

        [void]$builder.Append($char)
    }

    if ($builder.Length -gt 0) {
        $arguments.Add($builder.ToString())
    }

    return $arguments.ToArray()
}

function Join-Arguments {
    param(
        [string[]]$Arguments
    )

    if (-not $Arguments -or $Arguments.Count -eq 0) {
        return ''
    }

    $pieces = foreach ($argument in $Arguments) {
        if ($null -eq $argument) {
            continue
        }

        $value = [string]$argument
        if ($value -eq '') {
            '""'
            continue
        }

        if ($value -match '[\s"`]') {
            '"' + $value.Replace('"', '`"') + '"'
        } else {
            $value
        }
    }

    return ($pieces -join ' ')
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

$parsedExtraArgs = Split-ArgumentString -Value $env:HELEN_BACKEND_EXTRA_ARGS
$effectiveArgsList = New-Object System.Collections.Generic.List[string]
if ($parsedExtraArgs.Count -gt 0) {
    $effectiveArgsList.AddRange($parsedExtraArgs)
}

function Ensure-ArgumentPair {
    param(
        [System.Collections.Generic.List[string]]$List,
        [string]$Name,
        [string]$DefaultValue
    )

    for ($i = 0; $i -lt $List.Count; $i++) {
        $current = $List[$i]
        if ($current -eq $Name -or ($current -like "$Name=*")) {
            return
        }
    }

    $List.InsertRange(0, @($Name, $DefaultValue))
}

Ensure-ArgumentPair -List $effectiveArgsList -Name '--poll-interval' -DefaultValue '0.08'
Ensure-ArgumentPair -List $effectiveArgsList -Name '--frame-stride' -DefaultValue '2'
Ensure-ArgumentPair -List $effectiveArgsList -Name '--camera-height' -DefaultValue '720'
Ensure-ArgumentPair -List $effectiveArgsList -Name '--camera-width' -DefaultValue '1280'
Ensure-ArgumentPair -List $effectiveArgsList -Name '--camera-backend' -DefaultValue 'directshow'

$effectiveArgsArray = $effectiveArgsList.ToArray()

$env:HELEN_CAMERA_INDEX = $cameraIndex
$env:HELEN_BACKEND_EXTRA_ARGS = Join-Arguments -Arguments $effectiveArgsArray

$backendArgs = @('-m', 'backendHelen.server', '--host', '0.0.0.0', '--port', $Port, '--camera-index', $cameraIndex)
$backendArgs += $effectiveArgsArray

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
