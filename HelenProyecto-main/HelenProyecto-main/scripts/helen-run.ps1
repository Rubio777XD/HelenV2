param(
    [int]$Port = 5000,
    [int]$CameraIndex = 0,
    [string]$ExtraArgs = '',
    [switch]$SkipBrowser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-Python311 {
    $candidates = @(
        @{ Command = 'py'; Args = @('-3.11'); Description = 'Python Launcher 3.11' },
        @{ Command = 'python'; Args = @(); Description = 'python' },
        @{ Command = 'python3'; Args = @(); Description = 'python3' },
        @{ Command = 'C:\\Python311\\python.exe'; Args = @(); Description = 'Python 3.11 (C:\\Python311)' }
    )

    foreach ($candidate in $candidates) {
        try {
            $resolved = Get-Command $candidate.Command -ErrorAction Stop
        } catch {
            continue
        }

        try {
            $output = & $resolved.Source @($candidate.Args + @('-c', "import sys; print(sys.executable); print(sys.version_info.major); print(sys.version_info.minor)"))
        } catch {
            continue
        }

        $lines = $output -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
        if ($lines.Count -lt 3) {
            continue
        }

        $major = [int]$lines[1]
        $minor = [int]$lines[2]
        if ($major -eq 3 -and $minor -eq 11) {
            return [PSCustomObject]@{
                Executable = $lines[0]
                Description = $candidate.Description
            }
        }
    }

    throw "No se encontró Python 3.11. Instala Python 3.11 y agrégalo al PATH."
}

$pythonInfo = Get-Python311
$pythonExe = $pythonInfo.Executable
Write-Host "[HELEN] Usando Python 3.11 en $pythonExe ($($pythonInfo.Description))"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\\python.exe"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
if (-not (Test-Path $RequirementsPath)) {
    throw "No se encontró requirements.txt en $ProjectRoot"
}

if (Test-Path $VenvPython) {
    try {
        $venvVersion = (& $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    } catch {
        $venvVersion = ''
    }
    if ($venvVersion -ne '3.11') {
        Write-Warning "[HELEN] El entorno virtual actual usa Python $venvVersion. Se recreará con Python 3.11."
        Remove-Item -Recurse -Force $VenvDir
    }
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "[HELEN] Creando entorno virtual en $VenvDir"
    & $pythonExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo crear el entorno virtual en $VenvDir"
    }
}

Write-Host "[HELEN] Actualizando pip"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo actualizar pip"
}

Write-Host "[HELEN] Instalando dependencias desde requirements.txt"
& $VenvPython -m pip install -r $RequirementsPath
if ($LASTEXITCODE -ne 0) {
    throw "No se pudieron instalar las dependencias"
}

$defaultArgs = "--camera-backend directshow --camera-width 1280 --camera-height 720 --frame-stride 2 --poll-interval 0.08"
$extras = if ([string]::IsNullOrWhiteSpace($ExtraArgs)) { '' } else { " $ExtraArgs" }
$env:HELEN_CAMERA_INDEX = $CameraIndex
$env:HELEN_BACKEND_EXTRA_ARGS = "$defaultArgs$extras".Trim()

Write-Host "[HELEN] HELEN_CAMERA_INDEX=$CameraIndex"
Write-Host "[HELEN] HELEN_BACKEND_EXTRA_ARGS=$($env:HELEN_BACKEND_EXTRA_ARGS)"

$runScript = Join-Path $ScriptDir "run-windows.ps1"
if (-not (Test-Path $runScript)) {
    throw "No se encontró scripts\\run-windows.ps1"
}

$runParams = @{ Port = $Port }
if ($SkipBrowser) {
    $runParams['SkipBrowser'] = $true
}

& $runScript @runParams
