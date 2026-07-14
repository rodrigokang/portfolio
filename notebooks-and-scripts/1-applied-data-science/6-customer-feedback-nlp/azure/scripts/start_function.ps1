$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"
Write-Host " Starting Azure Function"
Write-Host "========================================"
Write-Host ""

$projectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$venvRoot = Join-Path $projectRoot ".venv"
$pythonPath = Join-Path $venvRoot "Scripts\python.exe"
$activatePath = Join-Path $venvRoot "Scripts\Activate.ps1"
$sitePackages = Join-Path $venvRoot "Lib\site-packages"

if (-not (Test-Path $pythonPath)) {
    throw "Virtual environment not found. Run .\scripts\setup.ps1 first."
}

if (-not (Get-Command func -ErrorAction SilentlyContinue)) {
    throw "Azure Functions Core Tools was not found in PATH."
}

Push-Location $projectRoot

try {
    # Dot-source the activation script so its changes apply in this scope.
    . $activatePath

    # Force Core Tools to use the project virtual environment.
    $env:languageWorkers__python__defaultExecutablePath = $pythonPath

    # Ensure packages remain visible even if Core Tools selects
    # another Python 3.11 worker executable.
    $env:PYTHONPATH = $sitePackages

    # Put the virtual environment first in PATH.
    $env:PATH = "$(Join-Path $venvRoot 'Scripts');$env:PATH"

    Write-Host "Python executable:"
    Write-Host $env:languageWorkers__python__defaultExecutablePath
    Write-Host ""

    & $pythonPath -c "import sys, numpy; print('Runtime:', sys.executable); print('NumPy:', numpy.__version__)"

    Write-Host ""
    func start
}
finally {
    Pop-Location
}