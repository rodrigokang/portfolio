Write-Host ""

Write-Host "========================================"
Write-Host " Setting up local environment"
Write-Host "========================================"
Write-Host ""

$projectRoot = Resolve-Path "$PSScriptRoot\.."
Push-Location $projectRoot
python3.11 -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
Write-Host ""
Write-Host "Environment created successfully."
Write-Host ""
Pop-Location
