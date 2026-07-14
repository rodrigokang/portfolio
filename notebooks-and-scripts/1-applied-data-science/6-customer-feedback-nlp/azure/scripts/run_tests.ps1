Write-Host ""
Write-Host "========================================"
Write-Host " Running unit tests"
Write-Host "========================================"
Write-Host ""
$projectRoot = Resolve-Path "$PSScriptRoot\.."
Push-Location $projectRoot
& ".\.venv\Scripts\Activate.ps1"
pytest -q
Pop-Location
