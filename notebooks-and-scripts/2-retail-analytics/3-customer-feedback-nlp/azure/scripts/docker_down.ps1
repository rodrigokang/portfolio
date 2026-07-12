$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"
Write-Host " Stopping Docker services"
Write-Host "========================================"
Write-Host ""

$projectRoot = (Resolve-Path "$PSScriptRoot\..").Path

Push-Location $projectRoot

try {
    docker compose down
}
finally {
    Pop-Location
}
