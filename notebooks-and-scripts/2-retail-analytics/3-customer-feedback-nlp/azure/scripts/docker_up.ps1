$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"
Write-Host " Starting Docker services"
Write-Host "========================================"
Write-Host ""

$projectRoot = (Resolve-Path "$PSScriptRoot\..").Path

Push-Location $projectRoot

try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker was not found in PATH. Install and start Docker Desktop."
    }

    docker compose up --build
}
finally {
    Pop-Location
}
