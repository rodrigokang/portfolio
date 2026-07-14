$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"
Write-Host " Starting Azurite"
Write-Host "========================================"
Write-Host ""

$projectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$azuriteData = Join-Path $projectRoot ".azurite"

if (-not (Get-Command azurite -ErrorAction SilentlyContinue)) {
    throw "Azurite was not found. Install it with: npm install -g azurite"
}

New-Item `
    -ItemType Directory `
    -Path $azuriteData `
    -Force | Out-Null

azurite `
    --location $azuriteData `
    --debug (Join-Path $azuriteData "debug.log") `
    --silent