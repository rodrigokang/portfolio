$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path "$PSScriptRoot\..").Path

Push-Location $projectRoot

try {
    docker compose logs --follow
}
finally {
    Pop-Location
}
