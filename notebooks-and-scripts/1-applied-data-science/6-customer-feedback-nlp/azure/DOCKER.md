# Docker execution

The Docker configuration runs two services:

- `function`: the Azure Functions Python application.
- `azurite`: the local Azure Storage emulator.

The Docker build context is the project root because the trained model artifacts
are shared by the `python` and `azure` directories.

## Prerequisite

Install and start Docker Desktop.

Verify the installation:

```powershell
docker --version
docker compose version
```

## Start the services

From the `azure` directory:

```powershell
.\scripts\docker_up.ps1
```

The initial build downloads the Azure Functions Python image and installs the
pinned Python dependencies.

## Test the API

In another terminal:

```powershell
.\scripts\health_check.ps1
.\scripts\invoke_local.ps1
```

The endpoints remain:

```text
GET  http://localhost:7071/api/health
POST http://localhost:7071/api/predict
```

## Stop the services

Press `Ctrl+C` in the foreground Compose terminal, then run:

```powershell
.\scripts\docker_down.ps1
```

## View logs

```powershell
.\scripts\docker_logs.ps1
```

## Reset Azurite data

To remove the persisted local storage volume:

```powershell
docker compose down --volumes
```

The Azurite account and key used by Compose are development-only values and are
not cloud credentials.
