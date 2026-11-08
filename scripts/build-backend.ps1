# Build the Python FastAPI sidecar with PyInstaller and stage it for Tauri.
# Usage:  .\scripts\build-backend.ps1   (works from any directory)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "python-backend"

Push-Location $backend
try {
    # 1. Build the single-file exe from python-backend/server.py
    & (Join-Path $root "venv\Scripts\python.exe") -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --name primal-backend `
        --add-data "fonts;fonts" `
        --add-data "data;data" `
        --add-data "models;models" `
        --add-data "symbols;symbols" `
        "server.py"

    # 2. Copy the artifact where the Tauri bundler expects it
    Copy-Item -Force (Join-Path $backend "dist\primal-backend.exe") `
        (Join-Path $root "src-tauri\binaries\primal-backend-x86_64-pc-windows-msvc.exe")

    Write-Host "Backend built and staged into src-tauri/binaries/"
}
finally {
    Pop-Location
}
