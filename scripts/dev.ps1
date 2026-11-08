# Development helper: run the Tauri dev shell (spawns the FastAPI sidecar).
# Usage:  .\scripts\dev.ps1

$ErrorActionPreference = "Stop"

Set-Location "src-tauri"
cargo tauri dev

Set-Location ".."
