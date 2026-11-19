# Starts the API on this computer (http://localhost:8000). Run from the repo folder:
#   powershell -ExecutionPolicy Bypass -File api\start.ps1
# PTA_DB: the database folder (a local copy is faster and does not depend on Drive being online).
if (-not $env:PTA_DB) { $env:PTA_DB = "C:\pta\database" }
Set-Location (Split-Path $PSScriptRoot -Parent)
py -m uvicorn api.app:app --host 127.0.0.1 --port 8000
