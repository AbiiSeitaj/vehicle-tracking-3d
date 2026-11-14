$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & py -3.14 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install CPython 3.14 with the Windows Python launcher, then retry.' }
}
& ./.venv/Scripts/python.exe -m pip install -r requirements-lock-win-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& ./.venv/Scripts/python.exe scripts/download_model.py
if ($LASTEXITCODE -ne 0) { throw 'Model preparation failed.' }
& ./.venv/Scripts/python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Verification failed.' }
Write-Host 'Environment ready. Run ./.venv/Scripts/python.exe run_demo.py --show --output outputs/walkthrough_preview.mp4'
