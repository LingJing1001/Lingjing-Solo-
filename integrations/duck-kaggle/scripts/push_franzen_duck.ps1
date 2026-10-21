# Pack Franzen overlay + rebuild notebook + push dataset & Duck kernel.
# Does NOT competition-submit (quota). Usage:
#   .\scripts\push_franzen_duck.ps1
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $PSScriptRoot
$Starter = Join-Path (Split-Path -Parent $Here) "ARC-AGI-3-Kaggle-Starter"
$py = Join-Path $Starter ".venv\Scripts\python.exe"
$kaggle = Join-Path $Starter ".venv\Scripts\kaggle.exe"
$tokenPath = Join-Path $Starter ".kaggle\access_token"
$env:KAGGLE_API_TOKEN = (Get-Content $tokenPath -Raw).Trim()
$env:HTTP_PROXY = ""; $env:HTTPS_PROXY = ""; $env:http_proxy = ""; $env:https_proxy = ""

Write-Host "==> pack overlay"
& $py (Join-Path $Here "scripts\pack_franzen_overlay.py")

$overlay = Join-Path $Here "overlay_dataset"
Write-Host "==> version overlay dataset (dir-mode zip)"
cmd /c "`"$kaggle`" datasets version -p `"$overlay`" -m `"Franzen Duck overlay`" --dir-mode zip"
if ($LASTEXITCODE -ne 0) {
    Write-Host "version failed; trying create..."
    cmd /c "`"$kaggle`" datasets create -p `"$overlay`" --dir-mode zip"
}

Write-Host "==> rebuild notebook"
& $py (Join-Path $Here "kaggle_push\patch_cp313.py")

# Sync patch into duck-harness copy for consistency
Copy-Item (Join-Path $Here "kaggle_push\patch_cp313.py") (Join-Path (Split-Path -Parent $Here) "duck-harness\kaggle_push\patch_cp313.py") -Force

Write-Host "==> push Duck kernel"
& $kaggle kernels push -p (Join-Path $Here "kaggle_push") 2>&1 | Out-String
Write-Host "Track: https://www.kaggle.com/code/liaojiaxuan/lingjing-duck-harness-v1"
