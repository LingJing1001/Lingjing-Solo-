# Windows entry: run Duck Kaggle submit via WSL.
param(
    [switch]$DryRun = $true,
    [string]$RunName = "lingjing-duck-harness-v1"
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$dh = (Resolve-Path (Join-Path $here "..")).Path
# Convert F:\... to /mnt/f/...
$drive = $dh.Substring(0, 1).ToLower()
$unix = "/mnt/$drive/" + ($dh.Substring(3) -replace "\\", "/")

$dry = if ($DryRun) { "true" } else { "false" }
$bash = @"
set -euo pipefail
cd '$unix'
python3 scripts/smoke_vendor.py
chmod +x scripts/submit_duck_kaggle.sh
export KAGGLE_DRY_RUN=$dry
export RUN_NAME='$RunName'
export KAGGLE_KERNEL_SLUG='$RunName'
bash scripts/submit_duck_kaggle.sh
"@

Write-Host "WSL path: $unix  DRY_RUN=$dry"
wsl -e bash -lc $bash
