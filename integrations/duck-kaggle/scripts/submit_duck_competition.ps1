# Competition-submit Duck kernel after COMPLETE.
# Usage:
#   .\scripts\submit_duck_competition.ps1 -Version 10
#   .\scripts\submit_duck_competition.ps1 -Version 10 -Message "Duck Franzen+Qwen3.8"
param(
    [Parameter(Mandatory = $true)][int]$Version,
    [string]$Message = "Duck Franzen overlay + Qwen3.8-27B-FP8 + UNDO/10x/guards"
)
$ErrorActionPreference = "Stop"
$Integrations = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Starter = Join-Path $Integrations "ARC-AGI-3-Kaggle-Starter"
$kaggle = Join-Path $Starter ".venv\Scripts\kaggle.exe"
$tokenPath = Join-Path $Starter ".kaggle\access_token"
$env:KAGGLE_API_TOKEN = (Get-Content $tokenPath -Raw).Trim()
$env:HTTP_PROXY = ""; $env:HTTPS_PROXY = ""; $env:http_proxy = ""; $env:https_proxy = ""

$kernel = "liaojiaxuan/lingjing-duck-harness-v1"
$status = (cmd /c "`"$kaggle`" kernels status $kernel 2>&1") | Out-String
Write-Host $status
if ($status -notmatch "COMPLETE") {
    throw "Duck kernel not COMPLETE yet — wait, then re-run with -Version $Version"
}

cmd /c "`"$kaggle`" competitions submit arc-prize-2026-arc-agi-3 -k $kernel -v $Version -f submission.parquet -m `"$Message`""
if ($LASTEXITCODE -ne 0) { throw "competition submit failed" }
Write-Host "OK submitted Duck v$Version"
cmd /c "`"$kaggle`" competitions submissions -c arc-prize-2026-arc-agi-3 -q" | Select-Object -First 5
