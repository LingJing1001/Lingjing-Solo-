# Push notebook and submit to ARC-AGI-3.
# Usage:
#   .\scripts\submit.ps1
#   .\scripts\submit.ps1 -Message "ScriptBank ls20+ar25"
#   .\scripts\submit.ps1 -SkipSubmit   # only build+push notebook
#
# Submits agent/my_agent.py + lingjing_solo (including planning/data/*_scripts.json).
# Before submit: run  python scripts/verify_scripts.py

param(
    [string]$Message = "CEAX unknown v3.6 (no plugins) focus8~2.26",
    [switch]$SkipSubmit
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Kaggle = Join-Path $Root ".venv\Scripts\kaggle.exe"

# Prefer env var; never hardcode secrets in repo
$Token = [Environment]::GetEnvironmentVariable("KAGGLE_API_TOKEN", "User")
if (-not $Token) {
    $Token = $env:KAGGLE_API_TOKEN
}
if (-not $Token) {
    $TokenPath = Join-Path $Root ".kaggle\access_token"
    if (Test-Path $TokenPath) { $Token = Get-Content $TokenPath -Raw }
}
if (-not $Token) { throw "KAGGLE_API_TOKEN not set. Run scripts/setup_kaggle_mcp.ps1 first." }
$env:KAGGLE_API_TOKEN = $Token.Trim()

Write-Host "== skip ScriptBank verify (CEAX unknown submission; no plugins/canned) ==" -ForegroundColor Cyan

Write-Host "== build notebook (CEAX v3.6 agent + lingjing_solo) ==" -ForegroundColor Cyan
& $Py (Join-Path $Root "scripts\build_notebook.py")

$pushOut = & $Kaggle kernels push -p (Join-Path $Root "notebooks") 2>&1 | Out-String
Write-Host $pushOut
$version = if ($pushOut -match "Kernel version (\d+)") { $Matches[1] } else { $null }

$meta = Get-Content (Join-Path $Root "notebooks\kernel-metadata.json") -Raw | ConvertFrom-Json
$kernel = $meta.id
Write-Host "Waiting for kernel run: $kernel" -ForegroundColor Cyan
for ($i = 1; $i -le 60; $i++) {
    $status = (& $Kaggle kernels status $kernel 2>&1 | Out-String).Trim()
    Write-Host "[$i] $status"
    if ($status -match "COMPLETE|ERROR|CANCEL") { break }
    Start-Sleep -Seconds 20
}
if ($status -notmatch "COMPLETE") {
    throw "Kernel did not complete. Check Kaggle notebook page."
}

if ($SkipSubmit) {
    Write-Host "SkipSubmit set — notebook ready, competition submit skipped." -ForegroundColor Yellow
    exit 0
}

if (-not $version) {
    $version = Read-Host "Enter kernel version number to submit (see Kaggle Versions tab)"
}

$submitOut = & $Kaggle competitions submit arc-prize-2026-arc-agi-3 `
    -k $kernel -v $version -f submission.parquet -m $Message 2>&1 | Out-String
Write-Host $submitOut
if ($LASTEXITCODE -ne 0 -or $submitOut -match "Client Error|Bad Request|Forbidden|Error") {
    Write-Host "COMPETITION SUBMIT FAILED (kernel v$version was pushed, but NOT on the leaderboard)." -ForegroundColor Red
    Write-Host "Common causes: daily quota=0, another submission still PENDING, or API reject." -ForegroundColor Yellow
    exit 1
}
Write-Host "Submitted OK: kernel v$version — $Message" -ForegroundColor Green
