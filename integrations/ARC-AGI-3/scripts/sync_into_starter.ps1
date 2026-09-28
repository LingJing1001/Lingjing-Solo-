# Sync the latest ARC submission agent and the canonical Lingjing package.
$ErrorActionPreference = "Stop"
$arc = Split-Path -Parent $PSScriptRoot
$integrations = Split-Path -Parent $arc
$workspace = Split-Path -Parent $integrations
$canonical = Join-Path $workspace "lingjing_solo"
$starter = Join-Path $integrations "ARC-AGI-3-Kaggle-Starter"
if (-not (Test-Path -LiteralPath $canonical)) { throw "Canonical package not found: $canonical" }
if (-not (Test-Path -LiteralPath $starter)) { throw "Starter not found: $starter" }

Copy-Item -LiteralPath (Join-Path $arc "agent\my_agent.py") -Destination (Join-Path $starter "agent\my_agent.py") -Force
robocopy $canonical (Join-Path $starter "lingjing_solo") /E /XD __pycache__ .pytest_cache .ruff_cache /XF *.pyc /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
Write-Output "Synced canonical package -> $starter"
Write-Output "Next: python $workspace\scripts\build_arc_notebook.py"
