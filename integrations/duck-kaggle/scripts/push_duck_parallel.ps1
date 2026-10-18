# Push the Duck/Qwen kernel only. Never touches the SmartRouter kernel.
# Usage (from integrations/duck-kaggle):
#   .\scripts\push_duck_parallel.ps1
# Does not competition-submit. Compare publicScore after the kernel COMPLETE.

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $PSScriptRoot
$MetaPath = Join-Path $Here "kaggle_push\kernel-metadata.json"
$meta = Get-Content $MetaPath -Raw | ConvertFrom-Json
if ($meta.id -match "smart-router") {
    throw "Refusing to push: kernel id looks like SmartRouter ($($meta.id))"
}
if ($meta.id -ne "liaojiaxuan/lingjing-duck-harness-v1") {
    throw "Unexpected Duck kernel id: $($meta.id)"
}
if ($meta.enable_internet -ne $false) {
    throw "Duck kernel enable_internet must be false"
}
Write-Host "DUCK_PARALLEL_OK id=$($meta.id) internet=false shape=$($meta.machine_shape)"
Write-Host "This does not replace SmartRouter. Push manually when GitHub/Kaggle is reachable:"
Write-Host "  kaggle kernels push -p `"$Here\kaggle_push`""
