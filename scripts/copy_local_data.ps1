param(
    [string]$SourceRoot = ".."
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $here "..")
$srcRoot = Resolve-Path (Join-Path $projectRoot $SourceRoot)
$srcData = Join-Path $srcRoot "data\processed"
$dstData = Join-Path $projectRoot "data\processed"

New-Item -ItemType Directory -Force -Path $dstData | Out-Null
Copy-Item -LiteralPath (Join-Path $srcData "factor_daily_alligned_krw.csv") -Destination $dstData -Force
Copy-Item -LiteralPath (Join-Path $srcData "score_vA_nonfx_features.csv") -Destination $dstData -Force

Write-Host "Copied data files to $dstData"

