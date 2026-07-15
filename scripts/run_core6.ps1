$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $here "..")

& C:\Python\python.exe (Join-Path $projectRoot "experiments\run_model_comparison.py") `
  --models oursmain foundation_relational mlp lstm gru gnn `
  --universe core6 `
  --lookback 10 `
  --epochs 80 `
  --seeds 42 123 456 `
  --output-dir (Join-Path $projectRoot "results\core6_compare")

