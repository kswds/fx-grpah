import os, json, random, math
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mean_ci(x, z: float = 1.96):
    """
    Return [mean, lo, hi] using normal approximation.
    This is a reporting summary for multi-seed experiments.
    """
    x = np.array([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(x) == 0:
        return [float("nan"), float("nan"), float("nan")]
    m = float(np.mean(x))
    if len(x) <= 1:
        return [m, float("nan"), float("nan")]
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return [m, m - z * se, m + z * se]

def save_results(obj: dict, output_dir: str = "results", filename: str = "results_all_compare_rawY.json"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"\n save completed: {path}")
