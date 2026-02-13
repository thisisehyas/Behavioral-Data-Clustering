import json
from pathlib import Path
import numpy as np

RUNS_ROOT = Path("experiments/lr-tuning/runs")
OUTPUT_DIR = Path("analysis/aggregated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_lr(lr_dir_name: str) -> float:
    """
    Convert 'lr_1e-02' -> 0.01
    """
    return float(lr_dir_name.replace("lr_", ""))

metrics_by_lr = {}

for lr_dir in sorted(RUNS_ROOT.iterdir()):
    if not lr_dir.is_dir():
        continue
    
    lr = parse_lr(lr_dir.name)
    metrics_by_lr[lr] = {}
    
    for seed_dir in lr_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        
        metrics_path = seed_dir / "run_metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path, "r") as f:
            run_metrics = json.load(f)

        sections = ["clustering_metrics", 
                    "entropy_metrics",
                    "optimization_metrics"
                ]
        for section in sections:
            for metric_name, value in run_metrics[section].items():
                metrics_by_lr[lr].setdefault(metric_name, []).append(value)
            
aggregated = {}

for lr, metrics in metrics_by_lr.items():
    if not metrics:
        print(f"[WARN] No metrics found for LR={lr}, skipping")
        continue
    
    aggregated[lr] = {
        "num_seeds": len(next(iter(metrics.values())))
    }

    for metric_name, values in metrics.items():
        values = np.array(values)
        aggregated[lr][metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

output_path = OUTPUT_DIR / "aggregated_lr_metrics.json"

with open(output_path, "w") as f:
    json.dump(aggregated, f, indent=2)
