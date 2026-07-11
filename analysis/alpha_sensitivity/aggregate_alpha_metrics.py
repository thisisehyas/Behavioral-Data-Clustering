import json
from pathlib import Path
import numpy as np

RUNS_ROOT = Path("experiments/alpha_sensitivity/runs")
OUTPUT_DIR = Path("analysis/alpha_sensitivity/aggregated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_SECTIONS = (
            "clustering_metrics", 
            "entropy_metrics",
            "optimization_metrics",
        )

def parse_alpha(alpha_dir_name: str) -> float:
    """
    Convert 'alpha_0.5' -> 0.5
    """
    return float(alpha_dir_name.replace("alpha_", ""))

metrics_by_alpha = {}

for alpha_dir in sorted(RUNS_ROOT.iterdir()):
    if not alpha_dir.is_dir():
        continue
    
    alpha = parse_alpha(alpha_dir.name)

    print(f"Aggregating alpha={alpha}")

    metrics_by_alpha[alpha] = {}
    
    for seed_dir in sorted(alpha_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        
        metrics_path = seed_dir / "run_metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path, "r") as f:
            run_metrics = json.load(f)

        for section in METRIC_SECTIONS:
            for metric_name, value in run_metrics[section].items():
                metrics_by_alpha[alpha].setdefault(metric_name, []).append(value)
            
aggregated = {}

for alpha, metrics in metrics_by_alpha.items():
    if not metrics:
        print(f"[WARN] No metrics found for alpha={alpha}, skipping")
        continue
    
    aggregated[alpha] = {
        "num_seeds": len(next(iter(metrics.values())))
    }

    for metric_name, values in metrics.items():
        values = np.array(values)
        aggregated[alpha][metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

output_path = OUTPUT_DIR / "aggregated_alpha_metrics.json"

with open(output_path, "w") as f:
    json.dump(aggregated, f, indent=2)

print(f"\nSaved aggregated metrics to: {output_path}")
