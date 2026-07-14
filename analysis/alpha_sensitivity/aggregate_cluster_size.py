import json
from pathlib import Path

import numpy as np

RUNS_DIR = Path("experiments/alpha_sensitivity/runs")
OUTPUT_DIR = Path("analysis/alpha_sensitivity/aggregated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANK_NAMES = [
    "smallest",
    "second_smallest",
    "middle",
    "second_largest",
    "largest",
]

def discover_alpha_dirs(runs_dir: Path):
    """
    Returns a dict:
    {
        "0.1": Path(.../alpha_0.1),
        "0.5": Path(.../alpha_0.5),
    }
    """
    alpha_dirs = {}

    for alpha_dir in sorted(runs_dir.iterdir()):
        if not alpha_dir.is_dir():
            continue

        if not alpha_dir.name.startswith("alpha_"):
            continue

        alpha_str = alpha_dir.name.replace("alpha_", "")
        try:
            alpha_value = str(float(alpha_str))
        except ValueError:
            continue

        alpha_dirs[alpha_value] = alpha_dir

    return alpha_dirs

def load_cluster_sizes_for_alpha(alpha_dir: Path):
    """
    Loads cluster sizes from all seed_* subfolders of an alpha directory

    Returns:
        dict[str, list[int]]
        {
            "smallest": [...],
            "second_smallest": [...],
            "middle": [...],
            "second_largest": [...],
            "largest": [...]
        }
    """
    cluster_sizes = {}

    for seed_dir in sorted(alpha_dir.iterdir()):
        if not seed_dir.is_dir():
            continue

        if not seed_dir.name.startswith("seed_"):
            continue

        metrics_path = seed_dir / "run_metrics.json"

        if not metrics_path.exists():
            print(f"Missing run_metrics.json in {seed_dir}")
            continue

        with open(metrics_path, "r") as f:
            run_metrics = json.load(f)

        sizes = run_metrics["cluster_size_metrics"]["cluster_sizes"]

        sorted_sizes = sorted(sizes.values())

        for rank, size in enumerate(sorted_sizes):
            cluster_sizes.setdefault(RANK_NAMES[rank], []).append(size)
        
    return cluster_sizes

alpha_dirs = discover_alpha_dirs(RUNS_DIR)

cluster_sizes_by_alpha = {}

for alpha, alpha_dir in alpha_dirs.items():
    cluster_sizes = load_cluster_sizes_for_alpha(alpha_dir)

    if not cluster_sizes:
        print(f"[WARN] No cluster sizes found for alpha={alpha}")
        continue

    cluster_sizes_by_alpha[alpha] = cluster_sizes


aggregated = {}

for alpha, clusters in cluster_sizes_by_alpha.items():

    aggregated[alpha] = {
        "num_seeds": len(next(iter(clusters.values())))
    }

    for rank_name, sizes in clusters.items():
        sizes = np.array(sizes)

        aggregated[alpha][rank_name] = {
            "mean": float(np.mean(sizes)),
            "std": float(np.std(sizes)),
        }


output_path = OUTPUT_DIR / "aggregated_cluster_sizes.json"

with open(output_path, "w") as f:
    json.dump(aggregated, f, indent=2)

print(f"\nSaved aggregated cluster sizes to: {output_path}")