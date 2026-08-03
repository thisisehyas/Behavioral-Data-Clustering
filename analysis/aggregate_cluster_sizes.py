import json
from pathlib import Path

import numpy as np

RANK_NAMES = [
    "smallest",
    "second_smallest",
    "middle",
    "second_largest",
    "largest",
]


def parse_parameter(directory_name: str, prefix: str) -> float:
    """
    Convert directory names such as:
        alpha_0.5 -> 0.5
        lr_1e-03 -> 0.001
    """
    return float(directory_name.replace(prefix, ""))


def discover_parameter_dirs(
    runs_root: Path,
    prefix: str,
):
    """
    Returns a dict mapping parameter value -> directory.

    Example:
        {
            "0.5": Path(...),
            "1.0": Path(...),
        }
    """

    parameter_dirs = {}

    for parameter_dir in sorted(runs_root.iterdir()):

        if not parameter_dir.is_dir():
            continue

        if not parameter_dir.name.startswith(prefix):
            continue

        parameter_str = parameter_dir.name.replace(prefix, "")

        try:
            parameter_value = str(float(parameter_str))
        except ValueError:
            continue

        parameter_dirs[parameter_value] = parameter_dir

    return parameter_dirs


def load_cluster_sizes_for_parameter(parameter_dir: Path):
    """
    Load cluster sizes from every seed directory belonging
    to one hyperparameter value.
    """

    cluster_sizes = {}

    seed_dirs = sorted(
        [
            d for d in parameter_dir.iterdir()
            if d.is_dir()
        ]
    )

    for seed_dir in seed_dirs:

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
            cluster_sizes.setdefault(
                RANK_NAMES[rank],
                [],
            ).append(size)

    return cluster_sizes


def aggregate_cluster_sizes(
    runs_root: Path,
    output_path: Path,
    prefix: str,
):
    """
    Aggregate cluster-size statistics across all seeds
    for each hyperparameter value.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameter_dirs = discover_parameter_dirs(
        runs_root,
        prefix,
    )

    cluster_sizes_by_parameter = {}

    for parameter_value, parameter_dir in parameter_dirs.items():

        cluster_sizes = load_cluster_sizes_for_parameter(
            parameter_dir
        )

        if not cluster_sizes:
            print(
                f"[WARN] No cluster sizes found for parameter={parameter_value}"
            )
            continue

        cluster_sizes_by_parameter[parameter_value] = cluster_sizes

    aggregated = {}

    for parameter_value, clusters in cluster_sizes_by_parameter.items():

        aggregated[parameter_value] = {
            "num_seeds": len(next(iter(clusters.values())))
        }

        for rank_name, sizes in clusters.items():

            sizes = np.array(sizes)

            aggregated[parameter_value][rank_name] = {
                "mean": float(np.mean(sizes)),
                "std": float(np.std(sizes)),
            }

    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nSaved aggregated cluster sizes to: {output_path}")

    return aggregated