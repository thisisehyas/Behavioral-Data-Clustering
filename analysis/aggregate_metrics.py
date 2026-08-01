import json
from pathlib import Path

import numpy as np

METRIC_SECTIONS = (
    "clustering_metrics",
    "entropy_metrics",
    "optimization_metrics",
)


def parse_parameter(directory_name: str, prefix: str) -> float:
    """
    Convert directory names such as:
        alpha_0.5 -> 0.5
        lr_1e-03 -> 0.001
    """
    return float(directory_name.replace(prefix, ""))


def aggregate_run_metrics(
    runs_root: Path,
    output_path: Path,
    prefix: str,
):
    """
    Aggregate run_metrics.json files across all seeds for each
    hyperparameter value.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_by_parameter = {}

    parameter_dirs = sorted(
        [
            d for d in runs_root.iterdir()
            if d.is_dir()
        ]
    )

    for parameter_dir in parameter_dirs:

        parameter_value = parse_parameter(
            parameter_dir.name,
            prefix,
        )

        print(f"Aggregating {parameter_dir.name}")

        metrics_by_parameter[parameter_value] = {}

        seed_dirs = sorted(
            [
                d for d in parameter_dir.iterdir()
                if d.is_dir()
            ]
        )

        for seed_dir in seed_dirs:

            metrics_path = seed_dir / "run_metrics.json"

            if not metrics_path.exists():
                continue

            with open(metrics_path, "r") as f:
                run_metrics = json.load(f)

            for section in METRIC_SECTIONS:
                for metric_name, value in run_metrics[section].items():
                    metrics_by_parameter[parameter_value].setdefault(
                        metric_name,
                        [],
                    ).append(value)

    aggregated = {}

    for parameter_value, metrics in metrics_by_parameter.items():

        if not metrics:
            print(
                f"[WARN] No metrics found for parameter={parameter_value}, skipping"
            )
            continue

        aggregated[parameter_value] = {
            "num_seeds": len(next(iter(metrics.values())))
        }

        for metric_name, values in metrics.items():

            values = np.array(values)

            aggregated[parameter_value][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nSaved aggregated metrics to: {output_path}")

    return aggregated