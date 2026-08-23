from pathlib import Path

from analysis.aggregate_cluster_sizes import aggregate_cluster_sizes


def main():
    aggregate_cluster_sizes(
        runs_root=Path("experiments/learning_rate/runs"),
        output_path=Path(
            "analysis/learning_rate/aggregated/aggregated_cluster_sizes.json"
        ),
        prefix="lr_",
    )


if __name__ == "__main__":
    main()