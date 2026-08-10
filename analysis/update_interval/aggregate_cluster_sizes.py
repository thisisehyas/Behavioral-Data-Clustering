from pathlib import Path

from analysis.aggregate_cluster_sizes import aggregate_cluster_sizes


def main():
    aggregate_cluster_sizes(
        runs_root=Path("experiments/update_interval/runs"),
        output_path=Path(
            "analysis/update_interval/aggregated/aggregated_cluster_sizes.json"
        ),
        prefix="update_interval_",
    )


if __name__ == "__main__":
    main()