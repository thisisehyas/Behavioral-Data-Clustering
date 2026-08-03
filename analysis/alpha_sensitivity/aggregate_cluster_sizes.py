from pathlib import Path

from analysis.aggregate_cluster_sizes import aggregate_cluster_sizes


def main():
    aggregate_cluster_sizes(
        runs_root=Path("experiments/alpha_sensitivity/runs"),
        output_path=Path(
            "analysis/alpha_sensitivity/aggregated/aggregated_cluster_sizes.json"
        ),
        prefix="alpha_",
    )


if __name__ == "__main__":
    main()