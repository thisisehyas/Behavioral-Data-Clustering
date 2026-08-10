from pathlib import Path

from analysis.aggregate_metrics import aggregate_run_metrics


def main():
    aggregate_run_metrics(
        runs_root=Path("experiments/update_interval/runs"),
        output_path=Path(
            "analysis/update_interval/aggregated/"
            "aggregated_update_interval_metrics.json"
        ),
        prefix="update_interval_",
    )


if __name__ == "__main__":
    main()