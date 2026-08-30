from pathlib import Path

from analysis.aggregate_metrics import aggregate_run_metrics

def main():
    aggregate_run_metrics(
        runs_root=Path("experiments/learning_rate/runs"),
        output_path=Path(
            "analysis/learning_rate/aggregated/aggregated_lr_metrics.json"
        ),
        prefix="lr_",
    )

if __name__ == "__main__":
    main()