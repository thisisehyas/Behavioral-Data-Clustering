from pathlib import Path

from analysis.aggregate_metrics import aggregate_run_metrics

def main():
    aggregate_run_metrics(
        runs_root=Path("experiments/alpha_sensitivity/runs"),
        output_path=Path(
            "analysis/alpha_sensitivity/aggregated/aggregated_alpha_metrics.json"
        ),
        prefix="alpha_",
    )

if __name__ == "__main__":
    main()