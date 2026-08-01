from pathlib import Path

from analysis.compute_run_metrics import compute_all_run_metrics

def main():
    compute_all_run_metrics(
        runs_root=Path("experiments/lr-tuning/runs"),
        sweep_parameter="learning_rate",
    )

if __name__ == "__main__":
        main()