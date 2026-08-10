from pathlib import Path

from analysis.compute_all_run_metrics import compute_all_run_metrics

def main():
    compute_all_run_metrics(
        runs_root=Path("experiments/update_interval/runs"),
        sweep_parameter="update_interval",
    )

if __name__ == "__main__":
        main()