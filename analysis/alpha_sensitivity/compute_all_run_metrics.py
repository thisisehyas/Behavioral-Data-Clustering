from pathlib import Path

from analysis.compute_run_metrics import compute_all_run_metrics

def main():
    compute_all_run_metrics(
        runs_root=Path("experiments/alpha_sensitivity/runs"),
        sweep_parameter="alpha",
    )

if __name__ == "__main__":
        main()