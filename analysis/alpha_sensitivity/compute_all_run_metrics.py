from pathlib import Path

from compute_run_metrics import (
    compute_metrics_for_run,
    save_run_metrics,
)

def main():
    runs_root = Path("experiments/alpha_sensitivity/runs")
    alpha_dirs = sorted(
        [
            d for d in runs_root.iterdir()
            if d.is_dir()
        ]
    )
    for alpha_dir in alpha_dirs:
        print(f"\nProcessing {alpha_dir.name}")
        seed_dirs = sorted(
            [
                d for d in alpha_dir.iterdir()
                if d.is_dir()
            ]
        )
        for seed_dir in seed_dirs:
            print(f"  Computing metrics for {seed_dir.name}")

            metrics = compute_metrics_for_run(seed_dir)
            save_run_metrics(seed_dir, metrics)
        
    print(f"Finished generating run_metrics.json for all runs.")

if __name__ == "__main__":
        main()
