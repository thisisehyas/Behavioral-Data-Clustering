from pathlib import Path

from compute_run_metrics import (
    compute_metrics_for_run,
    save_run_metrics,
)

def compute_all_run_metrics(runs_root: Path, sweep_parameter: str):

    parameter_dirs = sorted(
        [
            d for d in runs_root.iterdir()
            if d.is_dir()
        ]
    )
    for parameter_dir in parameter_dirs:
        print(f"\nProcessing {parameter_dir.name}")
        seed_dirs = sorted(
            [
                d for d in parameter_dir.iterdir()
                if d.is_dir()
            ]
        )
        for seed_dir in seed_dirs:
            print(f"  Computing metrics for {seed_dir.name}")

            metrics = compute_metrics_for_run(run_dir=seed_dir, sweep_parameter=sweep_parameter,)
            save_run_metrics(seed_dir, metrics)
        
    print(f"Finished generating run_metrics.json for all runs.")
