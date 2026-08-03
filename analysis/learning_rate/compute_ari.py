from pathlib import Path

from analysis.compute_ari import compute_ari


def main():

    compute_ari(
        runs_root=Path("experiments/lr-tuning/runs"),
        output_dir=Path(
            "analysis/learning_rate/aggregated"
        ),
        prefix="lr_",
    )


if __name__ == "__main__":
    main()