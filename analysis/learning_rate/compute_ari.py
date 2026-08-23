from pathlib import Path

from analysis.compute_ari import compute_ari


def main():

    compute_ari(
        runs_root=Path("experiments/learning_rate/runs"),
        output_dir=Path(
            "analysis/learning_rate/aggregated"
        ),
        prefix="lr_",
        parameter_name="lr"
    )


if __name__ == "__main__":
    main()