from pathlib import Path

from analysis.compute_ari import compute_ari


def main():

    compute_ari(
        runs_root=Path("experiments/update_interval/runs"),
        output_dir=Path(
            "analysis/update_interval/aggregated"
        ),
        prefix="update_interval_",
        parameter_name="update_interval"
    )


if __name__ == "__main__":
    main()