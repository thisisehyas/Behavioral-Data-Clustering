from pathlib import Path

from analysis.compute_ari import compute_ari


def main():

    compute_ari(
        runs_root=Path("experiments/alpha_sensitivity/runs"),
        output_dir=Path(
            "analysis/alpha_sensitivity/aggregated"
        ),
        prefix="alpha_",
        parameter_name="alpha",
    )


if __name__ == "__main__":
    main()