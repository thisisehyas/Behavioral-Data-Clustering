from pathlib import Path

from analysis.plot_kl_curves import plot_kl_curves


def main():

    plot_kl_curves(
        runs_root=Path("experiments/alpha_sensitivity/runs"),
        figures_dir=Path("analysis/alpha_sensitivity/figures"),
        parameter_prefix="alpha_",
        parameter_name="α",
        output_filename="kl_curves_by_alpha.png",
        show_train=False,
    )


if __name__ == "__main__":
    main()