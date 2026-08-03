from pathlib import Path

from analysis.plot_train_kl_curves import plot_train_kl_curves


def main():
    plot_train_kl_curves(
        runs_root=Path("experiments/alpha_sensitivity/runs"),
        figures_dir=Path("analysis/alpha_sensitivity/figures"),
        prefix="alpha_",
        parameter_name="alpha",
    )


if __name__ == "__main__":
    main()