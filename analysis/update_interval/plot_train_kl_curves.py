from pathlib import Path

from analysis.plot_train_kl_curves import plot_train_kl_curves


def main():
    plot_train_kl_curves(
        runs_root=Path("experiments/update_interval/runs"),
        figures_dir=Path("analysis/update_interval/figures"),
        prefix="update_interval_",
        parameter_name="Update Interval",
    )


if __name__ == "__main__":
    main()