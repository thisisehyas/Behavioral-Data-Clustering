from pathlib import Path

from analysis.plot_kl_curves import plot_kl_curves


def main():

    plot_kl_curves(
        runs_root=Path("experiments/update_interval/runs"),
        figures_dir=Path("analysis/update_interval/figures"),
        parameter_prefix="update_interval_",
        parameter_name="Update Interval",
        output_filename="kl_curves_by_update_interval.png",
        show_train=True,
    )


if __name__ == "__main__":
    main()