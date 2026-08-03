from pathlib import Path

from analysis.plot_kl_curves import plot_kl_curves


def main():

    plot_kl_curves(
        runs_root=Path("experiments/lr-tuning/runs"),
        figures_dir=Path("analysis/learning_rate/figures"),
        parameter_prefix="lr_",
        parameter_name="LR",
        output_filename="kl_curves_by_lr.png",
        show_train=True,
    )


if __name__ == "__main__":
    main()