from pathlib import Path

from analysis.plot_train_kl_curves import plot_train_kl_curves


def main():
    plot_train_kl_curves(
        runs_root=Path("experiments/lr-tuning/runs"),
        figures_dir=Path("analysis/learning_rate/figures"),
        prefix="lr_",
        parameter_name="learning_rate",
    )


if __name__ == "__main__":
    main()