from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CONDITIONS = ["frozen", "trainable"]


def load_kl_history(
    runs_root: Path,
    conditions: list[str],
) -> dict[str, np.ndarray]:
    """
    Load training KL-loss history for each ablation condition.
    """

    kl_by_condition = {}

    for condition in conditions:

        run_dir = runs_root / condition
        kl_path = run_dir / "train_kl.npy"

        if not kl_path.exists():
            raise FileNotFoundError(
                f"Missing train_kl.npy for condition "
                f"'{condition}': {kl_path}"
            )

        kl = np.load(kl_path)

        if kl.ndim != 1:
            raise ValueError(
                f"KL history for condition '{condition}' must be "
                f"1-dimensional, but got shape {kl.shape}."
            )

        if len(kl) == 0:
            raise ValueError(
                f"KL history for condition '{condition}' is empty."
            )

        if not np.all(np.isfinite(kl)):
            raise ValueError(
                f"KL history for condition '{condition}' "
                "contains non-finite values."
            )

        kl_by_condition[condition] = kl

        print(
            f"Loaded {condition}: "
            f"{len(kl)} epochs"
        )

    return kl_by_condition


def validate_kl_histories(
    kl_by_condition: dict[str, np.ndarray],
) -> None:
    """
    Verify that all conditions contain the same number of epochs.
    """

    lengths = {
        condition: len(kl)
        for condition, kl in kl_by_condition.items()
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Inconsistent number of epochs across conditions: "
            f"{lengths}"
        )

    n_epochs = next(iter(lengths.values()))

    print(
        f"\nBoth conditions contain {n_epochs} epochs."
    )


def create_kl_dataframe(
    kl_by_condition: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Create a long-format table containing epoch-wise KL loss.
    """

    n_epochs = len(
        next(iter(kl_by_condition.values()))
    )

    rows = []

    for condition, kl in kl_by_condition.items():
        for epoch, value in enumerate(kl, start=1):
            rows.append(
                {
                    "epoch": epoch,
                    "condition": condition,
                    "train_kl": float(value),
                }
            )

    return pd.DataFrame(rows)


def plot_kl_curves(
    kl_by_condition: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """
    Plot training KL loss across epochs for both conditions.
    """

    plt.figure(figsize=(7, 5))

    for condition in CONDITIONS:

        kl = kl_by_condition[condition]

        epochs = np.arange(
            1,
            len(kl) + 1,
        )

        plt.plot(
            epochs,
            kl,
            label=condition.capitalize(),
            linewidth=2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training KL loss")
    plt.title(
        "DEC Training Behavior: Frozen vs Trainable Encoder"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved KL curves to: {output_path}"
    )


def print_results(
    kl_by_condition: dict[str, np.ndarray],
) -> None:
    """
    Print only the basic training-dynamics statistics needed
    to inspect the two optimization trajectories.
    """

    print("\n" + "=" * 60)
    print("TRAINING BEHAVIOR — KL LOSS")
    print("=" * 60)

    for condition in CONDITIONS:

        kl = kl_by_condition[condition]

        print(
            f"\n{condition.capitalize()}:"
        )

        print(
            f"  Initial KL : {kl[0]:.6f}"
        )

        print(
            f"  Final KL   : {kl[-1]:.6f}"
        )

        print(
            f"  Minimum KL : {np.min(kl):.6f}"
        )

        print(
            f"  Epoch of minimum : "
            f"{np.argmin(kl) + 1}"
        )


def main():

    # Project paths
    PROJECT_ROOT = (
        Path(__file__).resolve().parents[2]
    )

    RUNS_ROOT = (
        PROJECT_ROOT
        / "experiments"
        / "encoder_ablation"
        / "runs"
    )

    ANALYSIS_ROOT = (
        PROJECT_ROOT
        / "analysis"
        / "encoder_ablation"
    )

    AGGREGATED_DIR = (
        ANALYSIS_ROOT / "aggregated"
    )

    FIGURES_DIR = (
        ANALYSIS_ROOT / "figures"
    )

    # Prepare output directories
    AGGREGATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("DEC ENCODER ABLATION — TRAINING BEHAVIOR")
    print("=" * 60)

    print(
        f"\nRuns root: {RUNS_ROOT}"
    )

    print(
        f"Output directory: {ANALYSIS_ROOT}"
    )

    print(
        f"Conditions: {CONDITIONS}"
    )

    # Load KL histories
    kl_by_condition = load_kl_history(
        runs_root=RUNS_ROOT,
        conditions=CONDITIONS,
    )

    # Validate
    validate_kl_histories(
        kl_by_condition
    )

    # Print basic results
    print_results(
        kl_by_condition
    )

    # Create and save epoch-wise table
    kl_df = create_kl_dataframe(
        kl_by_condition
    )

    csv_path = (
        AGGREGATED_DIR
        / "training_kl_comparison.csv"
    )

    kl_df.to_csv(
        csv_path,
        index=False,
    )

    print(
        f"\nSaved KL comparison to: {csv_path}"
    )

    # Plot
    plot_kl_curves(
        kl_by_condition=kl_by_condition,
        output_path=(
            FIGURES_DIR
            / "training_kl_curves.png"
        ),
    )

    print("\n" + "=" * 60)
    print("TRAINING BEHAVIOR ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()