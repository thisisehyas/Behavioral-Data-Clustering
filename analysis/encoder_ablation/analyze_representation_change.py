from pathlib import Path

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_embeddings(
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the initial pretrained representation and the final
    trainable-encoder representation.
    """

    initial_path = run_dir / "Z_initial.npy"
    final_path = run_dir / "Z_full.npy"

    if not initial_path.exists():
        raise FileNotFoundError(
            f"Missing Z_initial.npy: {initial_path}"
        )

    if not final_path.exists():
        raise FileNotFoundError(
            f"Missing Z_full.npy: {final_path}"
        )

    Z_initial = np.load(initial_path)
    Z_final = np.load(final_path)

    print(
        f"Loaded initial representation: "
        f"shape={Z_initial.shape}"
    )

    print(
        f"Loaded final representation: "
        f"shape={Z_final.shape}"
    )

    return Z_initial, Z_final


def validate_embeddings(
    Z_initial: np.ndarray,
    Z_final: np.ndarray,
) -> None:
    """
    Verify that the initial and final representations have
    compatible dimensions and correspond to the same observations.
    """

    if Z_initial.ndim != 2:
        raise ValueError(
            "Z_initial must be 2-dimensional, "
            f"but got shape {Z_initial.shape}."
        )

    if Z_final.ndim != 2:
        raise ValueError(
            "Z_full must be 2-dimensional, "
            f"but got shape {Z_final.shape}."
        )

    if Z_initial.shape != Z_final.shape:
        raise ValueError(
            "Initial and final representations must have "
            f"the same shape, but got "
            f"{Z_initial.shape} and {Z_final.shape}."
        )

    if not np.all(np.isfinite(Z_initial)):
        raise ValueError(
            "Z_initial contains non-finite values."
        )

    if not np.all(np.isfinite(Z_final)):
        raise ValueError(
            "Z_full contains non-finite values."
        )

    print(
        f"\nValidated representation dimensions: "
        f"{Z_initial.shape[0]} samples × "
        f"{Z_initial.shape[1]} latent dimensions."
    )


def compute_displacement(
    Z_initial: np.ndarray,
    Z_final: np.ndarray,
) -> np.ndarray:
    """
    Compute per-observation Euclidean displacement between
    the initial and final latent representations.
    """

    displacement = np.linalg.norm(
        Z_final - Z_initial,
        axis=1,
    )

    return displacement


def summarize_displacement(
    displacement: np.ndarray,
) -> dict:
    """
    Compute descriptive statistics and quantiles of
    latent representation displacement.
    """

    quantiles = {
        "q25": 0.25,
        "q50": 0.50,
        "q75": 0.75,
        "q90": 0.90,
        "q95": 0.95,
        "q99": 0.99,
    }

    summary = {
        "n_samples": int(len(displacement)),
        "mean_displacement": float(
            np.mean(displacement)
        ),
        "median_displacement": float(
            np.median(displacement)
        ),
        "std_displacement": float(
            np.std(displacement)
        ),
        "min_displacement": float(
            np.min(displacement)
        ),
        "max_displacement": float(
            np.max(displacement)
        ),
        "quantiles": {
            name: float(
                np.quantile(displacement, q)
            )
            for name, q in quantiles.items()
        },
    }

    return summary


def save_displacement_results(
    displacement: np.ndarray,
    summary: dict,
    aggregated_dir: Path,
) -> None:
    """
    Save per-observation displacement and summary statistics.
    """

    aggregated_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        aggregated_dir / "latent_displacement.npy",
        displacement,
    )

    summary_with_definition = {
        **summary,
        "distance_metric": "Euclidean",
        "definition": (
            "Per-observation displacement between the initial "
            "pretrained encoder representation and the final "
            "representation after trainable DEC optimization."
        ),
        "formula": (
            "Delta_Z_i = ||Z_final_i - Z_initial_i||_2"
        ),
        "initial_representation": (
            "trainable/Z_initial.npy"
        ),
        "final_representation": (
            "trainable/Z_full.npy"
        ),
    }

    with open(
        aggregated_dir / "representation_change_summary.json",
        "w",
    ) as f:
        json.dump(
            summary_with_definition,
            f,
            indent=2,
        )

    displacement_df = pd.DataFrame(
        {
            "sample_index": np.arange(
                len(displacement)
            ),
            "latent_displacement": displacement,
        }
    )

    displacement_df.to_csv(
        aggregated_dir
        / "latent_displacement_by_sample.csv",
        index=False,
    )

    print(
        f"Saved displacement values to: "
        f"{aggregated_dir / 'latent_displacement.npy'}"
    )

    print(
        f"Saved summary to: "
        f"{aggregated_dir / 'representation_change_summary.json'}"
    )

    print(
        f"Saved per-sample displacement table to: "
        f"{aggregated_dir / 'latent_displacement_by_sample.csv'}"
    )


def plot_displacement_distribution(
    displacement: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot the distribution of per-observation latent displacement.
    """

    plt.figure(figsize=(7, 5))

    plt.hist(
        displacement,
        bins=50,
    )

    plt.xlabel(
        "Latent representation displacement"
    )

    plt.ylabel(
        "Number of observations"
    )

    plt.title(
        "Change in Latent Representation After DEC Training"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved displacement distribution to: "
        f"{output_path}"
    )


def print_results(
    summary: dict,
) -> None:
    """
    Print the main representation-change statistics.
    """

    print("\n" + "=" * 60)
    print("REPRESENTATION CHANGE")
    print("=" * 60)

    print(
        f"\nNumber of samples : "
        f"{summary['n_samples']}"
    )

    print(
        f"Mean displacement : "
        f"{summary['mean_displacement']:.6f}"
    )

    print(
        f"Median displacement : "
        f"{summary['median_displacement']:.6f}"
    )

    print(
        f"Std displacement : "
        f"{summary['std_displacement']:.6f}"
    )

    print(
        f"Minimum displacement : "
        f"{summary['min_displacement']:.6f}"
    )

    print(
        f"Maximum displacement : "
        f"{summary['max_displacement']:.6f}"
    )

    print("\nDisplacement quantiles:")

    for name, value in summary["quantiles"].items():
        print(
            f"  {name:>3}: {value:.6f}"
        )


def main():

    # Project paths
    PROJECT_ROOT = (
        Path(__file__).resolve().parents[2]
    )

    RUN_DIR = (
        PROJECT_ROOT
        / "experiments"
        / "encoder_ablation"
        / "runs"
        / "trainable"
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
    print("DEC ENCODER ABLATION — REPRESENTATION CHANGE")
    print("=" * 60)

    print(
        f"\nRun directory: {RUN_DIR}"
    )

    print(
        f"Output directory: {ANALYSIS_ROOT}"
    )

    # Load representations
    Z_initial, Z_final = load_embeddings(
        run_dir=RUN_DIR
    )

    # Validate
    validate_embeddings(
        Z_initial=Z_initial,
        Z_final=Z_final,
    )

    # Compute displacement
    print(
        "\nComputing per-observation latent displacement..."
    )

    displacement = compute_displacement(
        Z_initial=Z_initial,
        Z_final=Z_final,
    )

    # Summarize
    summary = summarize_displacement(
        displacement
    )

    # Print results
    print_results(
        summary
    )

    # Save results
    save_displacement_results(
        displacement=displacement,
        summary=summary,
        aggregated_dir=AGGREGATED_DIR,
    )

    # Plot distribution
    plot_displacement_distribution(
        displacement=displacement,
        output_path=(
            FIGURES_DIR
            / "representation_displacement_distribution.png"
        ),
    )

    print("\n" + "=" * 60)
    print("REPRESENTATION CHANGE ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()