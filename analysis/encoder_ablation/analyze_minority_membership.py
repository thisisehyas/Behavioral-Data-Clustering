import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.dec.assets import load_features


# Paths

FROZEN_RUN = Path("experiments/encoder_ablation/runs/frozen")
TRAINABLE_RUN = Path("experiments/encoder_ablation/runs/trainable")

OUTPUT_DIR = Path("analysis/encoder_ablation/aggregated")
FIGURES_DIR = Path("analysis/encoder_ablation/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Feature names

FEATURE_NAMES = [
    "Lifestyle",
    "Communication",
    "Responsibility",
    "Openness",
    "Interests",
    "EmotionalStability",
    "SocialEngagement",
]


# Helper functions

def load_labels(run_dir):
    path = run_dir / "full_labels.npy"

    if not path.exists():
        raise FileNotFoundError(f"Labels not found: {path}")

    return np.load(path)


def identify_minority_cluster(labels):
    """Return the cluster label with the smallest membership."""
    unique, counts = np.unique(labels, return_counts=True)
    minority_cluster = unique[np.argmin(counts)]
    return minority_cluster


def summarize_group(X, indices, group_name):
    """Calculate descriptive statistics for one group."""
    group_X = X[indices]

    rows = []

    for feature_idx, feature_name in enumerate(FEATURE_NAMES):
        values = group_X[:, feature_idx]

        rows.append(
            {
                "group": group_name,
                "feature": feature_name,
                "n": len(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values, ddof=1),
                "q25": np.percentile(values, 25),
                "q75": np.percentile(values, 75),
                "min": np.min(values),
                "max": np.max(values),
            }
        )

    return pd.DataFrame(rows)


# Load data

print("Loading feature matrix...")
X = load_features()

if X.shape[1] != len(FEATURE_NAMES):
    raise ValueError(
        f"Expected {len(FEATURE_NAMES)} features, "
        f"but found {X.shape[1]}."
    )

print(f"X shape: {X.shape}")


print("\nLoading cluster assignments...")
frozen_labels = load_labels(FROZEN_RUN)
trainable_labels = load_labels(TRAINABLE_RUN)

if len(frozen_labels) != len(X):
    raise ValueError("Frozen labels and X have different numbers of samples.")

if len(trainable_labels) != len(X):
    raise ValueError("Trainable labels and X have different numbers of samples.")


# Identify minority clusters

frozen_minority = identify_minority_cluster(frozen_labels)
trainable_minority = identify_minority_cluster(trainable_labels)

frozen_minority_indices = np.flatnonzero(
    frozen_labels == frozen_minority
)

trainable_minority_indices = np.flatnonzero(
    trainable_labels == trainable_minority
)


print("\nMinority clusters:")
print(f"Frozen:    cluster {frozen_minority}, "
      f"n={len(frozen_minority_indices)}")
print(f"Trainable: cluster {trainable_minority}, "
      f"n={len(trainable_minority_indices)}")


# Identify newly added minority members

frozen_minority_set = set(frozen_minority_indices)
trainable_minority_set = set(trainable_minority_indices)

newly_added_indices = np.array(
    sorted(trainable_minority_set - frozen_minority_set),
    dtype=int,
)

retained_indices = np.array(
    sorted(frozen_minority_set & trainable_minority_set),
    dtype=int,
)

frozen_only_indices = np.array(
    sorted(frozen_minority_set - trainable_minority_set),
    dtype=int,
)

frozen_major_indices = np.flatnonzero(
    frozen_labels != frozen_minority
)

trainable_major_indices = np.flatnonzero(
    trainable_labels != trainable_minority
)


print("\nMembership comparison:")
print(f"Frozen minority:          {len(frozen_minority_indices)}")
print(f"Trainable minority:       {len(trainable_minority_indices)}")
print(f"Retained minority:        {len(retained_indices)}")
print(f"Newly added members:      {len(newly_added_indices)}")
print(f"Frozen-only members:      {len(frozen_only_indices)}")


# Save newly added observations

newly_added_df = pd.DataFrame(
    X[newly_added_indices],
    columns=FEATURE_NAMES,
)

newly_added_df.insert(0, "sample_index", newly_added_indices)

newly_added_df.to_csv(
    OUTPUT_DIR / "newly_added_minority_members.csv",
    index=False,
)


# Descriptive summaries

groups = {
    "frozen_minority": frozen_minority_indices,
    "newly_added_minority": newly_added_indices,
    "trainable_minority": trainable_minority_indices,
    "frozen_major": frozen_major_indices,
}

summary_frames = []

for group_name, indices in groups.items():
    summary_frames.append(
        summarize_group(X, indices, group_name)
    )

summary_df = pd.concat(summary_frames, ignore_index=True)

summary_df.to_csv(
    OUTPUT_DIR / "minority_group_summary.csv",
    index=False,
)


# Quantile-focused comparison

quantile_rows = []

for group_name, indices in groups.items():
    group_X = X[indices]

    for feature_idx, feature_name in enumerate(FEATURE_NAMES):
        values = group_X[:, feature_idx]

        quantile_rows.append(
            {
                "group": group_name,
                "feature": feature_name,
                "q05": np.percentile(values, 5),
                "q25": np.percentile(values, 25),
                "q50": np.percentile(values, 50),
                "q75": np.percentile(values, 75),
                "q95": np.percentile(values, 95),
            }
        )

quantiles_df = pd.DataFrame(quantile_rows)

quantiles_df.to_csv(
    OUTPUT_DIR / "minority_group_quantiles.csv",
    index=False,
)


# Print key comparison

print("\nMean feature profiles:")
mean_profile = (
    summary_df
    .pivot(index="feature", columns="group", values="mean")
)

print(mean_profile.round(3).to_string())

# Distance-based minority membership audit
# Compare newly added minority members with the frozen minority
# and frozen-major populations in the original 7D standardized space.

print("\nRunning distance-based minority membership audit...")

# Centroids in the original standardized feature space
frozen_minority_centroid = np.mean(
    X[frozen_minority_indices],
    axis=0,
)

frozen_major_centroid = np.mean(
    X[frozen_major_indices],
    axis=0,
)

# Distances for newly added minority members
newly_added_X = X[newly_added_indices]

distance_to_minority = np.linalg.norm(
    newly_added_X - frozen_minority_centroid,
    axis=1,
)

distance_to_major = np.linalg.norm(
    newly_added_X - frozen_major_centroid,
    axis=1,
)

distance_ratio = distance_to_minority / distance_to_major

closer_to_minority = distance_to_minority < distance_to_major

distance_summary = pd.DataFrame(
    {
        "sample_index": newly_added_indices,
        "distance_to_frozen_minority": distance_to_minority,
        "distance_to_frozen_major": distance_to_major,
        "minority_to_major_distance_ratio": distance_ratio,
        "closer_to_frozen_minority": closer_to_minority,
    }
)

# Save observation-level distances
distance_summary.to_csv(
    OUTPUT_DIR / "minority_membership_distance.csv",
    index=False,
)

# Aggregate summary
distance_stats = pd.DataFrame(
    {
        "metric": [
            "n_newly_added",
            "mean_distance_to_frozen_minority",
            "median_distance_to_frozen_minority",
            "mean_distance_to_frozen_major",
            "median_distance_to_frozen_major",
            "mean_distance_ratio",
            "median_distance_ratio",
            "min_distance_ratio",
            "max_distance_ratio",
            "n_closer_to_frozen_minority",
            "pct_closer_to_frozen_minority",
        ],
        "value": [
            len(newly_added_indices),
            np.mean(distance_to_minority),
            np.median(distance_to_minority),
            np.mean(distance_to_major),
            np.median(distance_to_major),
            np.mean(distance_ratio),
            np.median(distance_ratio),
            np.min(distance_ratio),
            np.max(distance_ratio),
            np.sum(closer_to_minority),
            100 * np.mean(closer_to_minority),
        ],
    }
)

distance_stats.to_csv(
    OUTPUT_DIR / "minority_membership_distance_summary.csv",
    index=False,
)

print("\nDistance comparison for newly added minority members:")
print(
    f"Mean distance to frozen minority: "
    f"{np.mean(distance_to_minority):.4f}"
)
print(
    f"Median distance to frozen minority: "
    f"{np.median(distance_to_minority):.4f}"
)
print(
    f"Mean distance to frozen major: "
    f"{np.mean(distance_to_major):.4f}"
)
print(
    f"Median distance to frozen major: "
    f"{np.median(distance_to_major):.4f}"
)
print(
    f"Mean minority/major distance ratio: "
    f"{np.mean(distance_ratio):.4f}"
)
print(
    f"Median minority/major distance ratio: "
    f"{np.median(distance_ratio):.4f}"
)
print(
    f"Closer to frozen minority: "
    f"{np.sum(closer_to_minority)} / "
    f"{len(newly_added_indices)} "
    f"({100 * np.mean(closer_to_minority):.2f}%)"
)

# Plot mean profiles

plt.figure(figsize=(11, 6))

x = np.arange(len(FEATURE_NAMES))

for group_name in groups:
    means = (
        summary_df[
            summary_df["group"] == group_name
        ]
        .set_index("feature")
        .loc[FEATURE_NAMES, "mean"]
        .values
    )

    plt.plot(
        x,
        means,
        marker="o",
        label=group_name,
    )

plt.xticks(x, FEATURE_NAMES, rotation=30, ha="right")
plt.axhline(0, linewidth=1)
plt.ylabel("Mean standardized feature value")
plt.title("Feature Profiles of Minority-Cluster Membership")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "minority_feature_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# Final checks

print("\nFiles written:")
print(OUTPUT_DIR / "minority_group_summary.csv")
print(OUTPUT_DIR / "minority_group_quantiles.csv")
print(OUTPUT_DIR / "newly_added_minority_members.csv")
print(FIGURES_DIR / "minority_feature_comparison.png")

print("\nMinority-membership audit completed.")