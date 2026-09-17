import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.dec.assets import load_features


# Paths

TRAINABLE_RUN = Path(
    "experiments/encoder_ablation/runs/trainable"
)

IDEC_RUN = Path(
    "experiments/idec_diagnostic/runs/seed_42"
)

OUTPUT_DIR = Path(
    "analysis/idec_diagnostic/aggregated"
)

FIGURES_DIR = Path(
    "analysis/idec_diagnostic/figures"
)

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
    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    return unique[np.argmin(counts)]


def summarize_profile(X, indices, condition):
    """Calculate mean feature profile for a minority subgroup."""
    group_X = X[indices]

    rows = []

    for feature_idx, feature_name in enumerate(FEATURE_NAMES):
        values = group_X[:, feature_idx]

        rows.append(
            {
                "condition": condition,
                "feature": feature_name,
                "n": len(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values, ddof=1),
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


# Load cluster assignments

print("\nLoading cluster assignments...")

trainable_labels = load_labels(TRAINABLE_RUN)
idec_labels = load_labels(IDEC_RUN)

if len(trainable_labels) != len(X):
    raise ValueError(
        "Trainable DEC labels and X have different "
        "numbers of samples."
    )

if len(idec_labels) != len(X):
    raise ValueError(
        "IDEC labels and X have different "
        "numbers of samples."
    )


# Identify minority clusters

trainable_minority = identify_minority_cluster(
    trainable_labels
)

idec_minority = identify_minority_cluster(
    idec_labels
)

trainable_minority_indices = np.flatnonzero(
    trainable_labels == trainable_minority
)

idec_minority_indices = np.flatnonzero(
    idec_labels == idec_minority
)

print("\nMinority clusters:")
print(
    f"Trainable DEC: cluster {trainable_minority}, "
    f"n={len(trainable_minority_indices)}"
)
print(
    f"IDEC:          cluster {idec_minority}, "
    f"n={len(idec_minority_indices)}"
)


# Membership overlap

trainable_set = set(trainable_minority_indices)
idec_set = set(idec_minority_indices)

intersection = trainable_set & idec_set
union = trainable_set | idec_set

n_trainable = len(trainable_set)
n_idec = len(idec_set)
n_intersection = len(intersection)
n_union = len(union)

trainable_retention = (
    100 * n_intersection / n_trainable
)

idec_retention = (
    100 * n_intersection / n_idec
)

jaccard = (
    n_intersection / n_union
)

print("\nMinority-membership overlap:")
print(f"Trainable DEC minority: {n_trainable}")
print(f"IDEC minority:          {n_idec}")
print(f"Shared observations:    {n_intersection}")
print(
    f"Trainable DEC retained: "
    f"{trainable_retention:.2f}%"
)
print(
    f"IDEC retained:          "
    f"{idec_retention:.2f}%"
)
print(f"Jaccard overlap:        {jaccard:.4f}")


# Save overlap summary

overlap_df = pd.DataFrame(
    [
        {
            "trainable_dec_minority_cluster":
                trainable_minority,
            "idec_minority_cluster":
                idec_minority,
            "trainable_dec_minority_n":
                n_trainable,
            "idec_minority_n":
                n_idec,
            "shared_n":
                n_intersection,
            "union_n":
                n_union,
            "trainable_dec_retention_pct":
                trainable_retention,
            "idec_retention_pct":
                idec_retention,
            "jaccard_overlap":
                jaccard,
        }
    ]
)

overlap_df.to_csv(
    OUTPUT_DIR / "minority_membership_overlap.csv",
    index=False,
)


# Observation-level membership comparison

shared_indices = np.array(
    sorted(intersection),
    dtype=int,
)

trainable_only_indices = np.array(
    sorted(trainable_set - idec_set),
    dtype=int,
)

idec_only_indices = np.array(
    sorted(idec_set - trainable_set),
    dtype=int,
)

membership_df = pd.DataFrame(
    {
        "sample_index": np.arange(len(X)),
        "trainable_dec_minority":
            trainable_labels == trainable_minority,
        "idec_minority":
            idec_labels == idec_minority,
    }
)

membership_df["membership_category"] = "neither"

membership_df.loc[
    membership_df["trainable_dec_minority"]
    & membership_df["idec_minority"],
    "membership_category"
] = "shared"

membership_df.loc[
    membership_df["trainable_dec_minority"]
    & ~membership_df["idec_minority"],
    "membership_category"
] = "trainable_DEC_only"

membership_df.loc[
    ~membership_df["trainable_dec_minority"]
    & membership_df["idec_minority"],
    "membership_category"
] = "IDEC_only"

membership_df.to_csv(
    OUTPUT_DIR / "minority_membership_observations.csv",
    index=False,
)


# Feature profiles

trainable_profile = summarize_profile(
    X,
    trainable_minority_indices,
    "trainable_DEC",
)

idec_profile = summarize_profile(
    X,
    idec_minority_indices,
    "IDEC",
)

profile_df = pd.concat(
    [trainable_profile, idec_profile],
    ignore_index=True,
)

profile_df.to_csv(
    OUTPUT_DIR / "minority_feature_profiles.csv",
    index=False,
)


# Print mean profiles

print("\nMinority mean feature profiles:")

mean_profile = (
    profile_df
    .pivot(
        index="feature",
        columns="condition",
        values="mean",
    )
    .loc[FEATURE_NAMES]
)

print(
    mean_profile.round(3).to_string()
)


# Profile difference

profile_difference = (
    mean_profile["IDEC"]
    - mean_profile["trainable_DEC"]
)

profile_difference_df = pd.DataFrame(
    {
        "feature": FEATURE_NAMES,
        "trainable_DEC_mean":
            mean_profile["trainable_DEC"].values,
        "IDEC_mean":
            mean_profile["IDEC"].values,
        "IDEC_minus_trainable_DEC":
            profile_difference.values,
    }
)

profile_difference_df.to_csv(
    OUTPUT_DIR / "minority_profile_difference.csv",
    index=False,
)

profile_distance = np.linalg.norm(
    profile_difference.values
)

print(
    f"\nEuclidean distance between minority "
    f"mean profiles: {profile_distance:.4f}"
)


# Plot mean feature profiles

plt.figure(figsize=(11, 6))

x = np.arange(len(FEATURE_NAMES))

for condition in ["trainable_DEC", "IDEC"]:

    means = (
        profile_df[
            profile_df["condition"] == condition
        ]
        .set_index("feature")
        .loc[FEATURE_NAMES, "mean"]
        .values
    )

    plt.plot(
        x,
        means,
        marker="o",
        label=condition,
    )

plt.xticks(
    x,
    FEATURE_NAMES,
    rotation=30,
    ha="right",
)

plt.axhline(
    0,
    linewidth=1,
)

plt.ylabel(
    "Mean standardized feature value"
)

plt.title(
    "Minority-Cluster Feature Profiles: "
    "Trainable DEC vs IDEC"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "minority_profile_trainable_vs_idec.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# Final checks

print("\nFiles written:")

print(
    OUTPUT_DIR / "minority_membership_overlap.csv"
)

print(
    OUTPUT_DIR / "minority_membership_observations.csv"
)

print(
    OUTPUT_DIR / "minority_feature_profiles.csv"
)

print(
    OUTPUT_DIR / "minority_profile_difference.csv"
)

print(
    FIGURES_DIR /
    "minority_profile_trainable_vs_idec.png"
)

print("\nStep 7 completed.")