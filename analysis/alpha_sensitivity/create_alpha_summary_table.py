import json
from pathlib import Path

import pandas as pd

AGG_METRICS_PATH = Path(
    "analysis/alpha_sensitivity/aggregated/aggregated_alpha_metrics.json"
)

ARI_PATH = Path(
    "analysis/alpha_sensitivity/aggregated/ari_summary_alpha.json"
)

CLUSTER_SIZE_PATH = Path(
    "analysis/alpha_sensitivity/aggregated/aggregated_cluster_sizes.json"
)

OUTPUT_PATH = Path(
    "analysis/alpha_sensitivity/aggregated/alpha_summary_table.csv"
)

MARKDOWN_OUTPUT_PATH = Path(
    "analysis/alpha_sensitivity/aggregated/alpha_summary_table.md"
)

# NEW
PAPER_OUTPUT_PATH = Path(
    "analysis/alpha_sensitivity/aggregated/paper_alpha_summary_table.md"
)


with open(AGG_METRICS_PATH, "r") as f:
    metrics = json.load(f)

with open(ARI_PATH, "r") as f:
    ari = json.load(f)

with open(CLUSTER_SIZE_PATH, "r") as f:
    cluster_sizes = json.load(f)


rows = []

for alpha in sorted(metrics.keys(), key=float):

    row = {
        "alpha": float(alpha),

        "train_kl_mean": metrics[alpha]["final_train_kl"]["mean"],
        "train_kl_std": metrics[alpha]["final_train_kl"]["std"],

        "val_kl_mean": metrics[alpha]["final_val_kl"]["mean"],
        "val_kl_std": metrics[alpha]["final_val_kl"]["std"],

        "silhouette_mean": metrics[alpha]["silhouette"]["mean"],
        "silhouette_std": metrics[alpha]["silhouette"]["std"],

        "davies_bouldin_mean": metrics[alpha]["davies_bouldin"]["mean"],
        "davies_bouldin_std": metrics[alpha]["davies_bouldin"]["std"],

        "calinski_harabasz_mean":
            metrics[alpha]["calinski_harabasz"]["mean"],
        "calinski_harabasz_std":
            metrics[alpha]["calinski_harabasz"]["std"],

        "entropy_mean": metrics[alpha]["mean_entropy"]["mean"],
        "entropy_std": metrics[alpha]["mean_entropy"]["std"],

        "mean_ari": ari[alpha]["mean_ari"],
        "std_ari": ari[alpha]["std_ari"],

        "smallest_cluster_mean":
            cluster_sizes[alpha]["smallest"]["mean"],
        "smallest_cluster_std":
            cluster_sizes[alpha]["smallest"]["std"],

        "largest_cluster_mean":
            cluster_sizes[alpha]["largest"]["mean"],
        "largest_cluster_std":
            cluster_sizes[alpha]["largest"]["std"],
    }

    rows.append(row)


df = pd.DataFrame(rows)

df = df.sort_values("alpha")

df.to_csv(OUTPUT_PATH, index=False)

with open(MARKDOWN_OUTPUT_PATH, "w") as f:
    f.write(df.round(4).to_markdown(index=False))


print("\nAlpha Summary Table:\n")
print(df.round(4).to_string(index=False))

print(f"\nSaved CSV table to: {OUTPUT_PATH}")
print(f"Saved Markdown table to: {MARKDOWN_OUTPUT_PATH}")

# ==========================================================
# Paper-ready summary table
# ==========================================================

def mean_std(mean, std, decimals=4):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


paper_rows = []

for alpha in sorted(metrics.keys(), key=float):

    paper_rows.append({

        "α": float(alpha),

        "Train KL": mean_std(
            metrics[alpha]["final_train_kl"]["mean"],
            metrics[alpha]["final_train_kl"]["std"],
        ),

        "Validation KL": mean_std(
            metrics[alpha]["final_val_kl"]["mean"],
            metrics[alpha]["final_val_kl"]["std"],
        ),

        "Silhouette": mean_std(
            metrics[alpha]["silhouette"]["mean"],
            metrics[alpha]["silhouette"]["std"],
        ),

        "DB Index": mean_std(
            metrics[alpha]["davies_bouldin"]["mean"],
            metrics[alpha]["davies_bouldin"]["std"],
        ),

        "CH Score": mean_std(
            metrics[alpha]["calinski_harabasz"]["mean"],
            metrics[alpha]["calinski_harabasz"]["std"],
            decimals=2,
        ),

        "Entropy": mean_std(
            metrics[alpha]["mean_entropy"]["mean"],
            metrics[alpha]["mean_entropy"]["std"],
        ),

        "ARI": mean_std(
            ari[alpha]["mean_ari"],
            ari[alpha]["std_ari"],
        ),

        "Smallest Cluster": mean_std(
            cluster_sizes[alpha]["smallest"]["mean"],
            cluster_sizes[alpha]["smallest"]["std"],
            decimals=1,
        ),

        "Largest Cluster": mean_std(
            cluster_sizes[alpha]["largest"]["mean"],
            cluster_sizes[alpha]["largest"]["std"],
            decimals=1,
        ),
    })


paper_df = pd.DataFrame(paper_rows)

with open(PAPER_OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(paper_df.to_markdown(index=False))

print(f"Saved paper table to: {PAPER_OUTPUT_PATH}")