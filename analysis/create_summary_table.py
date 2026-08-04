import json
from pathlib import Path

import pandas as pd


def mean_std(mean, std, decimals=4):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def create_summary_table(
    aggregated_metrics_path: Path,
    ari_summary_path: Path,
    cluster_sizes_path: Path,
    output_csv_path: Path,
    output_markdown_path: Path,
    paper_output_path: Path,
    parameter_name: str,
    paper_parameter_label: str,
):
    with open(aggregated_metrics_path, "r") as f:
        metrics = json.load(f)

    with open(ari_summary_path, "r") as f:
        ari = json.load(f)

    with open(cluster_sizes_path, "r") as f:
        cluster_sizes = json.load(f)

    rows = []

    for parameter in sorted(metrics.keys(), key=float):

        row = {
            parameter_name: float(parameter),

            "train_kl_mean": metrics[parameter]["final_train_kl"]["mean"],
            "train_kl_std": metrics[parameter]["final_train_kl"]["std"],

            "val_kl_mean": metrics[parameter]["final_val_kl"]["mean"],
            "val_kl_std": metrics[parameter]["final_val_kl"]["std"],

            "silhouette_mean": metrics[parameter]["silhouette"]["mean"],
            "silhouette_std": metrics[parameter]["silhouette"]["std"],

            "davies_bouldin_mean":
                metrics[parameter]["davies_bouldin"]["mean"],
            "davies_bouldin_std":
                metrics[parameter]["davies_bouldin"]["std"],

            "calinski_harabasz_mean":
                metrics[parameter]["calinski_harabasz"]["mean"],
            "calinski_harabasz_std":
                metrics[parameter]["calinski_harabasz"]["std"],

            "entropy_mean":
                metrics[parameter]["mean_entropy"]["mean"],
            "entropy_std":
                metrics[parameter]["mean_entropy"]["std"],

            "mean_ari":
                ari[parameter]["mean_ari"],
            "std_ari":
                ari[parameter]["std_ari"],

            "smallest_cluster_mean":
                cluster_sizes[parameter]["smallest"]["mean"],
            "smallest_cluster_std":
                cluster_sizes[parameter]["smallest"]["std"],

            "largest_cluster_mean":
                cluster_sizes[parameter]["largest"]["mean"],
            "largest_cluster_std":
                cluster_sizes[parameter]["largest"]["std"],
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(parameter_name)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv_path, index=False)

    with open(output_markdown_path, "w") as f:
        f.write(df.round(4).to_markdown(index=False))

    print(df.round(4).to_string(index=False))

    print(f"\nSaved CSV table to: {output_csv_path}")
    print(f"Saved Markdown table to: {output_markdown_path}")

    # ----------------------------
    # Paper table
    # ----------------------------

    paper_rows = []

    for parameter in sorted(metrics.keys(), key=float):

        paper_rows.append({

            paper_parameter_label: float(parameter),

            "Train KL": mean_std(
                metrics[parameter]["final_train_kl"]["mean"],
                metrics[parameter]["final_train_kl"]["std"],
            ),

            "Validation KL": mean_std(
                metrics[parameter]["final_val_kl"]["mean"],
                metrics[parameter]["final_val_kl"]["std"],
            ),

            "Silhouette": mean_std(
                metrics[parameter]["silhouette"]["mean"],
                metrics[parameter]["silhouette"]["std"],
            ),

            "DB Index": mean_std(
                metrics[parameter]["davies_bouldin"]["mean"],
                metrics[parameter]["davies_bouldin"]["std"],
            ),

            "CH Score": mean_std(
                metrics[parameter]["calinski_harabasz"]["mean"],
                metrics[parameter]["calinski_harabasz"]["std"],
                decimals=2,
            ),

            "Entropy": mean_std(
                metrics[parameter]["mean_entropy"]["mean"],
                metrics[parameter]["mean_entropy"]["std"],
            ),

            "ARI": mean_std(
                ari[parameter]["mean_ari"],
                ari[parameter]["std_ari"],
            ),

            "Smallest Cluster": mean_std(
                cluster_sizes[parameter]["smallest"]["mean"],
                cluster_sizes[parameter]["smallest"]["std"],
                decimals=1,
            ),

            "Largest Cluster": mean_std(
                cluster_sizes[parameter]["largest"]["mean"],
                cluster_sizes[parameter]["largest"]["std"],
                decimals=1,
            ),
        })

    paper_df = pd.DataFrame(paper_rows)

    with open(
        paper_output_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(paper_df.to_markdown(index=False))

    print(f"Saved paper table to: {paper_output_path}")