# Alpha Sensitivity Analysis (DEC)

This experiment evaluates the sensitivity of Deep Embedded Clustering (DEC)
to the Student-t distribution hyperparameter α on the roommate compatibility
dataset.

## Fixed Parameters

- Encoder: frozen pretrained encoder
- Latent dimension: 5
- Input features: 7 standardized composite features
- Number of clusters: 5
- Learning rate: 1e-3
- Update interval: 20
- Maximum epochs: 50

## Experimental Variables

- α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
- Random seed ∈ {0, 1, 2}

Total runs: 15

## Validation Protocol

- Fixed 80/20 train-validation split
- Validation data is never used for optimization
- Same data split for all experiments
- Results aggregated across three random seeds

## Metrics Evaluated

- Training KL divergence
- Validation KL divergence
- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Score
- Mean assignment entropy
- Pairwise Adjusted Rand Index (ARI)
- Ranked cluster-size statistics

## Per-Run Outputs

Each run directory contains:

- meta.json
- run_metrics.json
- train_kl.npy
- val_kl.npy
- val_labels.npy
- q_val.npy
- cluster_centers.npy
- Z_val.npy

## Aggregated Outputs

Analysis scripts generate:

- aggregated_alpha_metrics.json
- aggregated_cluster_sizes.json
- ari_summary_alpha.json
- ari_matrix_alpha_*.npy
- alpha_summary_table.csv
- alpha_summary_table.md