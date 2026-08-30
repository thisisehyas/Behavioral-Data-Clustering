# Update Interval Tuning (DEC)

This experiment evaluates the sensitivity of Deep Embedded Clustering (DEC)
to the frequency of target-distribution updates on the roommate compatibility
dataset.

## Fixed Parameters

- Encoder: frozen pretrained encoder
- Latent dimension: 5
- Input features: 7 standardized composite features
- Number of clusters: 5
- Alpha (Student-t parameter): 1.0
- Learning rate: 1e-3
- Maximum epochs: 50
- Batch size: 256
- KMeans initialization: `n_init = 20`

## Experimental Variables

- Update interval ∈ {1, 5, 10, 20, 50}
- Random seed ∈ {0, 1, 2, 3, 4}

Total runs: 25

## Validation Protocol

- Fixed 80/20 train-validation split
- KMeans initialization is performed using the training split only
- DEC optimization is performed using the training split only
- Validation data is never used for optimization
- The same data split is used for all experiments
- Results are aggregated across five random seeds

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

- aggregated_update_interval_metrics.json
- aggregated_cluster_sizes.json
- ari_summary_update_interval.json
- ari_matrix_update_interval_*.npy
- update_interval_summary_table.csv
- update_interval_summary_table.md
