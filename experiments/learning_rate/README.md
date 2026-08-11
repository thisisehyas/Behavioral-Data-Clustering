# Learning Rate Tuning (DEC)

This directory contains experiments for systematic learning rate tuning
of Deep Embedded Clustering (DEC) on the roommate compatibility dataset.

## Fixed parameters
- Encoder: frozen pretrained autoencoder
- Latent dimension: 5
- Input features: 7 standardized composite features
- Number of clusters: 5
- Alpha (Student t-parameter): 1.0
- Update interval: 20
- Max epochs: 50
- Batch size: 256

## Variables
- Learning rate ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
- Random seed ∈ {0, 1, 2, 3, 4}

Total runs: 25

## Validation protocol
- Fixed 80/20 train/validation split
- Validation data is not used for optimization
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

## Artifacts per run

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

- aggregated_lr_metrics.json
- aggregated_cluster_sizes.json
- ari_summary_learning_rate.json
- ari_matrix_lr_*.npy
- learning_rate_summary_table.csv
- learning_rate_summary_table.md