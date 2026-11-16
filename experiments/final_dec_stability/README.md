# Final DEC Stability Analysis

This experiment evaluates the stability of the final Deep Embedded Clustering (DEC)
solution across multiple random seeds on the full roommate compatibility dataset.

The purpose is to determine whether the final clustering structure is reproducible
or substantially dependent on random initialization.

## Fixed Parameters

- Encoder: frozen pretrained encoder
- Input features: 7 standardized composite features
- Latent dimension: 5
- Number of clusters: 5
- α: 0.5
- Learning rate: 3e-3
- Update interval: 20
- Batch size: 256
- Maximum epochs: 50
- KMeans initialization: n_init = 20

## Experimental Variable

- Random seed ∈ {42, 43, 44, 45, 46}

Total runs: 5

## Experimental Protocol

- Full dataset is used for every run
- No train/validation split is used
- No hyperparameter selection is performed
- All DEC hyperparameters are fixed to the canonical final configuration
- Only the random seed is varied
- Each run independently performs KMeans initialization and DEC training

## Metrics Evaluated

- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Score
- Mean assignment entropy
- Cluster sizes
- Pairwise Adjusted Rand Index (ARI)

## Per-Run Outputs

Each run directory contains:

- `meta.json`
- `train_kl.npy`
- `full_labels.npy`
- `q_full.npy`
- `Z_full.npy`
- `cluster_centers.npy`

## Purpose

The results are used to assess the reproducibility and stability of the
canonical final DEC clustering solution across random seeds.