# IDEC Stability Analysis

This experiment evaluates the stability of the final Improved Deep Embedded
Clustering (IDEC) solution across multiple random seeds on the full roommate
compatibility dataset.

The purpose is to assess whether the final IDEC clustering structure is
consistent across different random initializations.

## Fixed Parameters

- Encoder: trainable pretrained encoder
- Decoder: trainable
- Input features: 7 standardized composite features
- Latent dimension: 5
- Number of clusters: 5
- α: 0.5
- Learning rate: 3e-3
- Update interval: 20
- Batch size: 256
- Maximum epochs: 50
- KMeans initialization: n_init = 20
- Clustering weight (γ): 0.1

## Experimental Variable

- Random seed ∈ {42, 43, 44, 45, 46}

Total runs: 5

## Experimental Protocol

- The full dataset is used for every run.
- No train/validation split is used.
- No hyperparameter selection is performed.
- All IDEC parameters are fixed to the final configuration.
- Only the random seed is varied.
- Each run independently performs KMeans initialization and IDEC training.
- The same pretrained autoencoder is used as the initial model for all runs.

## Metrics Evaluated

- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Score
- Mean assignment entropy
- Cluster sizes
- Pairwise Adjusted Rand Index (ARI)
- Latent representation displacement
- Reconstruction loss

## Per-Run Outputs

Each run directory contains:

- `meta.json`
- `train_kl.npy`
- `train_reconstruction.npy`
- `full_labels.npy`
- `q_full.npy`
- `Z_full.npy`
- `Z_initial.npy`
- `cluster_centers.npy`

## Purpose

The results are used to assess the reproducibility and stability of the
final IDEC clustering solution across random seeds.