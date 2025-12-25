# Learning Rate Tuning (DEC)

This directory contains experiments for systematic learning rate tuning
of Deep Embedded Clustering (DEC) on the roommate compatibility dataset.

## Fixed parameters
- Encoder: frozen pretrained autoencoder
- Latent dimension: 5
- Input features: 7D standardized compressed features
- Number of clusters: 5
- Update interval: 20
- Max iterations: 50

## Variables
- Learning rate ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
- Random seed ∈ {0, 1, 2}

## Validation protocol
- Fixed 80/20 train/validation split
- Validation data is not used for optimization

## Artifacts per run
Each run directory contains:
- kl_train.npy
- kl_val.npy
- q_val.npy
- labels_val.npy
- cluster_centers.npy
- meta.json
