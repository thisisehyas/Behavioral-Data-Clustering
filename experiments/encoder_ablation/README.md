# DEC Encoder Trainability Ablation

This experiment evaluates the effect of allowing the pretrained encoder to update during Deep Embedded Clustering (DEC) training on the full roommate compatibility dataset.

The purpose is to determine whether keeping the encoder frozen, as in the canonical final DEC configuration, produces a substantially different clustering solution from allowing the encoder to be trained jointly with the DEC clustering layer.

## Fixed Parameters

- Input features: 7 standardized composite features
- Latent dimension: 5
- k: 5
- alpha: 0.5
- Learning rate: 3e-3
- Update interval: 20
- Batch size: 256
- Maximum epochs: 50
- KMeans initialization: n_init=20
- Random seed: 42

## Experimental Variable

- Encoder trainability:

  - Frozen pretrained encoder
  - Trainable pretrained encoder

## Experimental Protocol

- Full dataset is used for both conditions
- No train/validation split is used
- No hyperparameter selection is performed during the runs
- All DEC hyperparameters are fixed to the canonical final configuration
- Both conditions start from the same pretrained autoencoder
- Both conditions independently perform KMeans initialization and DEC training
- The only experimental difference is whether the encoder parameters are updated during DEC training

## Metrics Evaluated

- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Score
- Mean assignment entropy
- Cluster sizes
- Adjusted Rand Index between the two conditions
- Change in latent representation from the initial pretrained embedding

## Per-Run Outputs

Each run directory contains:

- `meta.json`
- `train_kl.npy`
- `full_labels.npy`
- `q_full.npy`
- `Z_full.npy`
- `Z_initial.npy`
- `cluster_centers.npy`

## Purpose

The results are used to assess whether encoder trainability materially affects the final DEC representation, clustering structure, and internal clustering quality, and to determine whether the frozen-encoder choice is supported as a reasonable modeling decision.
