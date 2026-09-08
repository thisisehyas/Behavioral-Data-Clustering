# Final DEC k-Sensitivity Experiment

This experiment evaluates the sensitivity of the final Deep Embedded Clustering (DEC) solution to the number of clusters on the full roommate compatibility dataset.

The purpose is to determine whether the selected value of k=5 is supported by the DEC clustering results or whether substantially different values of k produce preferable solutions.

The range k in {2,3,4,5,6} was selected based on the preliminary K-Means k-sensitivity analysis performed on the DEC latent representation, with particular attention to the stronger internal validation results observed for k=2 and k=3 and the motivation for evaluating k=5.

## Fixed Parameters

- Encoder: frozen pretrained encoder
- Input features: 7 standardized composite features
- Latent dimension: 5
- alpha: 0.5
- Learning rate: 3e-3
- Update interval: 20
- Batch size: 256
- Maximum epochs: 50
- KMeans initialization: n_init=20
- Random seed: 42

## Experimental Variable

- Number of clusters k in {2, 3, 4, 5, 6}

Total runs: 5

## Experimental Protocol

- Full dataset is used for every run
- No train/validation split is used
- No hyperparameter selection is performed during the runs
- All DEC hyperparameters are fixed to the canonical final configuration
- Only the number of clusters (k) is varied
- Each run independently performs KMeans initialization and DEC training

## Metrics Evaluated

- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Score
- Mean assignment entropy
- Cluster sizes

## Per-Run Outputs

Each run directory contains:

- `meta.json`
- `train_kl.npy`
- `full_labels.npy`
- `q_full.npy`
- `Z_full.npy`
- `cluster_centers.npy`

## Purpose

The results are used to assess the sensitivity of the DEC clustering solution to the number of clusters and to evaluate whether the canonical choice of (k=5) is supported by the experimental results.
