# IDEC Diagnostic Experiment

This experiment evaluates the effect of adding a reconstruction constraint to Deep Embedded Clustering (DEC) while keeping the pretrained encoder trainable. The experiment assesses whether reconstruction regularization changes the clustering solution and limits latent representation changes observed with trainable DEC.

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
- Clustering weight: 0.1

## Experimental Variable

- Reconstruction-constrained objective:

  - Reconstruction loss
  - KL clustering loss weighted by `clustering_weight`

## Experimental Protocol

- Full dataset is used
- No train/validation split is used
- No hyperparameter selection is performed during the run
- All DEC hyperparameters are fixed to the trainable DEC configuration
- The model starts from the same pretrained autoencoder used in the trainable DEC experiment
- KMeans initialization is performed independently before IDEC training
- Both the encoder and decoder are trainable
- The IDEC objective is:

$$
L = L_{reconstruction} + \gamma L_{KL}
$$

where $\gamma=0.1$.

The clustering weight is adopted from the original IDEC formulation by Guo et al. (IJCAI 2017), rather than tuned on the current dataset. The original implementation uses $\gamma=0.1$ and assigns a weight of 1 to the reconstruction loss.

## Metrics Evaluated

- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Score
- Mean assignment entropy
- Cluster sizes
- Cluster profiles
- Adjusted Rand Index against trainable DEC
- Change in latent representation from the initial pretrained embedding
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

The results are used to assess whether reconstruction regularization materially changes the trainable DEC solution and whether it constrains latent representation changes sufficiently to inform the choice between vanilla trainable DEC and an IDEC-style formulation.

## Reference

Guo, X., Gao, L., Liu, X., & Yin, J. (2017). *Improved Deep Embedded Clustering with Local Structure Preservation*. Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI-17).
