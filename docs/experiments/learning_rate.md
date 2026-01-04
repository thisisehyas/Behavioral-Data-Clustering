# Learning Rate Sensitivity Analysis (DEC)

The goal of this experiment is to systematically analyze the effect of the learning rate on the performance, stability, and convergence behavior of the Deep Embedded Clustering (DEC) model.

Learning rate is known to be a critical hyperparameter in DEC, as overly large values may destabilize cluster assignments, while overly small values may lead to slow or suboptimal convergence. Rather than selecting a learning rate heuristically, we perform a controlled sweep and evaluate results across multiple random seeds.


## Experimental Setup

### Dataset and Features
- Input space consists of compressed feature representations obtained from a pretrained autoencoder.
- The autoencoder weights are **fixed** during DEC training.
- Clustering is performed in the latent space of the encoder.
- Number of clusters is fixed across all runs.

### Train / Validation Split
- A fixed train–validation split is used across **all** runs.
- Validation indices are identical across learning rates and random seeds.
- All evaluation metrics are computed on the validation set only.

This ensures comparability across runs and enables cross-seed stability analysis.


## Learning Rate Sweep

The following learning rates were evaluated:

- 1e-2
- 3e-3
- 1e-3
- 3e-4
- 1e-4

For each learning rate, the model was trained using **three different random seeds**, resulting in fully controlled learning-rate × seed experiments.

## Logged Artifacts (Per Run)

Each run logs the following artifacts:

- `train_kl.npy` — training KL divergence per iteration
- `val_kl.npy` — validation KL divergence per iteration
- `val_labels.npy` — cluster assignments for validation samples
- `q_val.npy` — soft assignment distributions for validation samples
- `Z_val.npy` — latent embeddings of validation samples
- `cluster_centers.npy` — learned DEC cluster centers
- `meta.json` — run configuration and metadata
- `run_metrics.json` — post-hoc computed metrics

All artifacts are stored under a deterministic directory structure:
`experiments/lr-tuning/runs/lr_<value>/seed_<seed>/`


## Evaluation Metrics

Metrics are computed **post-training** to ensure identical evaluation logic across runs.

### Clustering Quality
- **Silhouette score** (higher is better)
- **Calinski–Harabasz index** (higher is better)
- **Davies–Bouldin index** (lower is better)

All clustering metrics are computed on the validation latent embeddings (`Z_val`) using predicted labels.

### Assignment Confidence
- **Mean entropy of soft assignments**
- **Standard deviation of entropy**

Lower entropy indicates more confident cluster assignments.


### Optimization Behavior
- Final training KL divergence
- Final validation KL divergence
- Minimum validation KL divergence
- KL gap (final validation − final training)

These metrics help assess convergence quality and overfitting behavior.

## Aggregation Across Seeds

For each learning rate:
- Metrics are aggregated across seeds
- Mean and standard deviation are reported

This allows us to distinguish genuine performance trends from seed-specific variance.

## Cross-Seed Stability

To assess clustering stability, we compute **Adjusted Rand Index (ARI)** between validation labelings obtained from different random seeds.

For each learning rate:
- All seed pairs are compared
- An ARI matrix is constructed
- Mean and standard deviation of ARI are reported

High ARI indicates that the clustering solution is reproducible across random initializations.

## Results Summary (Preliminary)

> *This section will be updated once all figures are finalized.*

Initial results indicate that:
- Extremely high learning rates tend to increase variance across seeds
- Extremely low learning rates reduce optimization instability but may limit cluster separation
- Intermediate learning rates provide a better balance between clustering quality and stability

## Decision Criteria

The final learning rate selection will be based on a combination of:
1. High silhouette score
2. Low variance across seeds
3. Strong ARI stability
4. Smooth KL convergence behavior


## Reproducibility Notes

- All random seeds are explicitly set
- All splits are fixed and reused
- No post-hoc tuning was performed after observing results

This ensures that reported conclusions are reproducible and unbiased.
