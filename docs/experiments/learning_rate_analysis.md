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

## Results Summary

This section summarizes the empirical evidence obtained from the learning rate sweep, combining quantitative metrics and qualitative visualizations.

### Silhouette vs Learning Rate

The silhouette score exhibits a clear performance peak at intermediate learning rates. While differences across the stable range are modest, learning rates around **1e-3** consistently achieve the highest mean silhouette score, with overlapping error bars across seeds. Extremely high learning rates show degraded clustering quality, while very small learning rates do not provide additional structural gains.

### Entropy vs Learning Rate

Mean assignment entropy increases monotonically as the learning rate decreases. This indicates that smaller learning rates produce softer cluster assignments. No entropy collapse or degenerate behavior is observed; however, the smallest learning rate results in overly soft assignments, suggesting under-confident clustering despite high stability.

### KL Convergence Dynamics

KL divergence curves reveal clear differences in optimization behavior:

* High learning rates exhibit noisier or less smooth convergence.
* Intermediate learning rates converge smoothly without oscillation.
* Very small learning rates converge slowly, with near-flat KL curves in later epochs, indicating under-driven optimization.

### Cross-Seed Stability (ARI)

ARI heatmaps and summary statistics show that stability improves as the learning rate decreases. High learning rates demonstrate increased variance and weaker diagonal structure, while lower learning rates produce strong diagonal coherence, indicating consistent clustering across epochs and seeds.


## Learning Rate Selection Criteria

The learning rate is selected using a **multi-criteria decision framework**, defined *prior to final selection*:

1. **Clustering Quality**
   Near-maximum silhouette score, within error bars of the best-performing configuration.

2. **Assignment Confidence**
   Low entropy indicating confident assignments, without evidence of degenerate collapse or premature hardening.

3. **Optimization Stability**
   Smooth KL divergence trajectories with no oscillation or divergence.

4. **Cross-Seed Reproducibility**
   High mean ARI and reasonable variance across random seeds.

No single metric is used in isolation; the final decision reflects a balance across all criteria.


## Failure Modes Observed

Explicit failure modes were identified for non-selected learning rates:

* **LR = 1e-2**
  High variance across seeds, weak ARI diagonal structure, and unstable clustering behavior.

* **LR = 3e-3**
  Marginal stability and relatively low entropy, suggesting a risk of premature hardening.

* **LR = 1e-4**
  Extremely stable but overly soft assignments, accompanied by slow and under-driven convergence.

These configurations were therefore excluded from final consideration.


## Final Learning Rate Decision

Based on the combined evidence from clustering quality, assignment confidence, optimization behavior, and cross-seed stability, we select:

```
Learning Rate = 1e-3
```

This value provides the best overall trade-off between performance, stability, and convergence efficiency.

## Learning Rate Freeze Statement

The learning rate is **fixed to 1e-3 for all subsequent experimental phases**.

No further learning rate tuning is performed beyond this point.
All results reported in later stages assume this fixed learning rate.

Any deviation from this value would require re-running the entire learning rate sensitivity analysis.
