# Alpha Sensitivity Analysis (DEC)

This experiment evaluates the sensitivity of Deep Embedded Clustering (DEC) to the Student-t distribution hyperparameter α on the roommate compatibility dataset.

This study examines how α affects optimization dynamics, clustering quality, reproducibility, assignment confidence, and the overall cluster structure.

# Objective

Evaluate the effect of the DEC Student-t distribution parameter α on:

- Optimization dynamics
- Validation behavior
- Clustering quality
- Assignment confidence
- Reproducibility across random seeds
- Cluster-size stability

The goal is to determine an appropriate α for the final DEC model.

# Fixed Parameters

- Encoder: Frozen pretrained encoder
- Latent dimension: 5
- Input features: 7 standardized composite features
- Number of clusters: 5
- Learning rate: 1e-3
- Update interval: 20
- Maximum epochs: 50

# Experimental Variables

Alpha values:

- 0.1
- 0.5
- 1.0
- 2.0
- 5.0

Random seeds:

- 0
- 1
- 2

Total runs: 15

# Validation Protocol

- Fixed 80/20 train-validation split
- Validation data was never used for optimization
- Each α value was evaluated using three independent random seeds
- Final results are reported as mean ± standard deviation across seeds
- All experiments used the same data split to ensure fair comparison.

# Metrics Evaluated

## Optimization

- Final Training KL Divergence
- Final Validation KL Divergence

## Clustering Quality

- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Score

## Assignment Confidence

- Mean assignment entropy

## Reproducibility

- Pairwise Adjusted Rand Index (ARI)
- ARI summary statistics
- ARI heatmaps

## Cluster Structure

- Ranked cluster-size statistics
- Cluster-size sensitivity plots

# Per-Run Outputs

Each experiment run contains:

- meta.json
- run_metrics.json
- train_kl.npy
- val_kl.npy
- val_labels.npy
- q_val.npy
- cluster_centers.npy
- Z_val.npy

Aggregated analysis produces:

- aggregated_alpha_metrics.json
- aggregated_cluster_sizes.json
- ari_summary_alpha.json
- ari*matrix_alpha*\*.npy
- alpha_summary_table.csv
- alpha_summary_table.md

Figures include:

- KL divergence curves
- Clustering metric comparisons
- Entropy analysis
- ARI heatmaps
- Cluster-size sensitivity plots

# Summary of Findings

The experiment revealed a clear distinction between optimization behavior and clustering performance.

Increasing α consistently reduced both training and validation KL divergence while also producing slightly more confident cluster assignments (lower entropy).

However, internal clustering quality metrics (Silhouette, Davies–Bouldin, and Calinski–Harabasz) remained essentially unchanged across the evaluated range.

Pairwise ARI analyses and ARI heatmaps showed highly consistent reproducibility for α values between 0.1 and 2.0, with only a modest decrease at α = 5.

Similarly, the ranked cluster-size distributions remained nearly identical across all α values, indicating that the overall partition structure was preserved.

Overall, these findings indicate that, within the evaluated range (α = 0.1–5.0), the DEC hyperparameter α primarily influences optimization dynamics and assignment confidence while exerting minimal influence on clustering quality, reproducibility, and the overall partition structure.

# Selected Alpha

Recommended value:

α = 2.0

Reasoning:

The alpha sensitivity study showed that increasing α consistently improved optimization (lower training and validation KL divergence) while producing only negligible changes in clustering quality metrics.

- Low training KL divergence
- Low validation KL divergence
- No degradation in clustering quality metrics
- Highest mean ARI among the evaluated values
- Stable cluster-size distribution
- Lower assignment entropy
- Improvements beyond α = 2.0 were negligible while α = 5.0 showed a
  slight reduction in reproducibility.

Accordingly, α = 2.0 was selected for all subsequent experiments because it provides the best overall balance between optimization performance, clustering quality, assignment confidence, and reproducibility.
