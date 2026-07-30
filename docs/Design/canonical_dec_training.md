# Canonical DEC Training Methodology

## Design Principles

The training procedure is based on the following principles:

- Every hyperparameter must be evaluated under the same optimization procedure that will be used for the final model.
- The only difference between tuning experiments should be the hyperparameter being swept.
- Validation data is used only for hyperparameter selection and never for optimization.
- After hyperparameters are selected, the final model is retrained on the full dataset.


# Canonical Training Procedure

## 1. Frozen Encoder

### Decision

The pretrained autoencoder remains frozen during DEC optimization.

### Reasoning

The objective of this work is to evaluate DEC as a clustering refinement method rather than jointly optimize the representation and clustering objective.

Keeping the encoder fixed:

- isolates the effect of DEC itself,
- improves experimental reproducibility,
- reduces the number of trainable parameters,
- makes hyperparameter comparisons more reliable.


## 2. Fresh KMeans Initialization

### Decision

A new KMeans model is fitted at the beginning of every training run.

During hyperparameter tuning:

- KMeans is trained only on the training embeddings.
- The initialization uses that run's assigned random seed.
- `n_init = 20`.

During final training:

- KMeans is trained on embeddings from the full dataset.
- A single pre-designated seed is used.
- `n_init = 20`.

### Reasoning

DEC refines the cluster centers produced by KMeans.

Each training run begins from a new KMeans initialization determined by its assigned random seed. Therefore, each seed captures the combined variability of both KMeans initialization and DEC optimization.

Reporting the mean ± standard deviation across multiple seeds evaluates the stability and expected performance of the complete DEC training pipeline rather than the performance obtained from one favorable initialization.


## 3. Mini-Batch DEC Optimization

### Decision

DEC is optimized using mini-batch gradient descent.

Configuration:

- Optimizer: Adam
- Batch size: 256
- Number of epochs: 50

### Reasoning

The dataset contains approximately 100,000 samples, making full-batch optimization unnecessarily expensive in both memory usage and computation time.

Mini-batch optimization is substantially more computationally efficient while remaining the standard optimization strategy for deep neural network training. It also makes the training procedure practical to reproduce on typical GPU hardware.


## 4. Target Distribution Updates

### Decision

The target distribution is recomputed periodically using the entire training set.

The optimization itself is performed using mini-batches, but every update interval:

1. compute embeddings for the complete training set,
2. compute soft assignments (`q`),
3. compute the target distribution (`p`),
4. continue mini-batch optimization using the updated targets.

During final training, the same procedure is applied to the full dataset.

### Reasoning

The DEC target distribution is defined globally over the dataset.

Updating it using the complete training set produces stable optimization while retaining the computational efficiency of mini-batch learning.


## 5. Train / Validation Split

### Decision

Hyperparameter tuning uses one fixed train/validation split.

- Only the training split is used during optimization.
- Validation samples are used exclusively for evaluation.

### Reasoning

A fixed validation set provides an unbiased estimate of hyperparameter performance while ensuring every configuration is evaluated on identical data.

Using the same split across all experiments isolates the effect of the hyperparameter being tuned.


## 6. Multiple Random Seeds

### Decision

Every hyperparameter configuration is evaluated using five random seeds:

```
{0, 1, 2, 3, 4}
```

Each seed changes:

- KMeans initialization,
- optimization randomness,
- mini-batch ordering.

These tuning seeds are separate from the seed used for final model training.

### Reasoning

DEC is sensitive to initialization.

Evaluating multiple seeds allows hyperparameters to be selected based on average performance and stability rather than a single favorable run.

Five seeds are used because previous tuning experiments revealed that differences between candidate hyperparameter values can be small. Increasing the number of seeds improves the reliability of distinguishing genuine effects from seed-to-seed variability.

All reported means and standard deviations are computed over these five tuning seeds.


## 7. Hyperparameter Selection

### Decision

Hyperparameters are selected using a decision rule fixed before any tuning experiment is analyzed.

For every candidate configuration:

1. Compute mean ± standard deviation across five seeds.
2. Rank primarily by the mean Silhouette Score.
3. If two configurations achieve comparable Silhouette Scores, the predefined tie-breakers are applied sequentially until one configuration is preferred.
4. Report the full table for transparency.


Primary metric:

1. Silhouette Score

Tie-breakers (applied in order):

1. Davies–Bouldin Index
2. Calinski–Harabasz Score
3. Mean Adjusted Rand Index (ARI)
4. Mean Assignment Entropy
5. Validation KL Divergence
6. Training KL Divergence

The complete results for every evaluated configuration are reported regardless of which configuration is ultimately selected.

### Reasoning

Defining the decision rule before inspecting the experimental results prevents post-hoc hyperparameter selection and improves the transparency and reproducibility of the study.


## 8. Final Model Training

### Decision

After hyperparameter tuning is complete:

- the selected hyperparameters are fixed,
- the train/validation split is removed,
- KMeans is fitted on embeddings from the full dataset,
- a single fixed random seed (42) is used for all components of the final training pipeline, including KMeans initialization, NumPy, Python, and PyTorch random number generators,
- DEC is trained on the full dataset,
- the final clustering solution is produced for downstream analysis.

### Reasoning

Hyperparameter tuning intentionally varies the random seed to measure the stability of the complete training pipeline.

The final reported model serves a different purpose: it must produce one specific, fully reproducible clustering solution that can be used consistently for visualization, cluster profiling, and discussion throughout the remainder of the project.

Using one fixed, pre-designated seed (42) guarantees that the final clustering solution can always be reproduced exactly. This seed is chosen before final training and is independent of the tuning seeds used to estimate model stability.