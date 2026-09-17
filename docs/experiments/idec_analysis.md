# IDEC Diagnostic Experiment

This experiment evaluates the effect of adding a reconstruction constraint to Deep Embedded Clustering (DEC) while keeping the pretrained encoder trainable. The purpose is to determine whether reconstruction regularization changes the trainable-DEC solution and moderates the latent representation adaptation observed in the encoder-trainability ablation.

## Research Question

Does reconstruction regularization substantially affect the trainable DEC solution and reduce the latent representation adaptation observed when the pretrained encoder is allowed to train?

## Experimental Design

The diagnostic compares:

1. **Trainable DEC:** pretrained encoder is trainable and optimization uses the DEC clustering objective.
2. **IDEC:** pretrained encoder and decoder are trainable, with reconstruction loss added to the DEC clustering objective.

The DEC hyperparameters are kept fixed between the two conditions.

The experiment is conducted on the full standardized roommate compatibility dataset without a train/validation split.

## Dataset

- Number of observations: 99,997
- Input features: 7 standardized composite features
- Features:
  - Lifestyle
  - Communication
  - Responsibility
  - Openness
  - Interests
  - EmotionalStability
  - SocialEngagement
- Latent dimension: 5
- Number of clusters: 5

## IDEC Formulation

The IDEC objective combines reconstruction loss and DEC clustering loss:

$$
L = L_{\mathrm{reconstruction}} + \gamma L_{\mathrm{KL}}
$$

where:

- $L_{\mathrm{reconstruction}}$ is the reconstruction loss between the input and decoder output.
- $L_{\mathrm{KL}}$ is the DEC KL-divergence clustering loss.
- $\gamma = 0.1$ is the clustering-loss weight.

The reconstruction loss uses mean squared error.

The reconstruction weight and optimization settings are not independently tuned on the current dataset.

## Source of the Reconstruction-Regularized Formulation

The reconstruction-constrained formulation follows the IDEC approach described by:

Guo, X., Gao, L., Liu, X., & Yin, J. (2017). *Improved Deep Embedded Clustering with Local Structure Preservation*. Proceedings of IJCAI-17, pp. 1753–1759. DOI: 10.24963/ijcai.2017/243.

The clustering weight $\gamma=0.1$ is adopted from the original IDEC formulation. It is not tuned on the current dataset.

This experiment does not constitute an exact reproduction of the original IDEC implementation because the present study retains the architecture and DEC optimization settings established for the current dataset.

## Fixed DEC Configuration

- Number of clusters (`k`): 5
- Student-t parameter (`alpha`): 0.5
- Learning rate: 3e-3
- Optimizer: Adam
- Target-distribution update interval: 20 epochs
- Batch size: 256
- Maximum epochs: 50
- KMeans initialization: `n_init=20`
- Random seed: 42
- Encoder: trainable
- Decoder: trainable
- Dataset usage: full dataset

The DEC hyperparameter configuration was selected in previous controlled sensitivity experiments with the encoder frozen and then held constant in the encoder-trainability ablation. The same configuration is retained here to isolate the effect of reconstruction regularization.

## Experimental Protocol

1. Load the standardized seven-feature dataset.
2. Load the pretrained autoencoder.
3. Generate the initial latent representation.
4. Initialize cluster centers independently using KMeans with `n_init=20`.
5. Construct the trainable DEC model.
6. Allow both encoder and decoder parameters to update.
7. Optimize the combined reconstruction and clustering objective.
8. Update the DEC target distribution every 20 epochs.
9. Obtain the final latent representation, soft assignments, cluster labels, and cluster centers.
10. Evaluate clustering quality, cluster structure, partition agreement, and latent representation change.

No hyperparameter selection is performed during the IDEC run.

## Evaluation

The diagnostic evaluates:

- Reconstruction loss
- KL-divergence loss
- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Score
- Mean assignment entropy
- Normalized mean entropy
- Cluster sizes
- Cluster profiles
- Partition agreement using Adjusted Rand Index (ARI)
- Latent representation displacement
- Minority-cluster membership

UMAP is used only for visualization and is not used as a quantitative evaluation metric.

## Training Behavior

The reconstruction loss decreased from 0.1169 at epoch 1 to 0.0828 at epoch 50. The decrease remained stable throughout training, with small fluctuations around target-distribution updates and no evidence of divergence.

The final KL loss was:

- Trainable DEC: 0.0493
- IDEC: 0.0639

The IDEC KL trajectory showed periodic increases following target-distribution updates around epochs 21 and 41, followed by subsequent decreases.

The higher final KL loss under IDEC is consistent with the additional reconstruction objective, which constrains optimization of the clustering objective alone.

## Internal Clustering Quality

| Metric | Trainable DEC | IDEC |
|---|---:|---:|
| Silhouette | 0.6189 | 0.4829 |
| Davies–Bouldin | 0.5360 | 0.7682 |
| Calinski–Harabasz | 213,778.8 | 145,009.0 |
| Mean entropy | 0.7718 | 1.0097 |
| Normalized mean entropy | 0.4796 | 0.6274 |

Adding reconstruction regularization reduced the internal clustering quality according to all evaluated clustering metrics.

IDEC produced a lower Silhouette score and Calinski–Harabasz score, a higher Davies–Bouldin index, and higher mean and normalized assignment entropy.

These metrics indicate that trainable DEC forms a more compact and separated clustering solution in its learned latent space. However, these metrics are also directly related to the clustering objective, so they should not be interpreted independently of the representation-change analysis.

## Cluster Sizes

### Trainable DEC

- C0: 929 (0.93%)
- C1: 24,030 (24.03%)
- C2: 25,683 (25.68%)
- C3: 27,174 (27.17%)
- C4: 22,181 (22.18%)

### IDEC

- C0: 488 (0.49%)
- C1: 22,534 (22.53%)
- C2: 26,361 (26.36%)
- C3: 29,612 (29.61%)
- C4: 21,002 (21.00%)

IDEC retained the five-cluster structure while reducing the smallest cluster from 929 observations to 488. The remaining four clusters remained broadly comparable in size, ranging from 21.0% to 29.6% of the dataset.

The IDEC minority-cluster size is also close to the 489 observations identified in the earlier frozen-encoder DEC condition.

## Cluster Profile Comparison

Cluster profiles were matched using minimum Euclidean distance in the original seven-dimensional standardized feature space.

| Trainable DEC | IDEC | Profile distance |
|---|---|---:|
| C0 | C0 | 5.034 |
| C1 | C1 | 0.309 |
| C2 | C2 | 0.143 |
| C3 | C3 | 0.107 |
| C4 | C4 | 0.283 |

The four major clusters show strong profile correspondence, with profile distances ranging from 0.107 to 0.309.

The smallest cluster differs substantially between the two formulations, with a profile distance of 5.034. Under IDEC, this cluster is smaller and has more negative standardized scores across all seven features.

Thus, the major behavioral and psychological cluster profiles are largely preserved, while the extreme low-profile minority region changes more substantially.

## Partition Agreement

The Adjusted Rand Index between trainable DEC and IDEC is:

**ARI = 0.6983**

This indicates substantial but incomplete agreement between the two partitions.

The result shows that reconstruction regularization changes a meaningful portion of individual cluster assignments while preserving considerable correspondence between the two clustering solutions.

## Latent Representation Change

Latent displacement is calculated as the Euclidean distance between each observation's initial pretrained latent representation and its final representation.

| Statistic | Trainable DEC | IDEC |
|---|---:|---:|
| Mean | 5.991 | 5.253 |
| Median | 5.837 | 5.063 |
| SD | 2.442 | 2.477 |
| Q25 | 3.826 | 3.080 |
| Q75 | 8.431 | 7.234 |
| Q90 | 8.859 | 8.628 |
| Q95 | 9.028 | 9.335 |
| Q99 | 10.200 | 10.780 |
| Maximum | 21.091 | 14.451 |

IDEC produced lower mean, median, and maximum latent displacement than trainable DEC.

The mean displacement decreased from 5.99 to 5.25, while the maximum decreased from 21.09 to 14.45.

These results indicate that reconstruction regularization moderated latent-space adaptation. However, substantial displacement remained, so the reconstruction constraint did not preserve the original latent representation completely.

## Minority-Cluster Membership Analysis

The smallest cluster showed a marked difference between the two formulations.

- Trainable DEC minority cluster: 929 observations
- IDEC minority cluster: 488 observations
- Shared observations: 488
- Trainable-DEC minority retained by IDEC: 52.53%
- Jaccard overlap: 0.5253

All 488 observations assigned to the IDEC minority cluster were already members of the trainable-DEC minority cluster.

The IDEC minority cluster had more negative mean standardized scores across all seven features, with a profile distance of 5.0335 relative to the trainable-DEC minority profile. Within-cluster variability was also lower across all seven features.

This indicates that IDEC primarily removed less-extreme observations from the expanded minority region while retaining its more extreme core.

## Diagnostic Interpretation

The IDEC diagnostic demonstrates that reconstruction regularization materially changes the trainable DEC solution.

Compared with trainable DEC, IDEC:

- reduces internal clustering-quality metrics;
- reduces mean and maximum latent displacement;
- reduces the size of the extreme minority cluster;
- retains all observations belonging to the IDEC minority cluster within the trainable-DEC minority group;
- preserves strong correspondence among the four major cluster profiles;
- produces a substantially different minority-cluster profile.

The results indicate that the stronger clustering metrics obtained by trainable DEC are accompanied by greater adaptation of the latent representation. Adding reconstruction regularization moderates this adaptation while preserving the main cluster structure.

The diagnostic therefore provides evidence that the clustering objective alone can substantially reshape the pretrained latent representation. Reconstruction regularization constrains this adaptation, although it does not eliminate it.

## Interpretation of the Clustering-Quality Trade-off

IDEC does not maximize the internal clustering metrics observed in this study. Its lower clustering scores are consistent with the additional reconstruction constraint, which limits the degree to which the latent representation can be optimized solely for cluster separation.

The purpose of this experiment is not to identify the formulation with the highest internal clustering scores. Instead, it evaluates the relationship between clustering quality, latent representation adaptation, and preservation of the input representation.

## Limitations of the Diagnostic

The IDEC experiment is a controlled diagnostic rather than an independently optimized comparison of DEC and IDEC.

The DEC hyperparameters were inherited from the previously selected configuration and were not independently re-tuned for the combined reconstruction-clustering objective. The reconstruction weight $\gamma=0.1$ was adopted from the original IDEC formulation rather than selected using the current dataset.

This design isolates the effect of adding reconstruction regularization under otherwise controlled conditions. Independent IDEC seed and sensitivity analyses may be used later to assess the stability of the final formulation.

## Outputs

The run directory contains:

- `meta.json`
- `train_kl.npy`
- `train_reconstruction.npy`
- `full_labels.npy`
- `q_full.npy`
- `Z_full.npy`
- `Z_initial.npy`
- `cluster_centers.npy`

The analysis directory contains:

- Training-loss figures
- Reconstruction-loss figure
- Quality-metric comparison
- Cluster-size comparison
- Cluster-profile comparison
- Profile matching results
- ARI comparison
- Representation-change summary
- Latent-displacement comparison
- Minority-membership analysis
