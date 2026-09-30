# IDEC Stability Analysis

This experiment evaluates the stability of the final Improved Deep Embedded
Clustering (IDEC) solution across multiple random seeds. The analysis examines
training behavior, internal clustering quality, cluster-size distributions,
partition agreement, cluster profiles, and latent representation changes.

## Research Question

Does the final IDEC formulation produce a consistent clustering solution
across different random initializations?

## Experimental Design

The final IDEC configuration was evaluated using five random seeds:

- 42
- 43
- 44
- 45
- 46

All other experimental parameters were kept fixed.

### Fixed Configuration

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
- Encoder: trainable pretrained encoder
- Decoder: trainable
- Dataset: full dataset
- Train/validation split: none

Only the random seed was varied between runs.

## Analysis

Stability was evaluated at several levels:

1. Training behavior
2. Internal clustering quality
3. Cluster-size distributions
4. Partition agreement
5. Cluster profile consistency
6. Latent representation displacement

Cluster labels were treated as arbitrary across independent runs. Hungarian
matching was therefore used when comparing cluster profiles across seeds.


## 1. Training Behavior

The training curves showed a consistent optimization pattern across all five
seeds. The KL loss decreased within each target-distribution update interval
but increased sharply immediately after the updates at epochs 21 and 41. The
loss subsequently decreased again within each interval.

The reconstruction loss followed a smoother trajectory, decreasing from
approximately 0.116–0.117 at epoch 1 to 0.082–0.084 at epoch 50 across the five
runs. No divergence was observed. Final KL losses ranged from 0.0640 to 0.0794,
while final reconstruction losses ranged from 0.0822 to 0.0836.

These results indicate consistent training behavior across random
initializations. The KL increases following target-distribution updates were
observed consistently across all runs and were followed by subsequent
decreases.


## 2. Internal Clustering Quality

Internal clustering metrics showed moderate variation across the five seeds.

- Silhouette: 0.386–0.473, mean 0.443 ± 0.035
- Davies–Bouldin: 0.763–0.865, mean 0.810 ± 0.039
- Calinski–Harabasz: 83,301–135,048, mean 118,939 ± 20,636
- Mean entropy: 1.025–1.084, mean 1.050 ± 0.024
- Normalized mean entropy: 0.637–0.674, mean 0.653 ± 0.015
- Entropy SD: 0.279–0.315, mean 0.300 ± 0.014

Seed 45 produced the lowest Silhouette score, highest Davies–Bouldin index,
lowest Calinski–Harabasz score, and highest normalized mean entropy among the
five runs. Seed 46 produced the most favorable values for most of the evaluated
metrics.

Despite this variation, the five runs remained within the same general range
of internal clustering quality.


## 3. Cluster-Size Stability

Four runs, corresponding to seeds 42, 43, 44, and 46, identified a small
cluster containing approximately 0.5% of the observations. The remaining four
clusters contained approximately 21.5–29.5% of the dataset in these runs.

Seed 45 produced a different cluster-size structure. Its five clusters were
more balanced, ranging from 17.7% to 21.2% of the observations, with no
comparable 0.5% minority cluster.

The identity of the smallest cluster differed between runs because cluster
labels are arbitrary and may be permuted during independent clustering. The
cluster-size analysis therefore focuses on the distribution of cluster sizes
rather than the numerical cluster labels.

The results indicate partial stability in cluster-size structure: a small
cluster was consistently identified in four of the five runs, while seed 45
produced a different distribution.


## 4. Partition Stability

Pairwise Adjusted Rand Index (ARI) was used to compare the observation-level
cluster assignments between runs.

Seeds 42, 43, 44, and 46 showed relatively high agreement, with pairwise ARI
values ranging from 0.810 to 0.880. In contrast, seed 45 showed substantially
lower agreement with the other four runs, with ARI values between 0.521 and
0.529.

Across all ten pairwise comparisons, the mean ARI was 0.7248 ± 0.1653, with a
median of 0.8325, a minimum of 0.5210, and a maximum of 0.8804.

The partition analysis therefore indicates partial but structured stability.
Four initializations converged to closely related partitions, whereas seed 45
produced a substantially different partition.


## 5. Cluster Profile Stability

Cluster profiles were compared after Hungarian matching to account for
arbitrary cluster-label permutations.

Seeds 42, 43, 44, and 46 showed strong profile consistency. Relative to seed 42,
the matched profile distances ranged from 0.054 to 0.299 across these runs.
The small extreme cluster identified in these four runs also showed highly
similar feature profiles, with its seven feature means remaining approximately
between −6.26 and −4.16.

Seed 45 showed greater profile differences. The largest difference occurred
for the cluster corresponding to the small extreme group in the reference
solution, with a profile distance of 3.820. The remaining matched profiles for
seed 45 had distances ranging from 0.106 to 0.588.

These results are consistent with the partition and cluster-size analyses:
the major cluster profiles are largely preserved across four runs, whereas
seed 45 produces a distinct cluster-level solution.


## 6. Latent Representation Stability

Latent representation displacement was measured relative to the initial
pretrained embedding for each run.

Mean displacement ranged from 4.99 to 5.31, with median values between 4.69
and 5.21. The standard deviation ranged from 2.40 to 2.52, while maximum
displacement ranged from 15.03 to 17.51.

Seed 45 showed the largest mean and median displacement, consistent with its
different clustering partition. However, the displacement distributions
remained broadly similar across all five runs, with substantial overlap in
their central ranges and upper tails.

These results indicate that the magnitude of latent-space adaptation was
generally consistent across random initializations. The displacement analysis
measures change from the common initial representation rather than direct
similarity between the final latent spaces of different runs.


## 7. Stability Assessment

The stability analysis shows a consistent pattern across the evaluated
levels. Training behavior and latent representation displacement were similar
across all five seeds. Internal clustering metrics also remained within a
similar range.

At the clustering-structure level, however, four runs (seeds 42, 43, 44, and
46) converged to closely related solutions, while seed 45 produced a distinct
cluster-size distribution, lower partition agreement, and larger differences
in cluster profiles.

The final IDEC solution can therefore be characterized as **partially stable
across random initializations**. The results do not support describing the
solution as fully stable, but they show that the principal clustering
structure is reproduced across four of the five evaluated seeds.

## Conclusion

The IDEC formulation exhibited consistent training behavior and broadly
similar latent representation changes across random seeds. Four of the five
runs produced closely related clustering structures, including similar major
cluster profiles and a recurring small extreme cluster. Seed 45 produced a
distinct partition and cluster-size structure, resulting in lower agreement
with the other runs.

Thus, the stability analysis indicates **partial rather than complete
reproducibility** of the final IDEC solution across random initializations.