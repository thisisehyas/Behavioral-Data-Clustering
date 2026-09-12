# DEC Encoder Trainability Ablation

## 1. Purpose

This experiment evaluates the effect of encoder trainability on the final Deep Embedded Clustering (DEC) solution.

The experiment compares two conditions:

1. **Frozen encoder:** the encoder of the pretrained autoencoder remains fixed during DEC training.
2. **Trainable encoder:** the pretrained encoder is updated jointly with the DEC clustering layer.

The objective is to determine whether allowing the encoder to adapt during DEC training changes the latent representation, clustering quality, cluster structure, and partition membership.

The experiment also examines whether changes in the final clustering are accompanied by substantial movement of the pretrained latent representation.

## 2. Experimental Design

The two conditions use the same pretrained autoencoder and the same DEC configuration. The only experimental variable is encoder trainability.

### Fixed configuration

| Parameter                 |                             Value |
| ------------------------- | --------------------------------: |
| Dataset size              |                            99,997 |
| Input features            | 7 standardized composite features |
| Latent dimension          |                                 5 |
| Number of clusters (`k`)  |                                 5 |
| Student-t parameter (`α`) |                               0.5 |
| Learning rate             |                             0.003 |
| Update interval           |                         20 epochs |
| Batch size                |                               256 |
| Maximum epochs            |                                50 |
| K-Means `n_init`          |                                20 |
| Random seed               |                                42 |
| Dataset usage             |                      Full dataset |
| Pretrained encoder        |                   Same checkpoint |

No train/validation split or hyperparameter selection was performed during this ablation.

Both conditions independently initialize the DEC cluster centers using K-Means from the pretrained encoder representation. The frozen condition keeps the encoder parameters fixed, whereas the trainable condition updates the encoder parameters jointly with the clustering layer.

The selected DEC hyperparameter configuration was retained unchanged across both conditions so that the effect of encoder trainability could be examined under controlled settings.

## 3. Evaluation Criteria

The experiment evaluates five aspects of the two conditions.

### 3.1 Clustering quality

Internal clustering metrics are calculated in the final DEC latent representation (`Z_full`):

- Silhouette Score — higher is better
- Davies–Bouldin Index — lower is better
- Calinski–Harabasz Score — higher is better
- Mean assignment entropy — lower indicates more concentrated assignments
- Normalized mean entropy — mean entropy divided by `log(k)`

The standard deviation of assignment entropy is also reported to assess how uniformly assignment certainty is distributed across observations.

### 3.2 Cluster structure

Cluster sizes and seven-feature behavioral profiles are calculated using the original standardized composite features (`X_scaled`).

This separates evaluation of latent-space geometry from interpretation of the discovered behavioral groups.

### 3.3 Partition agreement

The Adjusted Rand Index (ARI) is calculated between the frozen and trainable partitions.

A contingency table is also examined to identify the main sources of membership reassignment.

### 3.4 Representation change

For the trainable condition, the initial pretrained representation (`Z_initial`) is compared with the final representation (`Z_full`).

For each observation:

$$
\Delta Z_i =
\left\|
Z_{i,\mathrm{final}} -
Z_{i,\mathrm{initial}}
\right\|_2
$$

The distribution of these displacements is summarized using descriptive statistics and quantiles.

### 3.5 Training behavior

The epoch-wise training KL loss is compared between the two conditions to characterize optimization behavior.

KL loss is treated as a measure of training dynamics rather than as a standalone criterion for selecting the preferable clustering solution.

# 4. Results

## 4.1 Clustering Quality

Allowing the encoder to train produced a substantial change in all evaluated internal clustering metrics.

| Metric               |   Frozen |     Trainable | Direction             |
| -------------------- | -------: | ------------: | --------------------- |
| Silhouette ↑         |   0.2358 |    **0.6189** | Strong improvement    |
| Davies–Bouldin ↓     |   1.0984 |    **0.5360** | Strong improvement    |
| Calinski–Harabasz ↑  | 33,874.7 | **213,778.8** | Strong improvement    |
| Mean entropy ↓       |   1.4455 |    **0.7718** | Strong improvement    |
| Normalized entropy ↓ |   0.8982 |    **0.4796** | Strong improvement    |
| Entropy SD           |   0.0650 |    **0.4342** | Greater heterogeneity |

The trainable encoder therefore produced a latent representation with substantially stronger internal separation and more concentrated cluster assignments.

The increase in entropy standard deviation indicates that this improvement in assignment certainty was not uniform across observations. Some observations remained considerably more uncertain than others.

These results show that encoder trainability substantially affects the DEC solution. The improved internal metrics indicate that the trainable encoder produces a latent representation with stronger separation and more concentrated assignments. However, these metrics alone cannot establish whether the resulting behavioral structure is more meaningful, which motivates the subsequent analysis of cluster profiles, partition agreement, and representation change.


## 4.2 Cluster Structure

### 4.2.1 Cluster sizes

The frozen condition contains one small minority cluster consisting of **489 observations (0.49%)**. The other four clusters contain between 20.5% and 28.3% of the dataset.

In the trainable condition, the corresponding small cluster contains **929 observations (0.93%)**, while the four major clusters contain between 22.2% and 27.2%.

| Measure               |          Frozen |       Trainable |
| --------------------- | --------------: | --------------: |
| Smallest cluster      |     489 (0.49%) |     929 (0.93%) |
| Largest cluster       | 28,258 (28.26%) | 27,174 (27.17%) |
| Maximum/minimum ratio |           57.79 |           29.25 |
| Clusters below 1%     |               1 |               1 |

Thus, encoder training does not eliminate the small minority subgroup. Instead, its size increases while the overall size imbalance is reduced.

### 4.2.2 Major cluster profiles

The seven-feature profiles of the four major clusters remain highly similar between the two conditions.

The Euclidean distances between corresponding profiles are:

| Frozen → Trainable | Profile distance |
| ------------------ | ---------------: |
| C1 → C1            |        **0.078** |
| C2 → C2            |        **0.115** |
| C3 → C3            |        **0.035** |
| C4 → C4            |        **0.088** |

The small distances indicate that the main behavioral patterns identified by the frozen encoder are largely preserved after encoder training.

For example, the corresponding clusters retain their characteristic combinations of Lifestyle, Communication, Responsibility, Openness, Interests, Emotional Stability, and Social Engagement. The feature-level changes for these four clusters are generally small, with most differences below 0.07 standardized units.

### 4.2.3 Minority subgroup

The small cluster is affected more strongly by encoder training.

Its profile changes from approximately **−4.25 to −6.27 standard deviations** across the seven features in the frozen condition to approximately **−2.68 to −4.23 standard deviations** in the trainable condition.

The corresponding profile distance is **5.09**, substantially larger than the distances observed for the four major clusters.

Despite this change in magnitude, the direction of the profile is preserved: the subgroup remains strongly below the population mean across all seven behavioral dimensions.

Thus, encoder training produces a **larger and less extreme version of the same globally low-profile subgroup**, rather than removing the subgroup or changing its qualitative direction.

### 4.2.4 Interpretation

The quality and profile results provide complementary information. The trainable encoder substantially improves the geometry of the latent clustering, while the seven-feature profiles of the four major groups remain highly similar.

This suggests that the improvement is not primarily associated with discovering completely different behavioral groups. Instead, encoder training appears to reorganize the latent representation so that largely similar behavioral structures become more clearly separated.

The minority subgroup undergoes a larger change in both size and profile magnitude and therefore requires separate consideration in the partition analysis.


## 4.3 Partition Agreement

The frozen and trainable conditions produce an **ARI of 0.7537**.

This indicates substantial but imperfect agreement between the two partitions. Encoder training therefore changes cluster membership in a meaningful way, but does not produce an entirely different partition.

The observation-level correspondence is particularly strong for several clusters:

| Frozen cluster | Percentage remaining in corresponding trainable cluster |
| -------------- | ------------------------------------------------------: |
| C0             |                                             **100.00%** |
| C1             |                                              **82.55%** |
| C2             |                                              **99.03%** |
| C3             |                                              **96.95%** |
| C4             |                                              **83.20%** |

All **489 observations** belonging to the small frozen cluster remain together in the corresponding trainable cluster.

The largest sources of reassignment are:

- **10.02% of frozen C1 → trainable C2**
- **10.04% of frozen C4 → trainable C2**

Smaller fractions are redistributed among the other major clusters.

The ARI therefore supports a **partially preserved partition with meaningful membership reassignment**. Together with the cluster-profile results, this suggests that encoder trainability refines and redistributes observations among largely consistent behavioral structures rather than producing a completely different set of groups.

The ARI is calculated directly from the original cluster labels and is invariant to cluster-label permutations; no manual label matching is required for this comparison.


## 4.4 Representation Change

The trainable encoder substantially changes the latent representation during DEC optimization.

For the 99,997 observations:

| Statistic          | Latent displacement |
| ------------------ | ------------------: |
| Mean               |           **5.991** |
| Median             |           **5.837** |
| Standard deviation |           **2.442** |
| Minimum            |               0.268 |
| Q25                |               3.826 |
| Q50                |               5.837 |
| Q75                |               8.431 |
| Q90                |               8.859 |
| Q95                |               9.028 |
| Q99                |              10.200 |
| Maximum            |              21.091 |

The central 50% of observations therefore have displacement between **3.826 and 8.431**.

The distribution indicates that representation adaptation is not restricted to a small number of observations. Most observations undergo moderate-to-large movement, with relatively few observations showing displacement below 2. The distribution also contains two prominent regions, approximately around 4 and 8–9, together with a smaller right tail extending beyond 10.

These results indicate that the encoder does not merely undergo minor parameter adjustments while preserving the pretrained representation. Instead, DEC substantially reshapes the latent space when encoder parameters are trainable.

This provides a direct link between the mechanism of the ablation and the observed clustering results: the improvement in internal clustering quality is accompanied by substantial adaptation of the latent representation.


## 4.5 Training Behavior

The optimization trajectories differ substantially between the two conditions.

### Frozen encoder

The training KL decreases from **0.03519** at epoch 1 to a minimum of **0.03405 at epoch 4**. It then remains close to this value through epoch 20.

At epoch 21, the KL increases to **0.06259**, after which it remains near 0.062–0.063 until epoch 40. Following the next target-distribution update at epoch 41, it increases to **0.06514** and remains close to 0.065 through the end of training.

The final KL is **0.06506**.

### Trainable encoder

The trainable condition starts at a lower KL of **0.01265** and decreases throughout the first interval, reaching a minimum of **0.00432 at epoch 18**.

At epoch 21, the KL increases to **0.03795**, followed by continued optimization that reduces it to **0.02612 by epoch 40**.

At epoch 41, the KL increases to **0.07200** and subsequently decreases to **0.04926 at epoch 50**.

The increases at epochs 21 and 41 correspond to the periodic target-distribution updates used by DEC. They do not by themselves indicate optimization failure.

The trainable condition therefore shows substantially greater continued adaptation throughout training, whereas the frozen condition reaches a comparatively stable trajectory early within each target-distribution interval.

The lower final KL of the trainable condition should not, however, be used independently to rank the two clustering solutions. KL is the optimization objective, and the two conditions differ precisely in their ability to modify the latent representation.


# 5. Integrated Interpretation

The encoder-trainability ablation demonstrates that allowing the pretrained encoder to update materially affects DEC behavior.

The trainable condition produces substantially stronger internal clustering metrics and lower assignment entropy, accompanied by considerable movement of the latent representation. At the same time, the four major behavioral cluster profiles remain highly similar to those obtained with the frozen encoder. The ARI of **0.7537** confirms that cluster membership changes are meaningful but that substantial correspondence with the original partition remains.

The main structural difference concerns the small globally low-profile subgroup. It increases from **0.49% to 0.93%** of the sample and becomes less extreme in standardized feature space, while retaining its below-mean direction across all seven features.

Taken together, the results indicate that encoder trainability primarily **reorganizes the latent geometry and redistributes observations within largely consistent behavioral structures**, rather than generating an entirely different behavioral partition. The training trajectories support this interpretation by showing continued optimization and substantial adaptation in the trainable condition.


# 6. Final Conclusion

The ablation supports the use of a **trainable pretrained encoder** for the final DEC configuration. Encoder freezing produces a substantially different latent solution and limits the extent to which DEC can adapt the representation to the clustering objective.

The trainable condition provides stronger internal separation and assignment certainty while retaining substantial correspondence with the behavioral structures identified by the frozen condition. Thus, encoder trainability is treated as a meaningful component of the final DEC model rather than as a minor implementation choice.

The frozen-encoder condition is retained as an **ablation reference**, while the trainable-encoder condition is used as the final DEC formulation for subsequent analyses and reporting.
