# Final DEC k-Sensitivity Experiment

## 1. Purpose

This experiment evaluates the sensitivity of the final Deep Embedded Clustering (DEC) solution to the number of clusters, (k). The main objective is to determine whether the selected value of (k=5) provides a reasonable balance between latent-space clustering quality, assignment certainty, cluster granularity, and interpretability.

The experiment also examines whether structures identified at (k=5), particularly a small minority subgroup, remain present when the number of clusters is increased to (k=6).

## 2. Experimental Design

The final DEC configuration was kept fixed while varying only the number of clusters.

| Parameter                      | Setting                           |
| ------------------------------ | --------------------------------- |
| Number of clusters             | (k in {2,3,4,5,6})           |
| Random seed                    | 42                                |
| Input features                 | 7 standardized composite features |
| Input dimension                | 7                                 |
| Latent dimension               | 5                                 |
| Encoder                        | Pretrained and frozen             |
| Student-t parameter \alpha | 0.5                               |
| Learning rate                  | 0.003                             |
| Update interval                | 20 epochs                         |
| Batch size                     | 256                               |
| Maximum epochs                 | 50                                |
| K-Means initialization         |(n_init=20)            |
| Dataset                        | Full dataset                      |
| Data split                     | None                              |

Each value of k was evaluated independently using K-Means initialization followed by DEC optimization. No parameter other than k was changed between runs.

The tested range was selected based on the preliminary K-Means sensitivity analysis, which indicated stronger internal validation for k=2–3 and provided motivation to examine k=5 as a more granular alternative.


## 3. Evaluation Criteria

The solutions were evaluated using four internal clustering measures:

* **Silhouette coefficient:** higher values indicate better within-cluster cohesion and between-cluster separation.
* **Davies–Bouldin (DB) index:** lower values indicate better separation relative to within-cluster dispersion.
* **Calinski–Harabasz (CH) index:** higher values indicate greater between-cluster dispersion relative to within-cluster dispersion.
* **Normalized assignment entropy:** lower values indicate more concentrated soft assignments relative to the number of clusters.

Normalized entropy was used rather than raw entropy because raw entropy naturally increases with the number of possible clusters and is not directly comparable across different k values.

Final DEC KL divergence was recorded for completeness but was not used as a criterion for selecting k. Since the DEC objective changes with the number of clusters, its final KL values are not directly comparable as a standalone measure of the preferred k.

Cluster sizes and feature-level cluster profiles were also examined to assess balance, behavioral granularity, and interpretability.


## 4. Latent-Space Quality Results

| k | Silhouette ↑ |       DB ↓ |         CH ↑ | Normalized Entropy ↓ |
| ----: | -----------: | ---------: | -----------: | -------------------: |
| **2** |   **0.3146** |     1.1917 | **58,291.9** |           **0.8736** |
|     3 |       0.2138 |     1.4675 |     41,589.3 |               0.9270 |
|     4 |       0.2304 |     1.3156 |     38,120.9 |               0.9402 |
| **5** |       0.2358 | **1.0984** |     33,874.7 |           **0.8982** |
|     6 |       0.2089 |     1.1939 |     31,342.3 |               0.9157 |

### Interpretation

The results show a trade-off between geometric separation and cluster granularity.

The k=2 solution provides the strongest geometric separation. It has the highest Silhouette and CH values, the lowest normalized entropy, and a relatively low DB index. If geometric compactness and separation were the only objectives, k=2 would be the strongest choice.

However, increasing k from 2 to 3 does not improve the internal metrics. Silhouette decreases from 0.3146 to 0.2138, DB increases from 1.1917 to 1.4675, CH decreases, and normalized entropy becomes less favorable. Thus, the k=3 solution does not provide a strong metric-based improvement over k=2.

The transition from k=4 to k=5 is more favorable. Silhouette increases slightly from 0.2304 to 0.2358, DB decreases from 1.3156 to 1.0984, and normalized entropy decreases from 0.9402 to 0.8982. The k=5 solution therefore achieves the lowest DB value in the tested range and has relatively favorable assignment certainty.

Increasing k further to 6 does not improve the clustering metrics. Silhouette decreases to 0.2089, DB increases to 1.1939, CH decreases, and normalized entropy becomes less favorable. This provides no evidence that additional fragmentation beyond five clusters improves the latent-space structure.

These results do not establish k=5 as a universal metric optimum. Instead, they support k=5 as a reasonable compromise between geometric quality and the more granular behavioral structure required by the study. The k=2 solution is geometrically stronger but provides only a coarse partition, whereas k=5 provides additional structure without an improvement from further increasing k to 6.


## 5. Cluster-Size Analysis

| k | Cluster-size range | Min/Max ratio | Clusters <1% |
| ----: | ------------------ | ------------: | -----------: |
|     2 | 48.33%–51.67%      |          1.07 |            0 |
|     3 | 28.94%–39.91%      |          1.38 |            0 |
|     4 | 20.62%–28.11%      |          1.36 |            0 |
|     5 | 0.49%–28.26%       |         57.79 |            1 |
|     6 | 0.49%–26.06%       |         52.75 |            1 |

The k=2, k=3, and k=4 solutions produce relatively balanced partitions, with no cluster containing less than 1% of the observations.

At k=5, one cluster contains only 489 observations (0.49%), while the other four clusters contain between 20.52% and 28.26% of the dataset. The resulting imbalance is substantial. This cluster was therefore treated as a small minority cluster rather than being labeled an outlier cluster based on size alone.

A similar pattern appears at k=6, where one cluster contains 494 observations (0.49%) and the remaining clusters contain between 12.25% and 26.06% of the observations.

The recurrence of a subgroup of approximately 0.5% at both k=5 and k=6 suggests that the structure may persist across neighboring values of k. This was examined further using both cluster profiles and observation-level membership overlap.


## 6. Cluster Profile Analysis

Cluster profiles were calculated using the seven standardized interpretable input features.

### k=2

The two clusters are large and approximately balanced. They form opposing behavioral profiles. One group has relatively higher Lifestyle and Communication scores and lower Responsibility, Openness, Interests, Emotional Stability, and Social Engagement scores. The second group exhibits approximately the reverse pattern.

This solution has the strongest geometric separation but represents the behavioral space at a coarse level.

### k=3

Three interpretable profiles emerge, including a relatively mixed group, a group characterized by higher Lifestyle and Communication but lower Interests and Emotional Stability, and a group with higher Responsibility, Openness, Interests, Emotional Stability, and Social Engagement.

Despite this additional structure, the internal quality metrics are substantially weaker than those for k=2. Thus, k=3 does not provide a compelling improvement.

### k=4

The four clusters are all substantial and provide distinct behavioral profiles. The transition from k=4 to k=5 was examined using profile-level matching.

Three of the four k=4 profiles remain very similar at k=5, with profile distances of 0.058, 0.052, and 0.040. The remaining matched profile has a larger distance of 0.224.

This indicates that the k=5 solution does not represent a complete reorganization of the behavioral structure observed at k=4. Instead, much of the existing structure is preserved while additional structure is introduced.

### k=5

Four substantial clusters contain 28.26%, 20.52%, 25.48%, and 25.25% of the observations. A fifth cluster contains 489 observations (0.49%).

The minority cluster has strongly negative scores across all seven standardized features:

* Lifestyle: −6.269
* Communication: −5.135
* Responsibility: −6.249
* Openness: −5.861
* Interests: −4.254
* Emotional Stability: −4.895
* Social Engagement: −5.456

The pattern is therefore not driven by a single feature. It represents a globally low-profile subgroup across the compressed feature space. The subgroup is described as a small, globally low-profile cluster rather than an outlier cluster, since clustering alone does not establish whether these observations should be considered statistical or behavioral outliers.

### k=6

The small subgroup appears again with 494 observations (0.49%). Its profile is almost identical to the corresponding k=5 subgroup, with a profile distance of only 0.0652 and feature-level differences of at most approximately 0.041.

One other major profile is also preserved closely, with a profile distance of 0.0436. The remaining population, however, is divided into additional clusters.

Thus, increasing k from 5 to 6 preserves some existing structures but introduces further fragmentation without improving the internal quality metrics.

Profile matching across different values of k was used only as a descriptive comparison. Matching cluster profiles does not by itself establish that the same observations belong to the corresponding clusters.


## 7. Observation-Level Minority-Cluster Overlap

Because profile similarity does not establish membership consistency, the smallest k=5 and k=6 clusters were compared directly at the observation level.

| Measure                      |     Result |
| ---------------------------- | ---------: |
| k=5 minority size        |        489 |
| k=6 minority size        |        494 |
| Shared observations          |        488 |
| Union                        |        495 |
| k=5 recovered at k=6 | **99.80%** |
| k=6 recovered at k=5 | **98.79%** |
| Jaccard similarity           | **0.9859** |
| Binary minority-vs-rest ARI  | **0.9928** |

The two minority clusters share 488 observations. Thus, 99.80% of the k=5 minority cluster is retained at k=6, while 98.79% of the k=6 minority cluster was already present at k=5. The Jaccard similarity of 0.9859 and binary ARI of 0.9928 indicate very strong observation-level consistency.

This result complements the profile analysis. The subgroup is not only similar in size and feature profile across k=5 and k=6; its actual membership is also almost completely preserved.


## 8. Final Interpretation and Decision

The k-sensitivity analysis indicates that the choice of k affects both the geometric quality and the granularity of the resulting DEC partition.

The k=2 solution provides the strongest geometric separation according to Silhouette and CH and also has the lowest normalized entropy. However, it produces only two broad behavioral groups. The k=3 solution does not improve the internal metrics and therefore does not provide a strong alternative.

The k=4 solution provides a useful intermediate level of granularity. Moving from k=4 to k=5 preserves most of the major profile structures while introducing a small, globally low-profile subgroup. At k=5, the DB index is the lowest among all tested values, and normalized entropy is more favorable than at k=3, k=4, and k=6.

Increasing the number of clusters to k=6 does not improve the internal quality measures. Instead, it further fragments the substantial portion of the dataset while preserving the small minority subgroup almost exactly. The 99.80% observation-level recovery and 0.9859 Jaccard similarity show that this subgroup is highly persistent between k=5 and k=6.

Based on these results, k=5 was retained as the final clustering configuration. This decision is not based on a claim that k=5 is optimal according to every internal metric. Rather, it represents a reasonable compromise between geometric separation, assignment certainty, behavioral granularity, and structural persistence. The results also suggest that the behavioral structure in the dataset is not composed of sharply separated groups, but contains overlapping and continuous patterns that can be represented at different levels of granularity.

The final k-selection should therefore be understood as a research-design choice supported by multiple forms of evidence rather than as the unique mathematical optimum identified by a single clustering metric.
