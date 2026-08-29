# Alpha Sensitivity Analysis

## Objective

The objective of the alpha sensitivity analysis was to determine an appropriate value of the DEC clustering-layer parameter α for the final clustering configuration. Since α controls the shape of the soft assignment distribution, different values may produce different levels of assignment confidence, cluster separation, reproducibility, and optimization behavior.

The analysis was designed to avoid selecting α based on a single evaluation metric. Internal clustering quality, assignment behavior, KL optimization, reproducibility, and cluster-size structure were considered jointly. Particular attention was given to the disagreement between pairwise ARI and the internal clustering and optimization metrics, especially for α = 5.0.

## Experimental Setup

### Dataset

The experiment was conducted on the psychological, behavioral, and lifestyle dataset used throughout the DEC experiments. The dataset contains approximately 100,000 samples and was used without class labels during clustering.

### Train/validation split

The same train/validation split used in the preceding DEC experiments was retained. The split was generated with `SPLIT_SEED = 42` and a training ratio of 0.8.

### DEC architecture

The same DEC architecture and pretrained autoencoder initialization used in the preceding experiments were retained. K-Means initialization was performed before DEC optimization, and the encoder was kept frozen during the clustering optimization.

### Fixed hyperparameters

All hyperparameters other than α were kept fixed. The main training configuration used a batch size of 256, a maximum of 50 epochs, an update interval of 20 epochs, and the same learning rate selected in the preceding learning-rate experiment.

### Alpha values

Five α values were evaluated:

$$
\alpha \in \{0.1,\ 0.5,\ 1.0,\ 2.0,\ 5.0\}.
$$

### Random seeds

Five random seeds were evaluated for each α value. Seed-level results were retained to assess reproducibility and to determine whether observed differences were driven by individual initializations.

## Evaluation Criteria

### Clustering Quality

Clustering quality was evaluated using the Silhouette score, Calinski–Harabasz (CH) score, Davies–Bouldin (DB) index, and pairwise Adjusted Rand Index (ARI).

Silhouette and CH were used to assess intrinsic cluster separation and cohesion, while DB provided a complementary measure in which lower values indicate more compact and better-separated clusters. Pairwise ARI was used to measure agreement between clustering solutions obtained from different random seeds.

### Optimization and KL Behavior

Training and validation KL divergence were examined both at the end of training and throughout the 50 epochs. Minimum KL values and their trajectories were considered in addition to final KL values.

Because DEC periodically recalculates its target distribution, particular attention was given to the behavior around the scheduled target-distribution updates at epochs 20 and 40.

### Stability

Stability was assessed through seed-level ARI agreement, variation in clustering metrics across seeds, assignment entropy, and cluster-size distributions. Cluster-size imbalance was considered because highly confident and reproducible assignments can still correspond to an undesirable cluster structure.

## Results

### Main Results Table

The aggregated results show that no single α value dominates every evaluation criterion.

|   α | Train KL        | Validation KL   | Silhouette      | DB Index        | CH Score        | Entropy         | ARI             | Smallest Cluster   | Largest Cluster   |
|----:|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:-------------------|:------------------|
| 0.1 | 0.0733 ± 0.0010 | 0.0712 ± 0.0012 | 0.2299 ± 0.0148 | 1.1632 ± 0.1065 | 6717.67 ± 87.01 | 1.4640 ± 0.0268 | 0.7914 ± 0.1934 | 591.0 ± 986.0      | 5799.4 ± 66.9     |
| 0.5 | 0.0654 ± 0.0012 | 0.0636 ± 0.0007 | 0.2283 ± 0.0154 | 1.1524 ± 0.1144 | 6673.65 ± 97.59 | 1.4634 ± 0.0321 | 0.7874 ± 0.1929 | 600.6 ± 1006.7     | 5524.4 ± 73.7     |
| 1   | 0.0675 ± 0.0020 | 0.0658 ± 0.0011 | 0.2260 ± 0.0150 | 1.1482 ± 0.1309 | 6635.40 ± 93.48 | 1.4449 ± 0.0376 | 0.7858 ± 0.1910 | 655.6 ± 1115.7     | 5272.8 ± 79.2     |
| 2   | 0.0745 ± 0.0026 | 0.0729 ± 0.0013 | 0.2209 ± 0.0139 | 1.1496 ± 0.1539 | 6536.32 ± 78.71 | 1.4066 ± 0.0456 | 0.7977 ± 0.1827 | 756.6 ± 1317.2     | 5487.0 ± 447.1    |
| 5   | 0.0860 ± 0.0023 | 0.0875 ± 0.0008 | 0.2157 ± 0.0128 | 1.1702 ± 0.1701 | 6416.09 ± 80.20 | 1.3174 ± 0.0578 | 0.8124 ± 0.1623 | 710.2 ± 1223.4     | 5713.2 ± 542.2    |

The Silhouette and CH values at α = 0.5 and α = 1.0 were very close to those reported for the neighboring configurations, while α = 0.1 provided the highest values for both metrics. The DB index favored α = 1.0, although the difference from α = 0.5 was small. In contrast, α = 0.5 provided the lowest final training and validation KL.

The ARI results favored larger α values, with α = 5.0 obtaining the highest mean ARI. This disagreement with the internal clustering and KL metrics required additional diagnostics before selecting the final α.

### Training KL

Training KL showed a strong dependence on α during the early stages of optimization. Before the first target-distribution update, α = 5.0 produced the lowest training KL, followed by α = 2.0, α = 1.0, α = 0.5, and α = 0.1.

This apparent advantage of α = 5.0 did not persist after the target-distribution updates. Its KL increased substantially after the updates at epochs 20 and 40, resulting in the highest final training KL among the tested configurations.

In contrast, α = 0.5 maintained comparatively favorable KL behavior after the target-distribution updates and finished with the lowest final training KL, at 0.0654 ± 0.0012. α = 1.0 followed at 0.0675 ± 0.0020.

Thus, the low initial KL obtained with α = 5.0 does not indicate consistently better optimization. Its behavior is better characterized as a strong dependence on the changing DEC target distribution.

### Validation KL

Validation KL provided a clearer distinction between the candidate configurations. α = 0.5 achieved the lowest final validation KL, 0.0636 ± 0.0007, followed by α = 1.0 at 0.0658 ± 0.0011.

The validation KL increased for α = 0.1 and α = 2.0 and was substantially higher for α = 5.0, which reached 0.0875 ± 0.0008.

The advantage of α = 0.5 was not caused by the anomalous seed 0. When seeds 1–4 were considered separately, validation KL remained lower for α = 0.5 (0.0638 ± 0.0006) than for α = 1.0 (0.0663 ± 0.0007). This indicates that the KL advantage of α = 0.5 is consistent across the comparable seeds.

### Clustering Quality

The internal clustering metrics showed a different trend from ARI. Silhouette and CH decreased progressively as α increased. The direction of this trend remained consistent after excluding seed 0, indicating that it was not caused by a single initialization.

The best Silhouette and CH values were obtained at α = 0.1, while α = 5.0 produced the weakest values. The DB index favored α = 1.0, with a mean of 1.1482 ± 0.1309, although the difference from α = 0.5 and α = 2.0 was small.

When the comparison was restricted to seeds 1–4, α = 0.5 had a Silhouette of 0.2360 ± 0.0009 compared with 0.2336 ± 0.0009 for α = 1.0. The corresponding DB values were 1.0952 ± 0.0064 and 1.0827 ± 0.0068, respectively. Thus, α = 0.5 had a small advantage in Silhouette, while α = 1.0 had a consistent advantage in DB.

These results do not identify a single internally optimal α. Instead, they place α = 0.5 and α = 1.0 in the main candidate region, while providing evidence against the larger α values.

### Assignment Entropy

Mean assignment entropy decreased as α increased:

* α = 0.1: 1.4640 ± 0.0268
* α = 0.5: 1.4634 ± 0.0321
* α = 1.0: 1.4449 ± 0.0376
* α = 2.0: 1.4066 ± 0.0456
* α = 5.0: 1.3174 ± 0.0578

The results indicate that larger α values produce sharper assignments. However, lower entropy was not interpreted as an independent indicator of better clustering.

Across the experiment, entropy was negatively correlated with both inter-seed ARI (Pearson r = −0.632) and cluster-size imbalance (Pearson r = −0.621). Thus, lower entropy was associated with more reproducible assignments but also with greater cluster-size imbalance.

This relationship was particularly clear for α = 5.0. Among seeds 1–4, α = 5.0 produced lower entropy and slightly higher seed agreement than α = 2.0, but also a higher imbalance ratio. The results therefore suggest that the reduction in entropy at larger α values reflects sharper assignments rather than an unambiguous improvement in clustering structure.

### Stability Analysis

The seed-level diagnostics revealed a persistent initialization effect. Initial K-Means solutions for seeds 1–4 showed high agreement, with a mean pairwise ARI of 0.971 ± 0.014, whereas seed 0 showed substantially lower agreement with the other seeds (ARI = 0.598 ± 0.001).

This difference was present before DEC optimization and was independent of α. After DEC optimization, seeds 1–4 continued to show high agreement, with mean pairwise ARI values of approximately 0.942–0.949 across the tested α values. Seed 0 remained a clear outlier for every α.

Although α = 5.0 increased the agreement of seed 0 with the other seeds to approximately 0.614 ± 0.005, it did not eliminate the seed-specific difference. Consequently, the higher aggregate ARI of α = 5.0 cannot be interpreted simply as evidence of superior reproducibility.

### Cluster-Size Analysis

Cluster-size analysis confirmed that seed 0 represented a clearly different initialization. Seeds 1–4 consistently produced a very small fifth cluster containing approximately 97–99 samples, whereas seed 0 produced substantially larger minimum clusters.

When seeds 1–4 were considered separately, the minimum cluster size remained essentially unchanged across α. Therefore, the apparently larger minimum cluster size observed for some configurations, particularly α = 2.0, was primarily attributable to seed 0 rather than to an α-dependent improvement.

The largest cluster showed greater variation at α = 2.0 and α = 5.0. For seeds 1–4, α = 5.0 produced an imbalance ratio of 60.70 ± 0.94, compared with 58.24 ± 0.80 for α = 2.0. The smallest clusters remained approximately 98–99 samples, while the largest cluster under α = 5.0 reached approximately 5,979 samples.

The results indicate that increasing α does not resolve the persistent small-cluster structure and that the sharper assignments at α = 5.0 are accompanied by a more concentrated cluster structure.

## Diagnostic: Validation KL Around Target-Distribution Updates

The epoch-wise KL trajectories showed discontinuities around epochs 20 and 40 across the tested α values. These epochs correspond to the configured update interval of 20 epochs, at which the DEC target distribution is recalculated.

The changes were therefore not interpreted simply as optimization instability. They are consistent with the expected effect of changing the target distribution.

The α = 5.0 configuration nevertheless showed a clearly different response. Across all five seeds, validation KL increased substantially around the first target update. The jump from the epoch immediately preceding the update to the update point was approximately 0.009–0.011 across seeds. At the second update, the changes were smaller but the validation KL remained in a substantially higher regime.

This behavior was consistent across seeds, indicating that the different KL trajectory of α = 5.0 is systematic rather than caused by a single seed.

The diagnostic therefore provides additional evidence against α = 5.0. Its low pre-update KL does not persist as the target distribution evolves, and its higher post-update KL is not accompanied by better internal clustering quality.

## Discussion

The alpha sensitivity analysis produced a clear disagreement between different evaluation criteria. Larger α values, particularly α = 5.0, produced sharper assignments and higher pairwise ARI. However, increasing α was also associated with decreasing Silhouette and CH scores, greater cluster-size concentration, and less favorable KL behavior.

The α = 5.0 configuration illustrates this trade-off most clearly. It achieved the highest mean ARI (0.8124 ± 0.1623) and the lowest ARI standard deviation, but it simultaneously produced the worst Silhouette, CH, and DB values, the highest final training and validation KL, the lowest assignment entropy, and the greatest cluster-size imbalance among the comparable seeds. The additional KL diagnostic showed that this behavior was consistent across seeds and became particularly pronounced after the target-distribution updates.

For this application, reproducibility alone is not sufficient to establish that a clustering solution is desirable. A highly reproducible solution can still consistently represent an undesirable or overly concentrated cluster structure. Similarly, lower assignment entropy indicates sharper assignments but does not establish that the resulting clusters are more meaningful.

The remaining comparison is therefore primarily between α = 0.5 and α = 1.0. These configurations provide similar internal clustering quality, while differing in their secondary strengths. α = 1.0 provides a lower DB index and somewhat better cluster-size balance, whereas α = 0.5 provides lower training and validation KL and a slightly higher Silhouette score. The validation-KL advantage of α = 0.5 remains after excluding seed 0, confirming that it is not explained by the initialization outlier.

The balance advantage of α = 1.0 was treated as a genuine consideration rather than ignored. However, its improvement was accompanied by a measurable deterioration in validation KL without a corresponding improvement in Silhouette or CH.

## Selected α

**α = 0.5 was selected as the final value.**

## Rationale

### 1. Comparable clustering quality

α = 0.5 maintains clustering quality comparable to α = 1.0 and the other lower α values. Its Silhouette score is slightly higher than that of α = 1.0 when the comparable seeds are considered, while its CH performance remains close. Although α = 1.0 provides a lower DB index, this advantage is not accompanied by a corresponding improvement across the other internal metrics.

### 2. Strongest agreement among comparable optimization criteria

α = 0.5 provides the lowest final training KL and validation KL among all tested configurations. Its validation-KL advantage remains when seed 0 is excluded, with 0.0638 ± 0.0006 for α = 0.5 compared with 0.0663 ± 0.0007 for α = 1.0.

This makes the optimization advantage of α = 0.5 independent of the initialization outlier identified in the reproducibility analysis.

### 3. Avoidance of the high-α behavior observed at α = 2.0 and α = 5.0

The larger α values showed progressively weaker Silhouette and CH performance. α = 5.0 additionally exhibited substantially higher KL after target-distribution updates, lower entropy, and greater cluster-size concentration.

These results indicate that the sharper assignments obtained with large α values do not translate into a generally better clustering solution.

### 4. Favorable assignment behavior without excessive sharpening

α = 0.5 has an assignment entropy of 1.4634 ± 0.0321, close to α = 0.1 and higher than α = 2.0 and α = 5.0. Given the observed negative relationship between entropy and cluster-size imbalance, the more moderate assignment sharpness of α = 0.5 was preferred over the increasingly concentrated assignments produced by larger α values.

### 5. Consideration of cluster balance

α = 1.0 provides a measurable advantage in cluster balance. Among seeds 1–4, its imbalance ratio is 54.26 ± 0.35 compared with 56.65 ± 0.99 for α = 0.5. Its largest cluster is also smaller, while the smallest cluster size is essentially the same.

This advantage was not disregarded. It represents the main argument in favor of α = 1.0. However, both configurations retain the same small-cluster behavior, and the balance improvement is accompanied by higher validation KL and slightly lower Silhouette. Since the purpose is not simply to maximize cluster balance but to obtain meaningful and reasonably balanced clusters, α = 0.5 provides the better compromise across the evaluated criteria.

### 6. Summary

The final choice of α = 0.5 was not based on a single best metric. α = 0.1 provides the strongest Silhouette and CH values, α = 1.0 provides the lowest DB index and somewhat better cluster balance, and α = 5.0 provides the highest ARI and sharpest assignments. However, α = 0.5 provides the strongest combination of favorable validation and training KL, comparable internal clustering quality, moderate assignment sharpness, and avoidance of the increasingly concentrated and update-sensitive behavior observed at larger α values.

The selection therefore represents a trade-off between clustering quality, optimization behavior, reproducibility, and cluster structure rather than optimization of any single metric.

## Limitations

The alpha sensitivity analysis was performed over five random seeds and five α values. The persistent seed-specific behavior indicates that initialization can influence the resulting cluster structure, although the main comparison between α = 0.5 and α = 1.0 remains consistent across the comparable seeds.

The evaluation relies primarily on internal clustering metrics, assignment consistency, KL behavior, and cluster-size structure. Since the dataset does not contain ground-truth cluster labels, external validation of the semantic or psychological meaning of the resulting clusters remains limited.

The observed cluster-size imbalance, particularly the persistent small cluster among seeds 1–4, also remains an important limitation of the final clustering solution. The alpha experiment shows that changing α does not eliminate this structure.

The additional comparisons performed after excluding seed 0, including Silhouette, Davies–Bouldin, and cluster-size imbalance, are based on four seeds rather than the full five-seed set. These comparisons were used to assess whether the observed differences were driven by the initialization outlier, but the reduced number of seeds limits the strength of conclusions based on these comparisons.

## Conclusion

The alpha sensitivity analysis showed that α has a measurable effect on clustering quality, assignment sharpness, cluster-size structure, and KL optimization. No tested value was uniformly optimal across all criteria.

α = 5.0 was excluded despite its highest ARI because its stronger inter-seed agreement was accompanied by poorer internal clustering metrics, substantially higher final KL, stronger sensitivity to target-distribution updates, lower assignment entropy, and greater cluster-size imbalance.

The final comparison between α = 0.5 and α = 1.0 showed a genuine trade-off. α = 1.0 provided better DB and more balanced cluster sizes, whereas α = 0.5 provided lower training and validation KL and slightly better Silhouette performance while avoiding excessive assignment sharpening. The validation-KL advantage of α = 0.5 was also maintained after excluding the seed-specific initialization outlier.

Based on the joint evaluation of clustering quality, optimization behavior, reproducibility, assignment entropy, and cluster structure, **α = 0.5 was selected for the final DEC configuration**.
