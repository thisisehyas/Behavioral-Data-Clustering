# Update Interval Sensitivity Analysis

## Objective

The objective of this experiment was to evaluate the sensitivity of Deep Embedded Clustering (DEC) to the frequency at which the target distribution is updated during training. The update interval controls how often the soft cluster assignments are used to recompute the target distribution that guides DEC optimization.

A suitable update interval should provide good clustering quality while maintaining stable behavior across random initializations. The experiment therefore compared several update intervals using clustering-quality metrics, validation behavior, agreement between random seeds, and cluster-size statistics.

The analysis was also designed to identify whether unusually large variability in cluster sizes was caused by general instability across random seeds or by a specific initialization-dependent clustering structure.

## Experimental Setup

### Dataset

The experiment was performed on the roommate compatibility dataset using seven standardized composite features. The compressed feature representation was used as the input to DEC.

### Train/validation split

A fixed 80/20 train-validation split was used for all runs. The split was generated with `split_seed = 42`, resulting in 80,000 training samples and 20,000 validation samples.

KMeans initialization and DEC optimization were performed using the training split only. Validation data was not used during optimization. The same split was used for every update interval and random seed to ensure that differences between configurations were attributable to the experimental conditions rather than changes in the data split.

### DEC architecture

The DEC model used a pretrained, frozen encoder. The input consisted of seven standardized composite features and was mapped to a five-dimensional latent representation.

The model used five clusters and the Student-t distribution with `α = 1.0` for computing the soft cluster assignments.

### Fixed hyperparameters

The following parameters were kept fixed throughout the experiment:

| Parameter                 |                     Value |
| ------------------------- | ------------------------: |
| Input dimension           |                         7 |
| Latent dimension          |                         5 |
| Number of clusters        |                         5 |
| Student-t parameter (`α`) |                       1.0 |
| Learning rate             |                      1e-3 |
| Optimizer                 |                      Adam |
| Encoder                   | Frozen pretrained encoder |
| Maximum epochs            |                        50 |
| Batch size                |                       256 |
| KMeans `n_init`           |                        20 |
| Train/validation split    |                     80/20 |
| Split seed                |                        42 |

### Update intervals

Five update intervals were evaluated:

[
{1, 5, 10, 20, 50}.
]

The update interval determines how frequently the DEC target distribution is recomputed during training.

### Random seeds

Five random seeds were evaluated for each update interval:

[
{0,1,2,3,4}.
]

This resulted in 25 completed runs in total.

## Evaluation Criteria

The update interval was evaluated using clustering quality, optimization behavior, agreement across random seeds, and cluster-size structure.

### Clustering Quality

Three standard internal clustering metrics were used:

* **Silhouette Score:** higher values indicate better separation and within-cluster cohesion.
* **Davies–Bouldin (DB) Index:** lower values indicate better clustering, with smaller values corresponding to more compact and better-separated clusters.
* **Calinski–Harabasz (CH) Score:** higher values indicate stronger between-cluster separation relative to within-cluster dispersion.

Mean assignment entropy was also examined as a measure of the confidence of the soft cluster assignments. Lower entropy indicates more concentrated assignments, while higher entropy indicates greater uncertainty.

### Stability

Stability across random initializations was assessed using:

* Pairwise Adjusted Rand Index (ARI) between the validation cluster assignments of different seeds.
* The standard deviation of the pairwise ARI values.
* Ranked cluster-size statistics across seeds.
* Post-hoc analysis of the smallest cluster to determine whether large size variability reflected general sample reassignment or a consistent structural difference between clustering solutions.

## Results

### Main Results Table

The aggregated results across the five random seeds are shown below.

|   Update Interval | Train KL        | Validation KL   | Silhouette      | DB Index        | CH Score         | Entropy         | ARI             | Smallest Cluster   | Largest Cluster   |
|------------------:|:----------------|:----------------|:----------------|:----------------|:-----------------|:----------------|:----------------|:-------------------|:------------------|
|                 1 | 0.0646 ± 0.0029 | 0.0637 ± 0.0013 | 0.1945 ± 0.0069 | 1.2381 ± 0.1491 | 5850.95 ± 170.89 | 1.4169 ± 0.0500 | 0.7424 ± 0.2326 | 556.6 ± 969.2      | 7765.4 ± 1029.8   |
|                 5 | 0.0673 ± 0.0030 | 0.0651 ± 0.0015 | 0.2138 ± 0.0117 | 1.1661 ± 0.1471 | 6290.19 ± 54.24  | 1.4289 ± 0.0448 | 0.7281 ± 0.1945 | 598.6 ± 1042.2     | 6565.8 ± 761.2    |
|                10 | 0.0675 ± 0.0023 | 0.0656 ± 0.0013 | 0.2231 ± 0.0143 | 1.1435 ± 0.1412 | 6567.33 ± 81.29  | 1.4392 ± 0.0401 | 0.7715 ± 0.2005 | 648.8 ± 1107.6     | 5725.2 ± 300.7    |
|                20 | 0.0675 ± 0.0020 | 0.0658 ± 0.0011 | 0.2260 ± 0.0149 | 1.1475 ± 0.1305 | 6636.51 ± 91.52  | 1.4455 ± 0.0372 | 0.7842 ± 0.1863 | 652.4 ± 1109.3     | 5345.2 ± 91.0     |
|                50 | 0.0258 ± 0.0004 | 0.0645 ± 0.0008 | 0.2292 ± 0.0135 | 1.1644 ± 0.1245 | 6713.02 ± 78.56  | 1.4576 ± 0.0331 | 0.8083 ± 0.1775 | 662.0 ± 1118.0     | 5462.2 ± 93.1     |

The results show a clear improvement in clustering quality when moving from an update interval of 1 to larger intervals. Silhouette increased from 0.1945 at interval 1 to 0.2292 at interval 50, while the DB Index decreased from 1.2381 to approximately 1.14–1.16. The CH score also increased substantially across the tested range.

The main trade-off occurs between intervals 20 and 50. Interval 50 gives the highest Silhouette, CH, and ARI values, whereas interval 20 provides nearly the same clustering quality while avoiding the schedule-dependent behavior observed in the training KL.

### Validation KL

Validation KL decreased substantially from update interval 1 to the larger intervals. The mean final validation KL values were:

* Interval 1: 0.0637 ± 0.0013
* Interval 5: 0.0651 ± 0.0015
* Interval 10: 0.0656 ± 0.0013
* Interval 20: 0.0658 ± 0.0011
* Interval 50: 0.0645 ± 0.0008

The minimum validation KL was not used as a selection criterion. An independent check of all 25 runs showed that the minimum validation KL occurred at epoch 1 in every run and corresponded exactly to the first recorded validation value. Since this value was effectively determined before meaningful DEC optimization had taken place and was nearly independent of the update interval for a given seed, it does not provide useful information for comparing update schedules.

Final validation KL was therefore considered more informative than the minimum validation KL.

The final training KL at interval 50 requires separate interpretation. At this interval, the target distribution is computed before the first epoch and remains fixed throughout the 50-epoch training run. The final training KL is consequently measured against an unchanged target distribution. In contrast, intervals 1–20 repeatedly recompute the target distribution during optimization. The substantially lower training KL at interval 50 is therefore a consequence of the different target-update schedule and should not be interpreted as evidence of superior optimization.

The final validation KL at interval 50 does not show the same anomalous behavior: its value is comparable to the other configurations and does not by itself justify selecting interval 50.

### Clustering Quality

Clustering quality improved consistently as the update interval increased.

The Silhouette Score increased from 0.1945 at interval 1 to 0.2138, 0.2231, 0.2260, and finally 0.2292 at intervals 5, 10, 20, and 50, respectively. Thus, intervals 10, 20, and 50 all produced substantially stronger clustering than intervals 1 and 5.

The Davies–Bouldin (DB) Index showed a similar improvement from interval 1 to interval 10, decreasing from 1.2381 to 1.1435. Since lower DB values indicate better clustering, interval 1 produced the weakest result according to this metric, while interval 10 achieved the best mean value. Intervals 20 and 50 produced values of 1.1475 and 1.1644, respectively, which remain close to the best result. Thus, the DB Index does not indicate a continued improvement beyond interval 10, although the differences among intervals 10, 20, and 50 are relatively small.

The CH score increased from 5850.95 at interval 1 to 6713.02 at interval 50. Intervals 10 and 20 produced strong values of 6567.33 and 6636.51, respectively.

These results indicate that intervals 10, 20, and 50 form the strongest group of configurations. Interval 20 provides a particularly balanced result across the three clustering metrics: it is close to the best Silhouette and CH values while also producing the second-best DB Index.

Assignment entropy also increased with the update interval. Mean entropy was 1.4169 at interval 1 and increased gradually to 1.4576 at interval 50. Thus, the configurations producing stronger internal clustering metrics also produced somewhat less concentrated soft assignments. This indicates that assignment confidence and geometric clustering quality do not optimize at the same update interval.

### Stability Analysis

Pairwise ARI showed that intervals 10, 20, and 50 produced stronger agreement between random seeds than intervals 1 and 5.

The mean pairwise ARI values were:

| Update interval | Mean ARI | Std. ARI |
| --------------: | -------: | -------: |
|               1 |   0.7424 |   0.2326 |
|               5 |   0.7281 |   0.1945 |
|              10 |   0.7715 |   0.2005 |
|              20 |   0.7842 |   0.1863 |
|              50 |   0.8083 |   0.1775 |

The ARI statistics were calculated from the ten unique seed pairs in the upper triangle of the 5×5 pairwise ARI matrix. Self-comparisons and duplicate pairs were excluded.

Interval 50 achieved the highest mean ARI and the lowest ARI standard deviation, indicating the strongest agreement among the five final clustering solutions. Interval 20 was the next strongest configuration, with a mean ARI of 0.7842 and a standard deviation of 0.1863.

A further diagnostic analysis showed that the ARI variability was not simply caused by random reassignment of individual observations. Seed 0 consistently produced a distinct clustering solution, while seeds 1–4 showed high mutual agreement. Examination of the initial KMeans assignments showed that this difference was already present before DEC optimization. Pairwise ARI between the initial KMeans solution of seed 0 and seeds 1–4 was approximately 0.60, whereas agreement among seeds 1–4 was approximately 0.95–0.99.

This indicates that part of the observed variability is inherited from the KMeans initialization rather than being introduced entirely by DEC optimization.

### Cluster-Size Analysis

Cluster sizes were sorted within each run before calculating the smallest, second-smallest, middle, second-largest, and largest cluster statistics. This ranking is appropriate for describing the distribution of cluster sizes, but the ranked positions do not necessarily correspond to the same semantic cluster across different seeds.

The smallest-cluster statistics showed unusually large standard deviations for all update intervals. For example, interval 20 produced a smallest cluster size of 652.4 ± 1109.3. However, inspection of the individual seeds showed that this variability was driven primarily by seed 0:

* Seed 0: 2871 samples
* Seed 1: 98 samples
* Seed 2: 98 samples
* Seed 3: 98 samples
* Seed 4: 97 samples

The same pattern appeared across all update intervals. Seeds 1–4 consistently produced a small cluster containing approximately 72–103 validation samples, while seed 0 produced a much larger smallest cluster of approximately 2500–2900 samples.

The small clusters produced by seeds 1–4 had near-perfect Jaccard overlap. At interval 20, for example, the Jaccard overlap between seeds 1–3 was 1.00 and between seed 4 and the other seeds was approximately 0.99. The corresponding overlap with seed 0 was 0.00.

Tracing the small subgroup from seeds 1–4 into seed 0 showed that most of these observations were assigned to one of seed 0's larger clusters. At interval 20, approximately 70% of the subgroup was assigned to seed 0's cluster 2, while the remainder was distributed among several other clusters.

This indicates that the large standard deviation of the smallest cluster is not evidence that the membership of this subgroup is randomly changing across seeds. Instead, the results suggest an initialization-dependent structural choice: seeds 1–4 consistently separate a compact subgroup as its own cluster, whereas seed 0 tends to merge that subgroup into a neighboring larger cluster.

This behavior is consistent across update intervals and agrees with the corresponding post-hoc analysis performed during the α sensitivity experiment.

## Additional Diagnostic Checks

### Check 1 — Minimum Validation KL

The `min_val_kl` metric was independently verified across all 25 update-interval runs. In every run, the minimum validation KL occurred at epoch 1 and was exactly the first recorded validation KL value.

This occurs because validation KL is evaluated after each training epoch, and the first recorded value represents the first post-training-epoch validation evaluation. Subsequent validation KL values were higher than this first value in all 25 runs.

The first validation KL was also effectively independent of the update interval for a given seed. Consequently, `min_val_kl` is not an informative criterion for selecting the update interval and should not be interpreted as the best validation KL achieved during the meaningful optimization trajectory.

Final validation KL and clustering-quality metrics were therefore used instead.

### Check 2 — Interpretation of Training KL at Update Interval 50

At update interval 50, the training target distribution is computed once before the first epoch and remains fixed throughout the 50-epoch training run.

Consequently, the final training KL at interval 50 is measured against an unchanged target distribution, unlike intervals 1–20, where the training target distribution is repeatedly recomputed during optimization.

The substantially lower final training KL observed at interval 50 therefore reflects the different target-distribution update schedule and is not directly comparable to the final training KL values of configurations with more frequent target updates.

### Check 3 — ARI Calculation

Pairwise Adjusted Rand Index was calculated between the validation cluster assignments of all five seeds for each update interval.

The complete 5×5 ARI matrix contains self-comparisons on the diagonal, for which ARI = 1. However, the reported mean and standard deviation were calculated exclusively from the upper triangle, giving exactly ten unique seed pairs per update interval.

No self-comparisons or duplicate seed pairs were included in the summary. The mean ARI was calculated using `np.mean()`, and the reported standard deviation used `np.std(..., ddof=0)`, corresponding to the population standard deviation of the ten pairwise ARI values.

### Check 4 — Seed-0 Initialization Effect

Initial KMeans results showed that seed 0 already produced a substantially different clustering solution from seeds 1–4.

Pairwise ARI between seed 0 and seeds 1–4 was approximately 0.60, whereas agreement among seeds 1–4 was approximately 0.95–0.99. Seed 0 also differed in its initial cluster-size structure and did not produce the very small fifth cluster observed for seeds 1–4.

The distinctive final clustering behavior of seed 0 is therefore at least partly inherited from the initial KMeans solution rather than being introduced entirely during DEC optimization.

DEC training can modify the degree of disagreement, but seed 0 should not be treated simply as an anomalous or erroneous run.

### Check 5 — Smallest-Cluster Variability

The unusually large variance in the smallest cluster size was found to be driven primarily by a single initialization-dependent structural difference rather than broad instability across all random seeds.

Across all tested update intervals, seeds 1–4 consistently produced a very small cluster containing approximately 72–103 validation samples, while seed 0 consistently produced a substantially larger smallest cluster containing approximately 2500–2900 samples.

The smallest clusters obtained by seeds 1–4 showed near-perfect Jaccard overlap, indicating that they correspond to essentially the same subgroup of observations. Seed 0 showed zero overlap with this subgroup, and tracing these observations into seed 0 showed that most were absorbed into one of its larger clusters rather than being randomly distributed across the clustering solution.

The large standard deviation of the smallest ranked cluster should consequently be interpreted as an initialization-dependent difference in cluster separation rather than general instability in cluster membership.

The observed behavior can be summarized as follows:

```text
                 Small subgroup
                      │
          ┌───────────┴───────────┐
          │                       │
       Seeds 1–4                Seed 0
          │                       │
   separate cluster       merge into larger
      ~72–103              cluster(s)
```

The same pattern was observed across the update-interval sweep and is consistent with the post-hoc finding from the α sensitivity experiment.

## Discussion

The experiment shows that very frequent target-distribution updates are not beneficial for this dataset. Update interval 1 produced the weakest clustering results across the main internal metrics, with the lowest Silhouette and CH scores and the highest DB Index. Its ARI variability was also relatively high.

Increasing the update interval to 5 improved the internal clustering metrics, although its mean pairwise ARI was slightly lower than that of interval 1. The strongest and most consistent region of the sweep was formed by intervals 10, 20, and 50.

Interval 50 achieved the highest Silhouette score, CH score, and mean pairwise ARI. However, the advantage over interval 20 was small for the clustering metrics, while interval 50 introduced an important interpretational issue in the training KL. Because the target distribution remains fixed for the entire training run at interval 50, its final training KL is not directly comparable with the values obtained under more frequent target updates.

Interval 20 provided a similar level of clustering quality without this extreme target-update schedule. It also showed strong agreement across seeds, a relatively low ARI standard deviation, and the best cluster-size balance among the tested configurations.

The cluster-size diagnostics further showed that the large variance in the smallest cluster was mainly attributable to the initialization-dependent behavior of seed 0 and was present across the update-interval settings. This issue therefore does not provide evidence that interval 20 itself is unstable.

Taken together, the results suggest that the update interval primarily affects the frequency with which the clustering target is allowed to adapt during optimization. Very frequent updates appear to produce weaker clustering structure, while moderate update intervals provide a better balance between target adaptation and optimization stability. Increasing the interval beyond 20 provides only modest improvements in the main clustering metrics.

## Selected Update Interval

**The selected update interval for subsequent experiments is 20.**

## Rationale

Update interval 20 was selected as the default configuration for the subsequent experiments based on the following considerations:

1. **Strong clustering quality.**
   Interval 20 achieved a Silhouette Score of 0.2260 ± 0.0149, a DB Index of 1.1475 ± 0.1305, and a CH score of 6636.51 ± 91.52. These values are substantially better than those obtained at intervals 1 and 5 and are close to the best values observed at interval 50.

2. **Good stability across seeds.**
   The mean pairwise ARI at interval 20 was 0.7842 ± 0.1863, indicating strong agreement across random initializations. Although interval 50 achieved a higher ARI, the difference is not large enough to outweigh the other considerations.

3. **Good cluster balance.**
   Interval 20 produced the most favorable cluster-size balance according to the ranked cluster-size analysis. Its largest cluster had a mean size of 5345.2 ± 91.0, substantially lower than the corresponding values at intervals 1 and 5, while the remaining major clusters were also relatively balanced.

4. **Avoidance of the interval-50 training-KL anomaly.**
   At interval 50, the target distribution remains unchanged throughout training. The resulting final training KL of 0.0258 ± 0.0004 is therefore not directly comparable with the training KL values from intervals 1–20. Interval 20 avoids this extreme schedule while retaining clustering performance close to interval 50.

5. **Diminishing returns beyond interval 20.**
   Increasing the interval from 20 to 50 improved the Silhouette score from 0.2260 to 0.2292 and the CH score from 6636.51 to 6713.02. These improvements are relatively small compared with the improvement obtained when moving from interval 1 to the moderate intervals. Interval 20 therefore provides a practical balance between clustering quality and update frequency.

6. **Consistency with diagnostic analysis.**
   The large smallest-cluster variance was shown to be primarily an initialization-dependent structural effect associated with seed 0 and not a general instability specific to interval 20.

The selected value of 20 is therefore not the single best value on every individual metric. It was chosen as the most balanced configuration across clustering quality, seed stability, cluster-size structure, and interpretability of the optimization behavior.

## Limitations

Several limitations should be considered when interpreting this experiment.

First, only five random seeds were evaluated for each update interval. This provides useful information about variability but does not fully characterize the distribution of possible initializations.

Second, the update intervals were evaluated on a single dataset and under a fixed DEC architecture and hyperparameter configuration. The selected value of 20 may not generalize to other datasets, latent representations, or model configurations.

Third, the internal clustering metrics evaluate geometric properties of the learned clustering and do not by themselves establish the semantic validity of the resulting clusters. Interpretation of the cluster profiles will be assessed in the final DEC analysis using the selected hyperparameter configuration.

Fourth, the pairwise ARI analysis is dependent on the fixed validation split and measures agreement between the obtained clustering solutions rather than agreement with external ground-truth labels.

Finally, the initialization-dependent behavior identified in seed 0 indicates that KMeans initialization can influence the final DEC solution. Although the diagnostic analysis clarified the source of the smallest-cluster variability, it does not eliminate the underlying dependence on initialization.

## Conclusion

The update-interval sensitivity experiment evaluated five target-distribution update frequencies across five random seeds using clustering quality, validation behavior, assignment confidence, clustering agreement, and cluster-size statistics.

Very frequent updates, particularly interval 1, produced weaker clustering quality. Intervals 10, 20, and 50 provided the strongest results across the main internal clustering metrics. Interval 50 achieved the highest Silhouette, CH, and ARI values, but its final training KL was affected by the fact that its target distribution remained fixed throughout training, making this metric unsuitable for direct comparison with more frequently updated configurations.

Interval 20 was selected as the default update interval because it provided strong clustering quality, high agreement across seeds, favorable cluster-size balance, and a more interpretable optimization schedule. The difference between intervals 20 and 50 was relatively small for the main clustering metrics, making the additional complexity in interpreting interval 50 unnecessary for the subsequent experiments.

The diagnostic analyses also established that the large variance in the smallest cluster was primarily caused by an initialization-dependent structural difference: seeds 1–4 consistently separated a compact subgroup, whereas seed 0 tended to merge the same subgroup into a larger cluster. This behavior was observed across update intervals and was already present at the initial KMeans stage.

Accordingly, **update interval = 20** was adopted as the fixed update interval for the next stage of the DEC sensitivity analysis.
