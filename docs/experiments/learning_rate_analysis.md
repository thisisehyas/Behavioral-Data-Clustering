# Learning Rate Sensitivity Analysis

## Objective

This experiment evaluates the sensitivity of Deep Embedded Clustering (DEC) to the learning rate used during optimization. The learning rate determines the size of parameter updates during training and can affect both the optimization trajectory and the resulting clustering solution.

A suitable learning rate should provide good clustering quality while remaining stable across different random initializations. Five learning rates were compared using internal clustering metrics, KL-divergence behavior, soft-assignment entropy, agreement between random seeds, and cluster-size statistics.

The analysis also examined whether the differences observed between random seeds were caused by the learning rate itself or were already present in the KMeans initialization. In addition, validation KL behavior was examined around the periodic target-distribution updates.


## Experimental Setup

### Dataset

The experiment used the roommate compatibility dataset, represented by seven standardized composite features. This compressed feature representation served as the input to DEC.

### Train/validation split

All runs used a fixed 80/20 train-validation split, generated with `split_seed = 42` (80,000 training samples, 20,000 validation samples). KMeans initialization and DEC optimization were performed only on the training split; the validation data was not used for optimization. Using the same split across all learning rates and random seeds ensures that differences between configurations are attributable to the learning rate and initialization rather than changes in the data.

### DEC architecture

DEC used a pretrained, frozen encoder that maps the seven input features to a five-dimensional latent representation, with five clusters and a Student-t distribution (`α = 1.0`) for the soft cluster assignments.

### Fixed hyperparameters

| Parameter                 |                     Value |
| -------------------------- | ------------------------: |
| Input dimension             |                         7 |
| Latent dimension            |                         5 |
| Number of clusters          |                         5 |
| Student-t parameter (`α`)   |                       1.0 |
| Update interval             |                        20 |
| Optimizer                   |                      Adam |
| Encoder                     | Frozen pretrained encoder |
| Maximum epochs               |                        50 |
| Batch size                   |                       256 |
| KMeans `n_init`              |                        20 |
| Train/validation split       |                     80/20 |
| Split seed                   |                        42 |

### Learning rates

Five learning rates were evaluated:

[
1e-4, 3e-4, 1e-3, 3e-3, 1e-2
]

The update interval was fixed at 20, based on the result of the earlier update-interval sensitivity experiment.

### Random seeds

Five random seeds were evaluated for each learning rate:

[
{0,1,2,3,4}.
]

giving 25 runs in total.

## Evaluation Criteria

### Clustering Quality

Three internal clustering metrics were used: the Silhouette Score, where higher values indicate better separation and cohesion; the Davies–Bouldin (DB) Index, where lower values indicate more compact and better-separated clusters; and the Calinski–Harabasz (CH) Score, where higher values indicate stronger between-cluster separation relative to within-cluster dispersion. Mean assignment entropy was also examined as a measure of confidence in the soft cluster assignments, with lower entropy indicating more concentrated assignments.

### Optimization and KL Behavior

Final training KL and final validation KL were recorded for each run, together with the epoch at which the minimum validation KL occurred. Since the DEC target distribution is updated periodically during training, local changes in KL around these updates are expected as part of normal optimization and do not, by themselves, indicate instability.

### Stability

Stability across random initializations was assessed using the pairwise Adjusted Rand Index (ARI) between validation cluster assignments from different seeds, the standard deviation of these pairwise values, ranked cluster-size statistics across seeds, and a post-hoc diagnostic comparing the initial KMeans assignments across seeds. The latter diagnostic was used to determine whether the distinct behavior of one particular seed arose during DEC optimization or was already present before training.

## Results

### Main Results Table

| Learning Rate | Train KL        | Validation KL   | Silhouette      | DB Index        | CH Score        | Entropy         | ARI             | Smallest Cluster | Largest Cluster |
| ------------: | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- | :---------------- | :--------------- |
|        0.0001 | 0.0674 ± 0.0019 | 0.0658 ± 0.0010 | 0.2264 ± 0.0151 | 1.1480 ± 0.1261 | 6641.09 ± 93.92 | 1.4468 ± 0.0363 | 0.7784 ± 0.1877 | 633.8 ± 1071.6   | 5381.0 ± 148.3  |
|        0.0003 | 0.0675 ± 0.0020 | 0.0658 ± 0.0011 | 0.2260 ± 0.0149 | 1.1472 ± 0.1306 | 6634.45 ± 90.27 | 1.4459 ± 0.0370 | 0.7816 ± 0.1878 | 653.0 ± 1110.5   | 5379.6 ± 97.1   |
|         0.001 | 0.0675 ± 0.0020 | 0.0658 ± 0.0011 | 0.2260 ± 0.0149 | 1.1475 ± 0.1305 | 6636.51 ± 91.52 | 1.4455 ± 0.0372 | 0.7842 ± 0.1863 | 652.4 ± 1109.3   | 5345.2 ± 91.0   |
|         0.003 | 0.0675 ± 0.0020 | 0.0658 ± 0.0011 | 0.2260 ± 0.0150 | 1.1482 ± 0.1309 | 6635.40 ± 93.48 | 1.4449 ± 0.0376 | 0.7858 ± 0.1910 | 655.6 ± 1115.7   | 5272.8 ± 79.2   |
|          0.01 | 0.0676 ± 0.0021 | 0.0657 ± 0.0011 | 0.2253 ± 0.0152 | 1.1479 ± 0.1374 | 6617.58 ± 88.61 | 1.4445 ± 0.0378 | 0.7274 ± 0.1691 | 691.4 ± 1187.8   | 5331.2 ± 354.3  |

The main clustering metrics are largely insensitive to the learning rate within the tested range. Silhouette, Davies–Bouldin, and Calinski–Harabasz scores remain similar from **`1e-4`** through **`3e-3`**, while the clearest deterioration appears at **`1e-2`**, which has the lowest mean Silhouette and CH scores and the highest variability in several cluster-size statistics.

The mean pairwise ARI provides a clearer distinction between the configurations. The lower learning rates produce ARI values of approximately 0.778–0.786, whereas **`1e-2`** decreases to 0.7274, indicating lower reproducibility at the highest learning rate. As discussed below, this comparison should be interpreted with some caution because part of the observed variability is associated with a specific seed rather than the learning rate itself.

Taken together, the results indicate a broad stable region between **`1e-4`** and **`3e-3`**, rather than a single sharply optimal learning rate.


### Training KL

Final training KL is very similar across all five learning rates:

* `1e-4`: 0.0674 ± 0.0019
* `3e-4`: 0.0675 ± 0.0020
* `1e-3`: 0.0675 ± 0.0020
* `3e-3`: 0.0675 ± 0.0020
* `1e-2`: 0.0676 ± 0.0021

`1e-4` gives the lowest numerical mean, but the difference from the other configurations is small relative to the variability across seeds. The training-KL curves show similar late-stage behavior across all learning rates, with differences mainly appearing early in training, when `1e-4` reduces KL more slowly than the larger learning rates. The curves also show local jumps associated with changes in the DEC target distribution; these should not be interpreted as instability without considering the update schedule. Final training KL alone provides little basis for selecting a learning rate.

### Validation KL

Final validation KL is also largely flat across learning rates:

* `1e-4`: 0.0658 ± 0.0010
* `3e-4`: 0.0658 ± 0.0011
* `1e-3`: 0.0658 ± 0.0011
* `3e-3`: 0.0658 ± 0.0011
* `1e-2`: 0.0657 ± 0.0011

`1e-2` gives the lowest numerical mean, but the difference is negligible and is not supported by the clustering-quality or stability results.

One methodological point is important here: the validation target distribution used in this KL calculation is constructed from the same validation predictions being evaluated. The validation predictions are first used to construct their own target distribution, after which the KL divergence is computed against that target. This is not a conventional validation loss evaluated against a fixed, external target. It was therefore treated as a diagnostic of DEC's soft-assignment behavior rather than an independent generalization metric and given less weight than the clustering-quality and seed-stability measures in the final learning-rate decision.

### Minimum Validation KL and Its Trajectory

The minimum validation KL separates cleanly into two regimes. For `1e-4`, `3e-4`, and `1e-3`, the minimum occurs at the first recorded epoch in all five runs. For `3e-3` and `1e-2`, every run instead reaches its minimum at a later epoch:

| Learning Rate | Runs with minimum after epoch 1 | Interpretation                  |
| ------------: | -------------------------------: | -------------------------------- |
|        `1e-4` |                               0/5 | Minimum at epoch 1 in every run |
|        `3e-4` |                               0/5 | Minimum at epoch 1 in every run |
|        `1e-3` |                               0/5 | Minimum at epoch 1 in every run |
|        `3e-3` |                               5/5 | Later minimum in every run      |
|        `1e-2` |                               5/5 | Later minimum in every run      |

These later minima are not isolated events: for **`1e-2`**, they occur at epochs 6, 9, 12, 17, and 17; for **`3e-3`**, at epochs 9, 12, 15, 17, and 17. None occurs at epoch 20, which is the first scheduled target-distribution update, so the shift cannot be explained solely by that update.

The results indicate two distinct validation-KL trajectories. At the lower learning rates, validation KL is lowest at the beginning of training and increases thereafter. At the higher learning rates, it fluctuates around its initial value and occasionally reaches lower values later in training. Examining the KL change around each target-distribution update more closely shows that these updates frequently coincide with an immediate increase in validation KL:


| Learning Rate | Mean change after target update | Updates increasing KL |
| ------------: | --------------------------------: | ----------------------: |
|        `3e-4` |                        +0.000120 |                    5/10 |
|        `1e-3` |                        +0.000201 |                    6/10 |
|        `3e-3` |                        +0.000377 |                    7/10 |
|        `1e-2` |                        +0.000363 |                    7/10 |

This effect is more common at the higher learning rates, but it should not be interpreted as evidence that target-distribution updates destabilize DEC. Since validation KL is computed against a target constructed from the current validation predictions, changes in this quantity partly reflect how the soft assignments and their derived target move relative to each other, rather than necessarily indicating degradation in the underlying clustering. The observed pattern suggests that the learning rate affects the shape of the validation-KL trajectory: lower learning rates show an early minimum, whereas higher learning rates allow small reductions later in training.

### Clustering Quality

Between `1e-4` and `3e-3`, the internal clustering metrics show very little sensitivity to learning rate.

Silhouette ranges narrowly from 0.2260 to 0.2264 across these four configurations, with the highest value at `1e-4`:

* `1e-4`: 0.2264 ± 0.0151
* `3e-4`: 0.2260 ± 0.0149
* `1e-3`: 0.2260 ± 0.0149
* `3e-3`: 0.2260 ± 0.0150
* `1e-2`: 0.2253 ± 0.0152

These differences are small and substantially overlap across seeds, so the Silhouette Score does not provide a strong basis for preferring one configuration.

The Davies–Bouldin Index shows a similar pattern. The lowest value occurs at **`3e-4`** (1.1472 ± 0.1306), while the other configurations range from 1.1475 to 1.1482, except **`1e-2`**, which gives 1.1479 ± 0.1374. These differences are again small relative to the cross-seed variation.

The CH score is highest at **`1e-4`** (6641.09 ± 93.92), with similar values for **`3e-4`**, **`1e-3`**, and **`3e-3`** (6634.45, 6636.51, and 6635.40, respectively). The lowest value occurs at **`1e-2`** (6617.58 ± 88.61).

In summary, the clustering geometry is largely unchanged by the learning rate within the lower range. The case against **`1e-2`** is not based on a large decrease in these internal metrics, but on the combination of slightly weaker clustering scores and the substantially lower reproducibility discussed below.


### Assignment Entropy

Mean assignment entropy shows the same insensitivity:

* `1e-4`: 1.4468 ± 0.0363
* `3e-4`: 1.4459 ± 0.0370
* `1e-3`: 1.4455 ± 0.0372
* `3e-3`: 1.4449 ± 0.0376
* `1e-2`: 1.4445 ± 0.0378

There is a small downward trend as learning rate increases, but it is much smaller than the seed-to-seed variation and does not provide a useful basis for selecting a learning rate.

### Stability Analysis

Pairwise ARI gives the clearest distinction between the candidate learning rates. It was computed over the ten unique seed pairs in the upper triangle of the 5×5 pairwise ARI matrix (self-comparisons and duplicate pairs excluded):

| Learning Rate |   Mean ARI | Std. ARI |
| ------------: | ---------: | -------: |
|        `1e-4` |     0.7784 |   0.1877 |
|        `3e-4` |     0.7816 |   0.1878 |
|        `1e-3` |     0.7842 |   0.1863 |
|        `3e-3` | **0.7858** |   0.1910 |
|        `1e-2` | **0.7274** |   0.1691 |

**`3e-3`** gives the highest mean, but also the highest standard deviation among the four lower learning rates. At face value, this appears inconsistent: the highest average is paired with the largest spread. The reason becomes clearer when the ten pairwise comparisons are examined.

The pairwise ARI heatmaps show that seeds 1–4 agree closely with each other at every learning rate, while seed 0 consistently produces a different partition. Four of the ten pairs therefore involve seed 0, while the remaining six compare seeds 1–4 with each other. Combining these two groups produces a mean and standard deviation that do not represent either group well and instead largely reflect the difference between them.

To determine whether the discrepancy associated with seed 0 arises during DEC optimization or was already present at initialization, the initial KMeans assignments were compared directly before DEC training. Agreement between seed 0 and seeds 1–4 at initialization was low (mean ARI = 0.598 ± 0.001), whereas seeds 1–4 showed much higher agreement with each other (mean ARI = 0.971 ± 0.014). The initial cluster sizes show a corresponding pattern: seed 0 produces five relatively substantial clusters, while seeds 1–4 consistently isolate a much smaller fifth cluster containing 426 training samples. Since KMeans initialization occurs before the learning-rate-dependent optimization stage, this pattern is identical across all five learning rates by construction.

Splitting the post-training pairwise ARI into the two groups provides a clearer comparison:


| Learning Rate | Seed 0 vs. Seeds 1–4 |           Seeds 1–4 |
| ------------: | ---------------------: | --------------------: |
|      `0.0001` |       0.5488 ± 0.0111 |       0.9314 ± 0.0105 |
|      `0.0003` |       0.5520 ± 0.0074 |       0.9347 ± 0.0129 |
|       `0.001` |       0.5569 ± 0.0090 |       0.9357 ± 0.0200 |
|       `0.003` |       0.5521 ± 0.0084 |   **0.9416 ± 0.0072** |
|        `0.01` |   **0.5240 ± 0.0216** |   **0.8631 ± 0.0369** |

The seed-0 comparison remains within a narrow range (0.52–0.56) at every learning rate, which is consistent with an initialization effect. These four pairs therefore contribute a roughly constant penalty to the raw statistic regardless of the learning rate. The comparisons among seeds 1–4 show greater variation, with **`3e-3`** standing out by achieving both the highest agreement (0.9416) and the smallest spread (± 0.0072) among all configurations below **`1e-2`**.

This also explains the apparent inconsistency in the raw ARI table. Because the seed-0 pairs remain at approximately the same distance from the seeds-1–4 pairs across learning rates, a configuration in which seeds 1–4 agree particularly closely, as observed at **`3e-3`**, produces a *wider* spread when the seed-0 pairs are included, rather than a narrower one. The high raw standard deviation at **`3e-3`** is therefore not evidence of instability. Instead, it results from averaging two groups of pairs separated by a large, approximately constant gap, and appears most clearly when agreement among the comparable seeds is strongest.

**`1e-2`** shows a different pattern. Agreement among seeds 1–4 decreases from above 0.93 at every lower learning rate to 0.8631 ± 0.0369. Since this decrease occurs within a group of seeds that otherwise shows close agreement across the lower learning rates, it cannot be explained by the seed-0 initialization effect alone. This provides evidence that **`1e-2`** reduces reproducibility relative to the lower learning-rate configurations.


### Cluster-Size Analysis

Cluster sizes were ranked within each run from smallest to largest before computing the statistics. The ranked positions do not necessarily correspond to the same semantic cluster across seeds.

The cluster-size structure is broadly similar across learning rates, with no evidence of systematic cluster collapse or major redistribution among the five clusters at any configuration.

The smallest cluster has a large standard deviation everywhere:

* `1e-4`: 633.8 ± 1071.6
* `3e-4`: 653.0 ± 1110.5
* `1e-3`: 652.4 ± 1109.3
* `3e-3`: 655.6 ± 1115.7
* `1e-2`: 691.4 ± 1187.8

This is consistent with the initialization-dependent seed-0 effect described above rather than a learning-rate-specific problem, since the same pattern appears across all of the lower learning rates.

The largest cluster is comparatively stable at the lower learning rates, and its standard deviation decreases steadily from `1e-4` to `3e-3`:

* `1e-4`: 5381.0 ± 148.3
* `3e-4`: 5379.6 ± 97.1
* `1e-3`: 5345.2 ± 91.0
* `3e-3`: 5272.8 ± 79.2

`1e-2` produces 5331.2 ± 354.3, with substantially more variability than any of the lower learning rates. This gives further evidence against `1e-2`, though it does not, on its own, distinguish clearly between the lower-learning-rate candidates.

## Diagnostic: Validation KL Around Target-Distribution Updates

A closer look at validation KL around each target-distribution update was performed to check whether the changes visible in the KL curves were systematically tied to these updates.

Target-distribution updates frequently coincide with an immediate rise in validation KL:

| Learning Rate | Mean change after target update | Updates increasing KL |
| ------------: | --------------------------------: | ----------------------: |
|        `3e-4` |                        +0.000120 |                    5/10 |
|        `1e-3` |                        +0.000201 |                    6/10 |
|        `3e-3` |                        +0.000377 |                    7/10 |
|        `1e-2` |                        +0.000363 |                    7/10 |

This effect occurs more frequently at the larger learning rates, but it should not be interpreted as evidence that target-distribution updates destabilize DEC. The validation target distribution is derived from the same predictions used to compute the KL value, so this metric is not equivalent to a conventional loss evaluated against a fixed target. Changes following target updates may reflect the changing relationship between the current predictions and their derived target rather than a genuine change in clustering quality. This diagnostic indicates that the learning rate affects how DEC responds to the iterative target-update procedure, rather than showing that these updates are themselves a source of instability.


## Discussion

DEC is relatively insensitive to the learning rate across the **`1e-4`**–**`3e-3`** range based on the main internal clustering metrics. Silhouette, Davies–Bouldin, and Calinski–Harabasz scores remain close across these configurations, with differences small relative to the variation across random seeds. Mean assignment entropy shows the same pattern and does not, by itself, favor any particular value.

The clearest distinction appears at **`1e-2`**. This learning rate produces the lowest mean pairwise ARI, lower agreement among seeds 1–4, greater variability in the largest cluster, and slightly weaker internal clustering metrics. These results provide sufficient grounds to exclude it from subsequent experiments.

The validation-KL analysis shows a difference between the lower and higher learning rates. At **`1e-4`**, **`3e-4`**, and **`1e-3`**, the minimum validation KL occurs at the first recorded epoch in every run, whereas **`3e-3`** and **`1e-2`** both reach later minima in every run, before the first target-distribution update at epoch 20. These later reductions are small and are often followed by increases. Since validation KL is calculated using a target derived from the same predictions being evaluated, it is not treated as a conventional independent validation loss. It was consequently given less weight in the final decision than the clustering-quality and reproducibility results.

The seed-level diagnostic explains much of the ARI variability observed above. Seed 0 already produces a distinct clustering solution during the KMeans stage, before DEC training begins, while seeds 1–4 show close agreement from the outset. This pattern is identical across learning rates and persists after training, indicating that the large variation in the smallest cluster is better explained by an initialization effect than by a learning-rate-specific failure.

Among the lower learning rates, the comparison restricted to seeds 1–4 provides a more informative measure of reproducibility because the seed-0 pairs introduce a roughly constant offset regardless of the learning rate. By this measure, **`3e-3`** gives the highest agreement (0.9416 ± 0.0072), with a substantially tighter spread than **`1e-3`** (0.9357 ± 0.0200). Its internal clustering metrics also remain essentially indistinguishable from those of **`1e-4`**, **`3e-4`**, and **`1e-3`**.

Taken together, the results support treating **`1e-4`** to **`3e-3`** as a stable operating region, while **`1e-2`** is excluded based on reproducibility. Within this region, **`3e-3`** provides the strongest reproducibility evidence without compromising clustering quality.


## Selected Learning Rate

**The selected learning rate for subsequent experiments is 0.003 (`3e-3`).**

## Rationale

### 1. Comparable clustering quality

At `3e-3`, clustering quality is essentially indistinguishable from that of the other lower learning rates: Silhouette Score 0.2260 ± 0.0150, DB Index 1.1482 ± 0.1309, and CH score 6635.40 ± 93.48. `1e-4` gives a marginally higher Silhouette and CH score, while `3e-4` gives the lowest DB Index. These differences are small relative to the cross-seed spread and do not indicate a meaningful advantage for these configurations.

### 2. Strongest agreement among comparable seeds

The raw mean pairwise ARI at `3e-3` is 0.7858, the highest among the tested learning rates. However, this value combines two different types of seed comparisons: those involving seed 0, which the initialization diagnostic shows starts from a different KMeans solution regardless of learning rate, and those among seeds 1–4, which are directly comparable across configurations.

Restricting the comparison to seeds 1–4 removes this confound. `3e-3` gives the highest agreement within this group (0.9416 ± 0.0072), improving on `1e-3` (0.9357 ± 0.0200) in both mean agreement and spread. The raw standard deviation at `3e-3` is nominally the highest among the lower learning rates (0.1910), but this results from the seed-0 confound described in the Stability Analysis section rather than from instability specific to this learning rate.

This provides the primary evidence for selecting `3e-3` as the most reproducible configuration within the stable lower-learning-rate range.

### 3. Exclusion of the highest learning rate

`1e-2` gives a substantially lower mean pairwise ARI (0.7274 vs. 0.7858 at `3e-3`), while agreement among seeds 1–4 decreases from 0.9416 ± 0.0072 at `3e-3` to 0.8631 ± 0.0369. This decline occurs within a group of seeds that otherwise agree closely at the other learning rates and is therefore not explained by the seed-0 confound. `1e-2` also shows greater variability in cluster sizes. Taken together, these results are sufficient to exclude `1e-2` from subsequent experiments.

### 4. Similar optimization behavior to the other lower learning rates

Final training and validation KL at `3e-3` (0.0675 ± 0.0020 and 0.0658 ± 0.0011, respectively) are close to those of the other lower learning rates. Its validation-KL trajectory differs somewhat from `1e-4`–`1e-3`, as its minimum occurs later in training rather than at epoch 1. However, these later reductions are small and do not result in a clear advantage in the final clustering metrics.

### 5. Stable cluster structure

The largest cluster at `3e-3` (5272.8 ± 79.2) is the most stable among the lower learning rates and is clearly more stable than at `1e-2` (5331.2 ± 354.3). The large variation in the smallest cluster is present at every lower learning rate and can be traced to the initialization-dependent seed-0 effect rather than to a specific instability at `3e-3`.

### 6. Summary

The initialization diagnostic shows that the learning rate is not responsible for the distinct seed-0 solution, which is already present before DEC training begins. After accounting for this effect, **`3e-3`** shows the most consistent agreement among seeds 1–4 while maintaining clustering quality comparable to the other configurations in the stable range.

The selected value is not the best-performing configuration on every individual metric. It was selected as the most defensible choice within a broad region of similarly performing configurations, based on clustering quality, reproducibility among comparable seeds, cluster-size stability, and the clear exclusion of **`1e-2`**.

## Limitations

Only five random seeds were evaluated for each learning rate. This provides a useful view of initialization-dependent variability, but does not fully characterize the range of possible random initializations.

Restricting the reproducibility analysis to seeds 1–4 removes the seed-0 confound, but this comparison is based on only six pairwise ARI values per learning rate. The resulting standard deviations are informative but not precise; a larger number of seeds would be required to characterize this variability more rigorously.

The learning rates were evaluated on a single dataset using a fixed DEC architecture and hyperparameter configuration. The selected value of **`3e-3`** may not generalize to other datasets, latent representations, or model configurations.

The internal clustering metrics describe geometric properties of the learned clusters and do not, by themselves, establish their semantic validity. This will be examined separately in the final DEC analysis using the selected configuration.

The pairwise ARI analysis depends on the fixed validation split and measures agreement between clustering solutions rather than agreement with external ground-truth labels.

The validation KL used here is not a conventional loss evaluated against an independently defined target, since its target distribution is constructed from the same validation predictions being evaluated. Validation-KL values should thus be interpreted as a diagnostic of soft-assignment behavior rather than as an independent measure of generalization.

Finally, the initialization diagnostic shows that KMeans initialization can substantially influence the final DEC solution. The seed-0 effect is present before DEC optimization and remains visible after training. Although this diagnostic explains a large part of the observed variation, it does not eliminate DEC's underlying dependence on initialization.


## Conclusion

This experiment evaluated five learning rates across five random seeds using internal clustering metrics, KL-divergence behavior, assignment entropy, seed agreement, and cluster-size statistics.

The main clustering metrics are largely insensitive to learning rate from **`1e-4`** to **`3e-3`**. Silhouette, Davies–Bouldin, and Calinski–Harabasz all remain within a narrow range, indicating that the geometric structure of the resulting clusters is not strongly affected by learning rate in this region.

**`1e-2`** shows clearly lower reproducibility than the other configurations. Its mean pairwise ARI decreases to 0.7274, compared with 0.7784–0.7858 for the lower learning rates, and agreement among seeds 1–4 also decreases substantially. On this basis, it was excluded from subsequent experiments.

The validation-KL analysis identifies two trajectory regimes. At **`1e-4`**, **`3e-4`**, and **`1e-3`**, the minimum occurs at the first recorded epoch in every run, whereas at **`3e-3`** and **`1e-2`**, every run reaches a later minimum before the first target-distribution update at epoch 20. These later minima are small and are often followed by increases. Thus, learning rate is interpreted here as affecting how DEC responds to its iterative optimization process, rather than as evidence that the higher learning rates provide better validation behavior.

The initialization diagnostic accounts for most of the seed-specific ARI variability. Seed 0 produces a substantially different clustering already at the KMeans stage, and this difference persists after DEC training at every learning rate. The large variance in the smallest cluster is therefore interpreted as an initialization-dependent structural difference rather than as general instability or a problem specific to any one learning rate.

Among the lower learning rates, **`3e-3`** gives the strongest agreement among seeds 1–4 after accounting for the seed-0 effect, while maintaining clustering quality comparable to the other configurations in this range. It was therefore selected as the default learning rate for the subsequent DEC experiments.

**Learning rate = `3e-3`** is adopted as the fixed learning rate for the next stage of the DEC sensitivity analysis.
