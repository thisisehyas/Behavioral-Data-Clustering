# Final DEC Stability Experiment

## 1. Objective

This experiment evaluates the sensitivity of the final Deep Embedded Clustering (DEC) solution to random initialization. The objective is to determine whether the selected DEC configuration produces a consistent clustering solution when only the random seed is changed.

The experiment uses the final DEC configuration selected for the canonical run:

- Number of clusters: (k=5)
- Student-t parameter: (alpha=0.5)
- Learning rate: (0.003)
- Batch size: (256)
- Update interval: (20) epochs
- Maximum epochs: (50)
- Optimizer: Adam
- Encoder: frozen pretrained encoder
- K-Means initialization: (n_init=20)
- Training data: full processed dataset
- Number of samples: (99,997)
- Input representation: 7 standardized interpretable features
- DEC latent dimension: 5

Five random seeds were evaluated: **42, 43, 44, 45, and 46**. Seed 42 was predefined as the canonical random seed and was included as one of the five independent stability runs.

The stability assessment combines four complementary analyses:

1. pairwise Adjusted Rand Index (ARI) between cluster assignments;
2. cluster-size comparison;
3. correspondence of cluster profiles across seeds;
4. clustering-quality and assignment-uncertainty metrics in the DEC latent space.

This combination was used because similar clustering-quality scores alone do not establish that different runs produce the same partition.


## 2. Experimental Protocol

Each seed was run independently using the same dataset, pretrained autoencoder, DEC architecture, hyperparameters, and training procedure. Only the random seed was changed.

Seed 42 was predefined as the canonical random seed and was included as one of the five stability runs. It was also used as the reference seed for cluster-profile correspondence.

For each run, the following outputs were saved:

- `Z_full.npy`: final DEC latent representation;
- `full_labels.npy`: final hard cluster assignments;
- `q_full.npy`: final soft cluster assignments;
- `cluster_centers.npy`: final cluster centers;
- `train_kl.npy`: KL-divergence trajectory during training;
- `meta.json`: configuration and run metadata.

The raw run outputs are stored under:

```text
experiments/final_stability/runs/
```

## 3.1 Partition Stability Analysis Using ARI

Pairwise ARI was calculated between the final cluster assignments of all five seeds. ARI was selected because it is invariant to permutations of cluster labels and therefore measures agreement between partitions rather than numerical cluster identifiers.

The resulting pairwise ARI matrix was:

| Seed   |     42 |     43 |     44 |     45 |     46 |
| ------ | -----: | -----: | -----: | -----: | -----: |
| **42** | 1.0000 | 0.9347 | 0.8980 | 0.5578 | 0.8906 |
| **43** | 0.9347 | 1.0000 | 0.9342 | 0.5793 | 0.9204 |
| **44** | 0.8980 | 0.9342 | 1.0000 | 0.5925 | 0.9517 |
| **45** | 0.5578 | 0.5793 | 0.5925 | 1.0000 | 0.5880 |
| **46** | 0.8906 | 0.9204 | 0.9517 | 0.5880 | 1.0000 |

Across all non-identical seed pairs, the mean ARI was **0.7847** with a standard deviation of **0.1687**, with values ranging from **0.5578 to 0.9517**.

Four seeds (42, 43, 44, and 46) showed high agreement, with pairwise ARI values between **0.8906 and 0.9517**. In contrast, seed 45 produced substantially lower agreement with every other seed, with ARI values between **0.5578 and 0.5925**.

Thus, the results do not support a claim of complete seed invariance. Instead, they indicate a stable solution for four seeds together with one substantially different solution.


## 3.2 Cluster-Size Stability Analysis

Cluster sizes were compared across runs. Because cluster labels are arbitrary, numerical cluster identifiers were not directly matched across seeds in this analysis.

The results were:

| Seed | Cluster sizes                                                                       |
| ---- | ----------------------------------------------------------------------------------- |
| 42   | 489 (0.49%), 28,258 (28.26%), 20,515 (20.52%), 25,483 (25.48%), 25,252 (25.25%)     |
| 43   | 26,802 (26.80%), 24,950 (24.95%), 27,327 (27.33%), 20,437 (20.44%), 481 (0.48%)     |
| 44   | 24,807 (24.81%), 21,278 (21.28%), 27,799 (27.80%), 484 (0.48%), 25,629 (25.63%)     |
| 45   | 20,901 (20.90%), 16,687 (16.69%), 12,937 (12.94%), 20,908 (20.91%), 28,564 (28.56%) |
| 46   | 21,406 (21.41%), 27,976 (27.98%), 23,673 (23.67%), 26,453 (26.45%), 489 (0.49%)     |

Seeds 42, 43, 44, and 46 consistently produced one very small cluster containing approximately **0.5% of the observations**. The numerical label of this cluster changed between runs, as expected from the arbitrary nature of cluster labels.

Seed 45 differed substantially. It did not produce a cluster of comparable size; its smallest cluster contained **12,937 observations (12.94%)**. Instead, all five clusters represented substantial portions of the dataset.

This result provides an initial explanation for the lower ARI values involving seed 45 and motivated the subsequent analysis of cluster-profile correspondence.

## 3.3 Cluster-Profile Correspondence Analysis

To determine whether differences between seeds were caused only by label permutation or represented genuine changes in the discovered behavioral structure, cluster profiles were compared across runs.

For each cluster, the mean of the seven original standardized interpretable features was calculated. Cluster correspondence was then established using one-to-one optimal assignment via the Hungarian algorithm, using Euclidean distance between the seven-dimensional cluster-mean profiles.

Cluster correspondence was assessed in the original seven-dimensional interpretable feature space, rather than in the learned latent space, to determine whether clusters obtained under different random seeds represented similar behavioral profiles.

### 3.3.1 Consistency across seeds 42, 43, 44, and 46
After accounting for arbitrary cluster labels, seeds 42, 43, 44, and 46 recovered highly similar cluster profiles.

The small cluster identified in seed 42 (C0) was matched to:

- seed 43: C4;
- seed 44: C3;
- seed 46: C4.

The profile distances from the seed-42 profile were:

- seed 42 → seed 43: **0.088**
- seed 42 → seed 44: **0.055**
- seed 42 → seed 46: **0.000**

The corresponding cluster sizes were also highly consistent:

- seed 42: **489 (0.49%)**
- seed 43: **481 (0.48%)**
- seed 44: **484 (0.48%)**
- seed 46: **489 (0.49%)**

The seed-42 profile of this subgroup was characterized by extremely low values across all seven standardized features, approximately **−4 to −6 standard deviations** from the respective feature means. Its repeated recovery across four independent seeds indicates that this subgroup is not specific to the canonical seed.

The remaining four clusters also showed generally small profile distances after optimal correspondence, indicating that the major structure of the behavioral space was reproduced across these four runs.

### 3.3.2 Difference of seed 45

Seed 45 did not recover the same small extreme-profile subgroup.

The optimal profile correspondence produced the following distances from the seed-42 reference clusters:

| Seed-42 cluster | Matched seed-45 cluster | Profile distance |
| --------------- | ----------------------- | ---------------: |
| C0              | C0                      |      **14.5069** |
| C1              | C3                      |           0.6396 |
| C2              | C2                      |           0.3906 |
| C3              | C1                      |           0.3850 |
| C4              | C4                      |           0.1772 |

The distance of **14.5069** between seed-42 C0 and its matched seed-45 cluster is substantially larger than the corresponding distances observed for the other seeds. This confirms that the difference involving seed 45 cannot be explained by a simple permutation of cluster labels.

Seed 45 therefore converged to a genuinely different partition of the data. In particular, it did not isolate the extreme minority subgroup consistently identified by seeds 42, 43, 44, and 46.

## 3.4 Clustering Quality and Assignment Uncertainty Analysis

Internal clustering-quality metrics were calculated using the **DEC latent representation (Z_full)**, which is the representation on which the DEC clustering operates. The same latent-space evaluation was used in the canonical final notebook.

The following metrics were calculated:

- Silhouette coefficient;
- Davies–Bouldin (DB) index;
- Calinski–Harabasz (CH) index;
- mean assignment entropy;
- standard deviation of assignment entropy;
- final training KL divergence.

The results were:

| Seed | Silhouette |     DB |        CH | Mean entropy | Final KL |
| ---- | ---------: | -----: | --------: | -----------: | -------: |
| 42   |     0.2358 | 1.0984 | 33,874.73 |       1.4455 | 0.065057 |
| 43   |     0.2347 | 1.1110 | 33,779.78 |       1.4487 | 0.065215 |
| 44   |     0.2358 | 1.0982 | 33,834.49 |       1.4469 | 0.065142 |
| 45   |     0.2012 | 1.3780 | 32,817.13 |       1.5260 | 0.066014 |
| 46   |     0.2362 | 1.0988 | 33,840.16 |       1.4479 | 0.065128 |

Seeds 42, 43, 44, and 46 produced highly similar quality metrics. Their Silhouette values ranged from **0.2347 to 0.2362**, DB from **1.0982 to 1.1110**, and CH from **33,779.78 to 33,874.73**. Their mean assignment entropy was also closely grouped between **1.4455 and 1.4487**.

Seed 45 showed a different pattern. Its Silhouette coefficient decreased to **0.2012**, while the DB index increased to **1.3780** and the CH score decreased to **32,817.13**. Its mean assignment entropy was also higher (**1.5260**), indicating greater assignment uncertainty under the same soft-clustering formulation.

The final KL divergence, however, remained close to the other runs:

- seeds 42, 43, 44, and 46: **0.065057–0.065215**
- seed 45: **0.066014**

Thus, the final KL divergence alone did not clearly distinguish the alternative seed-45 solution, whereas partition agreement, cluster profiles, internal clustering metrics, and assignment entropy did.

This indicates that a similar optimization objective value does not necessarily imply that two DEC runs have converged to the same clustering structure or to solutions with equivalent geometric properties.

## 4. Summary of Findings

The four analyses provide a consistent picture.

Seeds **42, 43, 44, and 46** converged to highly similar solutions. They produced:

- pairwise ARI values of approximately **0.89–0.95**;
- the same small minority subgroup containing approximately **0.5%** of the observations;
- highly similar cluster profiles after label matching;
- nearly identical Silhouette, DB, and CH values;
- similar assignment entropy;
- similar final KL divergence.

Seed **45** converged to a substantially different solution. It:

- had ARI values of only **0.56–0.59** against the other runs;
- did not isolate the approximately 0.5% minority subgroup;
- produced substantially different cluster-profile correspondence;
- had lower Silhouette and CH values;
- had a higher DB index;
- had higher mean assignment entropy;
- produced a similar, but slightly higher, final KL divergence.

The evidence therefore does not support describing the final DEC configuration as completely seed-invariant. Instead, the configuration demonstrates **predominant stability with a meaningful seed-sensitive alternative solution**.

The four consistent runs also provide support for the canonical seed-42 solution. In particular, seeds 43, 44, and 46 independently recovered the same small extreme-profile subgroup and similar major cluster profiles. Thus, the canonical result is not supported only by a single random initialization.

At the same time, seed 45 demonstrates that the optimization can reach an alternative partition under the same hyperparameters. This should be reported rather than treated as a simple label-permutation effect or omitted from the analysis.


## 5. Conclusion

The final DEC configuration exhibits **substantial but imperfect stability across random seeds**. Four of the five evaluated seeds recovered essentially the same clustering structure, including a consistent approximately 0.5% extreme-profile subgroup and similar major cluster profiles. One seed produced a distinct partition with weaker internal clustering quality and higher assignment uncertainty.

The stability analysis therefore supports the use of the seed-42 solution as a representative canonical solution, while indicating that DEC optimization under the selected configuration retains some sensitivity to random initialization.

The result also demonstrates the importance of evaluating stability using multiple complementary criteria. Partition agreement, profile correspondence, cluster-size structure, internal clustering metrics, and assignment uncertainty reveal differences between runs that are not apparent from the final KL objective alone.
