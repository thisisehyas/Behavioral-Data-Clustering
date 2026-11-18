# Clustering Psychological and Behavioral Data: From Classical Methods to Deep Embedded Clustering

_A two-part machine learning project, comprising a bachelor's thesis and a research investigation, with roommate compatibility as a motivating case study._

## 1. Overview

This repository documents unsupervised-learning research conducted on a large-scale dataset of psychological, behavioral, and lifestyle survey responses, roughly 100,000 individuals described across 50 numerical features. The central question driving this research is methodological: can unsupervised learning identify behavioral structure in high-dimensional survey data that is both meaningful and _reliable_, and how sensitive is that structure to the modeling choices (representation, initialization, training regime, evaluation protocol) used to obtain it?

The research began with a personal observation: there is no scientifically grounded way to choose a roommate. Universities and shared-housing platforms that attempt any form of matching typically rely on a limited number of superficial preferences while ignoring the deeper psychological and behavioral traits that may have a greater influence on day-to-day compatibility. That gap is what first motivated this research.

However, what started as an applied question evolved substantially as the work progressed. This repository presents a set of interpretable cluster profiles extracted from a large real-world dataset. Its second and more extensive stage also investigates the reliability of these profiles: whether they remain consistent across clustering families, random initializations, encoder-training regimes, and evaluation spaces.

The research is organized into two stages:

- **Part I** (bachelor's thesis) applies classical unsupervised clustering methods to the dataset and builds an exploratory pairwise compatibility-scoring system based on the resulting clusters.
- **Part II** (research investigation) shifts the focus from an applied tool to a methodological question. It introduces Deep Embedded Clustering (DEC) and its reconstruction-regularized variant (IDEC), and evaluates the DEC family, together with the classical K-Means baseline, using a more demanding set of methodological analysis.

The two stages share the same dataset, research domain, and motivating question. However, Part II is best understood as a substantial deepening of method and experimental rigor rather than a direct technical continuation of the thesis pipeline.

## 2. Research Stages

### Part I — Bachelor's Thesis: _Modeling Roommate Compatibility Through Psychological and Behavioral Data_

#### Objective

To examine whether individuals with different psychological, behavioral, and lifestyle profiles could be grouped into behaviorally coherent clusters using classical unsupervised learning, and whether a quantitative pairwise compatibility score could be meaningfully derived from that cluster structure.

#### Methods

- **Preprocessing:** EDA, data cleaning, missing-value handling, outlier inspection, and feature standardization. The original 50 features were additionally compressed into seven interpretable composite dimensions (group means), used alongside the raw feature set in different experiments.
- **Clustering algorithms**, applied to multiple feature representations (PCA components, the top-10 individually most informative features, and the composite features):
  - K-Means
  - Spectral Clustering
  - DBSCAN
  - Hierarchical Clustering (both Agglomerative and Divisive)
  - Kernel K-Means (RBF approximation)
- **A compatibility-scoring system**, combining cosine similarity (weighted 0.7) with cluster-membership similarity (weighted 0.3), rescaled to a 0–4 range and calculated for approximately 500,000 individual pairs.
- **Evaluation:** Silhouette Score, Davies–Bouldin Index, and Calinski–Harabasz Index, applied consistently across algorithms and representations.
- **Visualization and behavioral cluster profiling**

#### Key Results

- Clustering on the composite (grouped) features consistently performed better than clustering on raw or PCA-reduced features, across several tested algorithms.
- Spectral Clustering on composite features and DBSCAN on the full feature set achieved the strongest internal cluster separation observed in this stage (Silhouette ≈ 0.71 and ≈ 0.69, respectively).
- A divisive hierarchical pipeline applied to the full dataset identified seven interpretable behavioral clusters. These ranged from one highly coherent, well-separated subgroup (Silhouette ≈ 0.78) to several moderately overlapping groups.
- Kernel K-Means was applied within individual hierarchical branches to investigate non-linear substructure. It largely reproduced the groupings identified by the hierarchical pipeline, providing initial evidence that the recovered structure was not specific to a single clustering method.
- The compatibility-scoring system identified a small group of individuals with structurally low compatibility (133 of 100,000, with an average top-5 match score below 2.5). These individuals were not distributed randomly across clusters: one cluster contained a disproportionately large number of them. They were characterized by comparatively low emotional stability, responsibility, or social engagement.
- The thesis received the highest possible grade (20/20).

**Relevant notebooks:**

- notebooks/thesis_kmeans_clustering.ipynb
- notebooks/thesis_comparative_clustering.ipynb
- notebooks/thesis_clustering_and_compatibility_scoring.ipynb

---

### Part II — Research Extension: _Deep Embedded Clustering for Behavioral Data: Evaluation, Stability, and Representation Analysis_

#### Objective

This stage uses the same dataset, the same composite behavioral dimensions, and the same underlying motivation as the thesis, but extends the scope and experimental analysis. Part II investigates deep clustering through a systematic set of experiments, examining hyperparameter sensitivity, sensitivity to random initialization, encoder-training regimes, and reconstruction regularization. It also compares classical and deep clustering methods under controlled evaluation conditions, investigates how evaluation space influences clustering conclusions, analyzes the extent to which deep clustering preserves the structure of the original behavioral feature space, and assesses the stability and interpretability of the resulting cluster profiles.

#### Methods

- A **canonical DEC training pipeline**, based on a pretrained autoencoder (architecture 7→32→16→5) and a KL-divergence-based self-training clustering layer. The pipeline standardized across subsequent experiment (fixed batch size, fresh K-Means initialization per run, and a consistent separation between tuning and final-fit stages).
- **Systematic hyperparameter sensitivity analysis** covering the Student-t parameter (α), the learning rate, and the target-distribution update interval, each evaluated across five random seeds.
- A **direct, partition-level random-seed stability analysis**, using pairwise Adjusted Rand Index across five seeds. This provides a more direct assessment of reproducibility than metric variance alone.
- A **cluster-number (k) sensitivity sweep** from k = 2 to k = 6, considering clustering quality, assignment certainty, behavioral granularity, and the persistence of a small minority subgroup.
- An **encoder-trainability ablation**, comparing a frozen pretrained encoder with a jointly fine-tuned (trainable) encoder under otherwise identical DEC configurations.
- An **IDEC (Improved Deep Embedded Clustering) diagnostic**, introducing a reconstruction-regularization term to investigate whether it limits the representation distortion caused by unconstrained encoder fine-tuning.
- A **controlled comparison in a common evaluation space** of frozen DEC, trainable DEC, IDEC, and K-Means. All four models were evaluated using identical internal clustering metrics in the original seven-dimensional feature space.
- An **exact, full-dataset representation-preservation analysis** — pairwise-distance correlation and k-nearest-neighbor preservation between the original feature space and each model's learned latent space, computed over all ≈5 billion unique observation pairs without any sampling.
- A **structured literature review** of roughly 17 papers across six thematic areas (seed and initialization stability, representation learning and reconstruction regularization, evaluation-space dependence, minority-cluster robustness, hyperparameter sensitivity, and cluster-number selection). The review was used to place the findings in the context of existing deep clustering research.
- **Cluster profiling and visualization**

#### Key Findings

- **Evaluation-space dependence.** When evaluated using identical internal metrics in the same original seven-feature space, K-Means outperformed all tested deep-clustering variants (frozen DEC, trainable DEC, and IDEC) on all three metrics. The large internal-metric advantages typically reported for deep clustering were observed only when each model was evaluated in its own learned latent space. This provides direct empirical evidence that the apparent superiority of deep-clustering can depend substantially on the evaluation space.
- **Partial reproducibility across random seeds.** A direct, partition-level comparison across seeds showed that the DEC family was only partially reproducible on this data. Four of the five seeds converged to a similar overall partition, with the same small, extreme behavioral subgroup recurring across these runs. The remaining seed produced a substantially different result.
- **Representation distortion and its partial mitigation.** An exact, full-dataset analysis of structure preservation showed that unconstrained encoder fine-tuning (trainable DEC) substantially distorted the original behavioral feature space, preserving as little as 1.8% of original nearest-neighbor relationships. Reconstruction regularization (IDEC) moderated this distortion, with a greater effect on local neighborhood structure than on global distance relationships. The frozen-encoder configuration also showed meaningful distortion, despite the encoder remaining unchanged during clustering. This indicates that the pretrained autoencoder itself introduces some representation change, independently of the clustering objective.
- **Final model selection.** Based on these findings, rather than clustering-quality scores alone, IDEC was selected as the final model. It improved on frozen DEC's clustering objective, moderated the representation distortion observed under unconstrained fine-tuning, and converged to a minority-subgroup structure consistent with that of the frozen baseline.
- **Final cluster structure.** The selected model identified five behavioral clusters across the dataset's seven composite dimensions: four larger, behaviorally distinguishable but partially overlapping groups, and one small (≈0.5%) subgroup with consistently and extremely low scores across every behavioral dimension.

**Notebooks, experiments, and documentation:**

- notebooks/research_initial_dec.ipynb
- notebooks/research_final_dec.ipynb
- notebooks/research_final_idec.ipynb
- src/dec/
- final_dec_results/
- final_idec_results/
- experiments/
- analysis/
- embeddings/
- docs/

## 3. Dataset

Both stages of this research use the same underlying dataset which contains 100,000 individuals, each described by 50 numerical features on a 0–5 scale, covering domains such as lifestyle habits, cleanliness, social activity, personality traits, and personal interests. The data was collected through self-reported responses to a questionnaire designed around validated psychological instruments, including the [IPIP Big-Five Factor Markers](https://openpsychometrics.org/tests/IPIP-BFFM/) from the [International Personality Item Pool](https://ipip.ori.org/new_ipip-50-item-scale.htm). The raw dataset originates from a public GitHub repository: [Link](https://github.com/Pradyumna-yes/Roommate-Pairing-ML-Model).

To reduce dimensionality and improve interpretability, the 50 original features were grouped into seven composite behavioral dimensions. Each dimension was represented by the mean of its constituent features:

| Composite Dimension                | General Concept                                                               |
| ---------------------------------- | ----------------------------------------------------------------------------- |
| Lifestyle                          | Sleep patterns, eating habits, hygiene, and related daily routines            |
| Communication and Interaction      | Listening skills, conflict-resolution ability, participation in social events |
| Responsibility and Organization    | Work ethic, punctuality, attention to detail, reliability                     |
| Openness and Core Beliefs          | Religious beliefs, cultural values, curiosity                                 |
| Interests and Activities           | Sports, food, art, music, film, reading, and similar interests                |
| Emotional Regulation and Stability | Anger and stress management, anxiety levels, compassion                       |
| Social Engagement and Behavior     | Cooperation, trust in others, altruism, enthusiasm for social gatherings      |

**Usage across the two stages:** Part I used both the original 50-feature set (for PCA-based and top-feature-based clustering experiments) and the composite 7-dimension representation. Part II used the standardized 7-dimension composite representation exclusively, as the sole input to the autoencoder and to every subsequent deep-clustering experiment.

## 4. Repository Structure

```text
.
├── dataset/
│   ├── primary_dataset.xlsx
│   ├── compressed_features.xlsx
│   ├── top5_matches.xlsx
│   └── X_scaled.csv
│
├── notebooks/
│   ├── thesis_kmeans_clustering.ipynb
│   ├── thesis_comparative_clustering.ipynb
│   ├── thesis_clustering_and_compatibility_scoring.ipynb
│   ├── research_initial_dec.ipynb
│   ├── research_final_dec.ipynb
│   └── research_final_idec.ipynb
│
├── models/
│   └── autoencoder_weights.pth
│
├── src/
│   └── dec/
│       ├── autoencoder.py
│       ├── clustering_layer.py
│       ├── dec.py
│       ├── target_distribution.py
│       ├── train_dec.py
│       ├── training.py
│       ├── utils.py
│       └── assets.py
│
├── experiments/
│   ├── alpha_sensitivity/
│   ├── learning_rate/
│   ├── update_interval/
│   ├── final_dec_stability/
│   ├── final_dec_k_sensitivity/
│   ├── encoder_ablation/
│   ├── idec_diagnostic/
│   ├── idec_stability/
│   ├── baseline_comparison/
│   ├── representation_preservation/
│   └── val_train_splits/
│       └── [experiment-specific scripts, configurations,
│           READMEs, and run outputs]
│
├── analysis/
│   ├── alpha_sensitivity/
│   ├── learning_rate/
│   ├── update_interval/
│   ├── final_stability/
│   ├── final_k_sensitivity/
│   ├── encoder_ablation/
│   ├── idec_diagnostic/
│   ├── idec_stability/
│   └── [analysis, aggregation, and visualization scripts]
│
├── docs/
│   ├── Design/
│   │   └── canonical_dec_training.md
│   └── experiments/
│       ├── alpha_sensitivity_analysis.md
│       ├── learning_rate_analysis.md
│       ├── update_interval_analysis.md
│       ├── final_stability_analysis.md
│       ├── final_k_stability_analysis.md
│       ├── encoder_ablation_analysis.md
│       ├── idec_analysis.md
│       └── idec_stability_analysis.md
│
├── final_dec_results/
├── final_idec_results/
│
├── requirements.txt
└── README.md
```

The repository separates experimental implementation, post-experiment analysis, and research documentation. The `experiments/` directory contains experiment-specific scripts, configurations, documentation, and run outputs. The `analysis/` directory contains scripts for processing experimental results, computing metrics, aggregating findings, and generating visualizations. The `docs/` directory provides methodological and analytical documentation for the experiments.

The `final_dec_results/` and `final_idec_results/` directories contain the saved results, metrics, cluster profiles, and visualizations associated with the respective final DEC and IDEC models.

## 5. Reproducibility and Requirements

- **Language:** Python 3.12.10
- **Deep learning:** PyTorch (autoencoder pretraining and DEC/IDEC implementation)
- **Classical machine learning:** scikit-learn (K-Means, Spectral Clustering, DBSCAN, Agglomerative Clustering, Kernel K-Means, and internal validation metrics)
- **Dimensionality reduction for visualization:** UMAP (`umap-learn`)
- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Statistics:** SciPy, used occasionally for supporting calculations

Random seeds are fixed and reported throughout: a reference seed (42) for the final model, and five-seed sweeps for every sensitivity and stability experiment in Part II. The representation-preservation analysis in Part II computes the reported correlations over the complete set of observation pairs, without sampling, to eliminate sampling uncertainty from the reported figures.

```bash
pip install -r requirements.txt
```

## 6. Publication Status

- **Part I** (bachelor's thesis) was completed and defended, and evaluated with the highest possible grade (20/20).
- **Part II** (deep clustering research) is ongoing. Current work focuses on investigating recent advances in clustering methods and exploring potential methodological extensions.

This README will be updated with the progress and publication details once available.
