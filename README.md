# Roommate Compatibility Analysis

_Modeling Roommate Compatibility Through Psychological and Behavioral Data_

This project explores how individuals with different psychological, behavioral, and lifestyle traits can be grouped into compatible roommate matches. It addresses a common question for anyone living in a shared space, such as dormitories: _"Who would be a better roommate for me?"_

To achieve this, a large-scale dataset of **100,000 records and 50 features** was analyzed using various clustering techniques, including **K-Means, Hierarchical Clustering, DBSCAN, and Spectral Clustering**. In addition, a **compatibility scoring system** was developed, combining clustering results with cosine similarity to assign each pair of individuals a numerical compatibility score.

## Motivation

Choosing the right roommate(s) can have a big impact on a person's mental health, productivity and overall quality of life. Living with someone who isn't compatible can easily lead to stress, arguments and an uncomfortable home environment — especially in shared spaces like dormitories or student housing.

Traditional roommate matching methods usually focus on only a few simple preferences such as cleanliness or sleep schedules and ignore deeper psychological or behavioral traits. This project aims to close that gap by using data-driven clustering and similarity analysis to discover meaningful compatibility patterns between people.

By analyzing psychological, behavioral and lifestyle features, this project shows how machine learning can be used to create more balanced and comfortable living environments through clear and objective compatibility analysis.

## Dataset

The dataset used in this project was extracted from a [public GitHub repository](https://github.com/Pradyumna-yes/Roommate-Pairing-ML-Model). It contains **100,000 rows and 50 features** representing **psychological, behavioral and lifestyle traits**, with all feature values ranging from **0 to 5**.

The dataset was collected through self-reported responses to a comprehensive questionnaire. The questions were designed based on validated psychological and behavioral criteria, including the [IPIP Big-Five Factor Markers](https://openpsychometrics.org/tests/IPIP-BFFM/) and related resources from the [International Personality Item Pool (IPIP)](https://ipip.ori.org/new_ipip-50-item-scale.htm). This ensured that the data captured reliable psychological and behavioral dimensions, making it suitable for analytical and clustering approaches.

Some algorithms were applied directly to the original dataset, while others used the **`compressed_features`** version, which groups related features together. The mean value of each feature group represents a higher-level characteristic. In total, **seven feature groups** were created.

<details>
<summary><strong>Feature Categories in <code>compressed_features</code></strong></summary>

<details>
<summary><strong>1. Lifestyle</strong></summary>
<ul>
  <li>Sleeping Patterns</li>
  <li>Eating Habits</li>
  <li>Exercise Routines</li>
  <li>Cleanliness and Hygiene</li>
  <li>Smoking and Drinking Habits</li>
  <li>Attitudes Toward Money and Finances</li>
  <li>Preference for Quiet vs. Lively Environment</li>
  <li>Views on Personal and Shared Spaces</li>
</ul>
</details>

<details>
<summary><strong>2. Interests & Activities</strong></summary>
<ul>
  <li>Music Preferences</li>
  <li>Movie/TV Show Genres</li>
  <li>Reading and Literature</li>
  <li>Sports and Outdoor Activities</li>
  <li>Cooking and Culinary Interests</li>
  <li>Travel and Exploration Interests</li>
  <li>Artistic Interests</li>
  <li>Creativity</li>
</ul>
</details>

<details>
<summary><strong>3. Communication & Interaction</strong></summary>
<ul>
  <li>Listening Skills</li>
  <li>Conflict Resolution Abilities</li>
  <li>Openness to Feedback</li>
  <li>Expressiveness</li>
  <li>Empathy in Communication</li>
  <li>Frequency of Hosting Guests</li>
  <li>Participation in Community Events</li>
  <li>Networking and Socializing Habits</li>
</ul>
</details>

<details>
<summary><strong>4. Responsibility & Organization</strong></summary>
<ul>
  <li>Work Ethic</li>
  <li>Organizational Skills</li>
  <li>Reliability</li>
  <li>Punctuality</li>
  <li>Attention to Detail</li>
</ul>
</details>

<details>
<summary><strong>5. Social Engagement & Behavior</strong></summary>
<ul>
  <li>Talkativeness</li>
  <li>Enthusiasm for Social Gatherings</li>
  <li>Energy Levels</li>
  <li>Sociability</li>
  <li>Assertiveness</li>
  <li>Cooperation</li>
  <li>Trust in Others</li>
  <li>Altruism</li>
  <li>Modesty</li>
</ul>
</details>

<details>
<summary><strong>6. Emotional Regulation & Stability</strong></summary>
<ul>
  <li>Compassion</li>
  <li>Anger Management</li>
  <li>Stress Management</li>
  <li>Emotional Stability</li>
  <li>Anxiety Levels</li>
  <li>Mood Swings</li>
</ul>
</details>

<details>
<summary><strong>7. Openness & Core Beliefs</strong></summary>
<ul>
  <li>Intellectual Curiosity</li>
  <li>Preference for Novelty</li>
  <li>Openness to Diverse Cultures</li>
  <li>Political Views</li>
  <li>Religious Beliefs</li>
  <li>Cultural Values</li>
</ul>
</details>

</details>

## Methods

This section explains the steps we followed to prepare the data and perform clustering. It includes data cleaning, exploratory analysis, dimensionality reduction, testing clustering methods (and sub-clustering), and building a compatibility scoring system. Cluster quality was measured using silhouette score, Davies–Bouldin, and Calinski–Harabasz indices. Visualizations were produced using UMAP, bar plots, and heatmaps.

<details>
<summary><strong>Exploratory Analysis & Data Preprocessing</strong></summary>

- **Initial EDA:** Dataset has ~100,000 samples and 50 numeric features (values 0–5). Most columns were complete; only a few missing values.
- **Missing values:** Filled using median per column to preserve distribution.
- **Outliers:** Checked with boxplots and z-scores. All values were within ±3, no removal needed.
- **Distributions & correlations:** Some features skewed; others roughly normal (e.g., creativity, curiosity). Correlation heatmap showed no features correlated >0.8.

</details>

<details>
<summary><strong>Standardization & Dimensionality Reduction</strong></summary>

- Standardized numeric features to mean 0 and standard deviation 1 for fair distance calculation.
- **PCA** used for dimensionality reduction and visualization. 30 components retained ~90–95% variance. PCA plots suggested possible subgroups.

</details>

<details>
<summary><strong>Feature Engineering: <code>compressed_features</code></strong></summary>

- Related features grouped into 7 categories:
  <details>
  <summary>The categories</summary>
  Lifestyle  <br>
  Interests & Activities  <br>
  Communication & Interaction  <br>
  Responsibility & Organization  <br>
  Social Engagement & Behavior  <br>
  Emotional Stability  <br>
  Openness & Core Beliefs
  </details>
- For each group, calculated the mean of features → `compressed_features` dataset, easier to interpret and used in some experiments.

</details>

<details>
<summary><strong>Clustering Experiments</strong></summary>

#### K-Means (Centroid-Based)

- Applied on 30 PCA components, tested 2–11 clusters, chose using silhouette score.
- Feature-importance pipeline: K-Means on 50 features, Random Forest predicted cluster labels → top 10 features selected.
- Refined K-Means: ran on top 10 features, tested 2–6 clusters using silhouette, Davies–Bouldin, Calinski–Harabasz scores.
- Composite-feature K-Means: ran on `compressed_features` to interpret clusters psychologically.

#### Spectral Clustering

- Too large for 100,000 samples → used **random 3,000-sample subset**.
- Tested using few composite features, 2–6 clusters. Reported results only for subset.

#### Density-Based Clustering (DBSCAN / HDBSCAN)

- Two modes:
  1. Top 10 features after UMAP → many small clusters and noise.
  2. Full dataset → 5 main clusters; evaluated with metrics and UMAP plots.

#### Hierarchical Clustering

- Agglomerative (Ward linkage) on 15,000 samples → two main clusters with subclusters.
- K-Means + Agglomerative: K-Means split dataset into two parts, agglomerative applied, labels generalized via k-NN classifier.
- Divisive (custom): recursive K-Means splitting.
  - Stopping: max depth 3, cluster < 2000, silhouette < 0.12.
  - Multi-level clusters created (e.g., 0 → 0.0 → 0.0.0) with metrics at each step.

#### Kernel K-Means (RBF Approximation)

- Detect non-linear cluster shapes in major subclusters.
- Function `find_best_rbf_kernel_kmeans` selected best gamma (RBF width) and number of clusters.
- Each subcluster split hierarchically → up to 8 final clusters.
- Cluster profiles calculated using 7 composite groups.

</details>

<details>
<summary><strong>Compatibility Scoring System</strong></summary>

- **Goal:** For each person, find top-5 roommate matches with scores and score breakdown.
- **Inputs:** normalized features, divisive-hierarchical labels, number of recommendations.
- **Mechanism:**
  1. Cluster-based similarity: similarity of cluster centers.
  2. Cosine similarity: similarity between individuals.
  3. Weighted combination: `Final Score = w_cos * Cosine + w_clust * Cluster`.
- **Hyperparameters:** Tested weights (0.5,0.5), (0.7,0.3), (0.9,0.1); chose (0.7,0.3).
- Scores rescaled to 0–4. Output: top-5 matches per person with breakdown.

</details>

<details>
<summary><strong>Evaluation & Visualization</strong></summary>

- **Metrics:** silhouette score, Davies–Bouldin, Calinski–Harabasz.
- **Cluster numbers:** initially 2–11, refined to 2–6.
- **Visualizations:** UMAP 2D projections, dendrograms, bar plots of cluster feature means, heatmaps of cluster vs feature groups.
- Final clusters’ psychological profiles summarized using 7 composite feature groups.

</details>

<details>
<summary><strong>Outputs & Artifacts</strong></summary>

- Cluster profiles (means of 7 composite groups) and visualizations.
- Compatibility results saved with top-5 matches and score breakdown.
- Logs from divisive algorithm with metrics and cluster IDs for reproducibility.

> Notes: Large dataset and memory limits shaped design choices (sampling for spectral clustering, RBF approximations). Detailed results and plots are available in the notebooks.

</details>

## Evaluation

The clustering models were evaluated using three internal validation metrics: **Silhouette Score, Davis-Bouldin Index,** and **Calinski-Harabasz Index**. These metrics assess the quality of cluster separation and cohesion without requiring labeled data. Each algorithm was tested under comparable conditions, with different feature subsets (PCA, Top Features, Composite Features) or sampling strategies when necessary.

| Algorithm                                | Number of Clusters | Silhouette                 | Davies–Bouldin          | Calinski–Harabasz          |
| ---------------------------------------- | ------------------ | -------------------------- | ----------------------- | -------------------------- |
| KMeans (PCA Features)                    | 2                  | 0.1047                     | –                       | –                          |
| KMeans (Top 10 Features)                 | 2                  | 0.2126                     | 1.7053                  | 31,502.90                  |
| KMeans (Composite Features)              | 2                  | **0.5372**                 | **1.0677**              | **71,027.18**              |
| Spectral Clustering (3,000 Samples)      | 2                  | 0.5100                     | 0.6879                  | 70.39                      |
| Spectral Clustering (Composite Features) | 2                  | **0.7131**                 | **0.2254**              | 140.67                     |
| DBSCAN (All Features)                    | 5                  | **0.6900**                 | **0.2094**              | 3,871.81                   |
| Divisive Hierarchical Clustering         | 7                  | 0.134–0.146 (one = 0.7837) | Avg = 2.26 (one = 0.37) | Avg = 8,549 (one = 19,585) |
| Kernel K-Means (RBF Approximation)       | 8                  | 0.13–0.15 (one = 0.6926)   | –                       | –                          |

> **Note:** Higher Silhouette and Calinski–Harabasz values, and lower Davies–Bouldin values, indicate better cluster separation and internal consistency.  
> Overall, Spectral Clustering on composite features and Density-Based Clustering showed the strongest cluster structure.
