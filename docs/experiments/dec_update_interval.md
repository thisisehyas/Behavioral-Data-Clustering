# Effect of Target Distribution Update Interval in DEC

We studied how the interval of updating the target distribution in the Deep Embedded Clustering model affects its convergence behavior. This update interval is related to the frequency of recalculating the auxiliary target distribution p from the model's soft assignments, a factor that controls the balance between stability and refinement during training.

We ran experiments with update intervals of {1, 5, 10, 20, 50}, while maintaining all other training hyperparameters constant. In each setting, we tracked the KL divergence loss between the target distribution (p) and the soft assignment distribution (q) across 50 training epochs.

Very frequent updates (interval = 1) resulted in unstable optimization and showed higher loss values, with slower convergence, probably due to excessive target drift. On the other hand, very infrequent updates, interval = 50, resulted in low loss values, but it made the model behave as if optimizing against a nearly fixed target distribution, which limits useful cluster refinement.

Intermediate ranges of 5, 10, and 20 exhibit a clear phase-wise pattern: the loss values remained low between updates and sharply rose right after each one, indicating sudden shifts in the optimization goal. Of these, 10 and 20 provided a good balance between stability and gradual refinement.

Considering these observations, update intervals of 10 and 20 were selected for detailed evaluations using external clustering quality metrics.

## Experimental Setup
- Model: Deep Embedded Clustering (DEC)
- Encoder: pretrained autoencoder (frozen)
- Number of clusters: 5
- Optimizer: Adam (lr = 1e-3)
- Epochs: 50
- Batch size: 256
- Dataset: latent embeddings (X_scaled)

## Decision
Based on loss behavior and stability, update intervals of 10 and 20 were selected for downstream evaluation.