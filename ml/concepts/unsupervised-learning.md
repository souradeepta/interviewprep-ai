# Unsupervised Learning

**TL;DR:** Unsupervised learning finds structure in inputs without supplied target labels, supporting clustering, density estimation, anomaly discovery, and representation learning.

## Prerequisites

Prerequisites: [probability and statistics](probability-statistics.md), feature scaling, and basic linear algebra. You can distinguish clustering from dimensionality reduction and anomaly detection, select an objective appropriate to data geometry, evaluate results without pretending labels exist, and identify when an apparent cluster may be an artifact. This ML lesson covers the general problem; CV, NLP, and RL tracks apply unsupervised methods to their own modalities.

## Learning Outcomes

- Explain the central assumptions and mechanism of unsupervised learning.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

In supervised learning, examples arrive with answers. In unsupervised learning, the algorithm sees only observations and must find a useful organization, such as repeated patterns or low-dimensional structure. Imagine sorting unfamiliar coins by markings and size without knowing their monetary value. A clustering algorithm can group similar coins, but the groups do not tell us their intended meaning. A person must decide whether that grouping is useful for the downstream question.

Unsupervised objectives often reward a mathematical property—compact clusters, reconstructed inputs, or high density—not semantic truth. A cluster can reflect camera type or customer activity rather than the product categories a business cares about. Validation therefore uses stability, domain review, downstream utility, and robustness to preprocessing, not only one internal score.

## How It Works

Clustering partitions observations into groups under a distance or density notion. K-means alternates assigning each point to the nearest of k centroids and recomputing centroids as within-cluster means. It minimizes within-cluster squared Euclidean distance, so it favors roughly spherical groups and is sensitive to scale and outliers. Hierarchical clustering builds nested merges or splits and exposes a dendrogram, but naïve pairwise methods can require O(n²) memory. DBSCAN joins dense neighborhoods and marks sparse points as noise; it can find non-spherical shapes but is sensitive to radius and struggles when densities vary.

Dimensionality reduction maps high-dimensional inputs to fewer coordinates. PCA centers data and finds orthogonal directions of maximum variance; the top k components preserve as much linear variance as possible. It is useful for compression and visualization, but high variance need not equal task relevance. t-SNE and UMAP emphasize local neighborhoods for visualization; global distances and cluster sizes can be distorted, so a two-dimensional plot should not be interpreted as a faithful metric space. Autoencoders learn nonlinear representations by reconstructing inputs, but can learn identity-like copying unless constrained or bottlenecked.

Density estimation models p(x), while anomaly detection flags observations with low estimated likelihood, large reconstruction error, or unusual distance from a learned representation. Rare does not necessarily mean erroneous or harmful. An anomaly detector trained on mostly normal data may flag legitimate new populations after shift. The threshold is an operational decision and must be reviewed with expected false-alert cost.

Preprocessing defines geometry. Standardization makes Euclidean distance treat features on comparable scales; one-hot encoding can create a geometry where category mismatches are equally distant. Missing-value imputation, feature selection, and dimensionality reduction should be fitted on training data when later evaluation is intended. In high dimensions, pairwise distances concentrate and neighborhood intuitions degrade, making feature selection or learned representations important.

Cluster labels themselves have no canonical names or order. If a run assigns cluster 0 to one group and the next run assigns it to another, the numeric ID change does not imply that the underlying groups changed. To compare runs, align clusters using shared examples or an assignment procedure, then inspect membership overlap and representative records. This is especially important when a downstream workflow attaches a persistent action to a cluster ID. A useful segmentation should have stable, understandable membership or demonstrate value in an explicitly measured downstream task.

Many unsupervised methods have random initialization, optimization, or sampling. Fix seeds for reproducibility, but also test stability across multiple seeds and data resamples: a single fixed run can be reproducible yet fragile. For a candidate clustering, summarize group size, within-cluster distances, feature distributions, and examples near boundaries. Tiny clusters may represent meaningful rare behavior, outliers, or algorithmic artifacts; their interpretation requires domain review rather than automatic deletion.

## Trade-offs and Selection Boundary

| Method | Strength | Boundary |
|---|---|---|
| K-means | Fast and interpretable centroids; scales well | Needs k and suits compact Euclidean clusters |
| DBSCAN | Finds dense non-spherical clusters and labels noise | Sensitive to radius/min_samples and varying density |
| PCA | Deterministic linear compression with interpretable directions | Preserves variance, not necessarily class/task information |
| t-SNE/UMAP | Useful visual exploration of local neighborhoods | Plot geometry can mislead; embedding depends on settings |
| Autoencoder | Learns nonlinear compressed representation | Reconstruction quality may not align with downstream utility |
| Density model | Scores likelihood and unusualness | Low density may be valid; high-dimensional estimation is difficult |

Choose K-means for a clear centroid-based segmentation hypothesis after scaling and checking cluster stability. Use density-based methods when shapes vary and noise points matter. Use PCA for linear compression or denoising; use nonlinear visualization for exploration, not as proof of semantic groups. If the task requires predicting known outcomes, a supervised target and baseline may be more appropriate than discovering clusters.

## Implementation

```python
def kmeans_two(points, centers, steps=5):
    for _ in range(steps):
        groups=[[],[]]
        for p in points:
            idx=min(range(2), key=lambda j: sum((a-b)**2 for a,b in zip(p,centers[j])))
            groups[idx].append(p)
        centers=[tuple(sum(p[d] for p in g)/len(g) for d in range(len(g[0]))) for g in groups]
    return centers, groups

points=[(0.,0.), (0.,1.), (5.,5.), (6.,5.)]
centers, groups=kmeans_two(points, [(0.,0.),(6.,5.)])
assert sorted(len(g) for g in groups) == [2,2]
assert centers == [(0.0,.5),(5.5,5.0)]
```

## Failure Modes and Diagnostics

1. **Scale dominates distance:** one numeric feature determines all clusters. Standardize using training data and compare membership before/after scaling.
2. **Chosen k mistaken for truth:** K-means returns groups even when there is no natural partition. Test stability across seeds and k values and validate downstream usefulness.
3. **Visualization interpreted globally:** t-SNE islands appear far apart or large/small. Treat the map as a local-neighborhood view and confirm patterns in original space.
4. **Anomaly score equated with risk:** rare valid events flood alerts. Review labeled samples, set thresholds by capacity/cost, and monitor prevalence and drift.
5. **High-dimensional distances become uninformative:** nearest and farthest neighbors have similar distances. Inspect distance distributions, reduce irrelevant dimensions, or learn representations.

## Interview Quick Reference

| Question | Answer |
|---|---|
| How does K-means work? | Alternate nearest-centroid assignments and centroid means to reduce within-cluster squared distance. |
| How evaluate clusters without labels? | Use stability, internal measures such as silhouette with care, domain review, and downstream utility. |
| Design follow-up: customer clusters change every retrain. | Check scaling, feature drift, initialization, and cluster alignment; assess stability and whether membership changes affect a real decision. |

## Related Concepts

- [Feature engineering](feature-engineering.md), [probability and statistics](probability-statistics.md), and [anomaly detection via supervised evaluation](evaluation-metrics.md)
- [Matching notebook](../notebooks/39-unsupervised-learning.ipynb)
- [Interview practice: clustering](../interview-prep/ml-theory-questions.md#q-explain-the-em-algorithm-with-an-example)

## References

- Jolliffe and Cadima, [Principal Component Analysis: A Review and Recent Developments](https://doi.org/10.1098/rsta.2015.0202).
- scikit-learn, [Clustering](https://scikit-learn.org/stable/modules/clustering.html).
