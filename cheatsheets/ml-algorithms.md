# ML Algorithms Quick Reference

## Master Comparison Table

| Algorithm | Type | Train Time | Inference | Memory | Best For | Avoid When |
|-----------|------|------------|-----------|--------|----------|------------|
| Linear Regression | Supervised / Regression | O(nd²) | O(d) — very fast | O(d) | Continuous targets, interpretability required, baselines | Non-linear relationships, noisy outliers (use Huber loss) |
| Logistic Regression | Supervised / Classification | O(ndi) iters | O(d) — very fast | O(d) | Binary/multi-class, calibrated probabilities, sparse features | Complex non-linear decision boundaries |
| Decision Tree | Supervised / Both | O(nd log n) | O(log n) | O(tree size) | Interpretability, categorical features, non-linear data | Deep trees overfit; avoid on high-dim sparse data |
| Random Forest | Supervised / Ensemble | O(T · nd log n) | O(T · log n) | O(T · tree size) | Tabular data, robust to hyperparams, feature importance | Very high-dim sparse data, strict real-time latency |
| Gradient Boosting (XGBoost) | Supervised / Ensemble | Slow (sequential) | Fast (cached) | Moderate | Tabular data competitions, structured features, ranking | Unstructured data (images/text), tiny datasets |
| SVM | Supervised / Both | O(n²–n³) | O(n_sv · d) | O(n_sv) | High-dim, small-medium datasets, clear margin | Large n (slow training), noisy labels, need probabilities |
| KNN | Supervised / Both | O(1) (no training) | O(nd) | O(nd) | Similarity search, small/medium datasets, no training budget | Large n (slow inference), high-d (curse of dimensionality) |
| K-Means | Unsupervised / Clustering | O(nkdi) | O(kd) per point | O(k + nd) | Spherical clusters, customer segmentation, fast clustering | Non-convex clusters, varying densities, categorical data |
| Neural Network (MLP) | Supervised / Both | Slow (GPU-hours) | O(params) | O(params) | Complex patterns, large data, feature learning | Small datasets (<1K rows), interpretability required |
| Transformer | Supervised / Both | Very slow (GPU-days) | O(n² · d) or cached | O(n · d) | NLP, vision, sequential data, large-scale pretraining | Small datasets, low-latency edge inference, tiny budgets |

n = samples, d = features, T = trees, k = clusters, i = iterations

---

## When to Use Each — Decision Guide

### Linear / Logistic Regression
- Always use as a **baseline** before complex models
- Regulatory or audit environments requiring **interpretability**
- Small datasets (< 10K rows) with hand-crafted features
- Features are already well-engineered and relationships are mostly linear

### Decision Tree
- Need to **explain individual decisions** as a decision path
- Mix of categorical and numerical features without preprocessing
- Keep depth <= 5 for human-readable trees; otherwise use ensemble

### Random Forest
- Tabular data, medium dataset (10K–1M rows)
- Want robustness with minimal hyperparameter tuning
- Need reliable **feature importance** estimates
- Parallel training available (each tree is independent)

### Gradient Boosting (XGBoost / LightGBM / CatBoost)
- **Structured tabular competitions** or production ranking tasks
- Many engineered features with complex interactions
- Use **LightGBM** for very large datasets (leaf-wise splits, faster)
- Use **CatBoost** when many categorical features (handles natively)

### SVM
- High-dimensional data, small-medium n (e.g., text with TF-IDF)
- Clear **margin** expected between classes
- Kernel trick needed for non-linearity (RBF kernel most common)

### KNN
- No training time budget; supports online updates trivially
- **Similarity retrieval** use case (product search, deduplication)
- Need interpretable rationale ("similar to these past examples")
- Use approximate NN libs (FAISS, ScaNN) for production scale

### K-Means
- Fast prototype clustering when cluster structure is unknown
- Post-embedding clustering (cluster sentence/image embeddings)
- Initial step before applying finer-grained algorithms
- Elbow method + silhouette score to choose k

### Neural Networks (MLP)
- Unstructured data (images, text, audio) with large datasets
- Complex patterns that tree methods plateau on
- Transfer learning from pretrained checkpoints available

### Transformers
- NLP tasks (classification, generation, Q&A, summarization)
- Long-range dependencies in sequences
- Large-scale pretraining or fine-tuning from HuggingFace Hub

---

## Key Hyperparameters

| Algorithm | Primary Hyperparams | Typical Starting Values | Most Impact |
|-----------|--------------------|-----------------------|-------------|
| Linear Regression | alpha (L1/L2 penalty) | alpha=1.0 | alpha |
| Logistic Regression | C (inverse regularization), max_iter | C=1.0, 1000 | C |
| Decision Tree | max_depth, min_samples_leaf, criterion | None, 1, gini | max_depth |
| Random Forest | n_estimators, max_depth, max_features | 100, None, "sqrt" | max_features |
| XGBoost | n_estimators, learning_rate, max_depth, subsample | 300, 0.1, 6, 0.8 | learning_rate |
| SVM | C, kernel, gamma | 1.0, rbf, "scale" | C and gamma |
| KNN | n_neighbors, metric, weights | 5, euclidean, uniform | n_neighbors |
| K-Means | n_clusters, init, n_init | 8, k-means++, 10 | n_clusters |
| Neural Network | lr, batch_size, depth, width, dropout | 1e-3, 128, 3, 256, 0.1 | lr and depth |
| Transformer | lr, warmup_steps, dropout, num_heads | 1e-4, 1000, 0.1, 8 | lr warmup schedule |

---

## Common Pitfalls

| Algorithm | Pitfall | Symptom | Fix |
|-----------|---------|---------|-----|
| Linear Regression | Multicollinearity among features | Unstable coefficients, huge confidence intervals | Use Ridge regression; remove correlated features |
| Logistic Regression | Unbalanced class distribution | High accuracy, low recall on minority class | class_weight="balanced"; use F1, not accuracy |
| Decision Tree | Unconstrained depth | Perfect train accuracy, poor validation | Set max_depth <= 8; increase min_samples_leaf |
| Random Forest | Too many trees in production | High inference latency | Reduce n_estimators; switch to LightGBM |
| XGBoost | Data leakage through future features | Inflated offline metric, bad live performance | Use TimeSeriesSplit; audit feature timestamps |
| SVM | Slow training on large n | Training takes hours/days | Switch to LinearSVC or SGDClassifier |
| KNN | Brute-force inference too slow | High per-query latency | Use approximate NN: FAISS, Annoy, ScaNN |
| K-Means | Wrong number of clusters | Forced groupings, poor silhouette score | Elbow method + silhouette; try DBSCAN |
| Neural Network | Vanishing gradient | No learning, loss stays flat | Add BatchNorm, residual connections; reduce depth |
| Transformer | OOM on long sequences | CUDA out-of-memory at train or inference | Flash attention; gradient checkpointing; chunking |
