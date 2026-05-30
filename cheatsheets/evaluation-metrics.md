# Evaluation Metrics Quick Reference

---

## Classification Metrics

| Metric | Formula | Range | When to Use | What It Misses |
|--------|---------|-------|-------------|----------------|
| Accuracy | (TP+TN) / N | [0, 1] | Balanced classes, simple baseline | Completely misleading on imbalanced data |
| Precision | TP / (TP+FP) | [0, 1] | When FP is costly (spam filter, alerts) | Ignores false negatives (missed cases) |
| Recall (Sensitivity) | TP / (TP+FN) | [0, 1] | When FN is costly (cancer detection, fraud) | Ignores false positives |
| F1 | 2PR / (P+R) | [0, 1] | Imbalanced classes, need P and R both | Treats P and R equally; use F-beta for asymmetric cost |
| F-beta | (1+β²)PR / (β²P+R) | [0, 1] | β>1 weights recall; β<1 weights precision | More complex to explain to stakeholders |
| AUC-ROC | Area under TPR vs FPR | [0.5, 1] | Threshold-invariant ranking quality | Insensitive to class imbalance; use AUC-PR instead |
| AUC-PR | Area under Precision-Recall curve | [0, 1] | Imbalanced classes, minority class matters most | Harder to compute; no baseline = class frequency |
| MCC | (TP·TN - FP·FN) / sqrt(...) | [-1, 1] | Balanced measure for all 4 confusion matrix cells | Less widely understood; range is less intuitive |
| Log Loss | -mean(y log p + (1-y)log(1-p)) | [0, inf) | Probabilistic calibration quality | Penalizes confident wrong predictions heavily |

**MCC formula (full):**
```
MCC = (TP·TN - FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Confusion matrix reminder:**
```
                  Predicted Positive   Predicted Negative
Actual Positive       TP                    FN
Actual Negative       FP                    TN
```

---

## Regression Metrics

| Metric | Formula | Range | When to Use | What It Misses |
|--------|---------|-------|-------------|----------------|
| MSE | mean((y - ŷ)²) | [0, inf) | Penalize large errors heavily, smooth optimization | Sensitive to outliers; same unit as y² |
| RMSE | sqrt(MSE) | [0, inf) | Same unit as y, easy interpretation | Still outlier-sensitive |
| MAE | mean(\|y - ŷ\|) | [0, inf) | Robust to outliers, median regression | Not differentiable at zero |
| MAPE | mean(\|y-ŷ\|/\|y\|) * 100 | [0, inf)% | Relative error, scale-invariant | Undefined when y=0; asymmetric (over vs under-forecast) |
| SMAPE | mean(2\|y-ŷ\|/(|y|+|ŷ|)) * 100 | [0, 200]% | Symmetric version of MAPE | Still biased; not truly symmetric |
| R² | 1 - SS_res/SS_tot | (-inf, 1] | Proportion of variance explained | Can be negative; penalizes model with wrong mean |
| Adjusted R² | 1 - (1-R²)(n-1)/(n-p-1) | (-inf, 1] | Penalizes adding unnecessary features | More complex; not standard across fields |

```
SS_res = Σ(y_i - ŷ_i)²
SS_tot = Σ(y_i - ȳ)²
```

---

## Ranking Metrics

| Metric | Formula | Range | When to Use |
|--------|---------|-------|-------------|
| MRR | (1/Q) Σ 1/rank_q | (0, 1] | Single relevant item per query (Q&A, factual search) |
| NDCG@K | DCG@K / IDCG@K | [0, 1] | Graded relevance, position-sensitive ranking |
| MAP | mean AP over queries | [0, 1] | Binary relevance, multiple relevant items per query |
| Hit Rate@K | Fraction of queries with relevant in top-K | [0, 1] | Recommendation: did user's item appear in top-K? |
| Precision@K | Relevant items in top-K / K | [0, 1] | Web search: how many of top K are good? |
| Recall@K | Relevant items in top-K / Total relevant | [0, 1] | Retrieval: how many relevant items did we find? |

### NDCG@K (Detailed)
```
DCG@K = Σ_{i=1}^{K}  rel_i / log_2(i + 1)

For binary relevance (rel_i in {0,1}):
  DCG@3 = rel_1/log_2(2) + rel_2/log_2(3) + rel_3/log_2(4)
        = rel_1/1 + rel_2/1.585 + rel_3/2

IDCG@K = DCG@K for the ideal (perfect) ranking

NDCG@K = DCG@K / IDCG@K
```
Discount = 1/log_2(i+1) means rank 1 is full credit, rank 2 is 63%, rank 3 is 50%.

### MAP (Detailed)
```
AP(q) = (1/R_q) Σ_{k=1}^{n} P(k) · rel(k)

P(k) = precision at rank k
rel(k) = 1 if item at rank k is relevant, 0 otherwise
R_q = total relevant items for query q

MAP = mean AP across all queries
```

---

## Generation Metrics

| Metric | Formula / Method | Range | When to Use | What It Misses |
|--------|-----------------|-------|-------------|----------------|
| BLEU | Geometric mean of n-gram precision + BP | [0, 1] | Machine translation, code generation | Recall, paraphrase, semantic similarity |
| ROUGE-N | N-gram recall between hypothesis and reference | [0, 1] | Summarization, headline generation | Precision, semantic equivalence |
| ROUGE-L | LCS-based recall | [0, 1] | Captures sentence-level structure better than N-gram | Still surface-form dependent |
| BERTScore | Cosine similarity of BERT embeddings (F1) | [-1, 1] | Semantic similarity evaluation, QA | Compute-heavy; correlated with BERT biases |
| Perplexity | exp(-mean log p(token)) | [1, inf) | Language model quality on held-out text | Task-independent; lower is better but not sufficient |
| METEOR | Harmonic mean P and R with stemming + synonyms | [0, 1] | Better than BLEU; handles morphology | Slower, less standard in current LLM eval |

### BLEU (simplified)
```
BLEU = BP · exp(Σ_n w_n log p_n)

BP = exp(1 - r/c) if c < r, else 1    (brevity penalty)
p_n = clipped n-gram precision
w_n = 1/N for uniform weighting (typically N=4)
```
Clipping: count each reference n-gram at most as often as it appears in the reference.

### Perplexity
```
PP(W) = exp( -1/T Σ_{t=1}^{T} log p(w_t | w_1,...,w_{t-1}) )
```
Lower = model assigns higher probability to held-out text = better language model.

---

## Clustering Metrics

| Metric | Range | Requires Labels | When to Use |
|--------|-------|-----------------|-------------|
| Silhouette Score | [-1, 1] | No (intrinsic) | General clustering quality; compare k values |
| Davies-Bouldin Index | [0, inf) | No | Lower is better; alternative to silhouette |
| Adjusted Rand Index (ARI) | [-1, 1] | Yes (ground truth) | When you have ground truth cluster assignments |
| Normalized Mutual Info (NMI) | [0, 1] | Yes | Cluster purity vs. ground truth; information-theoretic |
| V-Measure | [0, 1] | Yes | Harmonic mean of homogeneity and completeness |
| Inertia (WCSS) | [0, inf) | No | Within-cluster sum of squares; used for elbow method |

### Silhouette Score (Detailed)
```
For sample i:
  a(i) = mean distance to other points in same cluster
  b(i) = mean distance to points in nearest other cluster

s(i) = (b(i) - a(i)) / max(a(i), b(i))

Score = mean s(i) over all samples
```
Score near +1: well-separated clusters. Near 0: overlapping. Near -1: wrong cluster assignment.

---

## Metric Selection Guide

```
Problem type and key question -> Recommended metric(s)

Binary classification, balanced classes -> AUC-ROC + F1
Binary classification, severe imbalance -> AUC-PR + F1 (minority class)
Multi-class classification -> Macro-F1 or Weighted-F1 + Confusion matrix
Regression, outliers present -> MAE or Huber loss; not MSE
Regression, percentage errors matter -> MAPE (if no zeros), else SMAPE
Ranking, single relevant item -> MRR
Ranking, graded relevance -> NDCG@K
Recommendation, top-K appearance -> Hit Rate@K + NDCG@K
Translation / summarization -> BLEU or ROUGE + human eval
LLM output quality -> BERTScore + human eval; never BLEU alone
Clustering, no ground truth -> Silhouette score + elbow/Davies-Bouldin
Clustering, with ground truth labels -> ARI + NMI
```
