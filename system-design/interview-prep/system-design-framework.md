# ML System Design Framework

## The Framework

Use this 45-minute structure for any ML system design question.

### 1. Clarifying Questions (5 min)
- **Scale:** DAU, QPS, data volume?
- **Latency:** response time SLA?
- **Accuracy:** business metric (precision, recall, AUC)?
- **Data:** labeled data available? How much?

### 2. Problem Formulation (5 min)
Define the ML task precisely:
- **Type:** classification, regression, ranking, clustering?
- **Labels:** what defines positive/negative?
- **Output:** class scores, probabilities, scores for ranking?
- **Challenges:** class imbalance, concept drift, data sparsity?

### 3. High-Level Architecture (10 min)

**Two stages (most common):**
- **Candidate generation:** find top-K candidates quickly (ANN, two-tower, collaborative filtering)
- **Ranking:** score/rank top-K with rich features (gradient boosted trees, small NN)

**Key components:**
- **Feature store:** where do features come from? Precomputed or real-time?
- **Training pipeline:** how often retrain? Batch or online?
- **Serving:** online (request-time) or offline (batch)?

### 4. Data Processing (5 min)
- **Collection:** event logging, labels
- **Preprocessing:** deduplication, filtering
- **Features:** what raw features? How engineered?
- **Splits:** train/val/test strategy? Handle temporal ordering?

### 5. Model Selection (5 min)
- **Baseline:** simple rule or heuristic for comparison
- **Main model:** justify choice (fast for latency? interpretable? data requirements?)
- **Ensemble:** combine multiple models if time allows

### 6. Offline Evaluation (3 min)
- **Metrics:** precision, recall, AUC, NDCG@K, watch time, etc.
- **Evaluation set:** what makes a good holdout test set?
- **Offline results target:** aim for 5-10% improvement over baseline?

### 7. Online Evaluation (2 min)
- **A/B test:** what's the primary metric? Sample size? Duration?
- **Guardrails:** don't degrade other metrics while improving primary

## Template Answers

**Q: "How would you design a recommendation system?"**

1. **Clarifying:** 50M users, 10M items, <100ms latency, optimize for watch time
2. **Problem:** Ranking — given user, rank items by predicted watch time
3. **Architecture:** Two-tower (candidate) + LightGBM (ranker)
4. **Features:** User embeddings, item embeddings, user history, context
5. **Model:** Two-tower for speed; LightGBM for accuracy
6. **Offline eval:** NDCG@20, AUC on heldout users
7. **Online eval:** A/B test (5% traffic), primary metric = avg watch time per session

**Q: "Design a fraud detection system."**

1. **Clarifying:** 1M transactions/min, <50ms latency, cost(FN) >> cost(FP)
2. **Problem:** Binary classification — fraud vs legitimate
3. **Architecture:** Rule engine (instant) + ML model (async backup)
4. **Features:** Transaction amount, user history, velocity, network
5. **Model:** Gradient boosting (interpretable + fast)
6. **Offline eval:** Precision-recall curve, target 95% recall at 10% FPR
7. **Online eval:** Monitor fraud rate, false positive rate, review queue load

## Red Flags

If you say this, the interviewer will note it down:
- "I'll use deep learning" (without justifying why)
- "I'll just use accuracy as the metric" (ignores class imbalance or business costs)
- "The model never needs retraining" (concept drift is real)
- "I'll collect all possible features" (feature explosion, data leakage risk)

## Success Signals

The interviewer looks for:
- ✓ Understanding of the business problem (not just ML)
- ✓ Trade-off awareness (latency vs accuracy, recall vs precision)
- ✓ Scalability thinking (will this break at 100M users?)
- ✓ Online evaluation design (A/B testing, guardrails)
- ✓ Problem-specific insights (e.g., concept drift in fraud, cold-start in recs)

## Related Topics
- [Online vs Batch Inference](../patterns/online-vs-batch-inference.md) — [Feature Store](../patterns/feature-store.md)
