# ML Case Studies

End-to-end ML system design. Simulate a 45-minute interview.

---

## Case Study: Content Recommendation System

**Scenario:** Senior ML engineer at a streaming platform. Design the ML system that recommends
the next piece of content to each user.

### Step 1 — Clarifying Questions to Ask
- "What scale? DAU, catalog size, latency SLA?"
- "Business metric: clicks, watch time, or retention?"
- "Real-time personalization or precomputed?"
- "Cold-start users?"

*Assumed: 50M DAU, 10M items, <100ms response, optimize watch time, real-time for existing users.*

### Step 2 — ML Problem Formulation
- **Task:** ranking — given user u, rank candidates by predicted watch time
- **Labels:** implicit (watched ≥ X min = positive, skip = negative)
- **Output:** relevance score per (user, item) pair

### Step 3 — System Design

**Candidate Generation (offline, batch):**
- Two-tower model: user embedding × item embedding → cosine similarity → top 500 candidates
- Rerun nightly; triggered by large user activity shift

**Ranking (online, <100ms):**
- Features: user embedding, item embedding, user-item interactions, context (time, device)
- Model: LightGBM or small 3-layer MLP (low latency)
- Output: ranked top-20 items

**Feature Store:**
- User features (30d history, demographics) → Redis (precomputed)
- Item features (genre, popularity, embeddings) → Redis
- Session features (last 3 items watched) → Flink stream processing

**Training Pipeline:**
- Event logs → data warehouse → feature engineering → weekly retrain
- Offline metrics: AUC, NDCG@20
- Online: A/B test (5% traffic), primary metric = watch time per session

**Cold Start:**
- New users: popularity + content-based from onboarding preferences
- New items: content-based embedding until sufficient interaction data

### Step 4 — Key Trade-offs
- Two-tower vs MF: two-tower supports richer features but more complex to train/serve
- Online vs offline ranking: online = fresher, but latency risk; offline = fast, but stale
- Explore vs exploit: ε-greedy or UCB bandit for content exploration

### Step 5 — Failure Modes and Mitigations
- **Filter bubble:** homogeneous recommendations → add diversity constraint (MMR), exploration budget
- **Popularity bias:** popular items dominate → add popularity feature + regularize, or debias labels
- **Feedback loop:** model reinforces itself → counterfactual logging, propensity scoring

### Step 6 — Follow-up Questions
- "New user with no history?" → Content-based (item features) + global popularity + onboarding quiz
- "Model degrading in production?" → Monitor online metrics vs baseline; feature distribution drift alerts
- "Reduce latency from 100ms to 30ms?" → Precompute more offline, ANN for retrieval (FAISS), cache user embeddings

---

## Case Study: Fraud Detection System

**Scenario:** ML engineer at a payments company. Design a fraud detection system that evaluates
every transaction in real time.

### Step 1 — Clarifying Questions to Ask
- "Transaction volume? Latency budget?"
- "What is the cost of FP (blocking legitimate transaction) vs FN (missed fraud)?"
- "Do we have labeled data? How fresh?"
- "Online or batch scoring?"

*Assumed: 1M transactions/minute, <50ms latency, FN much more costly than FP, labeled historical data.*

### Step 2 — ML Problem Formulation
- **Task:** binary classification — fraud (1) or legitimate (0)
- **Labels:** chargebacks + manual review outcomes (delayed, noisy)
- **Challenge:** extreme class imbalance (~0.1% fraud), concept drift (fraud patterns evolve)

### Step 3 — System Design

**Feature Engineering (real-time + batch):**
- Transaction: amount, merchant, category, country, time-of-day
- User history (streaming): velocity features (transactions in last 1h/24h/7d), avg amount, deviation from baseline
- Network: graph features — is merchant/card/IP associated with known fraud network?
- Device: device fingerprint, location vs home country

**Model Stack:**
- Rule engine (instant, <1ms): hard rules (transaction > $10k from new device → flag)
- ML model (gradient boosting, <20ms): LightGBM with 200 features
- Neural network (optional, async): GNN over transaction network for complex patterns

**Handling Imbalance:**
- Use precision-recall AUC, not ROC-AUC (misleading with class imbalance)
- Oversample fraud / undersample legit with SMOTE or class weights
- Calibrate model outputs: Platt scaling or isotonic regression

**Training:**
- Labels arrive with delay (chargebacks take days) → training set is always stale
- Rolling window retraining: retrain weekly on 90-day window
- Champion/challenger: new model in shadow mode before promoting

### Step 4 — Key Trade-offs
- Recall vs precision: catch more fraud (high recall) vs fewer false positives (high precision)
- Rule engine vs ML: rules are fast and interpretable but brittle; ML generalizes but opaque
- Latency vs accuracy: complex models need async scoring or pre-computation

### Step 5 — Failure Modes and Mitigations
- **Concept drift:** fraud patterns evolve → monitor feature distributions, trigger retraining on drift
- **Label delay:** chargebacks delayed by 30d → use weak labels (merchant dispute) for faster feedback
- **Adversarial adaptation:** fraudsters adapt to model → regularly update features, add noise perturbation robustness

### Step 6 — Follow-up Questions
- "How do you handle false positives at scale?" → Tiered review: low-risk FP auto-approved, high-risk FP go to human review queue
- "Explainability requirement?" → SHAP values for each decision; rule-based fallback for regulatory audits
- "New type of fraud not in training data?" → Anomaly detection layer (isolation forest, autoencoder) in parallel with supervised model

---

*Additional case studies (same format):*
*- Search Query Understanding and Ranking*
*- ML-Powered Ads Click Prediction at 1M QPS*
