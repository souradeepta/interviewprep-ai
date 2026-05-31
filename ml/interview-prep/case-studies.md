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

---

## Case Study 03: Ad Click Prediction

**Scenario:** Senior ML engineer at a social media company. Design the ML system predicting whether a user will click on an ad.

### Step 1 — Clarifying Questions to Ask
- "What is the impression volume and latency SLA?"
- "What business metric are we optimizing — CTR, RPM, or something else?"
- "Do we have access to user identity or are we relying on cookie-level signals?"
- "What is the expected CTR and how severe is the class imbalance?"

*Assumed: 1B impressions/day, <10ms latency, 100K QPS peak, optimize CTR as primary and RPM as secondary, ~0.2% CTR (heavy imbalance), user identity available via login.*

### Step 2 — ML Problem Formulation
- **Task:** binary classification — will user u click on ad a given context c?
- **Labels:** click = positive (1), impression with no click = negative (0)
- **Challenge:** class imbalance at 0.2% CTR means 500 negatives per positive. Random sampling destroys precision. Use negative downsampling (keep 10% of negatives) and correct for it during calibration.
- **Output:** P(click | user, ad, context) — must be well-calibrated, not just rank-ordered, because downstream bid optimization uses raw probabilities

### Step 3 — System Design

**Data:**
- Impression logs (1B/day), click logs (2M/day), user activity stream (Kafka), ad metadata
- Feature freshness: user features updated hourly (batch), context features real-time

**Features:**
- User embedding (256-dim, trained on interaction history, updated daily)
- Ad embedding (256-dim, trained on ad content + historical engagement)
- Context: hour-of-day, day-of-week, device type, placement (feed vs sidebar), app version
- Interaction history: CTR on this ad category in last 7d, last 30d, all-time
- Cross features: user age group × ad category, device × ad format

**Model:** DLRM-style two-tower architecture
- Sparse features (user ID, ad ID) → embedding lookup (tables: 100M users × 256-dim, 10M ads × 256-dim)
- Dense features (context, statistics) → batch norm → MLP
- Interaction layer: dot products of all embedding pairs + dense features
- Final scoring MLP: 512→256→128→1, sigmoid output
- Retrain daily on 7-day sliding window; fine-tune hourly on last 1h of data

**Serving:**
- User and ad embeddings precomputed and cached in Redis (<1ms lookup)
- Scoring MLP served via TorchServe on GPU nodes: <5ms
- Total pipeline: feature fetch (3ms) + scoring (5ms) = <10ms P95

### Step 4 — Offline Evaluation
- **Metrics:** AUC-ROC (ranking quality), log-loss (calibration quality), Normalized Cross-Entropy (NCE)
- **Calibration check:** plot predicted probability deciles vs actual CTR — they should match
- **Holdout:** time-based split — train on days 1–27, validate on day 28, test on day 29–30. Never random split (leaks future signal)
- **Baseline:** logistic regression on hand-crafted features. Beat it by >1% NCE to ship.

### Step 5 — Online Evaluation and A/B Test
- **Primary metric:** CTR (click-through rate per impression)
- **Secondary metric:** RPM (revenue per thousand impressions) — CTR × bid price
- **Guardrail metrics:** user satisfaction signals (scroll-stop rate, hide-ad rate), session length (ads must not degrade feed engagement)
- **Test design:** 2% traffic hold-out (1% control, 1% treatment), 7-day minimum run for weekly seasonality
- **Rollout:** 2% → 10% → 50% → 100% with metric check at each gate

### Step 6 — Production Monitoring
- **Data drift:** PSI on top-20 features daily. Alert if PSI > 0.2.
- **Prediction drift:** monitor daily mean predicted CTR vs actual CTR. Gap > 0.05% triggers recalibration.
- **Model performance:** compare daily AUC on logged data vs baseline. Drop >0.005 triggers incident.
- **Latency:** P50, P95, P99 serving latency. Alert if P95 > 10ms.
- **Retraining triggers:** scheduled (daily retrain + hourly fine-tune) + drift alert (emergency retrain if PSI > 0.3)

---

## Case Study 04: Real-Time Fraud Detection

**Scenario:** ML engineer at a payments company. Build the ML system detecting fraudulent transactions in real time.

### Step 1 — Clarifying Questions to Ask
- "What is the transaction volume and latency budget?"
- "What is the fraud rate and the cost ratio of false negative vs false positive?"
- "Do we block the transaction synchronously or flag for async review?"
- "What labeled data exists — chargebacks, manual review outcomes?"

*Assumed: 10K transactions/second, <50ms end-to-end, 0.1% fraud rate (1:1000 imbalance), FN cost is 10x FP cost (missed fraud >> false block), synchronous decision required, 18 months of labeled chargeback data.*

### Step 2 — ML Problem Formulation
- **Task:** binary classification — fraudulent (1) or legitimate (0)
- **Labels:** chargebacks confirmed as fraud (true positives); manual review clearances (true negatives). Labels are delayed 30–90 days.
- **Challenge 1:** extreme imbalance (1:1000). ROC-AUC is misleading; use precision-recall AUC and F1 at the operating threshold.
- **Challenge 2:** concept drift — fraud patterns evolve monthly as fraudsters adapt to defenses
- **Challenge 3:** label delay — training data is always stale relative to current fraud patterns. Mitigate with semi-supervised learning on recent unlabeled data.

### Step 3 — System Design

**Data:**
- Transaction stream (Kafka, 10K/s), user account features (MySQL), device fingerprint (Redis), merchant network graph (Neo4j)
- Feature freshness: velocity features computed in real-time via Flink; account history from Redis cache

**Features:**
- Transaction: amount, currency, merchant category code (MCC), time-of-day, online vs in-person
- Velocity: transactions in last 1h, 6h, 24h; total spend in last 24h; unique merchants in last 7d
- Deviation: amount vs user's 30d avg (z-score), merchant vs user's historical merchants (new=1)
- Device: is device new (registered <7d), location mismatch vs home country (yes/no), device fingerprint match
- Network: merchant fraud rate (7d rolling), IP subnet fraud rate, BIN (Bank ID Number) fraud rate

**Model Stack:**
- Layer 1 — Rule engine (<1ms): hard rules for obvious patterns (e.g., amount > $10K on new device, card used in 2 countries within 1h). Catches 30% of fraud instantly.
- Layer 2 — LightGBM scorer (<20ms): 200 features, trained on 18-month labeled set with class weights (10:1). Catches 60% of remaining fraud at 3% FPR.
- Layer 3 — GNN (async, <500ms): transaction + merchant + device graph embeddings. Used for post-hoc review queue prioritization, not real-time blocking.

**Imbalance handling:** Undersample negatives 10:1, use scale_pos_weight=10 in LightGBM, calibrate outputs with isotonic regression.

**Serving:**
- Rule engine → LightGBM scoring → threshold → block/approve
- All features fetched from Redis: <5ms. Model inference: <15ms. Total: <25ms P95.

### Step 4 — Offline Evaluation
- **Metrics:** Precision@K, Recall@K (at operating threshold K), F1, PR-AUC. Never use ROC-AUC as primary (misleading at 1:1000 imbalance).
- **Holdout:** temporal split — train on months 1–15, validate on month 16, test on months 17–18
- **Baseline:** rule engine alone. Beat it by >10% recall at same FPR.
- **Threshold tuning:** choose threshold that minimizes (10 × FN_cost + FP_cost) on validation set.

### Step 5 — Online Evaluation and A/B Test
- **Special constraint:** never A/B test a fraud model on 50% of live traffic. Fraudsters in the control group go undetected, causing real losses. Use shadow deployment: run new model in parallel, log its decisions, compare offline.
- **Primary metric:** fraud catch rate (recall at 0.5% FPR threshold)
- **Guardrail metric:** false positive rate (legitimate transaction decline rate). Target <0.5% to protect user experience.
- **Promotion criteria:** shadow model beats champion by >5% catch rate with <0.1% FPR increase, validated over 30-day shadow window.

### Step 6 — Production Monitoring
- **Data drift:** PSI on top-10 features daily. Alert PSI > 0.1.
- **Prediction drift:** mean model score on legitimate vs fraud segment daily. Unexpected convergence signals fraud pattern shift.
- **Performance proxy:** chargeback rate (7-day lag). Rising rate with stable model score = new fraud vector.
- **Retraining triggers:** weekly scheduled retrain on 90-day rolling window + alert-based retrain if chargeback rate rises >20% week-over-week.
- **Logging:** log all blocked transactions with reason code for model retraining and regulatory audit.

---

## Case Study 05: Search Ranking

**Scenario:** ML engineer at an e-commerce company. Redesign the product search ranking ML system.

### Step 1 — Clarifying Questions to Ask
- "What is the query volume and catalog size?"
- "What is the primary business metric — clicks, add-to-cart, or revenue?"
- "Is there existing click log data? How is it labeled?"
- "Real-time re-ranking or batch precomputation?"

*Assumed: 50M queries/day, catalog 10M products, <100ms response, optimize add-to-cart rate (not just CTR), 2 years of click logs available, fully online ranking.*

### Step 2 — ML Problem Formulation
- **Task:** learning-to-rank (LTR) — given query q and candidate set C, produce ranked list maximizing add-to-cart probability
- **Labels:** implicit feedback — clicks are weak positive signal; add-to-cart is stronger; purchases are strongest. Use a graded relevance label (0=skip, 1=click, 2=cart, 3=purchase).
- **Challenge:** position bias — rank-1 items get 10x more clicks than rank-10 regardless of relevance. Naive training on raw clicks learns position, not relevance. Must debias.

### Step 3 — System Design

**Data:**
- Query logs (50M/day), click/cart/purchase events, product catalog, user profiles
- Historical click data with position information for propensity estimation

**Features:**
- Query-product relevance: BM25 score (keyword overlap), BERT cross-encoder score (semantic match)
- Product quality: avg rating, review count, return rate (negative), inventory depth
- User personalization: user embedding similarity to product embedding, brand affinity, price range preference
- Context: query recency (trending vs long-tail), session position (first query vs refinement)
- Business: profit margin, seller tier, sponsored flag (must be surfaced but not over-ranked)

**Model:** two-stage pipeline
- Stage 1 — Retrieval (<20ms): BM25 (Elasticsearch) + ANN on product embeddings (FAISS) → top-500 candidates. Merge via reciprocal rank fusion.
- Stage 2 — Reranking (<80ms): LambdaRank with LightGBM. Input: 200 features per (query, product) pair. Output: relevance score. Rank top-500, serve top-20.

**Position bias correction:** train an inverse propensity scoring (IPS) model: P(click | position) estimated from randomized traffic (1% of queries get randomly shuffled results). Weight training examples by 1/propensity.

**Training:** weekly retrain on 30-day click log with IPS weights. Fine-tune personalization layer daily.

### Step 4 — Offline Evaluation
- **Metrics:** NDCG@10 (primary), MRR@10, Recall@100 (retrieval stage)
- **Position bias check:** compare NDCG on clicked-at-rank-1 vs clicked-at-rank-5 items. If rank-1 NDCG >> rank-5, bias is present.
- **Holdout:** temporal split. Last 7 days = test set. Evaluate on queries with ≥3 interactions.
- **Baseline:** BM25 alone. LTR model must beat by >5% NDCG@10.

### Step 5 — Online Evaluation and A/B Test
- **Primary metric:** add-to-cart rate per search session
- **Secondary metric:** revenue per session (RPM equivalent for search)
- **Guardrail:** zero-result rate (search quality), query abandonment rate (user left without clicking)
- **Test design:** 10% traffic (5% control, 5% treatment), 14-day run, stratified by query frequency (head/torso/tail separately)
- **Rollout:** 10% → 25% → 100% with metric checks at each stage

### Step 6 — Production Monitoring
- **Relevance drift:** NDCG@10 on a held-out annotated query set (500 queries with human labels), measured weekly
- **Feature drift:** PSI on BM25 score distribution (catalog changes affect BM25), product quality score distribution
- **Business metrics:** add-to-cart rate vs 7-day rolling average, daily. Alert on >10% relative drop.
- **Retraining triggers:** weekly scheduled + alert if add-to-cart rate drops >15% vs 7-day average

---

## Case Study 06: Customer Lifetime Value Prediction

**Scenario:** ML engineer at a subscription company. Predict 12-month CLV for each customer at signup.

### Step 1 — Clarifying Questions to Ask
- "How is CLV defined — revenue, profit, or gross margin?"
- "Are customers still active? How do we handle censored observations?"
- "What business decision does CLV feed — paid acquisition budget, upsell targeting, or something else?"
- "What signals do we have at signup vs after 30 days of behavior?"

*Assumed: CLV = total revenue over 12 months; 40% of customers are still active at 12m (censored); used to set paid acquisition bid caps; only signup + first 7 days of behavior available at prediction time.*

### Step 2 — ML Problem Formulation
- **Challenge:** censored data — for customers still active, we only know CLV so far, not final CLV. Standard regression on observed revenue underestimates CLV for active customers.
- **Two-part model formulation:**
  - Part A: P(churn within 12 months) — binary classifier
  - Part B: E[revenue | active for X months] — regression
  - Final CLV = P(stay) × E[revenue | stay] + P(churn at month t) × E[revenue | churn at t], integrated over t
- **Alternative:** Weibull-Gamma-Gamma model (BG/NBD for subscription context) — handles censoring natively via survival analysis

### Step 3 — System Design

**Data:**
- Signup features: acquisition channel, plan type, device, geo, promo code used
- Early behavior (days 1–7): logins, features used, content consumed, support tickets, payment success
- Historical cohorts: 36 months of subscribers with full 12-month outcome (training set)
- Censored observations: model with Weibull survival or use IPCW (inverse probability of censoring weights)

**Features:**
- Acquisition: channel (organic vs paid vs referral), promo depth (% discount), signup plan (monthly vs annual)
- Early engagement: DAU in week 1, feature adoption count, profile completion %, onboarding completion flag
- Payment: first payment amount, payment method (card vs PayPal), billing country risk score
- Content: content category affinity, search queries in week 1, social actions (shares, saves)

**Model:**
- Churn model: LightGBM classifier, trained on 36-month cohort with 12-month churn label. AUC target >0.78.
- Revenue model: XGBoost regressor on non-churned customers. Target RMSE <$15.
- Final CLV: churn_prob × 0 + (1 - churn_prob) × predicted_revenue (simplified 2-part model)

**Serving:** batch scoring at end of day 7 for all new signups. Store CLV score in data warehouse and feature store for downstream use.

### Step 4 — Offline Evaluation
- **Churn model:** AUC, PR-AUC, calibration curve
- **Revenue model:** RMSE, MAE, % within $20 of actual
- **CLV model:** Spearman rank correlation between predicted CLV and actual CLV (validated on older cohorts with complete 12-month data)
- **Business validation:** does top-decile CLV group generate 3x more revenue than bottom decile? (Lorenz curve check)

### Step 5 — Online Evaluation and A/B Test
- **Evaluation lag:** 12-month CLV requires 12 months to validate. Use 3-month CLV as a proxy (validate correlation between 3m and 12m CLV on historical cohorts first).
- **Business experiment:** use CLV score to set acquisition bid cap. Treatment: bid up to 1.2× CLV. Control: fixed bid. Measure CAC payback period at 6 months.
- **Guardrail:** don't overspend on customers predicted high-CLV but who churn early — monitor 30-day churn rate in high-CLV cohort.

### Step 6 — Production Monitoring
- **Prediction drift:** weekly distribution of CLV scores for new signups. Sudden shift = feature drift or model degradation.
- **Early validation:** track 30-day and 90-day actual revenue for each CLV decile. Compare to prediction. Escalate if top decile actual revenue < 80% of predicted.
- **Cohort tracking:** for each monthly cohort, track actual CLV progression at 3m, 6m, 9m, 12m vs prediction. Waterfall chart reviewed monthly.
- **Retraining:** semi-annual with latest 36-month cohort data. More frequent if cohort validation shows systematic bias.

---

## Case Study 07: Spam Detection

**Scenario:** ML engineer at an email provider. Build ML system classifying incoming emails as spam.

### Step 1 — Clarifying Questions to Ask
- "What is the email volume and latency budget?"
- "What is the cost asymmetry between false positive (legitimate email marked spam) and false negative (spam reaching inbox)?"
- "Do we have user feedback signals — 'mark as spam' and 'not spam'?"
- "Are spammers actively probing our system (adversarial setting)?"

*Assumed: 5B emails/day (~60K/sec), <200ms classification, FP (legitimate in spam) is severe — users lose important mail; adversarial setting where spammers actively try to evade detection.*

### Step 2 — ML Problem Formulation
- **Task:** binary classification — spam (1) or ham (0)
- **Labels:** user feedback ("mark as spam" = positive), user rescue ("not spam" from spam folder = negative override), automated honeypot catches (high-confidence positives)
- **Challenge 1:** adversarial — spammers reverse-engineer signal by probing edge cases; model must be robust
- **Challenge 2:** feedback loop — if a spam campaign evades detection for 1 hour, millions of users receive it before the model adapts
- **False positive asymmetry:** missing a spam email is annoying; sending a legitimate email to spam (e.g., a job offer) is catastrophic for trust

### Step 3 — System Design

**Data:**
- Incoming SMTP stream (60K/sec), user feedback events, honeypot inbox captures (100% spam, high confidence), historical labeled corpus (5 years)
- Sender reputation database updated hourly

**Features:**
- **Sender signals:** sender IP reputation score, domain age, DMARC/DKIM/SPF pass/fail, sending rate (anomaly vs baseline)
- **Content — text:** TF-IDF unigrams/bigrams on subject + body, BERT embedding of first 512 tokens, URL density, image-to-text ratio, HTML complexity
- **Content — structure:** attachment type and size, header anomalies, base64 encoding in body, obfuscated characters
- **Behavioral:** recipient engagement with this sender historically (open rate, click rate), user-level spam complaint rate for this sender

**Model:** two-path architecture
- Fast path (<5ms): rule engine + logistic regression on sparse TF-IDF features. Catches obvious spam; handles 70% of volume.
- Slow path (<200ms): BERT fine-tuned on spam corpus + gradient boosted ensemble. Applied to borderline cases from fast path.
- Ensemble: logistic combination of fast-path and slow-path scores with sender reputation as additional input.

**Adversarial robustness:** retrain weekly on new spam patterns from honeypots. Feature hashing for robustness to vocabulary injection attacks. Rate-limit probing via IP-level query throttling.

### Step 4 — Offline Evaluation
- **Primary metrics:** Recall@0.1%FPR (catch rate at very low false positive rate), PR-AUC
- **FP constraint:** precision at operating threshold must be >99.9% (less than 1 in 1000 legitimate emails marked spam)
- **Holdout:** temporal — last 7 days of labeled corpus. Stratified by email category (transactional, marketing, personal).
- **Adversarial eval:** held-out set of adversarially generated spam (generated by red team) — model must catch >80% with <0.01% FPR.

### Step 5 — Online Evaluation and A/B Test
- **Primary metric:** spam escape rate (spam reaching inbox / total spam sent), measured via honeypot daily
- **Guardrail metric:** false positive rate on sampled legitimate email stream (audited by human reviewers on 1K samples/day)
- **Test design:** shadow deployment. New model runs in parallel for 14 days; compare spam escape rate and FP rate vs champion. Promote only if FP rate does not increase.
- **Rollout:** shadow → 1% → 5% → 100%. Aggressive FP monitoring at each stage.

### Step 6 — Production Monitoring
- **Spam escape rate:** measured via honeypot hourly. Alert if >5% escape rate for any sender cluster.
- **FP rate:** human audit of 1K randomly sampled quarantined emails daily. Alert if FP rate > 0.05%.
- **User feedback rate:** "mark as spam" click rate and "not spam" rescue rate, daily by segment. Rising rescue rate = FP issue.
- **New campaign detection:** cluster new spam emails by content embedding. Novel cluster appearing rapidly = zero-day campaign → alert on-call, add to blocklist, trigger emergency retrain.
- **Retraining:** weekly scheduled + emergency trigger on new campaign detection.

---

## Case Study 08: Enterprise Semantic Search (LLM)

**Scenario:** ML engineer at a SaaS company. Build semantic search over 10M internal documents (wikis, tickets, contracts, runbooks).

### Step 1 — Clarifying Questions to Ask
- "What is the query volume and latency SLA?"
- "Are documents structured (same format) or heterogeneous?"
- "Do we have labeled query-document relevance pairs for fine-tuning?"
- "Privacy constraints — can documents leave the corporate boundary (cloud API vs on-prem)?"

*Assumed: 500K queries/day, <500ms, heterogeneous documents (PDF, HTML, Markdown), 5K labeled query-document pairs from user feedback, on-prem deployment required (data security).*

### Step 2 — ML Problem Formulation
- **Task:** retrieve and rank top-K documents most relevant to a natural language query
- **Output:** ranked list of document chunks with relevance scores and source citations
- **Formulation:** two-stage — dense retrieval (bi-encoder) + neural reranking (cross-encoder)
- **Why not BM25 alone:** BM25 fails on synonym matching ("layoff" vs "reduction in force"), paraphrase ("revenue" vs "income"), and concept queries ("what's the on-call process?")

### Step 3 — System Design

**Data:**
- 10M documents ingested via Kafka pipeline; chunked, embedded, indexed
- 5K labeled pairs: (query, relevant_doc, irrelevant_doc) triples for fine-tuning

**Chunking strategy:**
- Split documents into 512-token chunks with 50-token overlap (preserves cross-sentence context)
- Metadata preserved: document title, source URL, last-modified date, author, access control list (ACL)
- Chunk count: ~50M chunks for 10M documents at average 5 chunks/doc

**Embedding model:**
- Base: `intfloat/e5-large-v2` (768-dim) fine-tuned on company-specific data using contrastive learning with hard negative mining
- Fine-tuning: 5K labeled triples + 50K in-batch negatives. Triplet loss with margin=0.2.
- Embedding inference: batched, GPU-accelerated, <2ms per query at serving time

**Index:** FAISS IVF-PQ index over 50M chunks (768-dim → 64-dim PQ)
- Build time: ~4 hours on 8 GPUs
- Query time: <20ms for top-100 retrieval
- Index size: ~12GB in memory

**Reranking:** cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` fine-tuned)
- Input: (query, chunk_text) pair. Output: relevance score 0–1.
- Rerank top-50 retrieved chunks. Serve top-5.
- Latency: <50ms for 50 pairs on GPU

**Hybrid retrieval:** merge BM25 (Elasticsearch) and dense retrieval results using Reciprocal Rank Fusion (RRF) with k=60. This improves recall@10 by ~8% vs dense alone.

**Serving latency budget:**
- Query embedding: <10ms
- ANN retrieval (FAISS): <20ms
- Reranking (50 pairs): <50ms
- Total: <80ms P95 (well within 500ms SLA)

**Access control:** filter retrieved chunks by user ACL before reranking. Never surface documents the user does not have permission to view.

### Step 4 — Offline Evaluation
- **Retrieval quality:** Recall@100 (does the correct document appear in top-100?), target >90%
- **Ranking quality:** MRR@10, NDCG@10, target NDCG@10 >0.72 vs BM25 baseline of 0.61
- **Human eval:** 200-query annotation set with 3 human judges per query. Use Krippendorff's alpha >0.6 for inter-annotator agreement.
- **Holdout:** 500 labeled queries held out, not used in fine-tuning.

### Step 5 — Online Evaluation and A/B Test
- **Primary metric:** user satisfaction (did user click a result and spend >30s on the document?)
- **Secondary metric:** search-and-done rate (user searched, found answer, did not re-query)
- **Guardrail:** query abandonment rate (user clicked nothing)
- **Test:** 10% of users (5% control = BM25, 5% treatment = bi-encoder + reranker), 14-day run

### Step 6 — Production Monitoring
- **Index freshness:** new documents indexed within 1 hour via streaming pipeline (Kafka → embed → FAISS append). Monitor index lag metric.
- **Query latency:** P50, P95, P99 daily. Alert if P95 > 200ms.
- **Relevance drift:** weekly NDCG@10 on annotated set. Drop >5% triggers investigation.
- **Zero-result rate:** % of queries returning 0 results above threshold score. Rising rate = embedding drift or data quality issue.
- **Feedback loop:** thumbs up/down on results feeds back into fine-tuning data weekly.

---

## Case Study 09: Code Completion

**Scenario:** ML engineer at a developer tools company (GitHub Copilot-style). Build inline code completion for an IDE.

### Step 1 — Clarifying Questions to Ask
- "What languages are in scope? What is the latency SLA for inline completion?"
- "Cloud inference or on-device (local) deployment?"
- "What is the primary success metric — acceptance rate, user retention, or something else?"
- "What code context is available — current file only, or project-wide?"

*Assumed: top-5 languages (Python, JS, TypeScript, Java, Go), <50ms for single-line completion, <200ms for multi-line (5–10 lines), cloud inference (user must be online), project-wide context (open files + current file).*

### Step 2 — ML Problem Formulation
- **Task:** next-token prediction conditioned on code context — a standard language modeling task fine-tuned on code
- **Challenge 1:** latency — LLM inference is slow. A 70B model at 30 tokens/sec = 5s for a 150-token completion. Must use a smaller model or speculative decoding.
- **Challenge 2:** context window management — current file can be 10K tokens; project context adds more. Must intelligently truncate.
- **Challenge 3:** evaluation is hard — "correct" completion is subjective. Acceptance rate is the gold metric but has selection bias.

### Step 3 — System Design

**Data:**
- Pre-training: public code corpus (GitHub permissive licenses, 100B tokens)
- Fine-tuning: company/user-specific code patterns (opt-in only), 10B tokens
- Evaluation: 10K hand-annotated completions with human preference labels

**Model:**
- Base: 1.3B parameter decoder-only transformer (CodeLlama-style), optimized for low-latency inference
- Context window: 4K tokens — fill from: (1) prefix before cursor, (2) relevant open files (BM25-selected top-3), (3) suffix after cursor (fill-in-middle training)
- Speculative decoding: draft model (130M parameters) generates 5 candidate tokens per step; verification model (1.3B) validates in parallel. 2–3x throughput improvement at same quality.

**Context window management:**
- If current file > 2K tokens: keep 1K before cursor (most recent context) + 500 tokens from file top (imports, class definitions)
- Cross-file context: embed open files, retrieve top-3 similar snippets by BM25+embedding, inject as prefix comments
- Truncate from middle, not start — keeps imports and recent code both visible

**Serving:**
- Edge inference server co-located with IDE (plugin calls local or nearby server)
- INT8 quantized model: 2x throughput vs FP32
- Request batching: queue requests for 5ms, batch up to 8 for GPU efficiency
- P95 latency: <45ms for single-line, <180ms for multi-line (5 tokens)

### Step 4 — Offline Evaluation
- **Metrics:** HumanEval pass@1 (functional correctness on coding benchmarks), MBPP pass@1, edit similarity (how similar is generated code to what user actually typed), compilation success rate
- **Latency benchmark:** P50, P95, P99 across 10K test queries on target hardware
- **Baseline:** previous model version. Ship only if pass@1 improves by >2% with no latency regression.

### Step 5 — Online Evaluation and A/B Test
- **Primary metric:** acceptance rate — % of shown completions that the user accepts (does not delete within 30s)
- **Quality proxy:** edit distance between accepted completion and final committed code (low edit distance = high quality)
- **Guardrail:** user retention in IDE (do users keep the plugin enabled?), completion shown rate (are completions triggering at appropriate moments?)
- **Test:** 5% of users, 14-day run. Statistical significance at p<0.05 with Bonferroni correction for multiple metrics.

### Step 6 — Production Monitoring
- **Acceptance rate:** daily by language, file type, context length. Alert on >15% relative drop.
- **Latency:** P95 by model size and context length. Alert if P95 > 50ms for single-line.
- **Model staleness:** weekly evaluation on HumanEval benchmark. If pass@1 drops >3% vs baseline, trigger investigation (can indicate model weight corruption or serving bug).
- **Toxicity/security:** scan accepted completions for hardcoded secrets, license-incompatible code (snippet verbatim from GPL corpus). Alert and filter.
- **Retraining triggers:** monthly scheduled fine-tune on new accepted completions (with user opt-in). Quality regression triggers emergency investigation.

---

## Case Study 10: Document Q&A System (RAG)

**Scenario:** ML engineer at a law firm. Build a Q&A system over 1M legal documents (contracts, case law, briefs).

### Step 1 — Clarifying Questions to Ask
- "What is the query type — factual lookup, contract clause extraction, or legal reasoning?"
- "What is the tolerance for hallucination? (Legal context: zero tolerance.)"
- "What languages and jurisdictions are covered?"
- "How fresh must the document index be?"

*Assumed: factual lookup + clause extraction (not open-ended legal reasoning), zero hallucination tolerance (every claim must cite a source), English documents only, US and UK law, new documents indexed within 1 hour.*

### Step 2 — ML Problem Formulation
- **Task:** retrieval-augmented generation (RAG) — retrieve relevant document chunks, synthesize a grounded answer with citations
- **Constraint:** answer must be grounded — every sentence in the output must be traceable to a specific retrieved chunk. Hallucinated content is a legal liability.
- **Formulation:** retrieve top-K chunks → rerank → generate answer conditioned on chunks + explicit "only use the provided context" instruction → verify grounding

### Step 3 — System Design

**Data:**
- 1M legal documents (PDF → text extraction via pdfplumber), structured metadata (case number, jurisdiction, date, parties)
- Chunking: 512 tokens with 50-token overlap. Legal documents need sentence boundary respect — use spaCy sentence splitter before chunking.
- Chunk count: ~5M chunks total

**Retrieval pipeline:**
- Embed chunks with domain-specific legal embedding model (`legal-bert-base-uncased` fine-tuned on legal corpus)
- Store in pgvector (PostgreSQL extension) — chosen for transactional consistency and easy ACL enforcement
- At query time: embed query (768-dim), ANN search for top-100 chunks (<30ms)
- Hybrid: BM25 (for exact legal citation match: "42 USC 1983") + dense (for concept match). RRF merge.

**Reranking:** cross-encoder trained on legal Q&A pairs. Rerank top-100 → select top-5 for generation.

**Generation:**
- LLM (GPT-4-turbo or Claude 3 Sonnet) with system prompt: "Answer using ONLY the provided documents. If the answer is not in the documents, say 'Not found in provided documents.' Cite every claim with [Doc ID, paragraph]."
- Max input: top-5 chunks (~2500 tokens) + query + instruction = ~3500 tokens
- Output: answer with inline citations [1][2] linked to source chunks

**Hallucination detection:**
- NLI classifier (DeBERTa fine-tuned on NLI): for each sentence in answer, check if it is entailed by at least one retrieved chunk. Score = % of sentences entailed.
- Threshold: if entailment score < 0.95, flag answer for human review. Do not surface to user.

**Citation accuracy:** post-process answer to verify each [Doc ID] reference resolves to an actual retrieved chunk. Broken citations are rejected.

**Index freshness:** new documents ingested via Kafka → text extraction → chunking → embedding (batch GPU) → pgvector upsert. Target: <1 hour from document upload to searchable.

### Step 4 — Offline Evaluation
- **Retrieval:** Recall@5 (does correct document appear in top-5 chunks?), target >85% on 500 labeled queries
- **Answer quality:** Citation precision@3 (are top-3 cited documents actually relevant?), ROUGE-L vs human-written reference answers
- **Grounding rate:** % of answers with entailment score >0.95 on 200 human-annotated QA pairs
- **Hallucination rate:** human expert review of 100 random answers — target 0 factual errors

### Step 5 — Online Evaluation and A/B Test
- **Primary metric:** user satisfaction (thumbs up/down on each answer), target >85% thumbs up
- **Citation click rate:** do users follow citations to source documents? High click rate = trust in citations.
- **Guardrail:** hallucination incident rate (human review queue flag rate). Must stay <1%.
- **Test:** 20% of queries routed to new RAG pipeline vs keyword search baseline. 30-day run.

### Step 6 — Production Monitoring
- **Grounding score:** monitor NLI entailment score distribution daily. Drop in mean score = generation model drift or retrieval degradation.
- **Retrieval quality:** Recall@5 on 50-query daily probe set (manually labeled). Alert if drops below 80%.
- **Index lag:** document-to-searchable latency, P95. Alert if >2 hours.
- **Answer latency:** P50, P95 end-to-end (retrieval + generation). Alert if P95 > 8 seconds.
- **Human review queue:** review flagged answers within 4 hours. Patterns in flags → model improvement feedback.
- **Retraining:** fine-tune retriever monthly on user feedback (thumbs up/down converted to relevance labels). LLM prompt tuning quarterly.

---

## Case Study 11: Real-Time Personalization Engine

**Scenario:** ML engineer at a news app. Personalize the article feed for 50M users in real time.

### Step 1 — Clarifying Questions to Ask
- "What is the refresh rate for the feed and latency SLA?"
- "What user signals are available — explicit (ratings) or implicit (scroll, time-on-article)?"
- "How do we handle new users (cold start) and new articles?"
- "What is the primary business metric — engagement, session length, or subscriptions?"

*Assumed: feed refreshed on every app open (~5 refreshes/day/user = 250M requests/day), <150ms, implicit signals only (scroll depth, read time, shares), optimize session engagement (articles read per session), cold start is a first-class problem.*

### Step 2 — ML Problem Formulation
- **Task:** ranking — given user u at time t, rank candidate articles by predicted engagement probability
- **Labels:** read (scroll to bottom) = strong positive; click (opened) = weak positive; scroll past = negative; share/comment = very strong positive
- **Challenge 1:** feature freshness — user's interests change within a session (reading about a sports event → only wants more sports). Need real-time features, not just daily batch.
- **Challenge 2:** cold start — 5% of DAU are new users with no history. Must serve relevant feed without personalization data.
- **Challenge 3:** publisher fairness — don't starve small publishers by over-indexing on top-10 sources.

### Step 3 — System Design

**Feature Store architecture:**
- **Online features (Redis, <1ms lookup):** user's engagement in last 1 hour (article categories clicked, read time, searches), current session context (device, location, local time)
- **Offline features (Hive, refreshed daily):** user's 30-day category preferences, author affinities, publisher subscriptions, content engagement history
- **Item features (Redis, updated on publish):** article embedding (topic model), publisher, author, recency score (exponential decay by age), predicted virality

**Model:** two-tower neural network
- **User tower:** concat(user_embedding_256d, real_time_engagement_features) → 3-layer MLP → 128-dim user state
- **Article tower:** article embedding (128-dim, cached) — precomputed at publish time, never recomputed at serving
- **Score:** cosine similarity between user state and article embedding, scaled by recency and quality signals
- **Retraining:** daily on 7-day engagement log. Fine-tune real-time layer every 6 hours on last 6h of data.

**Cold start strategy:**
- 0 interactions: serve trending articles (top viral, geographically relevant) + topic diversification (one article per of 5 categories)
- 1–9 interactions: content-based (match article features to engaged articles). Weight recent actions 3x.
- 10+ interactions: transition to collaborative (two-tower model activates fully)
- Onboarding quiz (optional): ask for 3 topic interests at signup. Use as prior for first 10 interactions.

**Candidate generation:**
- For established users: ANN search on user embedding against article index (FAISS, 100K fresh articles) → top-200 candidates
- For new users: pre-computed trending bucket + geographic bucket → 200 candidates
- Reranking: two-tower score × recency × diversity penalty (reduce score if same publisher already in top-5)

**Publisher fairness:** constrained re-ranking — top-10 articles must include ≥5 unique publishers.

### Step 4 — Offline Evaluation
- **Metrics:** NDCG@10 on held-out engagement set, precision@5 (top-5 articles: did user engage?), cold-start NDCG (separate eval for new users)
- **Temporal split:** last 2 days = test set. No user/article overlap with training.
- **Baseline:** most-read articles globally (popularity baseline). Beat by >15% NDCG@10.

### Step 5 — Online Evaluation and A/B Test
- **Primary metric:** articles read per session (depth of engagement)
- **Secondary:** session length (minutes), return rate (7-day retention)
- **Guardrail:** publisher diversity score (HHI of publishers in served feed), click-bait rate (articles with high CTR but low read time)
- **Test:** 5% of users, 21-day run (capture weekly content patterns)

### Step 6 — Production Monitoring
- **Real-time features:** monitor Redis feature freshness lag. Alert if user real-time features are >15 min stale.
- **Model staleness:** compare daily NDCG on probe set (500 labeled sessions). Alert if drops >5% vs 7-day average.
- **Diversity:** track publisher HHI daily per user segment. Alert if top-10 publishers receive >60% of impressions.
- **Cold start quality:** track NDCG@10 separately for users with <10 interactions. Alert if drops >10%.
- **Feedback loop:** monitor whether model is creating filter bubbles — diversity score in user category distribution over 30 days. If category entropy drops >10%, inject exploration budget.
- **Retraining:** daily full retrain + 6-hourly fine-tune scheduled. Alert-based trigger on NDCG drop >8%.

---

## Case Study 12: Model Monitoring at Scale

**Scenario:** ML platform engineer. Design a monitoring system for 50 production ML models at a large tech company.

### Step 1 — Clarifying Questions to Ask
- "What types of models — classifiers, rankers, regressors, LLMs?"
- "What is the acceptable response time for detecting a problem?"
- "Do we have ground truth labels in real time or with a delay?"
- "What alerting and on-call infrastructure already exists?"

*Assumed: mix of 30 classifiers, 10 rankers, 5 regressors, 5 LLM-based; detection within 1 hour for severe issues, 24 hours for gradual drift; ground truth delayed by 1–7 days for most models; PagerDuty for alerting.*

### Step 2 — ML Problem Formulation
- **Problem:** model degradation in production manifests as data drift (input distribution shifts), concept drift (input-output relationship changes), or infrastructure failures (serving bugs, feature pipeline errors). Each requires different detection and response.
- **Challenge:** 50 models × many features = thousands of metrics. Alert fatigue if not carefully designed. Must prioritize signal-to-noise.
- **Framework:** four monitoring layers — data quality → feature drift → prediction drift → outcome drift (when labels arrive)

### Step 3 — System Design

**Monitoring pipeline:**
- Each model's inference logs streamed to Kafka (request features + predictions)
- Monitoring service consumes Kafka: computes metrics on sliding windows, writes to time-series DB (InfluxDB)
- Alerting: Grafana dashboards + PagerDuty for P1 alerts

**Layer 1 — Data quality (<1 min detection):**
- Null rate per feature. Alert if null rate > 2× baseline (suggests upstream feature pipeline failure).
- Schema validation: unexpected feature type or missing feature. Alert immediately.
- Volume anomaly: sudden drop in request rate >50% vs 24h average. Alert immediately.

**Layer 2 — Feature drift (hourly):**
- Population Stability Index (PSI) on top-30 features per model, computed hourly on last 1h vs reference distribution (30-day baseline)
- PSI interpretation: <0.1 = stable, 0.1–0.2 = monitor, >0.2 = alert
- Alert threshold: PSI > 0.2 on any top-10 feature, or average PSI > 0.15 across all features

**Layer 3 — Prediction drift (hourly):**
- For classifiers: KL divergence between current prediction distribution and 30-day reference. Alert if KL > 0.1.
- For rankers: monitor mean predicted score and score variance. Alert on >20% relative shift.
- For regressors: monitor predicted value distribution (mean, std, percentiles). Alert on >2σ shift.

**Layer 4 — Outcome drift (when labels arrive, daily/weekly):**
- For each model, compute accuracy proxy as labels trickle in (e.g., chargeback labels for fraud model)
- Track rolling 7-day AUC/RMSE vs 30-day baseline. Alert on >5% relative degradation.
- For models with no direct labels: use business proxy metrics (e.g., conversion rate for ranking model)

**Shadow model infrastructure:**
- Every production model has a challenger model running in shadow mode (receives all production traffic, logs predictions, never serves users)
- Weekly automated comparison: if challenger outperforms champion by >3% on monitored metric, alert for human review and potential promotion.

**Retraining triggers:**
- Scheduled: weekly retrain for all models on latest data
- Alert-based: PSI > 0.3 on critical feature → emergency retrain within 24h
- Outcome-based: rolling AUC drops >5% → retrain with extended lookback window

### Step 4 — Offline Evaluation
- **Monitoring system quality:** precision and recall of alert system on historical incidents. Target >80% recall (don't miss real problems), <20% false alert rate.
- **Drift detector calibration:** validate PSI thresholds on synthetic drift injection — inject known data shift, verify alert fires within expected window.

### Step 5 — Online Evaluation and A/B Test
- **Not applicable in traditional sense** — monitoring system is always-on
- **Rollout:** new monitoring system deployed alongside existing, alerts compared for 4 weeks before decommissioning old system

### Step 6 — Production Monitoring (Meta-monitoring)
- **Monitor the monitor:** track alert precision (are alerts actionable?) weekly via on-call post-mortems
- **Alert fatigue metric:** number of PagerDuty pages per week per engineer. Target <5. If >10, review and raise thresholds.
- **SLA:** P1 alerts (data quality failures) acknowledged within 15 minutes; P2 alerts (feature drift) reviewed within 4 hours; P3 alerts (prediction drift) reviewed within 24 hours.
- **Coverage:** % of models with full monitoring stack (all 4 layers). Target 100%. Audit quarterly.
- **Incident retrospectives:** after each model incident, root cause categorized and monitoring gap identified. Update alert thresholds accordingly.

---

## Case Study 13: Price Optimization

**Scenario:** ML engineer at a ride-sharing company. Design a surge pricing ML system.

### Step 1 — Clarifying Questions to Ask
- "What is the objective — maximize revenue, driver utilization, or completed rides?"
- "What geographic granularity — city, neighborhood, or 1km hexagon cells?"
- "Are there regulatory constraints on maximum surge multipliers?"
- "How frequently can prices update and how do we communicate changes to users?"

*Assumed: objective is to balance supply and demand (maximize completed rides, not raw revenue); geographic granularity = H3 hexagon cells (~1km radius, ~10K cells per city); regulatory cap = 3x surge in most markets; prices update every 5 minutes; users see surge multiplier before booking.*

### Step 2 — ML Problem Formulation
- **Task:** for each (cell, time_5min_window), predict optimal price multiplier m ∈ [1.0, 3.0] that maximizes completed rides
- **Two-sided market:** must model both demand elasticity (how does demand drop as price rises?) and supply response (how many drivers arrive in this cell within 10 min as a function of earnings?)
- **Formulation:** completed_rides = min(supply(m, t), demand(m, t)) — a constraint optimization problem, not a standard regression
- **Causal challenge:** price affects demand — observational data conflates high-price times (high natural demand) with price response. Need causal estimation of price elasticity.

### Step 3 — System Design

**Data:**
- Historical ride requests per cell per 5-min window (2 years), price multiplier applied, acceptance rate, completion rate
- Driver GPS pings (real-time), driver earnings per cell (rolling 30 min)
- External: local events (concerts, sports), weather, public transit disruptions
- Randomized pricing experiments: 5% of cells receive random price each 5-min window for causal identification

**Demand elasticity model:**
- Input: cell features (neighborhood type, day-of-week, hour, weather, event indicator), current demand pressure (queue depth), proposed price multiplier
- Output: P(user accepts ride at this price) — trained on randomized experiment data (causal)
- Model: XGBoost regression on log-demand. Price elasticity varies by cell type (airport = inelastic, suburban = elastic).

**Supply response model:**
- Input: cell, time, current driver earnings in adjacent cells, proposed multiplier
- Output: expected driver arrivals in next 10 minutes
- Model: LSTM over 2-hour driver position history per cell. Trained on historical driver movement.

**Optimization:**
- For each cell at each 5-min window: grid search over multipliers m ∈ {1.0, 1.1, ..., 3.0} to find m* = argmax(min(supply(m), demand(m)))
- Smooth price changes: cap multiplier change at 0.5x per 5-min window to avoid price shock
- Regulatory cap: clip at 3.0x. During disasters: clip at 1.0x (no surge allowed).

**Exploration via multi-armed bandit:**
- Each cell treated as a bandit arm. Use Thompson sampling to explore price multipliers with uncertainty.
- Exploration budget: 3% of cells receive a random multiplier each interval for continuous causal learning.

**Serving:** precomputed demand and supply model scores cached per cell; price computation runs in <100ms per cell; 10K cells updated every 5 minutes.

### Step 4 — Offline Evaluation
- **Demand model:** log-loss and calibration on held-out randomized experiment data. Must be calibrated: predicted acceptance at price X should match actual acceptance.
- **Supply model:** MAE on predicted driver arrivals vs actual arrivals, per cell type.
- **Simulation:** replay 30 days of historical data with proposed pricing policy vs baseline (average multipliers). Measure simulated completed rides, driver earnings, demand unfilled rate.

### Step 5 — Online Evaluation and A/B Test
- **Primary metric:** completed rides per hour per active cell
- **Secondary metric:** driver earnings per hour (must not decrease — driver trust), demand unfilled rate (requests with no driver matched)
- **Guardrail:** user satisfaction score, refund rate, app uninstall rate during high-surge periods
- **Test design:** city-level holdout (not user-level — price must be consistent within a cell). Treatment: 3 cities with new pricing. Control: 3 matched cities with existing pricing. Run 4 weeks.

### Step 6 — Production Monitoring
- **Real-time supply-demand balance:** unfilled demand rate per cell, every 5 minutes. Alert if >30% of requests in a cell go unmatched for >15 minutes.
- **Surge cap violations:** monitor for multiplier > regulatory cap. Zero tolerance — alert and auto-correct immediately.
- **Model drift:** demand elasticity estimate drift (causal estimate from ongoing randomized 3% budget). Alert if elasticity shifts >20% from baseline.
- **Driver earnings:** monitor P25 driver earnings per hour daily. If drops >10% vs 30-day average in any market, investigate and freeze pricing changes.
- **Public relations trigger:** monitor social media complaint rate about surge pricing (sentiment API). Spike in complaints triggers manual review of pricing events.
- **Retraining:** weekly demand model retrain, monthly supply model retrain. Continuous causal update from 3% exploration budget.

---

## Case Study 14: Multimodal Product Search

**Scenario:** ML engineer at an e-commerce platform. Enable image+text product search over 100M items.

### Step 1 — Clarifying Questions to Ask
- "What query modalities must be supported — image only, text only, or image+text combined?"
- "What is the catalog size and how frequently do new items appear?"
- "What is the latency SLA?"
- "Do we have human-labeled image-query relevance pairs for training?"

*Assumed: all three modalities (image, text, image+text), catalog 100M items, 10K new items/day, <300ms end-to-end, 20K labeled image-query relevance triples.*

### Step 2 — ML Problem Formulation
- **Task:** given a query q (image, text, or both), retrieve and rank the top-K most visually and semantically relevant products
- **Multimodal alignment:** queries and items must be embedded in a shared vector space where similarity is meaningful across modalities (image query can match text-described item)
- **Model family:** CLIP-style dual encoder — separate towers for query and item, trained with contrastive loss to align modalities in shared embedding space

### Step 3 — System Design

**Data:**
- 100M product items, each with: product image (primary), title (avg 15 words), attributes (color, size, material, brand), category path
- Query types: image crop (user uploads photo), text query ("blue suede shoes"), combined ("shoes like [image] but in red")
- Training data: 20K labeled triples (query, relevant_item, irrelevant_item) + 2M weak pairs from click logs (query, clicked_item)

**Model: CLIP-style dual encoder**
- **Query encoder:**
  - Text queries: BERT-base encoder → 512-dim projection
  - Image queries: ViT-B/16 encoder → 512-dim projection
  - Combined queries: concat(text_emb, image_emb) → MLP → 512-dim fusion embedding
- **Item encoder:**
  - Image: ViT-B/16 → 256-dim
  - Title + attributes: BERT → 256-dim
  - Concat → 512-dim item embedding (precomputed at index time, never at serving)
- **Training:** InfoNCE contrastive loss on in-batch negatives (batch size 4096) + hard negatives from BM25 top-20
- **Fine-tuning:** start from OpenCLIP checkpoint, fine-tune on 20K labeled triples + 2M click pairs for 3 epochs

**Index:**
- 100M × 512-dim FAISS IVF-PQ index
  - IVF: 65536 centroids (Voronoi cells) for fast coarse search
  - PQ: 64 subspaces × 8 bits = 64-byte per vector (vs 2KB for full float32)
  - Index size: ~6.4GB in memory for 100M items
  - Build time: ~4 hours on 8 GPUs
  - Query time: <30ms for top-100 retrieval (nprobe=128)

**Reranking:** cross-encoder over top-100 results. Input: (query_image + query_text, item_image + item_title). Output: relevance score 0–1. Latency: <80ms for 100 pairs.

**New item indexing:** streaming pipeline — new item published → image + text encoded (GPU batch) → embedded vector appended to FAISS index via online update. Target: <15 minutes from item publish to searchable.

**Serving latency budget:**
- Query encoding: <20ms (GPU)
- ANN retrieval (FAISS): <30ms
- Reranking (100 pairs): <80ms
- Total: <130ms P95 (well within 300ms SLA)

### Step 4 — Offline Evaluation
- **Retrieval:** Recall@10 (target >75%), Recall@100 (target >90%) on held-out 2K labeled queries
- **Ranking:** NDCG@10, MRR@10 on 500 human-annotated image queries (3 human raters, Krippendorff alpha >0.65)
- **Modality breakdown:** evaluate separately for text-only, image-only, and combined queries. Model must not regress any modality vs previous text-only baseline.
- **Baseline:** BM25 text search on product titles. Beat by >20% NDCG@10 on image queries.

### Step 5 — Online Evaluation and A/B Test
- **Primary metric:** add-to-cart rate per search session
- **Secondary metric:** image search adoption rate (% of searches using image input)
- **Guardrail:** text search relevance must not regress (NDCG@10 on text-only query sample)
- **Test:** 10% of users, 21-day run. Separate analysis for image search and text search user segments.
- **Rollout:** image search feature flagged behind beta → 5% → 25% → 100%, with metric check at each gate.

### Step 6 — Production Monitoring
- **Index freshness:** new-item-to-searchable latency P95. Alert if >1 hour.
- **Query latency:** P50, P95, P99 by query modality. Alert if P95 > 150ms.
- **Retrieval quality:** weekly Recall@10 on probe set of 200 annotated queries. Alert if drops >5%.
- **Embedding drift:** cosine similarity between today's and last-week's item embeddings for 10K sampled items. High drift signals model or data pipeline issue.
- **Cold start quality:** track Recall@10 separately for items indexed <7 days old vs established items. New items should achieve >60% of established item recall within 24 hours.
- **Retraining:** monthly fine-tune on new click logs (weak supervision). Quarterly full retrain on updated labeled set. Emergency retrain on embedding drift alert.

---

## Case Study 15: Cost Optimization for ML Platform

**Scenario:** ML platform lead. The GPU bill is $5M/month. Design a strategy to reduce it by 40% ($2M/month in savings) without degrading model performance.

### Step 1 — Clarifying Questions to Ask
- "What is the split between training compute and inference compute?"
- "What models drive the majority of cost — top-5 models by spend?"
- "Is there a latency SLA that constrains inference optimization?"
- "What is acceptable performance regression — 0%, <1%, or budget for trade-off discussion?"

*Assumed: 60% inference, 40% training; top-5 models drive 70% of cost; inference P95 SLA is 100ms for user-facing; <1% performance regression acceptable; current GPU utilization averages 35%.*

### Step 2 — ML Problem Formulation
- **Task:** cost optimization under performance and latency constraints
- **This is not an ML modeling problem** — it is an ML systems engineering problem with a measurement + optimization + validation loop
- **Framework:** Audit → Quick wins (no performance risk) → Model efficiency (measured regression) → Infrastructure (scheduling + hardware) → Sustain (measurement culture)

### Step 3 — System Design

**Phase 1 — Audit (Week 1, target: identify $3M in savings opportunities):**
- Pull GPU-hours and cost per model, per job type (training vs serving), per team
- GPU utilization report: average utilization, P5 utilization (identifies idle machines), utilization distribution
- Findings expected: 20–30% of training clusters idle overnight/weekends; serving replicas over-provisioned by 2–3x for most models; dev/experiment clusters running 24/7

**Phase 2 — Quick wins (Weeks 2–4, target: $1M/month savings, zero performance risk):**
- **Auto-shutdown idle dev clusters:** clusters idle >30 min automatically shut down. Saves 15–20% of dev compute. Implementation: spot instance + auto-shutdown policy. Estimated saving: $500K/month.
- **Spot instances for non-critical training:** fault-tolerant training jobs (can restart from checkpoint) moved to spot instances (60–70% cheaper than on-demand). Target: 70% of training jobs are fault-tolerant. Estimated saving: $400K/month.
- **Right-size serving replicas:** set min replicas based on P95 traffic + 30% headroom (not 2× buffer). Implement auto-scaling with custom metrics (GPU utilization, queue depth). Estimated saving: $300K/month.
- **Quick wins total: ~$1.2M/month**

**Phase 3 — Model efficiency (Weeks 4–12, target: $700K/month savings):**
- **Knowledge distillation for top-5 models:** train student models (3–5x smaller) using teacher outputs. Average 3x inference throughput improvement at <0.5% accuracy loss. Requires 2–4 weeks of engineering per model. Estimated saving: $400K/month (top-5 models drive 70% of inference cost).
- **INT8 quantization:** quantize top-10 inference models from FP32 to INT8 using TensorRT or ONNX Runtime. 1.5–2x throughput improvement, <0.3% accuracy loss on most models. 1-week per model. Estimated saving: $200K/month.
- **Flash Attention for transformer models:** replace standard attention with Flash Attention v2. 2–4x memory reduction, 1.5x throughput. Zero accuracy change. 2-day implementation per model. Estimated saving: $100K/month.
- **Model efficiency total: ~$700K/month**

**Phase 4 — Infrastructure (Weeks 8–16, target: $300K/month savings):**
- **Batch inference consolidation:** consolidate models serving <100 QPS onto shared GPU instances with time-sharing. Low-QPS models don't need dedicated GPU capacity. Estimated saving: $150K/month.
- **Multi-model serving:** deploy multiple small models on same GPU using NVIDIA MPS (Multi-Process Service). GPU memory: 40GB A100 can host 4–8 small models (≤7B parameters each). Estimated saving: $100K/month.
- **Reserved instance commitment:** for baseline serving capacity (bottom 70% of traffic), commit to 1-year reserved instances (40% cheaper than on-demand). Estimated saving: $50K/month.

### Step 4 — Offline Evaluation
- **For each optimization:** run A/B comparison on offline benchmark before deploying
  - Distilled model: AUC, NDCG, RMSE vs original model on held-out test set
  - Quantized model: same metrics + latency benchmark (P50, P95, P99)
  - Acceptance criteria: <1% relative metric degradation + no latency regression
- **Measurement baseline:** establish cost-per-prediction metric for each model before optimizations. Track improvement.

### Step 5 — Online Evaluation and A/B Test
- **For each model optimization:** shadow deployment for 7 days, then 1% A/B test for 7 days, then full rollout if metrics hold
- **Primary metric for optimization models:** same as the model's production metric (CTR, AUC, NDCG)
- **Guardrail:** latency SLA — optimized model must meet or beat original P95 latency
- **Cost tracking:** GPU-hours/prediction daily. Report weekly savings vs pre-optimization baseline.

### Step 6 — Production Monitoring (Sustainability)
- **Weekly cost report:** GPU-hours and cost by model, team, job type. Shared with all ML teams. Public dashboard visible to leadership.
- **Efficiency scorecard:** cost-per-prediction for top-20 models, tracked weekly. Teams own their model's efficiency score.
- **Budget alerting:** per-team GPU budget. Alert at 80% consumption. Require approval for >20% overage.
- **GPU utilization:** fleet-wide utilization target >60% (up from 35% baseline). Monitor daily. Underutilized clusters trigger auto-consolidation.
- **Performance regression monitoring:** all optimized models retain their original production monitoring. Any regression >1% triggers rollback and investigation.
- **Target achievement:** combined optimizations deliver ~$2.2M/month savings (44% reduction), exceeding the 40% target. Review quarterly and identify next $1M in opportunities.

---
