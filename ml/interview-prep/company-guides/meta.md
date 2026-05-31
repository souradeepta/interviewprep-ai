# ML Interview at Meta

## What They're Looking For
Meta's ML challenges center on ranking and recommendations at 3B+ DAU scale. They value engineers who think in terms of real-time feature engineering, two-sided marketplace dynamics, and the tension between engagement metrics and long-term user wellbeing. PyTorch was built here — deep framework familiarity is expected. ML integrity (content moderation, misinformation detection) is a growing focus alongside ads and recommendations.

## Interview Rounds
| Round | Type | Duration | What's Tested |
|-------|------|----------|--------------|
| Phone Screen | ML System Design | 45 min | One system design at scale |
| Onsite 1 | ML Coding | 60 min | Implement ML from scratch (no sklearn) |
| Onsite 2 | ML System Design | 60 min | Large-scale ranking or recommendation system |
| Onsite 3 | ML Theory | 45 min | Deep theory: optimization, statistics, modeling |
| Onsite 4 | Behavioral | 45 min | Meta core values (Move Fast, Be Bold) |

## Most Common Question Topics
1. **Ranking systems** — News Feed, Reels, Marketplace. Expect: "How would you redesign News Feed ranking?"
2. **Ads optimization** — Click prediction at 10M QPS, position bias, calibration
3. **Embedding systems** — Two-tower models, embedding tables for sparse IDs (billions of user/item IDs)
4. **Integrity ML** — Content moderation, hate speech, misinformation. Adversarial dynamics.
5. **Feature engineering at scale** — Real-time feature pipelines, feature freshness tradeoffs
6. **Experimentation** — A/B testing at scale, network effects, interference between users

## Their ML Tech Stack (Known)
- **Framework:** PyTorch (primary), TorchScript for production
- **Training:** FBLearner Flow (internal ML platform), Tupperware (distributed training)
- **Serving:** Triton Inference Server, custom C++ serving
- **Features:** Internal feature store (Zipline), real-time + batch features
- **Search:** Faiss (they built it — vector search)
- **Data:** Hive, Presto, Spark

## Sample Questions from This Company

### Q: How would you redesign News Feed ranking to reduce misinformation without hurting engagement?
**What they're testing:** Values alignment + ML tradeoff reasoning

**Green flags:** Acknowledge tension between engagement and safety, propose multi-objective ranking (engagement + integrity score), A/B testing with both metrics, iterate based on data

**Red flags:** Ignoring the tradeoff, pure engagement optimization, no measurement plan

### Q: Design an ad click prediction system serving 10M QPS at <10ms latency.
**What they're testing:** Large-scale ML system design

**Green flags:** Two-tower model, embedding table sharding, near-real-time feature updates, calibration, position bias correction

**Red flags:** Suggesting training a new model per user, ignoring latency constraints

### Q: Your ranking model shows +2% engagement but -0.5% retention at 7 days. Do you ship?
**What they're testing:** Metric reasoning and business judgment

**Green flags:** Retention is a stronger signal; ask about statistical significance on retention; propose longer test; query which metric aligns with stated product goal

**Red flags:** "Ship it, engagement is up"

### Q: How do you handle position bias in training data from ranked lists?
**What they're testing:** Practical ML knowledge for ranking

**Green flags:** Inverse Propensity Scoring, counterfactual learning to rank, position-debiased models, unbiased evaluation (offline replay)

**Red flags:** Ignoring the bias, naive CTR as label

### Q: Walk me through how you'd build a two-tower model for recommendation.
**What they're testing:** Core Meta architecture pattern

**Green flags:** User tower (ID embedding + context), item tower (ID + features, cacheable), dot-product similarity, offline ANN index (Faiss), online scoring of top-K candidates

**Red flags:** Single model that takes (user, item) pair — doesn't scale to 10M items

## 3-Week Prep Strategy
**Week 1:** Study ranking systems. Read "Deep Learning Recommendation Model" (DLRM) paper. Implement two-tower from scratch. Review position bias and IPW.

**Week 2:** Practice large-scale system design. Do 3 mock designs (news feed, ads, content moderation). Study PyTorch internals and training at scale.

**Week 3:** Mock interviews. Behavioral prep with Meta values (Move Fast, Be Bold, Be Open, Build Social Value). Review A/B testing with network effects.

## Insider Tips
- Every system design answer should mention scale (orders of magnitude: QPS, users, items)
- Meta loves to ask about metrics tradeoffs — always name both the business metric and the guardrail metric
- Be prepared to code ML from scratch: logistic regression, gradient descent, AUC calculation — no sklearn
- Integrity (safety, misinformation) questions are increasingly common; know the ML approaches
