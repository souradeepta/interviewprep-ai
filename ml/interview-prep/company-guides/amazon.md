# ML Interview at Amazon

## What They're Looking For
Leadership Principles (LPs) are non-negotiable at Amazon — every ML answer should implicitly demonstrate an LP. Customer Obsession means starting with the customer impact, not the model. Dive Deep means being prepared for 5-level follow-up questions on any technical claim. Amazon's ML domains: demand forecasting (their lifeblood), product recommendations, Alexa NLP, AWS ML services (SageMaker, Rekognition, Comprehend).

## Interview Rounds
| Round | Type | Duration | What's Tested |
|-------|------|----------|--------------|
| Phone Screen | Behavioral + Technical | 60 min | 2 LP stories + 1 technical question |
| Onsite 1 | ML System Design | 60 min | Customer-facing ML system |
| Onsite 2 | ML Coding | 60 min | Implement ML algorithms |
| Onsite 3 | Behavioral | 60 min | 4-6 LP stories (STAR format) |
| Onsite 4 (Bar Raiser) | Mixed | 60 min | Raises the bar — can veto hire |

## Most Common Question Topics
1. **Demand forecasting** — Core Amazon business. Hierarchical forecasting, cold start for new products, uncertainty quantification
2. **Recommendation systems** — "Customers who bought X also bought Y", collaborative filtering, exploration
3. **LP-connected ML decisions** — How your ML choices reflect Customer Obsession, Dive Deep, Bias for Action
4. **NLP/Alexa** — NLU, slot filling, dialogue management, intent classification
5. **AWS ML services** — How to productionize ML on SageMaker; know the service portfolio
6. **Fairness and responsible AI** — Bias detection in recommendations, disparate impact across demographics

## Their ML Tech Stack (Known)
- **Framework:** MXNet (legacy), PyTorch (current), TensorFlow
- **Platform:** SageMaker (AWS), internal A9 search ML, Alexa NLU platform
- **Data:** AWS Redshift, S3, Kinesis, DynamoDB
- **Orchestration:** Step Functions, Airflow on MWAA

## Sample Questions from This Company

### Q: Tell me about a time you had to make a data-driven decision with incomplete information. (Bias for Action)
**What they're testing:** Bias for Action LP — act despite uncertainty

**Green flags:** Quantify what data you had, what was missing, what assumptions you made, how you validated afterward, what you'd do differently

**Red flags:** Waited for perfect data, no mention of the uncertainty tradeoff, no measurable result

### Q: Design a demand forecasting system for a new product category with no historical data.
**What they're testing:** ML problem solving + Customer Obsession

**Green flags:** Cold start via similar product features, hierarchical borrowing from category, causal variables (marketing spend, seasonality), Bayesian prior from domain knowledge, rapid iteration with early sales data

**Red flags:** "We can't forecast without data" — no creative solution

### Q: Your model is 95% accurate but customers are complaining about recommendations. What do you investigate?
**What they're testing:** Customer Obsession — metrics can lie

**Green flags:** Accuracy is the wrong metric. Check recommendation diversity, popular item bias, temporal relevance (stale items), business constraint violations (out-of-stock items recommended)

**Red flags:** "95% is good, ignore complaints" or no root cause investigation

### Q: How would you detect data drift in a demand forecasting model before it causes fulfillment failures?
**What they're testing:** Production ML monitoring and Dive Deep

**Green flags:** Monitor input feature distributions (KS test or PSI), track prediction distribution over time, set alert thresholds for key features (e.g., price, seasonality index), use holdout windows to detect degradation before it hits live predictions

**Red flags:** "Retrain weekly" without monitoring — reactive not proactive

### Q: Design the ML system for Alexa to understand "remind me to take my medication at 8pm daily."
**What they're testing:** NLU pipeline design end-to-end

**Green flags:** Intent classification (set_reminder), slot filling (entity: medication, time: 8pm, recurrence: daily), ambiguity resolution (which timezone?), confirmation dialog, downstream integration with reminder service

**Red flags:** Single classifier — misses slot filling, multi-turn dialog, and confirmation logic

## Amazon Leadership Principles — ML Context
Every behavioral story must connect to at least one LP. The most-tested LPs for ML roles:

| LP | ML Context | Story Pattern |
|----|-----------|---------------|
| Customer Obsession | Chose harder ML approach because it improved customer experience | Metric that mattered to customers vs. metric that was easier to optimize |
| Dive Deep | Investigated root cause of model failure 5 levels deep | Found data pipeline bug, not model bug, after weeks of investigation |
| Bias for Action | Shipped imperfect model with monitoring vs. waiting for perfect data | 70% solution shipped fast + iterated vs. 90% solution shipped 6 months later |
| Invent and Simplify | Built simpler ML solution that outperformed complex one | Replaced ensemble with single model, less maintenance, better accuracy |
| Are Right, A Lot | Changed ML approach based on data against team consensus | Disagreed with team, ran experiment, results proved new approach better |

## 3-Week Prep Strategy
**Week 1:** Write and practice 10 STAR stories covering all 16 Leadership Principles. Every story needs: specific numbers, YOUR decision (not team's), measurable result, what you learned.

**Week 2:** Technical prep — demand forecasting methods (DeepAR, N-BEATS, hierarchical models), collaborative filtering, SageMaker architecture. Practice 3 ML system designs.

**Week 3:** Mock behavioral interviews (record yourself). Connect every technical answer back to an LP. Research Amazon's recent ML blog posts and AWS re:Invent ML talks.

## Insider Tips
- Never say "we did X" — always say "I did X, the team did Y." Interviewers want YOUR contribution.
- Quantify everything: "improved accuracy by 8%" not "improved accuracy significantly"
- The Bar Raiser specifically looks for LPs you haven't demonstrated yet — prepare all 16, not just your favorites
- Amazon cares about business impact: end every ML story with revenue, cost savings, or customer metric
- Prepare 2 stories per LP minimum — Bar Raiser asks for different examples than prior rounds
