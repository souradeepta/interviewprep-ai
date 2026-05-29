# Model Debugging

## TL;DR
Why is model wrong? Debug: (1) training data quality (bad labels), (2) feature engineering (wrong features), (3) model capacity (too simple), (4) hyperparameters (suboptimal). Use error analysis (slice predictions by cohort).

## Core Intuition
Model makes wrong predictions. Where does error come from? Trace back: data → features → training → hyperparameters.

## How It Works

**Error analysis framework:**
1. Collect errors (predictions where label ≠ prediction)
2. Slice by cohort (age, geography, user segment)
3. Identify patterns (where does model fail most?)
4. Root cause: data quality, feature engineering, model capacity
5. Fix: collect more data, engineer better features, increase capacity

| Error Type | Example | Root Cause | Fix |
|-----------|---------|-----------|-----|
| High on old users | 70% error on age>60 | Insufficient training data for age>60 | Collect more old users |
| High on rare items | 60% error on new products | Features don't capture new item patterns | Feature engineering |
| Systematic low confidence | Always <0.6 confidence | Model capacity too low | Use larger model |

## Key Properties / Trade-offs
- Time: error analysis is manual and slow
- Coverage: focus on high-impact errors first
- Complexity: root cause often multi-factorial

## Detailed Trade-off Analysis

| Aspect | Manual Error Analysis | Automated Slicing | ML-Based Anomaly | Active Learning |
|--------|----------------------|------------------|-----------------|-----------------|
| Detection time | 1-2 weeks | 1-2 days | Hours | Continuous |
| False positive rate | Low (curated) | 10-20% noise | 5% | Low (labeled) |
| Actionability | High (expert review) | Medium (automated) | High (novel patterns) | Highest (human loop) |
| Scalability | Low (manual) | Medium (rules) | High (patterns) | Medium (requires labeling) |
| Cost | Engineer time $10K | $500 tools | $1K tools | $2K+labels |

**Decision:** Small issues → manual. Scale issues → automated. Novel patterns → ML. Continuous → active learning.

---

## Production Failure Scenarios

**Scenario 1: Fix wrong thing**
- High error on age>60. Engineer assumes model capacity too low. Retrains on larger model. Error stays 70%.
- Root cause: training data had <1% age>60. Real fix: collect stratified data for age groups.
- Prevention: ask "why" 5 times before fixing. Verify hypothesis with ablation.

**Scenario 2: Label errors confound analysis**
- Error analysis shows "high error on region X". Start collecting more region X data.
- Later discovery: region X labels were wrong in original dataset (80% mislabeled).
- Prevention: validate label quality before error analysis. Run consensus labeling on subset.

**Scenario 3: Overfitting to error cohort**
- Fix age>60 errors by training specifically on age>60 data. Age>60 accuracy improves but age<30 accuracy drops.
- Prevention: error analysis includes fairness check—ensure fix doesn't degrade other cohorts.

**Scenario 4: Confusing correlation with causation**
- Error analysis: high error on Mondays. Add "day_of_week" feature. Accuracy improves.
- But root cause: model performs poorly on Sunday predictions (system issue), carry-over to Monday.
- Prevention: dig deeper into Monday errors. Are they truly Monday-specific or spillover from Sunday?

---

## Implementation Guidance

**Wrong:** Look at overall accuracy, make broad changes (retrain on more data).
**Right:** Slice errors by cohort, identify pattern, test hypothesis with ablation, fix root cause.

**Wrong:** Trust training labels. Assume they're correct.
**Right:** Validate 100 random labels before analysis. If >5% errors, stop and clean data first.

---

## Sophisticated Interview Q&A

**Q1: Model accuracy 90%. Where would you improve first?**
A: Error analysis to find low-accuracy cohorts. Example: 95% on common cases, 60% on rare cases. Prioritize by impact: rare case errors × business cost. If rare=0.1% of traffic but 100x higher cost, fix rare. Otherwise fix common.

**Q2: Age>60 has 70% error, age<20 has 80%. Which fix first?**
A: Calculate impact. Age>60: 5% of traffic × 70% error × cost. Age<20: 10% of traffic × 80% error × cost. If costs equal, fix age<20 (larger segment). Consider: which is easier to fix? (More data available? Simpler patterns?)

**Q3: Error decreases but fairness metrics worsen. What happened?**
A: Trade-off scenario. Reducing overall error may harm minority groups. Example: fix age<20 errors by favoring age<20 in training, accidentally decrease age>60 accuracy. Resolve: define acceptable fairness thresholds upfront. May not be possible to maximize accuracy AND fairness simultaneously.

**Q4: How do you prevent fixing the wrong problem?**
A: Hypothesis-driven debugging. (1) State hypothesis (e.g., "age>60 data insufficient"). (2) Design test (e.g., train on more age>60 data only). (3) Measure (did accuracy improve?). (4) Validate (does improvement persist on holdout?). If hypothesis fails, move to next.

---

## Cost & Resource Analysis

**Manual error analysis:** 1 engineer × 1 week = $5K. Identifies ~3-5 actionable cohorts per model.
**Automated slicing tools:** Evidently, Datarobot, etc. $200-500/month. Identifies patterns 10x faster.
**Active learning (ML-assisted):** Label unclear cases incrementally. Cost: $50-100 per label × 1000 labels = $50-100K for one iteration.

**ROI:** Error analysis preventing 1% accuracy drop on $10M platform = $100K value. Cost breakeven at one analysis per month.

---

## Monitoring & Observability

**Key metrics:** Error rate by cohort, error count by cause (label quality, feature missing, data insufficient), time spent in error analysis, cost to fix vs cost of error

**Alerts:** New cohort with >10% error rate, error rate drift (increasing errors on previously good cohort), label quality drops below threshold

## Common Mistakes / Gotchas
- Looking at overall accuracy only (missing specific cohort failures)
- Assuming training data is correct (actually has label errors)
- Fixing wrong thing (improve feature engineering when real issue is data)
- No prioritization (fix rare edge case instead of high-impact error)

## Best Practices
- **Start with overall accuracy:** understand baseline before diving into cohorts
- **Slice systematically:** age, geo, user segment, time period
- **Count errors:** high-impact cohort = many errors × high cost
- **Root cause analysis:** ask "why" 5 times before fixing
- **Validate fix:** make change, remeasure accuracy on error cohort

## Code Example
```python
import pandas as pd

def error_analysis(predictions, labels, features):
    errors = predictions != labels
    error_df = features[errors].copy()
    error_df['error'] = 1
    
    # Slice by age
    for age_group in ['<20', '20-40', '40-60', '>60']:
        mask = error_df['age'].isin(age_groups[age_group])
        error_rate = error_df[mask]['error'].mean()
        print(f"Age {age_group}: {error_rate:.1%} error")
```

## Interview Q&A
**Q: Model has 85% accuracy. Where to improve?**
A: Error analysis. (1) Find high-error cohorts. (2) Fix highest-impact one. (3) Remeasure. Example: model is 95% accurate for users >30, 70% for <30. Focus on <30: collect more data, engineer better features, or use separate model.

## Interview Quick-Reference
| Analysis | Output | Action |
|----------|--------|--------|
| Overall accuracy | 85% | Proceed to slices |
| Age slice | <20: 60%, >60: 90% | Focus on <20 |
| Root cause | Insufficient training data | Collect more |

## Failure Scenarios

### Failure 1: Debugging Iteration Takes Too Long
**Symptom:** Each debugging cycle takes 4+ hours to run, so the team can only test 1-2 hypotheses per day. Root cause identification drags on for weeks.
**Root Cause:** The team is running every debugging experiment on the full production dataset (100M rows). There is no curated debug dataset.
**Detection:** Measure time-to-reproduce a known failure. If reproducing a known issue takes more than 30 minutes of compute, the debug cycle is too slow for effective iteration.
**Fix:** Maintain a curated 1% debug dataset (approximately 1M rows) that reproduces all known failure modes in under 10 minutes. Update this dataset monthly by adding new failure patterns as they are discovered. All debugging experiments run on this dataset first; only validated fixes are tested at full scale.

### Failure 2: Error Analysis on a Uniform Random Sample
**Symptom:** Error analysis finds that the model struggles with "elderly users," but the team later discovers the actual failure is on a rare demographic intersection (elderly users making their first purchase) that represents only 0.3% of traffic. The uniform random sample had only 3 examples of this cohort.
**Root Cause:** Uniform random sampling is biased toward majority patterns. Rare-but-important failure modes are statistically underrepresented.
**Detection:** Compare the error breakdown from your sample against the known class distribution from production. If any class with > 5% production prevalence is represented by fewer than 100 samples in your analysis set, the sample is not representative enough for that class.
**Fix:** Use stratified sampling by confidence bucket: 0-20%, 20-40%, 40-60%, 60-80%, 80-100% confidence. Sample at least 200 examples per bucket. This ensures low-confidence (likely-to-fail) predictions are over-represented in the analysis, making rare failure modes visible.

### Failure 3: Train-Serve Skew Undetected
**Symptom:** Debugging identifies a feature preprocessing bug in the training code. The fix improves accuracy by 3% on the validation set. After deployment, production accuracy is unchanged.
**Root Cause:** The same preprocessing bug exists in the serving code. The training fix was valid but incomplete — the serving pipeline has its own implementation of the same logic, and it was not updated.
**Detection:** Run a parity test pipeline: generate features for 1,000 representative inputs using both the training pipeline and the serving pipeline, then compare distributions using a KS test. Flag any feature where KS test p < 0.05.
**Fix:** Implement parity testing as a blocking step in the deployment checklist. Standardize on a single feature computation library used by both training and serving, eliminating the dual-implementation problem at the source.

### Failure 4: Overfitting to the Debug Dataset
**Symptom:** The team fixes the model by iterating on a specific debug set until all known failure cases are resolved. After deployment, production error rate is unchanged.
**Root Cause:** The debug dataset is too small and biased toward specific failure types the team already knew about. The model learned to handle those exact cases but did not generalize to similar-but-different production failures.
**Detection:** Run the "fixed" model in shadow mode against production traffic for 48 hours before full rollout. Compare error rates between shadow and production distributions.
**Fix:** Maintain three separate debug sets: (1) known failures (curated); (2) a stratified random sample from recent production traffic; (3) adversarial examples generated by the team. A fix must show improvement on all three sets, not just the curated set. Use the shadow evaluation as the final gate before promotion.

### Failure 5: Confusing Correlation with Causation in Error Analysis
**Symptom:** Error analysis shows high failure rate on Monday predictions. Team adds a `day_of_week` feature. Monday accuracy improves. But two weeks later, the Monday pattern returns.
**Root Cause:** The true root cause was a Sunday batch pipeline job that runs weekly and corrupts the feature store for approximately 4 hours Sunday night. Predictions made in that window (late Sunday and early Monday) use corrupt features. `day_of_week` is a proxy for this pipeline timing issue, not a causal driver.
**Detection:** Drill down further: are the Monday failures concentrated in a specific hour of Monday? If so, the temporal pattern is more granular than day-of-week and indicates an event-driven cause.
**Fix:** When error analysis reveals a temporal cohort pattern, investigate the pipeline schedule for that time window before adding features. Map the failure window against all scheduled jobs that touch feature inputs.

---

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| ML engineer debugging time | $200/hr | 8 hr/week | $6,400 |
| Debug dataset labeling (initial creation) | $0.10/sample | 10,000 samples | $1,000 (one-time) |
| Debug dataset labeling (monthly refresh) | $0.10/sample | 2,000 samples/month | $200 |
| GPU compute for debug experiment runs | $2/hr | 40 hr/month | $80 |
| Automated slicing tool (Evidently/DataRobot) | — | SaaS license | $300 |
| Log and prediction storage for analysis | $0.023/GB | 5 GB/day | $3.50 |
| **Total** | | | **~$7,984/month** |

Engineer time dominates the cost of model debugging at approximately 80% of the total. Every hour saved in the debug cycle (faster reproduction, automated slicing, pre-built cohort dashboards) has an outsized ROI. Automated slicing tools that identify high-error cohorts in hours instead of days can reduce debugging cycles from 2 weeks to 2 days, saving $8-16K per investigation in engineer time.

---

## Interview Q&A

**Q1: Model accuracy 90%. Where would you improve first?**
A: Run error analysis to find low-accuracy cohorts. Example: 95% overall hides 60% accuracy on a high-value user segment that represents 15% of revenue. Prioritize by impact: error_rate × segment_size × business_value. Fix the highest-impact cohort first, not the largest absolute error count.

**Q2: Age > 60 has 70% error, age < 20 has 80% error. Which to fix first?**
A: Calculate business impact: (segment_size × error_rate × cost_per_error). Age < 20 may be 10% of traffic while age > 60 is only 5% — at equal per-error cost, fix age < 20 first. Also consider ease of fix: if more training data is available for age > 60, that may be the faster win. Fix the problem you can actually solve soonest with highest impact.

**Q3: Error decreases but fairness metrics worsen. What happened?**
A: Classic accuracy-fairness trade-off. Optimizing for overall accuracy often harms minority groups because the training objective is indifferent to cohort-level performance. Example: adding more training data for age < 20 improves that cohort but the model re-allocates capacity away from age > 60. Resolution: define fairness constraints (minimum accuracy per protected group) before training and optimize under those constraints, not unconstrained accuracy.

**Q4: How do you prevent fixing the wrong problem?**
A: Hypothesis-driven debugging: (1) state the hypothesis ("age > 60 data is insufficient"); (2) design a minimal test (train on a dataset with 3× more age > 60 samples, all else equal); (3) measure (did age > 60 accuracy improve?); (4) validate on a held-out set the test never touched. If the hypothesis fails, move to the next. Never make multiple changes simultaneously — you cannot isolate the cause.

**Q5: When would you NOT do manual error analysis?**
A: When the model has thousands of output classes (e.g., a multi-label tagger with 5,000 categories), manual cohort analysis does not scale. Switch to automated slicing (Slice Finder, Domino) combined with ML-based anomaly detection on error patterns. Also skip manual analysis when the error rate is below the noise floor for your sample size (e.g., 0.1% errors need millions of samples to slice meaningfully).

**Q6: What breaks first when model debugging scales to 100 models in production?**
A: The debug dataset maintenance becomes a bottleneck — 100 curated debug datasets require dedicated data engineering effort to keep fresh. Automate: route a stratified sample of recent production predictions (including errors) to a shared debug dataset store daily, shared across all models. Each model's debug set refreshes automatically without manual curation.

**Q7: Label errors are found in 15% of training samples. What do you do?**
A: Do not proceed with error analysis until label quality is fixed. Label errors confound every downstream analysis. First: identify the labeling pipeline failure (UI bug, ambiguous guidelines, annotator bias). Second: re-label the affected subset. Third: audit a random 500-sample holdout from each labeler for ongoing quality control. Error analysis on dirty labels inverts priorities — you will "fix" the wrong cohorts.

**Q8: Model has 85% accuracy. Where to improve?**
A: Error analysis in order: (1) find high-error cohorts by slicing (age, geography, recency, user segment); (2) validate labels in the highest-error cohort to rule out annotation errors; (3) diagnose root cause (data volume, feature quality, model capacity); (4) fix and remeasure. The most common finding: the highest-error cohort is systematically underrepresented in training data.

## Related Topics
- [Evaluation Metrics](12-evaluation-metrics.md) in AI section
- [Drift Detection](15-drift-detection.md)

## Resources
- [Error Analysis Framework](https://www.deeplearning.ai/)
