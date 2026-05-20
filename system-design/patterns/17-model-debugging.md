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

## Related Topics
- [Evaluation Metrics](12-evaluation-metrics.md) in AI section
- [Drift Detection](15-drift-detection.md)

## Resources
- [Error Analysis Framework](https://www.deeplearning.ai/)
