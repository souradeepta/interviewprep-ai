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
