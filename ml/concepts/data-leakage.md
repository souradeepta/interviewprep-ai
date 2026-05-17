# Data Leakage

## TL;DR
Information from outside the training set leaks into features, causing inflated metrics that don't generalize. Examples: using future information, test data in preprocessing, time-based leakage. The most common source of "my model works in offline evaluation but fails in production."

## Core Intuition
Cheating. You're using an answer key during training, then surprised the model doesn't work when the answer key is gone. Every leakage source must be identified and eliminated.

## How It Works

**Temporal leakage:** using future information.
- Feature = "average purchase amount next 30 days" — won't be available at prediction time
- Fix: compute all features from data available at prediction time

**Scale leakage:** fitting on full dataset before train/test split.
- Scaling: compute mean/std on full data, apply to train and test → test data influenced training
- Fix: fit scaler on train only, apply to test

**Target leakage:** feature is derived from target.
- Example: predicting income, feature = "amount spent on luxury goods" (people with high income spend more)
- Fix: check if feature is available before the target event

**Group leakage:** test set contains similar examples as train.
- Time series: train on 2020, test on 2021 is OK. Train on 2021, test on 2020 is leakage
- User-based: if test set contains same users as train, model exploits user-specific patterns
- Fix: ensure test set is truly held-out (temporally later, different groups)

## Key Properties / Trade-offs
- **Hard to detect:** requires domain understanding
- **Cost of leakage:** high metrics → false confidence → production failure
- **Prevention:** rigorous train/test separation, think about prediction time, feature validation

## Common Mistakes / Gotchas
- **"But it works offline":** classic sign of leakage
- **Cross-validation without leakage:** still hard. StratifiedKFold handles class distribution but not temporal order
- **Feature selection:** if done on full data before CV, you've leaked information about test set

## Checklist
- ✓ All features available at prediction time?
- ✓ Preprocessing (scaling, encoding) fit only on train?
- ✓ Test set temporally later (for time series)?
- ✓ Test set contains different users/groups?
- ✓ Target information excluded from features?

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "What is data leakage?" | Test/future information contaminating training. Causes inflated metrics. |
| "Most common source?" | Preprocessing on full dataset. Always fit scaler/encoder on train only. |
| "How to detect?" | Large gap between offline and online metrics. Domain review of features. |

## Related Topics
- [Cross-Validation Strategies](cross-validation-strategies.md) — [Supervised Learning](supervised-learning.md)

## Resources
- [A Dirty Dozen: Twelve Common Data Mining Mistakes (Kaggle)](https://www.kaggle.com/wiki/FeatureLeak)
