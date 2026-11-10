# Bootstrap and Resampling

TL;DR: Bootstrap resampling approximates an estimator’s sampling uncertainty by repeatedly drawing independent observational units with replacement from the observed sample.

## Prerequisites

Review [probability fundamentals](01-probability-fundamentals.md), [estimation](04-maximum-likelihood-estimation.md), and [confidence intervals](06-confidence-intervals.md).

## Learning Outcomes

- Construct percentile bootstrap intervals for a statistic by resampling the correct observational unit.
- Explain why the empirical distribution stands in for repeated sampling and identify its assumptions.
- Distinguish ordinary, paired, cluster, and block resampling for common data structures.
- Recognize small-sample, extreme-tail, and non-identifiability limitations that more replicates cannot fix.

## Core Intuition

Imagine that the observed sample is the only miniature population available. A bootstrap replicate draws from its observed members with replacement, preserving the sample size, then recomputes the statistic. Repeating this gives an empirical picture of how the statistic would vary if similar samples were collected. An observation can appear multiple times in a replicate or not at all; that variability is the mechanism, not a bug.

For example, a service team has a small sample of response times and wants uncertainty for the median. The median’s sampling distribution is awkward to derive under a long right tail. Resampling response times and recalculating the median can expose skew in the estimator’s uncertainty. The method cannot invent unseen slow events, correct a biased sample, or turn ten observations into evidence about a rare 99.9th percentile. Its miniature population is only as representative as the original sample.

## How It Works

Let the observed independent data units be (X_1,ldots,X_n), with empirical distribution (hat F_n), assigning probability (1/n) to each observation. For bootstrap replicate (b), draw (X^*_{b1},ldots,X^*_{bn}) independently from (hat F_n), then calculate (hat\theta_b^*=T(X^*_{b1},\ldots,X^*_{bn})). The distribution of (hat\theta_b^*) approximates sampling variation of (hat\theta) under regularity and representativeness assumptions. A percentile interval takes quantiles of these replicate estimates, such as the 2.5th and 97.5th percentiles for nominal 95% coverage.

The replicate count B controls Monte Carlo error in estimated quantiles, not the statistical information in the data. Increasing B from 1,000 to 100,000 may make the empirical quantile smoother; it does not reduce sampling uncertainty or fix poor coverage. Computational cost is generally O(Bn) statistic evaluations, with additional storage from O(B) values if all replicates are retained. Streaming quantile algorithms can reduce memory for very large B.

The resampling unit must match the independent sampling unit. If users contribute repeated sessions, resample users as clusters and retain all sessions within each selected user. For time series, ordinary resampling destroys temporal dependence; block bootstrap resamples contiguous segments under assumptions about stationarity and block length. For a paired experiment, resample pairs, not treatment and control observations independently. This is the same design discipline used for standard errors: dependence belongs in the uncertainty calculation.

## Trade-offs

| Method | Appropriate structure | Strength | Main limitation |
|---|---|---|---|
| Percentile bootstrap | Independent, representative units; regular statistic | Simple and captures asymmetry | Coverage can be biased in small samples or for irregular statistics |
| BCa interval | Smooth statistic with enough observations | Adjusts bias and acceleration | More implementation complexity; fails for degenerate resamples |
| Parametric bootstrap | Trusted parametric model | Can extrapolate beyond observed support | Wrong model assumptions create misleading intervals |
| Cluster bootstrap | Independent clusters with within-cluster dependence | Respects grouped observations | Few clusters provide weak approximation |
| Block bootstrap | Dependent time series with appropriate stationarity | Preserves local dependence | Results depend on block design and do not handle arbitrary regime shifts |

## Implementation

This dependency-light example resamples units and computes a percentile interval for the median. A fixed seed makes the result reproducible. The example estimates uncertainty conditional on the sampled values; it does not perform bias correction or claim good tail coverage from a tiny sample.

```python
import random

random.seed(16)
sample = [2, 3, 3, 4, 5, 9, 13, 21]

def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2

replicates = [
    median([random.choice(sample) for _ in sample])
    for _ in range(5000)
]
replicates.sort()
interval = (replicates[124], replicates[4874])
assert len(replicates) == 5000
assert interval[0] <= median(sample) <= interval[1]
print("median and percentile interval:", median(sample), interval)
```

For a cluster bootstrap, first group rows by user, sample user IDs with replacement, and append all rows belonging to each sampled ID. If a user is selected twice, the cluster contributes twice. For a paired treatment design, represent each subject’s before/after values as one resampling unit. These adaptations preserve the original dependence structure rather than pretending every row was independently sampled.

## Failure Modes

- **Wrong resampling unit:** Sampling individual events from clustered users understates uncertainty. Diagnose repeated IDs; resample the top-level independently sampled entity or use an explicit hierarchical model.
- **Unrepresentative empirical support:** A convenience sample omits a population segment. Bootstrap replicates cannot recreate the missing segment; fix sampling or state the target population limitation.
- **Very small samples or extreme quantiles:** Replicates contain only a few unique values, and the empirical 99th percentile cannot reveal rare outcomes. Collect more independent units or use a defensible tail model and sensitivity analysis.
- **Non-smooth statistics:** The ordinary bootstrap may fail for boundary estimates, maxima, or some selected-model statistics. Compare against theory, use a method designed for the estimator, and avoid assuming more replicates guarantee validity.
- **Dependent time series:** IID resampling destroys autocorrelation. Inspect dependence and use block-based inference or a time-series model whose assumptions fit the process.
- **Treating Monte Carlo precision as data precision:** More bootstrap draws stabilize computed quantiles but do not narrow the true sampling uncertainty. Report both sample size and interval method.

## Interview Quick Reference

| Question | Answer | Follow-up |
|---|---|---|
| What does the bootstrap approximate? | The sampling distribution of a statistic using the empirical distribution as a proxy for the population. | Which units are independent? |
| How would you bootstrap repeated measures? | Resample subjects or clusters and retain their associated observations. | How many independent clusters do you have? |
| When can bootstrap intervals fail? | Small samples, irregular estimators, extreme tails, dependence, or an unrepresentative sample can invalidate the approximation. | Debug a suspiciously narrow interval. |

## Related Concepts

Continue with [confidence intervals](06-confidence-intervals.md) and [Monte Carlo sampling](11-monte-carlo-sampling.md). The paired notebook is [16-bootstrap-resampling.ipynb](../notebooks/16-bootstrap-resampling.ipynb). Practice the selection question in [Statistics interview practice](../interview-prep/statistics-interview-questions.md#6-select-a-bootstrap-procedure).

References: [Efron, “Bootstrap Methods: Another Look at the Jackknife” (1979)](https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full) and [Davison and Hinkley, *Bootstrap Methods and Their Application*](https://www.cambridge.org/core/books/bootstrap-methods-and-their-application/6F0D89E22F9A82F6F8339B2F0D13D5E1).
