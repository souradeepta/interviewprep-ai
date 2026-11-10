# Probability and Statistics

**TL;DR:** Probability describes uncertainty in outcomes and models; statistics uses finite observations to estimate parameters, test claims, and quantify how uncertain those conclusions remain.

## Prerequisites

Prerequisites: algebra, proportions, and [supervised learning](supervised-learning.md). You can distinguish conditional from marginal probability, explain likelihood and maximum likelihood estimation, interpret sampling uncertainty and a p-value correctly, and identify when a statistical assumption is implausible. This ML lesson supplies the inferential foundation; the statistics track develops experiments and specialized methods in greater depth.

## Learning Outcomes

- Explain the central assumptions and mechanism of probability statistics.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Machine learning rarely gives certainty. A classifier's probability, an experiment's measured lift, and an estimated model parameter all depend on data that could have varied. Probability gives a language for uncertainty before or after observing evidence; statistics asks what a finite sample can tell us about an underlying process. The difference between a real pattern and sampling noise is not settled by a model score alone.

Suppose a medical test is positive. The probability of disease given a positive result depends on disease prevalence, test sensitivity, and false-positive rate; sensitivity alone does not answer the question. This base-rate example captures why conditional direction matters. Similarly, a small p-value is computed under a null model and does not tell us the probability the null is true or whether the effect is useful.

## How It Works

For events A and B with P(B)>0, Bayes' theorem is P(A|B)=P(B|A)P(A)/P(B). P(A) is prior prevalence, P(B|A) is the likelihood of evidence under A, and P(A|B) is the posterior after observing B. Random variables have distributions: Bernoulli represents one binary event, Binomial counts successes across repeated trials, Poisson models counts under assumptions about rate and independence, and Normal distributions model continuous variation and arise as approximations in many sampling settings.

Given independent observations x₁,…,xₙ from a model with parameter θ, likelihood is L(θ)=∏ᵢp(xᵢ|θ). Maximum likelihood chooses θ maximizing L, equivalently maximizing log L=Σᵢlog p(xᵢ|θ), which avoids numerical underflow and converts products to sums. For Gaussian observations with unknown mean and variance, the MLE mean is the sample mean and the variance estimate uses divisor n; the unbiased sample variance uses n−1. Maximum a posteriori estimation adds a prior, often corresponding to a regularization penalty. Likelihood is not the probability that a parameter value is true; it is the probability of observed data as a function of the parameter.

Sampling distributions describe how a statistic would vary across repeated samples. The standard error estimates this sampling variation; for an independent sample mean with finite variance, it is approximately s/√n. A confidence interval procedure with 95% coverage would contain the fixed target in 95% of repeated samples under its assumptions. It does not mean that a computed interval has 95% probability of containing a fixed frequentist parameter. Bootstrap resampling approximates uncertainty by repeatedly sampling observed rows with replacement, though clustered or time-dependent data require block/group-aware resampling.

Hypothesis tests compare data with a null model. A p-value is the probability, assuming the null and test assumptions, of observing a statistic at least as extreme as the one obtained. A significance threshold controls a decision rule, not the probability that a finding is false. Effect size and uncertainty should accompany significance. Testing many hypotheses raises the chance of at least one false positive; Bonferroni controls family-wise error, while Benjamini-Hochberg controls expected false discovery proportion under its assumptions. These corrections trade power for error control.

The central limit theorem says that, under conditions such as independent observations with finite variance, a suitably standardized sample mean approaches a Normal distribution as sample size grows. It does not claim raw data become Normal or that a small sample always qualifies. Dependence, heavy tails, selection bias, and measurement error can invalidate simple standard errors and tests. In ML, this perspective guides uncertainty estimates for metrics and experiments, but resampling units must match the data-generating process.

## Trade-offs and Selection Boundary

| Tool | Answers | Does not answer |
|---|---|---|
| Bayesian posterior | Parameter/event uncertainty conditional on prior and model | Whether prior/model assumptions are adequate |
| MLE | Parameter value maximizing observed-data likelihood | Uncertainty or regularization by itself |
| Confidence interval | Coverage-based uncertainty under a sampling procedure | Probability a fixed parameter lies in this realized interval |
| p-value | Compatibility of data with a specified null model | Probability the null is true or effect is important |
| Bootstrap | Empirical uncertainty approximation from resampled data | Valid inference when sampling units/dependence are wrong |

Use a statistical procedure only after defining the estimand: the quantity you want to learn. Specify population, outcome, comparison, and time window. Then check whether sampling, independence, distributional, and measurement assumptions are credible. For randomized experiments, preserve randomization units in analysis; for repeated users or sites, account for clustering. When assumptions are uncertain, report sensitivity analyses or use a design that collects better data rather than relying on a more elaborate formula.

## Implementation

This deterministic example computes a Bernoulli maximum-likelihood estimate and a simple bootstrap interval for the sample proportion.

```python
import random

observed = [1, 0, 1, 1, 0, 1, 0, 1]
p_hat = sum(observed) / len(observed)
assert p_hat == 5/8

rng = random.Random(17)
boot = []
for _ in range(2000):
    sample = [rng.choice(observed) for _ in observed]
    boot.append(sum(sample)/len(sample))
boot.sort()
lower, upper = boot[49], boot[1949]  # approximate central 95% interval
assert 0 <= lower <= p_hat <= upper <= 1
```

## Failure Modes and Diagnostics

1. **Base-rate fallacy:** a high-sensitivity test is assumed to imply a high posterior probability. Use Bayes' rule with actual prevalence and false-positive rate.
2. **p-value interpreted as truth probability:** “p=.03 means 97% chance the effect is real.” State the null, assumptions, and tail probability, then report effect size and interval.
3. **Multiple testing ignored:** searching many features or metrics produces chance winners. Predefine primary hypotheses or control family-wise/FDR error and validate on fresh data.
4. **Standard error mistaken for standard deviation:** uncertainty in the mean is confused with individual variation. Label which quantity is reported and show sample size.
5. **Invalid independence assumption:** repeated measurements from one person are treated as independent rows. Resample or model at the randomized/entity level and compare with cluster-aware uncertainty.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What does Bayes' theorem do? | Combines prior probability with the likelihood of evidence to obtain a posterior probability. |
| What is an MLE? | The parameter maximizing the likelihood of observed data; it is not itself a probability distribution over parameters. |
| Design follow-up: an experiment reports p<.05 from 40 metrics. | Ask which outcome was primary, quantify multiplicity, inspect effect size/interval, and validate the finding on independent data. |

## Related Concepts

- [Naive Bayes](naive-bayes.md), [evaluation metrics](evaluation-metrics.md), [class imbalance](class-imbalance.md)
- [Matching notebook](../notebooks/30-probability-statistics.ipynb)
- [Interview practice: A/B testing](../../system-design/patterns/14-ab-testing.md)

## References

- Wasserman, [All of Statistics](https://www.stat.cmu.edu/~larry/all-of-statistics/).
- NIST/SEMATECH, [e-Handbook of Statistical Methods](https://www.itl.nist.gov/div898/handbook/).
