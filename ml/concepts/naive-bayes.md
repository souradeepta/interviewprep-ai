# Naive Bayes

**TL;DR:** Naive Bayes combines class priors with feature likelihoods under conditional independence, producing a fast probabilistic baseline that is especially effective for sparse text.

## Prerequisites

Prerequisites: Bayes' rule, conditional probability, and [supervised learning](supervised-learning.md). After this lesson, you can derive the Naive Bayes decision rule, choose among Gaussian, multinomial, and Bernoulli variants, apply smoothing and log probabilities, and explain why useful classification accuracy does not imply calibrated probabilities. This ML lesson focuses on the model's assumptions and boundaries; the AI material introduces classification more broadly.

## Learning Outcomes

- Explain the central assumptions and mechanism of naive bayes.
- Apply the method to a small deterministic example and interpret its output.
- Compare this method with a plausible alternative using a task-relevant criterion.
- Diagnose a characteristic failure and choose a validation or mitigation step.

## Core Intuition

Suppose an email contains the words “prize,” “claim,” and “meeting.” Bayes' rule starts with how common spam is, then updates that prior using how likely each observed word is under spam versus legitimate mail. Naive Bayes treats the words as conditionally independent once the class is known, so it multiplies simple likelihoods instead of modeling every word combination. The independence assumption is often wrong: “not good” is a phrase whose meaning depends on word interaction. Yet the resulting ranking can still be effective because a decision boundary may be good even when the probability estimates are imperfect.

“Naive” describes the factorization, not the quality of the method. Its low estimation variance is valuable when data is scarce or features are high-dimensional and sparse. Conversely, correlated predictors can cause the model to count the same evidence repeatedly. A strong baseline is useful for diagnosing whether a complex model's improvement is real.

## How It Works

For class y and feature vector x=(x₁,…,x_d), Bayes' rule gives P(y|x)∝P(y)P(x|y). Conditional independence approximates P(x|y)=∏ⱼP(xⱼ|y). Prediction selects the class maximizing log P(y)+Σⱼlog P(xⱼ|y); taking logs avoids underflow from multiplying many small values. Priors P(y) come from class counts or explicit assumptions. Likelihood estimates depend on feature type.

Gaussian Naive Bayes models each continuous feature within each class as a Gaussian with a class-specific mean and variance. Multinomial Naive Bayes suits nonnegative counts such as bag-of-words term frequencies; its likelihood is parameterized by category counts. Bernoulli Naive Bayes models binary feature presence and absence, so absent words also contribute evidence. These distributions are not interchangeable: feeding standardized negative values to MultinomialNB violates its count interpretation.

For multinomial counts cⱼᵧ, additive smoothing estimates P(word j|y)=(cⱼᵧ+α)/(Σₖcₖᵧ+αV), where vocabulary size V and α>0. Smoothing assigns nonzero mass to unseen words but also spreads probability, so α is a hyperparameter. Complexity is roughly O(nd) to estimate sparse feature counts and O(nnz(x)) per prediction, where nnz counts nonzero input features. The model is easy to update from sufficient statistics for some variants, though class vocabulary changes and forgetting old data require explicit policy.

Predicted probabilities may be poorly calibrated, particularly when correlated evidence is counted several times. If the deployment action depends on calibrated risk, evaluate reliability and consider held-out calibration such as sigmoid or isotonic mapping. For text classification where ranking or top class is the goal, Naive Bayes may still be a useful fast model even with imperfect probability values.

Vocabulary construction and tokenization are part of the fitted model. Case folding, punctuation handling, stemming, and n-gram choices change which events are counted. Fit a vocabulary on training text only; if a validation or test corpus contributes its words, this is usually a transductive assumption and should be stated. Out-of-vocabulary tokens are ignored or mapped according to the implementation, so a large production vocabulary shift can silently reduce evidence. Monitor unknown-token rate and compare the training and serving preprocessing versions.

Class priors deserve explicit attention. A prior based on the training sample is sensible only when that sample reflects deployment prevalence or when prior shift is corrected. If the dataset was deliberately balanced for convenience, the posterior ranking may remain useful while probabilities and default decisions are shifted. Decide whether prior correction is justified from operational evidence, and validate after correction at natural prevalence rather than assuming the original output is a production risk estimate.

## Trade-offs and Selection Boundary

| Variant | Data assumption | Good fit / boundary |
|---|---|---|
| Gaussian NB | Continuous feature likelihood is approximately normal per class | Fast numeric baseline; correlated features can distort confidence |
| Multinomial NB | Nonnegative counts/frequencies | Sparse text counts; invalid for arbitrary negative-valued features |
| Bernoulli NB | Binary feature presence and absence both matter | Short text or binary indicators; absent-feature modeling matters |
| Complement NB | Class-conditional complement counts address skewed text classes | Often useful for imbalanced text; validate against ordinary NB |

Prefer Naive Bayes when latency, simplicity, small data, or a sparse-text baseline matters. Prefer logistic regression when you want a linear discriminative model that does not need the feature-independence likelihood, or a tree/boosting model for nonlinear tabular interactions. Use [class imbalance](class-imbalance.md) methods when priors and costs matter; do not assume the learned prior matches production prevalence after resampling.

## Implementation

This miniature multinomial classifier uses deterministic token counts and additive smoothing, then calculates a stable log score.

```python
from collections import Counter
import math

docs = [("cheap prize now", "spam"), ("cheap offer", "spam"),
        ("team meeting", "ham"), ("project meeting now", "ham")]
classes = sorted({label for _, label in docs})
vocab = sorted({word for text,_ in docs for word in text.split()})
alpha = 1.0
counts = {c: Counter() for c in classes}
totals = Counter()
for text, label in docs:
    for word in text.split(): counts[label][word] += 1; totals[label] += 1

def score(text, label):
    token_counts = Counter(text.split())
    prior = sum(y == label for _,y in docs) / len(docs)
    return math.log(prior) + sum(n * math.log((counts[label][w]+alpha) /
        (totals[label]+alpha*len(vocab))) for w,n in token_counts.items())

prediction = max(classes, key=lambda c: score("cheap meeting", c))
assert prediction in classes
assert math.isfinite(score("unseen token", "spam"))  # smoothing prevents -infinity
```

## Failure Modes and Diagnostics

1. **Raw probability underflow:** multiplying many tiny likelihoods returns zero for every class. Sum log probabilities and use a stable implementation.
2. **Wrong likelihood family:** MultinomialNB receives centered or negative features. Check feature domain; choose GaussianNB or another classifier for continuous signed inputs.
3. **Overconfident predictions from correlated words:** independence counts near-duplicate evidence multiple times. Compare reliability curves and log loss, not only accuracy; calibrate when probability semantics matter.
4. **Zero counts dominate:** unseen words make a class score negative infinity. Apply additive smoothing and tune α on validation data.
5. **Prior shift after resampling:** training class frequencies do not reflect deployment. Evaluate with natural prevalence and adjust priors or calibration based on justified assumptions.

## Interview Quick Reference

| Question | Answer |
|---|---|
| What does “naive” mean? | Conditional independence of features given class, which makes likelihood estimation tractable. |
| Why can NB work despite a false assumption? | The assumption can yield a useful decision ranking with low estimation variance, even when probabilities are imperfect. |
| Design follow-up: a text model has good accuracy but poor risk estimates. | Keep NB as a ranking baseline if useful, assess calibration on held-out natural-prevalence data, and calibrate or use a discriminative model for risk decisions. |

## Related Concepts

- [Probability and statistics](probability-statistics.md), [class imbalance](class-imbalance.md), and [evaluation metrics](evaluation-metrics.md)
- [Matching notebook](../notebooks/26-naive-bayes.ipynb)
- [Interview practice: classification metrics](../interview-prep/evaluation-experiment-questions.md)

## References

- McCallum and Nigam, [A Comparison of Event Models for Naive Bayes Text Classification](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf).
- scikit-learn, [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html).
