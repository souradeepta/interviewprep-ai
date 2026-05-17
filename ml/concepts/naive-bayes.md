# Naive Bayes

## TL;DR
A probabilistic classifier based on Bayes' theorem with a strong independence assumption: features are conditionally independent given the class. Simple, fast, and surprisingly effective for text classification and spam filtering despite the naive assumption.

## Core Intuition
Bayes' rule: $P(y|x) = \frac{P(x|y)P(y)}{P(x)}$. The "naive" assumption is that features don't interact — knowing feature 1 doesn't change feature 2's probability given the class. This is almost never true, but simplifies computation dramatically.

## How It Works

**Assumption:** $P(x_1, x_2, ..., x_d | y) = \prod_i P(x_i | y)$

**Posterior:** $P(y|x) \propto P(y) \prod_i P(x_i | y)$

**Prediction:** $\hat{y} = \arg\max_y P(y) \prod_i P(x_i | y)$

**For text:** count word frequencies. $P(word_i | spam)$ = (times word appears in spam + 1) / (total words in spam + vocab size). Add-one smoothing prevents zero probabilities.

## Key Properties / Trade-offs
- **Speed:** O(d) per prediction — linear in feature count
- **Small data:** works well with limited training examples
- **Independence violation:** performs well anyway (bias-variance: high bias, low variance)
- **Probabilities:** outputs calibrated probabilities, not just class assignments

## Common Mistakes / Gotchas
- **Laplace smoothing:** always use it or zero probabilities destroy everything
- **Numerical underflow:** use log probabilities instead of raw probabilities
- **Feature scaling:** doesn't need it (probability-based, not distance-based)

## Code Example
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = ["win money now", "meeting tomorrow", "free cash"]
labels = [1, 0, 1]  # 1=spam, 0=ham
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
nb = MultinomialNB()
nb.fit(X, labels)
print(nb.predict(vectorizer.transform(["win free cash"])))  # [1]
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "What does naive mean?" | Assumes features are conditionally independent given class. Rarely true, but works in practice. |
| "When use Naive Bayes?" | Text classification (spam, sentiment), quick baseline, small datasets. |
| "Why Laplace smoothing?" | Prevents zero probabilities from zero-count features. |

## Related Topics
- [Supervised Learning](supervised-learning.md) — [Probability & Statistics](probability-statistics.md)

## Resources
- [Naive Bayes: Simple but Effective](https://towardsdatascience.com/naive-bayes-explained-9fefb1c78ecf)
