# K-Means Clustering

## Detailed Explanation

Partitions data minimizing within-cluster sum of squares...

## Core Intuition

A key technique in machine learning.

## How It Works

1. Step 1
2. Step 2
3. Step 3

```mermaid
graph TD
    A[Input] --> B[Process]
    B --> C[Output]
```

## Architecture / Trade-offs

Trade-off 1 vs trade-off 2

## Interview Q&A

**Q: When would you use K-Means Clustering?**
A: Context-dependent, varies by problem type.

**Q: What are the main trade-offs?**
A: Refer to Architecture / Trade-offs section above.

**Q: How do you choose hyperparameters?**
A: Cross-validation, grid/random/Bayesian search, domain knowledge.

**Q: What are common failure modes?**
A: Refer to Common Pitfalls section below.

## Best Practices

- Practice 1
- Practice 2
- Practice 3

## Common Pitfalls

- Pitfall 1
- Pitfall 2


## Code Examples

### Example 1: Basic K-Means

```python
from sklearn.cluster import KMeans
from sklearn import datasets

X = datasets.load_iris()[0]

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

print(f"Cluster sizes: {np.bincount(labels)}")
print(f"Inertia: {kmeans.inertia_:.2f}")
```

### Example 2: Elbow Method

```python
inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 10), inertias, 'o-')
plt.xlabel('k'), plt.ylabel('Inertia')
plt.title('Elbow Method'), plt.show()
```

### Example 3: K-Means++ Initialization

```python
from sklearn.cluster import KMeans

# Standard k-means
km_random = KMeans(n_clusters=3, init='random', n_init=1, random_state=42)
km_random.fit(X)

# K-Means++
km_kpp = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
km_kpp.fit(X)

print(f"Random init inertia: {km_random.inertia_:.2f}")
print(f"K-means++ inertia: {km_kpp.inertia_:.2f}")
```

## Related Concepts

- [Gradient Descent](./01-gradient-descent.md)
- [Cross-Validation](./22-cross-validation.md)
- [Hyperparameter Tuning](./26-hyperparameter-tuning.md)
