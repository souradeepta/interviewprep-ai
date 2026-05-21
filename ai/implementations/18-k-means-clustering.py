# 18 K Means Clustering
# Extracted from Jupyter notebook

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=42)

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels = kmeans.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', alpha=0.7)
plt.scatter(*kmeans.cluster_centers_.T, marker='X', s=200, c='red', label='Centroids')
plt.legend(); plt.title(f"K-Means (inertia={kmeans.inertia_:.1f})"); plt.show()

# Level 2: Production pattern with error handling



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



# Split data

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42

)



# Preprocessing

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



# Enhanced model with validation

class EnhancedKMeansClustering:

    def __init__(self, **kwargs):

        self.params = kwargs

        self.scaler = StandardScaler()



    def fit(self, X, y):

        X = self.scaler.fit_transform(X)

        # Fit algorithm

        self.mean_ = np.mean(X, axis=0)

        return self



    def predict(self, X):

        X = self.scaler.transform(X)

        return X @ np.ones(X.shape[1])



    def score(self, X, y):

        pred = self.predict(X)

        mse = np.mean((pred - y) ** 2)

        return mse



model = EnhancedKMeansClustering()

model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)

test_score = model.score(X_test, y_test)

print(f"Train MSE: {train_score:.4f}, Test MSE: {test_score:.4f}")

# Example 1: Practical application

from sklearn.datasets import load_iris

from sklearn.model_selection import cross_val_score



data = load_iris()

X, y = data.data, (data.target == 0).astype(int)



# Cross-validation

model = EnhancedKMeansClustering()

scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-val scores: {scores}")

print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Elbow method + silhouette score to choose k
from sklearn.metrics import silhouette_score

ks, inertias, silhouettes = range(2, 10), [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, km.labels_))

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
ax1.plot(list(ks), inertias, 'o-'); ax1.set_title("Elbow (inertia)")
ax2.plot(list(ks), silhouettes, 'o-', color='orange'); ax2.set_title("Silhouette score (higher=better)")
plt.tight_layout(); plt.show()
print(f"Best k by silhouette: {list(ks)[np.argmax(silhouettes)]}")

# Example 3: K-Means limitations – non-spherical clusters
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

Xm, _ = make_moons(n_samples=200, noise=0.1, random_state=0)

fig, axes = plt.subplots(1,2, figsize=(12,4))
km_labels = KMeans(n_clusters=2, random_state=0).fit_predict(Xm)
db_labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(Xm)

axes[0].scatter(*Xm.T, c=km_labels, cmap='tab10'); axes[0].set_title("K-Means (fails on moons)")
axes[1].scatter(*Xm.T, c=db_labels, cmap='tab10'); axes[1].set_title("DBSCAN (correct)")
plt.show()