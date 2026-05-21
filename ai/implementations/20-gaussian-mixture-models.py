# 20 Gaussian Mixture Models
# Extracted from Jupyter notebook

from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=[0.5,1.0,0.7], random_state=42)

gmm = GaussianMixture(n_components=3, covariance_type='full', n_init=5, random_state=42)
gmm.fit(X)

hard  = gmm.predict(X)
soft  = gmm.predict_proba(X)

plt.scatter(*X.T, c=hard, cmap='tab10', alpha=0.7)
plt.scatter(*gmm.means_.T, marker='X', s=200, c='red', label='Means')
plt.title(f"GMM (BIC={gmm.bic(X):.1f})"); plt.legend(); plt.show()

print("Soft assignment for first sample:", soft[0].round(3))

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

class EnhancedGaussianMixtureModels:

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



model = EnhancedGaussianMixtureModels()

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
model = EnhancedGaussianMixtureModels()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Choose n_components via BIC / AIC
bics, aics = [], []
ks = range(1, 9)
for k in ks:
    g = GaussianMixture(n_components=k, n_init=5, random_state=0).fit(X)
    bics.append(g.bic(X)); aics.append(g.aic(X))

plt.plot(list(ks), bics, 'o-', label='BIC')
plt.plot(list(ks), aics, 's-', label='AIC')
plt.xlabel("n_components"); plt.ylabel("Score (lower=better)")
plt.legend(); plt.title("Model selection: BIC / AIC"); plt.show()
print(f"BIC-optimal k={list(ks)[np.argmin(bics)]}, AIC-optimal k={list(ks)[np.argmin(aics)]}")

# Example 3: GMM vs K-Means on elongated clusters
np.random.seed(0)
X_ell = np.vstack([
    np.random.randn(150,2) @ [[3,0],[0,0.5]] + [0,0],
    np.random.randn(150,2) @ [[0.5,0],[0,3]] + [5,5]])

from sklearn.cluster import KMeans
km_labels  = KMeans(n_clusters=2, random_state=0).fit_predict(X_ell)
gmm_labels = GaussianMixture(n_components=2, covariance_type='full', random_state=0).fit_predict(X_ell)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
ax1.scatter(*X_ell.T, c=km_labels,  cmap='tab10'); ax1.set_title("K-Means (poor on ellipses)")
ax2.scatter(*X_ell.T, c=gmm_labels, cmap='tab10'); ax2.set_title("GMM (handles ellipses)")
plt.show()