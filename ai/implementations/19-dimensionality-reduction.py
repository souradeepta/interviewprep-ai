# 19 Dimensionality Reduction
# Extracted from Jupyter notebook

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

pca = PCA(n_components=4)
pca.fit(X)

# Scree plot
ev = pca.explained_variance_ratio_
plt.bar(range(1,5), ev, label='Individual')
plt.step(range(1,5), np.cumsum(ev), where='mid', color='red', label='Cumulative')
plt.axhline(0.95, linestyle='--', color='gray', label='95% threshold')
plt.xlabel("Component"); plt.ylabel("Explained variance ratio")
plt.legend(); plt.title("PCA Scree Plot"); plt.show()

pca2 = PCA(n_components=2)
X_2d = pca2.fit_transform(X)
plt.scatter(*X_2d.T, c=y, cmap='tab10', alpha=0.8)
plt.title(f"PCA 2D  (explains {pca2.explained_variance_ratio_.sum():.1%})")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.colorbar(); plt.show()

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

class EnhancedDimensionalityReduction:

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



model = EnhancedDimensionalityReduction()

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
model = EnhancedDimensionalityReduction()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: t-SNE vs PCA on MNIST-like data
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

digits = load_digits()
X_d, y_d = digits.data[:500], digits.target[:500]

X_pca  = PCA(n_components=2).fit_transform(X_d)
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_d)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))
ax1.scatter(*X_pca.T,  c=y_d, cmap='tab10', alpha=0.7); ax1.set_title("PCA")
ax2.scatter(*X_tsne.T, c=y_d, cmap='tab10', alpha=0.7); ax2.set_title("t-SNE")
plt.tight_layout(); plt.show()

# Example 3: PCA as pre-processing to speed up training
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

X_d2, y_d2 = load_digits().data, load_digits().target
Xtr, Xte, ytr, yte = train_test_split(X_d2, y_d2, test_size=0.2, random_state=42)
sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

for n_comp in [None, 30, 15]:
    if n_comp:
        pca = PCA(n_components=n_comp).fit(Xtr_s)
        Xtr_p, Xte_p = pca.transform(Xtr_s), pca.transform(Xte_s)
    else:
        Xtr_p, Xte_p = Xtr_s, Xte_s
    t0 = time.time()
    svm = SVC().fit(Xtr_p, ytr)
    elapsed = time.time() - t0
    print(f"Dims={Xtr_p.shape[1]:3d}  acc={svm.score(Xte_p,yte):.4f}  time={elapsed:.2f}s")