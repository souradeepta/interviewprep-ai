# 21 Bias Variance Tradeoff
# Extracted from Jupyter notebook

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.sort(np.random.uniform(0,1,200))
y = np.sin(2*np.pi*X) + np.random.randn(200)*0.2

Xtr, Xte, ytr, yte = train_test_split(X[:,None], y, test_size=0.3, random_state=0)

fig, axes = plt.subplots(1,3, figsize=(15,4))
for ax, degree, label in zip(axes, [1,4,15], ["Underfit (deg=1)","Good (deg=4)","Overfit (deg=15)"]):
    poly = PolynomialFeatures(degree); Xp = poly.fit_transform(Xtr)
    m = LinearRegression().fit(Xp, ytr)
    Xs = np.linspace(0,1,200)[:,None]
    ax.scatter(*Xte.flatten()[:30], yte[:30], alpha=0.5, label="Test")
    ax.plot(Xs.flatten(), m.predict(poly.transform(Xs)), 'r')
    tr_err = np.mean((m.predict(Xp)-ytr)**2); te_err = np.mean((m.predict(poly.transform(Xte))-yte)**2)
    ax.set_title(f"{label}\nTrain={tr_err:.3f} Test={te_err:.3f}")
plt.tight_layout(); plt.show()

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

class EnhancedBiasVarianceTradeoff:

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



model = EnhancedBiasVarianceTradeoff()

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

model = EnhancedBiasVarianceTradeoff()

scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-val scores: {scores}")

print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Bias-variance decomposition
def bias_variance(model_fn, X_full, y_full, n_runs=50, test_size=0.3):
    preds = []
    Xte = np.linspace(0,1,30)[:,None]
    for _ in range(n_runs):
        idx = np.random.choice(len(X_full), size=100)
        m = model_fn().fit(X_full[idx], y_full[idx])
        preds.append(m.predict(Xte))
    preds = np.array(preds)
    bias2   = np.mean((preds.mean(0) - np.sin(2*np.pi*Xte.flatten()))**2)
    variance = np.mean(preds.var(0))
    return bias2, variance

for deg in [1, 4, 12]:
    poly = PolynomialFeatures(deg)
    def model_fn(d=deg):
        from sklearn.pipeline import make_pipeline
        return make_pipeline(PolynomialFeatures(d), LinearRegression())
    b, v = bias_variance(model_fn, X[:,None], y)
    print(f"degree={deg:2d}  bias²={b:.4f}  variance={v:.4f}  total≈{b+v:.4f}")

# Example 3: Regularisation shifts the tradeoff
from sklearn.linear_model import Ridge

Xp_full = PolynomialFeatures(15).fit_transform(X[:,None])
for alpha in [1e-5, 0.01, 1.0, 10.0]:
    m = Ridge(alpha).fit(Xp_full[:140], y[:140])
    tr = np.mean((m.predict(Xp_full[:140])-y[:140])**2)
    te = np.mean((m.predict(Xp_full[140:])-y[140:])**2)
    print(f"Ridge α={alpha:.4f}  Train={tr:.4f}  Test={te:.4f}")