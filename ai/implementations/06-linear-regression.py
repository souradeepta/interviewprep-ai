# 06 Linear Regression
# Extracted from Jupyter notebook

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

X, y, coef = make_regression(n_samples=200, n_features=3, noise=10, coef=True, random_state=42)

# Closed-form OLS
X1 = np.c_[np.ones(len(X)), X]
theta_ols = np.linalg.lstsq(X1, y, rcond=None)[0]
y_pred = X1 @ theta_ols

mse  = np.mean((y - y_pred)**2)
r2   = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
print(f"OLS  MSE={mse:.2f}  R²={r2:.4f}")
print(f"True coef:    {np.round(coef,2)}")
print(f"Learned coef: {np.round(theta_ols[1:],2)}")

plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("OLS Predictions"); plt.show()

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

class EnhancedLinearRegression:

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



model = EnhancedLinearRegression()

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
model = EnhancedLinearRegression()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Ridge vs Lasso – sparsity comparison
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# High-dimensional: 3 true features, 17 noise
X_hd, y_hd = make_regression(n_samples=150, n_features=20, n_informative=3,
                               noise=10, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X_hd, y_hd, test_size=0.2, random_state=0)

sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

for Model, name in [(LinearRegression,"OLS"),(Ridge(1.0),"Ridge"),(Lasso(1.0),"Lasso")]:
    m = Model.fit(Xtr_s, ytr)
    r2 = m.score(Xte_s, yte)
    zeros = np.sum(np.abs(m.coef_) < 1e-4)
    print(f"{name:6s}  R²={r2:.4f}  zero-coefs={zeros}/20")

# Example 3: Regularisation path (Lasso)
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt

alphas, coefs, _ = lasso_path(Xtr_s, ytr)

plt.figure(figsize=(10,5))
for c in coefs:
    plt.plot(-np.log10(alphas), c)
plt.xlabel("-log10(alpha)  ← more regularisation   more features →")
plt.ylabel("Coefficient value")
plt.title("Lasso Regularisation Path – coefficients driven to zero")
plt.axhline(0, color='k', linewidth=0.5)
plt.show()