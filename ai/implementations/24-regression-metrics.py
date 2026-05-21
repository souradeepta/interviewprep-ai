# 24 Regression Metrics
# Extracted from Jupyter notebook

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X, y = fetch_california_housing(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

reg = RandomForestRegressor(n_estimators=50, random_state=42).fit(Xtr, ytr)
ypred = reg.predict(Xte)

mse  = mean_squared_error(yte, ypred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(yte, ypred)
r2   = r2_score(yte, ypred)
mape = np.mean(np.abs((yte - ypred) / (np.abs(yte)+1e-9))) * 100

print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}  (same units as target)")
print(f"MAE : {mae:.4f}  (robust to outliers)")
print(f"R²  : {r2:.4f}  (1.0 = perfect, 0.0 = predict mean)")
print(f"MAPE: {mape:.2f}%")

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

class EnhancedRegressionMetrics:

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



model = EnhancedRegressionMetrics()

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
model = EnhancedRegressionMetrics()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Residual analysis
residuals = yte - ypred

fig, axes = plt.subplots(1,3, figsize=(15,4))
axes[0].scatter(ypred, residuals, alpha=0.3)
axes[0].axhline(0, color='red'); axes[0].set_title("Residuals vs Predicted")

axes[1].hist(residuals, bins=30); axes[1].set_title("Residual Distribution")

import scipy.stats as stats
stats.probplot(residuals, plot=axes[2]); axes[2].set_title("Q-Q Plot")

plt.tight_layout(); plt.show()
print(f"Residual mean: {residuals.mean():.4f} (should be ~0)")
print(f"Residual std : {residuals.std():.4f}")

# Example 3: MSE vs MAE – sensitivity to outliers
np.random.seed(42)
y_clean   = np.random.randn(100)
y_clean_pred = y_clean + np.random.randn(100)*0.5

y_out = y_clean.copy(); y_out[0] = 100  # one extreme outlier

for ytrue, name in [(y_clean,"Clean"),(y_out,"Outlier")]:
    mse_ = mean_squared_error(ytrue, y_clean_pred)
    mae_ = mean_absolute_error(ytrue, y_clean_pred)
    print(f"{name:8s}  MSE={mse_:.2f}  MAE={mae_:.2f}")
print("MSE inflated by outlier; MAE is robust")