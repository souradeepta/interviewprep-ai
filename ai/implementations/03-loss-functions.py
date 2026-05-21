# 03 Loss Functions
# Extracted from Jupyter notebook

import numpy as np
import matplotlib.pyplot as plt

# Core loss functions from scratch
def mse(y, yhat):   return np.mean((y - yhat)**2)
def mae(y, yhat):   return np.mean(np.abs(y - yhat))
def bce(y, p):      p = np.clip(p, 1e-7, 1-1e-7); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
def hinge(y, yhat): return np.mean(np.maximum(0, 1 - y*yhat))  # y in {-1,+1}

np.random.seed(42)
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=float)
p_good = np.array([0.9,0.1,0.85,0.8,0.2,0.1,0.75,0.15])
p_bad  = np.array([0.6,0.4,0.55,0.5,0.5,0.4,0.6, 0.45])

for name, p in [("Good predictor", p_good), ("Bad predictor", p_bad)]:
    print(f"{name}: BCE={bce(y_true,p):.4f}, MSE={mse(y_true,p):.4f}")

# Visualise MSE vs MAE on a prediction error range
errors = np.linspace(-3, 3, 200)
plt.plot(errors, errors**2, label="MSE (squares errors)")
plt.plot(errors, np.abs(errors), label="MAE (linear)")
plt.legend(); plt.title("MSE vs MAE Loss Shape"); plt.xlabel("Error"); plt.show()

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

class EnhancedLossFunctions:

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



model = EnhancedLossFunctions()

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
model = EnhancedLossFunctions()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Focal loss for class imbalance
def focal_loss(y, p, gamma=2.0):
    p = np.clip(p, 1e-7, 1-1e-7)
    pt = np.where(y==1, p, 1-p)
    ce = -np.log(pt)
    return np.mean((1-pt)**gamma * ce)

# Imbalanced dataset: 95% negative, 5% positive
np.random.seed(0)
y_imb  = np.array([0]*190 + [1]*10)
p_imb  = np.random.beta(2, 5, 200)   # model biased toward low probs

print(f"BCE      : {bce(y_imb, p_imb):.4f}")
print(f"Focal γ=2: {focal_loss(y_imb, p_imb, gamma=2):.4f}")
print(f"Focal γ=5: {focal_loss(y_imb, p_imb, gamma=5):.4f}")
print("Higher γ → more focus on hard/minority examples")

# Example 3: Custom loss – asymmetric MAE (penalise underestimates more)
def asymmetric_loss(y, yhat, alpha=2.0):
    err = y - yhat
    return np.mean(np.where(err > 0, alpha*err**2, err**2))

np.random.seed(1)
y_reg  = np.random.exponential(5, 200)      # right-skewed (e.g. revenue)
yhat_reg = y_reg + np.random.randn(200)*0.5

sym_loss = mse(y_reg, yhat_reg)
asym_loss = asymmetric_loss(y_reg, yhat_reg, alpha=3.0)

print(f"Symmetric MSE   : {sym_loss:.4f}")
print(f"Asymmetric loss : {asym_loss:.4f}")
print("Use asymmetric when underestimating is more costly (e.g. inventory, safety)")