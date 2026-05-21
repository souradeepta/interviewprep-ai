# 16 Regularization
# Extracted from Jupyter notebook

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Over-parameterised problem
X, y = make_regression(n_samples=80, n_features=1, noise=15, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

poly = PolynomialFeatures(degree=12, include_bias=False)
Xtr_p, Xte_p = poly.fit_transform(Xtr), poly.transform(Xte)

for Model, name in [(Ridge(0.01),"Ridge λ=0.01"),(Ridge(10),"Ridge λ=10"),(Lasso(0.1),"Lasso")]:
    Model.fit(Xtr_p, ytr)
    tr, te = Model.score(Xtr_p, ytr), Model.score(Xte_p, yte)
    zeros = np.sum(np.abs(Model.coef_) < 1e-6)
    print(f"{name:20s}  Train R²={tr:.3f}  Test R²={te:.3f}  Zero coefs={zeros}")

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

class EnhancedRegularization:

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



model = EnhancedRegularization()

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
model = EnhancedRegularization()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Dropout in PyTorch
import torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import make_classification

X, y = make_classification(300, 20, n_informative=6, random_state=0)
Xt, yt = torch.FloatTensor(X), torch.LongTensor(y)

class DropoutNet(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20,128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64,2))
    def forward(self, x): return self.net(x)

for p in [0.0, 0.3, 0.5]:
    m = DropoutNet(p); o = optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(200):
        m.train(); o.zero_grad(); nn.CrossEntropyLoss()(m(Xt), yt).backward(); o.step()
    m.eval()
    with torch.no_grad(): acc = (m(Xt).argmax(1).numpy()==y).mean()
    print(f"Dropout p={p}: acc={acc:.4f}")

# Example 3: Early stopping
import torch, torch.nn as nn, torch.optim as optim

Xtr_t = Xt[:240]; ytr_t = yt[:240]; Xval_t = Xt[240:]; yval_t = yt[240:]

m = nn.Sequential(nn.Linear(20,128), nn.ReLU(), nn.Linear(128,2))
o = optim.Adam(m.parameters(), lr=1e-3)

best_val, best_ep, patience = 1e9, 0, 20
for epoch in range(500):
    m.train(); o.zero_grad(); nn.CrossEntropyLoss()(m(Xtr_t), ytr_t).backward(); o.step()
    m.eval()
    with torch.no_grad():
        val_loss = nn.CrossEntropyLoss()(m(Xval_t), yval_t).item()
    if val_loss < best_val: best_val, best_ep = val_loss, epoch
    elif epoch - best_ep > patience:
        print(f"Early stop at epoch {epoch}, best epoch={best_ep}"); break