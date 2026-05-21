# 15 Weight Initialization
# Extracted from Jupyter notebook

import numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt

def activation_stats(init_fn, n_layers=10, n_neurons=256, n_samples=500):
    X = np.random.randn(n_samples, n_neurons)
    stds = [X.std()]
    for _ in range(n_layers):
        W = init_fn(n_neurons)
        X = np.maximum(0, X @ W)   # ReLU
        stds.append(X.std())
    return stds

zero_init   = lambda n: np.zeros((n,n))
small_init  = lambda n: np.random.randn(n,n) * 0.01
xavier_init = lambda n: np.random.randn(n,n) * np.sqrt(1/n)
he_init     = lambda n: np.random.randn(n,n) * np.sqrt(2/n)

plt.figure(figsize=(10,5))
for name, fn in [("zero",zero_init),("small 0.01",small_init),("xavier",xavier_init),("He",he_init)]:
    plt.plot(activation_stats(fn), label=name, marker='o')
plt.xlabel("Layer"); plt.ylabel("Activation std"); plt.legend()
plt.title("Activation std across layers by init strategy"); plt.show()

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

class EnhancedWeightInitialization:

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



model = EnhancedWeightInitialization()

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
model = EnhancedWeightInitialization()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: He vs Xavier in PyTorch training
import torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import make_classification

X, y = make_classification(300, 20, n_informative=10, random_state=0)
Xt, yt = torch.FloatTensor(X), torch.LongTensor(y)

def make_model(init='he'):
    m = nn.Sequential(nn.Linear(20,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,2))
    for layer in m.modules():
        if isinstance(layer, nn.Linear):
            if init=='he':     nn.init.kaiming_normal_(layer.weight)
            elif init=='xavier': nn.init.xavier_normal_(layer.weight)
            elif init=='zero': nn.init.zeros_(layer.weight)
    return m

for name in ['he','xavier','zero']:
    m = make_model(name)
    o = optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(200):
        o.zero_grad(); nn.CrossEntropyLoss()(m(Xt), yt).backward(); o.step()
    with torch.no_grad():
        acc = (m(Xt).argmax(1).numpy() == y).mean()
    print(f"{name:8s} init → acc={acc:.4f}")

# Example 3: Batch Norm rescues poor init
m_bad  = nn.Sequential(nn.Linear(20,64), nn.ReLU(), nn.Linear(64,2))
m_good = nn.Sequential(nn.Linear(20,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64,2))

# Give both terrible init
for model in [m_bad, m_good]:
    for l in model.modules():
        if isinstance(l, nn.Linear): nn.init.normal_(l.weight, std=5)

for model, name in [(m_bad,"No BN"),(m_good,"With BN")]:
    o = optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(200):
        o.zero_grad(); nn.CrossEntropyLoss()(model(Xt), yt).backward(); o.step()
    with torch.no_grad():
        acc = (model(Xt).argmax(1).numpy() == y).mean()
    print(f"{name}: acc={acc:.4f}")