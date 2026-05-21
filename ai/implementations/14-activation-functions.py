# 14 Activation Functions
# Extracted from Jupyter notebook

import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 300)
acts = {
    "ReLU":       np.maximum(0, z),
    "Leaky ReLU": np.where(z>0, z, 0.1*z),
    "Sigmoid":    1/(1+np.exp(-z)),
    "Tanh":       np.tanh(z),
    "GELU":       z * 0.5*(1+np.tanh(np.sqrt(2/np.pi)*(z+0.044715*z**3))),
}

fig, axes = plt.subplots(1, 2, figsize=(14,4))
for name, a in acts.items():
    axes[0].plot(z, a, label=name)
axes[0].set_title("Activations"); axes[0].legend(); axes[0].grid()

# Gradients
grads = {
    "ReLU":       (z > 0).astype(float),
    "Leaky ReLU": np.where(z>0, 1, 0.1),
    "Sigmoid":    (1/(1+np.exp(-z))) * (1 - 1/(1+np.exp(-z))),
    "Tanh":       1 - np.tanh(z)**2,
}
for name, g in grads.items():
    axes[1].plot(z, g, label=name)
axes[1].set_title("Gradients"); axes[1].legend(); axes[1].grid()
plt.tight_layout(); plt.show()

print("Sigmoid/Tanh gradients saturate to 0 → vanishing gradients in deep nets")
print("ReLU gradient is 0 or 1 → no saturation, but dead neuron risk")

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

class EnhancedActivationFunctions:

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



model = EnhancedActivationFunctions()

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
model = EnhancedActivationFunctions()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Dead ReLU detection
import torch, torch.nn as nn

def count_dead_relu(model, X):
    hooks = []
    dead_counts = []
    def hook_fn(m, inp, out):
        dead_counts.append((out <= 0).float().mean().item())
    for layer in model.modules():
        if isinstance(layer, nn.ReLU):
            hooks.append(layer.register_forward_hook(hook_fn))
    with torch.no_grad():
        model(X)
    [h.remove() for h in hooks]
    return dead_counts

X_test = torch.randn(100, 10)
model_bad  = nn.Sequential(nn.Linear(10,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,2))

# Simulate dying ReLU by initialising with large negative biases
for m in model_bad.modules():
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, -5)

dead = count_dead_relu(model_bad, X_test)
print(f"Dead neuron % per ReLU layer: {[f'{d:.1%}' for d in dead]}")
print("Fix: He init, batch norm, or use LeakyReLU")

# Example 3: Activation choice in a real model
import torch.optim as optim
from sklearn.datasets import make_moons

X_m, y_m = make_moons(n_samples=300, noise=0.2, random_state=42)
Xt = torch.FloatTensor(X_m); yt = torch.LongTensor(y_m)

results = {}
for act_name, act in [("ReLU",nn.ReLU()), ("Tanh",nn.Tanh()), ("GELU",nn.GELU())]:
    m = nn.Sequential(nn.Linear(2,16), act, nn.Linear(16,8), act, nn.Linear(8,2))
    o = optim.Adam(m.parameters(), lr=5e-3)
    for _ in range(300):
        o.zero_grad(); nn.CrossEntropyLoss()(m(Xt), yt).backward(); o.step()
    with torch.no_grad():
        acc = (m(Xt).argmax(1).numpy() == y_m).mean()
    results[act_name] = acc

for k,v in results.items(): print(f"{k:8s}: {v:.4f}")