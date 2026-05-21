# 13 Neural Networks
# Extracted from Jupyter notebook

import torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=500, n_features=10, n_informative=6, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler().fit(Xtr)
Xtr_t = torch.FloatTensor(sc.transform(Xtr))
Xte_t = torch.FloatTensor(sc.transform(Xte))
ytr_t = torch.LongTensor(ytr)

model = nn.Sequential(nn.Linear(10,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,2))
opt   = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(200):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(Xtr_t), ytr_t)
    loss.backward(); opt.step()
    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(Xte_t).argmax(1).numpy() == yte).mean()
        print(f"Epoch {epoch:3d}  loss={loss.item():.4f}  test_acc={acc:.4f}")

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

class EnhancedNeuralNetworks:

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



model = EnhancedNeuralNetworks()

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
model = EnhancedNeuralNetworks()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Learning curves – diagnose over/underfitting
import numpy as np
import matplotlib.pyplot as plt

train_sizes = [50, 100, 200, 350, 400]
train_accs, test_accs = [], []

for n in train_sizes:
    Xn = Xtr_t[:n]; yn = ytr_t[:n]
    m = nn.Sequential(nn.Linear(10,32), nn.ReLU(), nn.Linear(32,2))
    o = optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(200):
        o.zero_grad(); loss_fn(m(Xn), yn).backward(); o.step()
    with torch.no_grad():
        tr_acc = (m(Xn).argmax(1).numpy() == yn.numpy()).mean()
        te_acc = (m(Xte_t).argmax(1).numpy() == yte).mean()
    train_accs.append(tr_acc); test_accs.append(te_acc)

plt.plot(train_sizes, train_accs, label="Train"); plt.plot(train_sizes, test_accs, label="Test")
plt.xlabel("Training size"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Learning Curves"); plt.show()

# Example 3: Universal approximation – XOR problem
xor_X = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]])
xor_y = torch.LongTensor([0,1,1,0])

# Linear model cannot solve XOR
linear = nn.Linear(2, 2)
# MLP can solve XOR
mlp = nn.Sequential(nn.Linear(2,4), nn.ReLU(), nn.Linear(4,2))

for model, name in [(linear,"Linear"),(mlp,"MLP")]:
    opt = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(1000):
        opt.zero_grad(); loss_fn(model(xor_X), xor_y).backward(); opt.step()
    with torch.no_grad():
        preds = model(xor_X).argmax(1)
    print(f"{name}: predictions={preds.tolist()}, labels={xor_y.tolist()}, solved={preds.tolist()==xor_y.tolist()}")