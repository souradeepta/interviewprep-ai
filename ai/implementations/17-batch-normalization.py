# 17 Batch Normalization
# Extracted from Jupyter notebook

import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X, y = make_classification(400, 20, n_informative=8, random_state=0)
Xt, yt = torch.FloatTensor(X), torch.LongTensor(y)

class PlainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,2))
    def forward(self, x): return self.layers(x)

class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20,64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64,32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32,2))
    def forward(self, x): return self.layers(x)

def train(model, epochs=200, lr=0.01):
    opt = optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        l = nn.CrossEntropyLoss()(model(Xt), yt)
        l.backward(); opt.step(); losses.append(l.item())
    return losses

l_plain = train(PlainNet())
l_bn    = train(BNNet())

plt.plot(l_plain, label="No BN"); plt.plot(l_bn, label="With BN")
plt.legend(); plt.title("Batch Normalization Effect on Training Loss"); plt.show()

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

class EnhancedBatchNormalization:

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



model = EnhancedBatchNormalization()

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
model = EnhancedBatchNormalization()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: BN vs Layer Norm for small batches
class LayerNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20,64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64,2))
    def forward(self, x): return self.layers(x)

# Simulate small batch (LN works better)
for bs in [4, 16, 64]:
    m_bn = BNNet(); m_ln = LayerNormNet()
    for m, name in [(m_bn,"BN"),(m_ln,"LN")]:
        o = optim.Adam(m.parameters())
        for _ in range(100):
            idx = torch.randperm(len(Xt))[:bs]
            o.zero_grad(); nn.CrossEntropyLoss()(m(Xt[idx]), yt[idx]).backward(); o.step()
        with torch.no_grad():
            acc = (m(Xt).argmax(1).numpy()==y).mean()
        print(f"bs={bs:3d}  {name}: acc={acc:.4f}")

# Example 3: Higher learning rate with BN
for lr in [0.001, 0.01, 0.05, 0.1]:
    m = BNNet(); o = optim.SGD(m.parameters(), lr=lr)
    try:
        for _ in range(100):
            m.train(); o.zero_grad(); l=nn.CrossEntropyLoss()(m(Xt),yt); l.backward(); o.step()
        with torch.no_grad(): acc=(m(Xt).argmax(1).numpy()==y).mean()
        print(f"lr={lr:.3f}: acc={acc:.4f}")
    except Exception as e:
        print(f"lr={lr:.3f}: diverged ({e})")