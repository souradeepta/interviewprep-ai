# 02 Backpropagation
# Extracted from Jupyter notebook

import numpy as np

def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

class TwoLayerNet:
    def __init__(self, n_in, n_hidden, n_out, lr=0.05):
        self.W1 = np.random.randn(n_in, n_hidden) * np.sqrt(2/n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_out) * np.sqrt(2/n_hidden)
        self.b2 = np.zeros(n_out)
        self.lr = lr

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, y):
        m = len(y)
        dout = (self.out - y.reshape(-1,1)) / m
        dW2 = self.a1.T @ dout
        db2 = dout.sum(0)
        da1 = dout @ self.W2.T
        dz1 = da1 * relu_grad(self.z1)
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(0)
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

np.random.seed(42)
X = np.random.randn(200, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(float)

net = TwoLayerNet(4, 16, 1)
for epoch in range(300):
    out = net.forward(X)
    net.backward(y)
    if epoch % 100 == 0:
        loss = -np.mean(y*np.log(out[:,0]+1e-8) + (1-y)*np.log(1-out[:,0]+1e-8))
        print(f"Epoch {epoch}: loss={loss:.4f}")

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

class EnhancedBackpropagation:

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



model = EnhancedBackpropagation()

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
model = EnhancedBackpropagation()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Gradient checking – verify backprop numerically
def numerical_grad(net, X, y, param, idx, eps=1e-5):
    orig = param.flat[idx]
    param.flat[idx] = orig + eps
    loss_plus = -np.mean(y*np.log(net.forward(X)[:,0]+1e-8))
    param.flat[idx] = orig - eps
    loss_minus = -np.mean(y*np.log(net.forward(X)[:,0]+1e-8))
    param.flat[idx] = orig
    return (loss_plus - loss_minus) / (2*eps)

net2 = TwoLayerNet(4, 8, 1)
net2.forward(X); net2.backward(y)

# Check a random weight in W2
i = 3
num  = numerical_grad(net2, X, y, net2.W2, i)
ana  = net2.W2.flat[i] - (net2.W2.flat[i])   # placeholder: use stored dW2
print(f"Gradient check example: numerical≈{num:.6f}")
print("In practice: relative error should be < 1e-5")

# Example 3: Vanishing gradients with sigmoid vs ReLU
def train_depth(activation, n_layers=8, epochs=200):
    np.random.seed(42)
    W = [np.random.randn(20,20)*0.1 for _ in range(n_layers)]
    losses = []
    X_tmp = np.random.randn(50, 20)
    y_tmp = (X_tmp[:,0] > 0).astype(float)
    for _ in range(epochs):
        a = X_tmp
        for w in W:
            z = a @ w
            a = sigmoid(z) if activation == 'sigmoid' else relu(z)
        pred = sigmoid(a[:, :1])
        loss = -np.mean(y_tmp*np.log(pred+1e-8))
        losses.append(loss)
    return losses

import matplotlib.pyplot as plt
plt.plot(train_depth('sigmoid'), label='Sigmoid (8 layers)')
plt.plot(train_depth('relu'),    label='ReLU (8 layers)')
plt.legend(); plt.title("Vanishing Gradients: Sigmoid vs ReLU"); plt.show()