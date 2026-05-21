# 04 Optimization Algorithms
# Extracted from Jupyter notebook

import numpy as np
import matplotlib.pyplot as plt

class SGDOptimizer:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, w, grad):
        return w - self.lr * grad

class MomentumOptimizer:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum = lr, momentum
        self.v = None
    def update(self, w, grad):
        if self.v is None: self.v = np.zeros_like(w)
        self.v = self.momentum * self.v - self.lr * grad
        return w + self.v

class AdamOptimizer:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999):
        self.lr, self.b1, self.b2, self.t = lr, b1, b2, 0
        self.m = self.v = None
    def update(self, w, grad):
        if self.m is None:
            self.m, self.v = np.zeros_like(w), np.zeros_like(w)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad**2
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

np.random.seed(42)
X = np.random.randn(100, 5)
y = X[:, :2].sum(axis=1) + np.random.randn(100) * 0.1

optimizers = {'SGD': SGDOptimizer(0.1), 'Momentum': MomentumOptimizer(0.1), 'Adam': AdamOptimizer(0.1)}
all_losses = {}
for name, opt in optimizers.items():
    theta = np.random.randn(5) * 0.01
    losses = []
    for _ in range(60):
        pred = X @ theta
        grad = (2/len(y)) * X.T @ (pred - y)
        theta = opt.update(theta, grad)
        losses.append(np.mean((X @ theta - y)**2))
    all_losses[name] = losses

plt.figure(figsize=(10, 4))
for name, losses in all_losses.items():
    plt.plot(losses, label=name)
plt.xlabel('Iteration'), plt.ylabel('MSE')
plt.title('SGD vs Momentum vs Adam'), plt.legend(), plt.show()
print(f"Final MSE — SGD: {all_losses['SGD'][-1]:.4f}, Momentum: {all_losses['Momentum'][-1]:.4f}, Adam: {all_losses['Adam'][-1]:.4f}")

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

class EnhancedOptimizationAlgorithms:

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



model = EnhancedOptimizationAlgorithms()

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
model = EnhancedOptimizationAlgorithms()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Grid search parameter tuning

from sklearn.model_selection import GridSearchCV



param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Placeholder: in practice use real parameters for your model

print(f"Parameter grid would be: {param_grid}")



# In practice:

# grid = GridSearchCV(model, param_grid, cv=5)

# grid.fit(X, y)

# print(f"Best params: {grid.best_params_}")

# Example 3: Full pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', EnhancedOptimizationAlgorithms()),
])

# Train and evaluate
pipeline.fit(X_train, y_train)
final_score = pipeline.score(X_test, y_test)
print(f"Pipeline score: {final_score:.4f}")