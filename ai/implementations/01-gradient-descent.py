"""
Auto-generated from 01-gradient-descent.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Gradient Descent
# ## Learning Objectives
# 1. Understand how gradient descent optimizes model weights
# 2. Implement GD variants (batch, SGD, mini-batch)
# ======================================================================

# ======================================================================
# ## Level 1: Manual Gradient Descent
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, y, lr=0.01, epochs=200):
    theta = np.zeros(X.shape[1])
    losses = []
    for _ in range(epochs):
        pred = X @ theta
        grad = (2/len(y)) * X.T @ (pred - y)
        theta -= lr * grad
        losses.append(np.mean((pred - y)**2))
    return theta, losses

np.random.seed(42)
X = np.c_[np.ones(200), np.random.randn(200, 2)]
true_w = np.array([1.0, 2.0, -1.5])
y = X @ true_w + np.random.randn(200)*0.3

theta, losses = gradient_descent(X, y, lr=0.05)
print(f"True:    {true_w}")
print(f"Learned: {np.round(theta,3)}")
plt.plot(losses); plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Convergence"); plt.show()


# ======================================================================
# ## Level 2: Mini-Batch with Momentum
# ======================================================================

class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.0, batch_size=32):
        self.lr = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.velocity = None

    def __call__(self, X, y, epochs=100):
        m = len(y)
        theta = np.zeros(X.shape[1])
        self.velocity = np.zeros_like(theta)
        costs = []

        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(m)
            X_shuffled, y_shuffled = X[indices], y[indices]

            epoch_cost = 0
            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                # Gradient
                pred = X_batch @ theta
                grad = (2/len(y_batch)) * X_batch.T @ (pred - y_batch)

                # Momentum
                self.velocity = self.momentum * self.velocity - self.lr * grad
                theta += self.velocity

                epoch_cost += np.mean((pred - y_batch) ** 2)

            costs.append(epoch_cost / (m / self.batch_size))

        return theta, costs

# Compare optimizers
X = np.c_[np.ones(200), np.random.randn(200, 5)]
y = np.sum(X[:, 1:3], axis=1) + np.random.randn(200) * 0.1

results = {}
for momentum in [0.0, 0.9]:
    opt = GradientDescentOptimizer(lr=0.1, momentum=momentum, batch_size=32)
    theta, costs = opt(X, y, epochs=50)
    results[f"momentum={momentum}"] = costs

plt.figure(figsize=(10, 5))
for label, costs in results.items():
    plt.plot(costs, label=label)
plt.xlabel('Epoch'), plt.ylabel('Loss')
plt.legend()
plt.title('Impact of Momentum on Convergence')
plt.show()


# ======================================================================
# ## Real-World Example 1: Learning Rate Schedules
# ======================================================================

def cosine_annealing(epoch, initial_lr=0.1, T_max=100):
    '''Cosine annealing schedule.'''
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / T_max))

def step_decay(epoch, initial_lr=0.1, drop=0.5, epochs_per_drop=10):
    '''Step decay schedule.'''
    return initial_lr * (drop ** (epoch // epochs_per_drop))

# Compare schedules
epochs = 100
schedules = {
    'constant': [0.1] * epochs,
    'cosine': [cosine_annealing(e) for e in range(epochs)],
    'step': [step_decay(e) for e in range(epochs)],
}

plt.figure(figsize=(12, 4))
for name, schedule in schedules.items():
    plt.plot(schedule, label=name)
plt.xlabel('Epoch'), plt.ylabel('Learning Rate')
plt.legend()
plt.title('Learning Rate Schedules')
plt.show()


# ======================================================================
# ## Real-World Example 2: Adam Optimizer
# ======================================================================

# Example 2: Compare batch, stochastic, mini-batch gradient descent
def mini_batch_gd(X, y, batch_size=32, lr=0.05, epochs=50):
    m = len(y); theta = np.zeros(X.shape[1]); losses = []
    for epoch in range(epochs):
        idx = np.random.permutation(m)
        for i in range(0, m, batch_size):
            batch = idx[i:i+batch_size]
            Xb, yb = X[batch], y[batch]
            theta -= lr * (2/len(yb)) * Xb.T @ (Xb @ theta - yb)
        losses.append(np.mean((X @ theta - y)**2))
    return theta, losses

_, batch_losses = gradient_descent(X, y, lr=0.05, epochs=50)
_, mini_losses  = mini_batch_gd(X, y, batch_size=32, lr=0.05, epochs=50)

plt.plot(batch_losses, label="Batch GD")
plt.plot(mini_losses,  label="Mini-Batch GD (bs=32)")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.title("Batch vs Mini-Batch Convergence"); plt.show()


# ======================================================================
# ## Real-World Example 3: Learning Rate Tuning
# ======================================================================

# Example 3: Adam optimizer from scratch
class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999):
        self.lr, self.b1, self.b2 = lr, b1, b2
        self.m = self.v = self.t = 0

    def step(self, theta, grad):
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*grad
        self.v = self.b2*self.v + (1-self.b2)*grad**2
        m_hat = self.m / (1-self.b1**self.t)
        v_hat = self.v / (1-self.b2**self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

adam = Adam(lr=0.1)
theta = np.zeros(X.shape[1])
losses = []
for _ in range(100):
    pred = X @ theta
    grad = (2/len(y)) * X.T @ (pred - y)
    theta = adam.step(theta, grad)
    losses.append(np.mean((pred - y)**2))

plt.plot(losses); plt.title("Adam Optimizer Convergence"); plt.xlabel("Iteration"); plt.show()
print(f"Learned: {np.round(theta, 3)}")


# ======================================================================
# ## Key Takeaways
# **Core idea:** Iteratively move weights in the direction that reduces loss.
# **Variants:**
# ======================================================================
