# 05 Learning Rate Scheduling
# Extracted from Jupyter notebook

import numpy as np
import matplotlib.pyplot as plt

def step_decay(epoch, lr0=0.1, drop=0.5, every=20):
    return lr0 * (drop ** (epoch // every))

def exponential_decay(epoch, lr0=0.1, gamma=0.95):
    return lr0 * (gamma ** epoch)

def cosine_annealing(epoch, T_max=100, lr_max=0.1, lr_min=1e-5):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T_max))

def warmup_cosine(epoch, warmup=10, T_max=100, lr_max=0.1):
    if epoch < warmup:
        return lr_max * (epoch / warmup)
    return cosine_annealing(epoch - warmup, T_max - warmup, lr_max)

epochs = np.arange(100)
schedules = {
    'Constant (0.1)': np.full(100, 0.1),
    'Step Decay':     [step_decay(e) for e in epochs],
    'Exponential':    [exponential_decay(e) for e in epochs],
    'Cosine':         [cosine_annealing(e) for e in epochs],
    'Warmup+Cosine':  [warmup_cosine(e) for e in epochs],
}

plt.figure(figsize=(12, 5))
for name, lrs in schedules.items():
    plt.plot(epochs, lrs, label=name, linewidth=2)
plt.xlabel('Epoch'), plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedules Comparison')
plt.legend(), plt.show()

# Simulate training with each schedule
np.random.seed(42)
X = np.random.randn(100, 5)
y = X[:, :2].sum(axis=1) + np.random.randn(100) * 0.1

for name, lrs in schedules.items():
    theta = np.random.randn(5) * 0.01
    for epoch, lr in enumerate(lrs):
        pred = X @ theta
        grad = (2/len(y)) * X.T @ (pred - y)
        theta -= lr * grad
    final_mse = np.mean((X @ theta - y)**2)
    print(f"{name:25s}: final MSE={final_mse:.4f}")

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

class EnhancedLearningRateScheduling:

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



model = EnhancedLearningRateScheduling()

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
model = EnhancedLearningRateScheduling()
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
    ('model', EnhancedLearningRateScheduling()),
])

# Train and evaluate
pipeline.fit(X_train, y_train)
final_score = pipeline.score(X_test, y_test)
print(f"Pipeline score: {final_score:.4f}")