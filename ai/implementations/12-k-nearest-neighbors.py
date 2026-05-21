# 12 K Nearest Neighbors
# Extracted from Jupyter notebook

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(Xtr_s, ytr)
print(f"Test accuracy: {knn.score(Xte_s, yte):.4f}")

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

class EnhancedKNearestNeighbors:

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



model = EnhancedK-NearestNeighbors()

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

model = EnhancedKNearestNeighbors()

scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-val scores: {scores}")

print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: k selection – bias-variance tradeoff
k_values = range(1, 30)
test_accs, train_accs = [], []
for k in k_values:
    m = KNeighborsClassifier(n_neighbors=k).fit(Xtr_s, ytr)
    train_accs.append(m.score(Xtr_s, ytr))
    test_accs.append(m.score(Xte_s, yte))

import numpy as np
best_k = list(k_values)[np.argmax(test_accs)]
plt.plot(k_values, train_accs, label="Train")
plt.plot(k_values, test_accs, label="Test")
plt.axvline(best_k, linestyle='--', color='red', label=f"Best k={best_k}")
plt.xlabel("k"); plt.ylabel("Accuracy"); plt.legend()
plt.title("KNN: Bias-Variance Tradeoff"); plt.show()

# Example 3: Effect of distance metric
metrics = ['euclidean', 'manhattan', 'chebyshev']
for metric in metrics:
    acc = KNeighborsClassifier(n_neighbors=best_k, metric=metric).fit(Xtr_s, ytr).score(Xte_s, yte)
    print(f"{metric:12s}: {acc:.4f}")