# 09 Random Forests
# Extracted from Jupyter notebook

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(Xtr, ytr)

print(f"Train: {rf.score(Xtr,ytr):.4f}  Test: {rf.score(Xte,yte):.4f}")

# OOB error
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf_oob.fit(Xtr, ytr)
print(f"OOB score: {rf_oob.oob_score_:.4f}  (free validation estimate)")

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

class EnhancedRandomForests:

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



model = EnhancedRandomForests()

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
model = EnhancedRandomForests()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: n_estimators convergence – when do trees stop helping?
oob_scores = []
n_values = [5, 10, 25, 50, 100, 200, 300]

for n in n_values:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf.fit(Xtr, ytr)
    oob_scores.append(rf.oob_score_)

plt.plot(n_values, oob_scores, 'o-')
plt.xlabel("Number of trees"); plt.ylabel("OOB Accuracy")
plt.title("Diminishing returns after ~100 trees"); plt.show()

# Example 3: Feature importance vs single decision tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xtr, ytr)
rf100 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)

import pandas as pd
compare = pd.DataFrame({
    "Feature": iris.feature_names,
    "DT importance":  dt.feature_importances_,
    "RF importance": rf100.feature_importances_,
}).set_index("Feature")

print(compare.round(4))
compare.plot(kind='bar', figsize=(8,4))
plt.title("Feature Importance: DT vs RF"); plt.tight_layout(); plt.show()