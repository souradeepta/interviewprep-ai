# 08 Decision Trees
# Extracted from Jupyter notebook

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
dt.fit(Xtr, ytr)

print(f"Train acc: {dt.score(Xtr,ytr):.4f}")
print(f"Test  acc: {dt.score(Xte,yte):.4f}")
print(f"Tree depth: {dt.get_depth()}, Leaves: {dt.get_n_leaves()}")

plt.figure(figsize=(16,6))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree (max_depth=3)"); plt.show()

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

class EnhancedDecisionTrees:

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



model = EnhancedDecisionTrees()

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
model = EnhancedDecisionTrees()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Depth vs accuracy – find optimal depth
import numpy as np
from sklearn.model_selection import cross_val_score

depths = range(1, 15)
train_scores, cv_scores = [], []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(Xtr, ytr)
    train_scores.append(dt.score(Xtr, ytr))
    cv_scores.append(cross_val_score(dt, Xtr, ytr, cv=5).mean())

plt.plot(depths, train_scores, label="Train")
plt.plot(depths, cv_scores,   label="CV Mean")
plt.xlabel("Max Depth"); plt.ylabel("Accuracy")
plt.legend(); plt.title("Depth vs Accuracy (overfitting visible)"); plt.show()

best_depth = depths[np.argmax(cv_scores)]
print(f"Best CV depth: {best_depth}")

# Example 3: Feature importance + pruning via min_impurity_decrease
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

dt_tuned = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=3,
                                   min_impurity_decrease=0.001, random_state=42)
dt_tuned.fit(Xtr, ytr)
print(f"Tuned test acc: {dt_tuned.score(Xte,yte):.4f}")

feat_imp = pd.Series(dt_tuned.feature_importances_, index=iris.feature_names).sort_values(ascending=True)
feat_imp.plot(kind='barh'); plt.title("Feature Importance"); plt.show()