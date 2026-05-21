# 10 Gradient Boosting
# Extracted from Jupyter notebook

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
gb.fit(Xtr, ytr)
print(f"Train: {gb.score(Xtr,ytr):.4f}  Test: {gb.score(Xte,yte):.4f}")

# Staged predictions – see how test accuracy improves with each tree
test_scores = [gb.score(Xte, yte) for _ in gb.staged_predict(Xte)]  # not exactly – use staged_predict_proba

import matplotlib.pyplot as plt
train_deviance = gb.train_score_
plt.plot(range(1, len(train_deviance)+1), train_deviance, label="Train loss")
plt.xlabel("Boosting iterations"); plt.ylabel("Deviance"); plt.legend()
plt.title("Gradient Boosting Training Curve"); plt.show()

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

class EnhancedGradientBoosting:

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



model = EnhancedGradientBoosting()

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
model = EnhancedGradientBoosting()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Early stopping with validation set
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

Xtr2, Xval, ytr2, yval = train_test_split(Xtr, ytr, test_size=0.2, random_state=0)

gb_es = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05,
                                    max_depth=3, validation_fraction=0.15,
                                    n_iter_no_change=15, random_state=42)
gb_es.fit(Xtr, ytr)
print(f"Trees used (early stop): {gb_es.n_estimators_}")
print(f"Test accuracy: {gb_es.score(Xte,yte):.4f}")

# Example 3: XGBoost vs sklearn GradientBoosting
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                         use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(Xtr, ytr)
    print(f"XGBoost test acc: {xgb.score(Xte,yte):.4f}")
except ImportError:
    print("xgboost not installed; run: pip install xgboost")

gb_sk = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
gb_sk.fit(Xtr, ytr)
print(f"sklearn  GB test acc: {gb_sk.score(Xte,yte):.4f}")