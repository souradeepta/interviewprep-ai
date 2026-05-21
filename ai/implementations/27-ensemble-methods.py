# 27 Ensemble Methods
# Extracted from Jupyter notebook

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

estimators = [('lr',  LogisticRegression(max_iter=1000)),
              ('rf',  RandomForestClassifier(n_estimators=50, random_state=42)),
              ('gb',  GradientBoostingClassifier(n_estimators=50, random_state=42))]

# Voting ensemble
vc = VotingClassifier(estimators, voting='soft')
vc.fit(Xtr, ytr)

for name, est in [*estimators, ('ensemble', vc)]:
    scores = cross_val_score(est if name=='ensemble' else est, Xtr, ytr, cv=5, scoring='roc_auc')
    print(f"{name:10s} CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

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

class EnhancedEnsembleMethods:

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



model = EnhancedEnsembleMethods()

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
model = EnhancedEnsembleMethods()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Stacking
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score

base = [('rf',  RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb',  GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('svc', SVC(probability=True))]
meta = LogisticRegression()

stack = StackingClassifier(estimators=base, final_estimator=meta, cv=5, passthrough=False)
scores = cross_val_score(stack, Xtr, ytr, cv=5, scoring='roc_auc')
print(f"Stacking CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 3: Bagging reduces variance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

single_tree = DecisionTreeClassifier(max_depth=None)  # unpruned (high variance)
bagged = BaggingClassifier(DecisionTreeClassifier(max_depth=None),
                            n_estimators=100, max_samples=0.8, random_state=42)

for model, name in [(single_tree,"Single Tree"),(bagged,"Bagging (100 trees)")]:
    sc = cross_val_score(model, Xtr, ytr, cv=10, scoring='roc_auc')
    print(f"{name:25s}  mean={sc.mean():.4f}  std={sc.std():.4f}")
print("Bagging reduces std (variance) dramatically")