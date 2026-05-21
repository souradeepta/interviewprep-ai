# 23 Classification Metrics
# Extracted from Jupyter notebook

import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, average_precision_score, f1_score)
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
ypred = clf.predict(Xte); yproba = clf.predict_proba(Xte)[:,1]

cm = confusion_matrix(yte, ypred)
print("Confusion matrix:\n", cm)
print(f"\nTP={cm[1,1]}  FP={cm[0,1]}  TN={cm[0,0]}  FN={cm[1,0]}")
print("\n", classification_report(yte, ypred, target_names=['malignant','benign']))
print(f"ROC-AUC : {roc_auc_score(yte, yproba):.4f}")
print(f"Avg Prec: {average_precision_score(yte, yproba):.4f}")

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

class EnhancedClassificationMetrics:

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



model = EnhancedClassificationMetrics()

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
model = EnhancedClassificationMetrics()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: ROC and Precision-Recall curves
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(yte, yproba)
prec, rec,  _ = precision_recall_curve(yte, yproba)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
ax1.plot(fpr, tpr, label=f"AUC={roc_auc_score(yte,yproba):.3f}")
ax1.plot([0,1],[0,1],'k--'); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC Curve"); ax1.legend()

ax2.plot(rec, prec, label=f"AP={average_precision_score(yte,yproba):.3f}")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("Precision-Recall Curve"); ax2.legend()
plt.tight_layout(); plt.show()

# Example 3: Choosing threshold for business need
thresholds = np.arange(0.1, 0.9, 0.05)
results = []
for thr in thresholds:
    ypred_t = (yproba >= thr).astype(int)
    cm = confusion_matrix(yte, ypred_t)
    tp, fp, tn, fn = cm[1,1], cm[0,1], cm[0,0], cm[1,0]
    precision = tp/(tp+fp+1e-9); recall = tp/(tp+fn+1e-9)
    results.append((thr, precision, recall, f1_score(yte, ypred_t)))

import pandas as pd
df = pd.DataFrame(results, columns=['threshold','precision','recall','f1'])
print(df.to_string(index=False))
print("\nChoose threshold based on cost of FP vs FN!")