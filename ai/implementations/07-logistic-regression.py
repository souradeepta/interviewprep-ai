# 07 Logistic Regression
# Extracted from Jupyter notebook

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                            n_clusters_per_class=1, random_state=42)
X = (X - X.mean(0)) / X.std(0)
Xb = np.c_[np.ones(len(X)), X]

theta = np.zeros(Xb.shape[1])
for _ in range(300):
    p    = sigmoid(Xb @ theta)
    grad = Xb.T @ (p - y) / len(y)
    theta -= 0.5 * grad

acc = ((sigmoid(Xb @ theta) > 0.5) == y).mean()
print(f"Accuracy: {acc:.4f}")

# Decision boundary
x0 = np.linspace(X[:,0].min(), X[:,0].max(), 100)
x1 = -(theta[0] + theta[1]*x0) / theta[2]
plt.scatter(*X[y==0].T, label="Class 0", alpha=0.6)
plt.scatter(*X[y==1].T, label="Class 1", alpha=0.6)
plt.plot(x0, x1, 'k--', label="Decision boundary")
plt.legend(); plt.title("Logistic Regression"); plt.show()

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

class EnhancedLogisticRegression:

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



model = EnhancedLogisticRegression()

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
model = EnhancedLogisticRegression()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Multiclass logistic regression (softmax)
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

clf = LogisticRegression(multi_class='multinomial', max_iter=500, C=1.0)
clf.fit(Xtr, ytr)
print(classification_report(yte, clf.predict(Xte), target_names=iris.target_names))

# Example 3: Calibration – are probabilities reliable?
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

clf2 = LogisticRegression(C=0.1).fit(Xtr, (ytr==0).astype(int))
prob = clf2.predict_proba(Xte)[:,1]
y_bin = (yte==0).astype(int)

frac_pos, mean_pred = calibration_curve(y_bin, prob, n_bins=10)
plt.plot(mean_pred, frac_pos, 'o-', label='Logistic')
plt.plot([0,1],[0,1],'k--', label='Perfect')
plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
plt.title("Calibration Curve"); plt.legend(); plt.show()