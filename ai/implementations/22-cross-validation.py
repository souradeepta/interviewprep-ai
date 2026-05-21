# 22 Cross Validation
# Extracted from Jupyter notebook

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

model = RandomForestClassifier(n_estimators=50, random_state=42)

# K-Fold
kf  = KFold(n_splits=5, shuffle=True, random_state=0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

kf_scores  = cross_val_score(model, X, y, cv=kf,  scoring='roc_auc')
skf_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

print(f"KFold   ROC-AUC: {kf_scores.mean():.4f} ± {kf_scores.std():.4f}")
print(f"Strat.  ROC-AUC: {skf_scores.mean():.4f} ± {skf_scores.std():.4f}")
print("Stratified CV recommended for classification (preserves class ratio)")

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

class EnhancedCrossValidation:

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



model = EnhancedCrossValidation()

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

model = EnhancedCrossValidation()

scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-val scores: {scores}")

print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Nested CV for hyperparameter tuning + evaluation
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([('sc', StandardScaler()), ('svm', SVC())])
param_grid = {'svm__C': [0.1,1,10], 'svm__kernel':['linear','rbf']}

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

# Nested: inner loop tunes hyperparams, outer loop evaluates
inner_gs = GridSearchCV(pipe, param_grid, cv=inner_cv)
nested_scores = cross_val_score(inner_gs, X, y, cv=outer_cv, scoring='roc_auc')

print(f"Nested CV ROC-AUC: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
print("Nested CV is the gold standard – avoids optimistic bias from hyperparameter search")

# Example 3: Time-series cross-validation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
import numpy as np

np.random.seed(0)
t = np.linspace(0,4*np.pi,200)
X_ts = np.c_[np.sin(t), np.cos(t), t]
y_ts = np.sin(t+0.5) + np.random.randn(200)*0.1

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(Ridge(), X_ts, y_ts, cv=tscv, scoring='r2')
print(f"Time-series CV R²: {scores}")
print(f"Mean: {scores.mean():.4f}")
print("TimeSeriesSplit ensures past → future only (no data leakage)")