# 26 Hyperparameter Tuning
# Extracted from Jupyter notebook

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Grid search
param_grid = {'n_estimators':[50,100], 'max_depth':[3,5,None], 'min_samples_leaf':[1,2,5]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(Xtr, ytr)
print(f"Grid  – best params: {gs.best_params_}, CV AUC: {gs.best_score_:.4f}, Test AUC: {gs.score(Xte, yte):.4f}")

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

class EnhancedHyperparameterTuning:

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



model = EnhancedHyperparameterTuning()

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
model = EnhancedHyperparameterTuning()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Random search – faster than grid for same budget
from scipy.stats import randint, uniform

param_dist = {'n_estimators': randint(50,300), 'max_depth': randint(2,20),
              'min_samples_leaf': randint(1,10), 'max_features': uniform(0.3,0.7)}
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist,
                         n_iter=30, cv=5, scoring='roc_auc', random_state=0, n_jobs=-1)
rs.fit(Xtr, ytr)
print(f"Random – best params: {rs.best_params_}, CV AUC: {rs.best_score_:.4f}, Test AUC: {rs.score(Xte,yte):.4f}")

# Example 3: Bayesian optimisation via scikit-optimize
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer

    space = {'n_estimators':Integer(50,300), 'max_depth':Integer(2,20),
             'min_samples_leaf':Integer(1,10)}
    bo = BayesSearchCV(RandomForestClassifier(random_state=42), space,
                       n_iter=20, cv=5, scoring='roc_auc', random_state=0)
    bo.fit(Xtr, ytr)
    print(f"Bayes  – best: {bo.best_params_}, AUC: {bo.best_score_:.4f}")
except ImportError:
    print("scikit-optimize not installed: pip install scikit-optimize")
    print("Bayesian optimisation finds good HPs with fewer evaluations than grid/random")