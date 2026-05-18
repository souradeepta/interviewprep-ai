#!/usr/bin/env python3
"""Fix pass placeholders in ai/concepts 21-28 Code Examples."""

import os

os.chdir("/home/sbisw/github/interviewprep-ml/ai/concepts")

EXAMPLES = {
    "21-bias-variance-tradeoff.md": [
        ("Polynomial Degree vs Bias-Variance", """import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

np.random.seed(42)
X_raw = np.linspace(0, 1, 100)
y_raw = np.sin(2 * np.pi * X_raw) + np.random.randn(100) * 0.2

X = X_raw.reshape(-1, 1)

train_errors, test_errors = [], []
degrees = range(1, 12)
for d in degrees:
    pipe = Pipeline([('poly', PolynomialFeatures(d)), ('lr', LinearRegression())])
    # Train on first 60
    pipe.fit(X[:60], y_raw[:60])
    train_errors.append(np.mean((pipe.predict(X[:60]) - y_raw[:60])**2))
    test_errors.append(np.mean((pipe.predict(X[60:]) - y_raw[60:])**2))

plt.figure(figsize=(10, 5))
plt.plot(degrees, train_errors, label='Train Error', marker='o')
plt.plot(degrees, test_errors, label='Test Error', marker='s')
plt.xlabel('Polynomial Degree'), plt.ylabel('MSE')
plt.title('Bias-Variance: Effect of Model Complexity')
plt.legend(), plt.show()
print(f"Best degree: {degrees[np.argmin(test_errors)]}")"""),
        ("Bias-Variance Decomposition", """from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X_all = np.linspace(0, 1, 200).reshape(-1, 1)
y_true = np.sin(2 * np.pi * X_all.ravel())

n_bootstraps = 50
predictions = {d: [] for d in [2, 5, 10]}

for _ in range(n_bootstraps):
    idx = np.random.choice(len(X_all), 100, replace=True)
    X_b, y_b = X_all[idx], y_true[idx] + np.random.randn(100) * 0.2
    for d in [2, 5, 10]:
        model = DecisionTreeRegressor(max_depth=d)
        model.fit(X_b, y_b)
        predictions[d].append(model.predict(X_all))

for d in [2, 5, 10]:
    preds = np.array(predictions[d])  # (n_bootstraps, n_points)
    bias_sq = np.mean((preds.mean(axis=0) - y_true)**2)
    variance = np.mean(preds.var(axis=0))
    print(f"Depth {d:2d}: Bias²={bias_sq:.4f}, Variance={variance:.4f}, Total={bias_sq+variance:.4f}")"""),
        ("Regularization to Control Overfitting", """from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score

np.random.seed(42)
n_samples, n_features = 100, 50
X = np.random.randn(n_samples, n_features)
# Only first 5 features matter
y = X[:, :5] @ np.array([1, 2, -1, 0.5, 3]) + np.random.randn(n_samples) * 0.5

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
ridge_scores, lasso_scores = [], []

for alpha in alphas:
    r = cross_val_score(Ridge(alpha), X, y, cv=5, scoring='neg_mean_squared_error')
    l = cross_val_score(Lasso(alpha, max_iter=5000), X, y, cv=5, scoring='neg_mean_squared_error')
    ridge_scores.append(-r.mean())
    lasso_scores.append(-l.mean())

best_ridge = alphas[np.argmin(ridge_scores)]
best_lasso = alphas[np.argmin(lasso_scores)]
print(f"Best Ridge alpha: {best_ridge}, CV MSE: {min(ridge_scores):.4f}")
print(f"Best Lasso alpha: {best_lasso}, CV MSE: {min(lasso_scores):.4f}")"""),
    ],
    "22-cross-validation.md": [
        ("K-Fold Cross-Validation", """import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42)

# Standard k-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)

kf_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
skf_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print(f"KFold:     {kf_scores.mean():.4f} ± {kf_scores.std():.4f}")
print(f"StratKFold:{skf_scores.mean():.4f} ± {skf_scores.std():.4f}")

# Class distribution per fold
for fold_i, (_, test_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold_i+1} class ratio: {y[test_idx].mean():.3f}")"""),
        ("Nested Cross-Validation", """from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC

X, y = make_classification(n_samples=300, n_features=15, n_informative=8, random_state=42)

# Inner CV: hyperparameter search
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# Outer CV: performance estimation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']}
clf = GridSearchCV(SVC(), param_grid, cv=inner_cv, scoring='accuracy')

# Nested CV gives unbiased estimate
nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring='accuracy')
# Non-nested (optimistic bias)
clf_best = GridSearchCV(SVC(), param_grid, cv=inner_cv, scoring='accuracy').fit(X, y)
non_nested = cross_val_score(clf_best.best_estimator_, X, y, cv=outer_cv, scoring='accuracy')

print(f"Nested CV:     {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
print(f"Non-nested CV: {non_nested.mean():.4f} ± {non_nested.std():.4f}")
print(f"Optimism bias: {(non_nested.mean() - nested_scores.mean()):.4f}")"""),
        ("Time-Series Cross-Validation", """import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

np.random.seed(42)
n = 200
t = np.arange(n)
y_ts = np.sin(0.1 * t) + 0.5 * np.sin(0.05 * t) + np.random.randn(n) * 0.2

# Build lag features
def make_lag_features(y, lags=5):
    X_lag = np.column_stack([y[i:-(lags-i)] for i in range(lags)])
    return X_lag, y[lags:]

X_lag, y_lag = make_lag_features(y_ts)

tscv = TimeSeriesSplit(n_splits=5)
scores = []
for train_idx, test_idx in tscv.split(X_lag):
    model = Ridge(alpha=1.0)
    model.fit(X_lag[train_idx], y_lag[train_idx])
    pred = model.predict(X_lag[test_idx])
    mse = np.mean((pred - y_lag[test_idx])**2)
    scores.append(mse)
    print(f"Fold train size={len(train_idx)}, test size={len(test_idx)}, MSE={mse:.4f}")

print(f"Mean MSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")"""),
    ],
    "23-classification-metrics.md": [
        ("Confusion Matrix and F1 Score", """import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              precision_recall_curve, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                            weights=[0.7, 0.3], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'), plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")"""),
        ("ROC and Precision-Recall Curves", """from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)
precision, recall, pr_thresh = precision_recall_curve(y_test, y_proba)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(fpr, tpr, label=f'AUC={roc_auc_score(y_test, y_proba):.3f}')
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlabel('FPR'), ax1.set_ylabel('TPR')
ax1.set_title('ROC Curve'), ax1.legend()

ap = average_precision_score(y_test, y_proba)
ax2.plot(recall, precision, label=f'AP={ap:.3f}')
ax2.axhline(y=y_test.mean(), color='k', linestyle='--', label='Random')
ax2.set_xlabel('Recall'), ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve'), ax2.legend()

plt.tight_layout(), plt.show()"""),
        ("Threshold Selection by Business Metric", """# Choose threshold based on cost matrix
# False negative (missing fraud) costs 10x more than false positive
cost_fn, cost_fp = 10, 1

thresholds = np.linspace(0.01, 0.99, 100)
costs = []
for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    cost = fn * cost_fn + fp * cost_fp
    costs.append(cost)

best_thresh = thresholds[np.argmin(costs)]
y_pred_opt = (y_proba >= best_thresh).astype(int)

plt.plot(thresholds, costs)
plt.axvline(best_thresh, color='r', linestyle='--', label=f'Optimal={best_thresh:.2f}')
plt.xlabel('Threshold'), plt.ylabel('Total Cost')
plt.title('Threshold vs Business Cost'), plt.legend(), plt.show()

print(f"Default 0.5 threshold:")
print(classification_report(y_test, (y_proba >= 0.5).astype(int), digits=3))
print(f"Optimal {best_thresh:.2f} threshold:")
print(classification_report(y_test, y_pred_opt, digits=3))"""),
    ],
    "24-regression-metrics.md": [
        ("MSE, MAE, R² Comparison", """import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)
X, y = make_regression(n_samples=300, n_features=10, noise=20, random_state=42)

# Add outliers
outlier_idx = np.random.choice(len(y), 20)
y[outlier_idx] += np.random.randn(20) * 200

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {'OLS': LinearRegression(), 'Huber': HuberRegressor()}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{name}: RMSE={mse**0.5:.2f}, MAE={mae:.2f}, R²={r2:.4f}")"""),
        ("Residual Analysis", """from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
residuals = y_test - pred

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(pred, residuals, alpha=0.5)
axes[0].axhline(0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted'), axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted')

axes[1].hist(residuals, bins=30, edgecolor='k')
axes[1].set_title('Residual Distribution')

# QQ plot
from scipy import stats
stats.probplot(residuals, dist='norm', plot=axes[2])
axes[2].set_title('QQ Plot')

plt.tight_layout(), plt.show()
print(f"Shapiro-Wilk normality p-value: {stats.shapiro(residuals[:50]).pvalue:.4f}")"""),
        ("MAPE and Custom Metrics", """def mape(y_true, y_pred, epsilon=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)

print(f"MAPE:       {mape(y_test, pred):.2f}%")
print(f"SMAPE:      {smape(y_test, pred):.2f}%")
print(f"R²:         {r2:.4f}")
print(f"Adj. R²:    {adjusted_r2(r2, len(y_test), X_test.shape[1]):.4f}")
print(f"Max Error:  {np.max(np.abs(y_test - pred)):.2f}")"""),
    ],
    "25-feature-engineering.md": [
        ("Encoding and Scaling", """import numpy as np
import pandas as pd
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                    LabelEncoder, OneHotEncoder)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

np.random.seed(42)
n = 200
df = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'income': np.random.exponential(50000, n),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n),
    'edu': np.random.choice(['HS', 'BS', 'MS', 'PhD'], n),
    'target': np.random.randint(0, 2, n)
})

# ColumnTransformer: numeric → StandardScaler, categorical → OneHotEncoder
numeric_features = ['age', 'income']
categorical_features = ['city', 'edu']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X = df.drop('target', axis=1)
y = df['target']
X_transformed = preprocessor.fit_transform(X)
print(f"Original shape: {X.shape}, Transformed shape: {X_transformed.shape}")"""),
        ("Polynomial and Interaction Features", """from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=300, n_features=5, n_informative=4, random_state=42)
X = (X - X.mean(axis=0)) / X.std(axis=0)

results = {}
for degree in [1, 2, 3]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = LogisticRegression(max_iter=1000, C=0.1)
    cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='accuracy')
    results[degree] = (X_poly.shape[1], cv_scores.mean(), cv_scores.std())
    print(f"Degree {degree}: {X_poly.shape[1]:4d} features, "
          f"CV accuracy={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# Feature names
poly_d2 = PolynomialFeatures(degree=2, include_bias=False)
poly_d2.fit(X[:, :3])
print(f"\\nSample feature names: {poly_d2.get_feature_names_out(['x1','x2','x3'])}")"""),
        ("Feature Selection", """from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=300, n_features=20, n_informative=5, random_state=42)

# Filter: ANOVA F-test
selector_f = SelectKBest(f_classif, k=5)
X_f = selector_f.fit_transform(X, y)
selected_f = np.where(selector_f.get_support())[0]

# Wrapper: RFE
rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=5)
rfe.fit(X, y)
selected_rfe = np.where(rfe.support_)[0]

# Embedded: Random Forest importance
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
top5_rf = np.argsort(rf.feature_importances_)[-5:][::-1]

print(f"ANOVA selected:      {sorted(selected_f)}")
print(f"RFE selected:        {sorted(selected_rfe)}")
print(f"RF importance top-5: {sorted(top5_rf)}")
print(f"True informative: features 0-4")"""),
    ],
    "26-hyperparameter-tuning.md": [
        ("Grid Search vs Random Search", """import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import time

X, y = make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42)

# Grid search
grid_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
t0 = time.time()
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid_params, cv=3, n_jobs=-1)
gs.fit(X, y)
gs_time = time.time() - t0

# Random search (same budget as 1/3 of grid configs)
from scipy.stats import randint
rand_params = {
    'n_estimators': randint(50, 300),
    'max_depth': [3, 5, 10, None],
    'min_samples_split': randint(2, 20)
}
t0 = time.time()
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), rand_params,
                         n_iter=12, cv=3, n_jobs=-1, random_state=42)
rs.fit(X, y)
rs_time = time.time() - t0

print(f"Grid Search:   best={gs.best_score_:.4f}, time={gs_time:.1f}s, configs={gs.cv_results_['mean_test_score'].shape[0]}")
print(f"Random Search: best={rs.best_score_:.4f}, time={rs_time:.1f}s, configs=12")
print(f"Grid best params: {gs.best_params_}")
print(f"Random best params: {rs.best_params_}")"""),
        ("Bayesian Optimization with Optuna", """# Bayesian optimization conceptual example (without optuna dependency)
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=300, n_features=10, n_informative=6, random_state=42)

# Simulate Bayesian optimization: Gaussian Process surrogate
# For real code: pip install optuna
# import optuna
# def objective(trial):
#     C = trial.suggest_float('C', 0.01, 100, log=True)
#     gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
#     svc = SVC(C=C, gamma=gamma)
#     return cross_val_score(svc, X, y, cv=3).mean()
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)
# print(f"Best: {study.best_value:.4f}, params: {study.best_params}")

# Manual approximation: log-uniform sampling
np.random.seed(42)
best_score, best_C = 0, None
for _ in range(30):
    C = 10 ** np.random.uniform(-2, 2)  # Log-uniform in [0.01, 100]
    gamma = np.random.choice(['scale', 'auto'])
    score = cross_val_score(SVC(C=C, gamma=gamma), X, y, cv=3).mean()
    if score > best_score:
        best_score, best_C, best_gamma = score, C, gamma

print(f"Best score: {best_score:.4f}, C={best_C:.4f}, gamma={best_gamma}")"""),
        ("Early Stopping and Halving", """from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import randint, uniform
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(2, 8),
    'learning_rate': uniform(0.01, 0.3),
    'min_samples_leaf': randint(1, 20)
}

# Successive Halving: start with many configs, eliminate worst each round
sh = HalvingRandomSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions,
    n_candidates=100,
    factor=3,
    cv=3,
    random_state=42,
    n_jobs=-1
)
sh.fit(X, y)

print(f"Best score: {sh.best_score_:.4f}")
print(f"Best params: {sh.best_params_}")
print(f"Iterations run: {sh.n_iterations_}")
print(f"Total fits: {sh.n_resources_}")"""),
    ],
    "27-ensemble-methods.md": [
        ("Voting and Averaging Ensemble", """import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=600, n_features=20, n_informative=12, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Individual models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000)
svc = SVC(probability=True)

# Hard voting
hard_voter = VotingClassifier([('rf', rf), ('gb', gb), ('lr', lr)], voting='hard')
# Soft voting (uses predicted probabilities)
soft_voter = VotingClassifier([('rf', rf), ('gb', gb), ('svc', svc)], voting='soft')

for name, model in [('RF', rf), ('GB', gb), ('LR', lr), ('Hard', hard_voter), ('Soft', soft_voter)]:
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name:6s}: {scores.mean():.4f} ± {scores.std():.4f}")"""),
        ("Stacking Ensemble", """from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score

base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000)),
]
# Meta-learner (level-1 model)
meta_learner = LogisticRegression(max_iter=1000)

stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,  # k-fold for generating meta-features
    passthrough=False  # Don't pass original features to meta-learner
)

scores = cross_val_score(stacking, X_train, y_train, cv=5, scoring='accuracy')
print(f"Stacking CV: {scores.mean():.4f} ± {scores.std():.4f}")

stacking.fit(X_train, y_train)
test_acc = accuracy_score(y_test, stacking.predict(X_test))
print(f"Stacking test accuracy: {test_acc:.4f}")"""),
        ("Boosting vs Bagging Analysis", """from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=500, n_features=15, n_informative=8, random_state=42)

n_estimators_range = range(10, 201, 10)
results = {'Bagging': [], 'AdaBoost': [], 'RandomForest': []}

for n in n_estimators_range:
    bagging = BaggingClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=n, random_state=42)
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n, random_state=42)
    rf = RandomForestClassifier(n_estimators=n, random_state=42)

    for name, model in [('Bagging', bagging), ('AdaBoost', ada), ('RandomForest', rf)]:
        score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
        results[name].append(score)

plt.figure(figsize=(10, 5))
for name, scores in results.items():
    plt.plot(list(n_estimators_range), scores, label=name, marker='o', markersize=3)
plt.xlabel('n_estimators'), plt.ylabel('CV Accuracy')
plt.title('Ensemble Size: Boosting vs Bagging vs Random Forest')
plt.legend(), plt.show()"""),
    ],
    "28-bayesian-inference.md": [
        ("Bayesian Coin Flip Update", """import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Prior: Beta distribution (represents beliefs about p_heads)
alpha_prior, beta_prior = 2, 2  # Weak prior: slightly prefer fair coin

# Observe coin flips
np.random.seed(42)
true_p = 0.7
observations = np.random.binomial(1, true_p, size=100)

p_range = np.linspace(0, 1, 500)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, n_obs in enumerate([0, 5, 20, 100]):
    heads = observations[:n_obs].sum() if n_obs > 0 else 0
    tails = n_obs - heads
    # Posterior: Beta(alpha_prior + heads, beta_prior + tails)
    alpha_post = alpha_prior + heads
    beta_post = beta_prior + tails
    posterior = stats.beta(alpha_post, beta_post)

    axes[i].plot(p_range, stats.beta(alpha_prior, beta_prior).pdf(p_range), 'b--', label='Prior', alpha=0.5)
    axes[i].plot(p_range, posterior.pdf(p_range), 'r-', label='Posterior')
    axes[i].axvline(true_p, color='g', linestyle=':', label='True p')
    axes[i].set_title(f'After {n_obs} flips (H={heads})')
    axes[i].set_xlabel('p(heads)'), axes[i].legend()

plt.tight_layout(), plt.show()
print(f"Final posterior: mean={alpha_post/(alpha_post+beta_post):.3f}, "
      f"95% CI={stats.beta(alpha_post, beta_post).interval(0.95)}")"""),
        ("MAP vs MLE Estimation", """import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
# Small dataset: n=10, true mean=5, true sigma=2
n = 10
true_mu, true_sigma = 5.0, 2.0
data = np.random.normal(true_mu, true_sigma, n)

# MLE: just the sample mean and std
mle_mu = data.mean()
mle_sigma = data.std()

# MAP with Gaussian prior on mean: mu_prior=0, sigma_prior=3
mu_prior, sigma_prior = 0.0, 3.0
# MAP posterior mean: weighted average of prior and MLE
precision_prior = 1 / sigma_prior**2
precision_likelihood = n / true_sigma**2
map_mu = (precision_prior * mu_prior + precision_likelihood * mle_mu) / (precision_prior + precision_likelihood)

print(f"True mu:  {true_mu:.3f}")
print(f"MLE mu:   {mle_mu:.3f}  (shrinkage toward prior: {abs(mle_mu - true_mu):.3f} error)")
print(f"MAP mu:   {map_mu:.3f}  (shrinkage toward prior: {abs(map_mu - true_mu):.3f} error)")
print(f"\\nMAP pulls estimate from MLE={mle_mu:.3f} toward prior={mu_prior:.1f}")
print(f"MAP estimate: {map_mu:.3f} (halfway between for this prior strength)")"""),
        ("Bayesian Linear Regression", """import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
n = 30
X_data = np.linspace(-3, 3, n)
true_slope, true_intercept = 2.0, 1.0
y_data = true_intercept + true_slope * X_data + np.random.randn(n)

# Bayesian linear regression (conjugate prior)
# Prior: w ~ N(0, alpha^-1 * I), noise precision beta
alpha, beta = 1.0, 1.0  # Prior precision, noise precision

# Design matrix
Phi = np.c_[np.ones(n), X_data]  # (n, 2)

# Posterior: S_N = (alpha*I + beta * Phi^T Phi)^-1
S_N_inv = alpha * np.eye(2) + beta * Phi.T @ Phi
S_N = np.linalg.inv(S_N_inv)
m_N = beta * S_N @ Phi.T @ y_data  # Posterior mean

print(f"True:              intercept={true_intercept:.2f}, slope={true_slope:.2f}")
print(f"Posterior mean:    intercept={m_N[0]:.2f}, slope={m_N[1]:.2f}")
print(f"Posterior std:     intercept={np.sqrt(S_N[0,0]):.4f}, slope={np.sqrt(S_N[1,1]):.4f}")

# Predictive distribution
X_test = np.linspace(-4, 4, 100)
Phi_test = np.c_[np.ones(100), X_test]
pred_mean = Phi_test @ m_N
pred_var = 1/beta + np.sum(Phi_test @ S_N * Phi_test, axis=1)

plt.figure(figsize=(10, 5))
plt.scatter(X_data, y_data, alpha=0.6, label='Data')
plt.plot(X_test, pred_mean, 'r-', label='Posterior mean')
plt.fill_between(X_test, pred_mean - 2*np.sqrt(pred_var),
                  pred_mean + 2*np.sqrt(pred_var), alpha=0.2, color='r', label='95% CI')
plt.legend(), plt.title('Bayesian Linear Regression'), plt.show()"""),
    ],
}

for filename, examples in EXAMPLES.items():
    with open(filename, 'r') as f:
        content = f.read()

    # Find the Code Examples section
    start = content.find("## Code Examples")
    end = content.find("## Related Concepts", start)
    if start == -1:
        print(f"⚠ No Code Examples in {filename}")
        continue

    new_examples = "## Code Examples\n\n"
    for i, (title, code) in enumerate(examples, 1):
        new_examples += f"### Example {i}: {title}\n\n```python\n{code}\n```\n\n"

    new_content = content[:start] + new_examples + content[end:]
    with open(filename, 'w') as f:
        f.write(new_content)
    print(f"✓ {filename}")

print("\n✅ Fixed Code Examples for all 8 remaining concepts")
