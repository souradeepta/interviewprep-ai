# 28 Bayesian Inference
# Extracted from Jupyter notebook

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Bayesian coin flip: prior + data → posterior
# Prior: Beta(alpha, beta) – belief before seeing data
# Likelihood: Binomial
# Posterior: Beta(alpha + heads, beta + tails)  [conjugate]

heads, tails = 6, 4   # observed data
prior_a, prior_b = 2, 2  # weakly informative prior

post_a = prior_a + heads
post_b = prior_b + tails

theta = np.linspace(0, 1, 200)
prior     = stats.beta.pdf(theta, prior_a, prior_b)
likelihood = stats.binom.pmf(heads, heads+tails, theta)
posterior = stats.beta.pdf(theta, post_a, post_b)

plt.plot(theta, prior/prior.max(),     label=f"Prior Beta({prior_a},{prior_b})", linestyle='--')
plt.plot(theta, likelihood/likelihood.max(), label="Likelihood (scaled)", linestyle=':')
plt.plot(theta, posterior/posterior.max(), label=f"Posterior Beta({post_a},{post_b})", linewidth=2)
plt.xlabel("θ (coin bias)"); plt.ylabel("Density (scaled)")
plt.legend(); plt.title("Bayesian Update: Prior × Likelihood → Posterior"); plt.show()

print(f"Prior mean:     {prior_a/(prior_a+prior_b):.3f}")
print(f"MLE estimate:   {heads/(heads+tails):.3f}")
print(f"Posterior mean: {post_a/(post_a+post_b):.3f}")

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

class EnhancedBayesianInference:

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



model = EnhancedBayesianInference()

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
model = EnhancedBayesianInference()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Bayesian linear regression vs OLS
from sklearn.datasets import make_regression
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=50, n_features=10, noise=20, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

ols = LinearRegression().fit(Xtr, ytr)
bay = BayesianRidge().fit(Xtr, ytr)

ols_r2 = r2_score(yte, ols.predict(Xte))
bay_r2 = r2_score(yte, bay.predict(Xte))

print(f"OLS R²     : {ols_r2:.4f}")
print(f"Bayesian R²: {bay_r2:.4f}")

# Bayesian gives uncertainty estimates
y_pred, y_std = bay.predict(Xte, return_std=True)
print(f"\nPrediction uncertainty (std): {y_std[:5].round(2)}")
print("OLS gives no uncertainty – Bayesian tells you confidence intervals!")

# Example 3: MAP estimation vs MLE
np.random.seed(0)
n = 30
X_obs = np.random.randn(n)   # True mean = 0, std = 1

# MLE: maximise P(data | theta)
mle_mean = X_obs.mean()

# MAP: maximise P(theta | data) ∝ P(data|theta) * P(theta)
# Prior: mean ~ N(mu0=2, sigma0=1)  (strong prior belief mean is 2)
mu0, sigma0 = 2.0, 1.0
sigma = 1.0

# MAP for Gaussian: weighted average of prior and data
map_mean = (mu0/sigma0**2 + X_obs.sum()/sigma**2) / (1/sigma0**2 + n/sigma**2)

print(f"True mean  : 0.00")
print(f"MLE mean   : {mle_mean:.3f}  (data only)")
print(f"MAP mean   : {map_mean:.3f}  (data + prior, prior pulls toward {mu0})")
print(f"\nWith more data, MAP → MLE (data overwhelms prior)")

# Show convergence with data size
for n_ in [10, 50, 200, 1000]:
    X_n = np.random.randn(n_)
    map_n = (mu0/sigma0**2 + X_n.sum()) / (1/sigma0**2 + n_)
    print(f"  n={n_:5d}  MAP={map_n:.4f}")