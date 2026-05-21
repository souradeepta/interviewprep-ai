# 25 Feature Engineering
# Extracted from Jupyter notebook

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Base features
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

base = RandomForestRegressor(n_estimators=50, random_state=0).fit(Xtr, ytr)
print(f"Base R²: {base.score(Xte, yte):.4f}")

# Add interaction features
X_eng = X.copy()
X_eng['rooms_per_person'] = X['AveRooms'] / X['AveOccup']
X_eng['pop_per_house']    = X['Population'] / X['HouseAge']

Xtr_e, Xte_e = train_test_split(X_eng, test_size=0.2, random_state=42)[0::1][:2]
Xtr_e, Xte_e = train_test_split(X_eng, test_size=0.2, random_state=42)[:2]
eng = RandomForestRegressor(n_estimators=50, random_state=0).fit(Xtr_e, ytr)
print(f"Engineered R²: {eng.score(Xte_e, yte):.4f}")

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

class EnhancedFeatureEngineering:

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



model = EnhancedFeatureEngineering()

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
model = EnhancedFeatureEngineering()
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# Example 2: Categorical encoding strategies
np.random.seed(42)
n = 400
df = pd.DataFrame({
    'city': np.random.choice(['A','B','C','D'], n),
    'size': np.random.randn(n),
    'price': np.zeros(n)
})
city_effect = {'A': 1.0, 'B': 2.5, 'C': -0.5, 'D': 1.8}
df['price'] = df['city'].map(city_effect) + df['size']*0.5 + np.random.randn(n)*0.2

# One-hot
ohe = pd.get_dummies(df[['city','size']])
# Ordinal
df['city_ord'] = df['city'].map({'A':0,'B':1,'C':2,'D':3})

for feats, name in [(ohe,"One-hot"), (df[['city_ord','size']],"Ordinal")]:
    from sklearn.linear_model import LinearRegression
    m = LinearRegression().fit(feats, df['price'])
    print(f"{name}: R²={m.score(feats, df['price']):.4f}")

# Example 3: Feature selection with importance
from sklearn.feature_selection import SelectFromModel

X, y = fetch_california_housing(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr)
selector = SelectFromModel(rf, threshold='mean', prefit=True)
Xtr_sel, Xte_sel = selector.transform(Xtr), selector.transform(Xte)

rf_sel = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr_sel, ytr)
print(f"All features ({X.shape[1]}):  R²={rf.score(Xte,yte):.4f}")
print(f"Selected ({Xtr_sel.shape[1]}): R²={rf_sel.score(Xte_sel,yte):.4f}")