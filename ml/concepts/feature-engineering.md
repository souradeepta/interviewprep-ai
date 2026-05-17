# Feature Engineering

## TL;DR
Transforming raw data into informative inputs for ML models. Often the highest-leverage activity
in applied ML — better features beat better models. Includes encoding categoricals, handling
missing values, creating interaction features, and scaling.

## Core Intuition
Models can only learn from what you give them. A raw date "2023-11-15" is opaque to a linear model
but powerful when decomposed into day_of_week=2, month=11, is_holiday=False. Feature engineering
encodes domain knowledge into a form models can exploit.

## How It Works

**Numerical:** StandardScaler (zero mean, unit variance), log transform for right-skewed distributions, binning for tree models.

**Categorical:**
- One-hot encoding: low-cardinality (< 50 categories)
- Target encoding: high-cardinality — replace category with target mean (use inside CV folds to prevent leakage)
- Embeddings: deep learning, high-cardinality

**Missing values:** mean/median imputation (simple), model-based imputation, add "was_missing" binary feature.

**Interactions:** price_per_sqft = price / sqft, polynomial features (x², xy), date decomposition.

## Key Properties / Trade-offs
- Tree models (XGBoost, RF) don't need feature scaling; linear/distance models do
- Target encoding risks leakage if not done inside CV folds
- One-hot + high cardinality = high-dimensional sparse matrix (bad)

## Common Mistakes / Gotchas
- Scaling on full dataset before train/test split → data leakage
- One-hot encoding a feature with thousands of unique values
- Ignoring missingness pattern — "missing" can be informative

## Code Example
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer([
    ('num', Pipeline([('scaler', StandardScaler())]), ['age', 'salary']),
    ('cat', Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))]), ['city'])
])
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Target vs one-hot encoding?" | One-hot for low-cardinality; target encoding for high-cardinality. Target encoding must be inside CV folds to avoid leakage. |
| "Do tree models need scaling?" | No — tree splits on thresholds, invariant to monotonic transforms. Linear models and KNN do. |
| "Missing value strategies?" | Mean/median imputation + add binary "was_missing" indicator column. Or model-based imputation for complex missingness. |

## Related Topics
- [Supervised Learning](supervised-learning.md) — [Evaluation Metrics](evaluation-metrics.md)

## Resources
- [Feature Engineering and Selection](http://www.feat.engineering/) — Kuhn & Johnson. Free online.
