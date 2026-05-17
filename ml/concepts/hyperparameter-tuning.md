# Hyperparameter Tuning

## TL;DR
Finding optimal hyperparameters (learning rate, tree depth, regularization) that maximize validation performance. Grid search exhausts all combinations; random search samples; Bayesian optimization intelligently explores. Trade-off: thorough search takes time but finds better params.

## Core Intuition
Every model has dials: learning rate, dropout, regularization strength, etc. Turn them the wrong way and performance tanks. Tuning finds the sweet spot.

## How It Works

**Grid Search:** specify ranges, try all combinations.
- Params: `lr ∈ [0.001, 0.01, 0.1]`, `dropout ∈ [0.2, 0.5]`
- Total: 3 × 2 = 6 combinations to try
- Exhaustive, guaranteed to find best in the grid

**Random Search:** sample hyperparams randomly.
- Try N random combinations
- Faster than grid if N << total combinations
- Often finds better params (distribution)

**Bayesian Optimization:** model performance as function of params, iteratively sample promising regions.
- Requires fewer evaluations than grid/random
- Harder to implement (use Hyperopt, Optuna)
- Best for expensive evaluations (large models)

**Early Stopping:** stop training when validation loss plateaus.
- Prevents overfitting to training set
- Saves compute time

## Key Properties / Trade-offs
- **Compute cost:** grid O(total_combos), random O(N), Bayesian O(N + model cost)
- **Coverage:** grid exhaustive, random probabilistic, Bayesian focused
- **Curse of dimensionality:** with 10 params and 10 values each, grid = 10^10 combos (infeasible)

## Common Mistakes / Gotchas
- **Tuning on test set:** data leakage. Use validation set or CV
- **Nested CV:** if tuning inside CV loop, need nested CV (outer for eval, inner for tuning)
- **Log scale:** for learning rates and regularization, use log scale (0.001, 0.01, 0.1) not linear
- **One param at a time:** interactive tuning misses interactions

## Code Example
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
```

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Grid vs random search?" | Grid exhaustive, random faster. Random often better (distribution). |
| "Why nested CV?" | Tuning on same CV folds as evaluation → optimistic bias. Outer CV for eval, inner for tuning. |
| "When Bayesian optimization?" | Expensive models or many hyperparams. Standard grid/random sufficient for most cases. |

## Related Topics
- [Model Selection](model-selection.md) — [Cross-Validation Strategies](cross-validation-strategies.md)

## Resources
- [Hyperopt: Bayesian Optimization](http://hyperopt.github.io/hyperopt/)
- [Optuna: Modern Hyperparameter Optimization](https://optuna.org/)
