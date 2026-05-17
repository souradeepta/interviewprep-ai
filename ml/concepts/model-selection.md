# Model Selection

## TL;DR
Choosing the right model class (linear regression, tree, neural net, SVM) for your problem. Trade-offs: interpretability vs accuracy, speed vs richness, data requirements. Baseline (simple, interpretable) → main model (accuracy) → ensemble (combine multiple).

## Core Intuition
Different tools for different jobs. A hammer (simple model) works fast; a drill (complex model) is powerful but slower and overkill for nails.

## How It Works

**Decision tree:**
- ✓ Fast, interpretable, handles nonlinear + categorical
- ✗ Overfits, high variance
- Use: baseline or with bagging/boosting

**Linear model (logistic/ridge regression):**
- ✓ Fast, interpretable, small data, online learning
- ✗ Poor nonlinear fit
- Use: baseline, when interpretability critical

**Random Forest:**
- ✓ Robust, handles mixed data types, feature importance
- ✗ Slower than trees, less interpretable
- Use: general-purpose, default choice

**Gradient Boosting (XGBoost, LightGBM):**
- ✓ Highest accuracy on tabular data, handles nonlinearity + interactions
- ✗ Slower to train, harder to tune
- Use: when accuracy most important

**Neural Networks:**
- ✓ Handles images, sequences, highly nonlinear
- ✗ Needs lots of data, slow to train, less interpretable
- Use: images, text, if tabular models insufficient

**SVM:**
- ✓ Good generalization, kernel trick, interpretable margin
- ✗ Slow to train, scales poorly
- Use: small datasets, when margin interpretation matters

## Key Properties / Trade-offs
| Model | Speed | Accuracy | Data Req | Interpret | Tune Effort |
|---|---|---|---|---|---|
| Linear | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| Tree | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| RF | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| XGBoost | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| NN | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |

## Common Mistakes / Gotchas
- **Jumping to complex models:** always start with linear/tree baseline
- **No feature engineering for NN:** NNs benefit from good features too
- **Ignoring inference latency:** fast training ≠ fast serving

## Framework
1. **Baseline:** simple model (logistic regression, decision tree)
2. **Main model:** higher-capacity model (RF, XGBoost)
3. **Optional ensemble:** combine multiple models
4. **Production choice:** balance accuracy, latency, maintainability

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Which model to use?" | Start with linear/tree baseline. If insufficient, try RF/XGBoost. NN only if images/sequences. |
| "NN vs tree?" | Tree fast + interpretable. NN powerful but needs data + compute. |
| "No free lunch?" | No model dominates all problems. Match model to data and task. |

## Related Topics
- [Hyperparameter Tuning](hyperparameter-tuning.md) — [Evaluation Metrics](evaluation-metrics.md)

## Resources
- [Scikit-learn Algorithm Cheat Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
