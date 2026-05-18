# AI Fundamentals

Comprehensive coverage of machine learning and deep learning fundamentals, from core optimization through classical ML algorithms to neural networks and advanced evaluation techniques.

## Overview

**28 concepts** across 6 major areas:
- **5 Core Optimization:** Gradient descent, backpropagation, loss functions, optimization algorithms, learning rate scheduling
- **7 Classical ML:** Linear regression, logistic regression, decision trees, random forests, gradient boosting, SVMs, KNN
- **5 Neural Networks:** Architecture, activation functions, weight initialization, regularization, batch normalization
- **3 Unsupervised Learning:** K-means, dimensionality reduction, GMMs
- **4 Model Evaluation:** Bias-variance, cross-validation, classification metrics, regression metrics
- **4 Advanced Topics:** Feature engineering, hyperparameter tuning, ensemble methods, Bayesian inference

## Learning Path

### Beginner
1. [Gradient Descent](./concepts/01-gradient-descent.md) — Foundation of all learning
2. [Loss Functions](./concepts/03-loss-functions.md) — What we optimize
3. [Linear Regression](./concepts/06-linear-regression.md) — First supervised model
4. [Classification Metrics](./concepts/23-classification-metrics.md) — How to evaluate

### Intermediate
5. [Backpropagation](./concepts/02-backpropagation.md) — Training neural networks
6. [Neural Networks](./concepts/13-neural-networks.md) — Multi-layer models
7. [Decision Trees](./concepts/08-decision-trees.md) — Non-linear patterns
8. [Cross-Validation](./concepts/22-cross-validation.md) — Honest evaluation
9. [Regularization](./concepts/16-regularization.md) — Fighting overfitting

### Advanced
10. [Optimization Algorithms](./concepts/04-optimization-algorithms.md) — Adam, RMSprop, etc.
11. [Ensemble Methods](./concepts/27-ensemble-methods.md) — Combining models
12. [Hyperparameter Tuning](./concepts/26-hyperparameter-tuning.md) — Grid/Bayesian search
13. [Bayesian Inference](./concepts/28-bayesian-inference.md) — Probabilistic ML

## All Concepts

### Group 1: Core Optimization (5)

| # | Concept | Key Topics |
|---|---------|-----------|
| 01 | [Gradient Descent](./concepts/01-gradient-descent.md) | SGD, momentum, learning rate, convergence |
| 02 | [Backpropagation](./concepts/02-backpropagation.md) | Chain rule, computational graphs, vanishing gradients |
| 03 | [Loss Functions](./concepts/03-loss-functions.md) | MSE, cross-entropy, custom losses |
| 04 | [Optimization Algorithms](./concepts/04-optimization-algorithms.md) | Adam, RMSprop, Adagrad, AdamW |
| 05 | [Learning Rate Scheduling](./concepts/05-learning-rate-scheduling.md) | Warmup, cosine annealing, step decay |

### Group 2: Classical ML - Supervised (7)

| # | Concept | Key Topics |
|---|---------|-----------|
| 06 | [Linear Regression](./concepts/06-linear-regression.md) | OLS, Ridge, Lasso, regularization |
| 07 | [Logistic Regression](./concepts/07-logistic-regression.md) | Binary/multinomial classification |
| 08 | [Decision Trees](./concepts/08-decision-trees.md) | CART, Gini, entropy, pruning |
| 09 | [Random Forests](./concepts/09-random-forests.md) | Bagging, feature importance |
| 10 | [Gradient Boosting](./concepts/10-gradient-boosting.md) | XGBoost, LightGBM, AdaBoost |
| 11 | [Support Vector Machines](./concepts/11-support-vector-machines.md) | Margin, kernel trick, SVMs |
| 12 | [K-Nearest Neighbors](./concepts/12-k-nearest-neighbors.md) | Distance metrics, k selection |

### Group 3: Neural Networks (5)

| # | Concept | Key Topics |
|---|---------|-----------|
| 13 | [Neural Networks](./concepts/13-neural-networks.md) | MLPs, forward pass, universal approximation |
| 14 | [Activation Functions](./concepts/14-activation-functions.md) | ReLU, sigmoid, tanh, GELU |
| 15 | [Weight Initialization](./concepts/15-weight-initialization.md) | Xavier, He initialization |
| 16 | [Regularization](./concepts/16-regularization.md) | L1, L2, dropout, early stopping |
| 17 | [Batch Normalization](./concepts/17-batch-normalization.md) | Internal covariate shift, layer norm |

### Group 4: Unsupervised Learning (3)

| # | Concept | Key Topics |
|---|---------|-----------|
| 18 | [K-Means Clustering](./concepts/18-k-means-clustering.md) | Elbow method, silhouette score |
| 19 | [Dimensionality Reduction](./concepts/19-dimensionality-reduction.md) | PCA, t-SNE, UMAP |
| 20 | [Gaussian Mixture Models](./concepts/20-gaussian-mixture-models.md) | EM algorithm, soft assignments |

### Group 5: Model Evaluation (4)

| # | Concept | Key Topics |
|---|---------|-----------|
| 21 | [Bias-Variance Tradeoff](./concepts/21-bias-variance-tradeoff.md) | Decomposition, underfitting/overfitting |
| 22 | [Cross-Validation](./concepts/22-cross-validation.md) | k-fold, stratified, time-series CV |
| 23 | [Classification Metrics](./concepts/23-classification-metrics.md) | Accuracy, precision, recall, F1, ROC-AUC |
| 24 | [Regression Metrics](./concepts/24-regression-metrics.md) | MSE, MAE, R², MAPE |

### Group 6: Advanced Foundations (4)

| # | Concept | Key Topics |
|---|---------|-----------|
| 25 | [Feature Engineering](./concepts/25-feature-engineering.md) | Encoding, scaling, feature selection |
| 26 | [Hyperparameter Tuning](./concepts/26-hyperparameter-tuning.md) | Grid search, random search, Bayesian |
| 27 | [Ensemble Methods](./concepts/27-ensemble-methods.md) | Stacking, blending, voting |
| 28 | [Bayesian Inference](./concepts/28-bayesian-inference.md) | Priors, posteriors, MAP estimation |

## Using This Section

### For Learning
Each concept includes:
- **Markdown concept file** — Theory, intuition, best practices, common pitfalls, code examples
- **Jupyter notebook** — 3-level implementation (basic, advanced, 3 real-world examples)
- **Mermaid diagrams** — Visual representation of algorithms and flows

### For Interviews
- Read the markdown for conceptual understanding
- Study the Interview Q&A sections for judgment-based questions
- Review Best Practices and Common Pitfalls for system design context
- Run the notebooks to solidify understanding

### For Implementation
- Copy code examples from notebooks (100% real, no pseudo-code)
- Use as reference during development
- Adapt examples to your specific problem domain

## Technology Stack

- **NumPy** — Numerical computation, gradient descent implementations
- **Scikit-learn** — Classical ML algorithms, metrics, preprocessing
- **Matplotlib/Seaborn** — Visualizations, loss curves, decision boundaries
- **PyTorch** (optional) — Neural network implementations
- **Pandas** — Data manipulation and analysis

## Structure

```
ai/
├── concepts/               # 28 markdown files
│   ├── 01-gradient-descent.md
│   ├── 02-backpropagation.md
│   └── ... (all 28 concepts)
├── notebooks/              # 28 Jupyter notebooks
│   ├── 01-gradient-descent.ipynb
│   ├── 02-backpropagation.ipynb
│   └── ... (all 28 concepts)
└── README.md              # This file
```

## Best Practices for Using This Section

1. **Understand the Theory First** — Read the markdown for intuition
2. **Run the Code** — Execute notebooks to see algorithms in action
3. **Modify and Experiment** — Change hyperparameters, try different datasets
4. **Connect Concepts** — Follow the "Related Concepts" links to build mental models
5. **Interview Prep** — Focus on Q&A sections for judgment-based reasoning

## Common Learning Paths

### ML Engineer Interview
1. Fundamentals: 01, 02, 03, 04, 06, 07
2. Deep: 13, 14, 15, 16, 17
3. Evaluation: 21, 22, 23, 24
4. Advanced: 26, 27

### Data Scientist Interview
1. Fundamentals: 06, 07, 08, 09, 10
2. Evaluation: 21, 22, 23, 24
3. Advanced: 25, 26, 27
4. Bayesian: 28

### System Design Interview
1. Optimization: 01, 04, 05
2. Scaling: 22, 25, 27
3. Evaluation: 23, 24
4. Production patterns: Review Best Practices sections

## What You'll Learn

After working through this section:
- ✅ Understand how neural networks learn (backprop + gradient descent)
- ✅ Choose appropriate algorithms for different problem types
- ✅ Implement and tune models in Python
- ✅ Evaluate models rigorously (metrics, cross-validation, bias-variance)
- ✅ Optimize hyperparameters systematically
- ✅ Combine multiple models (ensembles)
- ✅ Handle real-world challenges (imbalance, missing data, scaling)

## Quick Start

1. Start with [Gradient Descent](./concepts/01-gradient-descent.md) to understand optimization
2. Follow with [Backpropagation](./concepts/02-backpropagation.md) for neural networks
3. Jump to [Neural Networks](./concepts/13-neural-networks.md) for architecture
4. Use [Classification Metrics](./concepts/23-classification-metrics.md) for evaluation
5. Review [Hyperparameter Tuning](./concepts/26-hyperparameter-tuning.md) for optimization

## Related Sections

- **[Agentic AI](../agentic-ai/)** — 52 concepts on building intelligent agents
- **[LLM](../llm/)** — 32 concepts on transformer-based language models
- **[ML Ops](../mlops/)** — Production ML systems and deployment

---

**Status:** ✅ All 28 concepts complete with markdown + notebooks
**Total Code Examples:** 400+ production patterns
**Ready for:** Interviews, education, reference, implementation
