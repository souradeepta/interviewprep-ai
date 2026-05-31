# Statistics for Machine Learning Interviews

Statistics is the mathematical foundation of every machine learning algorithm. Understanding statistics at interview depth means being able to answer not just "what is this?" but "when would you use it, what breaks if you don't, and how would you debug it in production?" This section covers the 15 core statistical concepts that appear most frequently in ML engineering and data science interviews.

## Prerequisites

- **Calculus**: partial derivatives, chain rule, optimization (gradient, Hessian)
- **Linear Algebra**: matrix multiplication, eigendecomposition, positive definiteness, SVD
- **Basic Probability**: random variables, expectations, variance, conditional probability

No prior deep knowledge of statistics is assumed, but comfort with notation like E[X], Var[X], and p(x|y) is expected.

---

## Concept Table

| # | Name | Category | Notebook |
|---|------|----------|---------|
| 01 | Probability Fundamentals | Foundation | [01-probability-fundamentals.ipynb](notebooks/01-probability-fundamentals.ipynb) |
| 02 | Distributions | Foundation | [02-distributions.ipynb](notebooks/02-distributions.ipynb) |
| 03 | Probability Distributions | Foundation | [03-probability-distributions.ipynb](notebooks/03-probability-distributions.ipynb) |
| 04 | Estimation Theory | Foundation | [04-estimation-theory.ipynb](notebooks/04-estimation-theory.ipynb) |
| 05 | Statistical Inference | Inference | [05-statistical-inference.ipynb](notebooks/05-statistical-inference.ipynb) |
| 06 | Hypothesis Testing | Inference | [06-hypothesis-testing.ipynb](notebooks/06-hypothesis-testing.ipynb) |
| 07 | Regression Analysis | Modeling | [07-regression-analysis.ipynb](notebooks/07-regression-analysis.ipynb) |
| 08 | Bayesian Inference | Bayesian | [08-bayesian-inference.ipynb](notebooks/08-bayesian-inference.ipynb) |
| 09 | Experimental Design | Inference | [09-experimental-design.ipynb](notebooks/09-experimental-design.ipynb) |
| 10 | Resampling Methods | Computation | [10-resampling-methods.ipynb](notebooks/10-resampling-methods.ipynb) |
| 11 | Monte Carlo Sampling | Computation | [11-monte-carlo-sampling.ipynb](notebooks/11-monte-carlo-sampling.ipynb) |
| 12 | Markov Chains and MCMC | Bayesian | [12-markov-chains-mcmc.ipynb](notebooks/12-markov-chains-mcmc.ipynb) |
| 13 | Multivariate Statistics | Multivariate | [13-multivariate-statistics.ipynb](notebooks/13-multivariate-statistics.ipynb) |
| 14 | Time Series Statistics | Temporal | [14-time-series-statistics.ipynb](notebooks/14-time-series-statistics.ipynb) |
| 15 | Statistical-ML Connections | Synthesis | [15-statistical-ml-connections.ipynb](notebooks/15-statistical-ml-connections.ipynb) |

---

## Learning Paths

Different roles and interview types emphasize different subsets of these concepts. Use these curated paths to focus your preparation.

### Stats for ML Interviews (Core Track)
**Best for**: MLE, Data Scientist, Applied Scientist roles at any company.
**Concepts**: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08

This track covers the absolute fundamentals that appear in virtually every ML interview: probability, common distributions, estimation, hypothesis testing, and regression. Complete this track before anything else.

**Key questions you will be able to answer after this track:**
- Why is the normal distribution so central? (CLT, MLE under Gaussian noise)
- What is a p-value really? Why is p < 0.05 not the same as "important"?
- When would you use OLS vs regularized regression?
- What is the difference between Bayesian credible intervals and frequentist confidence intervals?

---

### Bayesian Track
**Best for**: Research scientists, probabilistic ML roles, NLP/CV researchers.
**Concepts**: 01 → 03 → 08 → 12

This track goes deep on the Bayesian perspective: prior-posterior updating, MCMC for posterior sampling, and the connections between Bayesian inference and regularization.

**Key questions you will be able to answer:**
- How do you choose a prior, and how sensitive are results to that choice?
- What is the difference between MCMC and variational inference?
- Why does Ridge regression correspond to a Gaussian prior? What prior does Lasso correspond to?
- How do you diagnose MCMC convergence, and why is ESS more informative than sample count?

---

### Production ML Statistics
**Best for**: MLE at companies running A/B tests, deploying models at scale, or working on data infrastructure.
**Concepts**: 05 → 06 → 07 → 08 → 09 → 10

This track focuses on the statistical skills needed for production ML: A/B testing, experiment design, regression diagnostics, uncertainty quantification, and resampling for robust estimates.

**Key questions you will be able to answer:**
- How do you design an A/B test with sufficient power to detect a 2% lift?
- What is the difference between statistical significance and practical significance?
- How do you validate that a model works on out-of-distribution data?
- When would you use the bootstrap instead of asymptotic standard errors?

---

### Advanced Computation Track
**Best for**: Research engineers, ML infrastructure, probabilistic programming.
**Concepts**: 10 → 11 → 12 → 13 → 15

This advanced track covers computational statistics: Monte Carlo methods, MCMC, multivariate analysis, and the statistical foundations of ML regularization and optimization.

**Key questions you will be able to answer:**
- How does importance sampling reduce the samples needed to estimate rare-event probabilities?
- What is the bias-variance tradeoff, and how does it connect to regularization?
- When does PCA give misleading results, and what do you use instead?
- How is the EM algorithm related to variational inference and KL divergence?

---

## How to Use This Material

Each concept has two files:
- **Markdown concept file** (`concepts/NN-name.md`): 8-section deep-dive with Detailed Explanation, Core Intuition, How It Works, Architecture/Trade-offs, Interview Q&A, Best Practices, Common Pitfalls, and Related Concepts.
- **Jupyter notebook** (`notebooks/NN-name.ipynb`): 12-cell hands-on implementation with three levels of complexity (basic → advanced → real-world examples) and a comparative visualization.

**Recommended study method:**
1. Read the concept markdown file end-to-end (15-20 minutes)
2. Run the notebook, stopping to understand each cell (30-60 minutes)
3. Review the Interview Q&A section and answer each question aloud before reading the answer
4. Try the exercises at the end of each notebook

**Available libraries for code**: `numpy`, `scipy`, `scipy.stats`, `matplotlib`, `sklearn`
