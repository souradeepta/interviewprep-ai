#!/usr/bin/env python3
"""Replace placeholder Best Practices and Common Pitfalls content in ML interview prep markdown files."""

import re
import os

REPO = "/home/sbisw/github/interviewprep-ml"

CONTENT = {
    # ai/concepts/
    "06-linear-regression.md": {
        "practices": [
            "Always scale features before regression",
            "Use Ridge over OLS when features are correlated (multicollinearity)",
            "Plot residuals vs fitted values to detect non-linearity",
            "Check VIF (Variance Inflation Factor) to diagnose multicollinearity",
            "Use adjusted R² not R² for comparing models with different feature counts",
            "Consider log-transforming skewed targets",
            "Validate homoscedasticity with Breusch-Pagan test",
        ],
        "pitfalls": [
            "Including highly correlated features causes unstable coefficients (multicollinearity)",
            "Forgetting to scale features makes coefficient magnitudes incomparable",
            "Assuming linearity when the true relationship is non-linear (check residual plots)",
            "Evaluating on training data only — always hold out a test set",
        ],
    },
    "07-logistic-regression.md": {
        "practices": [
            "Scale features — logistic regression is sensitive to feature magnitude",
            "Use class_weight='balanced' for imbalanced datasets",
            "Tune regularization C with cross-validation (log scale: 0.001 to 100)",
            "Use predict_proba not predict for ranking/scoring tasks",
            "Check calibration curve — logistic regression is well-calibrated by default",
            "Monitor log-loss not just accuracy",
            "Use L1 regularization for feature selection",
        ],
        "pitfalls": [
            "Using accuracy on imbalanced classes hides poor performance on minority class",
            "Assuming predicted probabilities are perfectly calibrated without checking",
            "Forgetting to handle class imbalance (use class_weight or resample)",
            "Using default threshold 0.5 without considering business costs of FP vs FN",
        ],
    },
    "08-decision-trees.md": {
        "practices": [
            "Set max_depth (3-6) or min_samples_leaf to prevent overfitting",
            "Use feature importances to identify noisy features",
            "Prune trees post-training for interpretability",
            "Visualize with sklearn.tree.plot_tree or export_graphviz",
            "Use Gini for classification speed, entropy when you need information gain interpretation",
            "Always validate depth with cross-validation, not just training accuracy",
            "Use min_impurity_decrease to stop splits below a threshold",
        ],
        "pitfalls": [
            "Unpruned trees perfectly memorize training data (max depth=None means overfit)",
            "Biased toward high-cardinality features in splitting criteria",
            "Unstable — small data changes create very different trees",
            "Single trees are weak learners — use ensembles (RF, GBM) in production",
        ],
    },
    "09-random-forests.md": {
        "practices": [
            "Start with n_estimators=100-500 (more is rarely worse, just slower)",
            "Use oob_score=True for free validation without a holdout set",
            "Feature importances via feature_importances_ — but prefer permutation importance for correlated features",
            "Tune max_features first (sqrt for classification, log2 or 0.3 for regression)",
            "Use n_jobs=-1 for parallel training",
            "For imbalanced data use class_weight='balanced_subsample'",
            "Set random_state for reproducibility",
        ],
        "pitfalls": [
            "Feature importances are biased toward high-cardinality and correlated features",
            "Extrapolation is poor — trees can't predict beyond training range",
            "Memory-intensive for large n_estimators + deep trees",
            "Ignoring correlation structure in feature importance leads to misleading rankings",
        ],
    },
    "10-gradient-boosting.md": {
        "practices": [
            "Use small learning_rate (0.05-0.1) with more estimators for better generalization",
            "Set subsample=0.8 (stochastic GBM) to reduce overfitting and speed up",
            "Tune max_depth=3-6 first, then learning_rate",
            "Use early_stopping_rounds in XGBoost/LightGBM to find optimal n_estimators",
            "Use LightGBM for datasets >100k rows — much faster than sklearn GBM",
            "Monitor train vs validation loss curves to detect overfitting",
            "Use scale_pos_weight for imbalanced binary classification in XGBoost",
        ],
        "pitfalls": [
            "Learning rate too high causes poor generalization",
            "Not using early stopping wastes compute on too many estimators",
            "Over-tuning on validation set — use a separate final test set",
            "LightGBM default grows leaf-wise trees which need min_child_samples tuning to avoid overfit",
        ],
    },
    "11-support-vector-machines.md": {
        "practices": [
            "Always scale features — SVMs are distance-based and scale-sensitive",
            "Use RBF kernel as default; only use linear kernel for text/very high-dimensional data",
            "Tune C and gamma together on log-scale grid",
            "Use probability=True only when calibrated probabilities needed (it's slower)",
            "For n > 50k samples consider LinearSVC or SGDClassifier instead",
            "Use class_weight='balanced' for imbalanced data",
            "Cache kernel computations with cache_size=2000 for large datasets",
        ],
        "pitfalls": [
            "Forgetting to scale features completely breaks SVM performance",
            "Training O(n²) to O(n³) makes vanilla SVM impractical beyond ~50k samples",
            "Tuning C and gamma independently — they interact and need joint grid search",
            "Using probability=True adds Platt scaling overhead — avoid when not needed",
        ],
    },
    "12-k-nearest-neighbors.md": {
        "practices": [
            "Always normalize features before KNN — Euclidean distance is scale-sensitive",
            "Choose k with cross-validation (plot CV error vs k)",
            "Use KD-tree or Ball-tree for datasets <100k — much faster than brute force",
            "Compute distances in lower-dimensional space after PCA for high-dim data",
            "Weight neighbors by distance (weights='distance') for smoother boundaries",
            "Use leaf_size parameter to tune tree-build vs query speed trade-off",
            "Consider approximate nearest neighbors (FAISS, ANNOY) at large scale",
        ],
        "pitfalls": [
            "Slow at prediction time — O(n·d) per query with brute force",
            "Suffers from curse of dimensionality — distances become meaningless in high dimensions",
            "Sensitive to irrelevant features — feature selection helps",
            "No model to inspect — completely non-parametric, hard to interpret",
        ],
    },
    "13-neural-networks.md": {
        "practices": [
            "Use ReLU for hidden layers as default; LeakyReLU or ELU if dying ReLU is a problem",
            "Initialize with He init for ReLU, Xavier for tanh/sigmoid",
            "Always use batch normalization before or after non-linearities in deep networks",
            "Use dropout (0.2-0.5) in fully connected layers for regularization",
            "Start with Adam optimizer, switch to SGD+momentum for final fine-tuning",
            "Monitor gradient norms — exploding/vanishing signals architecture problems",
            "Use learning rate warmup for large networks",
        ],
        "pitfalls": [
            "Using sigmoid/tanh in deep networks — leads to vanishing gradients",
            "Not normalizing inputs causes slow or failed convergence",
            "Too large batch size reduces generalization (sharp minima)",
            "No validation monitoring — can't detect overfitting early",
        ],
    },
    "14-activation-functions.md": {
        "practices": [
            "Use ReLU as default for hidden layers in feedforward networks",
            "Use GELU for transformers and attention-based models (standard in BERT/GPT)",
            "Use sigmoid only for binary output layer probabilities",
            "Use softmax only for multiclass output layer",
            "Use tanh for RNNs where negative outputs matter",
            "Monitor dead neuron rate (fraction with zero gradient) when using ReLU",
            "Try Swish/Mish if ReLU is causing issues — often small accuracy gains",
        ],
        "pitfalls": [
            "Using sigmoid in hidden layers of deep networks — vanishing gradient kills learning",
            "Dead ReLU neurons (always outputting 0) caused by high learning rates or bad initialization",
            "Applying softmax in hidden layers instead of output — it squashes gradients",
            "Forgetting that activation choice affects initialization — must pair ReLU with He init",
        ],
    },
    "15-weight-initialization.md": {
        "practices": [
            "Use He initialization for ReLU networks; Xavier/Glorot for tanh/sigmoid",
            "Use PyTorch defaults (Kaiming He) — they are already correct for most architectures",
            "Never use zero initialization for weights — breaks symmetry and all neurons learn identically",
            "Small random noise initialization works for shallow networks but fails for deep ones",
            "Initialize biases to zero — this is fine",
            "For transformers, follow architecture-specific init (often truncated normal with std=0.02)",
        ],
        "pitfalls": [
            "Zero initialization: all neurons compute identical outputs and receive identical gradients — complete training failure",
            "Too large initial weights cause exploding activations and NaN loss",
            "Too small initial weights cause vanishing gradients in deep networks",
            "Mismatching initialization to activation (e.g., Xavier with ReLU) leads to suboptimal variance scaling",
        ],
    },
    "16-regularization.md": {
        "practices": [
            "Start with L2 (weight decay) — it's differentiable and works well with Adam",
            "Use dropout rate 0.1-0.3 for convolutional layers, 0.3-0.5 for dense layers",
            "Combine early stopping + L2 for best generalization",
            "Set weight_decay in optimizer (AdamW) rather than adding L2 manually",
            "Use data augmentation as implicit regularization for images and text",
            "Apply gradient clipping (max_norm=1.0) alongside regularization for RNNs",
            "Monitor train-val gap — large gap = underregularized, small gap but high loss = overregularized",
        ],
        "pitfalls": [
            "Adding L1 regularization to Adam breaks adaptive learning rates — use AdamW with L2 instead",
            "Dropout during inference without model.eval() adds noise to predictions",
            "Too high regularization causes underfitting — tune with cross-validation",
            "Applying dropout before batch norm breaks the normalization statistics",
        ],
    },
    "17-batch-normalization.md": {
        "practices": [
            "Apply BatchNorm before the activation function (or after — results are similar, try both)",
            "Use smaller learning rates without BN; with BN can use 10x higher LR",
            "Disable BN (model.eval()) during inference — uses running statistics not batch stats",
            "Use LayerNorm instead of BatchNorm for NLP/transformers and small batch sizes",
            "Use GroupNorm for object detection/segmentation with small batches",
            "Keep default momentum=0.1 — rarely needs tuning",
            "Don't use BN after dropout — the noise disrupts normalization",
        ],
        "pitfalls": [
            "Using BN with batch_size=1 — variance is undefined for single samples",
            "Forgetting model.eval() at inference — batch stats from test data leak",
            "BN + dropout ordering matters — wrong order hurts performance",
            "BN requires synchronization across GPUs in distributed training (use SyncBatchNorm)",
        ],
    },
    "18-k-means-clustering.md": {
        "practices": [
            "Always run k-means++ initialization (default in sklearn) — much better than random",
            "Run multiple restarts (n_init=10) and keep best inertia",
            "Scale features before clustering — Euclidean distance is scale-sensitive",
            "Use elbow method + silhouette score together to pick k",
            "For large datasets use MiniBatchKMeans — similar results, 10-100x faster",
            "Visualize clusters in 2D after PCA/t-SNE for sanity check",
            "Set random_state for reproducibility",
        ],
        "pitfalls": [
            "k-means assumes spherical, equal-size clusters — fails on elongated or irregular shapes",
            "Sensitive to outliers — one outlier can pull a centroid far from the cluster",
            "Number of clusters k must be specified — no automatic determination",
            "Results depend on initialization — always use k-means++ and multiple restarts",
        ],
    },
    "19-dimensionality-reduction.md": {
        "practices": [
            "Always scale features before PCA (StandardScaler)",
            "Use explained_variance_ratio_ to pick n_components (aim for 90-95% explained variance)",
            "Use PCA for preprocessing before ML models, t-SNE/UMAP only for visualization",
            "Set perplexity=30-50 for t-SNE on most datasets",
            "UMAP is faster than t-SNE and preserves more global structure — prefer it for large datasets",
            "Use PCA to remove noise before applying t-SNE (reduces compute)",
            "Set random_state for reproducibility of t-SNE/UMAP",
        ],
        "pitfalls": [
            "t-SNE is non-parametric — you can't transform new points, only fit_transform",
            "t-SNE distances between clusters are not meaningful — don't interpret cluster separation as distance",
            "PCA loses non-linear structure — use kernel PCA or autoencoders for non-linear reduction",
            "Using too many components defeats the purpose — check scree plot",
        ],
    },
    "20-gaussian-mixture-models.md": {
        "practices": [
            "Use BIC (lower is better) or AIC to select number of components — plot for k=1..15",
            "Always run multiple restarts (n_init=5-10) — GMM can converge to local optima",
            "Use covariance_type='full' for flexibility but 'diag' for speed on high-dim data",
            "Initialize GMM with k-means centroids for better convergence",
            "Validate with held-out log-likelihood, not just training BIC",
            "Use soft assignments (predict_proba) when downstream task benefits from uncertainty",
            "Add regularization_covar=1e-6 to prevent covariance matrices from becoming singular",
        ],
        "pitfalls": [
            "EM algorithm is not guaranteed to find global optimum — always use multiple restarts",
            "Too many components can overfit — use BIC/AIC to penalize complexity",
            "Full covariance matrix with high-dimensional data requires many samples to estimate reliably",
            "Doesn't handle heavy-tailed distributions well — consider t-mixture models",
        ],
    },
    "21-bias-variance-tradeoff.md": {
        "practices": [
            "Plot learning curves (train vs val vs training set size) to diagnose bias vs variance",
            "Use cross-validation to estimate true generalization error",
            "High bias → more complex model, more features, lower regularization",
            "High variance → more data, more regularization, simpler model, ensemble methods",
            "Use bootstrap resampling to empirically measure variance",
            "Don't rely on a single train/test split — variance across splits is informative",
            "Use ensemble methods (random forests, boosting) to reduce variance without increasing bias",
        ],
        "pitfalls": [
            "Treating bias and variance as independent — increasing model complexity reduces bias but increases variance simultaneously",
            "Overfitting the validation set through repeated hyperparameter tuning — use nested CV",
            "Assuming more data always helps — high bias models (underfitting) benefit little from more data",
            "Ignoring irreducible error — perfect fit is impossible with noisy labels",
        ],
    },
    "22-cross-validation.md": {
        "practices": [
            "Use StratifiedKFold for classification to preserve class ratios across folds",
            "For time-series data always use TimeSeriesSplit — never shuffle",
            "Use nested CV for unbiased hyperparameter tuning AND performance estimation",
            "Report mean ± std across folds, not just mean",
            "Use k=5 or k=10 as default — LOOCV only for very small datasets (<50 samples)",
            "Pipeline preprocessing inside CV to prevent data leakage",
            "Use cross_val_score with n_jobs=-1 for parallelism",
        ],
        "pitfalls": [
            "Fitting scaler/imputer on all data before CV — leaks statistics from test folds",
            "Using regular KFold on time-series data — future data leaks into training",
            "Picking the best fold's score instead of the mean — optimistic bias",
            "Treating the CV estimate as exact — it has variance, report confidence intervals",
        ],
    },
    "23-classification-metrics.md": {
        "practices": [
            "Never use accuracy alone on imbalanced datasets — use F1, ROC-AUC, or PR-AUC",
            "Use PR-AUC (average precision) for heavily imbalanced problems — more informative than ROC-AUC",
            "Report confusion matrix alongside scalar metrics",
            "Tune threshold based on business cost matrix (FP cost vs FN cost)",
            "Use macro-averaged F1 for multiclass when all classes equally important",
            "Use weighted F1 when class frequencies should influence the metric",
            "Monitor calibration (reliability diagram) when predicted probabilities matter",
        ],
        "pitfalls": [
            "Using accuracy on 99/1 imbalanced data — predicting all majority gets 99% accuracy",
            "ROC-AUC is optimistic with severe imbalance — use PR-AUC instead",
            "Default threshold 0.5 is rarely optimal — always evaluate threshold vs metric curves",
            "Reporting only training metrics — models memorize training data",
        ],
    },
    "24-regression-metrics.md": {
        "practices": [
            "Always plot residuals vs fitted values to check for patterns (non-linearity, heteroscedasticity)",
            "Use RMSE when large errors are especially bad (it penalizes them more)",
            "Use MAE when you want a robust metric less sensitive to outliers",
            "Use MAPE only when target values are always positive and far from zero",
            "Report multiple metrics — RMSE and MAE together reveal outlier influence",
            "Check residual distribution for normality (QQ plot) if confidence intervals are needed",
            "Use adjusted R² when comparing models with different numbers of features",
        ],
        "pitfalls": [
            "MAPE blows up when true values are near zero — use SMAPE or MAE instead",
            "High R² doesn't mean the model generalizes — check on held-out data",
            "Evaluating on training data only — always use cross-validation or a test set",
            "Assuming residuals are normally distributed without checking",
        ],
    },
    "25-feature-engineering.md": {
        "practices": [
            "Create interaction features for known domain relationships before trying automated methods",
            "Use log transformation for right-skewed features and targets",
            "Bin continuous features (age groups, income brackets) when non-linear boundaries are expected",
            "Use target encoding carefully — always apply within cross-validation folds to prevent leakage",
            "Compute feature importances first to identify what to engineer",
            "Apply polynomial features only up to degree 2 for most tabular tasks",
            "Document every feature transformation for reproducibility",
        ],
        "pitfalls": [
            "Target encoding without cross-validation causes severe data leakage",
            "Creating polynomial features on unscaled data creates numerically unstable large values",
            "Feature selection on the full dataset before CV leaks information",
            "Forgetting to apply the same transformations to test/inference data",
        ],
    },
    "26-hyperparameter-tuning.md": {
        "practices": [
            "Tune hyperparameters in order of importance: learning rate first, then capacity (depth, width), then regularization",
            "Use log-scale for learning rates and regularization strengths",
            "Start with RandomizedSearchCV (30-100 iterations) before refining with GridSearch",
            "Use HalvingRandomSearchCV or Optuna for large search spaces",
            "Always tune inside cross-validation, not on a single val split",
            "Set n_jobs=-1 for parallel search",
            "Document the search space — easy to forget why you chose specific ranges",
        ],
        "pitfalls": [
            "Tuning on the test set and reporting test performance as final — use a separate final eval",
            "Grid search exponentially expensive in high dimensions — use random or Bayesian instead",
            "Overfitting to the validation set through many tuning iterations — use separate test set",
            "Forgetting that hyperparameter sensitivity varies by dataset size",
        ],
    },
    "27-ensemble-methods.md": {
        "practices": [
            "Use soft voting (probability averaging) over hard voting when models are calibrated",
            "Stack with a simple meta-learner (logistic regression, ridge) — complex meta-learners overfit",
            "Use out-of-fold predictions to generate meta-features for stacking",
            "Ensure base models are diverse (different algorithms, feature subsets) — correlated models don't help",
            "Use cross_val_predict with cv=5 for stacking meta-features",
            "Blend models trained on different data subsets or preprocessing pipelines",
            "Monitor whether ensemble improves over best single model — overhead may not be worth it",
        ],
        "pitfalls": [
            "Blending base models trained on the full training set — meta-learner overfits to training predictions",
            "Stacking many similar models doesn't add diversity — only reduces variance marginally",
            "Ensemble overhead at inference — 10 models = 10x prediction cost",
            "Forgetting to calibrate probabilities before soft voting averages them",
        ],
    },
    "28-bayesian-inference.md": {
        "practices": [
            "Use conjugate priors (Beta-Binomial, Normal-Normal) for analytical posteriors when possible",
            "Encode prior beliefs with weak priors (high variance) unless domain knowledge is strong",
            "Use MCMC (PyMC, Stan) for non-conjugate models",
            "Check posterior predictive distributions — not just point estimates",
            "Compare models with WAIC or LOO cross-validation, not just likelihood",
            "Use half-Normal or Exponential priors for scale parameters (must be positive)",
            "Visualize prior vs posterior to verify data is updating beliefs as expected",
        ],
        "pitfalls": [
            "Using flat (improper) priors in MCMC can cause sampling problems — use weakly informative priors instead",
            "Ignoring prior sensitivity — results should be robust to reasonable prior changes",
            "Confusing MAP estimate with the full posterior — MAP loses uncertainty information",
            "MCMC convergence must be checked (R-hat < 1.01, trace plots) before trusting results",
        ],
    },
    # agentic-ai/concepts/
    "observability-for-agents.md": {
        "practices": [
            "Instrument every tool call with input/output/latency/error logging",
            "Use structured logging (JSON) for machine-parseable traces",
            "Attach trace IDs to every agent turn so you can reconstruct full execution chains",
            "Log token counts at each step to monitor cost growth",
            "Set up dashboards for P50/P95/P99 latency and error rate per tool",
            "Alert on error rate spikes > 5% or p95 latency doubling",
            "Use sampling (1-10%) for high-volume agents rather than logging everything",
        ],
        "pitfalls": [
            "Logging only final outputs — losing intermediate reasoning steps that explain failures",
            "Not correlating traces across agent turns — impossible to debug multi-step failures",
            "PII in logs — scrub user data before storing traces",
            "Over-logging everything at full volume — storage costs and query latency become prohibitive",
        ],
    },
    "retrieval-augmented-generation.md": {
        "practices": [
            "Chunk documents at semantic boundaries (paragraphs, sections), not arbitrary token counts",
            "Store both dense (embedding) and sparse (BM25) indexes for hybrid retrieval",
            "Re-rank top-k (20-50) results with a cross-encoder before passing top 3-5 to LLM",
            "Include document metadata (source, date, confidence) in retrieved context",
            "Evaluate retrieval quality separately from generation quality",
            "Use chunk overlap (10-15%) to avoid context loss at chunk boundaries",
            "Cache embeddings for static document sets",
        ],
        "pitfalls": [
            "Semantic search alone misses exact keyword matches — always use hybrid retrieval",
            "Passing too many retrieved chunks exceeds context window or dilutes relevant content",
            "Not filtering stale/outdated documents — retrieved context can be wrong even if relevant",
            "Embedding the query without preprocessing (lowercase, remove stopwords) reduces retrieval quality",
        ],
    },
    "memory-types.md": {
        "practices": [
            "Use episodic memory (conversation history) with a fixed window (last 10-20 turns) to prevent context overflow",
            "Store semantic memory externally (vector DB) and retrieve selectively rather than injecting everything",
            "Use procedural memory (tool definitions) as system prompt — it rarely changes",
            "Compress old episodic memory into summaries before dropping it",
            "Separate short-term (in-context) from long-term (external DB) memory architecturally",
            "Version semantic memory — stale facts cause hallucinations",
            "Profile memory retrieval latency — it's on the critical path",
        ],
        "pitfalls": [
            "Injecting all memory into every prompt — context grows unbounded and costs explode",
            "Not invalidating stale semantic memory — agents confidently state outdated facts",
            "Conflating episodic and semantic memory — different retrieval strategies needed",
            "Memory without forgetting — accumulating irrelevant noise degrades retrieval quality",
        ],
    },
    "reflection-and-self-improvement.md": {
        "practices": [
            "Use structured reflection prompts: 'What went wrong? What would you do differently?'",
            "Set a reflection budget (max 2-3 iterations) to prevent infinite loops",
            "Log reflection reasoning — helps debug why agents get stuck",
            "Separate critic and actor roles for better reflection quality",
            "Validate that reflection actually changes behavior — measure task success before and after",
            "Use Constitutional AI principles as reflection constraints",
            "Cache successful reflection patterns for reuse",
        ],
        "pitfalls": [
            "Reflection loops without convergence criteria run forever — always set max iterations",
            "Sycophantic reflection — LLM convinces itself everything is fine rather than critiquing",
            "Reflection on wrong metric — improving reasoning trace without improving task outcome",
            "Over-reflecting on simple tasks adds latency with no benefit",
        ],
    },
    "web-agents.md": {
        "practices": [
            "Use headless Playwright/Puppeteer over Selenium for reliability and speed",
            "Implement exponential backoff with jitter for rate limiting",
            "Cache page content aggressively — avoid re-fetching unchanged pages",
            "Use robots.txt compliance checks before scraping",
            "Extract structured data with CSS selectors over regex on HTML",
            "Handle JavaScript-rendered content with waitForSelector",
            "Set realistic User-Agent and request timing to avoid blocks",
        ],
        "pitfalls": [
            "Not handling pagination — only scraping page 1 of multi-page results",
            "Ignoring rate limits and getting IP-blocked",
            "Brittle selectors that break when sites update their HTML",
            "Not validating extracted data — silently returning empty or wrong content",
        ],
    },
    "knowledge-graphs.md": {
        "practices": [
            "Model entities and relationships explicitly — avoid storing facts as unstructured text in KG",
            "Use standardized ontologies (Schema.org, DBpedia) where possible",
            "Index both node properties and edge labels for fast traversal",
            "Use SPARQL or Cypher query languages for complex relationship queries",
            "Validate graph consistency (no dangling edges, correct types) on ingestion",
            "Combine KG with vector search for hybrid factual + semantic retrieval",
            "Version the knowledge graph schema separately from data",
        ],
        "pitfalls": [
            "Over-engineering the schema — start simple and evolve",
            "Not handling missing relationships — open-world assumption means absence ≠ false",
            "KG can become stale — build update pipelines from the start",
            "Circular relationships causing infinite traversal — always limit traversal depth",
        ],
    },
    "finance-agents.md": {
        "practices": [
            "Always validate numerical outputs against business rules (no negative prices, rate limits)",
            "Use deterministic computation for monetary arithmetic — avoid floating point",
            "Log all financial decisions with full reasoning chain for audit trails",
            "Implement hard limits on trade size and frequency",
            "Require human approval for actions above risk thresholds",
            "Backtest agent strategies on historical data before live deployment",
            "Use read-only mode for data fetching, separate authorized mode for transactions",
        ],
        "pitfalls": [
            "LLM hallucinating financial figures — always ground in real-time data feeds",
            "No kill switch for runaway trading agents",
            "Missing edge cases in market hours, holidays, circuit breakers",
            "Insufficient sandboxing — test agent connecting to production systems",
        ],
    },
    "mcts-for-agents.md": {
        "practices": [
            "Tune exploration constant C (UCB1 formula) for your reward scale — default √2 may be wrong",
            "Use domain-specific rollout policies rather than random rollouts for efficiency",
            "Cache previously visited states to avoid redundant computation",
            "Set simulation budget based on available latency — MCTS scales gracefully",
            "Use progressive widening for large branching factors",
            "Parallelize tree search across CPU cores",
            "Store the full search tree for debugging and post-hoc analysis",
        ],
        "pitfalls": [
            "Random rollouts give noisy value estimates — use learned value functions for better signal",
            "Tree grows too large in memory for long horizons — limit tree depth or prune",
            "Poor reward shaping leads MCTS to find unintended shortcuts",
            "Forgetting that MCTS finds optimal play, not human-like play — may be hard to explain",
        ],
    },
    "tree-of-thought.md": {
        "practices": [
            "Define evaluation criteria explicitly in the value function prompt — vague criteria give inconsistent scores",
            "Use beam search (k=3-5) over DFS to explore multiple promising paths",
            "Cache intermediate thoughts to avoid re-generating similar branches",
            "Set depth limit (3-5 levels) to prevent exponential compute growth",
            "Log the full tree for debugging — which branches were pruned and why",
            "Use ToT selectively — simple tasks don't benefit from the overhead",
            "Combine with MCTS value estimation for more principled search",
        ],
        "pitfalls": [
            "Exponential branching without pruning — cost grows as b^d",
            "Inconsistent evaluation scoring across different reasoning branches",
            "Over-splitting simple problems — not every task needs multi-path reasoning",
            "LLM value estimates are unreliable — validate with external feedback when possible",
        ],
    },
    "structured-output.md": {
        "practices": [
            "Use Pydantic v2 models for schema validation with automatic error messages",
            "Define strict schemas with Field validators for business rules (e.g., price > 0)",
            "Use Optional[T] for fields that may not always be present",
            "Implement retry logic when structured output parsing fails",
            "Log raw LLM output alongside parsed output for debugging",
            "Use json_mode (OpenAI) or response_format for reliable JSON",
            "Test schemas against edge cases before deploying",
        ],
        "pitfalls": [
            "Overly complex nested schemas cause more hallucination and parsing failures",
            "No fallback when schema validation fails — agent silently returns wrong data",
            "Forgetting that LLMs can violate schemas even with strict prompting",
            "Not versioning schemas — downstream consumers break when schemas change",
        ],
    },
    "hierarchical-agents.md": {
        "practices": [
            "Define clear interfaces between orchestrator and sub-agents — treat sub-agents as black boxes",
            "Use async dispatch for independent sub-agent tasks to reduce latency",
            "Implement timeout and fallback for each sub-agent call",
            "Log orchestrator decisions and which sub-agents were invoked",
            "Design sub-agents to be stateless and idempotent where possible",
            "Use capability registries to dynamically discover available sub-agents",
            "Test orchestrator logic independently from sub-agent implementations",
        ],
        "pitfalls": [
            "Tight coupling between orchestrator and sub-agents — hard to swap implementations",
            "No timeout on sub-agent calls — one slow agent blocks the entire pipeline",
            "Orchestrator doing too much reasoning — sub-agents should be specialized, not generic",
            "Circular delegation between agents — detect and break cycles",
        ],
    },
    "error-recovery.md": {
        "practices": [
            "Classify errors as retryable (transient) vs fatal before deciding recovery strategy",
            "Implement exponential backoff with jitter for transient errors",
            "Log full error context — input, state, error type — for debugging",
            "Use circuit breakers to prevent retry storms on persistently failing services",
            "Define fallback responses for each failure mode",
            "Test error recovery paths explicitly — don't assume they work",
            "Set maximum retry count to bound latency on persistent failures",
        ],
        "pitfalls": [
            "Retrying non-idempotent operations — double-charging, double-sending",
            "No maximum retry limit — agent retries forever on hard failures",
            "Swallowing errors silently — agent continues with bad state",
            "Retrying immediately without backoff — DDOS-ing the failing service",
        ],
    },
    "agent-cost-optimization.md": {
        "practices": [
            "Cache LLM responses for identical inputs using content-addressed hashing",
            "Use the smallest model that achieves required quality — benchmark before defaulting to large models",
            "Implement request batching for non-interactive workflows",
            "Monitor cost per task type and set budget alerts",
            "Use prompt compression (summarization) to reduce input token count",
            "Route simple tasks to cheaper models, complex to expensive",
            "Profile which agent steps consume the most tokens — usually the most optimizable",
        ],
        "pitfalls": [
            "Premature optimization before establishing baseline costs",
            "Caching responses that depend on time or external state — serving stale answers",
            "Using large models for simple classification tasks — overkill and expensive",
            "No cost visibility — costs grow silently until the bill arrives",
        ],
    },
    "tracing-agents.md": {
        "practices": [
            "Generate unique trace IDs at agent session start and propagate through all sub-calls",
            "Use OpenTelemetry standard for vendor-agnostic tracing",
            "Record span start/end times for every tool call and LLM inference",
            "Tag spans with model name, token counts, and tool names",
            "Export traces to a searchable backend (Jaeger, Honeycomb, Langfuse)",
            "Set sampling rate based on volume — 100% in dev, 1-10% in prod",
            "Alert on spans with latency > P99 baseline",
        ],
        "pitfalls": [
            "Tracing overhead adds latency — use async export, not synchronous",
            "Not propagating trace context across process boundaries",
            "Storing raw prompts in traces without PII scrubbing",
            "Trace IDs not correlated to user sessions — hard to debug user-reported issues",
        ],
    },
    "tool-calling.md": {
        "practices": [
            "Write clear, specific tool descriptions — the LLM uses them to decide when to call",
            "Use strict JSON schema for tool parameters with required fields specified",
            "Validate tool inputs before execution — LLMs sometimes generate out-of-range values",
            "Return structured error messages from tools, not exceptions",
            "Log every tool call with input/output for debugging",
            "Limit tools per prompt to 10-15 — too many options degrades selection quality",
            "Test tool descriptions independently by asking the LLM which tool to use in scenarios",
        ],
        "pitfalls": [
            "Vague tool descriptions cause wrong tool selection",
            "No input validation — tool receives malformed arguments and fails silently",
            "Returning raw exceptions to the LLM — it doesn't know how to recover",
            "Infinite tool call loops — always set max tool call depth",
        ],
    },
    "simulation-for-agents.md": {
        "practices": [
            "Use deterministic seeds for reproducible simulation runs",
            "Simulate failure modes (network timeout, API errors) explicitly, not just happy path",
            "Log every agent action and environment response for replay and analysis",
            "Calibrate simulation fidelity to match production distribution",
            "Run simulations in parallel for statistical significance",
            "Use simulation for regression testing after agent updates",
            "Define clear success/failure criteria before running simulations",
        ],
        "pitfalls": [
            "Simulation distribution shift — agents trained in simulation fail in production",
            "Over-optimizing for simulation metrics that don't correlate with real performance",
            "No randomization in simulation — agents overfit to deterministic scenarios",
            "Simulations too fast — missing timing-dependent agent behaviors",
        ],
    },
    "latency-optimization-agents.md": {
        "practices": [
            "Profile before optimizing — measure where latency actually comes from",
            "Parallelize independent tool calls and LLM requests with asyncio.gather",
            "Use streaming responses for user-facing agents — reduces perceived latency",
            "Cache embeddings, retrieved documents, and LLM responses where safe",
            "Use smaller, faster models for latency-sensitive steps with fallback to larger models",
            "Implement request timeouts and return partial results rather than failing",
            "Pre-warm model inference endpoints to eliminate cold-start latency",
        ],
        "pitfalls": [
            "Optimizing the wrong bottleneck — spend time on network when the LLM is the bottleneck",
            "Aggressive caching of responses that depend on dynamic state",
            "Parallelizing steps that have data dependencies — race conditions and wrong results",
            "Trading accuracy for latency without measuring the accuracy impact",
        ],
    },
    "skill-composition.md": {
        "practices": [
            "Define skills with clear input/output schemas and single responsibilities",
            "Use skill registries for dynamic discovery — don't hardcode skill lists",
            "Test each skill in isolation before testing compositions",
            "Version skills independently and use semantic versioning",
            "Document skill prerequisites and post-conditions",
            "Use interface contracts to allow skill implementations to be swapped",
            "Monitor which skills are used most — optimize them first",
        ],
        "pitfalls": [
            "Skills with side effects that aren't documented — unexpected state mutations",
            "No versioning — updating a shared skill breaks all callers",
            "Skills that are too coarse (doing too much) or too fine (requiring excessive composition)",
            "Circular skill dependencies that create deadlocks",
        ],
    },
    "agent-monitoring.md": {
        "practices": [
            "Monitor task completion rate, not just uptime",
            "Collect user feedback signals (thumbs up/down, edit rate) as quality metrics",
            "Set anomaly detection on token usage, cost, and error rate",
            "Track agent reasoning quality with LLM-as-judge scoring on sampled outputs",
            "Use RED metrics (Rate, Errors, Duration) for each agent endpoint",
            "Dashboard agent performance by task type, not just overall",
            "Alert on degradation within 5 minutes of deployment",
        ],
        "pitfalls": [
            "Monitoring infrastructure metrics (CPU, memory) but not agent quality metrics",
            "Alert fatigue from too many low-threshold alerts — tune thresholds carefully",
            "Not baselining before deployment — can't detect regression without baseline",
            "Monitoring only happy-path scenarios — missing tail-case failures",
        ],
    },
    "agent-evals.md": {
        "practices": [
            "Define evals before building the agent — let evaluation drive design",
            "Use a holdout eval set the agent has never seen",
            "Include adversarial examples (jailbreaks, edge cases) in eval suite",
            "Combine automated metrics with human evaluation for final quality assessment",
            "Run evals on every code change in CI/CD pipeline",
            "Use LLM-as-judge for open-ended evaluation with rubrics",
            "Track eval scores over time — detect regressions early",
        ],
        "pitfalls": [
            "Evaluating on training distribution only — agents generalize poorly to real user inputs",
            "Contaminating the eval set — using eval examples during agent development",
            "Single metric evals miss important failure modes",
            "Eval infrastructure not versioned — can't reproduce historical scores",
        ],
    },
    "safety-alignment.md": {
        "practices": [
            "Apply safety checks at multiple layers: input filtering, output filtering, and action validation",
            "Use constitutional AI principles as explicit constraints in system prompts",
            "Log all refused requests for safety audit and false positive analysis",
            "Test safety measures with red-teaming before deployment",
            "Implement human review queues for high-risk agent actions",
            "Use principle of least privilege for tool access",
            "Monitor safety override attempts and alert immediately",
        ],
        "pitfalls": [
            "Single-layer safety (output-only) — jailbreaks bypass output filters via indirect injection",
            "Safety measures not tested adversarially — fail against simple jailbreaks",
            "Over-refusal reduces utility — safety and helpfulness must be balanced",
            "Safety audits as one-time events — threat models evolve, so should safety measures",
        ],
    },
    "agent-debugging.md": {
        "practices": [
            "Log the full agent state at each step (prompt, tool calls, outputs) for replay",
            "Use deterministic seeds during debugging to reproduce non-deterministic failures",
            "Implement step-by-step execution mode for inspection",
            "Compare agent traces between working and failing runs side by side",
            "Use smaller, faster models for faster debug iteration cycles",
            "Isolate components — test each tool and LLM call independently",
            "Build a test harness that can replay production traces",
        ],
        "pitfalls": [
            "Debugging only the final output without examining intermediate steps",
            "Not logging enough — impossible to reconstruct agent state post-failure",
            "Using production models during debugging — slow and expensive iteration",
            "Assuming the LLM is wrong — often the bug is in tool implementation or schema",
        ],
    },
    "tool-use.md": {
        "practices": [
            "Validate all tool inputs against a schema before execution",
            "Return consistent error types and messages from all tools",
            "Keep tools stateless and idempotent where possible",
            "Throttle tool calls to prevent rate-limit violations",
            "Document tool behavior including side effects and latency expectations",
            "Test tools independently from the LLM that calls them",
            "Use timeout wrappers on all external tool calls",
        ],
        "pitfalls": [
            "LLM inventing tool parameters not in the schema — always validate inputs",
            "Tools with hidden side effects (modifying state) called speculatively",
            "No rate limiting — LLM can trigger thousands of tool calls in a loop",
            "Error messages that reference internal details — LLM may use them to probe the system",
        ],
    },
    "cooperative-agents.md": {
        "practices": [
            "Define explicit communication protocols between agents — don't rely on natural language for inter-agent calls",
            "Assign clear roles and responsibilities to prevent task duplication",
            "Use shared state stores (Redis, database) for coordination, not in-context passing",
            "Implement consensus mechanisms for decisions requiring agreement",
            "Monitor coordination overhead — communication can dominate task time",
            "Test agents in isolation then together — emergent failures only appear in cooperation",
            "Use task queues to decouple agents and handle load spikes",
        ],
        "pitfalls": [
            "Agents duplicating work without coordination leading to race conditions",
            "Circular wait between agents — deadlock detection required",
            "Over-communication — agents checking in too frequently drowns in noise",
            "No fallback when a cooperating agent fails — cascading failures",
        ],
    },
    "agent-testing.md": {
        "practices": [
            "Unit test tools in isolation from the LLM",
            "Integration test common agent workflows end-to-end",
            "Use deterministic mock LLMs for unit and regression tests",
            "Build an eval harness that runs automatically on PRs",
            "Include adversarial test cases — malformed inputs, edge cases, prompt injections",
            "Test failure modes explicitly — what happens when a tool times out",
            "Use property-based testing for input validation logic",
        ],
        "pitfalls": [
            "Only testing happy path — production has adversarial inputs",
            "Non-deterministic LLM responses make regression tests flaky — use temp=0 or mocks",
            "No tests for error recovery paths",
            "Testing in isolation from production environment — environment differences cause failures",
        ],
    },
    "competitive-agents.md": {
        "practices": [
            "Define game rules and reward functions explicitly and unambiguously",
            "Use self-play to generate diverse training opponents at all skill levels",
            "Implement Elo or TrueSkill rating systems to track agent performance over time",
            "Separate exploration (training) from exploitation (evaluation) environments",
            "Test agents against fixed, known-quality opponents for benchmarking",
            "Log full game transcripts for post-hoc analysis",
            "Use parallel game execution to increase training throughput",
        ],
        "pitfalls": [
            "Reward hacking — agents find loopholes that maximize reward without playing correctly",
            "Overfitting to a specific opponent strategy — test against diverse opponents",
            "Forgetting that self-play can diverge — monitor strategy diversity",
            "Evaluation against only current best agents — misses regressions on older strategies",
        ],
    },
    "context-window-management.md": {
        "practices": [
            "Summarize old conversation turns progressively — don't drop them abruptly",
            "Use token counting (tiktoken) before every LLM call to detect near-limit situations",
            "Store and retrieve conversation history from external DB rather than in-context",
            "Compress system prompts — they consume tokens in every call",
            "Prioritize recent and relevant context over older context",
            "Implement a context budget per agent turn and enforce it",
            "Use structured formats (JSON, bullet lists) which tokenize more efficiently than prose",
        ],
        "pitfalls": [
            "Silent truncation when context limit is hit — agent loses critical history",
            "No token counting — running out of context mid-conversation",
            "Injecting full conversation history into every turn — cost grows quadratically",
            "System prompt bloat — every added instruction consumes tokens forever",
        ],
    },
    "function-calling.md": {
        "practices": [
            "Use the model's native function calling API (OpenAI tools, Anthropic tool_use) — more reliable than prompt-based parsing",
            "Define required vs optional parameters in the schema",
            "Return function results in the same format the model expects",
            "Validate function call arguments before executing",
            "Use parallel function calling when tools are independent",
            "Cache function results for identical inputs within a session",
            "Log every function call with its arguments for debugging",
        ],
        "pitfalls": [
            "Ignoring the finish_reason — not detecting when the model wants to call a function",
            "No input validation — executing function calls with malformed arguments",
            "Deeply nested function schemas confuse the model — keep schemas flat",
            "Infinite function call recursion without depth limit",
        ],
    },
    "react-reasoning-acting.md": {
        "practices": [
            "Use explicit Thought/Action/Observation structure in prompts for clarity",
            "Limit observation length — long tool outputs overflow the context",
            "Set max steps (10-15) to prevent infinite loops",
            "Parse Thought content for reasoning traces — useful for debugging",
            "Use scratchpad reasoning in Thought before every Action",
            "Validate that Action format matches expected tool call schema",
            "Monitor Thought quality — degenerate reasoning patterns predict task failure",
        ],
        "pitfalls": [
            "No max step limit — agent loops forever on unsolvable tasks",
            "Mixing Thought and Action in a single output — breaks parsing",
            "Passing full raw tool output as Observation — truncate to relevant parts",
            "Allowing the model to fabricate Observations — always get real tool results",
        ],
    },
    "planning-reasoning.md": {
        "practices": [
            "Break plans into atomic, verifiable steps rather than vague goals",
            "Use plan validation before execution — check prerequisites are met",
            "Allow plan revision at each step based on observed outcomes",
            "Log the plan and each deviation for debugging",
            "Use hierarchical planning for complex tasks — high-level plan with low-level sub-plans",
            "Verify plan feasibility before committing resources",
            "Set plan expiry — stale plans based on outdated state cause failures",
        ],
        "pitfalls": [
            "Over-planning before execution — spending more time planning than acting",
            "No replanning when execution diverges from plan — agent proceeds with stale plan",
            "Plans too rigid — real-world tasks require adaptation",
            "Confusing planning (deciding what to do) with reasoning (deciding how to do it)",
        ],
    },
    "autonomous-agents.md": {
        "practices": [
            "Define explicit task boundaries and stop conditions before deployment",
            "Implement human-in-the-loop checkpoints for irreversible actions",
            "Log all autonomous decisions with full reasoning for audit",
            "Use principle of least privilege — only grant tools the agent actually needs",
            "Test in sandboxed environments before production deployment",
            "Set resource limits (API call budgets, time limits) to prevent runaway agents",
            "Monitor autonomous agents in real time, not just post-hoc",
        ],
        "pitfalls": [
            "No stop condition — agent runs indefinitely consuming resources",
            "Irreversible actions without confirmation — deleting data, sending emails, making purchases",
            "Insufficient sandboxing during testing — test agents accidentally affect production",
            "Over-trusting agent judgment — autonomous agents make mistakes that cascade",
        ],
    },
    # system-design/patterns/
    "model-versioning.md": {
        "practices": [
            "Tag every model artifact with a semantic version (major.minor.patch) tied to a git commit",
            "Store model metadata (training data hash, hyperparameters, metrics) alongside the artifact",
            "Use a model registry (MLflow, Weights & Biases) — don't just store files in S3 with dates",
            "Never overwrite model versions — treat them as immutable artifacts",
            "Automate promotion criteria (must pass eval thresholds before promotion to staging/prod)",
            "Keep model lineage — which dataset and code version produced this model",
            "Test model versions in shadow mode before promoting to production",
        ],
        "pitfalls": [
            "Storing models without metadata — can't reproduce or audit production models",
            "Overwriting model files in place — no rollback path",
            "Manual promotion process — inconsistent checks, human errors",
            "Not versioning preprocessing pipelines alongside the model — causes silent prediction errors",
        ],
    },
    "data-governance.md": {
        "practices": [
            "Classify all data assets by sensitivity (PII, confidential, public) before building pipelines",
            "Document data lineage from source through transformations to model features",
            "Implement access controls at the data layer, not just application layer",
            "Audit all data access for compliance — log who accessed what and when",
            "Define data retention policies and automate deletion",
            "Version datasets like code — immutable snapshots with metadata",
            "Conduct regular data quality audits and alert on schema drift",
        ],
        "pitfalls": [
            "PII in training data without proper anonymization — compliance risk",
            "No lineage tracking — can't identify which models are affected by a data issue",
            "Access controls only at application level — direct DB access bypasses them",
            "Dataset mutations without versioning — can't reproduce previous model training runs",
        ],
    },
    "ml-governance.md": {
        "practices": [
            "Document model cards for every production model (intended use, limitations, metrics by subgroup)",
            "Require approval gates before production deployment — human sign-off for high-risk models",
            "Log model predictions in production for monitoring and audit",
            "Track model performance by demographic subgroups to detect disparate impact",
            "Set model expiration dates — trigger retraining reviews periodically",
            "Maintain rollback procedures with tested SLA",
            "Version control all configuration files, not just model weights",
        ],
        "pitfalls": [
            "Deploying models without documented intended use and limitations",
            "No monitoring of production model behavior — model drift goes undetected",
            "Governance as checkbox compliance rather than actual risk management",
            "Missing subgroup performance analysis — aggregate metrics hide demographic disparities",
        ],
    },
    "shadow-mode.md": {
        "practices": [
            "Route 100% of production traffic to shadow model — don't subsample if data volume allows",
            "Compare distributions of predictions, not just aggregate metrics",
            "Set a fixed time window for shadow evaluation (1-2 weeks) before deciding",
            "Measure shadow model latency independently — it must meet production SLA",
            "Log shadow model outputs with the same schema as production",
            "Run shadow evaluation for full business cycles (weekdays + weekends, seasonality)",
            "Define promotion criteria upfront, not after seeing shadow results",
        ],
        "pitfalls": [
            "Shadow model affecting production by sharing state or side-effecting",
            "Evaluating shadow model on only a subset of traffic — misses rare but important cases",
            "No automatic comparison dashboard — shadow evaluation is manual and inconsistent",
            "Forgetting to test shadow model's resource consumption — may be 2x production cost",
        ],
    },
    "fairness-metrics.md": {
        "practices": [
            "Measure fairness across multiple metrics simultaneously — no single metric captures all notions of fairness",
            "Use disaggregated evaluation by subgroup before and after model training",
            "Apply fairness constraints during training when post-processing is insufficient",
            "Document which fairness definition was optimized and why",
            "Use Aequitas or Fairlearn libraries for systematic fairness auditing",
            "Monitor fairness metrics in production — distribution shift can re-introduce bias",
            "Involve domain experts and affected communities in defining fairness criteria",
        ],
        "pitfalls": [
            "Optimizing one fairness metric while violating another (impossibility theorems)",
            "Fairness evaluation only at training time — production data distribution shifts",
            "Proxy discrimination through correlated features even without protected attributes",
            "Treating fairness as a binary pass/fail rather than a continuous metric to track",
        ],
    },
    "privacy-preserving-ml.md": {
        "practices": [
            "Apply differential privacy with epsilon < 1.0 for high-sensitivity data",
            "Use federated learning when data cannot leave client devices",
            "Anonymize training data before use — pseudonymization is not anonymization",
            "Apply privacy budget tracking across multiple model versions",
            "Use secure aggregation in federated settings to prevent server from seeing individual updates",
            "Conduct privacy audits using membership inference attack benchmarks",
            "Encrypt model weights at rest and in transit for sensitive domains",
        ],
        "pitfalls": [
            "Assuming anonymization means privacy — re-identification attacks can de-anonymize",
            "No privacy budget management — epsilon accumulates across queries",
            "Federated learning without secure aggregation — individual updates reveal local data",
            "Differential privacy noise level set too low — privacy guarantees are not meaningful",
        ],
    },
    "interpretability.md": {
        "practices": [
            "Choose interpretability method matching model complexity — LIME for local, SHAP for global",
            "Always validate feature importances against domain knowledge — spurious correlations look important",
            "Use SHAP summary plots for global importance, waterfall plots for individual predictions",
            "Test explanations on adversarial examples to verify faithfulness",
            "Provide explanations in the user's language, not ML jargon",
            "Document which features are most influential for regulatory compliance",
            "Combine model-agnostic methods with model-specific interpretations",
        ],
        "pitfalls": [
            "LIME explanations are local approximations — inconsistent across nearby points",
            "SHAP values for correlated features split importance arbitrarily — interpret jointly",
            "Trusting feature importance without causal validation — correlation ≠ causation",
            "Using interpretability as post-hoc rationalization rather than genuine model understanding",
        ],
    },
    "reproducibility.md": {
        "practices": [
            "Pin all library versions in requirements.txt or conda environment file",
            "Set random seeds everywhere — Python, NumPy, PyTorch, TensorFlow",
            "Use deterministic algorithms when available (PyTorch: torch.use_deterministic_algorithms)",
            "Store training data snapshots as versioned artifacts, not live queries",
            "Record hardware specs — GPU type affects floating point results",
            "Containerize the training environment with Docker",
            "Automate full reproduction pipeline from data → trained model → metrics",
        ],
        "pitfalls": [
            "Non-deterministic data loading order without fixed seed",
            "Forgetting to pin transitive dependencies — library updates break reproducibility",
            "Recording only final metrics, not training config, data version, and code commit",
            "GPU non-determinism from cuDNN — must explicitly enable deterministic mode",
        ],
    },
    "disaster-recovery.md": {
        "practices": [
            "Define RPO (Recovery Point Objective) and RTO (Recovery Time Objective) before designing recovery",
            "Test recovery procedures quarterly — untested backups often fail",
            "Store backups in a different geographic region from primary",
            "Automate failover — manual failover during incidents is slow and error-prone",
            "Practice chaos engineering to validate recovery before disasters happen",
            "Document step-by-step recovery runbooks with owner assignments",
            "Keep shadow warm standby systems for critical ML inference endpoints",
        ],
        "pitfalls": [
            "Backups that are never tested — corrupt or incomplete backups discovered during actual disaster",
            "No RTO/RPO targets — recovery takes as long as it takes",
            "Single-region backups — regional outage takes out backups and primary",
            "Recovery procedures documented but not practiced — execution fails under pressure",
        ],
    },
    "inference-caching.md": {
        "practices": [
            "Use content-addressed caching (hash of input) for exact-match cache hits",
            "Implement semantic caching (embedding similarity) for near-duplicate queries",
            "Set TTL based on how frequently the underlying model or data changes",
            "Cache at the right granularity — response-level for full outputs, embedding-level for representations",
            "Monitor cache hit rate and latency reduction — validate caching is worth the complexity",
            "Implement cache warming for predictable request patterns",
            "Separate cache storage from model serving to scale independently",
        ],
        "pitfalls": [
            "Caching responses that depend on time or user context — serving stale personalized content",
            "Cache invalidation not triggered by model updates — serving old model outputs after redeployment",
            "No cache size limits — unbounded cache growth causes memory issues",
            "Caching raw LLM outputs without validating they're still valid after model version changes",
        ],
    },
    "feature-importance-tracking.md": {
        "practices": [
            "Track feature importances across model versions to detect distribution shift in inputs",
            "Alert when a previously unimportant feature becomes top-ranked — often indicates data issues",
            "Combine multiple importance methods (permutation, SHAP, gradient) for robustness",
            "Track feature importance stability across cross-validation folds",
            "Use importance tracking to guide feature engineering for the next model iteration",
            "Visualize importance trends over time, not just point-in-time values",
            "Separate importance from model version metadata",
        ],
        "pitfalls": [
            "Feature importance without context of training data — can't explain why importances changed",
            "Single-method importance is biased — permutation importance biased by correlated features",
            "Acting on importance without causal analysis — correlation-driven features can disappear with distribution shift",
            "Importance computed on training set — use validation set for unbiased importance",
        ],
    },
    "load-balancing.md": {
        "practices": [
            "Use health checks with warmup awareness — newly started model servers need time before accepting traffic",
            "Implement circuit breakers to stop routing to unhealthy backends",
            "Use weighted routing for gradual rollouts (canary deployments)",
            "Monitor P95/P99 latency per backend, not just average",
            "Use consistent hashing for stateful inference (session-pinned requests)",
            "Auto-scale backends based on queue depth, not just CPU — ML inference is memory-bound",
            "Test load balancer behavior during backend restarts",
        ],
        "pitfalls": [
            "Health checks that pass during warm-up but fail under load",
            "No circuit breakers — routing to failing backends until they recover",
            "Equal distribution to backends with different model sizes — slower backends get overwhelmed",
            "Missing sticky sessions for stateful agents — each request hits a different backend with no context",
        ],
    },
    "differential-privacy.md": {
        "practices": [
            "Start with epsilon=1.0 as a baseline and tighten based on privacy requirements",
            "Use RDP (Rényi Differential Privacy) accountant for tight composition bounds",
            "Clip gradients before adding noise — gradient norm must be bounded",
            "Apply DP at the model level, not the dataset level, for training",
            "Use DP-SGD (opacus library for PyTorch) for end-to-end differentially private training",
            "Publish epsilon values alongside model performance metrics",
            "Test that privacy guarantees hold with membership inference attack evaluation",
        ],
        "pitfalls": [
            "Setting epsilon too high (>10) — provides negligible privacy guarantee",
            "Not accounting for privacy budget across multiple training runs and queries",
            "Applying DP without gradient clipping — noise calibration is meaningless without bounded sensitivity",
            "Confusing differential privacy with anonymization — they are different guarantees",
        ],
    },
    "federated-learning.md": {
        "practices": [
            "Apply differential privacy at the client level to protect individual contributions",
            "Use secure aggregation to prevent server from seeing individual client updates",
            "Implement client selection strategies to handle heterogeneous data and compute",
            "Use FedAvg with momentum for faster convergence",
            "Communicate only model deltas, not full weights, to reduce bandwidth",
            "Monitor for Byzantine clients and implement robust aggregation",
            "Test with heterogeneous data (non-IID) distributions — federated data is never IID",
        ],
        "pitfalls": [
            "Assuming federated = private — model inversion attacks can still extract training data from gradients",
            "Non-IID data causes model drift — requires careful aggregation weighting",
            "Communication round bottleneck — too many rounds makes FL impractical",
            "Server becoming a single point of failure for federated training",
        ],
    },
    "model-registry.md": {
        "practices": [
            "Store model artifacts, metadata, and metrics together as a single versioned unit",
            "Implement promotion workflows: experiment → staging → production with approval gates",
            "Use tags for searchability (model type, task, dataset, team)",
            "Automate metric comparison before promotion — require improvement over current prod",
            "Integrate registry with CI/CD for automatic registration on successful training runs",
            "Store model cards and bias evaluation alongside model artifacts",
            "Keep registry access audited — log all reads, writes, and promotions",
        ],
        "pitfalls": [
            "Registry used as a file store only — no metadata, metrics, or lineage",
            "Manual promotion with no automated quality gates — inconsistent standards",
            "No rollback procedure linked to registry entries",
            "Multiple teams using the same registry with no namespace isolation",
        ],
    },
    "bias-detection.md": {
        "practices": [
            "Define protected attributes and fairness criteria before building models",
            "Measure bias on held-out test set, not training data",
            "Disaggregate metrics by subgroup combinations (intersectionality), not just single attributes",
            "Use statistical tests to determine if performance differences are significant",
            "Run bias evaluation as part of CI/CD pipeline — not one-time audits",
            "Monitor bias metrics in production — distribution shift causes bias to reappear",
            "Involve domain experts in interpreting bias metrics — not all disparities are problematic",
        ],
        "pitfalls": [
            "Testing for bias only on training data — doesn't reflect deployment distribution",
            "Evaluating single protected attributes — intersectional bias is often more severe",
            "Fixing aggregate bias without checking subgroup performance doesn't improve individual fairness",
            "Treating bias detection as one-time pre-launch activity — bias must be monitored continuously",
        ],
    },
    "model-explainability.md": {
        "practices": [
            "Choose explanation method based on stakeholder: SHAP values for data scientists, natural language for end users",
            "Validate explanations against domain knowledge — implausible explanations signal model issues",
            "Use local explanations (LIME, SHAP waterfall) for individual predictions, global for model behavior",
            "Generate explanations at inference time for production use cases requiring regulatory compliance",
            "Test explanation stability — similar inputs should yield similar explanations",
            "Include counterfactual explanations ('if X were different, outcome would change')",
            "Document explanation limitations alongside the explanations themselves",
        ],
        "pitfalls": [
            "Explanations that are not faithful to the model's actual computation — post-hoc rationalization",
            "Overconfident explanations — SHAP and LIME approximate, they don't perfectly explain",
            "Explanations for the model, not the data — model explanations can still reflect spurious correlations",
            "Treating explainability as a technical output rather than a communication tool for the audience",
        ],
    },
}

# Map filename slugs to directory paths
FILE_PATHS = {}

# ai/concepts/ files
ai_files = [
    "06-linear-regression.md", "07-logistic-regression.md", "08-decision-trees.md",
    "09-random-forests.md", "10-gradient-boosting.md", "11-support-vector-machines.md",
    "12-k-nearest-neighbors.md", "13-neural-networks.md", "14-activation-functions.md",
    "15-weight-initialization.md", "16-regularization.md", "17-batch-normalization.md",
    "18-k-means-clustering.md", "19-dimensionality-reduction.md", "20-gaussian-mixture-models.md",
    "21-bias-variance-tradeoff.md", "22-cross-validation.md", "23-classification-metrics.md",
    "24-regression-metrics.md", "25-feature-engineering.md", "26-hyperparameter-tuning.md",
    "27-ensemble-methods.md", "28-bayesian-inference.md",
]
for f in ai_files:
    FILE_PATHS[f] = os.path.join(REPO, "ai", "concepts", f)

# agentic-ai/concepts/ files
agentic_files = [
    "observability-for-agents.md", "retrieval-augmented-generation.md", "memory-types.md",
    "reflection-and-self-improvement.md", "web-agents.md", "knowledge-graphs.md",
    "finance-agents.md", "mcts-for-agents.md", "tree-of-thought.md", "structured-output.md",
    "hierarchical-agents.md", "error-recovery.md", "agent-cost-optimization.md",
    "tracing-agents.md", "tool-calling.md", "simulation-for-agents.md",
    "latency-optimization-agents.md", "skill-composition.md", "agent-monitoring.md",
    "agent-evals.md", "safety-alignment.md", "agent-debugging.md", "tool-use.md",
    "cooperative-agents.md", "agent-testing.md", "competitive-agents.md",
    "context-window-management.md", "function-calling.md", "react-reasoning-acting.md",
    "planning-reasoning.md", "autonomous-agents.md",
]
for f in agentic_files:
    FILE_PATHS[f] = os.path.join(REPO, "agentic-ai", "concepts", f)

# system-design/patterns/ files
sysdesign_files = [
    "model-versioning.md", "data-governance.md", "ml-governance.md", "shadow-mode.md",
    "fairness-metrics.md", "privacy-preserving-ml.md", "interpretability.md",
    "reproducibility.md", "disaster-recovery.md", "inference-caching.md",
    "feature-importance-tracking.md", "load-balancing.md", "differential-privacy.md",
    "federated-learning.md", "model-registry.md", "bias-detection.md",
    "model-explainability.md",
]
for f in sysdesign_files:
    FILE_PATHS[f] = os.path.join(REPO, "system-design", "patterns", f)


def build_bullet_block(items):
    """Build a markdown bullet list string from a list of strings."""
    return "\n".join(f"- {item}" for item in items)


def replace_placeholder_block(text, placeholder_pattern, real_items):
    """
    Replace a consecutive block of placeholder lines matching
    `- Practice N` or `- Pitfall N` with real_items bullet list.
    The pattern matches 1 or more consecutive placeholder lines.
    """
    # Build regex that matches one or more consecutive placeholder lines
    # e.g. "- Practice 1\n- Practice 2\n- Practice 3\n" or similar
    pattern = re.compile(
        r'((?:' + re.escape(placeholder_pattern) + r'\s*\d+[^\n]*\n?)+)',
        re.MULTILINE
    )
    replacement = build_bullet_block(real_items) + "\n"
    new_text, count = pattern.subn(replacement, text)
    return new_text, count


def fix_file(filename, filepath, content_spec):
    if not os.path.exists(filepath):
        print(f"  SKIP (not found): {filepath}")
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    original = text
    total_replacements = 0

    if "practices" in content_spec:
        text, count = replace_placeholder_block(text, "- Practice", content_spec["practices"])
        total_replacements += count

    if "pitfalls" in content_spec:
        text, count = replace_placeholder_block(text, "- Pitfall", content_spec["pitfalls"])
        total_replacements += count

    if text == original:
        print(f"  NO CHANGE (no placeholders found): {filename}")
        return False

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"  OK {filename} ({total_replacements} block(s) replaced)")
    return True


def main():
    fixed = 0
    skipped = 0
    for filename, content_spec in CONTENT.items():
        if filename not in FILE_PATHS:
            print(f"  ERROR: No path mapping for {filename}")
            skipped += 1
            continue
        filepath = FILE_PATHS[filename]
        result = fix_file(filename, filepath, content_spec)
        if result:
            fixed += 1
        else:
            skipped += 1

    print(f"\nDone: {fixed} files fixed, {skipped} files skipped/unchanged.")


if __name__ == "__main__":
    main()
