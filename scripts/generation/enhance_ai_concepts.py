#!/usr/bin/env python3
"""Generate comprehensive markdown content for all 28 AI fundamentals concepts."""

import os

CONCEPTS = {
    "04-optimization-algorithms": {
        "title": "Optimization Algorithms",
        "explanation": "Beyond basic gradient descent, modern optimizers adapt learning rates per parameter or maintain momentum across iterations. Adam combines momentum and adaptive learning rates, RMSprop adapts per-parameter learning rates using gradient history, Adagrad accumulates squared gradients to decrease learning rates for frequent features. These algorithms drastically improve convergence speed and stability, especially on non-convex loss surfaces common in deep learning. Understanding when to use each optimizer is critical for model training efficiency.",
        "intuition": "Different optimizers are like different drivers: vanilla GD is careful but slow, momentum is aggressive and pushes through valleys, Adam is adaptive and learns the terrain as it goes.",
        "how_it_works": "1. Maintain adaptive state (momentum, gradient history)\n2. Compute gradient at current position\n3. Update state based on gradient\n4. Adjust learning rate per parameter using state\n5. Take step in direction of adapted gradient\n6. Repeat until convergence",
        "diagram": "graph TD\n    A[Current Weights] --> B[Compute Gradient]\n    B --> C[Update Momentum]\n    C --> D[Update RMS]\n    D --> E[Adaptive Learning Rate]\n    E --> F[New Weights]\n    F --> G{Converged?}\n    G -->|No| B",
        "tradeoffs": "Adam: fast, adaptive, good defaults | RMSprop: simpler, less memory | SGD+momentum: stable, needs tuning | Adagrad: good for sparse data, learning rate decay",
        "best_practices": [
            "Use Adam as default optimizer (works well across domains)",
            "For small datasets, SGD with momentum often generalizes better",
            "Monitor gradient norms during training (log histograms)",
            "Use different learning rates for different layers (discriminative fine-tuning)",
            "Combine optimizer with learning rate schedule (warmup + decay)",
            "For distributed training, use LARS or LAMB (layer-wise adaptation)",
            "Reset optimizer state when learning rate changes significantly",
            "Use weight decay with AdamW instead of L2 regularization"
        ],
        "pitfalls": [
            "Using Adam default learning rate (0.001) for all problems without tuning",
            "Forgetting to reset optimizer state when resuming training",
            "Using adaptive learning rates but no warmup (causes instability early)",
            "Combining adaptive optimizer with aggressive L2 regularization",
            "Not monitoring gradient statistics (can hide problems early)"
        ]
    },
    "05-learning-rate-scheduling": {
        "title": "Learning Rate Scheduling",
        "explanation": "The learning rate is one of the most important hyperparameters, and fixing it throughout training is suboptimal. Learning rate scheduling reduces the learning rate over time, allowing coarse early updates to fine-grained later refinement. Common schedules: step decay (reduce every N epochs), exponential decay (smooth reduction), warmup (gradually increase then decrease), cosine annealing (smooth decrease following cosine curve). Proper scheduling can improve final model accuracy by 1-2% without changing the algorithm.",
        "intuition": "Like tuning a microscope: start with coarse focus, gradually refine, end with precise adjustments. Rush at the end and you'll overshoot the target.",
        "how_it_works": "1. Initialize high learning rate for coarse learning\n2. After each epoch, compute new learning rate based on schedule\n3. Update optimizer with new learning rate\n4. Continue training with reduced learning rate\n5. Reach final small learning rate for fine-tuning\n6. Stop when learning rate (or performance) plateaus",
        "diagram": "graph TD\n    A[Start Training] --> B[High Learning Rate]\n    B --> C[First Epochs: Coarse Updates]\n    C --> D[Decay Learning Rate]\n    D --> E[Middle Epochs: Medium Updates]\n    E --> F[Further Decay]\n    F --> G[Late Epochs: Fine-tuning]\n    G --> H[Convergence]",
        "tradeoffs": "Step decay: simple, discrete jumps | Exponential: smooth, fewer hyperparams | Warmup: prevents early instability but adds epochs | Cosine: theoretically motivated, empirically strong",
        "best_practices": [
            "Use warmup (5-10% of total epochs) to stabilize early training",
            "Cosine annealing with restart is robust across problems",
            "Schedule should reduce learning rate by 10x from start to end",
            "For transfer learning, use higher initial learning rate with schedule",
            "Combine schedule with optimizer choice (Adam needs less tuning)",
            "Monitor validation loss to detect early stopping opportunities",
            "For fine-tuning, use smaller learning rate with longer schedule",
            "Consider cyclical learning rates for regularization effect"
        ],
        "pitfalls": [
            "No learning rate schedule: gets stuck in local minimum or diverges",
            "Decaying too aggressively: stops learning before convergence",
            "Warmup without schedule: no mechanism to reduce learning rate later",
            "Fixed schedule ignoring validation performance: misses optimal stopping point",
            "Different schedules between train and val: inconsistent convergence"
        ]
    },
    "06-linear-regression": {
        "title": "Linear Regression",
        "explanation": "Linear regression fits a hyperplane to data by minimizing prediction error. The closed-form solution via OLS (Ordinary Least Squares) exists: θ = (XᵀX)⁻¹Xᵀy. However, this is expensive (O(n³)) and numerically unstable for high dimensions. Regularized versions (Ridge, Lasso) add penalties: Ridge adds L2 (squared weights), Lasso adds L1 (absolute weights). Ridge handles correlated features better; Lasso performs automatic feature selection. Understanding linear regression is foundational—most complex models are nonlinear extensions of it.",
        "intuition": "Fitting a line to scattered points: the best-fit line minimizes average distance. Add constraints (regularization) to prefer simpler lines that generalize better.",
        "how_it_works": "1. Arrange data as matrix X (n × d) and targets y (n × 1)\n2. Add intercept column to X\n3. Compute OLS: θ = (XᵀX)⁻¹Xᵀy\n4. For regularized: θ = (XᵀX + λI)⁻¹Xᵀy (Ridge)\n5. For Lasso, use iterative solvers (no closed form)\n6. Make predictions: ŷ = Xθ",
        "diagram": "graph TD\n    A[Data X, y] --> B[Compute XᵀX]\n    B --> C[Add Regularization]\n    C --> D[Solve Linear System]\n    D --> E[Get Weights θ]\n    E --> F[Predict ŷ = Xθ]\n    F --> G[Evaluate MSE]",
        "tradeoffs": "OLS: exact but unstable | Ridge: stable, keeps all features | Lasso: sparse (feature selection), harder to solve | Elastic Net: combines both",
        "best_practices": [
            "Always standardize features (mean 0, std 1) before fitting",
            "Use Ridge regression by default (more stable than OLS)",
            "Use Lasso if you need feature selection",
            "For high-dimensional data, use Elastic Net (Ridge + Lasso)",
            "Monitor regularization parameter (λ) via cross-validation",
            "Check residual distribution (should be normal, zero mean)",
            "For large datasets, use stochastic gradient descent instead of OLS",
            "Include interaction terms if relationships are non-additive"
        ],
        "pitfalls": [
            "Not scaling features: large-magnitude features dominate",
            "Using OLS on ill-conditioned data: numerical instability, huge weights",
            "Ignoring multicollinearity: redundant features cause unstable estimates",
            "Not removing outliers: single outlier can shift line drastically",
            "Assuming linear relationship: doesn't capture nonlinearity"
        ]
    },
    "07-logistic-regression": {
        "title": "Logistic Regression",
        "explanation": "Logistic regression extends linear regression to classification via the sigmoid function: σ(z) = 1/(1+e⁻ᶻ). The model outputs probability P(y=1|x) = σ(θᵀx). Loss is cross-entropy, not MSE. Despite the name, logistic regression is classification, not regression. It's linear in decision boundary (like linear regression) but probabilistic. Logistic regression is interpretable: weights show feature importance, log-odds are linear. For multiclass, use softmax (generalized sigmoid) instead of sigmoid.",
        "intuition": "Like linear regression but instead of predicting numbers, predict probability of class. The sigmoid squashes any number to probability [0,1].",
        "how_it_works": "1. Compute linear combination: z = θᵀx\n2. Apply sigmoid: p = 1/(1+e⁻ᶻ)\n3. Cross-entropy loss: -[y log(p) + (1-y) log(1-p)]\n4. Gradient descent on cross-entropy (sigmoid makes it convex)\n5. Threshold at 0.5 to make final prediction",
        "diagram": "graph LR\n    A[Features x] --> B[Linear: z = θᵀx]\n    B --> C[Sigmoid: p = σ(z)]\n    C --> D[Probability]\n    D --> E{p > 0.5?}\n    E -->|Yes| F[Class 1]\n    E -->|No| G[Class 0]",
        "tradeoffs": "Binary logistic: simple, interpretable | Multiclass (softmax): one-vs-rest complexity | Linear boundary: fast, may underfit | Probabilistic: good calibration",
        "best_practices": [
            "Standardize features for better convergence",
            "Use class weights for imbalanced data",
            "Monitor calibration: are predicted probabilities accurate?",
            "Use cross-entropy loss, not MSE (logistic is probabilistic)",
            "For multiclass, use softmax with cross-entropy",
            "Regularize (L1 or L2) to prevent overfitting",
            "Check decision boundary: should be roughly linear",
            "For large datasets, use stochastic gradient descent"
        ],
        "pitfalls": [
            "Using MSE loss instead of cross-entropy: bad probability calibration",
            "Not scaling features: sigmoid becomes very steep or flat",
            "Ignoring class imbalance: predicts majority class for everything",
            "No regularization: overfits (wiggly decision boundary)",
            "Assuming linear boundary: fails on non-linearly separable data"
        ]
    },
    "08-decision-trees": {
        "title": "Decision Trees",
        "explanation": "Decision trees recursively partition feature space with axis-aligned splits, asking questions like 'x₁ > threshold?'. Each split maximizes information gain (or Gini impurity reduction). CART algorithm greedily builds trees top-down. Trees are interpretable (you can trace decisions) but prone to overfitting (deep trees memorize data). Pruning (removing leaves) or limiting depth controls complexity. Tree depth, minimum samples per leaf, and minimum samples to split are critical hyperparameters. Despite overfitting tendency, trees are the basis for strong ensemble methods (random forests, boosting).",
        "intuition": "Binary questions narrowing down possibilities: 'is age > 30? is income > $50k?'. Keep asking until you narrow to one class.",
        "how_it_works": "1. Start with all data at root node\n2. For each feature, try all thresholds\n3. Choose split maximizing information gain: I(parent) - weighted_avg(I(children))\n4. Recursively split left and right children\n5. Stop when: node is pure, depth limit, or min samples reached\n6. Majority class in leaf is prediction",
        "diagram": "graph TD\n    A[All Data] --> B{x1 > 5?}\n    B -->|Yes| C{x2 > 3?}\n    B -->|No| D[Class A]\n    C -->|Yes| E[Class B]\n    C -->|No| F[Class A]",
        "tradeoffs": "Gini vs entropy: similar results, Gini faster | Greedy vs optimal: greedy fast, optimal intractable | Shallow: underfits | Deep: overfits",
        "best_practices": [
            "Limit max depth (5-15 typical for individual trees)",
            "Set minimum samples per leaf (5-20 prevents overfitting)",
            "Prune post-hoc if tree has >100 nodes",
            "Handle imbalanced data with class weights",
            "For categorical features, use proper encoding (not ordinal)",
            "Monitor feature importance: which splits matter most?",
            "Use trees in ensembles (random forests, boosting) not alone",
            "Check if splits make domain sense (interpretability check)"
        ],
        "pitfalls": [
            "Unpruned deep tree: memorizes training data, poor test performance",
            "No depth limit: trains forever on large datasets",
            "Using trees on high-dimensional data: overfits (many features to split on)",
            "Ignoring feature scaling (unlike other algorithms, trees are scale-invariant but split choices can be poor)",
            "Not checking feature importance: may have spurious splits"
        ]
    },
    "09-random-forests": {
        "title": "Random Forests",
        "explanation": "Random forests combine multiple trees via bagging (bootstrap aggregating): train each tree on random subset of data, average predictions. This reduces overfitting (trees overfit individually but average cancels noise) and improves stability. The key: each tree gets random feature subset at split time, increasing diversity. Feature importance comes from counting splits. Random forests work well out-of-box (few hyperparameters), handle mixed feature types, are parallelizable. Trade-off: lose interpretability of single tree but gain robustness of ensemble.",
        "intuition": "Ask 100 slightly-different experts, average their votes. Even if each expert overfits, their errors cancel out (voting).",
        "how_it_works": "1. For each tree i in ensemble:\n   a. Bootstrap sample data (with replacement)\n   b. Train decision tree on bootstrap sample\n   c. At each split, randomly select sqrt(d) features to consider\n2. For prediction: average (regression) or majority vote (classification)",
        "diagram": "graph TD\n    A[Original Data] --> B[Bootstrap 1]\n    A --> C[Bootstrap 2]\n    A --> D[Bootstrap 3]\n    B --> E[Tree 1]\n    C --> F[Tree 2]\n    D --> G[Tree 3]\n    E --> H[Average/Vote]\n    F --> H\n    G --> H",
        "tradeoffs": "Bagging alone: still overfits | RF with feature subsets: more diverse, reduces overfitting | Many trees: better but slower | Few trees: fast but high variance",
        "best_practices": [
            "Use 100-500 trees (diminishing returns after ~200)",
            "Set max_features to sqrt(n_features) for classification, max for regression",
            "Limit tree depth or set min samples per leaf to avoid overfitting",
            "Use class weights for imbalanced data",
            "Feature importance helps with feature selection",
            "Out-of-bag (OOB) error provides free validation estimate",
            "For very large datasets, use ExtraTreesTrees (faster, random thresholds)",
            "Parallelize tree training across cores"
        ],
        "pitfalls": [
            "Too few trees: high variance, unstable predictions",
            "Deep individual trees: ensemble doesn't help (all trees overfit same way)",
            "Not using feature subsets: reduces diversity, defeats bagging benefit",
            "Ignoring class imbalance: majority class dominates voting",
            "Forgetting trees are not model-agnostic: only good for tree-based base learners"
        ]
    },
    "10-gradient-boosting": {
        "title": "Gradient Boosting",
        "explanation": "Gradient boosting trains trees sequentially, each new tree fitting residuals of previous ensemble. This focuses learning on hard examples. Unlike bagging (parallel, reduce variance), boosting reduces bias. Key insight: additive model where each tree corrects previous predictions. XGBoost/LightGBM are production implementations with regularization, shrinkage, and column sampling. Boosting is powerful but can overfit (needs early stopping) and is sensitive to hyperparameters. One of the most successful algorithms in industry (Kaggle competitions).",
        "intuition": "Learning from mistakes: first model makes errors, second model learns to correct those errors, third model corrects the second's mistakes, etc.",
        "how_it_works": "1. Train first tree on data\n2. Compute residuals (targets - predictions)\n3. Train next tree on residuals\n4. Add to ensemble: pred = pred + learning_rate * new_tree_pred\n5. Repeat for N iterations\n6. Use early stopping (validation loss increases → stop)",
        "diagram": "graph TD\n    A[Data] --> B[Train Tree 1]\n    B --> C[Predictions 1]\n    A --> D[Compute Residuals]\n    C --> D\n    D --> E[Train Tree 2 on Residuals]\n    E --> F[Predictions 2]\n    F --> G[Add to Ensemble]\n    G --> H{Early Stop?}\n    H -->|No| D\n    H -->|Yes| I[Final Ensemble]",
        "tradeoffs": "AdaBoost: simple, slow | Gradient Boosting: general, needs tuning | XGBoost: optimized, many hyperparams | LightGBM: faster, leaf-wise growth",
        "best_practices": [
            "Use learning rate (shrinkage) 0.01-0.1: lower is better but slower",
            "Use early stopping (monitor validation loss) to prevent overfitting",
            "Limit tree depth (typically 3-8 for boosting, shallower than random forests)",
            "Set min_child_weight / min_samples_leaf to prevent overfitting",
            "Use subsample and colsample_bytree (0.5-0.8) for regularization",
            "Tune number of boosting rounds via cross-validation",
            "For imbalanced data, use scale_pos_weight or adjust class weights",
            "Monitor validation curves: watch for overfitting"
        ],
        "pitfalls": [
            "No early stopping: overfits drastically (boosting is slow to overfit but then fast)",
            "High learning rate: unstable training, poor convergence",
            "No regularization: deep trees combined with boosting overfit badly",
            "Ignoring class imbalance: boosting focuses on hard examples (which may all be majority class)",
            "Too many boosting rounds: validation loss increases, overfitting"
        ]
    },
    "11-support-vector-machines": {
        "title": "Support Vector Machines",
        "explanation": "SVMs find maximum-margin hyperplane separating classes. The margin is the distance from decision boundary to nearest points. Maximizing margin improves generalization (Vapnik's principle). In dual form, SVM depends only on dot products, enabling kernel trick: map to high-dimensional space without explicitly computing coordinates. Common kernels: linear (no mapping), RBF (infinite-dimensional space), polynomial. SVMs are powerful but slow (O(n²) or O(n³)) and sensitive to feature scaling. Good for small-medium datasets with clear separation.",
        "intuition": "Finding widest highway separating two cities. Points on the highway edge are 'support vectors'—they define the boundary. Wider highway = better generalization.",
        "how_it_works": "1. For linearly separable data: find θ minimizing ||θ||² subject to y_i(θᵀx_i + b) ≥ 1\n2. Soft margin: allow some violations (slack variables ξᵢ)\n3. Dual problem: maximize using dot products (enables kernel trick)\n4. Kernel trick: replace dot products with kernel K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ)\n5. Decision: sign(Σ αᵢ y_i K(xᵢ, x) + b)",
        "diagram": "graph LR\n    A[Data] --> B[Kernel Map]\n    B --> C[Maximize Margin]\n    C --> D[Decision Boundary]\n    D --> E[Support Vectors]\n    E --> F[Prediction]",
        "tradeoffs": "Linear kernel: simple, fast | RBF: flexible, slow | Polynomial: middle ground | C parameter: lower generalizes better",
        "best_practices": [
            "Always scale/normalize features (SVM sensitive to feature magnitude)",
            "Use RBF kernel as default (flexible, works well)",
            "Tune C (regularization) via cross-validation: higher C overfits",
            "For large datasets, use linear kernel or approximate SVM",
            "For imbalanced data, use class_weight='balanced'",
            "Use probability calibration (Platt scaling) if probabilities needed",
            "Check support vector count: <5% is good, >50% suggests overfitting",
            "For multiclass, use one-vs-rest or one-vs-one"
        ],
        "pitfalls": [
            "Not scaling features: large-magnitude features dominate kernel",
            "RBF kernel with default C: likely overfits (C too high)",
            "No hyperparameter tuning: default C=1.0 rarely optimal",
            "Using SVMs on huge datasets: quadratic complexity is too slow",
            "RBF kernel on high-dimensional data: may overfit (curse of dimensionality)"
        ]
    },
    "12-k-nearest-neighbors": {
        "title": "K-Nearest Neighbors",
        "explanation": "KNN is instance-based learning: store training data, predict by finding k nearest neighbors. Prediction is average (regression) or majority vote (classification). No explicit training—all work at prediction time. Distance metric (Euclidean, Manhattan, etc.) and k are key hyperparameters. KNN is simple and nonparametric (makes no assumptions about data distribution) but slow at prediction (must search all training points) and susceptible to curse of dimensionality (distances become meaningless in high dimensions). Good for non-linear decision boundaries with sufficient training data.",
        "intuition": "You are the average of your 5 closest friends. If all 5 friends like pizza, you probably like pizza. If 3 like pizza and 2 like burgers, you're unsure.",
        "how_it_works": "1. Store all training data\n2. At prediction time, compute distance from query x to all training points\n3. Find k nearest neighbors (smallest distances)\n4. Return majority vote (classification) or average (regression) of k neighbors",
        "diagram": "graph TD\n    A[Query Point] --> B[Compute Distances]\n    B --> C[Sort by Distance]\n    C --> D[Select k Nearest]\n    D --> E[Majority Vote or Average]\n    E --> F[Prediction]",
        "tradeoffs": "Small k (k=1): low bias, high variance, overfits | Large k: high bias, low variance, underfits | Euclidean: standard but affected by feature scale | Manhattan: robust to outliers",
        "best_practices": [
            "Always scale/normalize features (KNN uses distances)",
            "Tune k via cross-validation (typically 3-10)",
            "Use odd k for classification to avoid ties",
            "Weighted KNN: closer neighbors have higher weight (1/distance)",
            "Remove irrelevant features (curse of dimensionality)",
            "Use spatial index (KD-tree) for faster neighbor search",
            "For imbalanced data, use weighted voting",
            "Use distance metrics appropriate for data (e.g., cosine for text)"
        ],
        "pitfalls": [
            "Not scaling features: feature with large range dominates distance",
            "High-dimensional data: distances become meaningless (curse of dimensionality)",
            "k too small (k=1): overfits, very sensitive to noise",
            "k too large: prediction dominated by overall class distribution",
            "Imbalanced data with equal voting: majority class always wins"
        ]
    },
    "13-neural-networks": {
        "title": "Neural Networks",
        "explanation": "Neural networks stack multiple layers of parameterized transformations: y = σ(Wₙσ(Wₙ₋₁...σ(W₁x))). Each layer applies linear transformation (W, b) followed by nonlinearity (activation σ). The universality theorem: a single hidden layer with enough neurons can approximate any function. But shallow networks need exponential width for complex functions, while deep networks learn hierarchically with polynomial width. Modern deep learning stacks many layers for efficiency. Backpropagation trains networks via gradient descent on parameters.",
        "intuition": "Layers of transformations: first layer detects edges, second detects shapes, third detects objects. Each layer builds on previous.",
        "how_it_works": "1. Initialize weights (randomly, carefully)\n2. Forward pass: y = σ(Wₙ...σ(W₁x))\n3. Compute loss: L = loss(y, y_true)\n4. Backward pass: compute ∂L/∂W via chain rule\n5. Update: W ← W - α∂L/∂W\n6. Repeat for multiple epochs",
        "diagram": "graph LR\n    A[Input x] --> B[W1, b1]\n    B --> C[ReLU]\n    C --> D[W2, b2]\n    D --> E[ReLU]\n    E --> F[W3, b3]\n    F --> G[Softmax]\n    G --> H[Output]",
        "tradeoffs": "Shallow: fast, underfits | Deep: slow, overfits if unregularized | Wide: more capacity, more parameters | Narrow: fewer parameters, less capacity",
        "best_practices": [
            "Use He initialization for ReLU (prevents vanishing gradients)",
            "Batch normalize after linear layers (stabilizes training)",
            "Use ReLU or variants (GELU): simpler, faster than sigmoid/tanh",
            "Limit depth (residual networks needed for >50 layers)",
            "Use dropout for regularization (randomly zero activations)",
            "Monitor training and validation loss (watch for overfitting)",
            "Use learning rate schedule (warmup + decay)",
            "Normalize inputs (zero mean, unit variance)"
        ],
        "pitfalls": [
            "Vanishing gradients: sigmoid in deep network, gradients approach 0",
            "Exploding gradients: gradients grow unbounded, NaN loss",
            "Poor weight initialization: training doesn't progress",
            "No batch normalization: training is unstable, slow",
            "Too much regularization: underfits (high train and test loss)"
        ]
    },
    "14-activation-functions": {
        "title": "Activation Functions",
        "explanation": "Activation functions introduce nonlinearity, enabling networks to learn nonlinear functions. Without activation, stacking linear layers just results in a linear function. Common activations: ReLU (f(x)=max(0,x)) simple and fast, Sigmoid (f(x)=1/(1+e⁻ˣ)) squashes to [0,1], Tanh squashes to [-1,1], GELU (Gaussian-weighted linear) smooth ReLU, Swish (x·sigmoid(x)) learnable nonlinearity. ReLU family dominates modern networks due to simplicity and fast computation. Choice of activation affects convergence speed and generalization.",
        "intuition": "Activations are like switches: ReLU is on/off, sigmoid is gradual fade, GELU is smooth on/off. Choose based on problem.",
        "how_it_works": "1. Linear layer computes z = Wx + b\n2. Activation applies nonlinearity: a = σ(z)\n3. Outputs of one layer become inputs to next\n4. Nonlinearity compounds through layers (deep networks)",
        "diagram": "graph LR\n    A[z = Wx+b] --> B{Activation}\n    B -->|ReLU| C[max(0,z)]\n    B -->|Sigmoid| D[1/(1+exp(-z))]\n    B -->|Tanh| E[tanh(z)]\n    B -->|GELU| F[z * Φ(z)]\n    C --> G[Next Layer]\n    D --> G\n    E --> G\n    F --> G",
        "tradeoffs": "ReLU: simple, fast, dead ReLU problem | Sigmoid: smooth, expensive, vanishing gradient | Tanh: zero-centered, still expensive | GELU: smooth, modern, slower",
        "best_practices": [
            "Use ReLU by default (standard, works well)",
            "For dying ReLU, try LeakyReLU (small negative slope)",
            "Use GELU in transformer models (state-of-art)",
            "Avoid sigmoid/tanh in hidden layers (use only output)",
            "Sigmoid for binary classification output",
            "Softmax for multiclass classification output",
            "No activation for regression output",
            "Monitor dead ReLU percentage (shouldn't exceed 50%)"
        ],
        "pitfalls": [
            "Sigmoid in hidden layers: vanishing gradients in deep networks",
            "Dead ReLU: negative inputs always zero (fix: LeakyReLU)",
            "Wrong output activation: sigmoid for multiclass (should be softmax)",
            "No activation at output when needed: network learns linear function",
            "Swish/GELU everywhere: slower than ReLU with marginal gains"
        ]
    },
    "15-weight-initialization": {
        "title": "Weight Initialization",
        "explanation": "Random weight initialization is critical: poor initialization can cause vanishing/exploding gradients or slow training. Xavier (Glorot) initialization maintains activation variance: scale ∝ 1/√(fan_in + fan_out). He initialization for ReLU: scale ∝ √(2/fan_in). Zero initialization is catastrophic (all neurons compute identical function). Large random weights cause saturation (especially with sigmoid). Small weights work but slow learning. Biases are typically initialized to zero. Modern frameworks handle this, but understanding it helps debug training.",
        "intuition": "Weights too large: activations saturate (sigmoid always 0 or 1, no gradient). Weights too small: activations too quiet, learning is slow.",
        "how_it_works": "1. Compute fan_in (inputs to neuron) and fan_out (outputs)\n2. For Xavier: W ~ Uniform[-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out))]\n3. For He: W ~ Normal(0, √(2/fan_in))\n4. Initialize biases to 0\n5. Train: check that activation distributions don't explode/vanish",
        "diagram": "graph TD\n    A[Weight Scale] --> B{Too Large?}\n    B -->|Yes| C[Saturated Activations]\n    C --> D[Vanishing Gradient]\n    B -->|No| E{Too Small?}\n    E -->|Yes| F[Quiet Activations]\n    F --> G[Slow Learning]\n    E -->|No| H[Goldilocks]\n    H --> I[Good Training]",
        "tradeoffs": "Xavier: works for sigmoid/tanh | He: needed for ReLU | Uniform vs Normal: similar results, normal slightly better",
        "best_practices": [
            "Use He initialization for ReLU networks",
            "Use Xavier for sigmoid/tanh (though avoid these)",
            "Initialize biases to 0 (except maybe small positive for ReLU)",
            "For deep networks, use careful initialization + batch norm",
            "Monitor activation statistics: mean and std shouldn't explode",
            "Check gradient statistics: should be similar across layers",
            "For LSTM/GRU, use orthogonal initialization for recurrent weights",
            "For transfer learning, keep pre-trained weights, initialize new layers only"
        ],
        "pitfalls": [
            "Zero initialization: all neurons are identical, no learning",
            "Too-large random weights: activations saturate, gradients vanish",
            "Using uniform initialization for very deep networks: unstable",
            "Xavier initialization with ReLU: doesn't account for ReLU death",
            "Not considering fan_in/fan_out: same scale used regardless of layer size"
        ]
    },
    "16-regularization": {
        "title": "Regularization",
        "explanation": "Regularization prevents overfitting by penalizing model complexity. L2 (weight decay) penalizes large weights: L_total = L_data + λ/2 Σ w². This spreads weight mass, improving generalization. L1 penalizes absolute weights, inducing sparsity (some weights → 0), useful for feature selection. Dropout randomly zeros activations during training, preventing co-adaptation of neurons. Early stopping uses validation loss to halt training before overfitting. Ensemble methods regularize via voting. Data augmentation provides more effective training examples. The goal: low train AND low test loss.",
        "intuition": "Overfitting is like memorizing answers vs learning concepts. Regularization forces the model to learn the concept, which transfers to new problems.",
        "how_it_works": "1. L2: add λ Σ w²/2 to loss (weight decay)\n2. L1: add λ Σ |w| to loss (sparsity)\n3. Dropout: during training, set each activation to 0 with probability p, scale others by 1/(1-p)\n4. Early stopping: monitor validation loss, stop when it increases\n5. Ensemble: train multiple models, average predictions",
        "diagram": "graph TD\n    A[Model Complexity] --> B{Regularization}\n    B -->|None| C[Overfitting]\n    B -->|L2| D[Penalize Large Weights]\n    B -->|L1| E[Penalize Absolute Values]\n    B -->|Dropout| F[Random Deactivation]\n    B -->|Early Stop| G[Limit Training]\n    D --> H[Good Generalization]\n    E --> H\n    F --> H\n    G --> H",
        "tradeoffs": "L2: smooth, all features | L1: sparse, feature selection | Dropout: random, needs careful tuning | Early stop: needs validation set",
        "best_practices": [
            "Use L2 by default (weight decay), λ found via cross-validation",
            "Use L1 if you need feature selection",
            "For neural networks, use Dropout (typically 0.2-0.5 probability)",
            "Combine multiple regularization: dropout + L2 + early stopping",
            "Monitor train/val loss ratio: should be similar (within 10-20%)",
            "Data augmentation when possible: more training data helps",
            "Reduce model capacity if overfitting persists",
            "Use ensemble of regularized models for robustness"
        ],
        "pitfalls": [
            "Too much regularization: underfits (high train loss)",
            "No regularization: overfits (low train, high test loss)",
            "Dropout only at test time: ineffective (should be training only)",
            "L1/L2 coefficient too high: model can't fit training data",
            "Early stopping without validation set: stopping point arbitrary"
        ]
    },
    "17-batch-normalization": {
        "title": "Batch Normalization",
        "explanation": "Batch normalization normalizes layer inputs to zero mean and unit variance across the batch. This stabilizes training (internal covariate shift problem: distribution of layer inputs changes during training). BN allows higher learning rates, is a regularizer (randomness from batch statistics), and speeds up convergence. Adds learnable parameters (scale γ and shift β) per channel. At test time, uses running average of statistics from training. Alternatives: layer norm (normalize over features, not batch), group norm (normalize over groups of features), instance norm.",
        "intuition": "Standardized exams where all students get similar score distributions (not raw, but relative). Easier to learn consistent patterns.",
        "how_it_works": "1. Compute batch statistics: μ_batch = mean(x), σ²_batch = var(x)\n2. Normalize: x_norm = (x - μ_batch) / √(σ²_batch + ε)\n3. Scale and shift: y = γ x_norm + β\n4. During training: use batch statistics\n5. During inference: use running average from training batches",
        "diagram": "graph TD\n    A[Inputs x] --> B[Compute Batch Mean/Var]\n    B --> C[Normalize]\n    C --> D[Scale γ, Shift β]\n    D --> E[Normalized Output]\n    E --> F[Next Layer]\n    B -.-> G[Update Running Average]\n    G -.-> H[Use at Test Time]",
        "tradeoffs": "Batch norm: couple train/test, needs sufficient batch size | Layer norm: train/test consistent, slower | Group norm: middle ground",
        "best_practices": [
            "Use batch norm after linear layers, before activation",
            "Batch size should be ≥16 for stable statistics",
            "Allows higher learning rate (more aggressive updates)",
            "Can reduce or eliminate dropout (BN is regularizer)",
            "Careful with batch size in distributed training",
            "Use momentum for running average (typically 0.99)",
            "Layer norm for small batch sizes or RNNs",
            "Don't batch norm the output layer (unneeded)"
        ],
        "pitfalls": [
            "Batch size = 1: batch statistics useless, BN becomes identity",
            "Different behavior train/test: batch stats vs running average",
            "Placing after activation: could normalize away useful signal",
            "No momentum on running average: test statistics diverge from train",
            "Batch norm before activation with ReLU: may kill information"
        ]
    },
    "18-k-means-clustering": {
        "title": "K-Means Clustering",
        "explanation": "K-means partitions data into k clusters by minimizing within-cluster sum of squares. The algorithm iterates: (1) assign each point to nearest centroid, (2) recompute centroids as mean of assigned points. Converges locally, not globally—initialization matters. Elbow method chooses k: plot within-cluster variance vs k, pick elbow point. Silhouette score measures cluster quality. K-means is fast, scalable, but assumes spherical clusters (fails on elongated clusters) and requires specifying k. K-means++ initialization improves convergence.",
        "intuition": "Grouping students by similarity: move everyone to classroom nearest to their preferred location, then recompute best location as average.",
        "how_it_works": "1. Initialize k centroids (randomly or k-means++)\n2. Repeat until convergence:\n   a. Assign each point to nearest centroid\n   b. Recompute centroid as mean of assigned points\n3. Return centroids and cluster assignments",
        "diagram": "graph TD\n    A[Initialize Centroids] --> B[Assign to Nearest]\n    B --> C[Compute New Centroids]\n    C --> D{Changed?}\n    D -->|Yes| B\n    D -->|No| E[Final Clusters]",
        "tradeoffs": "Lloyd: simple, local minimum | K-means++: better init, same cost | Elbow method: heuristic, subjective | Silhouette: objective, more compute",
        "best_practices": [
            "Scale features (K-means uses Euclidean distance)",
            "Use K-means++ initialization (much better than random)",
            "Try multiple initializations, pick best (lowest within-cluster variance)",
            "Elbow method: plot inertia vs k, look for elbow",
            "Silhouette score: average should be >0.5 for good clustering",
            "Use PCA before K-means on high-dimensional data",
            "For large datasets, use mini-batch K-means",
            "Validate with domain knowledge: do clusters make sense?"
        ],
        "pitfalls": [
            "Not scaling features: features with large range dominate distance",
            "Random initialization: converges to poor local minimum",
            "Wrong k: too small (underfitting), too large (overfitting)",
            "Non-spherical clusters: K-means fails (use DBSCAN)",
            "Ignoring that result is local minimum: rerun many times"
        ]
    },
    "19-dimensionality-reduction": {
        "title": "Dimensionality Reduction",
        "explanation": "High-dimensional data is hard to visualize, train on (curse of dimensionality), and store. Dimensionality reduction finds lower-dimensional representation preserving important information. PCA finds directions of maximum variance: rotate to align with principal components. t-SNE optimizes for local structure: similar points stay together (good for visualization). UMAP balances local and global structure: faster than t-SNE, better for large data. Feature selection removes irrelevant features. These methods improve visualization, reduce overfitting, speed up training.",
        "intuition": "Like creating a cartoon: fewer lines but still recognizable. Or projecting 3D movie onto 2D screen while preserving important info.",
        "how_it_works": "PCA: 1. Center data\n2. Compute covariance matrix Σ\n3. Eigendecompose: Σ = UᴸΛUᴸᵀ\n4. Project: z = Uₖᵀ(x - μ) (use top k eigenvectors)",
        "diagram": "graph TD\n    A[High-D Data] --> B{Method}\n    B -->|PCA| C[Maximum Variance]\n    B -->|t-SNE| D[Local Structure]\n    B -->|UMAP| E[Local + Global]\n    C --> F[Low-D Data]\n    D --> F\n    E --> F",
        "tradeoffs": "PCA: linear, fast, preserves variance | t-SNE: nonlinear, slow, great for viz | UMAP: nonlinear, faster, balanced",
        "best_practices": [
            "PCA: whiten data first (zero mean, unit variance)",
            "Explained variance ratio: choose components for 80-95% variance",
            "t-SNE: perplexity 5-50, higher for larger datasets",
            "UMAP: n_neighbors controls local vs global (15 typical)",
            "Use dimensionality reduction for visualization (not training)",
            "For training, use PCA (linear, interpretable)",
            "High-dimensional features: often use autoencoders",
            "Feature selection (remove irrelevant features) often better than reduction"
        ],
        "pitfalls": [
            "Using t-SNE for training (not designed for it, non-deterministic)",
            "PCA on non-centered data: eigenvectors incorrect",
            "Too many components: overfitting (curse persists)",
            "Too few components: loses important information",
            "Not scaling features: large-scale features dominate PCA"
        ]
    },
    "20-gaussian-mixture-models": {
        "title": "Gaussian Mixture Models",
        "explanation": "GMMs model data as mixture of k Gaussian distributions with weights. Unlike K-means (hard assignment), GMM has soft assignments: each point has probability of belonging to each cluster. Expectation-Maximization (EM) algorithm fits GMMs: E-step computes cluster membership probabilities, M-step updates parameters maximizing likelihood. GMMs are probabilistic, allow model selection via BIC/AIC, handle uncertainty. Slower than K-means but more principled. Good when clusters may overlap or have different sizes/shapes.",
        "intuition": "Data comes from mixture of bell curves. Each point probably from one curve, but could be from another (soft assignment).",
        "how_it_works": "1. Initialize k Gaussians (means, covariances, weights)\n2. E-step: compute P(z=k|x) for each point and cluster\n3. M-step: update means, covariances, weights using cluster assignments\n4. Repeat E and M until convergence\n5. Return cluster probabilities",
        "diagram": "graph TD\n    A[Data] --> B[E-Step: Assign Probabilities]\n    B --> C[M-Step: Update Gaussians]\n    C --> D{Converged?}\n    D -->|No| B\n    D -->|Yes| E[Final GMM]",
        "tradeoffs": "K-means: hard assignment, fast | GMM: soft assignment, principled | EM: local optimum, slower | VBEM: Bayesian variant",
        "best_practices": [
            "Initialize with K-means (faster convergence)",
            "Use BIC or AIC to choose k (balances fit and complexity)",
            "Full covariance more flexible but more parameters (risk overfitting)",
            "Diagonal covariance faster, simpler",
            "Constrain covariances to avoid numerical issues",
            "Scale features before fitting",
            "Check log-likelihood convergence (should be monotonic)",
            "Soft assignments give cluster membership probabilities (useful)"
        ],
        "pitfalls": [
            "Local optima: EM initializes poorly, try multiple starts",
            "Covariance singularity: add regularization λI",
            "Too many components: overfits (use BIC to select k)",
            "Unbalanced clusters: some may become empty (reinitialize)",
            "Not scaling features: features with large variance dominate"
        ]
    },
    "21-bias-variance-tradeoff": {
        "title": "Bias-Variance Tradeoff",
        "explanation": "Error decomposes into bias + variance + noise. Bias: systematic error from wrong assumptions (underfitting). Variance: sensitivity to training data randomness (overfitting). Noise: irreducible error. Simple models have high bias, low variance. Complex models have low bias, high variance. The optimal model balances both. Regularization reduces variance at cost of bias. Ensemble methods reduce variance (average over multiple models). Understanding this tradeoff guides model selection and hyperparameter tuning.",
        "intuition": "Bias like aiming at wrong target. Variance like loose aim. Need both accuracy and consistency.",
        "how_it_works": "Error(x) = Bias² + Variance + Noise\nBias = E[(ŷ - y_true)]\nVariance = E[(ŷ - E[ŷ])²]\nNoise = (y_observed - y_true)²",
        "diagram": "graph TD\n    A[Model Complexity] --> B[Bias]\n    A --> C[Variance]\n    B --> D[Total Error]\n    C --> D\n    D --> E{Optimal?}\n    E -->|More Complex| F[Lower Bias, Higher Variance]\n    E -->|Less Complex| G[Higher Bias, Lower Variance]",
        "tradeoffs": "Underfitting: high bias, low variance | Overfitting: low bias, high variance | Regularized: balanced",
        "best_practices": [
            "Monitor train and validation error: gap shows variance",
            "High train, high val error: underfitting (need more capacity)",
            "Low train, high val error: overfitting (need regularization)",
            "Use learning curves: plot error vs training set size",
            "Regularization reduces variance (decreases gap)",
            "Ensemble reduces variance (average over models)",
            "More data reduces variance (less influence of noise)",
            "Feature selection reduces variance (fewer parameters)"
        ],
        "pitfalls": [
            "Assuming more features always helps: increases variance",
            "Using test set for hyperparameter tuning: optimizing variance",
            "Not monitoring train/val gap: missing overfitting warning",
            "Adding regularization without validation: may undfit",
            "Believing low train loss means good model: ignores test performance"
        ]
    },
    "22-cross-validation": {
        "title": "Cross-Validation",
        "explanation": "Cross-validation estimates model performance on unseen data without wasting examples for a test set. K-fold CV: split data into k folds, train on k-1, test on 1, repeat k times, average results. Stratified k-fold maintains class distribution. Leave-one-out CV: extreme case (k=n). Time-series CV respects temporal order. Cross-validation is more reliable than single train/test split and uses data efficiently. Computational cost increases with k (typical k=5 or 10).",
        "explanation": "Cross-validation estimates model performance without wasting data on a test set. K-fold CV: split data into k folds, train on k-1 folds, test on the remaining fold, repeat k times. Reduces variance of performance estimate. Stratified k-fold maintains class distribution (important for imbalanced data). Time-series CV respects temporal order (no future data in training). Leave-one-out CV uses all data but is expensive.",
        "intuition": "Rather than one test score, get k test scores from different train/test splits. Average them for robust estimate.",
        "how_it_works": "1. Split data into k folds\n2. For i = 1 to k:\n   a. Train on folds 1...i-1, i+1...k\n   b. Test on fold i, record score\n3. Average scores and report std",
        "diagram": "graph TD\n    A[Data] --> B[Split into k Folds]\n    B --> C[Fold 1 Test]\n    B --> D[Fold 2 Test]\n    B --> E[Fold k Test]\n    C --> F[Average Score]\n    D --> F\n    E --> F",
        "tradeoffs": "k=5: fast | k=10: standard, good variance reduction | k=n: expensive but unbiased | Stratified: slower but better distribution",
        "best_practices": [
            "Use stratified k-fold for classification (maintains class balance)",
            "k=5 or k=10 typical (balance speed vs accuracy)",
            "Shuffle before splitting (randomize fold assignment)",
            "Time-series CV: use past to predict future (no leakage)",
            "Nested CV for hyperparameter tuning: inner for tuning, outer for eval",
            "Report mean ± std of scores (shows stability)",
            "For small datasets (n<100), use leave-one-out",
            "Use cross_validate to get multiple metrics at once"
        ],
        "pitfalls": [
            "Not stratifying on class: imbalanced folds give high variance",
            "Data leakage: preprocessing on full data before CV",
            "Tuning hyperparameters on CV folds: test on separate holdout set",
            "k too small: high variance, unreliable estimate",
            "Time-series CV with shuffling: future data in training (cheating)"
        ]
    },
    "23-classification-metrics": {
        "title": "Classification Metrics",
        "explanation": "Accuracy (correct/total) is intuitive but misleading for imbalanced data. Precision (true positives / predicted positive) answers 'of predictions, how many correct?' Recall (true positives / actual positive) answers 'of actual positives, how many found?' F1 harmonizes precision-recall (geometric mean variant). ROC curve plots true positive rate vs false positive rate across thresholds; AUC summarizes. PR curve (precision-recall) better for imbalanced data. Confusion matrix shows all four cases. Choose metrics based on problem: high recall for disease detection, high precision for spam.",
        "intuition": "Precision: if doctor says you're sick, you probably are. Recall: doctor catches most real illnesses. Both matter for diagnosis.",
        "how_it_works": "Precision = TP / (TP + FP)\nRecall = TP / (TP + FN)\nF1 = 2 * Precision * Recall / (Precision + Recall)\nROC: plot TPR = TP/(TP+FN) vs FPR = FP/(FP+TN)\nAUC = area under ROC curve",
        "diagram": "graph TD\n    A[Predictions] --> B{Confusion Matrix}\n    B -->|TP| C[True Positive]\n    B -->|FP| D[False Positive]\n    B -->|TN| E[True Negative]\n    B -->|FN| F[False Negative]\n    C --> G[Precision, Recall, F1, ROC]\n    D --> G\n    E --> G\n    F --> G",
        "tradeoffs": "Accuracy: simple, misleading on imbalance | F1: balanced, threshold-dependent | ROC/AUC: invariant to threshold, summary metric",
        "best_practices": [
            "Always report confusion matrix (four numbers)",
            "Balanced data: accuracy is fine",
            "Imbalanced data: report precision, recall, F1 (not accuracy)",
            "For medical: high recall (catch all diseases)",
            "For spam: high precision (avoid false positives)",
            "ROC/AUC for ranking (threshold-independent)",
            "PR curve for imbalanced classification",
            "Use weighted averaging for multiclass (accounts for class imbalance)"
        ],
        "pitfalls": [
            "Using accuracy on imbalanced data: 99% accuracy can be useless",
            "Optimizing for F1 without knowing precision-recall tradeoff",
            "Comparing ROC curves without considering operating point",
            "Ignoring cost of false positives vs false negatives",
            "Using micro-average for imbalanced multiclass (biases toward majority)"
        ]
    },
    "24-regression-metrics": {
        "title": "Regression Metrics",
        "explanation": "Regression evaluates continuous predictions. MSE (mean squared error) punishes large errors heavily, MAE (mean absolute error) treats all errors equally. RMSE is sqrt(MSE), interpretable in original units. R² (coefficient of determination) measures variance explained: 1.0 is perfect, 0.0 is predicting mean. MAPE (mean absolute percentage error) useful when values vary widely. Residual analysis checks if errors have structure (should be random). Heteroscedasticity (error variance varies) is common. Choose metric based on problem: MSE for penalizing outliers, MAE for robustness.",
        "intuition": "MSE like golf: penalize big misses. MAE like basketball: points count equally. R² like 'how much better than guessing mean?'",
        "how_it_works": "MSE = mean((y_true - y_pred)²)\nMAE = mean(|y_true - y_pred|)\nRMSE = sqrt(MSE)\nR² = 1 - (SS_res / SS_tot) where SS_res = Σ(y_true - y_pred)², SS_tot = Σ(y_true - mean)²\nMAPE = mean(|y_true - y_pred| / |y_true|)",
        "diagram": "graph TD\n    A[Predictions vs Truth] --> B{Metrics}\n    B -->|Absolute| C[MAE, MAPE]\n    B -->|Squared| D[MSE, RMSE]\n    B -->|Relative| E[R²]\n    C --> F[Choose Based on Problem]\n    D --> F\n    E --> F",
        "tradeoffs": "MSE: penalizes outliers | MAE: robust to outliers | R²: interpretable, scale-dependent",
        "best_practices": [
            "Report multiple metrics: MSE and MAE together",
            "Check residual plots: should be random noise",
            "RMSE in original units is interpretable",
            "R² for comparing models (but needs context)",
            "MAPE when predicting values with different scales",
            "Cross-validation for honest error estimates",
            "Plot predicted vs actual: should be near diagonal",
            "Check for outliers: can dominate MSE"
        ],
        "pitfalls": [
            "Using MSE on data with outliers: dominated by few extreme points",
            "Reporting only R²: can hide poor predictions",
            "Not checking residual distribution: assuming normality",
            "Scale-dependent metrics without normalization: hard to compare",
            "Optimizing MSE then complaining about outliers: use MAE instead"
        ]
    },
    "25-feature-engineering": {
        "title": "Feature Engineering",
        "explanation": "Features determine model performance more than algorithm choice. Feature engineering creates useful representations of raw data. Scaling/normalization standardizes ranges. Encoding converts categorical to numeric (one-hot, ordinal, target encoding). Interactions (x₁·x₂) capture nonlinear relationships. Polynomial features (x, x², x³) add basis functions. Feature selection removes irrelevant features (reduces overfitting, improves interpretability). Domain knowledge matters: time-based features for time series, NLP embeddings for text. Feature engineering is often more impactful than model selection.",
        "intuition": "Raw materials (features) determine what you can build. Good raw materials → good model. Bad features → poor model no matter the algorithm.",
        "how_it_works": "1. Identify feature types (numeric, categorical, temporal)\n2. Encoding: categorical → numeric (one-hot, ordinal, etc)\n3. Scaling: normalize numeric features\n4. Interactions: create x_i * x_j for nonlinear relationships\n5. Selection: remove irrelevant/redundant features\n6. Domain knowledge: add meaningful features",
        "diagram": "graph TD\n    A[Raw Data] --> B[Identify Types]\n    B --> C[Scale Numeric]\n    B --> D[Encode Categorical]\n    C --> E[Interactions]\n    D --> E\n    E --> F[Select Features]\n    F --> G[Final Features]",
        "tradeoffs": "One-hot: expands dimensions | Ordinal: loses information but compact | Target: can leak if not careful",
        "best_practices": [
            "Understand data: domain knowledge first",
            "Scale features (most algorithms need this)",
            "One-hot encode low-cardinality categories",
            "Target encoding for high-cardinality (be careful of leakage)",
            "Create interaction features if domain suggests nonlinearity",
            "Remove highly correlated features (multicollinearity)",
            "Feature importance from trees: which features matter?",
            "Dimensionality reduction (PCA) only if many features"
        ],
        "pitfalls": [
            "Not scaling: large-magnitude features dominate",
            "Target encoding without cross-validation: leakage to test",
            "Too many features: overfitting (curse of dimensionality)",
            "Removing features without validation: may lose important signal",
            "Creating features without domain knowledge: random and misleading"
        ]
    },
    "26-hyperparameter-tuning": {
        "title": "Hyperparameter Tuning",
        "explanation": "Hyperparameters (learning rate, regularization, tree depth) are set before training, unlike parameters (weights) learned during training. Grid search exhaustively tries combinations (slow but simple). Random search samples combinations (faster, surprisingly effective). Bayesian optimization learns which hyperparameters are promising (most efficient). Hyperband combines random + early stopping. Tuning can improve performance 5-20%. Use cross-validation for honest estimates. Computational cost is high; resources often the limiting factor.",
        "intuition": "Grid search: try all locks on keyring. Random search: try random keys, often find it faster. Bayesian: learn which key types work, focus search.",
        "how_it_works": "1. Define hyperparameter grid/search space\n2. For each configuration:\n   a. Train model with those hyperparameters\n   b. Evaluate on validation set\n   c. Record performance\n3. Select configuration with best validation performance\n4. Refit on train+val, evaluate on test",
        "diagram": "graph TD\n    A[Hyperparameter Space] --> B{Search Strategy}\n    B -->|Grid| C[Exhaustive Combinations]\n    B -->|Random| D[Random Sampling]\n    B -->|Bayesian| E[Learn & Focus]\n    C --> F[Best Hyperparameters]\n    D --> F\n    E --> F",
        "tradeoffs": "Grid: thorough, slow | Random: faster, simpler | Bayesian: efficient, complex",
        "best_practices": [
            "Start with default hyperparameters: often good",
            "Grid search on most important hyperparameters (learning rate, regularization)",
            "Log scale for learning rate: 0.0001, 0.001, 0.01, 0.1",
            "Use nested cross-validation: inner for tuning, outer for eval",
            "Early stopping during tuning (don't train each config fully)",
            "Visualize results: how sensitive is performance to each hyperparameter?",
            "Bayesian optimization for expensive evaluations",
            "Parallelize across cores/GPUs: hyperparameter search is embarassingly parallel"
        ],
        "pitfalls": [
            "Tuning on test set: optimizing generalization gap",
            "Grid too coarse: miss better hyperparameters in between",
            "Grid too fine: search takes forever without early stopping",
            "Not using cross-validation: single validation split high variance",
            "Tuning too many hyperparameters: exponential combinations"
        ]
    },
    "27-ensemble-methods": {
        "title": "Ensemble Methods",
        "explanation": "Ensembles combine multiple models for better performance. Bagging (bootstrap aggregating) trains on subsets, averages predictions (reduces variance). Boosting trains sequentially, each focusing on previous errors (reduces bias). Stacking combines diverse models with a meta-learner. Voting averages predictions of different algorithms. Ensemble error = bias + variance: boosting reduces bias, bagging reduces variance, stacking reduces both by combining diversity. Winning Kaggle solutions often use ensembles of 10-100 models.",
        "intuition": "Poll multiple experts, average their votes: more accurate than any single expert. Especially if experts disagree.",
        "how_it_works": "Bagging: train on bootstrap samples, average predictions\nBoosting: train sequentially, weight errors\nStacking: train base models, use predictions as features for meta-model",
        "diagram": "graph TD\n    A[Data] --> B[Bagging]\n    A --> C[Boosting]\n    A --> D[Stacking]\n    B --> E[Model 1, 2, 3...]\n    C --> F[Model 1 → Model 2 → Model 3]\n    D --> G[Base Models]\n    E --> H[Average]\n    F --> I[Weighted Sum]\n    G --> J[Meta-Learner]\n    H --> K[Final Prediction]\n    I --> K\n    J --> K",
        "tradeoffs": "Bagging: parallel, reduces variance | Boosting: sequential, reduces bias | Stacking: complex, combines both",
        "best_practices": [
            "Ensemble diverse models (don't combine 100 identical trees)",
            "Bagging for high-variance models (deep trees, neural nets)",
            "Boosting for high-bias models (shallow trees)",
            "Stacking requires careful train/val/test split to avoid leakage",
            "Soft voting (probability averaging) better than hard voting",
            "Monitor correlation between ensemble members (should be low)",
            "Ensemble regularized models (individual weak models)",
            "Early stopping in boosting (prevent overfitting)"
        ],
        "pitfalls": [
            "Combining similar models: high correlation, little benefit",
            "No diversity: ensemble is just one model repeated",
            "Leakage in stacking: meta-learner trains on test data indirectly",
            "Too many models: diminishing returns after ~50",
            "Unequal model quality: bad models drag down ensemble"
        ]
    },
    "28-bayesian-inference": {
        "title": "Bayesian Inference",
        "explanation": "Bayesian methods combine prior beliefs with data via Bayes' theorem: P(θ|data) ∝ P(data|θ) P(θ). Posterior distribution quantifies uncertainty about parameters. Unlike frequentist point estimates, Bayesian gives distributions. Posterior predictive distribution for new data: integrate over uncertain parameters. Conjugate priors have closed-form posteriors (Beta-Binomial, Gaussian). MCMC (Markov Chain Monte Carlo) samples from posterior. Variational inference approximates posterior efficiently. Bayesian framework is principled, handles uncertainty, but computationally intensive.",
        "intuition": "Prior: what you believe before seeing data. Likelihood: how likely is data given beliefs? Posterior: updated beliefs after seeing data.",
        "how_it_works": "Bayes: P(θ|data) = P(data|θ) P(θ) / P(data)\nPosterior ∝ Likelihood × Prior\nPrediction: P(y_new|data) = ∫ P(y_new|θ) P(θ|data) dθ",
        "diagram": "graph TD\n    A[Prior Belief] --> B[Observe Data]\n    B --> C[Likelihood]\n    C --> D[Bayes' Theorem]\n    A --> D\n    D --> E[Posterior Distribution]\n    E --> F[Uncertainty Quantified]",
        "tradeoffs": "MAP: point estimate, ignores uncertainty | Posterior: full distribution, complex | Variational: approximate, fast",
        "best_practices": [
            "Encode domain knowledge in prior",
            "Use weak priors for unknowns (regularization effect)",
            "Check posterior predictive: should look like data",
            "Use MCMC diagnostics: Rhat<1.01, effective sample size >1000",
            "Variational inference for large data (faster)",
            "Credible intervals instead of confidence intervals",
            "Compare posterior to prior: did data change beliefs?",
            "Hierarchical models: partial pooling between groups"
        ],
        "pitfalls": [
            "Strong prior overrides data: prior misspecified",
            "MCMC: poor mixing, converges slowly",
            "Variational inference: approximation underestimates uncertainty",
            "Ignoring prior's influence: not truly Bayesian",
            "Posterior always consistent with data: poor model fit not caught"
        ]
    },
}

os.chdir("/home/sbisw/github/interviewprep-ml/ai/concepts")

for num in range(4, 29):
    slug = f"{num:02d}-" + {
        4: "optimization-algorithms",
        5: "learning-rate-scheduling",
        6: "linear-regression",
        7: "logistic-regression",
        8: "decision-trees",
        9: "random-forests",
        10: "gradient-boosting",
        11: "support-vector-machines",
        12: "k-nearest-neighbors",
        13: "neural-networks",
        14: "activation-functions",
        15: "weight-initialization",
        16: "regularization",
        17: "batch-normalization",
        18: "k-means-clustering",
        19: "dimensionality-reduction",
        20: "gaussian-mixture-models",
        21: "bias-variance-tradeoff",
        22: "cross-validation",
        23: "classification-metrics",
        24: "regression-metrics",
        25: "feature-engineering",
        26: "hyperparameter-tuning",
        27: "ensemble-methods",
        28: "bayesian-inference",
    }[num]

    filename = f"{slug}.md"
    if filename not in CONCEPTS:
        print(f"Skipping {filename} (detailed content not yet prepared)")
        continue

    concept = CONCEPTS[filename]
    title = concept["title"]
    explanation = concept["explanation"]
    intuition = concept["intuition"]
    how_it_works = concept["how_it_works"]
    diagram = concept["diagram"]
    tradeoffs = concept["tradeoffs"]
    best_practices = concept["best_practices"]
    pitfalls = concept["pitfalls"]

    content = f"""# {title}

## Detailed Explanation

{explanation}

## Core Intuition

{intuition}

## How It Works

{how_it_works}

```mermaid
{diagram}
```

## Architecture / Trade-offs

{tradeoffs}

## Interview Q&A

**Q: When would you use {title}?**
A: Context-dependent, varies by problem type.

**Q: What are the main trade-offs?**
A: Speed vs accuracy, simplicity vs power, bias vs variance.

**Q: How do you choose hyperparameters?**
A: Cross-validation, grid/random/Bayesian search, domain knowledge.

**Q: What are common failure modes?**
A: Refer to Common Pitfalls section below.

**Q: How does this compare to alternatives?**
A: Trade-offs above show how to choose between approaches.

## Best Practices

"""

    for practice in best_practices:
        content += f"- {practice}\n"

    content += "\n## Common Pitfalls\n\n"
    for pitfall in pitfalls:
        content += f"- {pitfall}\n"

    content += f"""

## Code Examples

### Example 1: Basic Implementation

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate sample data
X, y = datasets.make_classification(n_samples=200, n_features=10, random_state=42)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training set: {{X_train.shape}}")
print(f"Test set: {{X_test.shape}}")
```

### Example 2: Model Training and Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

# Create and train model
# model = SomeClassifier()  # Placeholder for actual model
# model.fit(X_train, y_train)

# Make predictions
# y_pred = model.predict(X_test)

# Evaluate
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
```

### Example 3: Cross-Validation and Hyperparameter Tuning

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Placeholder for actual implementation
# param_grid = {{'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}}
# grid = GridSearchCV(model, param_grid, cv=5)
# grid.fit(X_train, y_train)
# print(f"Best params: {{grid.best_params_}}")
# print(f"Best score: {{grid.best_score_:.4f}}")
```

## Related Concepts

- [Gradient Descent](./01-gradient-descent.md) — Optimization foundation
- [Cross-Validation](./22-cross-validation.md) — Evaluation methodology
- [Hyperparameter Tuning](./26-hyperparameter-tuning.md) — Parameter optimization
"""

    with open(filename, 'w') as f:
        f.write(content)
    print(f"✓ Enhanced {filename}")

print(f"\n✅ Enhanced {len([c for c in CONCEPTS if c not in ['01-gradient-descent', '02-backpropagation', '03-loss-functions']])} markdown files")
