#!/usr/bin/env python3
"""Replace placeholder 'How It Works' steps in AI concepts markdown files."""

import os
import re

BASE_DIR = "/home/sbisw/github/interviewprep-ml/ai/concepts"

PLACEHOLDER = "1. Step 1\n2. Step 2\n3. Step 3"

CONTENT = {
    "06-linear-regression.md": """\
1. Represent the model as ŷ = Xθ, where X is the feature matrix and θ are the parameters to learn
2. Define a loss function — Mean Squared Error: L(θ) = (1/n) Σ(ŷᵢ − yᵢ)²
3. Minimize the loss either analytically via the Normal Equation: θ = (XᵀX)⁻¹Xᵀy, or iteratively with gradient descent
4. Compute the gradient: ∂L/∂θ = (2/n) Xᵀ(Xθ − y)
5. Update parameters: θ ← θ − α · ∂L/∂θ until convergence
6. Add L2 regularization (Ridge) by penalizing θᵀθ: θ = (XᵀX + λI)⁻¹Xᵀy
7. Evaluate fit using R², RMSE, and residual plots""",

    "07-logistic-regression.md": """\
1. Model the log-odds of the positive class as a linear function: log(p/(1−p)) = Xθ
2. Apply the sigmoid function to map log-odds to probabilities: p = 1/(1+e^(−Xθ))
3. Define cross-entropy loss: L(θ) = −(1/n) Σ[yᵢ log(pᵢ) + (1−yᵢ) log(1−pᵢ)]
4. Compute gradient: ∂L/∂θ = (1/n) Xᵀ(p − y)
5. Update parameters with gradient descent (or L-BFGS for small datasets)
6. Apply threshold (default 0.5) to predicted probabilities to get class labels
7. For multiclass, extend to softmax: pₖ = e^(Xθₖ) / Σⱼ e^(Xθⱼ)""",

    "08-decision-trees.md": """\
1. Start with the full training dataset at the root node
2. For each candidate feature and split threshold, compute impurity (Gini or entropy) of resulting child nodes
3. Select the split that maximizes information gain (parent impurity − weighted child impurity)
4. Recursively split each child node using the same procedure
5. Stop when a stopping criterion is met: max_depth reached, min_samples_leaf, or no impurity improvement
6. Assign each leaf node the majority class (classification) or mean value (regression) of its training samples
7. Optionally prune the tree post-hoc by removing splits that don't improve validation performance""",

    "09-random-forests.md": """\
1. Draw B bootstrap samples (with replacement) from the training data, one per tree
2. For each bootstrap sample, train a decision tree — but at each split, consider only a random subset of √p features (for classification) or p/3 features (for regression)
3. Repeat until B trees are trained (typical B = 100–500)
4. For prediction, pass the input through all B trees: take majority vote (classification) or mean (regression)
5. Samples not in a tree's bootstrap sample form its Out-of-Bag (OOB) set — use these for free internal validation
6. Compute feature importances by measuring average impurity decrease per feature across all trees
7. Tune max_features (controls diversity) and n_estimators (controls variance reduction)""",

    "10-gradient-boosting.md": """\
1. Initialize with a simple prediction: F₀(x) = argmin_γ Σ L(yᵢ, γ) (e.g., mean of y for regression)
2. Compute pseudo-residuals (negative gradient of loss w.r.t. current predictions): rᵢ = −∂L(yᵢ, F(xᵢ))/∂F(xᵢ)
3. Fit a weak learner (shallow decision tree) to the pseudo-residuals
4. Find the optimal step size γ by line search: γ = argmin_γ Σ L(yᵢ, F(xᵢ) + γ·hₜ(xᵢ))
5. Update the model: Fₜ(x) = Fₜ₋₁(x) + η·γ·hₜ(x), where η is the learning rate
6. Repeat steps 2–5 for T boosting rounds
7. Final prediction is the sum of all weak learners: F(x) = F₀(x) + η·Σₜ hₜ(x)""",

    "11-support-vector-machines.md": """\
1. Map inputs to a high-dimensional feature space φ(x) using the kernel trick (no explicit computation needed)
2. Find the hyperplane w·φ(x) + b = 0 that maximizes the margin 2/‖w‖ between the two classes
3. Formulate as a convex optimization: minimize ½‖w‖² subject to yᵢ(w·φ(xᵢ)+b) ≥ 1
4. Introduce slack variables ξᵢ for soft margin (allow some misclassifications): minimize ½‖w‖² + C·Σξᵢ
5. Solve the dual problem using Lagrange multipliers — only support vectors (points on or inside the margin) have non-zero multipliers
6. Use the kernel function K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ) to compute inner products implicitly (RBF: e^(−γ‖xᵢ−xⱼ‖²))
7. Predict: ŷ = sign(Σᵢ αᵢyᵢK(xᵢ,x) + b), summing only over support vectors""",

    "12-k-nearest-neighbors.md": """\
1. Store the entire training dataset (lazy learning — no explicit training phase)
2. Given a new query point x, compute its distance to every training point using a distance metric (Euclidean, Manhattan, cosine)
3. Retrieve the k training points with the smallest distances to x
4. For classification: assign the majority class among the k neighbors (optionally weighted by 1/distance)
5. For regression: predict the mean (or distance-weighted mean) of the k neighbors' target values
6. Choose k using cross-validation — small k = low bias/high variance, large k = high bias/low variance
7. Optimize with spatial data structures (KD-tree for d < 20, Ball-tree for higher dimensions) to avoid O(nd) brute force""",

    "13-neural-networks.md": """\
1. Define network architecture: input layer (d features) → hidden layers (each with n neurons + activation) → output layer
2. Forward pass: for each layer l, compute zˡ = Wˡaˡ⁻¹ + bˡ, then apply activation aˡ = σ(zˡ)
3. Compute loss at the output: L = loss(ŷ, y) (e.g., cross-entropy for classification)
4. Backward pass: compute ∂L/∂W and ∂L/∂b for every layer using chain rule (backpropagation)
5. Start from output layer: δᴸ = ∂L/∂z, then propagate back: δˡ = ((Wˡ⁺¹)ᵀ δˡ⁺¹) ⊙ σ'(zˡ)
6. Update parameters with an optimizer (SGD, Adam): W ← W − α·∂L/∂W
7. Repeat forward and backward passes over mini-batches for multiple epochs until convergence""",

    "14-activation-functions.md": """\
1. Receive pre-activation value z = w·x + b (weighted sum of inputs)
2. Apply the non-linear activation function: a = σ(z)
3. For ReLU: a = max(0, z) — zero for negative inputs, identity for positive
4. For sigmoid: a = 1/(1+e⁻ᶻ) — squashes output to (0,1), used for binary output
5. For softmax (output layer): aₖ = e^(zₖ)/Σⱼ e^(zⱼ) — produces probability distribution over K classes
6. The activation's derivative σ'(z) is used in backpropagation: δ = δ_next · σ'(z)
7. Choice of activation affects gradient flow — ReLU avoids vanishing gradients; sigmoid causes them in deep networks""",

    "15-weight-initialization.md": """\
1. Before training begins, assign initial values to all weight matrices W and bias vectors b
2. Set biases to zero — this is always safe and standard
3. For Xavier/Glorot initialization (tanh, sigmoid): draw W from Uniform(−√(6/(nᵢₙ+nₒᵤₜ)), √(6/(nᵢₙ+nₒᵤₜ))) or Normal(0, √(2/(nᵢₙ+nₒᵤₜ)))
4. For He initialization (ReLU): draw W from Normal(0, √(2/nᵢₙ)) — accounts for ReLU zeroing half the neurons
5. The goal is to keep activation variance stable across layers: Var(a^l) ≈ Var(a^(l-1))
6. Poor initialization (too large → exploding activations, too small → vanishing gradients) prevents gradient flow
7. Verify initialization quality by checking that activation statistics (mean, std) are reasonable in the first forward pass""",

    "16-regularization.md": """\
1. Define the base loss function L(θ) (e.g., cross-entropy or MSE)
2. Add a regularization term: L_reg(θ) = L(θ) + λ·Ω(θ)
3. For L2 (Ridge/weight decay): Ω(θ) = ½‖θ‖² — penalizes large weights, shrinks them toward zero
4. For L1 (Lasso): Ω(θ) = ‖θ‖₁ — induces sparsity, drives some weights exactly to zero
5. For dropout: during each forward pass, randomly set each neuron's output to zero with probability p; scale remaining by 1/(1−p)
6. For early stopping: monitor validation loss; stop training when validation loss stops improving for patience epochs
7. Update regularized gradient: ∂L_reg/∂W = ∂L/∂W + λW (for L2), then apply optimizer update""",

    "17-batch-normalization.md": """\
1. For a mini-batch B = {x₁,...,xₘ}, compute batch mean: μ_B = (1/m)Σxᵢ
2. Compute batch variance: σ²_B = (1/m)Σ(xᵢ − μ_B)²
3. Normalize each activation: x̂ᵢ = (xᵢ − μ_B) / √(σ²_B + ε), where ε prevents division by zero
4. Scale and shift with learnable parameters γ and β: yᵢ = γx̂ᵢ + β (allows the network to undo normalization if needed)
5. During training, maintain running estimates of mean and variance using exponential moving average
6. During inference, use the running estimates (not batch statistics) for deterministic behavior
7. Backpropagate gradients through the normalization: ∂L/∂γ = Σ∂L/∂yᵢ · x̂ᵢ, ∂L/∂β = Σ∂L/∂yᵢ""",

    "18-k-means-clustering.md": """\
1. Choose k (number of clusters) and initialize k centroids — randomly or using k-means++ (choose each centroid with probability proportional to distance from nearest existing centroid)
2. Assignment step: assign each point xᵢ to the nearest centroid: cᵢ = argminⱼ ‖xᵢ − μⱼ‖²
3. Update step: recompute each centroid as the mean of its assigned points: μⱼ = (1/|Cⱼ|) Σᵢ∈Cⱼ xᵢ
4. Repeat assignment and update steps until centroids stop moving (convergence) or max iterations reached
5. Measure quality with inertia (sum of squared distances to nearest centroid) — lower is better
6. Run multiple restarts with different initializations and keep the result with lowest inertia
7. Select k using the elbow method (inertia vs k curve) or silhouette score (measures cluster cohesion vs separation)""",

    "19-dimensionality-reduction.md": """\
1. Center the data by subtracting the mean: X_centered = X − mean(X)
2. Compute the covariance matrix: C = (1/(n−1)) X_centeredᵀ X_centered
3. Compute eigenvectors and eigenvalues of C: the eigenvectors are the principal components
4. Sort eigenvectors by decreasing eigenvalue magnitude — they capture the most variance first
5. Select the top k eigenvectors to form projection matrix W (d × k)
6. Project data into the k-dimensional space: Z = X_centered · W
7. The explained variance ratio for each component = eigenvalue / sum(all eigenvalues) — use to choose k (aim for 90-95% cumulative explained variance)""",

    "20-gaussian-mixture-models.md": """\
1. Initialize K Gaussian components with means μₖ, covariances Σₖ, and mixing weights πₖ (often via k-means)
2. E-step (Expectation): compute soft assignment (responsibility) of each point to each component: rᵢₖ = πₖ·N(xᵢ|μₖ,Σₖ) / Σⱼ πⱼ·N(xᵢ|μⱼ,Σⱼ)
3. M-step (Maximization): update parameters using the responsibilities as weights: Nₖ = Σᵢ rᵢₖ, μₖ = (1/Nₖ)Σᵢ rᵢₖxᵢ
4. Update covariances: Σₖ = (1/Nₖ) Σᵢ rᵢₖ(xᵢ−μₖ)(xᵢ−μₖ)ᵀ
5. Update mixing weights: πₖ = Nₖ/n
6. Repeat E and M steps until log-likelihood converges: log p(X) = Σᵢ log Σₖ πₖ·N(xᵢ|μₖ,Σₖ)
7. Select K using BIC = −2·log p(X) + K·log(n) — choose K that minimizes BIC""",

    "21-bias-variance-tradeoff.md": """\
1. Decompose the expected prediction error at a point x into three components: Error = Bias² + Variance + Irreducible noise
2. Bias measures how far the average prediction is from the true value: Bias = E[f̂(x)] − f(x)
3. Variance measures how much predictions vary across different training sets: Var = E[(f̂(x) − E[f̂(x)])²]
4. Irreducible noise is the inherent randomness in the labels — cannot be reduced by any model
5. Complex models (deep trees, high-degree polynomials) have low bias but high variance — they fit noise
6. Simple models (linear, shallow trees) have high bias but low variance — they underfit the signal
7. Optimal model complexity minimizes total error — tune via validation curve (error vs model complexity plot)""",

    "22-cross-validation.md": """\
1. Split the training data into k equal-sized folds (e.g., k=5 or k=10)
2. For fold i (i=1..k): train the model on all folds except fold i, then evaluate on fold i
3. Record the evaluation metric (e.g., accuracy, RMSE) for each fold
4. Repeat for all k folds — every data point is used exactly once as a test point
5. Report mean ± standard deviation of the k metric values as the generalization estimate
6. For stratified k-fold (classification): ensure each fold preserves the original class distribution
7. For nested CV: wrap an inner CV (hyperparameter search) inside the outer CV — produces unbiased estimate of tuned model performance""",

    "23-classification-metrics.md": """\
1. Obtain predicted class labels (or probabilities) from the model on a held-out test set
2. Build the confusion matrix: count True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
3. Compute Precision = TP/(TP+FP) — of all predicted positives, what fraction are correct?
4. Compute Recall (Sensitivity) = TP/(TP+FN) — of all actual positives, what fraction did we catch?
5. Compute F1 = 2·(Precision·Recall)/(Precision+Recall) — harmonic mean, penalizes extreme imbalance
6. For threshold-independent evaluation, compute ROC-AUC: area under the Receiver Operating Characteristic curve (TPR vs FPR at all thresholds)
7. For imbalanced classes, prefer PR-AUC (area under Precision-Recall curve) — more sensitive to minority class performance""",

    "24-regression-metrics.md": """\
1. Obtain predicted values ŷᵢ from the model on a held-out test set
2. Compute residuals: eᵢ = yᵢ − ŷᵢ for each prediction
3. Compute Mean Absolute Error: MAE = (1/n)Σ|eᵢ| — robust to outliers, same units as target
4. Compute Mean Squared Error: MSE = (1/n)Σeᵢ² — penalizes large errors more heavily
5. Compute RMSE = √MSE — returns to original units; compare directly to target scale
6. Compute R² = 1 − SSres/SStot, where SSres = Σeᵢ², SStot = Σ(yᵢ−ȳ)² — proportion of variance explained (1.0 is perfect)
7. Plot residuals vs fitted values and residual histogram — patterns reveal model misspecification""",

    "25-feature-engineering.md": """\
1. Explore the raw data: check distributions, missing values, cardinality of categoricals, correlations
2. Handle missing values: impute with mean/median (numeric) or mode/constant (categorical), or use indicator features for missingness
3. Encode categorical variables: one-hot for low-cardinality, target encoding for high-cardinality, ordinal for ordered categories
4. Scale numeric features: StandardScaler (zero mean, unit variance) for distance-based models; MinMaxScaler for neural networks; RobustScaler when outliers are present
5. Create interaction features: multiply or divide related features when domain knowledge suggests interactions
6. Transform skewed features: log(1+x) for right-skewed, Box-Cox for general skewness, binning for non-linear relationships
7. Select features: remove zero-variance features, apply correlation filtering (drop one of correlated pairs), then use model-based importance for final selection""",

    "26-hyperparameter-tuning.md": """\
1. Define the hyperparameter search space: specify ranges and scales (log-scale for learning rate, linear for dropout rate)
2. Choose a search strategy: grid search for small spaces (< 3 hyperparameters), random search for larger spaces, Bayesian optimization for expensive models
3. For random search: sample n_iter configurations uniformly at random from the search space
4. For Bayesian optimization: fit a surrogate model (Gaussian process) to observed (config, score) pairs; use an acquisition function (Expected Improvement) to select the next configuration
5. Evaluate each configuration with k-fold cross-validation to get an unbiased performance estimate
6. Track all configurations and scores in a registry — prevents redundant evaluations
7. Select the best configuration, retrain on the full training set, and evaluate once on the held-out test set""",

    "27-ensemble-methods.md": """\
1. Train multiple diverse base learners on the training data (can be same algorithm with different hyperparameters, or different algorithms)
2. For bagging (e.g., Random Forest): train each base learner on a bootstrap sample of the data; average predictions to reduce variance
3. For boosting (e.g., GBM): train base learners sequentially, where each learner corrects the errors of the previous ensemble
4. For stacking: generate out-of-fold predictions from base learners; use these as features to train a meta-learner
5. For voting classifiers: combine predictions by majority vote (hard voting) or average probabilities (soft voting)
6. Combine base learner predictions using the chosen aggregation: mean/vote (bagging), weighted sum (boosting), meta-learner output (stacking)
7. Validate that the ensemble outperforms the best individual base learner on a held-out test set""",

    "28-bayesian-inference.md": """\
1. Specify a prior distribution p(θ) encoding beliefs about parameters before observing data
2. Define a likelihood function p(D|θ) — the probability of observing data D given parameters θ
3. Apply Bayes' theorem: p(θ|D) = p(D|θ) · p(θ) / p(D), where p(D) = ∫p(D|θ)p(θ)dθ is the normalizing constant
4. For conjugate priors (e.g., Beta-Binomial, Normal-Normal), compute the posterior analytically
5. For non-conjugate models, approximate the posterior with MCMC (Markov Chain Monte Carlo) sampling or Variational Inference
6. Extract point estimates: MAP (mode of posterior) or posterior mean; extract uncertainty via credible intervals
7. Use the posterior predictive distribution p(x_new|D) = ∫p(x_new|θ)p(θ|D)dθ to make predictions with uncertainty quantification""",
}


def fix_file(filepath, filename, real_steps):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if PLACEHOLDER not in content:
        print(f"  SKIP (placeholder not found): {filename}")
        return False

    new_content = content.replace(PLACEHOLDER, real_steps, 1)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"  OK: {filename}")
    return True


def main():
    fixed = 0
    skipped = 0
    missing = 0

    for filename, real_steps in CONTENT.items():
        filepath = os.path.join(BASE_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  MISSING: {filepath}")
            missing += 1
            continue
        result = fix_file(filepath, filename, real_steps)
        if result:
            fixed += 1
        else:
            skipped += 1

    print(f"\nDone: {fixed} fixed, {skipped} skipped (no placeholder), {missing} missing files")


if __name__ == "__main__":
    main()
