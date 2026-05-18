#!/usr/bin/env python3
"""
Fix placeholder Interview Q&A sections in AI concepts markdown files.
Replaces generic placeholder answers with real, specific Q&A for each concept.
"""

import re
import os

CONCEPTS_DIR = "/home/sbisw/github/interviewprep-ml/ai/concepts"

# Real Q&A content for each file (keyed by filename)
QA_CONTENT = {
    "04-optimization-algorithms.md": """\
**Q: When would you choose SGD over Adam?**
A: Use SGD with momentum for computer vision tasks where generalization matters more than fast convergence — SGD often finds flatter minima that generalize better. Adam converges faster but can overfit, especially on small datasets. For production CV models (ResNets, VGGs), SGD+momentum+LR schedule is still the standard.

**Q: What's the difference between Adam and AdamW?**
A: AdamW decouples weight decay from the gradient update, fixing a subtle bug in Adam where L2 regularization is scaled by the adaptive learning rate (effectively weakening regularization for parameters with large gradients). For transformers and modern architectures, always use AdamW over Adam when regularization matters.

**Q: Why does the learning rate interact with optimizer choice?**
A: Each optimizer expects a different learning rate magnitude. Adam typically uses 1e-3 to 1e-4; SGD typically uses 0.01 to 0.1. Using Adam's typical LR with SGD would barely move the parameters; using SGD's LR with Adam would cause instability. Always re-tune LR when switching optimizers.

**Q: What happens when you apply L2 regularization with Adam instead of AdamW?**
A: The L2 penalty gets scaled by Adam's adaptive learning rate, meaning parameters with historically large gradients receive weaker regularization. This is mathematically inconsistent with the intended behavior. AdamW separates the weight decay step from the gradient update, restoring the correct regularization effect.

**Q: How would you debug optimizer divergence (NaN loss)?**
A: Check in order: (1) learning rate too high — reduce by 10x; (2) missing gradient clipping for RNNs/transformers — add clip_grad_norm_(1.0); (3) numerical instability in loss function (log of zero) — add epsilon; (4) exploding weights from bad initialization — check initial loss value. Log gradient norms to identify which layer is exploding.

**Q: What's the intuition behind momentum in SGD?**
A: Momentum accumulates a velocity vector in the direction of consistent gradients and dampens oscillations in directions with inconsistent gradients. Imagine rolling a ball down a loss landscape — momentum lets it build speed in consistent directions (valleys) and reduces zigzagging across ravines. Typical momentum=0.9 means 90% of previous velocity is retained each step.
""",

    "05-learning-rate-scheduling.md": """\
**Q: Why use learning rate warmup at all? Can you just start with the target LR?**
A: For large models and transformers, starting with a high LR causes unstable updates in early training when parameters are far from any good optimum. The gradient magnitudes are large and inconsistent, so large updates thrash rather than converge. Warmup lets the model first move to a reasonable region, then apply the full LR. For small models trained from scratch, warmup is often unnecessary.

**Q: What's the difference between cosine annealing and step decay in practice?**
A: Step decay drops LR abruptly at fixed intervals, which can cause instability at each drop. Cosine annealing smoothly reduces LR following a cosine curve, which typically produces smoother convergence. In practice, cosine annealing often achieves comparable or better final accuracy without requiring tuning of when to drop. Step decay is still used in image classification benchmarks where exact training protocols are replicated.

**Q: When would you use ReduceLROnPlateau vs a fixed schedule?**
A: ReduceLROnPlateau is adaptive — it reduces LR when validation loss stops improving for `patience` epochs. Use it when you don't know the optimal training duration or want automatic adaptation. Fixed schedules (cosine, step) are better for reproducibility and hyperparameter tuning since the schedule doesn't depend on training dynamics. For large-scale training with a fixed budget, use a fixed schedule.

**Q: How does learning rate interact with batch size?**
A: When you increase batch size by k, the gradient estimate is k times more accurate (lower variance), so you can safely increase LR by approximately √k (linear scaling rule for large k). This is why papers report "LR 0.1 with batch size 256" — doubling batch to 512 suggests LR ~0.14. Ignoring this causes undertraining with large batches.

**Q: What is cyclic learning rate and when does it help?**
A: Cyclic LR (CLR) alternates between a minimum and maximum LR on a fixed cycle, allowing the optimizer to periodically escape local minima and explore the loss landscape. It often reduces training time by 2-5x because the model doesn't need extensive LR search — a range test to find min/max LR is sufficient. Works particularly well for CNNs; less commonly used for transformers where cosine with warmup dominates.

**Q: How would you find a good initial learning rate without extensive grid search?**
A: Use the LR range test (Leslie Smith, 2017): train for 1-2 epochs with LR increasing exponentially from 1e-7 to 10, log loss vs LR, pick the LR just before loss starts diverging (steepest decline region). This typically identifies a good LR in 5-10 minutes. Tools: fastai's lr_find(), PyTorch Lightning's lr_finder.
""",

    "06-linear-regression.md": """\
**Q: When would you choose Ridge over Lasso, and vice versa?**
A: Choose Ridge when you expect most features to contribute (correlated features), since it shrinks all coefficients toward zero smoothly. Choose Lasso when you want sparse solutions — it drives some coefficients exactly to zero, performing implicit feature selection. Use Elastic Net when you want both: sparsity with grouping effect for correlated features.

**Q: Why is the closed-form Normal Equation not used for large datasets?**
A: The Normal Equation requires computing (XᵀX)⁻¹, which is O(p³) in the number of features and O(np²) to form XᵀX. For n=1M rows and p=10k features, this is computationally prohibitive and XᵀX may not fit in memory. Gradient descent scales linearly with data and features, making it the practical choice beyond ~10k features.

**Q: What does a violation of linear regression assumptions look like, and how do you fix it?**
A: Heteroscedasticity (variance of residuals increases with fitted values) shows as a fan pattern in residuals vs fitted plot — fix with log transformation of the target. Non-linearity shows as curved residual patterns — add polynomial or interaction terms. Multicollinearity shows as large standard errors on coefficients — fix with Ridge or remove correlated features.

**Q: How do you interpret coefficients when features have different scales?**
A: Raw coefficients reflect the unit scale of each feature, making them incomparable. To compare feature importance, standardize all features first — then coefficients represent the change in y per standard deviation of x. Always standardize before interpreting coefficient magnitudes, especially when features have different units.

**Q: What's the difference between p-values and regularization for feature selection?**
A: P-values test whether a coefficient is statistically different from zero given the data — they depend heavily on sample size and can be misleading with correlated features. Regularization (Lasso) penalizes complexity and drives irrelevant coefficients to zero in a way that's more robust to collinearity. For feature selection, Lasso regularization is more reliable than p-value filtering.

**Q: How does adding more features affect linear regression?**
A: Adding irrelevant features adds noise to the model and reduces generalization, though training error keeps falling. With p > n features the system is underdetermined and requires regularization. More features also increase multicollinearity risk. Always validate with cross-validation — training R² improvement doesn't mean generalization improvement.
""",

    "07-logistic-regression.md": """\
**Q: Why is logistic regression preferred over linear regression for classification?**
A: Linear regression can predict values outside [0,1] which are uninterpretable as probabilities, and minimizing MSE on binary labels is suboptimal. Logistic regression directly models P(y=1|x) using the sigmoid function, ensures outputs are valid probabilities, and optimizes cross-entropy which is the proper loss for Bernoulli-distributed outcomes.

**Q: What happens when classes are perfectly linearly separable?**
A: Without regularization, gradient descent pushes the decision boundary to infinity — coefficients grow unbounded because the loss can always decrease by making predictions more extreme (0 or 1). In sklearn, the solver will issue a convergence warning. Fix by adding regularization (C parameter) which penalizes large coefficients and keeps the model bounded.

**Q: How do you handle a highly imbalanced dataset (1% positive class)?**
A: Set class_weight='balanced' to upweight minority class in the loss, or resample (oversample minority with SMOTE or undersample majority). Evaluate with PR-AUC or F1, not accuracy — a model predicting all negatives gets 99% accuracy but is useless. Tune the decision threshold based on the precision-recall trade-off for your application's cost structure.

**Q: When does logistic regression fail even with good features?**
A: Logistic regression assumes a linear decision boundary in the feature space. It fails when the true boundary is non-linear (e.g., XOR pattern). Solutions: add polynomial/interaction features manually, or switch to a kernel SVM or tree-based model. If the features are linearly separable but with complex decision boundaries, logistic regression will underfit.

**Q: What's the difference between L1 and L2 regularization in logistic regression?**
A: L2 (default C parameter in sklearn) shrinks all coefficients toward zero but keeps all features — good when all features contribute. L1 (penalty='l1') drives some coefficients exactly to zero, performing feature selection — good for high-dimensional sparse data. Elastic Net combines both. The C parameter is the inverse of regularization strength (smaller C = more regularization).

**Q: How would you extend binary logistic regression to multiclass?**
A: Two approaches: One-vs-Rest (OvR) trains k binary classifiers (one per class) and predicts the class with highest confidence; Softmax (multinomial) learns a single model with k output nodes using softmax normalization, ensuring probabilities sum to 1. Softmax is more principled and better when classes overlap; OvR can be faster for large k.
""",

    "08-decision-trees.md": """\
**Q: Why do decision trees overfit, and what are the main ways to prevent it?**
A: Unpruned trees can grow to depth n-1 (one leaf per training point), perfectly memorizing labels. The fix is controlling capacity: set max_depth (3-6), min_samples_leaf (5-20), or min_impurity_decrease. These hyperparameters should be tuned with cross-validation. Alternatively, use tree ensembles (Random Forest, GBM) which inherently reduce overfitting through averaging.

**Q: When would you prefer a decision tree over a Random Forest?**
A: When interpretability is critical — a single shallow tree (depth 3-4) is fully explainable to non-technical stakeholders and can be printed as a flowchart. Decision trees are also faster to train and predict, and better for rule extraction (convert tree to if-else rules). For predictive performance, Random Forests almost always win; for transparency, single trees are preferred.

**Q: What is the difference between Gini impurity and entropy as splitting criteria?**
A: Both measure class impurity and usually yield very similar trees. Gini is slightly faster to compute (no log). Entropy (information gain) can sometimes create more balanced splits. In practice the difference is negligible — sklearn uses Gini by default. Choose entropy if you want information-theoretic interpretability; otherwise Gini is fine.

**Q: How are decision trees biased toward certain feature types?**
A: Trees are biased toward features with many possible split points (high-cardinality continuous features) and features with many unique values, because they offer more chances for a high-information-gain split purely by chance. This inflates their apparent importance. Use permutation importance instead of Gini importance to get unbiased feature rankings.

**Q: How would you debug a decision tree that performs well on training but poorly on validation?**
A: The tree is overfitting. Check tree depth (if unlimited, start by setting max_depth=5); check if training accuracy is near 100% while validation is much lower; plot validation accuracy vs max_depth (should peak then decline). Also check for data leakage — a feature that's a proxy for the target will look great in training but fail in production.

**Q: Can decision trees handle missing values natively?**
A: Standard CART (sklearn's implementation) cannot — you must impute before fitting. However, some implementations like XGBoost and LightGBM handle missing values natively by learning the best direction to send missing values at each split. If using sklearn trees, impute with median (numeric) or mode (categorical), or add a missing indicator feature.
""",

    "09-random-forests.md": """\
**Q: Why does random feature subsampling (max_features) in Random Forests reduce variance?**
A: If all trees use the same features, they'll learn similar patterns and be highly correlated — averaging correlated predictors barely reduces variance. By forcing each split to consider only a random subset of features (√p for classification), trees are decorrelated, so averaging them gives a much larger variance reduction. The randomness creates diversity which is the source of Random Forest's power.

**Q: When does adding more trees stop helping in a Random Forest?**
A: After ~100-500 trees, the out-of-bag error stabilizes and additional trees provide negligible variance reduction while adding compute and memory cost. The law of diminishing returns applies — the first 100 trees provide most of the benefit. Monitor OOB error convergence to find the practical stopping point for your dataset.

**Q: How would you use a Random Forest for feature selection?**
A: Compute feature_importances_ from the trained forest (mean decrease in Gini). However, this is biased toward high-cardinality and correlated features. More reliable: use permutation importance (sklearn's permutation_importance), which measures how much performance drops when each feature is randomly shuffled. Keep top k features by permutation importance and retrain.

**Q: What's the difference between Random Forest and Gradient Boosting — when do you choose each?**
A: Random Forest trains trees in parallel (independent) — faster, easier to parallelize, more robust to hyperparameter choices. Gradient Boosting trains trees sequentially (each corrects the previous) — usually higher accuracy but slower, more sensitive to hyperparameters, needs careful LR tuning. Random Forest for quick baselines and when robustness matters; GBM for competition-level accuracy.

**Q: Why is the Out-of-Bag (OOB) error useful?**
A: Each bootstrap sample leaves out ~37% of training points, which serve as an implicit validation set for that tree. Aggregating OOB predictions across all trees gives a nearly unbiased generalization estimate without requiring a separate holdout set. This is particularly valuable for small datasets. OOB error closely tracks cross-validation error in practice.

**Q: How does Random Forest handle class imbalance?**
A: Poorly by default — it optimizes Gini which can be dominated by the majority class. Solutions: class_weight='balanced_subsample' reweights splits in each tree; class_weight='balanced' weights globally; or use stratified bootstrap sampling. Additionally, you can lower the classification threshold on predict_proba to improve minority class recall.
""",

    "10-gradient-boosting.md": """\
**Q: Why does gradient boosting use shallow trees rather than deep ones?**
A: Shallow trees (max_depth=3-6) are "weak learners" — slightly better than random. Boosting's power comes from sequentially adding many weak learners, each correcting the previous ensemble's errors. Deep trees would be strong learners that fit noise, causing overfitting early. The combination of many small corrections is more generalizable than fewer large corrections.

**Q: What's the role of the learning rate in gradient boosting?**
A: The learning rate (eta/shrinkage) scales each tree's contribution: F_t = F_{t-1} + η·h_t. Small η (0.01-0.1) requires more trees but generalizes better because each update is conservative. High η converges faster but overfits. The key insight: n_estimators and learning_rate are inversely related — halving learning rate roughly requires doubling n_estimators for similar performance.

**Q: How does XGBoost differ from sklearn's GradientBoostingClassifier?**
A: XGBoost adds second-order gradient information (Newton boosting), built-in L1/L2 regularization on leaf weights, column/row subsampling for variance reduction, and efficient sparse matrix handling. It's typically 10-100x faster than sklearn's GBM due to parallel tree building and cache-aware access patterns. XGBoost and LightGBM are the standard choices for production; sklearn's GBM is mainly for teaching.

**Q: When would you NOT use gradient boosting?**
A: When interpretability is critical (single decision tree is more explainable), when data is very high-dimensional and sparse (linear models or SVMs may be better), when you need very low latency predictions and can't afford the sequential tree traversal cost, or when you have very little data (<500 samples) and need the regularization benefits of simpler models.

**Q: What is early stopping and why is it important for gradient boosting?**
A: Early stopping monitors a validation metric during training and stops adding trees when the metric stops improving for `early_stopping_rounds` rounds. Without it, you must manually tune n_estimators — too few underfit, too many overfit. Early stopping automates this: set n_estimators=10000 and let early stopping find the optimal count. Always use a separate eval_set, not the training set.

**Q: How would you explain a gradient boosting prediction to a non-technical stakeholder?**
A: GBM builds a sequence of simple decision rules that each fix the mistakes of the previous rules, similar to a committee of experts where each new expert focuses on the cases the previous experts got wrong. The final prediction is the sum of all experts' weighted opinions. For individual predictions, use SHAP values to show which features pushed the prediction higher or lower.
""",

    "11-support-vector-machines.md": """\
**Q: What is the kernel trick, and why is it powerful?**
A: The kernel trick allows SVM to find non-linear decision boundaries without explicitly computing the high-dimensional feature transformation φ(x). Instead, it computes the inner product K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ) directly, which can be done efficiently even when φ maps to infinite dimensions (RBF kernel). This gives SVMs the power of neural networks in some settings without the computational cost of explicit feature maps.

**Q: What does the C parameter control, and how do you tune it?**
A: C controls the trade-off between maximizing the margin and minimizing training errors. Small C (large margin): allows more misclassifications, more regularization, better generalization on noisy data. Large C (small margin): fits training data closely, lower bias but higher variance. Tune C on log scale (0.001, 0.01, 0.1, 1, 10, 100) with cross-validation. For RBF kernel, tune C and gamma jointly.

**Q: When would SVM outperform Random Forest or Gradient Boosting?**
A: SVMs often excel on small datasets with clear margins, high-dimensional data (text classification, genomics), and when the kernel function is well-matched to the data structure. For text, the linear kernel SVM is extremely competitive with deep learning methods. SVMs also have strong theoretical guarantees. However, for most tabular datasets with n > 10k, GBM typically wins.

**Q: Why does SVM training scale poorly with dataset size?**
A: SVM solves a quadratic programming problem with O(n²) memory (kernel matrix) and O(n²-n³) compute. For n=100k samples, the kernel matrix alone is 80GB. Solutions: use LinearSVC (O(n) solve using SGD), subsample the data, use approximate kernels (Nyström approximation), or switch to a gradient boosting model which scales to millions of samples easily.

**Q: What's the difference between hard margin and soft margin SVM?**
A: Hard margin SVM requires all training points to be correctly classified with margin ≥ 1 — only works when data is linearly separable. Soft margin SVM introduces slack variables ξᵢ ≥ 0 that allow points to violate the margin, penalized by C·Σξᵢ in the objective. In practice, data is almost never perfectly separable, so soft margin (with C tuning) is always used.

**Q: How does multiclass SVM work?**
A: SVM is inherently binary. For k classes, two strategies: One-vs-One (OvO) trains k(k-1)/2 binary classifiers, predicts by majority vote — sklearn default; One-vs-Rest (OvR) trains k binary classifiers, predicts the class with highest decision score. OvO is more accurate but slower to train; OvR is faster. For large k (>10 classes), consider switching to a neural network or GBM.
""",

    "12-k-nearest-neighbors.md": """\
**Q: How do you choose k in KNN?**
A: Use cross-validation: plot validation error vs k (1 to ~√n). k=1 perfectly memorizes training data (overfit), large k approximates the global mean (underfit). The optimal k is usually in the range 3-20, where the bias-variance tradeoff is best. Odd k values avoid ties in binary classification. Start with k=5 as a baseline and tune from there.

**Q: Why does KNN perform poorly in high-dimensional spaces?**
A: The curse of dimensionality: in high dimensions, all points become approximately equidistant from the query point (distances concentrate), making nearest neighbors no more similar than random points. Additionally, the volume of space grows exponentially with dimensions so the k nearest neighbors may be very far away and not representative. PCA or feature selection before KNN helps significantly.

**Q: What is the computational complexity of KNN at prediction time?**
A: Brute force: O(nd) per query (n training points, d dimensions) — very slow for large datasets. KD-tree: O(d·log(n)) for low dimensions (d < 20). Ball-tree: O(d·log(n)) for higher dimensions. For d > 50, both trees degrade and brute force may be faster. For very large n (millions), use approximate nearest neighbor libraries like FAISS or HNSW.

**Q: When would you use KNN over a tree-based model?**
A: KNN works well when decision boundaries are irregular and don't align well with axis-aligned splits (as in trees). It's also good for recommendation systems (user-item similarity) and anomaly detection (distance to k-th neighbor as anomaly score). For tabular data with many features, tree-based methods almost always outperform KNN. KNN shines for geometric similarity tasks.

**Q: How does the choice of distance metric affect KNN?**
A: Euclidean distance treats all features equally and is sensitive to scale and irrelevant features. Manhattan distance is more robust to outliers. Cosine similarity is appropriate for sparse high-dimensional data (text). Mahalanobis distance accounts for feature correlations. Always normalize features first, and consider learning a task-specific distance metric (metric learning) when accuracy is critical.

**Q: How does KNN handle class imbalance?**
A: KNN is sensitive to imbalance because majority class points dominate the neighborhood even if the query point is near a minority class. Solutions: weight votes inversely by distance (weights='distance' in sklearn); use SMOTE to oversample minority class; use a modified voting scheme that accounts for class prior probability. For severe imbalance, KNN performs poorly and tree-based methods or specialized classifiers are better.
""",

    "13-neural-networks.md": """\
**Q: What is the vanishing gradient problem and which architectures are vulnerable?**
A: In deep networks, gradients are multiplied by the activation derivative at each layer during backpropagation. Sigmoid and tanh saturate (derivative ≈ 0 at extremes), so gradients shrink exponentially with depth. For a 10-layer network with sigmoid activations, gradients at the first layer can be 10^-10 smaller than at the output. ReLU networks, BatchNorm, residual connections (ResNets), and proper initialization all mitigate this.

**Q: Why are deep networks better than wide shallow networks for the same parameter count?**
A: Deep networks learn hierarchical representations — early layers learn simple features (edges, textures), deeper layers compose them into complex patterns (objects, concepts). A shallow network would need exponentially more neurons to represent the same function. Practically, depth provides a powerful inductive bias for structured data (images, text, sequences) that matches how those signals are actually generated.

**Q: How would you diagnose whether your neural network is underfitting or overfitting?**
A: Plot training vs validation loss curves. Large gap (train loss low, val loss high) = overfitting — add regularization (dropout, weight decay), reduce model size, or get more data. Both losses high = underfitting — increase model capacity, train longer, reduce regularization. If val loss decreases then increases, use the checkpoint from minimum val loss.

**Q: What's the intuition behind batch size selection?**
A: Large batches give more accurate gradient estimates but: (1) use more memory, (2) converge to sharp minima that generalize worse (known as the "generalization gap" effect). Small batches add noise that acts as regularization and often find flatter minima. Practical range: 32-256 for most tasks. With large batch training, you must increase LR proportionally (linear scaling rule) to maintain training speed.

**Q: When would you use a neural network vs gradient boosting for tabular data?**
A: Gradient boosting typically outperforms neural networks on tabular data (structured features, heterogeneous types). Neural networks excel when: data is very large (>1M rows), features have spatial/temporal structure, transfer learning is available, or you need joint training with other modalities. For most tabular tasks up to ~10M rows, XGBoost/LightGBM is the baseline to beat.

**Q: What is the universal approximation theorem, and what are its practical limitations?**
A: The theorem states that a single hidden layer with enough neurons can approximate any continuous function to arbitrary accuracy. However, it says nothing about how many neurons are needed (could be exponential), how to find the weights (training), or how well the network generalizes. In practice, depth is more efficient than width, and generalization requires regularization — the theorem is theoretically important but practically misleading.
""",

    "14-activation-functions.md": """\
**Q: What is the dying ReLU problem and how do you fix it?**
A: ReLU outputs zero for any negative input, and its gradient is also zero there. If a neuron's weights are updated such that its pre-activation is always negative for all training examples, that neuron will never activate and its weights will never be updated — it's "dead". Causes: too large learning rate, bad initialization. Fixes: LeakyReLU (small negative slope), ELU (smooth negative region), or careful LR and initialization tuning.

**Q: Why is ReLU preferred over sigmoid for hidden layers?**
A: Sigmoid saturates at both extremes (output near 0 or 1), where the derivative approaches 0 — this causes vanishing gradients in deep networks. ReLU has constant gradient of 1 for positive inputs, so gradients flow easily through deep networks. ReLU also has sparser activations (many zeros), which can be computationally efficient. Sigmoid is still used for binary output layers where a probability interpretation is needed.

**Q: What is GELU and why is it used in transformers?**
A: GELU (Gaussian Error Linear Unit) computes x·Φ(x) where Φ is the standard Gaussian CDF. Unlike ReLU which has a sharp cutoff at 0, GELU smoothly gates inputs based on their value relative to the distribution. Empirically, GELU provides small but consistent accuracy gains in transformers (BERT, GPT use it). The smooth gradient may help optimization in attention-heavy architectures.

**Q: When would you use softmax vs sigmoid for the output layer?**
A: Sigmoid for binary classification (single output, probability of positive class) or multi-label classification where classes are independent (each label can be independently 0 or 1). Softmax for mutually exclusive multiclass classification where probabilities must sum to 1. Never use softmax in hidden layers — it kills gradient flow by squashing all activations to compete against each other.

**Q: How does activation function choice affect weight initialization?**
A: He initialization (variance = 2/n_in) is designed for ReLU, which zeroes half its inputs — the factor of 2 compensates for this. Xavier initialization (variance = 2/(n_in + n_out)) is for symmetric activations like tanh and sigmoid. Using the wrong pairing (e.g., Xavier with ReLU) causes the variance to shrink with depth, slowing convergence. Always match initialization to activation.

**Q: What is the exploding gradient problem and how does it differ from vanishing gradients?**
A: Exploding gradients occur when gradient magnitudes grow exponentially with depth, causing parameter updates that are too large and destabilize training (loss diverges to NaN). Most common in RNNs with long sequences. Fix: gradient clipping (clip_grad_norm_(1.0)). Vanishing gradients cause parameters in early layers to barely update, preventing deep networks from learning hierarchical features. Both arise from multiplying many values during backprop.
""",

    "15-weight-initialization.md": """\
**Q: Why does zero initialization fail for neural networks?**
A: If all weights start at zero, every neuron in a layer computes identical outputs (symmetry), receives identical gradients during backprop, and updates identically — they never differentiate. The network effectively has only one neuron per layer regardless of width. Breaking symmetry requires random initialization. Biases can be zero because the weight randomness already breaks symmetry.

**Q: What is the intuition behind He and Xavier initialization?**
A: Both aim to keep activation variance stable across layers: Var(output) ≈ Var(input). If variance grows, activations saturate or explode; if it shrinks, gradients vanish. Xavier assumes symmetric activations (tanh) where the full neuron output is used. He accounts for ReLU zeroing ~half its inputs, requiring twice the variance to compensate (factor of 2/n_in instead of 1/n_in).

**Q: How would you diagnose a weight initialization problem in a deep network?**
A: After the first forward pass (before any training), inspect activation statistics: (1) plot histogram of activations in each layer — should be roughly Gaussian, centered near zero; (2) check for all-zero layers (dead ReLU from bad init) or all-saturated layers (sigmoid/tanh with too-large init); (3) check gradient norms — should be similar magnitude across layers. Frameworks like PyTorch autograd make this easy.

**Q: What special initialization do transformers use and why?**
A: Transformers often use truncated normal with std=0.02 (GPT-2, BERT). The output projections of attention and MLP blocks are sometimes initialized with 1/√(2·n_layers) scaling to prevent the residual stream variance from growing with depth (similar to how GPT-2 scales by 1/√n). This "depth scaling" is critical for training very deep transformers (12-96 layers) stably.

**Q: When does initialization matter less?**
A: With batch normalization or layer normalization, initialization matters much less because normalization rescales activations at each layer — the network can recover from poor initialization more easily. With modern adaptive optimizers (Adam) and normalization layers, training is relatively robust to initialization choice. Initialization matters most for plain networks (no BN) or networks trained with SGD.

**Q: What is orthogonal initialization and when is it used?**
A: Orthogonal initialization sets weight matrices to random orthogonal matrices (preserving Euclidean distance), which ensures singular values are all 1 — neither amplifying nor suppressing signals. It's used for RNNs to help gradient flow through many time steps without exploding or vanishing. Also used for some transformer experiments as it provides strong theoretical guarantees about gradient propagation.
""",

    "16-regularization.md": """\
**Q: What is the difference between L1 and L2 regularization in terms of the solution they produce?**
A: L2 (Ridge) penalizes the sum of squared weights — it shrinks all weights toward zero but rarely to exactly zero, producing dense solutions. L1 (Lasso) penalizes the sum of absolute values — it produces sparse solutions with many weights exactly zero because the gradient of |w| is constant (not proportional to w), creating a tendency to eliminate small weights entirely. L1 is better for feature selection; L2 for general regularization.

**Q: Why does dropout work as a regularizer?**
A: Dropout prevents co-adaptation: neurons can't rely on specific other neurons always being present, so each learns more robust, independent features. This approximates training an ensemble of 2^n different architectures (all subnetworks of the full network), and using all weights at inference (scaled by 1-p) approximates averaging their predictions. The regularization effect is similar to noise injection and weight sharing.

**Q: When should you use early stopping vs explicit regularization (L2/dropout)?**
A: Use early stopping as a free baseline — it's always safe and costs nothing. Add L2/dropout when early stopping alone isn't sufficient. Early stopping is coarser (controls total training steps), while L2/dropout control model capacity more precisely. For deep learning, combine all three: use dropout in the architecture, L2 (weight_decay) in the optimizer, and early stopping to select the best checkpoint.

**Q: How does the dropout rate affect training vs inference behavior?**
A: During training, each neuron is dropped with probability p — effectively training a different subnetwork each step. During inference, dropout is disabled (model.eval()) and weights are scaled by (1-p) to maintain expected activation magnitude. Forgetting model.eval() is a common bug that adds noise to inference predictions. Higher dropout (0.5) gives more regularization but requires more training; lower (0.1-0.2) is standard for CNNs.

**Q: What is weight decay and how does it relate to L2 regularization?**
A: Weight decay directly multiplies weights by (1 - α·λ) at each step: w ← (1-λ)w - α·∇L. L2 regularization adds λ‖w‖²/2 to the loss, which produces a gradient -λw, giving the same update with standard optimizers (SGD). However, with adaptive optimizers like Adam, L2 regularization gets scaled by the adaptive learning rate, weakening its effect — hence AdamW implements true weight decay separately from gradient updates.

**Q: How would you choose the regularization strength (lambda/C)?**
A: Always use cross-validation on a log scale (1e-5, 1e-4, ..., 1, 10). Plot val error vs lambda to find the sweet spot — too small: overfitting; too large: underfitting. For neural networks, start with weight_decay=1e-4 and dropout=0.2-0.5 as defaults. Monitor the training-validation gap: if large, increase regularization; if small but both losses are high, decrease regularization.
""",

    "17-batch-normalization.md": """\
**Q: What problem does batch normalization solve, and why does it help training?**
A: BN addresses "internal covariate shift" — the distribution of layer inputs changing during training as previous layers update, forcing each layer to continuously adapt. By normalizing activations to zero mean and unit variance, BN stabilizes training, allows higher learning rates (10x faster), and reduces sensitivity to initialization. The learnable γ and β parameters restore representational power while normalization provides stability.

**Q: What's the difference between batch normalization and layer normalization?**
A: BatchNorm normalizes across the batch dimension (all samples, single feature) and maintains running statistics for inference. LayerNorm normalizes across the feature dimension (single sample, all features) and has no batch dependency. LayerNorm is preferred in transformers (variable-length sequences, small batches), RNNs, and any setting where batch statistics are unreliable. BatchNorm is standard for CNNs with large batches.

**Q: Why must you call model.eval() before inference when using batch normalization?**
A: During training, BN uses current batch statistics (mean and variance). During inference, you want deterministic behavior independent of the current batch size. model.eval() switches BN to use running statistics accumulated during training. Without it, a batch size of 1 would use a single sample's statistics (very noisy), and results would vary between calls. This is one of the most common production bugs with neural networks.

**Q: What happens when batch size is very small (1-4) with batch normalization?**
A: With small batches, the batch statistics (mean, variance) are noisy estimates of the true statistics — normalizing by them adds significant noise to training. Batch size 1 is particularly bad since variance is undefined. Solutions: use LayerNorm (size-independent), Group Normalization (normalizes within feature groups), or Instance Normalization. For object detection with small batches, Group Normalization is the standard replacement.

**Q: Where should batch normalization be placed — before or after the activation function?**
A: Original paper: Conv → BN → Activation. Many subsequent papers found Conv → Activation → BN works similarly or better for some architectures. In practice, try both. The key insight is that BN before activation can cause the normalized input to the activation to be roughly zero-mean, which is good for tanh/sigmoid but doesn't matter much for ReLU. For residual networks, placing BN after the final convolution and before the residual addition is common.

**Q: Can you use dropout and batch normalization together?**
A: Yes, but the order matters and interactions are tricky. BN → Dropout (in that order) is typical. Applying dropout before BN is problematic because dropout changes the variance of activations, which BN then normalizes away, reducing dropout's regularization effect. Also note that in some architectures, BN already provides sufficient regularization — adding aggressive dropout on top may cause underfitting.
""",

    "18-k-means-clustering.md": """\
**Q: What are the main failure modes of k-means and when does it break down?**
A: K-means assumes spherical, equal-size, equal-density clusters. It fails on: elongated clusters (use DBSCAN or GMM), clusters of very different sizes (small clusters get absorbed), non-convex shapes (concentric rings), and data with very different feature scales (must standardize first). A single outlier can pull a centroid far from its cluster, effectively emptying it.

**Q: How does k-means++ initialization differ from random initialization, and why does it matter?**
A: Random initialization can place multiple centroids in the same dense region, leaving other clusters unrepresented. K-means++ chooses each centroid with probability proportional to its squared distance from the nearest existing centroid — spreading centroids to cover the data space. This reduces the chance of poor convergence, improves final inertia by 2-5x on average, and is the sklearn default (init='k-means++').

**Q: How would you determine the optimal k for a dataset?**
A: Use the elbow method (plot inertia vs k, find the "elbow" where the curve flattens — diminishing returns) and silhouette score (ranges from -1 to 1; higher is better; peaks at the optimal k). Use both together: inertia always decreases with more k (never use it alone), while silhouette penalizes poor cluster separation. For domain problems, also consider the business interpretation of k clusters.

**Q: What is the time complexity of k-means, and how do you scale it to large datasets?**
A: Standard k-means: O(n·k·d·i) per iteration where n=samples, k=clusters, d=features, i=iterations. For large datasets, MiniBatchKMeans processes random batches of size b per iteration: O(b·k·d·i) — typically 3-10x faster with similar quality. For very large k or d, consider approximate methods or dimensionality reduction before clustering.

**Q: Why must features be scaled before applying k-means?**
A: K-means uses Euclidean distance. A feature with range 0-1000 will dominate over a feature with range 0-1, regardless of their actual importance. Scaling ensures each feature contributes proportionally. Use StandardScaler (zero mean, unit variance) or MinMaxScaler. Failing to scale is one of the most common mistakes — clusters will primarily reflect the high-magnitude features.

**Q: How would you evaluate the quality of k-means clusters when you have no ground truth labels?**
A: Use intrinsic metrics: silhouette score (cohesion vs separation — higher is better), Calinski-Harabasz index (ratio of between-cluster to within-cluster variance — higher is better), Davies-Bouldin index (average similarity of each cluster to its most similar cluster — lower is better). Also visualize in 2D with PCA or t-SNE, and check if clusters make semantic sense in the domain.
""",

    "19-dimensionality-reduction.md": """\
**Q: When should you use PCA vs t-SNE vs UMAP?**
A: PCA: for preprocessing before ML models (linear, preserves global variance), interpretable components, fast. t-SNE: for visualization only (2D/3D) — it's non-parametric (can't transform new points) and distances between clusters are not meaningful. UMAP: better than t-SNE for visualization (preserves more global structure, faster, can transform new points). Never use t-SNE/UMAP features as input to other models — use PCA for preprocessing.

**Q: What does explained variance ratio tell you in PCA?**
A: Each principal component's explained_variance_ratio_ is that component's eigenvalue divided by the sum of all eigenvalues — it's the fraction of total variance captured by that component. A cumulative explained variance of 95% means you've captured 95% of the information with far fewer dimensions. Plot the cumulative curve (scree plot) and select the "knee" or use the 95% rule to choose n_components.

**Q: How do you choose the number of components for PCA used as preprocessing?**
A: Use cross-validation: fit a pipeline (PCA → model) and evaluate CV performance for n_components in [0.8, 0.9, 0.95, 0.99] (fraction of variance) or a range of integers. More components = less information loss but slower downstream model. Common shortcut: use 95% explained variance as a starting point, then tune. Never just pick a number without validating on downstream model performance.

**Q: What are the limitations of PCA for non-linear data?**
A: PCA only finds linear projections — it can't capture non-linear manifolds (e.g., Swiss roll data). If the data lies on a curved manifold, PCA will project it to a hyperplane, potentially mixing different classes. Solutions: kernel PCA (uses kernel trick to implicitly map to higher dimensions), autoencoders (neural network-based non-linear reduction), or UMAP/t-SNE for visualization. For most practical preprocessing, PCA still works surprisingly well.

**Q: How is t-SNE different from PCA in terms of what it preserves?**
A: PCA preserves global structure (maximizes variance, preserves large pairwise distances) — far-apart points in high dimensions stay far apart. t-SNE preserves local structure — it keeps nearby points together but distorts global distances, so clusters in t-SNE plots may be closer or farther than they actually are. Never interpret t-SNE cluster separation distances as meaningful. t-SNE also uses a probabilistic model (KL divergence between high-dim and 2D neighbor distributions).

**Q: What is the curse of dimensionality and how does dimensionality reduction help?**
A: In high dimensions: (1) all distances become approximately equal — nearest neighbors are not meaningful; (2) volume grows exponentially — data becomes sparse; (3) many features are redundant or noisy — they add variance without signal. Dimensionality reduction removes these redundant/noisy dimensions, making distance metrics meaningful again. For KNN and SVMs, PCA before the model often improves performance significantly.
""",

    "20-gaussian-mixture-models.md": """\
**Q: What is the key difference between k-means and GMM clustering?**
A: K-means makes hard assignments — each point belongs to exactly one cluster. GMM makes soft assignments — each point has a probability of belonging to each cluster. GMM also models the shape of clusters through covariance matrices (can capture elongated, correlated clusters), while k-means assumes spherical, equal-size clusters. GMM generalizes k-means: k-means is a special case of GMM with spherical covariances and hard assignments.

**Q: What is the EM algorithm and why might it fail to find the global optimum?**
A: The Expectation-Maximization algorithm alternates between computing soft assignments (E-step) and updating parameters to maximize the likelihood given those assignments (M-step). It's guaranteed to converge to a local maximum of the likelihood but not the global maximum, because the likelihood surface is non-convex with many local optima. Always run multiple random restarts (n_init=5-10) and keep the solution with the highest log-likelihood.

**Q: How do you choose the number of components in a GMM?**
A: Use BIC (Bayesian Information Criterion) or AIC (Akaike Information Criterion): BIC = -2·log(L) + k·log(n). BIC penalizes complexity more than AIC. Fit GMMs for k=1..15, plot BIC vs k, choose k at the minimum (or where the curve starts to flatten for AIC). BIC typically selects more parsimonious models than AIC. BIC is preferred when model selection is the goal; AIC when predictive performance matters.

**Q: What are the different covariance types in GMM and when do you use each?**
A: 'full' — each component has its own full covariance matrix (most flexible, most parameters); 'tied' — all components share one covariance matrix (good if clusters have similar shape); 'diag' — diagonal covariance (faster, fewer parameters, assumes feature independence within clusters); 'spherical' — single variance per component (equivalent to k-means with EM). Start with 'full' for flexibility; use 'diag' if high-dimensional.

**Q: When would you use GMM for density estimation rather than clustering?**
A: GMM is a parametric density estimator: p(x) = Σₖ πₖ·N(x|μₖ,Σₖ). Use it for anomaly detection (low-probability regions under the fitted density are anomalies), generative modeling (sample from the fitted distribution), and likelihood-based model comparison. Unlike k-means, GMM gives a proper probability density, enabling log-likelihood evaluation on new data. This makes GMM useful as a prior in Bayesian models.

**Q: How does a GMM differ from a standard Gaussian (single component) in practice?**
A: A single Gaussian assumes unimodal, symmetric data — if data has multiple clusters or non-symmetric distribution, it fits poorly. GMM is a universal density approximator: with enough components, it can approximate any continuous distribution. In practice, even 2-5 components can dramatically improve fit for multimodal data. Score the fit on held-out data with log_prob() and compare single Gaussian vs GMM.
""",

    "21-bias-variance-tradeoff.md": """\
**Q: How do you diagnose whether your model has high bias or high variance from learning curves?**
A: Plot training and validation error vs training set size. High bias (underfitting): both training and validation error are high and converge to a high value — adding more data won't help, need more complex model. High variance (overfitting): large gap between low training error and high validation error — adding more data helps (curves converge), or add regularization.

**Q: Why does the bias-variance tradeoff differ between classical ML and modern deep learning?**
A: In classical ML, increasing model complexity monotonically increases variance and decreases bias (U-shaped test error curve). Modern deep learning exhibits "double descent" — after the classical overfitting peak, continuing to increase model size causes test error to decrease again. Very overparameterized models (GPT-3: 175B params for millions of training examples) can generalize well due to implicit regularization from gradient descent.

**Q: What's the effect of ensemble methods on bias and variance?**
A: Bagging (Random Forest): reduces variance by averaging uncorrelated trees — bias stays the same. Boosting (GBM): primarily reduces bias by sequentially correcting errors — slight variance increase. Stacking: can reduce both. This is why Random Forests excel when single trees overfit (high variance), and boosting excels when base models underfit (high bias, weak learners).

**Q: How does regularization affect the bias-variance tradeoff?**
A: Strong regularization increases bias (constrains model predictions toward simple functions) and decreases variance (model is less sensitive to training data noise). Zero regularization: low bias, potentially high variance (overfit). Too much regularization: high bias, low variance (underfit). The regularization hyperparameter (lambda, C, dropout rate) directly controls where on the bias-variance curve the model sits.

**Q: What is irreducible error and why can't it be eliminated?**
A: Irreducible error (Bayes error) is the noise in the labels themselves — if two identical input examples have different labels due to measurement noise, label ambiguity, or missing features, no model can predict correctly. It represents the minimum achievable error rate. You can reduce it by collecting better features, cleaning labels, or reducing measurement noise in the data collection process — not by improving the model.

**Q: How does cross-validation relate to the bias-variance tradeoff?**
A: Cross-validation gives an estimate of test error that includes both bias and variance components of the model. The mean CV score estimates the expected performance (inversely related to bias for a fixed model class), while the variance across folds estimates the sensitivity to training data choice. Highly variable CV scores indicate a high-variance model that needs regularization.
""",

    "22-cross-validation.md": """\
**Q: What is data leakage in cross-validation and how do you prevent it?**
A: Data leakage occurs when information from the test fold contaminates the training process. The most common form: fitting a scaler, imputer, or feature selector on the full dataset, then using those statistics in CV — the test fold's statistics influence the training preprocessing. Prevention: always wrap all preprocessing in a sklearn Pipeline so transformations are fit only on the training folds.

**Q: When should you use stratified k-fold vs regular k-fold?**
A: Use StratifiedKFold whenever the target class distribution matters — classification problems, especially with imbalanced classes. Regular KFold can produce folds with very different class ratios (especially for rare classes), making CV estimates noisy. For regression, regular KFold is standard; optionally use KFold with shuffle=True to ensure random fold assignment.

**Q: Why does nested cross-validation give a more honest performance estimate?**
A: Standard CV with hyperparameter tuning optimistically biases performance — you select hyperparameters based on validation performance, which inflates the estimate. Nested CV uses an outer CV loop for performance estimation and an inner CV loop for hyperparameter tuning, ensuring no hyperparameter choice information leaks into the outer performance estimate. The difference between nested and non-nested CV scores reveals the optimism bias.

**Q: How many folds should you use and what are the trade-offs?**
A: k=5 or k=10 are the standard choices. Large k (e.g., LOOCV): low bias (each test set is one point, training set is nearly full), high variance (highly variable scores), computationally expensive. Small k (e.g., k=2): high bias (training on only half the data), lower variance. k=5-10 balances bias, variance, and compute for most datasets. Use LOOCV only for very small datasets (<50 samples).

**Q: How does cross-validation for time-series differ from standard k-fold?**
A: Standard k-fold shuffles data, allowing future data to appear in the training fold — this is leakage for time-series. TimeSeriesSplit (sklearn) always trains on past data and tests on future: fold 1 trains on months 1-2, tests on month 3; fold 2 trains on months 1-3, tests on month 4; etc. This correctly simulates deployment: you predict future from past. Also consider gap periods between train and test to avoid autocorrelation.

**Q: What is the difference between CV for model selection vs CV for performance estimation?**
A: CV for model selection: compare multiple models or hyperparameter settings using the same CV splits — choose the model with the best mean CV score. CV for performance estimation: estimate how well the chosen model will generalize — report the CV score as the expected performance. If you use the same CV for both, you need nested CV to avoid optimistic bias in the performance estimate.
""",

    "23-classification-metrics.md": """\
**Q: When should you use AUC-ROC vs AUC-PR?**
A: Use AUC-ROC when the negative class matters (balanced or near-balanced datasets). Use AUC-PR (average precision) for imbalanced datasets — ROC is overly optimistic because it includes true negatives in the denominator, and rare positive predictions can look good by ROC but terrible by PR. As a rule of thumb: if positive class < 10% of data, AUC-PR is more meaningful. Both should be reported for imbalanced problems.

**Q: Why is accuracy a misleading metric for imbalanced datasets?**
A: With 99% negative and 1% positive, a model that always predicts negative gets 99% accuracy — but is completely useless. Accuracy treats all errors equally and ignores class frequency. Use F1 (balances precision and recall for the positive class), AUC-PR, or Matthews Correlation Coefficient (MCC) which accounts for all four confusion matrix cells and gives credit for correctly identifying both classes.

**Q: How do you choose the right classification threshold?**
A: The default threshold of 0.5 is arbitrary — it only makes sense when false positives and false negatives have equal cost. Define the business cost matrix (cost of FP vs FN), then find the threshold that minimizes expected cost: plot cost vs threshold and select the minimum. For medical diagnosis, FN (missed disease) costs more than FP (unnecessary test), so use a lower threshold. Use model.predict_proba() and threshold manually.

**Q: How do you evaluate a classifier on multiclass problems?**
A: Use classification_report() which shows per-class precision, recall, F1, and aggregate averages. Macro average treats all classes equally (good when class sizes are balanced). Weighted average weights by class size (penalizes poor performance on frequent classes more). For visualization, the confusion matrix shows which classes are confused with each other. Use the normalized confusion matrix (normalize='true') to see per-class recall regardless of class size.

**Q: What does high precision but low recall mean in practice?**
A: The model is conservative — it only predicts positive when very confident, so predictions are usually correct (high precision) but it misses many actual positives (low recall). Example: a fraud detection system that only flags obvious fraud has high precision (few false alarms) but low recall (lets subtle fraud through). Adjust the threshold downward to accept more false positives and improve recall. The trade-off is application-specific.

**Q: How would you evaluate a ranking model or retrieval system differently from a classifier?**
A: Ranking models need rank-aware metrics: NDCG (normalized discounted cumulative gain — rewards relevant items appearing earlier in the list), MAP (mean average precision — averages precision at each relevant item's position), and MRR (mean reciprocal rank — focuses on the first relevant result). These capture the quality of ordering, not just binary relevance. Scikit-learn doesn't include these natively; use ir_measures or trec_eval.
""",

    "24-regression-metrics.md": """\
**Q: When should you prefer MAE over RMSE and vice versa?**
A: Use RMSE when large errors are especially costly (financial loss prediction, safety-critical systems) — its squared error term penalizes outliers heavily. Use MAE when all errors should be treated proportionally (demand forecasting, general metrics) — it's more robust to outliers and directly interpretable in the target's units. Report both: if RMSE >> MAE, you have large outliers; understanding this matters for model diagnosis.

**Q: What does R² actually measure, and when is it misleading?**
A: R² = 1 - SS_res/SS_tot measures the proportion of variance in y explained by the model. It's misleading when: (1) used for non-linear models without checking residuals (high R² doesn't mean good fit); (2) compared across datasets with different y variance (R² depends on y variance, not just model quality); (3) reported on training data without cross-validation; (4) negative on test data — means the model is worse than predicting the mean, which is a red flag.

**Q: How would you diagnose heteroscedasticity and why does it matter?**
A: Plot residuals vs fitted values — heteroscedasticity shows as a funnel shape (variance grows with predicted values). It violates OLS assumptions, making confidence intervals and p-values unreliable (not the predictions themselves). Fix options: log-transform the target (if residuals have positive skew), use weighted least squares, or use robust regression (Huber regressor). Models like GBM are naturally robust to heteroscedasticity.

**Q: Why is MAPE problematic for certain prediction tasks?**
A: MAPE = mean |error|/|actual| blows up when actual values are near zero (e.g., demand forecasting with some zero-demand days), making it meaningless. It also asymmetrically penalizes over-prediction vs under-prediction. Use SMAPE (symmetric MAPE) for near-zero values, or simply use MAE with a note about the target scale. For demand forecasting, WMAPE (weighted MAPE, weighted by actual volume) is more stable.

**Q: How do you evaluate a regression model when the target is log-transformed?**
A: Transform predictions back to original scale before computing metrics: exp(y_pred) → original scale. Compute RMSE and MAE in original scale for interpretability. Log-transformed predictions tend to underestimate large values (log compression), so check residuals at both extremes of the target range. Don't compare RMSE in log-space vs original-space across models — always use the same scale.

**Q: What additional diagnostics would you run beyond standard metrics?**
A: (1) Residual histogram — should be roughly Gaussian for OLS assumptions; (2) QQ plot — check normality of residuals; (3) Residuals vs feature values — detect non-linear relationships not captured by the model; (4) Cook's distance — identify influential outliers that disproportionately affect coefficients; (5) Autocorrelation of residuals (Durbin-Watson test) for time-series data — residual correlation indicates missing temporal structure.
""",

    "25-feature-engineering.md": """\
**Q: What is target encoding and why can it cause data leakage if done incorrectly?**
A: Target encoding replaces a categorical value with the mean of the target for that category (e.g., "city=NYC" becomes the mean income of NYC residents in training data). If computed on the full dataset before CV, the validation fold's target values influence the encoding — this is leakage. Correct approach: fit target encoding only on the training folds inside each CV split, or use leave-one-out encoding which excludes the current sample.

**Q: When would you use one-hot encoding vs ordinal encoding for categorical features?**
A: One-hot: for nominal categories with no order (color, city) and low cardinality (<20 values) — creates interpretable binary features. Ordinal: for ordered categories (education level, rating) or high-cardinality categories with tree-based models (trees handle ordinal efficiently). For high-cardinality nominal features (1000+ categories), use target encoding or embeddings — one-hot creates too many sparse features.

**Q: How do you handle a new category in production that wasn't in training data?**
A: This is the "unseen category" or "cold start" problem. Solutions: (1) handle_unknown='ignore' in OneHotEncoder — maps unseen to zero vector; (2) handle_unknown='infrequent_if_exist' — maps rare/unseen to a special "other" bucket; (3) mean encoding fallback to global mean for unseen categories; (4) pre-define an "OTHER" category during training for any low-frequency categories. Always test your encoding pipeline on data with new categories.

**Q: What's the difference between feature selection and dimensionality reduction?**
A: Feature selection keeps a subset of original features unchanged — results are interpretable (you know which features matter). Dimensionality reduction (PCA) creates new features as linear combinations — more compact but harder to interpret. Use feature selection when interpretability matters (regulated industries, feature cost analysis). Use PCA when you need the most compact representation and don't care about feature interpretability.

**Q: How do interaction features affect model complexity and when do you add them?**
A: Interaction features (x1·x2) explicitly capture non-linear effects that linear models can't model from x1 and x2 separately. They increase dimensionality (p features → p² interactions), which can cause overfitting without regularization. Add them when domain knowledge suggests a multiplicative relationship (e.g., price × quantity = revenue), or when EDA reveals different slopes for x1 across x2 values. Tree-based models capture interactions implicitly — interaction features mainly help linear models.

**Q: How would you handle a feature with 30% missing values?**
A: First, understand the missingness mechanism: MCAR (missing completely at random — impute), MAR (missing at random given other features — impute with model-based imputation), MNAR (missing not at random — the missing indicator itself is informative, add as feature). Options: mean/median imputation for MCAR, KNN or iterative imputer for MAR, or add a binary "was_missing" feature to preserve the information that the value was missing, then impute.
""",

    "26-hyperparameter-tuning.md": """\
**Q: What is the difference between hyperparameter tuning and model selection?**
A: Model selection chooses between fundamentally different algorithm types (SVM vs Random Forest vs GBM). Hyperparameter tuning optimizes the configuration of a chosen model type (max_depth, learning_rate). They are distinct tasks and should be separated: first do model selection on default hyperparameters, then tune the winner. Conflating them by tuning all models simultaneously wastes compute and can lead to overfitting the validation set.

**Q: Why is random search often better than grid search for large hyperparameter spaces?**
A: Grid search evaluates every combination — with 5 hyperparameters × 5 values each, that's 5^5=3125 evaluations. Random search samples randomly from the joint distribution and, critically, each evaluation is independent — if some hyperparameters are unimportant, random search doesn't waste evaluations on their grid points. With the same compute budget, random search typically finds better configurations than grid search for spaces with ≥3 hyperparameters.

**Q: How does Bayesian optimization improve on random search?**
A: Bayesian optimization builds a surrogate model (typically a Gaussian process) of the objective function, updated with each new evaluation. An acquisition function (Expected Improvement) guides the next evaluation toward regions likely to improve on the current best. This uses past evaluations to intelligently explore the space — exploiting promising regions while still exploring. It typically requires 5-10x fewer evaluations than random search to find good configurations.

**Q: How do you avoid overfitting the validation set during hyperparameter tuning?**
A: Use k-fold cross-validation for each configuration evaluation (not a single train/val split), and keep a completely separate test set that is only evaluated once at the very end. The more configurations you evaluate, the higher the chance of getting lucky on the validation set. Use nested CV to get an honest performance estimate that accounts for hyperparameter selection optimism.

**Q: What is early stopping in hyperparameter search (successive halving) and when does it help?**
A: Successive halving allocates a small budget (few training epochs or data fraction) to many configurations initially, then doubles the budget and keeps only the top fraction, iterating until one winner remains. It's useful when evaluating a full configuration is expensive — instead of fully training 1000 models, train all 1000 for 10 epochs, keep top 333, train for 30 epochs, keep top 111, etc. sklearn's HalvingRandomSearchCV implements this.

**Q: Which hyperparameters matter most to tune for gradient boosting?**
A: In rough priority order: (1) learning_rate + n_estimators (use early stopping to auto-tune n_estimators for a given LR); (2) max_depth (3-8 range, use lower for noisy data); (3) subsample and colsample_bytree for stochastic boosting (0.6-0.9); (4) min_child_weight / min_samples_leaf for regularization; (5) reg_lambda / reg_alpha for L2/L1 regularization. Start with LR=0.1, use early stopping, then tune depth.
""",

    "27-ensemble-methods.md": """\
**Q: Why is diversity important in ensemble methods?**
A: If all models make identical predictions, averaging them doesn't help — the ensemble error equals the individual error. Diversity (uncorrelated errors) is what makes ensembles powerful: when models make different mistakes, averaging cancels out errors. Random Forests create diversity via feature subsampling; boosting creates diversity via sequential error-correction. An ensemble of 3 diverse models often outperforms an ensemble of 100 correlated models.

**Q: When does stacking improve over simple averaging?**
A: Stacking learns the optimal weighting (and interactions) between base models, while averaging assigns equal weight. Stacking helps when base models have different strengths on different parts of the input space — the meta-learner learns to route predictions. Simple averaging is sufficient when models are similar in quality and errors. Stacking often provides 0.5-2% accuracy gains in competitions but adds complexity in production.

**Q: What is the key risk in stacking implementation?**
A: Training the meta-learner on base model predictions generated from the same training data causes data leakage — the base models have memorized the training labels, so their training predictions are too optimistic. The meta-learner then learns to trust overfitted base models. Fix: generate meta-features using out-of-fold cross-validation predictions — base models are never evaluated on their own training data.

**Q: When would you NOT use an ensemble?**
A: When inference latency is critical (k models = k× prediction time); when model interpretability is required (stacked ensemble is a black box even if components are interpretable); when the dataset is too small (ensembles need diverse errors to be beneficial, hard with limited data); when you need to deploy, monitor, and retrain a single model (operational complexity multiplies with ensemble size).

**Q: How does boosting reduce bias while bagging reduces variance?**
A: Bagging trains models independently on bootstrap samples and averages — averaging reduces variance (reduces sensitivity to training data noise) without affecting bias. Boosting trains models sequentially, each correcting the previous model's errors — each iteration focuses the model on hard examples, reducing the bias (systematic underfitting) of the combined model. Boosting uses very shallow weak learners (stumps) to avoid adding variance.

**Q: What is model blending vs stacking?**
A: Blending trains base models on the full training set, generates predictions on a held-out validation set, and trains the meta-learner on those predictions. Stacking uses cross-validation to generate meta-features, so all training data is used and overfitting risk is lower. Blending is simpler and faster but wastes data (the held-out set isn't used for base model training). In competitions, stacking is preferred; in production, blending's simplicity often wins.
""",

    "28-bayesian-inference.md": """\
**Q: What is the difference between a frequentist confidence interval and a Bayesian credible interval?**
A: A 95% frequentist confidence interval means: if you repeated the experiment many times, 95% of the constructed intervals would contain the true parameter. It says nothing about the probability that this specific interval contains the true value. A 95% Bayesian credible interval means: given the observed data and prior, the probability that the parameter falls in this interval is 95%. The credible interval is more intuitively interpretable.

**Q: When would you choose Bayesian inference over frequentist methods?**
A: Use Bayesian inference when: prior knowledge is meaningful and should be incorporated (rare disease prevalence, regulatory constraints); you need principled uncertainty quantification for decision-making (not just a point estimate); you have small datasets where regularization through priors matters; or you need sequential updating (updating beliefs as new data arrives, like online learning). Frequentist methods are simpler and sufficient when data is abundant.

**Q: What is MAP estimation and how does it relate to regularization?**
A: Maximum A Posteriori (MAP) estimation finds the mode of the posterior: θ_MAP = argmax p(θ|D) = argmax p(D|θ)p(θ). With a Gaussian prior p(θ) ~ N(0, 1/λ), MAP is equivalent to L2 regularized MLE (Ridge regression). With a Laplace prior, it's equivalent to L1 regularization. This reveals a deep connection: regularization is implicit Bayesian inference with specific priors — the prior is the regularizer.

**Q: What are conjugate priors and why are they useful?**
A: A prior is conjugate to a likelihood if the posterior belongs to the same distribution family as the prior. Example: Beta prior + Binomial likelihood → Beta posterior. Conjugate priors allow analytical (closed-form) posterior computation — no MCMC needed. Useful for simple models (coin flips, Poisson processes, Gaussian observations) and as building blocks for more complex models. Non-conjugate models require approximate inference (MCMC or variational inference).

**Q: What is MCMC and when do you need it instead of analytical solutions?**
A: Markov Chain Monte Carlo samples from the posterior distribution by constructing a Markov chain whose stationary distribution equals the posterior. It's needed when the posterior has no closed-form solution (non-conjugate priors, complex likelihoods, hierarchical models). Modern MCMC (NUTS sampler in PyMC, Stan) scales to hundreds of parameters. For very large datasets or models, variational inference (approximates posterior with a simpler distribution) is faster but less accurate.

**Q: How do you validate a Bayesian model?**
A: (1) Prior predictive check: sample from the prior and check if simulated data looks plausible before seeing observations; (2) Posterior predictive check: generate data from the posterior and compare its distribution to real data — discrepancies reveal model misspecification; (3) MCMC diagnostics: R-hat < 1.01 (chain convergence), effective sample size > 400, trace plots show mixing; (4) Sensitivity analysis: check if conclusions change substantially with different reasonable priors.
""",
}


def replace_qa_section(filepath: str, new_qa: str) -> bool:
    """
    Replace the content between '## Interview Q&A' and '## Best Practices'
    with the provided Q&A content.
    Returns True if replacement was made, False otherwise.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the Interview Q&A section and replace up to (but not including) Best Practices
    pattern = r"(## Interview Q&A\n\n).*?(\n## Best Practices)"
    replacement = r"\g<1>" + new_qa.rstrip() + r"\g<2>"

    new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)

    if count == 0:
        print(f"  WARNING: Pattern not found in {filepath}")
        return False

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    return True


def main():
    fixed = 0
    skipped = 0
    errors = 0

    for filename, qa_content in QA_CONTENT.items():
        filepath = os.path.join(CONCEPTS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"SKIP (not found): {filename}")
            skipped += 1
            continue

        print(f"Fixing: {filename}")
        success = replace_qa_section(filepath, qa_content)
        if success:
            fixed += 1
        else:
            errors += 1

    print(f"\nDone: {fixed} fixed, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
