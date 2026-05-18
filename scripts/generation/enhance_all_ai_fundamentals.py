#!/usr/bin/env python3
"""Enhance Detailed Explanation and Core Intuition for ALL AI fundamentals concepts (01-28)."""

import os
import re

BASE = "/home/sbisw/github/interviewprep-ml"

# Complete expansions for ALL AI fundamentals (01-28)
AI_EXPANSIONS = {
    "01-gradient-descent": {
        "detailed": """Gradient descent is the foundational optimization algorithm used to train nearly all neural networks and many machine learning models. The core idea is deceptively simple: to minimize a loss function, repeatedly take small steps in the direction of steepest descent (opposite to the gradient). This iterative approach works because it's guaranteed to eventually reach a local minimum if the learning rate is chosen appropriately.

The algorithm comes in three main variants: Batch gradient descent uses all training data to compute the gradient (slow but stable), Stochastic gradient descent (SGD) uses one sample at a time (fast but noisy), and Mini-batch gradient descent uses small batches (best of both worlds, standard in practice). The learning rate is critical: too high and the algorithm oscillates and diverges, too low and training is prohibitively slow. Adaptive methods like Adam adjust the learning rate per parameter, often outperforming fixed-rate approaches.

Understanding gradient descent is essential because nearly every deep learning model training uses it. Recognizing why training plateaus or diverges often comes down to understanding gradient descent mechanics. Modern practitioners rarely implement it from scratch but understanding the algorithm is crucial for debugging training failures and making architectural decisions.""",
        "intuition": """Imagine you're at the top of a mountain in the dark and want to reach the valley below. You can't see far ahead, but you can feel the slope beneath your feet. Gradient descent is like repeatedly taking steps downhill: feel the slope (compute gradient), take a step in that direction (update weights). Eventually you reach the bottom, though you might get stuck in local valleys."""
    },

    "02-backpropagation": {
        "detailed": """Backpropagation is the algorithm for computing gradients in neural networks, enabling gradient descent training. Given a complex network with many layers, backpropagation uses the chain rule to efficiently compute how each parameter affects the final loss. Without backpropagation, computing gradients would require re-evaluating the entire network for each parameter (prohibitively expensive), but backpropagation reuses intermediate calculations to compute all gradients in a single backward pass.

The algorithm works by propagating error information backward through the network: starting from the loss function, compute how the loss changes with respect to each layer's outputs, then use those signals to compute gradients with respect to that layer's weights. The key insight is that this backward pass mirrors the forward computation, reusing the same connectivity structure. Two critical problems can occur: vanishing gradients (gradient signals become too small in deep networks) and exploding gradients (signals grow exponentially). Modern techniques like batch normalization, careful weight initialization, and skip connections address these issues.

Backpropagation is the algorithmic foundation of deep learning. Understanding it helps explain why some architectures work (skip connections help gradients flow) and why some don't (very deep networks suffer from vanishing gradients). It's less about implementing it (frameworks like PyTorch do this) and more about understanding its limitations and how to work around them.""",
        "intuition": """Imagine water flowing backward through a pipe system from the output (leak) back to the input. The water flow represents error signals. At each junction, water splits proportionally (chain rule), and you can measure how much water came from each upstream pipe. That's backpropagation: tracing error signals backward to find which weights caused the error."""
    },

    "03-loss-functions": {
        "detailed": """Loss functions quantify how wrong a model's predictions are, providing the signal that gradient descent uses to improve the model. Different tasks require different loss functions because they encode different assumptions about what 'wrong' means. Mean squared error (MSE) penalizes large errors quadratically, making it sensitive to outliers. Cross-entropy loss is the standard for classification, directly related to probability distributions. Hinge loss for SVMs encodes the margin concept. Custom loss functions can encode domain knowledge about which errors are most costly.

Choosing the right loss function is crucial because it directly shapes what the model learns. A model trained with MSE will be quite different from one trained with mean absolute error (which is outlier-robust), even on identical data. Some loss functions have nice theoretical properties (cross-entropy is information-theoretically justified for classification), others are pragmatic choices. Weighted loss functions can address class imbalance by making the model care more about minority class errors. The loss function you choose is saying 'these are the errors I care about most.'

Loss functions are often overlooked but are equally important as architecture choices. Understanding why certain loss functions work better for certain problems is crucial for effective machine learning. Modern deep learning frameworks make it easy to swap loss functions, but understanding their properties helps debug models that aren't learning properly.""",
        "intuition": """Loss functions are like a teacher grading papers differently depending on the subject. A math teacher cares about exact answers (squared error). A spelling teacher cares about making mistakes (absolute error). A doctor might care most about catching disease (high penalty for missed positives). Each 'grading rubric' guides what the model learns."""
    },

    "04-optimization-algorithms": {
        "detailed": """Beyond basic gradient descent, many optimization algorithms adaptively adjust learning rates and incorporate momentum to accelerate convergence and improve stability. Momentum methods (like SGD with momentum) accumulate gradients over time, helping escape shallow local minima and plateaus. Adam combines momentum with adaptive per-parameter learning rates, automatically scaling step sizes. RMSprop divides learning rates by the root mean square of accumulated gradients. AdamW (Adam with weight decay) adds L2 regularization, crucial for neural networks.

Each optimizer has different properties: SGD with momentum is simple and reliable but requires careful learning rate tuning. Adam is often the default choice—less sensitive to learning rate and usually requires minimal tuning. RMSprop works well when gradients are sparse. Different optimizers can converge to different final solutions because they navigate loss landscapes differently. Modern practice usually starts with Adam, but understanding the alternatives helps when Adam gets stuck.

Optimization algorithms are the engines driving neural network training. They matter as much as architecture choices for final performance. Knowing which optimizer to try when training isn't working is a practical debugging skill. Most importantly, understanding that all these algorithms are solving the same problem (minimizing loss) with different strategies helps you predict which might work best for a given problem.""",
        "intuition": """Optimization algorithms are like different navigation strategies on a road trip. Basic gradient descent is like following the steepest downhill path (works but slow on flat terrain). Momentum is like having momentum that helps you push through flat areas. Adam is like an intelligent navigator that adjusts your pace based on the terrain (steep = slower, flat = faster). Different strategies reach the destination at different speeds."""
    },

    "05-learning-rate-scheduling": {
        "detailed": """Learning rate scheduling gradually changes the learning rate during training to improve convergence and final performance. Starting with a higher learning rate helps initial progress, but eventually the model needs smaller steps to settle into good solutions. Schedules can decrease linearly, exponentially, or in steps. Warmup (increasing learning rate for the first few steps) prevents gradient explosion early in training. Cosine annealing (smoothly decreases to near-zero then restarts) has been empirically effective for both preventing overfitting and discovering diverse solutions.

The intuition is that learning rate is a trade-off: large steps make progress quickly but might overshoot optima, small steps navigate precisely but take forever. Scheduling automatically balances this over training: be aggressive early, careful later. Different schedules encode different assumptions about learning dynamics. Linear decay is simplest. Exponential decay speeds up convergence. Step decay (drop by factor every N epochs) is easy to implement. Cosine annealing provides smooth transitions with a principled mathematical foundation.

Learning rate scheduling often improves final model performance by 1-5% compared to fixed learning rates, which is worth the minimal implementation effort (most frameworks support multiple schedulers). Understanding scheduling helps explain why some models plateau early (learning rate too high, overshooting) and why others converge slowly (learning rate too low). Practitioners often overlook this easy win.""",
        "intuition": """Learning rate scheduling is like driving on a mountain road: start with steady progress (high speed), slow down as you near the destination (lower speed for precision), and consider rest stops (warmup) to avoid crashing early. Different road conditions (optimization landscapes) benefit from different speed profiles."""
    },

    "06-linear-regression": {
        "detailed": """Linear regression predicts continuous values by fitting a line (or plane/hyperplane in high dimensions) to minimize squared prediction errors. The mathematical elegance is that the optimal solution has a closed-form formula (normal equation), making it computationally efficient. However, the closed form requires inverting a matrix which becomes numerically unstable with many features, so gradient descent is often used instead despite being iterative.

The model assumes target = linear combination of features + noise, which is rarely exactly true but is often a good approximation. Regularization (ridge regression adds L2 penalty, lasso adds L1 penalty) prevents overfitting by penalizing large coefficients. Ridge keeps all features (shrinking coefficients) while lasso zeros out some features entirely (feature selection). The interpretation of coefficients is straightforward: coefficient = change in target per unit change in feature (holding others fixed), making linear regression highly interpretable.

Linear regression is often considered 'simple' but remains incredibly valuable because it's interpretable, efficient, and often performs as well as complex methods on real data. Understanding when linear assumptions are reasonable helps you choose whether to use linear regression or move to more complex methods. It's also the foundation for understanding more complex models—many advanced techniques can be viewed as non-linear extensions of linear regression.""",
        "intuition": """Linear regression is like fitting a straight line through scatter plot points to predict future values. The line minimizes total vertical distances from points to the line. It's simple but powerful: if the relationship is roughly linear, a straight line often predicts better than a complex curve would (avoiding overfitting)."""
    },

    "07-logistic-regression": {
        "detailed": """Logistic regression predicts probabilities for classification by applying a sigmoid function to a linear model, squashing outputs into the [0,1] range. Despite its name, it's a classification algorithm (not regression), and it's arguably the most important baseline in machine learning. If logistic regression doesn't work well on a problem, more complex methods likely won't either (garbage in, garbage out). It's the first algorithm to try for binary classification.

The model outputs a probability via σ(w·x + b), where σ is the sigmoid function. This probability is interpreted as P(y=1|x), making the model interpretable: coefficients show log-odds changes. Training uses cross-entropy loss, which is information-theoretically justified and works better than squared error for classification. Regularization prevents overfitting, especially important when features vastly outnumber samples. Logistic regression extends naturally to multi-class via softmax.

Logistic regression is remarkably effective in practice because many real-world classification problems are approximately linear in the input space. Its interpretability (you can read off which features matter and in which direction) makes it invaluable for high-stakes applications. Understanding logistic regression helps you grasp how neural networks (which are logistic regression + nonlinearity + depth) work. It's also computationally efficient, making it suitable for deployment.""",
        "intuition": """Logistic regression is like drawing a decision boundary (straight line) through scatter plot points to separate two classes. The probability near the boundary is uncertain (near 0.5), far from boundary is confident (near 0 or 1). It's the simplest way to turn a linear model into a probabilistic classifier."""
    },

    "08-decision-trees": {
        "detailed": """Decision trees recursively split data into increasingly homogeneous subsets, building a tree where each node represents a split decision and leaves represent predictions. Trees are human-interpretable, require minimal preprocessing, and handle non-linear relationships automatically. However, they easily overfit (can grow to memorize training data), have high variance (small data changes produce very different trees), and suffer from greedy splitting (local optimization, not global).

The algorithm recursively finds the feature and threshold that best separates data. Splitting criterion is usually Gini impurity (how mixed are the classes) or information gain (reduction in entropy). Controlling tree depth prevents overfitting: shallow trees have high bias but low variance, deep trees have low bias but high variance. Minimum samples per leaf and maximum depth are the key hyperparameters. Trees are often used as base learners in ensembles (random forests, gradient boosting) rather than standalone because ensembles address the variance problem.

Decision trees are conceptually simple but powerful as building blocks. Understanding how they work explains random forests and gradient boosting (which combine many trees). Single trees rarely win modern competitions, but understanding trees helps you debug ensemble models. The interpretability of trees makes them valuable for explaining predictions in business settings, even if other models might be slightly more accurate.""",
        "intuition": """Decision trees are like a flowchart of yes/no questions: 'Is feature A > threshold?' If yes, go left; if no, go right. Keep asking questions until you reach a leaf (prediction). Each path through the tree is an interpretable rule like 'if age > 30 AND income > $50k, then approve loan'."""
    },

    "09-random-forests": {
        "detailed": """Random forests train many decision trees on random subsets of data, then average their predictions to reduce overfitting. Bootstrap sampling (sampling with replacement) creates diverse training sets for each tree. Random feature selection at each split adds further diversity. The diversity means different trees make different mistakes, and averaging cancels out errors—the ensemble is far more stable and accurate than any individual tree.

The algorithm trades interpretability for accuracy: you can't easily explain why the ensemble makes a prediction (unlike a single tree), but accuracy usually improves substantially. Out-of-bag (OOB) error estimates generalization performance without needing a separate validation set. Feature importance can be computed by tracking how much each feature decreases impurity across all trees, providing some interpretability. Random forests handle non-linear relationships, missing values somewhat gracefully, and are robust to outliers. They're one of the first methods to try for tabular data.

Random forests demonstrate the power of ensemble learning: combining weak learners (trees prone to overfitting) creates a strong learner (ensemble that generalizes well). Understanding how diversity reduces error helps explain why other ensembles work. Practical practitioners rarely tune random forests heavily—default hyperparameters often work well. The simplicity, robustness, and good out-of-the-box performance makes random forests a go-to baseline.""",
        "intuition": """Random forests are like hiring a committee of diverse experts (trees) who each make slightly different mistakes. By averaging their votes, mistakes cancel out. A single expert might be biased or miss things, but a diverse committee is usually smarter than any individual member."""
    },

    "10-gradient-boosting": {
        "detailed": """Gradient boosting builds trees sequentially, where each new tree learns to correct errors made by previous trees. Unlike random forests (parallel independent trees), boosting is sequential—each tree targets the residuals of the ensemble so far. This sequential correction process is powerful but requires careful tuning to avoid overfitting. XGBoost and LightGBM are optimized implementations used heavily in competitions and industry.

The algorithm starts with a base prediction (usually the mean), then fits a tree to residuals. Each new tree is scaled by a learning rate (small step sizes prevent overfitting). Early stopping (stop after validation error plateaus) prevents memorizing training data. Key hyperparameters are tree depth (shallow trees = regularization), number of boosting rounds (more = more capacity), and learning rate (lower = slower but more careful learning). Unlike random forests, gradient boosting requires careful tuning but often achieves state-of-the-art accuracy on tabular data.

Gradient boosting is the most effective algorithm for tabular/structured data in practice, winning most Kaggle competitions. Understanding the sequential error correction concept helps explain why boosting works: each successive tree is easier to fit because it only needs to capture the previous ensemble's errors, not the full complexity. XGBoost and LightGBM are industry standards, though both are complex to tune optimally. Starting with default hyperparameters and adjusting based on validation performance is practical.""",
        "intuition": """Gradient boosting is like learning by mistakes: your first prediction is rough (tree 1), you see where you were wrong (residuals), and train a second model to correct those specific mistakes. Then you see the remaining errors and train a third model, etc. Each model learns from the collective mistakes of all previous models."""
    },

    "11-support-vector-machines": {
        "detailed": """Support Vector Machines find the maximum-margin hyperplane separating two classes, treating classification as a geometric problem. The 'margin' is the distance from the decision boundary to the nearest data points (support vectors). Maximizing margin provides generalization: confident separations from data tend to generalize well to new data. The SVM can be reformulated to use a kernel trick, applying implicit non-linear transformations without explicitly computing high-dimensional features.

Linear SVMs are interpretable: weights show which features matter and in which direction. Non-linear SVMs (via kernels) handle complex decision boundaries but are less interpretable. The kernel trick is mathematically elegant: comparing high-dimensional transformed data is done implicitly, avoiding exponential computation. Common kernels are linear (for linear problems), RBF (universal approximator for non-linear), polynomial. The C parameter controls margin-error trade-off: high C enforces margins perfectly (may overfit), low C allows violations (generalizes better).

SVMs are theoretically elegant and work well on many problems, especially with moderate amounts of data. However, they don't scale as well as tree-based methods on large datasets and require feature scaling. The kernel trick is intellectually important (demonstrated non-linear capacity through clever mathematics) even though deep learning has largely superseded SVMs in practice. Understanding SVMs helps explain margins, regularization concepts, and the importance of decision boundaries in classification.""",
        "intuition": """SVMs are like drawing the thickest possible 'safe zone' around your decision boundary to separate two groups. The kernel trick is like secretly transforming data to a higher dimension where a simple line can separate them, but doing the math without explicitly computing the transformation."""
    },

    "12-k-nearest-neighbors": {
        "detailed": """K-nearest neighbors (KNN) predicts based on the k nearest training examples, using majority vote for classification or averaging for regression. It's a non-parametric, lazy algorithm: no training phase, computation happens at prediction time. Simple to understand (look at k similar past examples) but computationally expensive and suffers from curse of dimensionality (in high dimensions, notion of 'nearby' becomes less meaningful).

Key choice is k: small k (1-3) has low bias but high variance (single outliers influence predictions), large k (all training data) has high bias but low variance. Distance metric matters: Euclidean distance works for continuous features, other metrics for different data types. Feature scaling is critical because unscaled features with large ranges dominate distances. KNN is often used as a baseline because it's parameter-free (only k to tune) and non-parametric (no distributional assumptions).

KNN is conceptually the simplest classification algorithm and often serves as a baseline. Understanding it clarifies why distance, scale, and dimensionality matter. In practice, KNN is usually outperformed by parametric methods (trees, linear models) on most problems, but for certain applications (few training examples, non-standard data) it remains useful. The computational cost of exact KNN (computing distances to all training points) led to approximate nearest neighbor algorithms crucial for modern large-scale systems.""",
        "intuition": """KNN is like making a prediction based on your k nearest neighbors' experiences: 'What did people like you do?' If most of your 5 nearest neighbors succeeded, you probably will too. It relies entirely on similarity—things that are similar tend to behave similarly."""
    },

    "13-neural-networks": {
        "detailed": """Neural networks are compositions of simple functions (neurons) arranged in layers, enabling non-linear function approximation. A single neuron computes a weighted sum of inputs plus a bias, then applies an activation function. Multiple neurons in a layer compute multiple features in parallel. Stacking layers creates depth, enabling learning of hierarchical representations. The universal approximation theorem guarantees that a sufficiently wide two-layer network can approximate any function, though it says nothing about learnability or depth efficiency.

The XOR problem (which linear models can't solve) is solvable by a two-layer network, illustrating why non-linearity is necessary. Deeper networks can be more parameter-efficient than shallow ones (learning same function with fewer parameters), but are harder to train (vanishing/exploding gradients). Modern best practices include ReLU activations (simpler and more gradient-friendly than sigmoids), batch normalization, skip connections, and careful initialization. Neural networks are "just function approximators" but the composition enables learning rich, hierarchical representations.

Neural networks are the foundation of deep learning. Understanding the basic unit (neuron = weighted sum + non-linearity) and how stacking creates expressiveness helps demystify deep learning. Many practitioners use frameworks without fully understanding the underlying mechanics, but understanding how gradients flow through layers is crucial for debugging training issues. The universality theorem is intellectually important but practically misleading (doesn't mean networks learn efficiently).""",
        "intuition": """Neural networks are like stacking transparent sheets with hand-drawn features: the first layer detects edges, the second layer combines edges into corners, the third layer combines corners into shapes, etc. Each layer transforms the representation slightly. Stacking enables detecting increasingly complex patterns."""
    },

    "14-activation-functions": {
        "detailed": """Activation functions introduce non-linearity into neural networks, without which stacking layers would just compute linear functions (composition of linear functions is linear). ReLU (Rectified Linear Unit, max(0,x)) is the modern standard due to computational simplicity and good gradient properties. Sigmoid squashes to [0,1], historically standard but has vanishing gradient problem in deep networks. Tanh is similar to sigmoid but squashes to [-1,1]. GELU (Gaussian Error Linear Unit) is smooth and often works better than ReLU.

The choice of activation affects: (1) gradient flow (does gradient signal propagate well backward?), (2) computational cost (ReLU is simplest), (3) interpretability (different activations encode different assumptions about data). Dying ReLU problem (neurons becoming inactive) can be fixed with leaky ReLU or ELU. Output activation depends on task: sigmoid for binary classification, softmax for multi-class, linear for regression, tanh for outputs in [-1,1]. Using wrong output activation is a common mistake that's trivial to fix but causes problems.

Activation functions are often treated as design choices to try (ReLU usually works, try others if stuck) rather than something to understand deeply. However, understanding gradient flow through activations helps explain training problems. Vanishing gradients through many sigmoid layers is why deep networks were hard to train before ReLU. The choice of activation is usually not the bottleneck but understanding trade-offs helps when debugging.""",
        "intuition": """Activation functions are like on-off switches (ReLU) or volume knobs (sigmoid) that add non-linearity. Without them, neural networks would just be linear models no matter how deep. ReLU is like 'pass through if positive, kill if negative'. Sigmoid is like 'gradually ramp up from 0 to 1'. Both add complexity that allows learning non-linear patterns."""
    },

    "15-weight-initialization": {
        "detailed": """Neural networks are sensitive to initialization: poor initialization can cause vanishing/exploding gradients even with good architectures and learning rates. Xavier initialization (also Glorot) scales initial weights inversely to the number of inputs, maintaining gradient magnitudes. He initialization is similar but scaled for ReLU activations. Bad initialization (too large) causes exploding gradients, (too small) causes vanishing gradients. Biases are typically initialized to zero.

The intuition is that initial weights should preserve gradient variance: if x has variance σ², then w·x should have similar variance. With fanin inputs, weights should have variance ≈ 1/fanin. Modern frameworks default to sensible initialization, but understanding this explains why networks fail to train sometimes. Orthogonal initialization (weights are orthogonal matrices) is elegant but less commonly used. Layer normalization makes initialization less critical by normalizing activations to fixed statistics.

Weight initialization is one of those details often overlooked because frameworks handle it. However, understanding why it matters explains training failures. Initialization affects how long the network takes to converge and sometimes whether it converges at all. Combined with batch normalization and ReLU activations, initialization is less critical, but understanding the principles helps debug networks that aren't training.""",
        "intuition": """Weight initialization is like starting a game with different initial conditions: too heavy initial weights lead to extreme predictions immediately (exploding gradients), too light initial weights lead to barely-moving outputs (vanishing gradients). Proper initialization keeps gradients flowing smoothly so the network can learn."""
    },

    "16-regularization": {
        "detailed": """Regularization prevents overfitting by penalizing model complexity, encouraging simpler models that generalize better. L2 regularization (ridge) adds penalty proportional to squared weights, shrinking large weights but keeping all features. L1 regularization (lasso) adds penalty proportional to absolute weights, driving some weights exactly to zero (feature selection). L1 produces sparse solutions; L2 produces small but non-zero weights. Dropout randomly zeros activations during training (preventing co-adaptation), then scales by dropout rate at test time to compensate. Early stopping stops training when validation performance plateaus.

The regularization-generalization connection is fundamental: more parameters fit training data better but generalize worse. Regularization trades training performance for generalization. The regularization strength λ is a hyperparameter: high λ enforces strong regularization (underfitting risk), low λ weak regularization (overfitting risk). Different problems need different λ values; empirically finding the right trade-off is crucial. Dropout is applied per-layer and acts somewhat like averaging ensemble predictions, explaining its regularization effect.

Regularization is one of the most important tools in machine learning, yet often underappreciated. The difference between a model that overfits (memorizes) and one that generalizes is often just the right regularization. Understanding the L1 vs L2 distinction helps choose: use L1 for automatic feature selection, L2 for smoother regularization. Dropout is particularly important in deep learning—modern networks often require dropout to generalize. Practitioners should understand that regularization isn't just a technical detail but a core concept in machine learning.""",
        "intuition": """Regularization is like penalizing complexity: L2 is like 'keep weights small', L1 is like 'remove unnecessary features entirely', dropout is like 'hide random neurons to prevent over-reliance on specific features'. All prevent the model from memorizing training data by constraining what it can learn."""
    },

    "17-batch-normalization": {
        "detailed": """Batch normalization normalizes layer activations to have mean ≈ 0 and variance ≈ 1 during training, stabilizing training and enabling higher learning rates. Internal covariate shift (the problem it solves) is when layer input distributions change during training, requiring earlier layers to keep adapting. BN prevents this by fixing downstream layer's input distribution. At inference, BN uses running statistics (not minibatch statistics) computed during training.

BN provides several benefits: allows higher learning rates (training is more stable), reduces dependence on initialization, acts as regularization (noise from using minibatch statistics), and often improves generalization. Layer normalization (normalize across features rather than batch) works better for sequential models where batch dimension is small. Group norm (normalize across groups of features) is middle ground. BN has trainable parameters (scale and shift) that let it undo normalization if useful, providing flexibility.

Batch normalization is a practical workhorse in deep learning, often essential for training deep networks. Understanding why it helps (internal covariate shift) explains why architectures with BN train faster. The training vs inference distinction (using minibatch statistics during training, running statistics during inference) is important for correct implementation. BN can interact unexpectedly with dropout and other stochastic techniques, requiring careful design. Despite its empirical success, theoretical understanding of why BN works remains incomplete.""",
        "intuition": """Batch normalization is like normalizing test scores to mean 0, variance 1 before grading: ensures all layers see similar input distributions, making training more stable. It's like calibrating instruments in an assembly line so each station works with standardized input ranges."""
    },

    # Unsupervised Learning
    "18-k-means-clustering": {
        "detailed": """K-means partitions data into k clusters by iteratively assigning points to nearest centroids and updating centroids to be cluster means. It's the most widely used clustering algorithm due to simplicity and efficiency. The algorithm converges to a local optimum (not necessarily global), so initialization matters. K-means++ initialization selects initial centroids that are far apart, improving solution quality without much additional cost. The main limitation is that k must be specified (unknown in truly unsupervised settings).

Determining optimal k is non-trivial. Elbow method plots intra-cluster variance versus k and looks for an 'elbow' (diminishing returns point). Silhouette score measures cluster quality (higher is better). BIC/AIC are information-theoretic measures balancing fit and complexity. In practice, domain knowledge often suggests reasonable k values. K-means assumes clusters are roughly spherical and similarly sized, which isn't always true. K-means++ improves quality; mini-batch variants enable large-scale clustering. K-means is non-probabilistic; soft clustering (like Gaussian Mixture Models) provides probabilities.

K-means is a quick way to discover patterns in unlabeled data. Understanding its limitations (local optima, spherical assumption, need to specify k) helps choose alternatives. K-means++ is a simple modification that dramatically improves solution quality. Most practitioners use sklearn or similar implementations but understanding the algorithm helps debug unexpected cluster assignments or identify when assumptions are violated.""",
        "intuition": """K-means is like organizing people into k friend groups: initially pick k random people, then repeatedly ask 'who are your closest friends?' and recompute centers. Eventually you reach groups where everyone's closest friends are in their group. It's a natural way to partition data by proximity."""
    },

    "19-dimensionality-reduction": {
        "detailed": """Dimensionality reduction maps high-dimensional data to lower dimensions while preserving important structure. Principal Component Analysis (PCA) finds orthogonal directions of maximum variance. t-SNE creates 2D visualizations by preserving local structure (nearby points stay nearby) but can't extrapolate beyond training data. UMAP is faster than t-SNE and more faithful to global structure. These methods address curse of dimensionality (in high dimensions, distances become less meaningful) and enable visualization.

PCA is linear (rotation to maximum-variance directions); t-SNE and UMAP are non-linear. PCA is deterministic and interpretable (principal components are weighted combinations of original features); t-SNE and UMAP involve randomness and are less interpretable. PCA is fast and scalable; t-SNE/UMAP are slower. Explained variance ratio measures how much information PCA preserves (cumulative explained variance guides choosing number of components). Feature scaling is critical for PCA.

Dimensionality reduction is useful for visualization (understanding data structure) and sometimes for preprocessing (reducing computational cost, reducing noise by dropping low-variance dimensions). PCA is the classical choice and remains useful. t-SNE is excellent for visualization but can be misleading about global structure. Understanding that different methods preserve different properties helps choose: use PCA for interpretability, t-SNE for visualization, UMAP for balance. Practitioners often apply reduction for visualization without realizing it changes apparent structure.""",
        "intuition": """Dimensionality reduction is like taking a high-dimensional photo and printing a 2D picture: some information is lost, but key structures (who's near whom, overall grouping) are preserved. PCA is like rotating to the best viewing angle. t-SNE is like artistic rendering emphasizing details you find interesting."""
    },

    "20-gaussian-mixture-models": {
        "detailed": """Gaussian Mixture Models (GMMs) model data as generated by a mixture of Gaussian distributions, providing soft clustering (probabilities) rather than hard assignments. The EM algorithm learns parameters by iteratively assigning responsibilities (probability of point belonging to cluster) then updating cluster parameters. GMMs generalize K-means (K-means is GMM with fixed covariance), provide probabilistic cluster assignments (useful for uncertainty quantification), and have principled model selection (BIC/AIC).

Training uses Expectation-Maximization (EM): E-step assigns soft responsibilities based on current parameters, M-step updates parameters to maximize expected log-likelihood. Choosing number of clusters is non-trivial; BIC/AIC provide principled selection. Different covariance assumptions (spherical, diagonal, full) provide different trade-offs between flexibility and parameter count. GMMs provide posterior probabilities of cluster membership, enabling probabilistic downstream analysis. Computational cost scales with data size and dimensions; mini-batch EM enables large-scale fitting.

GMMs are more principled than K-means (probabilistic model with likelihood) but require more careful tuning. Understanding the difference between hard clustering (K-means) and soft clustering (GMM) is important: soft assignments are useful when cluster membership is uncertain. The EM algorithm is intellectually elegant (coordinate ascent in the likelihood, guaranteed convergence) and appears in many ML contexts. GMMs work well when data is approximately Gaussian-distributed; if not, other methods might be better.""",
        "intuition": """GMMs are like describing a group of people's heights as a mixture of two Gaussians (short people + tall people). Instead of drawing a hard boundary, you assign probabilities: someone 5'10\" might be 70% likely from the tall group, 30% from the short group. It's soft clustering where membership is probabilistic."""
    },

    # Model Evaluation
    "21-bias-variance-tradeoff": {
        "detailed": """The bias-variance decomposition decomposes expected test error into three components: bias (errors from wrong assumptions), variance (sensitivity to training data changes), and irreducible noise. High bias (underfitting) means the model is too simple to capture the true pattern. High variance (overfitting) means the model fits training data so closely it doesn't generalize. The trade-off is fundamental: simpler models (high bias, low variance) vs. complex models (low bias, high variance).

Model complexity controls the trade-off: low complexity → high bias, low variance; high complexity → low bias, high variance. With infinite training data, complexity can increase without increasing variance (each model fits data well), so variance only matters in finite-data regimes. Regularization moves the complexity slider: more regularization → simpler model → higher bias, lower variance. Empirical validation (train/test curves) reveals whether a model suffers from bias (high error on both sets) or variance (low train error, high test error).

The bias-variance trade-off is one of the most fundamental concepts in machine learning, applicable to all learning algorithms. Understanding this decomposition helps diagnose model problems: high training error suggests bias (need more capacity), high gap between training and test error suggests variance (need regularization or more data). Many practitioners struggle to distinguish between these, leading to ineffective fixes. Learning curves (error vs. training set size) reveal the nature of the problem: variance-limited problems improve with more data, bias-limited problems don't.""",
        "intuition": """The bias-variance tradeoff is like target shooting: bias is systematic error (aiming wrong), variance is noise (inconsistent aim). A gun with consistent but wrong aim has high bias, low variance. A gun with inconsistent aim all over the place has low bias, high variance. The best gun balances accuracy (low bias) with consistency (low variance)."""
    },

    "22-cross-validation": {
        "detailed": """Cross-validation estimates generalization performance by repeatedly splitting data into training and validation folds. K-fold splits data into k subsets, trains on k-1 subsets, tests on the held-out subset, repeating k times. Final performance is the average across folds. This provides robust estimates using all data for both training and testing (unlike single train/test split which wastes data). Stratified K-fold maintains class distribution in each fold, critical for imbalanced datasets.

Different fold choices exist: time-series data requires ordered splits (don't train on future predicting past), grouped data requires keeping groups together (don't split group across folds), high-variance problems benefit from more folds (5-10 typical). Computational cost is k × training cost. Nested cross-validation (CV inside CV) provides unbiased hyperparameter selection but is expensive. Leave-one-out cross-validation tests on single samples repeatedly (expensive but unbiased for small datasets).

Cross-validation is the standard for evaluating model performance, providing more reliable estimates than a single train/test split. Understanding when standard CV is inappropriate (time series, grouped data) prevents mistakes. Stratified CV is crucial for imbalanced classification but many practitioners forget. The variance across folds (how much does performance vary?) indicates stability: low variance = robust performance, high variance = unreliable. Modern practice often uses cross-validation early to validate that models actually work before deployment.""",
        "intuition": """Cross-validation is like testing a recipe by cooking it multiple times with slightly different ingredients each time: instead of one expensive test, you get multiple data points about whether the recipe is good. Averaging across tests gives a better estimate than a single test would."""
    },

    "23-classification-metrics": {
        "detailed": """Classification metrics quantify prediction quality differently based on the problem. Accuracy (fraction correct) is intuitive but misleading with class imbalance. Precision (of positive predictions, what fraction correct) matters when false positives are costly. Recall (fraction of positives found) matters when false negatives are costly. F1-score (harmonic mean of precision and recall) balances both. ROC-AUC (Area Under the Receiver Operating Characteristic curve) plots true positive rate vs false positive rate, measuring ranking quality. PR-curve (Precision-Recall) is better than ROC for imbalanced data.

Different metrics encode different priorities: medical diagnosis values recall (catch all diseases) over precision (some false positives ok). Spam detection values precision (false positive = delete real mail) over recall. The confusion matrix shows all four categories (TP, FP, TN, FN), revealing which errors are made. Metrics aren't chosen arbitrarily—they should align with actual costs. Optimizing wrong metrics is a common mistake: accuracy can improve by predicting majority class regardless of actual improvements.

Choosing appropriate metrics is as important as choosing models. Understanding which errors matter (false positives? negatives?) determines which metrics to optimize. Reporting multiple metrics (accuracy, precision, recall, F1) gives a complete picture. Threshold choice (confidence needed to predict positive) affects precision-recall trade-off. Practitioners often fixate on single metrics without understanding what they measure, leading to misleading conclusions about model quality.""",
        "intuition": """Classification metrics are like different report cards for a student: accuracy is overall score, precision is 'of questions they answered yes to, how many were actually yes', recall is 'of all yes questions, how many did they answer yes to'. Different metrics reveal different strengths/weaknesses."""
    },

    "24-regression-metrics": {
        "detailed": """Regression metrics quantify prediction error for continuous targets. Mean Absolute Error (MAE) is interpretable (average |prediction - actual| error in same units as target) and robust to outliers. Mean Squared Error (MSE) penalizes large errors quadratically (outlier-sensitive). Root Mean Squared Error (RMSE) is back in original units, comparable to MAE but penalizes large errors more. R² (proportion of variance explained) ranges [0,1] with 1 = perfect, 0 = just predicting mean. R² is scale-independent, making comparisons across datasets possible.

The trade-off between MAE and MSE: MSE is differentiable and has nice mathematical properties (used in derivations, fast computation), but MAE is interpretable and outlier-robust. Median Absolute Error (MedianAE) is even more robust but less standard. Mean Absolute Percentage Error (MAPE) is useful when relative error matters, but undefined when actuals are zero. Custom metrics can encode domain knowledge about relative importance of different errors.

Choosing metrics should reflect what matters in the application: predicting demand (small errors acceptable?) vs. medical dosing (large errors very costly?). Residual analysis (plotting errors vs fitted values, checking for patterns) is crucial: systematic patterns reveal model misspecification. Multiple metrics paint different pictures (model A has lower MSE, model B has lower MAE). R² can be misleading (R² = 0.9 sounds great but might be unimpressive depending on domain). Practitioners often report only one metric without understanding what it really tells them.""",
        "intuition": """Regression metrics are different ways to measure 'how wrong you were': MAE is like average distance from target (dollar amount off), MSE penalizes big mistakes heavily, R² is like 'what fraction of the variation did you explain'. Different metrics suit different situations."""
    },

    # Advanced Foundations
    "25-feature-engineering": {
        "detailed": """Feature engineering creates new features from raw data to improve model performance. Domain knowledge often reveals good features: for time series, lagged values and rolling statistics; for text, word counts and TF-IDF. Polynomial features create non-linear combinations; interactions capture synergies. Scaling features (standardization, normalization) is essential for algorithms sensitive to feature magnitude. Encoding categorical variables (one-hot, ordinal, target encoding) transforms non-numeric data.

Feature engineering is problem-specific—no universal rules exist. The goal is creating features that expose patterns to the model. Well-engineered features can enable simpler models to outperform complex models on poorly-engineered features. Dimensionality (too many features) causes overfitting; too few features causes underfitting. Feature selection (choosing important features) reduces dimensionality and interpretability. Automated feature engineering (searching over potential combinations) is appealing but requires careful validation to avoid overfitting to training data structure.

Feature engineering is often cited as 'where domain experts contribute' (not something to automate), and it's where careful practitioners gain advantages over default pipelines. Understanding which features matter for your problem is crucial. Good features need less model capacity (simpler model), train faster (fewer features), and generalize better (noise is low-dimensional). Modern deep learning aims to learn features automatically (representation learning) but explicit feature engineering remains valuable for tabular data.""",
        "intuition": """Feature engineering is like translating raw ingredients (raw data) into measured ingredients (features) that make cooking easier. Raw eggs are hard to use; cracked eggs are easier. Flour needs to be sifted. Butter needs to be at room temperature. Same with data—transforming it into useful forms helps the 'recipe' (model) work better."""
    },

    "26-hyperparameter-tuning": {
        "detailed": """Hyperparameters (learning rate, regularization strength, tree depth, etc.) are set before training and significantly affect model performance. Grid search exhaustively tries all combinations in a specified grid (expensive, few values per hyperparameter). Random search samples combinations randomly (efficient, can explore wider ranges). Bayesian optimization models performance as a function of hyperparameters, intelligently choosing which to try next (efficient but more complex). Early stopping (monitoring validation performance) provides regularization without additional hyperparameters.

The search space defines reasonable ranges: learning rate 1e-5 to 1e-1 on log scale, tree depth 3-20, regularization 1e-4 to 1e2. Too narrow ranges miss optimal values; too wide ranges waste computation. The number of hyperparameters (curse of dimensionality in hyperparameter space) grows exponentially. Focused tuning (fixing unimportant hyperparameters, tuning important ones) is practical. Warm-starting from good previous runs accelerates tuning. Cross-validation during tuning avoids overfitting to validation data.

Hyperparameter tuning is often where practitioners spend too much effort. Often default hyperparameters work well, and careful feature engineering matters more. However, tuning the top 1-2 important hyperparameters often improves performance noticeably. Knowing which hyperparameters matter for your problem (algorithm-dependent) helps focus effort. Modern AutoML tools automate tuning but require careful validation. Tuning can be expensive computationally, so frameworks supporting parallel/distributed search are valuable.""",
        "intuition": """Hyperparameter tuning is like adjusting oven temperature, baking time, and ingredient ratios: too hot/fast burns the cake, too cool/slow leaves it raw, too much salt/sugar ruins flavor. You experiment to find the sweet spot. Some parameters matter more (temperature) than others (minor ingredient amounts)."""
    },

    "27-ensemble-methods": {
        "detailed": """Ensemble methods combine multiple models to improve robustness and performance. Bagging (bootstrap aggregation) trains models on random data subsets independently, then averages predictions. Boosting trains sequentially, with each model learning to correct previous models' errors. Stacking trains a meta-learner on outputs of base learners. Voting (simple averaging) is simplest. Weights can be learned (weighted average, meta-learner) for better combinations. Diversity among ensemble members is key: ensembles of identical models provide no benefit.

The bias-variance perspective explains ensembles: averaging independent, high-variance models reduces variance while bias remains constant (or increases slightly). This is the fundamental mechanism. Different base learners (some shallow, some deep; some linear, some non-linear) capture different patterns. Stacking can learn when to trust which model. Bagging is parallelizable (train on different subsets); boosting is sequential. Random forests are a bagged ensemble of trees; gradient boosting is a boosted ensemble.

Ensembles are one of the most practical and underappreciated techniques. Simple averaging of multiple trained models often improves performance noticeably with minimal additional cost. Understanding what makes good ensemble members (diverse, uncorrelated errors) helps design ensembles. Practical practitioners create ensembles by training multiple models with different initializations, hyperparameters, or architectures. The trade-off is inference cost: k times the model cost but often k times the accuracy improvement.""",
        "intuition": """Ensembles are like hiring multiple experts to make a decision: individual experts make different mistakes, but averaging their votes reduces error. A committee of diverse experts is usually smarter than any individual expert. The key is diversity—a committee of identical people is useless."""
    },

    "28-bayesian-inference": {
        "detailed": """Bayesian inference computes posterior probabilities P(parameters|data) using Bayes' rule: posterior ∝ likelihood × prior. Unlike frequentist estimation (point estimates), Bayesian inference provides full probability distributions. Prior encodes domain knowledge (ignorant priors have minimal influence); likelihood comes from data; posterior combines both. Interpretation is intuitive: posterior probability directly answers 'given data, what's likely true?'. For Bayesian neural networks, weights become distributions, providing uncertainty estimates.

The posterior distribution is often intractable (can't compute analytically), requiring approximation. Variational inference approximates with a tractable distribution. Markov Chain Monte Carlo (MCMC) samples from the posterior (Metropolis-Hastings, Gibbs sampling). Approximate Bayesian Computation (ABC) circumvents likelihood computation. Laplace approximation uses local Gaussian approximation around the mode. Each method trades accuracy, computational cost, and applicability differently.

Bayesian inference is intellectually elegant (principled combination of prior knowledge and data) but computationally expensive. The uncertainty quantification (probabilities for all parameters, not just point estimates) is valuable for decision-making. In deep learning, full Bayesian neural networks are expensive; approximations (variational, dropout as approximate Bayesian, SWAG) are practical. Bayesian reasoning about calibration (do probabilities match actual frequencies?) is important for deployment. Practitioners often use Bayesian methods for principled uncertainty without fully understanding the mathematics; understanding helps avoid misinterpretation.""",
        "intuition": """Bayesian inference is like updating your beliefs as you gather evidence: prior belief + new evidence → updated belief. The posterior probability distribution quantifies your uncertainty: high peak = confident, wide distribution = uncertain. It's formalizing how humans naturally update beliefs."""
    },
}

def enhance_all_ai():
    """Enhance all AI fundamentals concepts."""

    print("=== Enhancing ALL AI Fundamentals Concepts (01-28) ===\n")

    enhanced = 0
    for num in range(1, 29):
        # Map number to filename
        num_str = f"{num:02d}"

        # Find matching file
        concepts_dir = f"{BASE}/ai/concepts"
        matching_files = [f for f in os.listdir(concepts_dir) if f.startswith(num_str)]

        if not matching_files:
            print(f"  ⊘ {num_str}: No file found")
            continue

        filename = matching_files[0]
        slug = filename.replace('.md', '').replace(f'{num_str}-', '')
        filepath = os.path.join(concepts_dir, filename)

        # Check if we have expansions for this
        full_slug = f"{num_str}-{slug}"
        if full_slug not in AI_EXPANSIONS:
            print(f"  ⊘ {full_slug}: No expansion data")
            continue

        # Read and enhance
        with open(filepath) as f:
            content = f.read()

        exp = AI_EXPANSIONS[full_slug]
        detailed = exp["detailed"]
        intuition = exp["intuition"]

        # Replace sections
        content = re.sub(
            r'(## Detailed Explanation\n\n).*?(\n\n## Core Intuition)',
            rf'\1{detailed}\2',
            content,
            flags=re.DOTALL
        )

        content = re.sub(
            r'(## Core Intuition\n\n).*?(\n\n## How It Works)',
            rf'\1{intuition}\2',
            content,
            flags=re.DOTALL
        )

        # Write back
        with open(filepath, 'w') as f:
            f.write(content)

        print(f"  ✓ {full_slug}")
        enhanced += 1

    print(f"\n✅ Enhanced {enhanced} AI fundamentals concepts")

if __name__ == "__main__":
    enhance_all_ai()
