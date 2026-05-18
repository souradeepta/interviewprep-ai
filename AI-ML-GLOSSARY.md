# Comprehensive AI/ML/LLM Glossary

*A complete reference guide for terms, concepts, and abbreviations used throughout the AI Fundamentals, LLM, and Agentic AI curricula.*

---

## A

**Accuracy** - Fraction of correct predictions out of all predictions. Misleading for imbalanced datasets.
- Formula: (TP + TN) / (TP + TN + FP + FN)
- Best for: Balanced classification problems

**Adam** - Adaptive Moment Estimation optimizer. Combines momentum with adaptive per-parameter learning rates.
- Default choice for most deep learning
- Less sensitive to learning rate than SGD

**Adapter** - Parameter-efficient fine-tuning method. Small modules added between layers, frozen base model.
- Parameters: +0.5-2% relative to base model
- Efficient alternative to LoRA

**Advantage Function** - A(s,a) = Q(s,a) - V(s). Measures how much better action a is than average in state s.
- Used in Actor-Critic methods
- Reduces variance in policy gradient estimation

**Agent** - Autonomous entity that perceives environment and takes actions to maximize reward.
- Components: sensors, decision logic, actuators
- Examples: robots, game AI, conversational assistants

**Attention Mechanism** - Neural network component that learns to weight different inputs by importance.
- Query (Q), Key (K), Value (V) matrices
- Foundation of Transformers

**Autoencoder** - Neural network that compresses input into latent representation, then reconstructs.
- Uses: Dimensionality reduction, anomaly detection, data augmentation
- Variants: Variational (VAE), Denoising (DAE)

**Autoregressive** - Model predicts next token given previous tokens. Generates sequences one token at a time.
- Examples: GPT, language modeling
- Trade-off: Slow generation but stable training

---

## B

**Backpropagation** - Algorithm for computing gradients through neural networks using chain rule.
- Enables training deep networks
- Cost: ~2x forward pass time

**Batch** - Group of samples processed together.
- Typical: 32-256 samples
- Trade-off: Larger batch = smoother gradients but more memory

**Batch Normalization** - Normalize layer activations to mean 0, variance 1.
- Stabilizes training, allows higher learning rates
- Interaction: Train vs inference use different statistics

**Bayesian Inference** - Probability-based inference: P(θ|data) ∝ P(data|θ) * P(θ).
- Incorporates prior knowledge and uncertainty
- Trade-off: More principled but computationally expensive

**Beam Search** - Decoding algorithm that keeps top-k candidates at each step.
- Width k controls diversity vs speed
- Used in sequence-to-sequence models

**BPE (Byte Pair Encoding)** - Subword tokenization method. Merges frequent character pairs iteratively.
- Vocabulary size ~30k-50k typical
- Balances word-level and character-level benefits

---

## C

**Chain of Thought** - Prompting technique that asks model to show step-by-step reasoning.
- Improves accuracy on complex tasks
- Works especially well with larger models

**Classification** - Task of predicting discrete category for each sample.
- Binary: 2 classes (yes/no, fraud/legit)
- Multi-class: 3+ classes
- Multi-label: Sample can have multiple labels

**CNN (Convolutional Neural Network)** - Architecture specialized for spatial data (images).
- Uses convolution operations to detect local patterns
- Fewer parameters than fully connected networks

**Contrastive Learning** - Training by learning to recognize similar pairs and distinguish dissimilar pairs.
- Used in: embedding learning, self-supervised learning
- Examples: SimCLR, CLIP

**Convergence** - Training process reaches stable solution where loss stops decreasing.
- Measured by: Loss plateau, gradient norm <threshold
- Not guaranteed to find global optimum

**Cross-Entropy Loss** - Standard loss for classification. Measures divergence between predicted and true distributions.
- Binary: -[y*log(ŷ) + (1-y)*log(1-ŷ)]
- Multi-class: -Σ y_i * log(ŷ_i)

**Cross-Validation** - Technique to estimate generalization by repeatedly splitting data.
- K-fold: k splits, train k-1, test 1, repeat k times
- Prevents overfitting to specific train/test split

---

## D

**DAG (Directed Acyclic Graph)** - Graph structure used in Bayesian Networks. Represents causal relationships.
- Directed: Edges have direction
- Acyclic: No cycles/loops
- Used for: Probabilistic graphical models, causal inference

**Data Augmentation** - Creating synthetic training samples by transforming existing samples.
- Examples: image rotation, noise addition, back-translation
- Reduces overfitting when data is limited

**Decision Boundary** - Surface that separates different classes in feature space.
- Linear models: straight line/hyperplane
- Non-linear models: curved surfaces
- Visualizing helps understand model behavior

**Deep Learning** - Machine learning using neural networks with many layers (3+).
- Enables learning hierarchical representations
- Requires lots of data and computation

**Disentanglement** - Learned representations where each latent dimension captures independent factor of variation.
- Goal of: VAEs with β > 1, some representation learning
- Improves interpretability and transfer learning

**DPO (Direct Preference Optimization)** - Fine-tuning method that directly optimizes for user preferences without reward model.
- Simpler than RLHF (single model vs 3)
- More stable training dynamics

**Dropout** - Regularization technique. Randomly zero activations during training, scale by (1-p) at inference.
- p typically 0.1-0.5
- Prevents co-adaptation of neurons

**DQN (Deep Q-Network)** - Combines Q-Learning with neural networks. Uses experience replay and target network.
- First successful deep RL algorithm
- Enables Q-Learning in large state spaces

---

## E

**Embedding** - Dense vector representation of discrete entities (words, items, users).
- Dimension: typically 100-1024
- Learned: Updated during training
- Static: Pre-computed, not updated

**Entropy** - Measure of uncertainty. Higher entropy = more uniform distribution.
- Formula: -Σ p_i * log(p_i)
- Used in: information theory, decision trees, regularization

**Epoch** - One complete pass through the entire training dataset.
- Multiple epochs required for convergence
- Learning rate typically decays per epoch

**ε-Greedy** - Exploration strategy. With probability ε take random action, else exploit.
- ε typically 0.1 or decays from 1→0
- Balances exploration vs exploitation

**Evaluation Metric** - Quantitative measure of model performance on held-out data.
- Different tasks need different metrics
- Should align with business objective

**Exploding Gradient** - Gradients grow exponentially large during backpropagation.
- Causes: Poor initialization, large learning rate, deep networks
- Solution: Gradient clipping, careful initialization

---

## F

**F1-Score** - Harmonic mean of precision and recall. Balances both metrics.
- Formula: 2 * (precision * recall) / (precision + recall)
- Better than accuracy for imbalanced classification

**Feature** - Input variable/dimension to a model.
- Raw: Original from data source
- Engineered: Created to improve model

**Feature Engineering** - Process of creating new features from raw data.
- Domain knowledge crucial
- Huge impact on model performance

**FP (False Positive)** - Model predicts positive but actual is negative.
- Type I error
- Cost depends on application

**FN (False Negative)** - Model predicts negative but actual is positive.
- Type II error
- Often more costly than false positives in critical applications

**Fine-tuning** - Adapting a pre-trained model to a specific task.
- Transfer learning: Use representations learned on large data
- Typical: Lower learning rate, fewer iterations than pre-training

---

## G

**GAT (Graph Attention Networks)** - Graph neural network using attention to weight neighbor importance.
- Learns which neighbors matter most
- More interpretable than standard GCN

**GCN (Graph Convolutional Network)** - Neural network for graph-structured data.
- Aggregates neighbor features with mean operation
- Permutation invariant

**GELU (Gaussian Error Linear Unit)** - Smooth activation function. More stable than ReLU.
- Used in: Transformers, modern models
- Slower computation than ReLU

**Generalization** - Model performance on unseen data.
- Core goal of machine learning
- Trade-off with training performance (bias-variance tradeoff)

**Gini Impurity** - Measure of class mixedness in classification. Used in decision trees.
- Formula: 1 - Σ p_i²
- Lower = purer node

**Gradient** - Direction of steepest ascent of a function. Opposite points toward steepest descent.
- Computed by: Backpropagation
- Used in: Gradient descent, gradient-based optimization

**Gradient Descent** - Optimization algorithm. Iteratively move in negative gradient direction.
- Variants: Batch, SGD, Mini-batch
- Learning rate controls step size

**Gradient Clipping** - Limit magnitude of gradients to prevent explosion.
- Threshold typically 1.0-10.0
- Common in: RNNs, deep networks

---

## H

**Hallucination** - Model generates plausible but incorrect information.
- Common in: Large language models
- Mitigation: Retrieval augmentation, factuality training

**Hugging Face** - Popular library and model hub for NLP/transformers.
- Provides: Pre-trained models, tokenizers, training utilities
- Community: Millions of model variants

**Hyperparameter** - Model parameter set before training (not learned).
- Examples: learning rate, batch size, tree depth
- Tuned via: Grid search, random search, Bayesian optimization

**Hypothesis** - Proposed explanation or model in machine learning.
- Training: Find hypothesis that fits data well
- Testing: Evaluate hypothesis on unseen data

---

## I

**Inference** - Using trained model to make predictions on new data.
- Inference cost crucial for deployment
- Optimization: Quantization, distillation, pruning

**Information Gain** - Reduction in entropy from splitting. Used in decision trees.
- Formula: Entropy(parent) - Σ(weight * Entropy(child))
- Higher = better split

**In-Context Learning** - Large language model learns from examples in prompt without fine-tuning.
- Context length limits example count
- Enables rapid adaptation

**Instruction Tuning** - Fine-tuning language models on instruction-following tasks.
- Format: "Instruction: ... Input: ... Output: ..."
- Makes models better at following directives

---

## J

**Jacobian** - Matrix of all first-order partial derivatives.
- Size: output_dim × input_dim
- Used in: Neural network analysis, sensitivity analysis

---

## K

**KL Divergence** - Kullback-Leibler divergence. Asymmetric measure of distribution difference.
- Formula: Σ p_i * log(p_i / q_i)
- Used in: VAEs, mutual information, variational inference

**k-Nearest Neighbors (kNN)** - Non-parametric classifier. Predicts based on k nearest training examples.
- Simple but computationally expensive at inference
- No training phase

---

## L

**L1 Regularization** - Add penalty proportional to absolute weight magnitude. Sum of |w_i|.
- Effect: Zeros out some weights (sparse solution)
- Use: Feature selection

**L2 Regularization** - Add penalty proportional to squared weight magnitude. Sum of w_i².
- Effect: Shrinks all weights (smooth solution)
- Use: Prevent overfitting

**Laplacian Pyramid** - Multi-scale image representation. Used in some architectures.
- Creates hierarchical feature maps
- Fewer uses in modern deep learning

**Layer Normalization** - Normalize activations per sample across features (not per batch).
- Used in: Transformers, RNNs
- Batch-size independent, good for variable sequence lengths

**Learning Rate** - Hyperparameter controlling step size in gradient descent.
- Too high: Oscillates, diverges
- Too low: Slow convergence
- Scheduling: Decay over time improves convergence

**Learning Rate Scheduling** - Changing learning rate during training.
- Strategies: Step decay, exponential, cosine annealing, warmup
- Improves final performance 1-5% typically

**Leaky ReLU** - Activation function. Linear for negative inputs with small slope α (e.g. 0.01).
- Fixes dying ReLU problem
- Slightly more computation than ReLU

**Likelihood** - Probability of observed data given model parameters. P(data|θ).
- Higher likelihood = better model fit
- Used in: Maximum likelihood estimation, Bayesian inference

**Local Minimum** - Point where gradient is zero and all nearby points have higher loss.
- May not be global minimum
- Neural networks have many local minima

**LoRA (Low-Rank Adaptation)** - Parameter-efficient fine-tuning using low-rank matrix updates.
- Parameters: +0.1-0.5% relative to base model
- Inference: Merge with base model for no overhead

---

## M

**MAE (Mean Absolute Error)** - Average absolute prediction error. Robust to outliers.
- Formula: (1/n) Σ |y_i - ŷ_i|
- Same units as target

**Margin** - Distance from decision boundary in classification (especially SVMs).
- Larger margin = more confident classification
- Reduces overfitting risk

**Markov Property** - Future depends only on current state, not history.
- Enables: Efficient RL algorithms, HMMs
- Violated: Partial observability

**MDP (Markov Decision Process)** - Formal framework for sequential decision making.
- Components: States, Actions, Transitions, Rewards, Discount
- Foundation of: Reinforcement learning

**Mean Pooling** - Aggregate by taking mean over pool region.
- Used in: Convolutional networks, graph networks
- Alternative: Max pooling

**Memoization** - Caching computed values to avoid recomputation.
- Efficiency technique in: Dynamic programming, agents
- Trade-off: Memory for speed

**Mixture of Experts** - Model with multiple specialized sub-models (experts) and gating network.
- Gating: Learns which expert to use
- Benefits: Specialized capacity, sparse activation

**Model Collapse/Mode Collapse** - Generator produces limited variety. Common in GANs.
- Symptoms: Same output regardless of input variation
- Mitigation: Spectral normalization, Wasserstein loss

**Momentum** - Optimizer technique. Accumulates gradient direction over time.
- Helps escape plateaus
- Faster convergence than vanilla SGD

**MSE (Mean Squared Error)** - Average squared prediction error. Sensitive to outliers.
- Formula: (1/n) Σ (y_i - ŷ_i)²
- Units: target²

**Multi-Head Attention** - Multiple parallel attention computations in Transformer.
- Heads: Typically 8-12
- Benefit: Different heads learn different relationships

---

## N

**Natural Language Processing (NLP)** - AI field focused on processing human language.
- Tasks: Summarization, translation, question-answering, dialogue
- Enabled by: Large language models

**Neural Architecture Search (NAS)** - Automated search for optimal neural network architecture.
- Methods: Evolutionary algorithms, Bayesian optimization, reinforcement learning
- Challenge: Computationally very expensive

**Neuron** - Basic unit in neural network. Computes weighted sum + bias + activation.
- Formula: output = activation(Σ w_i * x_i + b)
- Millions used in modern networks

**Next Token Prediction** - Self-supervised task. Predict next token given previous tokens.
- Used for: Language model pre-training
- Foundation of: Autoregressive language models

**Normalizing Flow** - Architecture that transforms simple distribution to complex via invertible functions.
- Expressivity: Can represent any distribution
- Used in: Generative modeling, density estimation

**Null Hypothesis** - Default assumption in statistical testing (no effect exists).
- Rejection means: Effect likely exists with confidence
- Related: Frequentist vs Bayesian approaches

---

## O

**Off-Policy** - Learning target policy while following different behavior policy.
- Example: Q-Learning learns greedy policy while exploring with ε-greedy
- Advantage: Data efficiency
- Disadvantage: Stability issues (overestimation)

**One-Class SVM** - SVM variant for anomaly detection. Finds boundary around normal data.
- Unsupervised: Only normal data needed for training
- Hyperparameter ν controls outlier fraction

**Optimization** - Process of adjusting model parameters to minimize loss.
- Goal: Find parameters that generalize well
- Challenge: Loss landscape is non-convex with many local minima

**Optimizer** - Algorithm for optimization. Examples: SGD, Adam, RMSprop.
- Learning rate critical
- Adaptive optimizers less sensitive to LR than SGD

**Overfitting** - Model fits training data too well, performs poorly on test data.
- Symptoms: Training error low, test error high
- Causes: Model too complex, not enough data, wrong regularization
- Solutions: Simplify model, get more data, add regularization

**Overestimation Bias** - Q-Learning tendency to overestimate action values.
- Cause: Takes max of noisy Q estimates
- Solutions: Double Q-Learning, target networks

---

## P

**Padding** - Adding extra elements (usually zeros) to input.
- Convolutions: Preserve spatial dimensions
- Sequences: Align variable-length sequences

**Parameter** - Learned value in model (weights, biases).
- Updated during training
- Billions in modern large models

**Parameter-Efficient Fine-Tuning (PEFT)** - Techniques to reduce number of trainable parameters during fine-tuning.
- Methods: LoRA, Adapters, Prefix Tuning
- Benefit: Memory efficient, faster training

**Perplexity** - Metric for language models. Exponential of negative log-likelihood.
- Formula: exp(-1/n Σ log P(word_i))
- Lower = better
- Intuition: Average branching factor

**Policy** - Function mapping state to action (or distribution over actions).
- Deterministic: π(s) = a
- Stochastic: π(a|s) = probability

**Policy Gradient** - RL algorithm. Directly optimize policy by gradient of expected reward.
- Advantage: Works with continuous actions
- Disadvantage: High variance

**Positional Encoding** - Add position information to embeddings in Transformer.
- Method: sin/cos functions of different frequencies
- Benefit: Preserves sequence order information

**Precision** - Fraction of positive predictions that are correct.
- Formula: TP / (TP + FP)
- High precision: Few false alarms
- When: False positive costly

**Prefix Tuning** - Parameter-efficient fine-tuning using learned soft prompts.
- Parameters: +0.1% relative to base
- Performance: Slightly lower than LoRA

**Prompt** - Input text sent to language model.
- Prompt engineering: Crafting prompts for better performance
- Few-shot prompt: Includes examples

**Prompt Injection** - Security attack. Malicious input overrides system instructions.
- Prevention: Treat user input as data, not instructions
- Defense: Input validation, prompt filtering

**Pruning** - Removing unnecessary model components (weights, neurons, layers).
- Types: Weight pruning, neuron pruning, layer pruning
- Benefit: Smaller, faster model with minimal accuracy loss

---

## Q

**Q-Learning** - Off-policy RL algorithm. Learns Q-values (expected cumulative reward).
- Update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- Converges to optimal policy with sufficient exploration

**Q-Value** - Expected cumulative discounted reward from taking action a in state s.
- Q(s,a) = E[r_t + γ*r_{t+1} + γ²*r_{t+2} + ...]
- Higher = better action

---

## R

**R² (Coefficient of Determination)** - Fraction of variance in target explained by model.
- Range: [0, 1] (or negative for very bad models)
- Formula: 1 - (SS_res / SS_tot)
- Scale-independent (good for comparing datasets)

**RAG (Retrieval-Augmented Generation)** - Technique combining retrieval with generation.
- Retriever: Finds relevant documents
- Generator: Creates response using retrieved context
- Benefit: More accurate, interpretable answers

**Recall** - Fraction of positive samples correctly identified.
- Formula: TP / (TP + FN)
- High recall: Few missed positives
- When: False negative costly

**Recurrent Neural Network (RNN)** - Neural network with cycles. Has internal state.
- Limitation: Hard to train (vanishing/exploding gradients)
- Variants: LSTM, GRU (solve gradient problems)

**Reflection** - Agent analyzing its own outputs to identify and correct errors.
- Used in: Self-improving agents, interactive learning
- Benefit: Better accuracy through iteration

**Regularization** - Technique to reduce overfitting by constraining model complexity.
- Methods: L1/L2, Dropout, Early Stopping, Data Augmentation
- Strength λ controls trade-off between training fit and simplicity

**Regression** - Predicting continuous values (vs discrete in classification).
- Metrics: MSE, MAE, R²
- Output activation: Linear (unbounded)

**REINFORCE** - Policy gradient algorithm. On-policy, high variance.
- Update: θ ← θ + α * ∇log π(a|s) * R(τ)
- Baseline reduces variance

**ReLU (Rectified Linear Unit)** - Activation function. f(x) = max(0, x).
- Modern default for hidden layers
- Benefits: Computationally simple, avoids vanishing gradients
- Problem: Dying ReLU (outputs zero permanently)

**Residual Connection / Skip Connection** - Add input directly to layer output.
- Enables training very deep networks
- Benefit: Gradients flow directly to early layers

**RLHF (Reinforcement Learning from Human Feedback)** - Fine-tuning using human preferences as reward signal.
- Process: Collect preference data → train reward model → use for RLHF
- Challenge: Reward hacking (model exploits reward model)

**RNN (Recurrent Neural Network)** - Neural network with internal state for sequence processing.
- Sequential: Processes one element at a time
- Memory: Hidden state carries information forward
- Issue: Vanishing/exploding gradients in long sequences

**ROC-AUC (Receiver Operating Characteristic - Area Under Curve)** - Metric for binary classification.
- Plots: True positive rate vs false positive rate across thresholds
- AUC range: [0, 1], 0.5 = random, 1 = perfect
- Use: Ranking quality, threshold-independent evaluation

---

## S

**Sampling** - Generating outputs by sampling from model distribution.
- Temperature: Controls randomness (higher = more random)
- Top-k/Top-p: Limits vocabulary to top k/p probability mass
- Benefit: Diverse outputs

**Sampling Bias** - Training data not representative of deployment data.
- Causes: Poor data collection, distribution shift
- Mitigation: Careful data collection, online evaluation

**Scaling** - Changing the magnitude of features.
- Standardization: (x - mean) / std → mean 0, std 1
- Normalization: (x - min) / (max - min) → range [0, 1]
- Why: Some algorithms sensitive to magnitude

**Scatter Plot** - Plot with points showing relationship between two variables.
- Visual: Identify patterns, outliers, clusters
- Limitation: Only shows 2D (or 3D), may hide structure

**Self-Attention** - Attention mechanism on sequence itself (not external context).
- Used in: Transformers
- Benefit: Captures long-range dependencies

**Self-Supervised Learning** - Creating supervision signal from data itself (no labels).
- Methods: Masked language modeling, contrastive learning, next token prediction
- Benefit: Learn from unlabeled data

**Semantic Search** - Finding similar items based on meaning (vs keyword matching).
- Method: Embed query and documents, find nearest neighbors
- Benefit: Better results than keyword search

**Sequence-to-Sequence** - Model that maps input sequence to output sequence.
- Architecture: Encoder-decoder with attention
- Applications: Translation, summarization, dialogue

**SGD (Stochastic Gradient Descent)** - Gradient descent using single sample per update.
- Pro: Fast updates, explores widely
- Con: Noisy gradients, slow convergence
- Practice: Mini-batch SGD (compromise)

**Sigmoid** - Activation function. f(x) = 1/(1 + e^{-x}).
- Range: (0, 1)
- Vanishing gradient: ∂/∂x sigmoid = sigmoid*(1-sigmoid) ≤ 0.25
- Use: Binary classification output

**Silhouette Score** - Measure of clustering quality. Range [-1, 1].
- Score 1: Perfect clustering
- Score 0: On boundary between clusters
- Score -1: Misclassified

**Softmax** - Activation for multi-class output. Converts logits to probability distribution.
- Formula: σ(x_i) = e^{x_i} / Σ e^{x_j}
- Property: Outputs sum to 1

**Sparse** - Most elements are zero (vs dense where most are non-zero).
- Efficiency: Storage and computation proportional to non-zero count
- L1 regularization encourages sparsity

**Speculative Decoding** - Inference acceleration. Quickly generate candidates then verify with larger model.
- Speed-up: 2-3x typical
- Trade-off: Slightly higher inference cost for verification

**SSO (Sum of Squared Errors)** - Total squared errors in predictions.
- Formula: Σ (y_i - ŷ_i)²
- Used in: R² calculation, regression analysis

**Stacking** - Ensemble technique. Train meta-learner on outputs of base learners.
- Level 0: Train base models
- Level 1: Train meta-model on base predictions
- Benefit: Learns when to trust which base model

**Stateless** - System with no persistent internal state.
- Batch processing: Treat each batch independently
- vs Stateful: Maintains memory across batches

**Stochastic** - Involving randomness. Opposite of deterministic.
- Stochastic sampling: Random selection from distribution
- vs Deterministic: Same input always gives same output

**Summarization** - Condensing text while preserving key information.
- Extractive: Select important sentences from original
- Abstractive: Generate new text capturing essence

---

## T

**Tanh (Hyperbolic Tangent)** - Activation function. f(x) = (e^x - e^{-x})/(e^x + e^{-x}).
- Range: (-1, 1)
- Similar to sigmoid but centered at 0
- Use: RNNs, sometimes hidden layers

**Temperature** - Hyperparameter controlling randomness in sampling.
- T=1: No change (use model probabilities directly)
- T>1: Higher temperature, more random
- T<1: Lower temperature, more deterministic (greedy-like)

**Tensor** - Multi-dimensional array. Generalization of vectors/matrices.
- 0D: Scalar
- 1D: Vector
- 2D: Matrix
- 3D+: Tensor

**Test Set** - Held-out data for final model evaluation.
- Should be untouched until final evaluation
- Represents: Deployment data
- Size: 10-20% typical

**Threshold** - Decision boundary value.
- Classification: Predict positive if confidence > threshold
- Default: 0.5 for binary classification
- Tuning: Adjust for precision-recall trade-off

**Token** - Atomic unit of text. Word, subword, or character depending on tokenizer.
- Tokenization: Text → tokens → token IDs → embeddings
- Context length: Maximum tokens model can process

**Top-k Sampling** - During sampling, only consider top-k most likely tokens.
- k: Typically 40-50
- Benefit: Removes low-probability noise

**Top-p Sampling (Nucleus Sampling)** - During sampling, consider tokens with cumulative probability > p.
- p: Typically 0.9
- Benefit: Dynamic vocabulary size

**Training Set** - Data used to train model.
- Size: 60-80% typical
- Properties: Should be representative of deployment data

**Transfer Learning** - Using knowledge from pre-training on different task.
- Benefit: Better performance, less data needed, faster training
- Fine-tuning: Adapt pre-trained model to new task

**Transformer** - Architecture using multi-head self-attention instead of recurrence.
- Benefit: Parallelizable, captures long-range dependencies
- Used in: BERT, GPT, T5

**True Negative (TN)** - Model predicts negative and actual is negative (correct).
- Good outcome
- Counted in: Accuracy, specificity

**True Positive (TP)** - Model predicts positive and actual is positive (correct).
- Good outcome
- Counted in: Precision, recall, accuracy

---

## U

**Underfitting** - Model too simple to capture data patterns.
- Symptoms: High training error, high test error
- Causes: Low capacity, not enough training, high regularization
- Solutions: Increase capacity, reduce regularization, more features

**Universal Approximation** - Theorem: Sufficiently large neural network can approximate any function.
- Caveat: Says nothing about learnability or efficiency
- Implication: Depth often more efficient than width

**Unsupervised Learning** - Learning from unlabeled data.
- Examples: Clustering, dimensionality reduction, density estimation
- Challenge: No ground truth for evaluation

---

## V

**Validation Set** - Data used to tune hyperparameters and select models.
- Size: 10-20% typical
- Separate from: Training (learning) and test (final eval)
- Used for: Early stopping, hyperparameter selection

**Vanishing Gradient** - Gradients become exponentially small in deep networks.
- Cause: Chain rule with gradients < 1 multiplies many times
- Especially: Sigmoid, tanh activations
- Solution: ReLU, skip connections, batch normalization

**Value Function** - V(s) = expected cumulative discounted reward from state s.
- Used in: Value-based RL, Actor-Critic
- Estimated: Neural network function approximation

**Variable Length** - Inputs/outputs of different sizes.
- Challenge: Neural networks expect fixed-size inputs
- Solutions: Padding, bucketing by length, masking

**Variance** - In statistics: spread of values around mean.
- In ML bias-variance: Sensitivity to training data changes
- Model with high variance: Overfits

**Vector** - 1D array of numbers.
- Embedding vector: Learned representation
- Gradient vector: Partial derivatives per parameter

**Vocabulary** - Set of possible tokens.
- Typical: 20k-50k for BPE
- Size trade-off: Larger = better coverage, slower tokenization

---

## W

**Warmup** - Gradually increase learning rate at beginning of training.
- Strategy: Linear increase from 0 to max over N steps
- Benefit: Prevents gradient explosion in Transformers
- Duration: Typically 10% of total training

**Weight Decay** - Regularization adding penalty on magnitude of weights.
- L2: Penalty ∝ Σ w²
- L1: Penalty ∝ Σ |w|
- Effect: Smaller weights, simpler model

**Weight Initialization** - Setting initial values of neural network weights.
- Xavier: 1/√fan_in for sigmoid/tanh
- He: √(2/fan_in) for ReLU
- Importance: Affects gradient flow, convergence speed

---

## X

**Xavier Initialization** - Weight initialization method. Scale inversely to fan-in.
- Formula: Uniform[-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out))]
- Purpose: Keep activation variance consistent across layers
- For: Sigmoid, tanh activations

**XGBoost** - Optimized gradient boosting library.
- Features: Regularization, parallel training, missing value handling
- Performance: Often wins Kaggle competitions
- Trade-off: More complex, slower training than simple tree ensembles

---

## Y

**Yield** - Generator keyword in Python. Enables lazy evaluation.
- Efficient: Doesn't materialize entire sequence in memory
- Use in: Data loading, agent interactions

---

## Z

**Zero-shot Learning** - Predicting on classes not in training data.
- Method: Use class descriptions/attributes
- Requirement: Side information about classes
- Example: CLIP predicts any image label given class descriptions

**Zipfian Distribution** - Probability distribution where frequency ranks inversely with probability.
- Natural language: Word frequency follows Zipf's law
- Implication: Few common words, many rare words

---

## Acronyms Quick Reference

| Acronym | Full Name |
|---------|-----------|
| A2C | Advantage Actor-Critic |
| ABC | Approximate Bayesian Computation |
| ADAM | Adaptive Moment Estimation |
| AE | Autoencoder |
| AI | Artificial Intelligence |
| AUC | Area Under Curve |
| BN | Batch Normalization |
| BPE | Byte Pair Encoding |
| CAM | Class Activation Map |
| CNN | Convolutional Neural Network |
| CSRF | Cross-Site Request Forgery |
| CTC | Connectionist Temporal Classification |
| CV | Cross-Validation |
| DAE | Denoising Autoencoder |
| DAG | Directed Acyclic Graph |
| DAgger | Dataset Aggregation |
| DL | Deep Learning |
| DQN | Deep Q-Network |
| DPO | Direct Preference Optimization |
| ELBO | Evidence Lower BOund |
| EM | Expectation Maximization |
| F1 | F1-Score (harmonic mean) |
| FN | False Negative |
| FP | False Positive |
| GAN | Generative Adversarial Network |
| GAT | Graph Attention Network |
| GCN | Graph Convolutional Network |
| GELU | Gaussian Error Linear Unit |
| GIN | Graph Isomorphism Network |
| GLUE | General Language Understanding Evaluation |
| GPT | Generative Pre-trained Transformer |
| GRU | Gated Recurrent Unit |
| HMM | Hidden Markov Model |
| ICL | In-Context Learning |
| ICAS | Interpretable Classification |
| KDD | Knowledge Discovery in Databases |
| KL | Kullback-Leibler |
| KNN | K-Nearest Neighbors |
| LSTM | Long Short-Term Memory |
| MAE | Mean Absolute Error |
| MAP | Maximum A Posteriori |
| MAPE | Mean Absolute Percentage Error |
| MB | Mini-Batch |
| MCMC | Markov Chain Monte Carlo |
| MDP | Markov Decision Process |
| MLP | Multi-Layer Perceptron |
| MMLU | Massive Multitask Language Understanding |
| MoE | Mixture of Experts |
| MSE | Mean Squared Error |
| MTTNN | Multi-Task Tree Neural Network |
| NAS | Neural Architecture Search |
| NLP | Natural Language Processing |
| ONNX | Open Neural Network Exchange |
| OOB | Out-Of-Bag |
| PCA | Principal Component Analysis |
| PDF | Probability Density Function |
| PEFT | Parameter-Efficient Fine-Tuning |
| PG | Policy Gradient |
| POMDP | Partially Observable MDP |
| PPO | Proximal Policy Optimization |
| PR | Precision-Recall |
| PEFT | Parameter-Efficient FinetuneTraining |
| QLORA | Quantized LoRA |
| RAG | Retrieval-Augmented Generation |
| REINFORCE | REward INcrement = Noncriticized Reinforcement |
| ReLU | Rectified Linear Unit |
| RESMEM | Residual Memory |
| RLHF | Reinforcement Learning from Human Feedback |
| RMSE | Root Mean Squared Error |
| RMSprop | Root Mean Square Propagation |
| RNN | Recurrent Neural Network |
| ROC | Receiver Operating Characteristic |
| RPO | Relative Policy Optimization |
| SAG | Stochastic Average Gradient |
| SGD | Stochastic Gradient Descent |
| SHAP | SHapley Additive exPlanations |
| SQA | Software Quality Assurance |
| SVM | Support Vector Machine |
| t-SNE | t-Distributed Stochastic Neighbor Embedding |
| TPOT | Tree-based Pipeline Optimization Tool |
| TRPO | Trust Region Policy Optimization |
| TS | Time Series |
| UMAP | Uniform Manifold Approximation and Projection |
| VAE | Variational Autoencoder |
| VI | Variational Inference |
| ViT | Vision Transformer |
| VLM | Vision Language Model |
| VRAM | Video RAM |
| XAI | Explainable AI |

---

## Common Metrics by Domain

### Classification
- **Balanced Classes:** Accuracy, F1, ROC-AUC
- **Imbalanced Classes:** Precision, Recall, F1, PR-AUC
- **Multi-class:** Macro/Micro F1, weighted metrics
- **Ranking:** NDCG, MAP, MRR

### Regression
- **General:** MSE, RMSE, MAE, R²
- **Robust:** Median AE, Huber loss
- **Relative:** MAPE, RMAPE
- **Business:** Custom metrics aligned with objectives

### Language Models
- **Perplexity:** Exponential of cross-entropy loss
- **BLEU:** For machine translation
- **ROUGE:** For summarization
- **METEOR:** Translation quality
- **Benchmarks:** GLUE, MMLU, SuperGLUE

### Computer Vision
- **Object Detection:** mAP, IoU
- **Segmentation:** Dice, Jaccard/IoU
- **Image Classification:** Top-1, Top-5 Accuracy
- **Visual QA:** Accuracy, BLEU

### Reinforcement Learning
- **Episode Reward:** Cumulative return per episode
- **Success Rate:** Fraction of successful episodes
- **Sample Efficiency:** Performance vs steps taken
- **Wall-clock Time:** Real time to convergence

---

*This glossary covers core concepts used throughout the AI Fundamentals, LLM, and Agentic AI curricula. For detailed explanations, see the concept markdown files and notebooks.*
