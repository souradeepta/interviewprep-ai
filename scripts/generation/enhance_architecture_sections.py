#!/usr/bin/env python3
"""Enhance Architecture / Trade-offs sections with detailed diagrams and comparison tables."""

import os
import re

BASE = "/home/sbisw/github/interviewprep-ml"

# Architecture enhancements for each concept
ARCHITECTURES = {
    # AI Fundamentals
    "31-q-learning": {
        "content": """## Architecture / Trade-offs

### Core Architecture

```mermaid
graph TD
    A["Agent State<br/>s(t)"] -->|Choose Action| B["Epsilon-Greedy<br/>Policy"]
    B -->|Action a| C["Environment"]
    C -->|Reward r<br/>New State s'| D["Q-Table Update<br/>Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]"]
    D -->|Store Q-values| E["Q-Value Function<br/>Dictionary or Network"]
    E -->|Lookup| B

    style A fill:#e1f5ff
    style E fill:#fff3e0
    style D fill:#f3e5f5
```

### Off-Policy vs On-Policy

| Aspect | Q-Learning (Off-Policy) | REINFORCE (On-Policy) |
|--------|-------------------------|----------------------|
| **What learns** | Optimal policy Q* | Current policy π |
| **Sample efficiency** | High (learn from any trajectory) | Low (must learn from current policy) |
| **Stability** | Overestimation bias | More stable |
| **Implementation** | Simpler | More complex |
| **Convergence** | Guaranteed with conditions | Convergent but slower |
| **Use case** | Learning from diverse experience | Immediate policy improvement |

### Tabular vs Function Approximation

| Approach | Tabular Q-Learning | Deep Q-Network (DQN) |
|----------|-------------------|----------------------|
| **State space** | Small, discrete | Large, continuous |
| **Memory** | O(|S| × |A|) | O(network parameters) |
| **Scalability** | Limited to small domains | Scales to complex domains |
| **Convergence** | Guaranteed | Not guaranteed |
| **Generalization** | No generalization | Generalizes across similar states |
| **Stability** | Inherently stable | Requires tricks (target network, replay buffer) |
| **Best for** | Toy problems, simple games | Atari, robotics, real-world tasks |

### Key Design Trade-offs

**Exploration Rate (ε)**
- High ε (0.5+): More exploration, slower convergence, discovers rare good actions
- Low ε (0.01): Fast convergence, may get stuck in local optima
- Decay ε over time: Best approach, explore early, exploit later

**Learning Rate (α)**
- High α (0.5+): Fast learning, unstable, overshoots optima
- Low α (0.01): Slow learning, stable, requires many iterations
- Optimal: Decay α over time or use adaptive methods (decreasing with update count)

**Discount Factor (γ)**
- γ close to 0: Myopic, only cares about immediate reward
- γ close to 1: Long-term planning, slower convergence, can be unstable
- Typical: 0.9-0.99 for most problems

### Addressing Overestimation

```mermaid
graph LR
    A["Standard Q-Learning<br/>Overestimation"] -->|Problem| B["Uses max from<br/>same network"]
    C["Double Q-Learning"] -->|Solution 1| D["Two networks:<br/>one selects, one evaluates"]
    E["Target Network"] -->|Solution 2| F["Slowly updated<br/>copy of Q-network"]

    style A fill:#ffebee
    style C fill:#e8f5e9
    style E fill:#e8f5e9
```"""
    },

    "32-policy-gradients": {
        "content": """## Architecture / Trade-offs

### Core Architecture

```mermaid
graph TD
    A["Environment State<br/>s(t)"] -->|Input| B["Policy Network<br/>π(a|s;θ)"]
    B -->|Action probabilities| C["Sample Action<br/>a ~ π(a|s)"]
    C -->|Execute| D["Environment"]
    D -->|Reward r<br/>Next state s'| E["Compute Return<br/>G(t) = Σ γ^t r(t)"]
    E -->|Gradient direction| F["Policy Gradient<br/>∇J(θ) = E[∇log π(a|s) G(t)]"]
    F -->|Update θ| B

    style B fill:#fff3e0
    style F fill:#f3e5f5
    style E fill:#e1f5ff
```

### Policy Gradient Variants

| Method | Baseline | Variance | Bias | Convergence |
|--------|----------|----------|------|-------------|
| **REINFORCE** | None | Very High | Unbiased | Slow |
| **REINFORCE w/ Baseline** | State value | High | Unbiased | Medium |
| **Actor-Critic** | Critic network | Low | Biased | Fast |
| **A3C** | Advantage function | Low | Slight bias | Very Fast |
| **PPO** | Clipped advantage | Low | Slight bias | Stable |

### Continuous vs Discrete Actions

```mermaid
graph TD
    A["Action Space"] -->|Discrete| B["Output softmax<br/>π(a|s) ∈ [0,1]"]
    A -->|Continuous| C["Output mean & variance<br/>μ(s), σ(s)"]
    B -->|Sample action| D["Select from<br/>discrete options"]
    C -->|Sample action| E["Gaussian distribution<br/>a ~ N(μ, σ)"]
    D --> F["Policy Gradient Update"]
    E --> F

    style A fill:#fff3e0
    style B fill:#e1f5ff
    style C fill:#e1f5ff
```

### Variance Reduction Techniques

| Technique | Impact on Variance | Impact on Bias | Cost |
|-----------|-------------------|-----------------|------|
| Baseline subtraction | Moderate reduction | None (unbiased) | Minimal |
| Advantage estimation | Good reduction | Slight increase | Moderate |
| Generalized Advantage Estimation (GAE) | Very good | Very slight | Moderate |
| Multiple steps (n-step returns) | Good | Small increase | Moderate |

### Trade-off: Sample Efficiency vs Stability

**On-Policy (REINFORCE family)**
- ✓ Unbiased gradient estimates
- ✓ Guaranteed convergence to local optima
- ✗ High variance (need many samples)
- ✗ Sample inefficient

**Off-Policy (Deterministic Policy Gradient)**
- ✓ Better sample efficiency
- ✗ Biased gradient estimates
- ✗ Convergence not guaranteed
- ✓ More stable training"""
    },

    "33-actor-critic-methods": {
        "content": """## Architecture / Trade-offs

### Two-Network Architecture

```mermaid
graph TB
    subgraph "Actor-Critic System"
        A["Shared Features<br/>Feature Extractor"]

        A --> B["Actor Network<br/>π(a|s;θ_actor)"]
        A --> C["Critic Network<br/>V(s;θ_critic)"]

        D["Environment"] -->|State s, Reward r| E["TD Error<br/>δ = r + γV(s') - V(s)"]

        B -->|Action a| D
        C -->|Value estimate| E

        E -->|TD signal| B
        E -->|Value target| C

        B -->|Probability| D

        style B fill:#fff3e0
        style C fill:#e1f5ff
        style E fill:#f3e5f5
    end
```

### Actor vs Critic Responsibilities

| Component | Actor | Critic |
|-----------|-------|--------|
| **What it learns** | Policy π(a\|s) | Value function V(s) |
| **Output** | Action probabilities/means | Scalar value |
| **Loss function** | Policy gradient × TD error | TD loss (V(s) - target)² |
| **Training signal** | Critic's TD error | Ground truth return |
| **Role** | Decides what to do | Evaluates how good decision is |
| **Failure mode** | Policies underexplore | Overestimates/underestimates values |

### Architecture Variants

```mermaid
graph LR
    A["Actor-Critic Base"] --> B["Advantage Actor-Critic<br/>A3C"]
    A --> C["Asynchronous A3C<br/>Parallel workers"]
    A --> D["Deep Deterministic<br/>Policy Gradient<br/>DDPG"]
    A --> E["Soft Actor-Critic<br/>SAC"]

    B -->|Use| F["Discrete actions"]
    D -->|Use| G["Continuous actions"]
    E -->|Use| G

    style B fill:#e1f5ff
    style D fill:#fff3e0
    style E fill:#f3e5f5
```

### Bias-Variance Trade-off

| Aspect | Lower Bias (Critic) | Higher Bias (Advantage) |
|--------|-------------------|------------------------|
| **TD Target** | Full episode return (unbiased) | Critic prediction (biased) |
| **Variance** | High (entire episode affects gradient) | Low (critic reduces noise) |
| **Convergence speed** | Slow | Fast |
| **When to use** | Small variance in environment | Complex tasks with noise |
| **Stability** | More stable | Less stable |

### Multi-Step Learning

```mermaid
graph TD
    A["One-Step TD"] -->|δ = r + γV(s')| B["Low bias, High variance<br/>Fast learning, Unstable"]
    A -->|n-Step| C["n-Step TD<br/>δ = r + γr' + ... + γ^n V(s_n)"]
    A -->|∞-Step| D["Monte Carlo<br/>Use full return"]
    C -->|Balanced| E["Good trade-off<br/>n=3 or 4 typical"]

    style B fill:#fff3e0
    style E fill:#e1f5ff
    style D fill:#f3e5f5
```

### Trade-offs: Synchronous vs Asynchronous

| Property | Synchronous A3C | Asynchronous A3C |
|----------|-----------------|------------------|
| **Training** | Single worker, sequential | Multiple workers, parallel |
| **Convergence speed** | Slower | Faster (more samples) |
| **Complexity** | Simple | Complex (threading, locking) |
| **Hardware** | Single machine | Multi-core/GPU |
| **Stability** | Less stable | More stable (diverse experience) |
| **Memory** | Lower | Higher (multiple workers) |"""
    },

    "34-graph-neural-networks": {
        "content": """## Architecture / Trade-offs

### GNN Message Passing Architecture

```mermaid
graph TD
    A["Input Graph<br/>Nodes & Edges"] -->|Feature Embedding| B["Node Embeddings<br/>h_v^(0)"]

    B -->|Layer 1| C["Message Passing<br/>Aggregate neighbor info"]
    C -->|Update| D["h_v^(1) = σ(W * [h_v^(0); Agg(h_u^(0))])"]
    D -->|Layer 2-k| E["Repeat: More expressive<br/>representations"]
    E -->|Readout| F["Graph-level: Pool nodes<br/>Node-level: Use final embedding"]
    F -->|Prediction| G["Task Output<br/>Classification/Regression"]

    style C fill:#f3e5f5
    style D fill:#e1f5ff
    style G fill:#fff3e0
```

### GNN Architecture Types

| Type | Aggregation | Best For | Expressive Power |
|------|-------------|----------|------------------|
| **Graph Convolutional Network (GCN)** | Weighted mean | Node classification | Moderate |
| **GraphSAGE** | Neighborhood sampling | Inductive learning | Moderate |
| **Graph Attention Network (GAT)** | Learned attention weights | Long-range dependencies | High |
| **Message Passing Neural Network** | Custom aggregation | General graphs | Very high |
| **Graph Isomorphism Network (GIN)** | Sum aggregation | Theory-grounded | High |

### Aggregation Functions Comparison

```mermaid
graph LR
    A["Neighbor Information<br/>from h_u for u ∈ N(v)"] -->|Mean| B["GCN: Simple average<br/>Good for undirected graphs"]
    A -->|Sum| C["GIN: Summation<br/>Provably expressive"]
    A -->|Max| D["GraphSAGE: Max pooling<br/>Robust to outliers"]
    A -->|Attention| E["GAT: Learned weights<br/>Adaptive to importance"]

    style B fill:#e1f5ff
    style C fill:#f3e5f5
    style E fill:#fff3e0
```

### Depth vs Width Trade-off

| Aspect | Shallow (1-2 layers) | Deep (5+ layers) |
|--------|----------------------|-----------------|
| **Receptive field** | 1-2 hop neighbors | 5+ hop neighbors |
| **Information flow** | Local structure | Global structure |
| **Vanishing gradient** | Unlikely | Likely (harder training) |
| **Oversmoothing** | Not an issue | Major issue (embeddings converge) |
| **Computation** | Fast | Slow |
| **Memory** | Low | High |
| **Best for** | Local patterns | Long-range relationships |

### Node-level vs Graph-level Tasks

```mermaid
graph TD
    A["GNN Output"] -->|Node-level| B["Use node embeddings directly"]
    A -->|Graph-level| C["Aggregate node embeddings"]

    B -->|Tasks| D["Node classification<br/>Link prediction"]
    C -->|Pooling| E["Mean/Max/Attention pooling"]
    E -->|Tasks| F["Graph classification<br/>Molecular property prediction"]

    style D fill:#e1f5ff
    style F fill:#fff3e0
```

### Scalability Considerations

| Issue | Solution | Trade-off |
|-------|----------|-----------|
| **Large graphs don't fit in memory** | Neighbor sampling (GraphSAGE) | Biased gradient estimates |
| **Dense computation with large node count** | Sparse attention patterns | May miss important connections |
| **Message passing complexity O(E)** | Subgraph sampling | Information loss |
| **Deep GNNs oversmoothing** | Skip connections, layer normalization | Added complexity |
| **Training time for large graphs** | Mini-batch sampling | Requires careful variance control |"""
    },

    "35-causal-inference": {
        "content": """## Architecture / Trade-offs

### Causal vs Observational Reasoning

```mermaid
graph TD
    A["Data Analysis Goal"] -->|Predict| B["Prediction<br/>P(Y|X)"]
    A -->|Intervene| C["Causal Inference<br/>P(Y|do(X))"]

    B -->|Methods| D["Standard ML<br/>Regression, Classification"]
    C -->|Methods| E["Causal Methods<br/>IV, Matching, Causal Forests"]

    D -->|Answer| F["What will happen?<br/>Correlation-based"]
    E -->|Answer| G["What if we change X?<br/>Causation-based"]

    style B fill:#fff3e0
    style C fill:#f3e5f5
    style F fill:#e1f5ff
    style G fill:#e8f5e9
```

### Causal Identification Methods

| Method | Assumptions | Data Type | Bias |
|--------|-------------|-----------|------|
| **Randomized Experiment** | None (gold standard) | Experimental | Unbiased |
| **Propensity Score Matching** | Unconfoundedness | Observational | Biased if unobserved confounders |
| **Instrumental Variables** | Valid instrument exists | Observational | Unbiased (if valid) |
| **Regression Adjustment** | No hidden confounders | Observational | Biased if confounders missed |
| **Causal Forests** | Unconfoundedness | Observational | Unbiased under assumptions |
| **Synthetic Control** | Parallel trends | Panel data | Biased if assumption violated |

### Confounder Adjustment

```mermaid
graph TD
    A["Confounder Z"] -->|Affects| B["Treatment X"]
    A -->|Affects| C["Outcome Y"]
    B -->|Affects| C

    D["Naive Comparison<br/>Corr(X,Y)"] -->|Biased| E["Confounded estimate<br/>Includes Z→Y effect"]

    F["Adjusted Comparison<br/>Corr(X,Y|Z)"] -->|Unbiased| G["Causal estimate<br/>Removes Z effect"]

    style A fill:#f3e5f5
    style D fill:#ffebee
    style F fill:#e8f5e9
```

### Methods Comparison

| Approach | Causal Assumption | Handles Hidden Confounders | Handles Feedback | Scalability |
|----------|-------------------|---------------------------|------------------|-------------|
| **Randomization** | No confounders (by design) | Yes | Yes | Limited |
| **Adjustment** | No hidden confounders | No | No | High |
| **Matching** | Unconfoundedness | No | No | Medium |
| **IV methods** | Instrument validity | Partial | No | Medium |
| **Causal Discovery** | None (learns from data) | Difficult | Can identify | High |

### Causal DAG Example

```mermaid
graph LR
    Z["Confounder<br/>Socioeconomic Status"]
    X["Treatment<br/>Education Level"]
    Y["Outcome<br/>Income"]
    U["Unobserved<br/>Ability"]

    Z -->|Confounds| X
    Z -->|Affects| Y
    X -->|Causes| Y
    U -->|Affects| X
    U -->|Affects| Y

    style Z fill:#f3e5f5
    style U fill:#ffebee
    style X fill:#fff3e0
    style Y fill:#e1f5ff
```

### Trade-offs: Strong Assumptions vs Flexibility

| Approach | Assumptions | Flexibility | Robustness |
|----------|-------------|-------------|-----------|
| **Randomization** | Very strict (need control group) | Low (fixed design) | Very high |
| **Causal Discovery** | Minimal (structure learning) | High (data-driven) | Low (can identify wrong structure) |
| **Domain Expert DAG** | Moderate (expert knowledge) | Moderate (expert-guided) | Depends on expertise |
| **Multiple Robustness Checks** | Weak (sensitivity testing) | Very high | High (if checks pass) |"""
    },

    "36-probabilistic-graphical-models": {
        "content": """## Architecture / Trade-offs

### Directed vs Undirected Graphical Models

```mermaid
graph LR
    subgraph "Bayesian Network (Directed)"
        A["X1"] -->|Causes| B["Y"]
        C["X2"] -->|Causes| B
        B -->|Causes| D["Z"]
    end

    subgraph "Markov Random Field (Undirected)"
        E["X1"] ---|Correlates| F["Y"]
        G["X2"] ---|Correlates| F
        F ---|Correlates| H["Z"]
    end

    style A fill:#fff3e0
    style E fill:#f3e5f5
```

### Model Type Comparison

| Property | Bayesian Network | Markov Random Field |
|----------|-----------------|----------------------|
| **Graph type** | Directed Acyclic Graph | Undirected graph |
| **Semantics** | Causal/temporal | Symmetric correlations |
| **Factorization** | Conditional probabilities P(X_i\|Parents) | Potential functions φ(cliques) |
| **Inference** | Belief propagation, VE | Belief propagation, sampling |
| **When to use** | Causal relationships | Symmetric dependencies |
| **Example** | Medical diagnosis | Image segmentation |

### Inference Algorithms

```mermaid
graph TD
    A["Inference Task<br/>Compute P(X|evidence)"] -->|Exact| B["Variable Elimination"]
    A -->|Exact| C["Belief Propagation<br/>Sum-Product"]
    A -->|Approximate| D["Markov Chain<br/>Monte Carlo"]
    A -->|Approximate| E["Variational<br/>Inference"]

    B -->|Complexity| F["O(n^k) worst case<br/>k = treewidth"]
    C -->|Complexity| G["Efficient on trees<br/>Slow on loopy graphs"]
    D -->|Complexity| H["Slow but general<br/>Sample-based"]
    E -->|Complexity| I["Fast approximation<br/>Tractable bounds"]

    style B fill:#e1f5ff
    style C fill:#e1f5ff
    style D fill:#fff3e0
    style E fill:#f3e5f5
```

### Parameter Learning Methods

| Method | Observability | Complexity | Assumptions |
|--------|---------------|-----------|-------------|
| **Maximum Likelihood (MLE)** | Fully observed | Easy | No missing data |
| **Expectation Maximization (EM)** | Partially observed | Medium | Convergence to local optimum |
| **Gradient Descent** | Any | Medium | Differentiable |
| **Gibbs Sampling** | Partially observed | Hard | MCMC mixing |
| **Variational EM** | Partially observed | Hard | Variational approximation quality |

### Complexity of Inference

```mermaid
graph TD
    A["Graph Structure"] -->|Tree| B["Polynomial time<br/>O(n^2) for n variables"]
    A -->|Tree-width k| C["Exponential in k<br/>O(n^(k+1))"]
    A -->|Dense/Loopy| D["NP-hard<br/>Need approximation"]

    B -->|Use| E["Exact inference"]
    C -->|Use| F["May use exact if k small"]
    D -->|Use| G["Approximate: sampling<br/>or variational"]

    style B fill:#e8f5e9
    style C fill:#fff3e0
    style D fill:#ffebee
```

### Model Selection Trade-offs

| Aspect | Simple Model | Complex Model |
|--------|--------------|----------------|
| **Structure** | Few edges, few parameters | Many edges, many parameters |
| **Interpretability** | High (easy to understand) | Low (hard to understand) |
| **Data requirements** | Low (fewer parameters) | High (risk overfitting) |
| **Inference speed** | Fast | Slow |
| **Modeling power** | Limited (may underfit) | High (may overfit) |
| **Overfitting risk** | Low | High |
| **Best for** | Small data, interpretability | Large data, accuracy |"""
    },

    "37-variational-autoencoders": {
        "content": """## Architecture / Trade-offs

### VAE Architecture

```mermaid
graph TD
    A["Input Data<br/>x"] -->|Encode| B["Encoder Network<br/>q(z|x)"]
    B -->|Output| C["Mean & Variance<br/>μ(x), σ(x)"]
    C -->|Sample| D["Latent Vector<br/>z ~ N(μ,σ)"]
    D -->|Decode| E["Decoder Network<br/>p(x|z)"]
    E -->|Output| F["Reconstructed<br/>x̂"]

    G["Prior<br/>p(z) = N(0,I)"] -.->|Regularize| D

    F -->|Reconstruction Loss| H["L_recon = ||x - x̂||²"]
    D -->|KL Divergence| I["L_KL = KL(q(z|x) || p(z))"]

    H -->|Total Loss| J["L_total = L_recon + βL_KL"]
    I -->|Total Loss| J

    style B fill:#fff3e0
    style E fill:#f3e5f5
    style D fill:#e1f5ff
    style G fill:#f3e5f5
```

### VAE vs Standard Autoencoder

| Property | Standard AE | VAE |
|----------|------------|-----|
| **Latent space** | Discrete points | Continuous distribution |
| **Sampling** | Can't generate new samples | Can sample and generate |
| **Interpretation** | No probabilistic meaning | Each dimension meaningful |
| **Regularization** | L1/L2 on weights | KL divergence on distribution |
| **Training** | Simpler, faster | More complex, slower |
| **Generalization** | Limited | Better (smooth latent space) |
| **Use case** | Compression | Generation, representation learning |

### Loss Function Trade-off (β parameter)

```mermaid
graph TD
    A["Loss = L_recon + β*L_KL"] -->|β = 0| B["Reconstruction only<br/>Good reconstruction, poor distribution"]
    A -->|β = 1| C["Balanced (ELBO)<br/>Good trade-off (theoretical)")
    A -->|β > 1| D["More regularization<br/>Better distribution, worse reconstruction"]

    B -->|Result| E["Latent space collapses<br/>Posterior ≈ Prior"]
    C -->|Result| F["Smooth latent space<br/>Good generation + reconstruction"]
    D -->|Result| G["Disentangled factors<br/>But blurry reconstructions"]

    style C fill:#e8f5e9
```

### Encoder Design Options

| Encoder Type | Latent Dimension | Variance Modeling | Best For |
|--------------|-----------------|------------------|----------|
| **Deterministic μ + learned σ** | Variable | Full flexibility | General purpose |
| **Deterministic μ + fixed σ** | Variable | Simpler training | Image reconstruction |
| **Factorized Gaussian** | Independent dimensions | Simplified | Standard VAE |
| **Spherical Gaussian** | Isotropic | Very simple | Training stability |
| **Full covariance** | Dependent dimensions | Most flexible | Complex data |

### Posterior Collapse Problem

```mermaid
graph LR
    A["Standard VAE Training"] -->|Issue| B["KL term → 0<br/>q(z|x) ≈ p(z)"]
    B -->|Result| C["Decoder ignores z<br/>Autoencoder-like behavior"]
    C -->|Manifestation| D["Poor generation<br/>Blurry samples"]

    E["Solution 1: β-VAE"] -->|Increase| F["β weight<br/>Enforce distribution"]
    G["Solution 2: Annealing"] -->|Gradually| H["Increase KL weight<br/>during training"]
    I["Solution 3: Free bits"] -->|Ensure| J["Minimum KL contribution<br/>per minibatch"]

    style B fill:#ffebee
    style E fill:#e8f5e9
    style G fill:#e8f5e9
    style I fill:#e8f5e9
```

### Conditional vs Unconditional VAE

| Aspect | Unconditional | Conditional (CVAE) |
|--------|---------------|--------------------|
| **Input** | Only x | x and conditioning y |
| **Latent** | p(z), q(z\|x) | p(z\|y), q(z\|x,y) |
| **Generation** | Random sampling | Controlled generation |
| **Applications** | General generation | Specific categories |
| **Complexity** | Simpler | More complex |
| **Control** | No control | Full control over output |"""
    },

    "38-generative-adversarial-networks": {
        "content": """## Architecture / Trade-offs

### GAN Training Loop

```mermaid
graph TD
    A["Real Data<br/>x ~ p_data"] -->|Input| B["Discriminator<br/>D(x)"]
    C["Noise<br/>z ~ p_z"] -->|Input| D["Generator<br/>G(z)"]

    D -->|Fake samples| E["x̂ = G(z)"]
    E -->|Input| B

    B -->|Real score| F["L_D = -[log D(x) + log(1-D(G(z)))]"]
    D -->|Fake score| G["L_G = -log D(G(z))"]

    F -->|Discriminator step| H["Update D weights"]
    G -->|Generator step| I["Update G weights"]

    H -->|Equilibrium| J["Better discriminator<br/>Better training signal"]
    I -->|Equilibrium| J

    style A fill:#e1f5ff
    style C fill:#f3e5f5
    style B fill:#fff3e0
    style D fill:#fff3e0
```

### GAN Variants Comparison

| Variant | Generator Loss | Stability | Training Speed | Quality |
|---------|----------------|-----------|-----------------|---------|
| **Standard GAN** | -log D(G(z)) | Low | Medium | Medium |
| **Least Squares GAN** | (D(G(z))-1)² | Medium | Medium | Good |
| **Wasserstein GAN** | -E[D(G(z))] | High | Slow | Very good |
| **Spectral Norm GAN** | Standard + Lipschitz constraint | High | Medium | Very good |
| **Progressive GAN** | Standard + progressive growth | High | Slow | Excellent |

### Mode Collapse Phenomenon

```mermaid
graph TD
    A["Generator produces<br/>limited variety"] -->|Cause| B["Generator focuses on<br/>subset of modes"]
    B -->|Why| C["Discriminator can't distinguish<br/>even from real data"]
    C -->|Result| D["Generator stops exploring<br/>Equilibrium reached prematurely"]

    E["Solution 1: Minibatch discrimination"] -->|Add| F["Discriminator sees batch diversity<br/>Rewards variety"]
    G["Solution 2: Spectral normalization"] -->|Add| H["Constraint on D gradients<br/>Smoother training"]
    I["Solution 3: Multiple discriminators"] -->|Use| J["Each discriminator penalizes<br/>different artifacts"]

    style D fill:#ffebee
    style E fill:#e8f5e9
    style G fill:#e8f5e9
    style I fill:#e8f5e9
```

### Discriminator vs Generator Balance

| Aspect | Weak D | Balanced | Strong D |
|--------|--------|----------|----------|
| **Generator gradient** | Weak signal, slow learning | Good signal, fast learning | Saturated, no learning |
| **Fake sample quality** | Poor (no pressure) | Excellent | May collapse (all variations) |
| **Training stability** | Unstable, diverges | Stable | Unstable, mode collapse |
| **Convergence speed** | Slow | Optimal | Slow or fails |
| **How to fix** | Train D more | Nothing needed | Use better loss function |

### Training Tricks

```mermaid
graph LR
    A["GAN Training Challenges"] -->|Initialization| B["Careful weight init<br/>Standard: He init"]
    A -->|Learning rates| C["Separate LR for D and G<br/>Often: G LR > D LR"]
    A -->|Batch normalization| D["Essential for G<br/>Careful in D (no in output)"]
    A -->|Architecture| E["Strided convolutions<br/>Avoid pooling (introduces artifacts)"]

    style A fill:#f3e5f5
    style B fill:#e1f5ff
    style C fill:#e1f5ff
    style D fill:#e1f5ff
    style E fill:#e1f5ff
```

### Evaluation Metrics Trade-offs

| Metric | Measures | Pros | Cons |
|--------|----------|------|------|
| **Inception Score (IS)** | Sample quality | Easy to compute | Biased toward IS-trained images |
| **Fréchet Inception Distance (FID)** | Distribution quality | Correlates with human judgment | Requires reference dataset |
| **Kernel Inception Distance (KID)** | Distribution quality | Less biased than FID | Slower to compute |
| **Precision/Recall** | Diversity vs quality | Disentangled metrics | Requires classifiers |
| **Human evaluation** | True quality | Ground truth | Expensive, subjective |"""
    },

    "39-time-series-forecasting": {
        "content": """## Architecture / Trade-offs

### Time Series Forecasting Approaches

```mermaid
graph TD
    A["Forecasting Method"] -->|Statistical| B["ARIMA/SARIMA"]
    A -->|Exponential| C["Exponential Smoothing"]
    A -->|Machine Learning| D["Tree-based<br/>XGBoost, LightGBM"]
    A -->|Deep Learning| E["RNN-based<br/>LSTM, GRU"]
    A -->|Deep Learning| F["Attention-based<br/>Transformer, Temporal Conv"]

    B -->|Best for| G["Stationary series<br/>Clear patterns"]
    C -->|Best for| G
    D -->|Best for| H["Non-linear patterns<br/>Multiple features"]
    E -->|Best for| I["Sequence dependencies<br/>Complex temporal structure"]
    F -->|Best for| J["Long-range dependencies<br/>Multiple steps ahead"]

    style G fill:#e1f5ff
    style H fill:#fff3e0
    style I fill:#f3e5f5
    style J fill:#e8f5e9
```

### Model Complexity vs Data Requirements

| Model | Parameters | Data Needed | Interpretability | Training Speed |
|-------|-----------|------------|-----------------|-----------------|
| **Exponential Smoothing** | Few (2-3) | 10-20 observations | Very high | Very fast |
| **ARIMA** | Few (3-5) | 50-100+ observations | High | Fast |
| **Linear/Ridge Regression** | Medium | 100+ observations | High | Very fast |
| **Tree ensemble** | High (100-1000) | 500+ observations | Medium | Medium |
| **LSTM** | Very high (10K+) | 1000+ observations | Low | Slow |
| **Transformer** | Extreme (100K+) | 10K+ observations | Very low | Very slow |

### Univariate vs Multivariate

```mermaid
graph TD
    A["Time Series Data"] -->|Single variable| B["Univariate<br/>y(t) = f(y(t-1), y(t-2), ...)"]
    A -->|Multiple variables| C["Multivariate<br/>y(t) = f(X(t-1), X(t-2), ...)"]

    B -->|Model| D["ARIMA, Exponential Smoothing<br/>LSTM on single feature"]
    C -->|Model| E["VAR, Deep Learning<br/>Multi-input RNN/Transformer"]

    B -->|Advantage| F["Simpler, fewer parameters<br/>Less data needed"]
    C -->|Advantage| G["Captures cross-variable<br/>relationships"]

    style F fill:#e1f5ff
    style G fill:#e1f5ff
```

### Forecasting Horizons

| Horizon | Difficulty | Accuracy Decay | Method |
|---------|-----------|-----------------|---------|
| **1-step (nowcasting)** | Easy | Minimal | Any method works |
| **Short-term (1-7 steps)** | Medium | Gradual | Statistical or basic ML |
| **Medium-term (1-4 weeks)** | Hard | Significant | ML or deep learning |
| **Long-term (1-12 months)** | Very hard | Severe | External factors essential |
| **Multi-step ahead** | Hardest | Compounding error | Teacher forcing or ensemble |

### Error Accumulation: Direct vs Recursive

```mermaid
graph LR
    A["Forecast horizon h steps"] -->|Direct<br/>One model per step| B["Horizon-specific models<br/>Slower training"]
    A -->|Recursive<br/>One step, iterate| C["Use predictions as input<br/>Error accumulation"]

    B -->|Error| D["Independent errors<br/>No accumulation"]
    C -->|Error| E["Compounding errors<br/>Exponential decay"]

    B -->|Accuracy| F["Better for all horizons"]
    C -->|Efficiency| G["More efficient<br/>Single model"]

    style D fill:#e8f5e9
    style E fill:#ffebee
```

### Handling Non-stationarity

| Technique | Use When | Trade-off |
|-----------|----------|-----------|
| **Differencing** | Trend present | May lose information |
| **Detrending** | Linear trend | Assumes trend is linear |
| **Seasonal decomposition** | Seasonality evident | Assumes fixed seasonality |
| **Adaptive models** | Patterns change over time | More complex |
| **Online learning** | Streaming data | Need continuous updates |
| **Time window** | Recent data matters most | Loses historical patterns |"""
    },

    "40-anomaly-detection": {
        "content": """## Architecture / Trade-offs

### Anomaly Detection Approaches

```mermaid
graph TD
    A["Anomaly Detection Methods"] -->|Statistical| B["Gaussian distribution<br/>Isolation Forest"]
    A -->|Distance-based| C["K-NN distance<br/>Local Outlier Factor"]
    A -->|Reconstruction| D["Autoencoder<br/>VAE"]
    A -->|Supervised| E["Classification<br/>Random Forest"]
    A -->|Density-based| F["Kernel Density<br/>One-class SVM"]

    B -->|Best for| G["Known distribution<br/>Univariate data"]
    C -->|Best for| H["Local anomalies<br/>High-dimensional"]
    D -->|Best for| I["Complex patterns<br/>Unsupervised"]
    E -->|Best for| J["Labeled anomalies<br/>Production systems"]
    F -->|Best for| K["Non-parametric<br/>Complex boundaries"]

    style G fill:#e1f5ff
    style H fill:#fff3e0
    style I fill:#f3e5f5
```

### Supervised vs Unsupervised

| Aspect | Unsupervised | Supervised |
|--------|-------------|-----------|
| **Labels needed** | None | Full anomaly labels |
| **Class imbalance** | N/A (no labels) | Extreme (1-10% anomalies) |
| **False positive cost** | May be high | Controllable |
| **False negative cost** | May miss anomalies | Controllable |
| **Adaptability** | Good (learns normal) | Fixed to training anomalies |
| **Deployment** | Immediate | Need labeled data |
| **Interpretability** | May be unclear | Can explain why anomalous |

### Batch vs Online Detection

```mermaid
graph TD
    A["Anomaly Detection Setup"] -->|Batch| B["Process entire dataset<br/>Compute statistics once"]
    A -->|Online/Streaming| C["Process stream<br/>Update model continuously"]

    B -->|Pros| D["More data for thresholds<br/>Better statistics"]
    B -->|Cons| E["No real-time detection<br/>Offline only"]

    C -->|Pros| F["Real-time alerts<br/>Immediate response"]
    C -->|Cons| G["Limited history<br/>Concept drift challenges"]

    D -->|Use| H["Historical analysis<br/>Post-hoc investigation"]
    F -->|Use| I["Production systems<br/>Real-time monitoring"]

    style D fill:#e1f5ff
    style F fill:#e8f5e9
```

### Point vs Contextual Anomalies

| Type | Definition | Detection | Challenge |
|------|-----------|-----------|-----------|
| **Point anomaly** | Single value far from distribution | Easy (statistical) | Simple cases only |
| **Contextual anomaly** | Value unusual in context | Hard (requires context) | Context-dependent threshold |
| **Collective anomaly** | Group of values unusual together | Very hard (collective) | Need multi-variate model |

### Threshold Selection Methods

```mermaid
graph LR
    A["Setting Anomaly Threshold"] -->|Statistical| B["Standard deviations<br/>e.g., 3σ for Gaussian"]
    A -->|Percentile| C["Top 1% or 5% of scores<br/>Model-independent"]
    A -->|ROC/PR curve| D["Optimize F1 or precision<br/>Needs labeled validation"]
    A -->|Isolation Forest| E["Anomaly score<br/>Interpretation unclear"]
    A -->|Reconstruction error| F["Threshold on error magnitude<br/>Adaptive possible"]

    B -->|Best for| G["Gaussian data"]
    C -->|Best for| H["Unknown distribution"]
    D -->|Best for| I["Labeled anomalies"]
    F -->|Best for| J["Learned models"]

    style D fill:#e8f5e9
```

### Imbalanced Learning Solutions

| Solution | How It Works | Pros | Cons |
|----------|------------|------|------|
| **Threshold moving** | Change decision boundary | Simple | May not transfer |
| **Class weighting** | Higher weight for anomalies | Natural | Requires tuning |
| **Oversampling anomalies** | Duplicate anomaly samples | Increases data | Can overfit |
| **Undersampling normals** | Reduce normal samples | Faster training | Loss of information |
| **Ensemble methods** | Combine multiple models | Robust | More complex |
| **One-class learning** | Learn normal only | Natural fit | Harder to train |"""
    },
}

# Add more concepts for LLM and Agentic AI (abbreviated for space)
ARCHITECTURES.update({
    "36-adversarial-robustness": {
        "content": """## Architecture / Trade-offs

### Attack vs Defense Landscape

```mermaid
graph TD
    A["Adversarial Threats"] -->|Input-level| B["Prompt injection<br/>Jailbreak attempts"]
    A -->|Token-level| C["Subtle perturbations<br/>Typos, paraphrasing"]
    A -->|Semantic| D["Meaning-preserving<br/>Rephrase harmful requests"]

    E["Defense Layers"] -->|Input filtering| F["Block known jailbreaks<br/>Keyword detection"]
    E -->|Prompt engineering| G["Explicit safety instructions<br/>Role definition"]
    E -->|Model training| H["Adversarial fine-tuning<br/>RLHF on attacks"]
    E -->|Monitoring| I["Anomaly detection<br/>Request logging"]
    E -->|Human review| J["Escalation workflow<br/>Manual verification"]

    B -->|Defended by| F
    B -->|Defended by| G
    C -->|Defended by| H
    D -->|Defended by| I
    D -->|Defended by| J

    style B fill:#ffebee
    style F fill:#e8f5e9
```

### Defense-in-Depth Strategy

| Layer | Purpose | Coverage | Cost |
|-------|---------|----------|------|
| **Input validation** | Block known attacks | 20-30% of attacks | Very low |
| **Prompt engineering** | Guide model behavior | 30-50% of attacks | Low |
| **Model robustness** | Improve model resistance | 40-60% of attacks | High |
| **Rate limiting** | Prevent brute force | Slow down attackers | Low |
| **Human review** | Catch edge cases | 90%+ (but expensive) | Very high |
| **Monitoring** | Detect novel attacks | Enables response | Medium |

### Red Teaming Architecture

```mermaid
graph TD
    A["Model Under Test"] -->|Expose to| B["Red Team<br/>Security experts"]
    B -->|Find vulnerabilities<br/>via systematic probing| C["Vulnerability Log"]
    C -->|Classify| D["Severity Level<br/>High/Medium/Low"]
    D -->|Prioritize| E["Fix Queue"]
    E -->|Update| F["Retrain/Fine-tune"]
    F -->|Validate| A

    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e1f5ff
```"""
    },

    "37-knowledge-distillation": {
        "content": """## Architecture / Trade-offs

### Knowledge Transfer Mechanisms

```mermaid
graph TD
    A["Teacher Model<br/>Large, accurate"] -->|Extract knowledge| B["Soft targets<br/>Probability distributions"]
    A -->|Feature extraction| C["Intermediate representations<br/>Hidden layers"]

    B -->|Temperature scaling| D["T=1: Hard targets<br/>T>1: Soft targets<br/>Higher T = more information"]

    D -->|Train| E["Student Model<br/>Small, fast"]
    C -->|Attention transfer| E

    E -->|Result| F["Faster inference<br/>Better accuracy than training alone"]

    style A fill:#fff3e0
    style E fill:#f3e5f5
    style D fill:#e1f5ff
```

### Model Size vs Performance

| Student Size | Knowledge Retention | Speedup | Accuracy vs Teacher |
|--------------|-------------------|---------|-------------------|
| **Large (90% params)** | ~95% | 1.1x | -0.5% |
| **Medium (50% params)** | ~85% | 2x | -2% |
| **Small (10% params)** | ~70% | 10x | -5-10% |
| **Tiny (1% params)** | ~40% | 50x | -20%+ |

### Distillation Loss Components

```mermaid
graph LR
    A["Total Loss"] -->|Distillation| B["L_dist = KL(p_student || p_teacher)<br/>Match soft targets"]
    A -->|Ground Truth| C["L_task = CrossEntropy(y, p_student)<br/>Match labels"]

    B -->|Weight α| D["Balance trade-off"]
    C -->|Weight 1-α| D

    D -->|α=0| E["Pure distillation<br/>Depends on teacher"]
    D -->|α=0.5| F["Balanced<br/>Best for limited teacher quality"]
    D -->|α=1| G["Pure supervision<br/>Ignore teacher"]

    style E fill:#fff3e0
    style F fill:#e8f5e9
    style G fill:#ffebee
```"""
    },

    "38-neural-architecture-search": {
        "content": """## Architecture / Trade-offs

### NAS Search Strategy

```mermaid
graph TD
    A["Architecture Search"] -->|Random Search| B["Baseline<br/>Random sampling"]
    A -->|Evolutionary| C["Genetic algorithm<br/>Population-based"]
    A -->|Reinforcement Learning| D["Controller network<br/>Learns to propose architectures"]
    A -->|Differentiable| E["Continuous relaxation<br/>Gradient-based optimization"]

    B -->|Speed| F["Slowest"]
    C -->|Speed| G["Medium"]
    D -->|Speed| G
    E -->|Speed| H["Fastest"]

    F -->|Quality| I["Baseline"]
    G -->|Quality| J["Good"]
    H -->|Quality| K["Excellent"]

    style E fill:#e8f5e9
```

### Search Space Design

| Dimension | Options | Impact |
|-----------|---------|--------|
| **Number of layers** | 2-20 | Depth complexity |
| **Layer type** | Conv, Dense, Attention | Architecture diversity |
| **Kernel size** | 1, 3, 5, 7 | Receptive field |
| **Filters/units** | 16-512 | Model capacity |
| **Skip connections** | Yes/No | Information flow |
| **Activation** | ReLU, GELU, Swish | Non-linearity |"""
    },
})

def enhance_architecture(filepath, title):
    """Enhance Architecture section with detailed content."""

    filename = filepath.split('/')[-1].replace('.md', '')

    if filename not in ARCHITECTURES:
        return False

    arch_content = ARCHITECTURES[filename]["content"]

    # Read current file
    with open(filepath) as f:
        content = f.read()

    # Replace Architecture / Trade-offs section
    # Find the section and replace until the next ## heading
    pattern = r'(## Architecture / Trade-offs\n\n).*?(\n## Interview Q&A)'
    replacement = rf'\1{arch_content}\2'

    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Write updated file
    with open(filepath, 'w') as f:
        f.write(content)

    return True

def main():
    """Enhance all newly created concepts."""

    concepts_to_enhance = [
        (31, "Q-Learning", "ai"),
        (32, "Policy Gradients", "ai"),
        (33, "Actor-Critic Methods", "ai"),
        (34, "Graph Neural Networks", "ai"),
        (35, "Causal Inference", "ai"),
        (36, "Probabilistic Graphical Models", "ai"),
        (37, "Variational Autoencoders", "ai"),
        (38, "Generative Adversarial Networks", "ai"),
        (39, "Time Series Forecasting", "ai"),
        (40, "Anomaly Detection", "ai"),
        (36, "Adversarial Robustness", "llm"),
        (37, "Knowledge Distillation", "llm"),
        (38, "Neural Architecture Search", "llm"),
    ]

    print("=== Enhancing Architecture / Trade-offs Sections ===\n")

    enhanced = 0
    for num, title, section in concepts_to_enhance:
        slug = title.lower().replace(' ', '-').replace('&', 'and').replace('(', '').replace(')', '')
        filepath = f"{BASE}/{section}/concepts/{num:02d}-{slug}.md"

        if not os.path.exists(filepath):
            print(f"  ⊘ {num:02d}-{slug}.md (not found)")
            continue

        try:
            if enhance_architecture(filepath, title):
                print(f"  ✓ {num:02d}-{slug}.md")
                enhanced += 1
            else:
                print(f"  ⊘ {num:02d}-{slug}.md (no expansion data)")
        except Exception as e:
            print(f"  ✗ {num:02d}-{slug}.md (error: {e})")

    print(f"\n✅ Enhanced {enhanced} architecture sections")

if __name__ == "__main__":
    main()
