#!/usr/bin/env python3
"""Enhance Detailed Explanation and Core Intuition for AI fundamentals 29-40."""

import os
import re

BASE = "/home/sbisw/github/interviewprep-ml"

# Comprehensive expansions for AI concepts 29-40
AI_EXPANSIONS_29_40 = {
    "29-reinforcement-learning-basics": {
        "detailed": """Reinforcement Learning (RL) enables agents to learn optimal decision-making by interacting with an environment and receiving rewards. Unlike supervised learning which requires labeled examples, RL agents learn through trial-and-error: take action, receive reward/penalty, update understanding. This approach mirrors how humans and animals learn: exploring behaviors, receiving feedback, gradually improving. RL powers game-playing agents (AlphaGo, game AIs), robotic control, and autonomous systems.

The key concepts are: agents (decision-makers), environments (respond to actions), states (current situation), actions (choices), rewards (feedback signals), and policies (decision rules). The agent's goal is to maximize cumulative reward over time, not just immediate reward—this distinction makes planning and long-term thinking necessary. Exploration vs exploitation trade-off is central: exploring new strategies discovers better options but risks short-term losses; exploiting known good strategies is safe but may miss better options.

RL is powerful but challenging: requires careful reward design (wrong rewards lead to unintended behaviors), sample-inefficient (needs many interactions to learn), and non-stationary (environment changes, feedback depends on agent actions). Understanding RL helps explain how machines can learn autonomously and why alignment (ensuring learned behavior matches intent) is critical for advanced AI systems.""",
        "intuition": """Reinforcement Learning is like raising a dog: reward good behavior (sit, stay), penalize bad behavior (biting), and through repetition, the dog learns. The dog doesn't read instructions—it learns by trying behaviors and receiving feedback. The goal is teaching the dog to maximize rewards (treats, praise) through its actions."""
    },

    "30-markov-decision-processes": {
        "detailed": """Markov Decision Processes (MDPs) formalize sequential decision-making problems where outcomes depend on the current state and action but not on history. The Markov property (memorylessness) means the future is independent of the past given the current state, enabling efficient algorithms. MDPs model many real-world problems: navigation (state = position, action = move direction), finance (state = portfolio, action = buy/sell), game-playing (state = board configuration, action = move).

An MDP is defined by: states S, actions A, transition probabilities P(s'|s,a) (where we end up), rewards R(s,a,s') (what we receive), and discount factor γ (future vs immediate value). The discount factor balances short-term and long-term rewards: γ=0 cares only about immediate reward, γ=1 treats all times equally (can diverge), γ=0.9 is typical. Value iteration (Bellman equations) provides optimal policies by bootstrapping: value of state = immediate reward + discounted value of next state.

MDPs are the mathematical foundation for RL and planning algorithms. Understanding the Markov property helps recognize when it applies (often violated in practice: Partially Observable MDPs, game history matters). The Bellman equations are intellectually important even if rarely implemented directly. Modern practitioners use approximate value functions (neural networks) to handle large state spaces, but understanding exact algorithms clarifies how approximations work.""",
        "intuition": """MDPs are like a board game: each position (state) has possible moves (actions), each move leads to the next position (transition probability), and at each position you earn points (reward). The goal is choosing moves that maximize total points. The Markov property means your next position depends only on current position and move, not how you got there."""
    },

    "31-q-learning": {
        "detailed": """Q-Learning is a foundational off-policy RL algorithm that learns optimal decision-making by estimating Q-values: expected cumulative reward for taking action a in state s. The algorithm is simple but powerful: repeatedly take action, observe reward and next state, update Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)], then repeat. Off-policy means it learns the optimal policy while following a different exploratory policy, making it sample-efficient.

The core insight is that Q-values can be learned iteratively: the value of a state-action pair improves by comparing actual return (immediate reward + discounted future value) with current estimate, then moving toward the better estimate. Exploration-exploitation trade-off is managed with ε-greedy: explore with probability ε, exploit with probability 1-ε. Deep Q-Networks (DQN) use neural networks as function approximators for large state spaces, combined with experience replay (storing and sampling past experiences) and target networks (slowly updated copies) for stability.

Q-Learning is crucial to understand because it bridges simple models and deep RL. The algorithm introduces concepts (value iteration, off-policy learning, temporal difference learning) that appear throughout modern RL. Limitations include overestimation bias (which Double Q-Learning addresses) and difficulty with continuous action spaces. Most modern practical RL uses variants or alternatives, but Q-Learning concepts remain fundamental.""",
        "intuition": """Q-Learning is like learning which chess moves are good by playing repeatedly: estimate value of each (position, move) pair, try moves, see results, update estimates toward observed outcomes. The value estimate improves by bootstrapping: using estimated values of future positions to improve current estimates. It's learning value through iterative comparison of expectations vs. reality."""
    },

    "32-policy-gradients": {
        "detailed": """Policy Gradient methods directly optimize the policy (probability distribution over actions) by computing gradients with respect to policy parameters. Rather than learning value functions then deriving policies (value-based RL), policy-gradient methods directly improve the policy. The core idea: compute how policy parameters affect expected return, then take steps to increase that return. REINFORCE is the simplest algorithm: sample trajectories, compute returns, scale policy gradients by returns.

Policy gradients handle continuous action spaces naturally (policy outputs action probabilities or mean/variance of action distribution), unlike Q-Learning which requires discrete actions or complex approximations. Variance is high (samples are noisy) so variance reduction techniques are crucial: baselines subtract an estimate of value (reduces variance without changing gradient), advantage functions estimate how much better an action is than average. Actor-Critic methods combine policy gradients (actor: which action to take) with value functions (critic: how good is this state).

Policy gradients form the foundation of modern RL: Proximal Policy Optimization (PPO), Trust Region Policy Optimization (TRPO), and others. Understanding the policy gradient theorem (gradient of expected return = expected gradient of return) is important for understanding the landscape. Limitations include high variance (needs many samples) and local optima (hill-climbing in policy space). Modern methods add techniques like entropy regularization (encourages exploration) and clipped policy updates for stability.""",
        "intuition": """Policy gradients are like tweaking a recipe based on how good the food turns out: if dish tastes good, increase likelihood of parameters that led to it; if bad, decrease likelihood. It's directly improving the decision-making strategy (policy) based on observed outcomes, not learning values."""
    },

    "33-actor-critic-methods": {
        "detailed": """Actor-Critic methods combine policy gradients (actor: learns what action to take) with value functions (critic: evaluates quality of states/actions), balancing high variance of policy gradients with bias of value functions. The actor (policy network) generates actions; the critic (value network) estimates how good those actions were (temporal difference error). The actor uses critic's estimates to reduce variance in policy gradients. Advantage Actor-Critic (A2C) and Asynchronous A3C are practical algorithms combining both.

The actor-critic architecture mirrors human decision-making: the actor (conscious decision) is guided by the critic (evaluation/reward). Training uses two loss functions: actor loss (improve policy toward actions the critic values highly) and critic loss (make value estimates accurate). Entropy regularization encourages exploration by adding bonus for diverse policies. The advantage function (actual return - estimated value) reduces variance: tells actor which actions are better than average, focusing learning on important decisions.

Actor-Critic methods are widely used because they balance sample efficiency (value functions guide learning) with flexibility (policy gradients handle continuous actions). Understanding the actor-critic decomposition helps recognize similar architectures elsewhere: many modern systems combine prediction networks (critic) with generation networks (actor). Challenges include stable learning (both networks must improve together), function approximation errors in critic affecting actor, and tuning trade-offs between actor/critic learning rates.""",
        "intuition": """Actor-Critic methods are like having a performer (actor) and a coach (critic): the actor tries different performances, the coach evaluates each attempt and gives feedback. The actor improves based on feedback, and the coach improves at evaluating performances. Together they become better than either alone."""
    },

    "34-graph-neural-networks": {
        "detailed": """Graph Neural Networks (GNNs) extend neural networks to graph-structured data (networks of nodes and edges), learning representations that respect graph structure. Many real-world problems are naturally graphs: social networks (people = nodes, friendships = edges), molecules (atoms = nodes, bonds = edges), knowledge graphs (entities = nodes, relationships = edges). Standard neural networks ignore graph structure; GNNs exploit it for better representations and predictions.

GNNs work through message-passing: each node aggregates information from neighbors, then updates its representation. Graph Convolutional Networks (GCN) apply convolution-like operations on graphs. GraphSAGE samples neighbors for scalability. Graph Attention Networks (GAT) learn attention weights for important neighbors. Permutation invariance (output doesn't change if you reorder nodes) is key: aggregation functions must be invariant (like sum, max) rather than order-dependent. Different readout layers (graph-level predictions) aggregate node representations: mean pooling (average all nodes), hierarchical pooling, or attention-based.

GNNs are revolutionizing machine learning on structured data, outperforming traditional methods on molecules, knowledge graphs, social networks. Understanding permutation invariance clarifies why GNNs work: treating nodes equivalently respects symmetry. Modern architectures add techniques: skip connections, spectral methods, higher-order convolutions. Challenges include scalability to large graphs, learning robust representations, and oversmoothing (representations become similar after many layers). GNNs represent a shift from fixed-size inputs (images, text) to variable-size structured inputs.""",
        "intuition": """Graph Neural Networks are like gossiping in a community: each person learns from their friends, then shares with their friends, and gradually information propagates through the network. Different people learn different things based on who they're connected to and what they value in their friends' opinions."""
    },

    "35-causal-inference": {
        "detailed": """Causal inference determines which variables cause which outcomes (beyond correlation), crucial for understanding mechanisms and making effective interventions. The fundamental problem: from observational data, we only see correlations, but correlation doesn't imply causation (A correlated with B could mean A→B, B→A, or C→A,C→B). Causal inference uses graphical models (DAGs: directed acyclic graphs) to encode causal assumptions, then applies adjustment strategies to estimate effects of interventions.

Randomized controlled trials (RCTs) estimate causal effects by randomly assigning interventions (breaks all back-door paths from intervention to outcome). However, RCTs are expensive/unethical for many questions. Observational methods use causal graphs to identify confounders (variables affecting both intervention and outcome) and use adjustment (conditioning on confounders) or methods like instrumental variables (variables affecting outcome only through intervention) to estimate causal effects. The do-calculus (Pearl's framework) determines whether a causal effect is identifiable from observational data given a causal model.

Causal inference is increasingly important for policy decisions (what intervention improves outcomes?), medicine (does treatment cause cure?), and fairness (does algorithm discriminate?). Understanding that correlation ≠ causation is critical. Challenges include model misspecification (assumed causal graph might be wrong), hidden confounders, and multiple intervention pathways. Modern methods combine causal reasoning with machine learning for estimation, but understanding the principles is essential for responsible conclusions.""",
        "intuition": """Causal inference is like detective work: you observe correlations (X and Y are often together), but you need to figure out the mechanism (X causes Y, Y causes X, or both caused by Z). Causal graphs show your theories about mechanisms, and different techniques extract causal effects from data depending on those theories."""
    },

    "36-probabilistic-graphical-models": {
        "detailed": """Probabilistic Graphical Models (PGMs) represent joint probability distributions using graphs: nodes are variables, edges indicate dependencies (compact representation). Bayesian Networks (directed graphs) encode conditional independence: ancestors influence descendants; unrelated variables are independent given parents. Markov Networks (undirected graphs) encode Markov blankets: variables are independent of non-neighbors given neighbors. PGMs enable efficient inference and learning even with many variables.

Inference (computing probabilities) in PGMs uses algorithms exploiting structure: belief propagation (message-passing on trees/DAGs) computes probabilities exactly in polynomial time; loopy belief propagation (on graphs with cycles) is approximate. Variable elimination orders variables and eliminates them sequentially, reducing computation. Sampling methods (MCMC) approximate posteriors. Learning parameters (given structure) is often tractable; learning structure (which edges to include) is harder. Latent Variable Models (where some variables are unobserved) add complexity but enable discovering hidden factors.

PGMs are foundational for probabilistic reasoning: HMMs (Markov chains with observations), Kalman filters (continuous state spaces), Topic models (LDA). Modern deep learning often replaces PGMs for large, complex data, but PGMs remain valuable for interpretability, structured reasoning, and when data is limited. Understanding PGM principles helps recognize that many modern methods (attention mechanisms, diffusion models) implicitly use graphical model concepts. Challenges include scalability (inference is NP-hard in general), model selection, and computational efficiency on large graphs.""",
        "intuition": """Probabilistic Graphical Models are like family trees showing genetic inheritance: parents influence children, siblings are correlated through parents. If you know the parents' genetics, you don't need parents' parents' genetics—that's the independence structure. Different connections mean different influences, enabling efficient computation."""
    },

    "37-variational-autoencoders": {
        "detailed": """Variational Autoencoders (VAEs) are generative models learning to map data to a latent representation and generate new data from that representation. Unlike standard autoencoders (encoder → latent → decoder) which just compress, VAEs learn probability distributions: encoder outputs latent distribution parameters (mean, variance), decoder generates data from samples. The elegance: sampling from latent space generates diverse realistic data.

VAEs use variational inference: the encoder approximates intractable true posterior, the decoder learns likelihood. Loss has two terms: reconstruction loss (decoder should reconstruct data) and KL divergence (encoder distribution should be close to prior, usually Gaussian). This balances accurate reconstruction (low reconstruction loss) with learning useful latent structure (low KL divergence). Reparameterization trick (sample latent from encoder, backprop through sampling) enables gradient-based training. VAEs learn disentangled representations (separate latent factors for separate data factors) better than standard autoencoders.

VAEs enable principled generation and interpolation: smoothly transition between latents interpolates in data space. Applications include image generation, data augmentation, anomaly detection. Challenges include blurry generated images (optimization biases toward reconstruction), difficulty learning complex posteriors, and posterior collapse (KL divergence becomes zero, latent space unused). Modern improvements: β-VAE increases KL weight for better disentanglement, hierarchical VAEs stack latents, neural autoregressive models for better decoders. Understanding VAEs clarifies deep generative models and variational inference principles.""",
        "intuition": """VAEs are like compressing a book into notes and regenerating books from notes: the notes (latent representation) capture essential information. Adding noise to notes (sampling from distribution) creates variations. The challenge is finding notes detailed enough to reconstruct accurately but simple enough to generate variations easily."""
    },

    "38-generative-adversarial-networks": {
        "detailed": """Generative Adversarial Networks (GANs) train two competing networks: generator (creates fake data trying to fool discriminator) and discriminator (tries to distinguish real from fake data). This adversarial process creates tension: as generator improves, discriminator must improve, creating positive feedback. The result: generator learns to create highly realistic data. GANs have created photorealistic images, style transfer, and data augmentation.

The training process is a two-player game: discriminator loss penalizes classifying real as real and fake as fake (wants correct classification); generator loss penalizes discriminator correctly classifying fake (generator wants to fool discriminator). At equilibrium, discriminator can't distinguish real from fake, so generator output is indistinguishable from real data. Training is unstable: generator collapses to single mode (ignores variation in real data), discriminator overpowers generator, or vice versa. Techniques stabilize training: spectral normalization, progressive growing (grow networks gradually), hinge loss alternatives.

GANs are powerful but notoriously difficult to train. The adversarial objective is elegant theoretically but unstable practically. Many variants address stability: Wasserstein GANs use better distance metrics; Conditional GANs add class labels; StyleGAN disentangles style from content. Understanding why GANs are unstable (generator has infinite capacity, training is min-max not minimum) helps appreciate engineering solutions. Applications beyond generation: discriminator representations for downstream tasks, feature learning, anomaly detection (real data has low discriminator loss). GANs represent a paradigm shift from explicit likelihood maximization to implicit distribution learning.""",
        "intuition": """GANs are like art school: a forger (generator) creates fake paintings while an art critic (discriminator) learns to detect fakes. As the forger improves, the critic must improve, creating an arms race. Eventually, the forger creates paintings indistinguishable from real art. The competition drives quality."""
    },

    "39-time-series-forecasting": {
        "detailed": """Time Series Forecasting predicts future values given historical observations, crucial for planning and decision-making across finance, weather, demand. Time series have structure that standard models ignore: autocorrelation (values depend on recent history), seasonality (patterns repeat), trends (long-term direction). These patterns enable modeling and prediction but violate independence assumptions. Traditional methods (ARIMA, SARIMA, exponential smoothing) explicitly model these patterns; modern deep learning approaches (RNNs, Transformers) learn patterns implicitly.

Challenges are fundamental: future depends on factors not in historical data (sudden events, policy changes), distributions may change (non-stationary), high variance in nature (weather is inherently unpredictable). Model choices trade off: simple models (moving average, exponential smoothing) are interpretable, stable, but miss complex patterns; complex models (deep learning) capture patterns but need more data and careful validation. Walk-forward validation (train on past, test on recent, repeat) prevents overfitting to historical data. Probabilistic forecasting (predicting distributions not point estimates) quantifies uncertainty.

Time series forecasting is distinct from standard supervised learning: temporal structure matters, evaluation must respect time ordering, and fresh data continuously refutes old models. Understanding the limits of forecasting (some systems are inherently unpredictable) prevents over-confidence. Multivariate forecasting (multiple related series) adds complexity but enables leveraging relationships. Modern practical approaches combine statistical methods (handling known patterns) with deep learning (learning complex relationships), ensemble techniques (averaging predictions), and domain expertise (known seasonal adjustments).""",
        "intuition": """Time series forecasting is like predicting weather: past patterns (seasons, trends) provide clues, but you can't predict everything (storms are surprising). Better predictions combine pattern recognition (summers are hot, winters cold) with understanding limits (perfect prediction is impossible). Recent history matters most, but very old patterns reveal long-term trends."""
    },

    "40-anomaly-detection": {
        "detailed": """Anomaly Detection identifies unusual data points (outliers, anomalies) that deviate from normal patterns, crucial for fraud detection, equipment failure prediction, security. Normal data has patterns; anomalies violate those patterns. Approaches differ: supervised (label anomalies during training), unsupervised (find outliers without labels), or semi-supervised (labels only for normal, model flags deviations). Most real-world problems are unsupervised: anomalies are rare and unknown, so labeling is expensive and incomplete.

Unsupervised methods assume anomalies are rare and different from normal data. Isolation Forest isolates anomalies (separate from normal data) recursively; Local Outlier Factor detects points with low density relative to neighbors; One-Class SVM finds hyperplane separating normal from everything else. Deep learning approaches use autoencoders (anomalies have high reconstruction error) or Variational Autoencoders (anomalies have high log-likelihood under reconstruction). Time-series anomalies use forecasting: actual vs predicted values reveal anomalies. Streaming anomaly detection must detect changes in real-time with limited memory.

Anomaly detection is conceptually simple but practically complex: defining 'normal' is hard (operational drift, concept change), evaluation is tricky (ground truth is rare), false positive/negative costs matter. Combining methods (ensemble) often improves performance. Domain knowledge is crucial: knowing what anomalies matter in your context guides feature engineering and threshold selection. Understanding that anomalies span many forms (pointwise, contextual, collective) helps choose methods. Modern approaches incorporate deep learning for automatic feature learning but statistical methods remain valuable for interpretability and small-data regimes.""",
        "intuition": """Anomaly detection is like a security guard: learns what normal behavior looks like (people walking normally, cars driving normally), then flags unusual behavior (person running, car going backward). The challenge is defining 'normal' broadly enough (different people walk differently) but tightly enough (actually catch anomalies)."""
    },
}

def enhance_ai_29_40():
    """Enhance AI fundamentals 29-40."""

    print("=== Enhancing AI Fundamentals Concepts 29-40 ===\n")

    enhanced = 0
    for num in range(29, 41):
        num_str = f"{num:02d}"
        concepts_dir = f"{BASE}/ai/concepts"
        matching_files = [f for f in os.listdir(concepts_dir) if f.startswith(num_str)]

        if not matching_files:
            print(f"  ⊘ {num_str}: No file found")
            continue

        filename = matching_files[0]
        slug = filename.replace('.md', '').replace(f'{num_str}-', '')
        filepath = os.path.join(concepts_dir, filename)

        full_slug = f"{num_str}-{slug}"
        if full_slug not in AI_EXPANSIONS_29_40:
            print(f"  ⊘ {full_slug}: No expansion data")
            continue

        with open(filepath) as f:
            content = f.read()

        exp = AI_EXPANSIONS_29_40[full_slug]
        detailed = exp["detailed"]
        intuition = exp["intuition"]

        # Replace Detailed Explanation
        content = re.sub(
            r'(## Detailed Explanation\n\n).*?(\n\n## Core Intuition)',
            rf'\1{detailed}\2',
            content,
            flags=re.DOTALL
        )

        # Replace Core Intuition
        content = re.sub(
            r'(## Core Intuition\n\n).*?(\n\n## How It Works)',
            rf'\1{intuition}\2',
            content,
            flags=re.DOTALL
        )

        with open(filepath, 'w') as f:
            f.write(content)

        print(f"  ✓ {full_slug}")
        enhanced += 1

    print(f"\n✅ Enhanced {enhanced} AI concepts (29-40)")

if __name__ == "__main__":
    enhance_ai_29_40()
