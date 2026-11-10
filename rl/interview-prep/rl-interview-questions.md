# Reinforcement Learning Interview Questions

This bank covers core RL reasoning, stability, exploration, offline data, and alignment. Scenario prompts require explicit assumptions about the environment and reward.

### 1. Specify an MDP
**Prompt:** Define the MDP for a robot navigating a grid with obstacles and a goal.
**Role/domain:** Reinforcement learning; foundations. **Difficulty:** Easy.
**Expected answer:** Specify states, actions, transition probabilities, reward, discount factor, and terminal conditions. State must contain enough information for the Markov property; if velocity or hidden map state affects transitions, include it or acknowledge partial observability.
**Evaluation:** Complete tuple, Markov assumption, terminal semantics.
**Follow-up:** When should this be modeled as a POMDP?
**Common mistake:** Defining state as only the agent’s location when dynamics depend on omitted variables.

### 2. Explain a Bellman update
**Prompt:** For a fixed policy, write the value update and explain each term.
**Role/domain:** RL theory. **Difficulty:** Easy.
**Expected answer:** (V_{k+1}(s)=\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)[r(s,a,s')+\gamma V_k(s')]). It averages immediate reward and discounted next-state value under the policy and transition model.
**Evaluation:** Correct expectation, discount, policy versus optimality distinction.
**Follow-up:** How does the optimality operator change the equation?
**Common mistake:** Omitting expectation over stochastic transitions or using max for policy evaluation.

### 3. Debug an unstable off-policy learner
**Prompt:** Q-learning rewards swing sharply and values grow without bound. What do you inspect?
**Role/domain:** Deep RL; debugging. **Difficulty:** Hard.
**Expected answer:** Check reward scale, terminal masking, target computation, learning rate, replay distribution, bootstrap target drift, and overestimation. With function approximation, off-policy bootstrapping can destabilize; use target networks, replay controls, conservative updates, and compare to a tabular or on-policy baseline.
**Evaluation:** Update correctness, instability mechanisms, controlled diagnosis.
**Follow-up:** Why is the “deadly triad” relevant?
**Common mistake:** Tuning only exploration rate when target estimates or terminal handling are wrong.

### 4. Compare SARSA and Q-learning
**Prompt:** In a risky grid with a cliff, why may SARSA learn a safer route than Q-learning under epsilon-greedy behavior?
**Role/domain:** RL; control. **Difficulty:** Medium.
**Expected answer:** SARSA is on-policy and evaluates the exploratory behavior actually followed, including risk of random actions near the cliff. Q-learning targets the greedy next action and estimates the optimal policy despite exploratory behavior, which can favor a shorter but riskier path.
**Evaluation:** On/off-policy target distinction, risk interpretation.
**Follow-up:** Does SARSA always produce a safer policy?
**Common mistake:** Describing Q-learning as learning the behavior policy’s return.

### 5. Choose an exploration strategy
**Prompt:** A bandit has one uncertain arm with potentially high reward and several well-measured mediocre arms. Compare epsilon-greedy and UCB.
**Role/domain:** RL exploration. **Difficulty:** Medium.
**Expected answer:** Epsilon-greedy explores uniformly at random with probability ε; UCB adds an uncertainty bonus that prioritizes under-sampled arms. UCB can be more directed under stationary assumptions; nonstationarity, delayed feedback, and reward constraints may require adaptation.
**Evaluation:** Mechanism, assumptions, deployment caveat.
**Follow-up:** What would Thompson sampling represent probabilistically?
**Common mistake:** Assuming exploration can be eliminated once one arm leads early.

### 6. Debug reward hacking
**Prompt:** A simulated delivery agent maximizes reward by circling near the destination without completing delivery. How do you fix the objective?
**Role/domain:** RL safety; debugging. **Difficulty:** Hard.
**Expected answer:** Inspect reward loophole and terminal criteria, define task success independently, and add validated constraints or reward terms without simply increasing a penalty blindly. Build adversarial scenarios and measure true completion and side effects, not only shaped reward.
**Evaluation:** Objective mismatch, independent success metric, adversarial evaluation.
**Follow-up:** When can potential-based shaping preserve optimal policies?
**Common mistake:** Rewarding more intermediate progress without testing for new loopholes.

### 7. Handle sparse rewards
**Prompt:** An agent rarely receives reward because tasks require a long sequence. What options do you compare?
**Role/domain:** RL training; exploration. **Difficulty:** Medium.
**Expected answer:** Consider curriculum, demonstrations, goal relabeling, structured exploration, or potential-based shaping. Compare sample efficiency and policy quality against an unshaped baseline; ensure shaping does not change the intended optimum or encode unavailable evaluation information.
**Evaluation:** Appropriate methods, invariance, evaluation protocol.
**Follow-up:** Why can naive dense rewards change the task?
**Common mistake:** Assuming any additional reward signal is harmless.

### 8. Interpret an offline RL failure
**Prompt:** A policy trained on logged actions performs poorly in simulation despite high predicted Q-values. Why?
**Role/domain:** Offline RL; debugging. **Difficulty:** Hard.
**Expected answer:** The policy may choose actions outside dataset support, where value estimates extrapolate optimistically. Measure state-action coverage, compare behavior cloning and conservative baselines, use support constraints or conservative objectives, and validate with reliable off-policy evaluation before deployment.
**Evaluation:** Distribution shift, support coverage, safe baselines.
**Follow-up:** Why is random train/test splitting not enough to validate policy value?
**Common mistake:** Treating fitted Q-values as trustworthy for unseen actions.

### 9. Design a safe exploration experiment
**Prompt:** An online recommender can explore but user harm is costly. Propose a rollout and guardrails.
**Role/domain:** RL systems; system design. **Difficulty:** Hard.
**Expected answer:** Start in simulation/offline replay with explicit limits, then use constrained or low-risk exploration on a small eligible cohort. Define user-level safety metrics, exposure caps, stop rules, human oversight, and logging; use randomized evaluation where ethical and valid.
**Evaluation:** Staging, constraints, user-level metrics, stopping.
**Follow-up:** What does a simulator fail to represent?
**Common mistake:** Optimizing long-term reward while ignoring unacceptable short-term harm.

### 10. Diagnose policy evaluation bias
**Prompt:** You estimate a new policy’s value from logged actions, but the logging policy rarely chose the new policy’s preferred action. What is the issue?
**Role/domain:** Offline policy evaluation. **Difficulty:** Hard.
**Expected answer:** Importance weights have poor overlap and high variance; counterfactuals are unsupported. Inspect propensity support and effective sample size, report uncertainty, consider doubly robust estimators only under their assumptions, and avoid a confident value estimate outside support.
**Evaluation:** Positivity/overlap, variance, estimator limits.
**Follow-up:** How can clipping importance weights trade bias for variance?
**Common mistake:** Reporting a single weighted estimate without overlap diagnostics.

### 11. Compare model-based and model-free RL
**Prompt:** When might a learned dynamics model reduce sample cost, and how can it hurt?
**Role/domain:** RL; model-based planning. **Difficulty:** Medium.
**Expected answer:** A model can support simulated rollouts and planning when data are expensive and dynamics are learnable. Model error compounds across long horizons and can exploit inaccuracies; evaluate rollout horizon, uncertainty, and real-environment performance against model-free baselines.
**Evaluation:** Sample efficiency, compounding error, validation.
**Follow-up:** What is a practical way to limit model bias?
**Common mistake:** Treating a high one-step prediction score as evidence of accurate long-horizon planning.

### 12. Choose discount factor semantics
**Prompt:** How does changing γ affect a continuing task with delayed rewards?
**Role/domain:** RL theory. **Difficulty:** Medium.
**Expected answer:** A larger γ values longer-term reward and lengthens the effective horizon; a smaller γ emphasizes near-term reward and can stabilize learning. It should reflect task horizon and objective, not serve as an arbitrary performance knob; continuing tasks need return/convergence assumptions.
**Evaluation:** Horizon meaning, objective alignment, mathematical caveat.
**Follow-up:** How can reward rescaling interact with learning dynamics?
**Common mistake:** Saying γ only changes numerical convergence speed.

### 13. Review an RLHF pipeline
**Prompt:** Describe the roles of demonstrations, preference data, reward model, and policy optimization in a typical RLHF setup.
**Role/domain:** LLM alignment; RLHF. **Difficulty:** Medium.
**Expected answer:** Demonstrations commonly support supervised fine-tuning; comparisons train a reward model; policy optimization improves reward while often constraining divergence from a reference policy. Preference data and reward models are imperfect proxies, so monitor reward hacking, diversity, safety, and human judgments.
**Evaluation:** Pipeline stages, proxy limitations, monitoring.
**Follow-up:** How does DPO differ in what it optimizes directly?
**Common mistake:** Calling the learned reward model human preference itself.

### 14. Debug reward-model overoptimization
**Prompt:** Increasing policy optimization steps raises reward-model score but human preference declines. What happened and what would you test?
**Role/domain:** RLHF; debugging. **Difficulty:** Hard.
**Expected answer:** The policy likely exploited reward-model errors or moved out of its training distribution. Compare human preference and safety slices across checkpoints, monitor divergence and output diversity, improve preference coverage, and tune constraints/early stopping rather than selecting by reward score alone.
**Evaluation:** Proxy overoptimization, independent evaluator, corrective plan.
**Follow-up:** Why can a KL penalty reduce but not eliminate this failure?
**Common mistake:** Trusting the optimized proxy as ground truth.

### 15. Decide if RL is justified
**Prompt:** A team wants RL for a deterministic workflow that has clear input-output examples. What questions determine whether RL is appropriate?
**Role/domain:** Applied RL; method selection. **Difficulty:** Hard.
**Expected answer:** Ask whether actions affect future states, delayed outcomes matter, exploration is feasible, and a reliable reward can be measured. If the task is static supervised prediction or direct optimization with known objectives, simpler supervised methods or operations research may be cheaper and easier to validate.
**Evaluation:** Problem formulation, alternatives, data/safety cost.
**Follow-up:** What evidence would justify a contextual bandit formulation?
**Common mistake:** Choosing RL because the product is interactive, without defining sequential credit assignment.
