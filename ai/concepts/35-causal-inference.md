# Causal Inference

## Detailed Explanation

Causal inference determines which variables cause which outcomes (beyond correlation), crucial for understanding mechanisms and making effective interventions. The fundamental problem: from observational data, we only see correlations, but correlation doesn't imply causation (A correlated with B could mean A→B, B→A, or C→A,C→B). Causal inference uses graphical models (DAGs: directed acyclic graphs) to encode causal assumptions, then applies adjustment strategies to estimate effects of interventions.

Randomized controlled trials (RCTs) estimate causal effects by randomly assigning interventions (breaks all back-door paths from intervention to outcome). However, RCTs are expensive/unethical for many questions. Observational methods use causal graphs to identify confounders (variables affecting both intervention and outcome) and use adjustment (conditioning on confounders) or methods like instrumental variables (variables affecting outcome only through intervention) to estimate causal effects. The do-calculus (Pearl's framework) determines whether a causal effect is identifiable from observational data given a causal model.

Causal inference is increasingly important for policy decisions (what intervention improves outcomes?), medicine (does treatment cause cure?), and fairness (does algorithm discriminate?). Understanding that correlation ≠ causation is critical. Challenges include model misspecification (assumed causal graph might be wrong), hidden confounders, and multiple intervention pathways. Modern methods combine causal reasoning with machine learning for estimation, but understanding the principles is essential for responsible conclusions.

## Core Intuition

Causal inference is like detective work: you observe correlations (X and Y are often together), but you need to figure out the mechanism (X causes Y, Y causes X, or both caused by Z). Causal graphs show your theories about mechanisms, and different techniques extract causal effects from data depending on those theories.

## How It Works

1. Confounding: variable X affects both treatment T and outcome Y
2. DAG: directed acyclic graph showing causal structure
3. Adjustment: condition on confounders to isolate causal effect
4. Matching: match treated and control units on confounders
5. Propensity score: probability of treatment, use for matching or weighting
6. Instrumental variables: use variable that affects T but not Y (directly)
7. Difference-in-differences: compare treatment and control pre/post intervention

```mermaid
graph TD
    A[Input] -->|Process| B[Model/Algorithm]
    B -->|Output| C[Result]
```

## Architecture / Trade-offs

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
| **Multiple Robustness Checks** | Weak (sensitivity testing) | Very high | High (if checks pass) |
## Interview Q&A


**Q: What's the difference between correlation and causation?**
A: Correlation: variables move together. Causation: one causes change in other. Confounder example: ice cream sales correlate with drowning (both caused by summer weather, no direct causation). Causal methods isolate true effect.

**Q: How do you estimate causal effects from observational data?**
A: Assumption: no unmeasured confounders (observe all variables affecting outcome). Methods: (1) adjustment (condition on confounders), (2) matching (match treated/control on confounders), (3) propensity score (weight by inverse probability of treatment). All assume no unmeasured confounding.

**Q: What is a confounder and how do you handle it?**
A: Confounder: affects both treatment and outcome. Bias: if not adjusted, confounders bias causal estimate. Handle: (1) randomization (best, breaks confounding), (2) adjustment (condition on confounder), (3) matching (match on confounder).

**Q: What are instrumental variables and when do you use them?**
A: IV: variable Z affects treatment T but doesn't directly affect outcome Y. Use when: confounders unmeasured, can't randomize. Example: rainfall affects irrigation (T), affects crops (Y) but not through other mechanisms. Enables causal inference under more assumptions.

**Q: How do you validate causal inferences?**
A: Sensitivity analysis: how robust to unmeasured confounding? Placebo tests: effect on variables that shouldn't be affected. Heterogeneous effects: does effect differ by subgroup (should be consistent mechanism). Multiple methods: if all agree, confidence increases.


## Best Practices

- Apply best practices specific to this concept
- Consider edge cases and failure modes
- Test on representative data
- Evaluate comprehensively

## Common Pitfalls

- Avoid over-simplification
- Watch for incorrect assumptions
- Test edge cases thoroughly
- Monitor for degradation

## Code Examples

See the associated notebook for implementation and real-world examples.

## Related Concepts

- Understand prerequisites first
- Connect related topics
- Build integrated knowledge
