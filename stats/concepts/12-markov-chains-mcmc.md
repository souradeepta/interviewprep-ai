# Markov Chains and MCMC

## Detailed Explanation

A Markov chain is a stochastic process where the probability of transitioning to a future state depends only on the current state, not the full history: P(X_{t+1} | X_t, ..., X_0) = P(X_{t+1} | X_t). This Markov property enables enormous simplifications in probabilistic modeling and computation. The key long-run behavior is described by the stationary (invariant) distribution π: a probability distribution such that πP = π, where P is the transition matrix. Under mild regularity conditions (irreducibility and aperiodicity), any initial distribution converges to π.

Markov Chain Monte Carlo (MCMC) flips this construction: given a target distribution π (e.g., a Bayesian posterior), design a Markov chain that has π as its stationary distribution and simulate it to collect samples. The two canonical algorithms are Metropolis-Hastings (MH), which accepts proposed moves probabilistically, and Gibbs sampling, which updates each variable conditioned on all others. Because the chain must mix (converge to π), MCMC samples are correlated, unlike i.i.d. Monte Carlo samples. The effective sample size (ESS) quantifies how many independent samples the correlated chain is worth.

MCMC is indispensable in Bayesian statistics because most posteriors cannot be normalized analytically, especially in high dimensions. Practitioners use MCMC to sample posteriors over model parameters, enabling credible intervals, posterior predictive checks, and model comparison. Diagnostics — trace plots, autocorrelation functions, Gelman-Rubin statistic R̂ — are as important as the sampling algorithm itself, because a poorly mixed chain gives incorrect answers that look plausible.

## Core Intuition

MCMC is like exploring a landscape in fog: you cannot see the full terrain (the unnormalized posterior) but you can compare your current foothold to any proposed step and decide whether to move based on which is higher. Walk long enough and the fraction of time you spend at each location approximates the true elevation of the landscape.

## How It Works

1. **Choose a target distribution π(x)** (e.g., unnormalized posterior p(θ|data) ∝ p(data|θ)p(θ)). You only need to evaluate ratios π(x')/π(x).
2. **Initialize the chain** at some starting state x_0 (random or from a rough approximation).
3. **Metropolis-Hastings proposal**: at state x, propose x' ~ q(x'|x) from a proposal kernel (e.g., x' = x + ε, ε ~ N(0, σ²)).
4. **Accept or reject**: compute acceptance ratio α = min(1, π(x')/π(x) × q(x|x')/q(x'|x)). Draw u ~ Uniform(0,1); if u < α, move to x'; otherwise stay at x.
5. **Discard burn-in**: the first B samples are drawn before the chain has mixed; discard them. Keep every k-th sample (thinning) to reduce autocorrelation.
6. **Diagnose convergence**: inspect trace plots for stationarity, compute autocorrelation and ESS, and run Gelman-Rubin R̂ across multiple chains (R̂ < 1.05 indicates convergence).

```mermaid
graph TD
    A[Initialize x0] --> B[Propose x prime from q x prime | x]
    B --> C[Compute acceptance ratio alpha]
    C --> D{u < alpha?}
    D -- Yes --> E[Accept: x = x prime]
    D -- No --> F[Reject: x stays]
    E --> G{Burned in?}
    F --> G
    G -- No --> B
    G -- Yes --> H[Record sample]
    H --> I{Converged?}
    I -- No --> B
    I -- Yes --> J[Return samples]
```

## Architecture / Trade-offs

### MCMC Algorithm Comparison

| Algorithm | Requirement | Acceptance Rate | Mixing Speed | Dimensionality | Correlation |
|-----------|-------------|-----------------|--------------|----------------|-------------|
| Metropolis-Hastings | Unnormalized π only | Tunable (target 0.23) | Slow if poorly tuned | Scales badly | High if small steps |
| Gibbs Sampling | Conditional distributions | 100% (always accepts) | Good when conditionals easy | Moderate | Can be high |
| HMC/NUTS | Gradient of log π | ~65-90% | Fast, low correlation | Scales well | Low |
| Slice Sampling | Unnormalized π | 100% (adaptive) | Moderate | Low-medium dim | Moderate |

### Proposal Variance Trade-offs in MH

| Proposal std | Acceptance rate | Problem |
|--------------|-----------------|---------|
| Too small (0.01) | Very high (>95%) | Tiny steps, slow exploration, very high autocorrelation |
| Optimal (~2.38/√d) | ~23% | Good mixing, appropriate ESS |
| Too large (10x) | Very low (<5%) | Most proposals rejected, chain barely moves |

## Interview Q&A

**Q: What happens if your Metropolis-Hastings proposal variance is too small, and how would you detect it?**
A: With tiny proposals the chain accepts almost every step but moves very slowly, producing extremely autocorrelated samples. You would detect it from the trace plot (slow random walk pattern rather than fast mixing), the autocorrelation function (decays very slowly over many lags), and a very small ESS (e.g., ESS = 50 from N = 10,000 nominal samples). Fix by increasing proposal variance until acceptance rate is near 23% for high-dimensional targets.

**Q: Why do we discard burn-in, and how do you decide how much to discard?**
A: The chain starts at an arbitrary x_0 which may be in a low-probability region of π. Early samples reflect the starting point, not the target distribution. A common rule is to run two chains from overdispersed starting points and discard samples until R̂ < 1.05, or inspect the trace plot for when the chain settles from its initial transient. In practice 10-20% of total samples as burn-in is typical, but some posteriors require much longer.

**Q: Why is ESS much smaller than the number of MCMC samples, and what does it mean for your analysis?**
A: MCMC samples are correlated (each depends on the previous), unlike i.i.d. samples. ESS = N / (1 + 2Σ ρ_k) where ρ_k is lag-k autocorrelation. A highly correlated chain with τ_integrated = 50 gives ESS ≈ N/100. This means running 10,000 MCMC steps might give only 100 effectively independent samples. For uncertainty quantification, compute confidence intervals using ESS, not N.

**Q: When would you choose Gibbs sampling over Metropolis-Hastings?**
A: Use Gibbs when full conditional distributions p(x_i | x_{-i}) are available in closed form and easy to sample from (e.g., conjugate models, Gaussian graphical models). Gibbs always accepts and is simpler to implement. Use MH when conditionals are intractable or the model is non-conjugate. In practice, modern probabilistic programming languages like PyMC use combinations — Gibbs for conjugate subgraphs, HMC for the rest.

**Q: How do you diagnose whether your Markov chain has converged?**
A: Run at least 2-4 chains from different starting points and compute R̂ = sqrt(Var_between / Var_within). R̂ > 1.1 signals non-convergence. Also inspect trace plots for stationarity (the chain should look like white noise, not trending), autocorrelation plots (should decay quickly), and compute ESS (aim for ESS > 400 for reliable 95% credible intervals). A single chain can fool all diagnostics if it gets stuck in one mode.

**Q: What is the role of detailed balance in MCMC, and why does MH satisfy it?**
A: Detailed balance (reversibility) is a sufficient condition for π to be the stationary distribution: π(x)T(x'|x) = π(x')T(x|x'). MH satisfies this by construction: the acceptance probability α = min(1, π(x')q(x|x')/(π(x)q(x'|x))) ensures that the net flow between any two states is balanced. This means the chain will converge to π regardless of the proposal q, as long as q has the same support as π.

## Best Practices

- Target acceptance rate of 20-30% for random-walk MH in high dimensions; use adaptive MCMC during warm-up to tune proposal variance automatically.
- Always run multiple chains (at least 4) from overdispersed starting points; a single chain can get stuck in a mode and appear converged while missing large probability mass elsewhere.
- Compute ESS for every parameter and require ESS > 400 before reporting credible intervals; running longer is always preferable to under-sampling.
- Use log-probabilities throughout to avoid numerical underflow in high-dimensional or small-probability calculations; never compute π(x') and π(x) separately and then divide.
- Thin the chain (keep every k-th sample) only to reduce storage, not to reduce autocorrelation — thinning wastes compute but does not improve statistical efficiency.
- Monitor trace plots and R̂ before publishing any MCMC results; R̂ < 1.05 is a minimum bar, not a guarantee of correctness.
- For high-dimensional posteriors (>20 dimensions), use HMC or NUTS (via PyMC or Stan) instead of random-walk MH; HMC exploits gradient information to take much larger steps.

## Common Pitfalls

- **Single-chain analysis**: Running one chain and accepting convergence based on its trace plot is unreliable — if the chain gets stuck in one mode, it will appear stationary. Fix: always run multiple chains from different starting points and check R̂.

- **Treating nominal N as effective N**: Reporting credible intervals based on N = 10,000 MCMC steps when ESS = 200 gives artificially narrow intervals and overconfident conclusions. Fix: always report ESS and use it for uncertainty calculations, not N.

- **Ignoring burn-in**: Including early samples before the chain has mixed introduces bias toward the starting point. Fix: visually inspect trace plots and discard the initial transient; use R̂ convergence criterion.

- **Symmetric proposal assumption when it is not**: The MH acceptance ratio simplifies to π(x')/π(x) only when q(x'|x) = q(x|x'). For asymmetric proposals (e.g., log-normal), omitting the q ratio gives a biased sampler. Fix: always include the proposal ratio q(x|x')/q(x'|x) in the acceptance probability.

- **Floating-point overflow in acceptance ratio**: Computing exp(log π(x') - log π(x)) is numerically stable; computing π(x')/π(x) directly overflows or underflows for unnormalized posteriors. Fix: work in log space throughout.

## Related Concepts

- [11-monte-carlo-sampling.md](./11-monte-carlo-sampling.md) — MCMC is a special case of MC sampling where i.i.d. draws are replaced by correlated chain samples
- [08-bayesian-inference.md](./03-bayesian-inference.md) — MCMC is the workhorse for Bayesian posterior computation when conjugacy fails
- [02-distributions-reference.md](./02-distributions-reference.md) — Understanding target distributions requires knowledge of common parametric families
- [15-statistical-ml-connections.md](./15-statistical-ml-connections.md) — VI and EM are deterministic alternatives to MCMC for approximate Bayesian inference
