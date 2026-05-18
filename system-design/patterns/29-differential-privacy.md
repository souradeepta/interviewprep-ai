# Differential privacy

## TL;DR
Core ML system design pattern for production.

## Core Intuition
[Intuitive explanation]

## How It Works
[Technical details]

## Key Properties / Trade-offs
- Property 1
- Property 2

## Common Mistakes / Gotchas
- Mistake 1
- Mistake 2

## Best Practices
- Start with epsilon=1.0 as a baseline and tighten based on privacy requirements
- Use RDP (Rényi Differential Privacy) accountant for tight composition bounds
- Clip gradients before adding noise — gradient norm must be bounded
- Apply DP at the model level, not the dataset level, for training
- Use DP-SGD (opacus library for PyTorch) for end-to-end differentially private training
- Publish epsilon values alongside model performance metrics
- Test that privacy guarantees hold with membership inference attack evaluation

## Interview Q&A

**Q: What does epsilon in differential privacy mean intuitively and what values are considered strong?**
A: Epsilon bounds the privacy loss: an attacker's ability to distinguish whether a specific individual was in the training set improves by at most e^epsilon. Small epsilon means strong privacy. Interpretations: epsilon=0: perfect privacy (no information leakage); epsilon=1: 2.7x advantage for attacker with one individual's data; epsilon=8: 3000x advantage (barely better than no privacy). In practice: epsilon<1 is strong but often impractical (too much noise); epsilon=1-10 is commonly used; epsilon>10 provides weak protection. The appropriate epsilon depends on your threat model and how sensitive the data is.

**Q: How do you compose privacy budgets across multiple DP computations?**
A: Basic composition: running k DP mechanisms with epsilon values epsilon_1 through epsilon_k gives total privacy budget epsilon_total = sum of all epsilon values—budget accumulates with each computation on the same data. This means you can't run unlimited DP queries—the privacy budget depletes. Advanced composition (Renyi DP, moments accountant): achieves tighter bounds, allowing more queries within the same total budget. In practice: use Opacus's privacy engine which automatically tracks cumulative privacy loss using moments accountant. Set a total budget and stop training when it's exhausted.

**Q: When is DP-SGD preferable to aggregation-level DP (adding noise to final statistics)?**
A: Aggregation-level DP: add noise to aggregate statistics (counts, means) after computation—simpler and allows tighter privacy guarantees for simple statistics. Sufficient for: releasing demographic statistics, publishing aggregate model metrics. DP-SGD: clips and noisifies gradients during training—provides protection for each training step but accumulates privacy budget. Necessary for: training neural networks where you want the model itself to be DP. If you just need to release model performance statistics publicly, aggregation-level DP is cheaper. If you need the model weights to be DP (e.g., to prevent model inversion), use DP-SGD.

**Q: How does clipping gradient norms interact with differential privacy in DP-SGD?**
A: DP-SGD clips each per-sample gradient to a maximum L2 norm C before adding Gaussian noise. Clipping bounds the sensitivity—the noise level is calibrated to C. Too small C: clips most gradients, destroying signal and degrading accuracy significantly. Too large C: gradients barely clipped, noise is small relative to gradient magnitude but privacy loss per step is lower. Tune C to the typical per-sample gradient norm: use the median gradient norm from the first few non-DP training steps as a starting point, then adjust based on accuracy vs. privacy budget.

**Q: What are the implementation pitfalls when applying DP to ML training?**
A: Common mistakes: applying DP after batch normalization (batch norm leaks per-sample info through batch statistics—use group norm or layer norm instead), forgetting to clip per-sample gradients (not batch-averaged gradients), reusing the privacy budget for hyperparameter tuning (each evaluation depletes budget—use public validation data for tuning), and applying DP only to training but not to the evaluation data (evaluation can also leak information). Use Opacus (PyTorch) or TF Privacy (TensorFlow) rather than implementing DP from scratch.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
