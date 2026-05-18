# Privacy preserving ml

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
- Apply differential privacy with epsilon < 1.0 for high-sensitivity data
- Use federated learning when data cannot leave client devices
- Anonymize training data before use — pseudonymization is not anonymization
- Apply privacy budget tracking across multiple model versions
- Use secure aggregation in federated settings to prevent server from seeing individual updates
- Conduct privacy audits using membership inference attack benchmarks
- Encrypt model weights at rest and in transit for sensitive domains

## Interview Q&A

**Q: What are the main privacy-preserving ML techniques and when do you use each?**
A: Differential privacy (DP): adds calibrated noise during training to prevent individual data points from being identifiable in model outputs—use for models where model weights or outputs could reveal training data. Federated learning: train without centralizing data—use when data can't leave devices or organizational silos. Secure multi-party computation: compute on encrypted data—use for collaborative ML between competitors. Homomorphic encryption: compute on fully encrypted data—use for inference on sensitive data without decrypting. Match the technique to the threat model—DP doesn't help if an attacker has access to the training logs.

**Q: How does differential privacy affect model accuracy and how do you tune the privacy budget?**
A: DP adds Gaussian noise scaled to the sensitivity of the computation. Higher privacy (lower epsilon) requires more noise, reducing model accuracy. In practice: for epsilon=8 (strong privacy), image classifiers lose 5-15% accuracy; for epsilon=1, loss can be 20-30%. Tune epsilon by: determining the minimum acceptable accuracy on your task, testing DP training at different epsilon values, and choosing the smallest epsilon where accuracy meets the threshold. Document the privacy budget in the model card and treat it as a model property that informs deployment decisions.

**Q: What is model memorization and how do you test for it?**
A: Model memorization: the model stores specific training examples and can reproduce them when queried. Test with: membership inference attacks (can you tell if a specific example was in the training set?), training data extraction (can you prompt the model to reproduce verbatim training data?), and shadow model attacks (train a shadow model on similar data and compare behavior). LLMs are particularly susceptible to memorization of frequently-repeated text (addresses, emails, code). Use differential privacy training, deduplication of training data, and output filtering to mitigate.

**Q: How do you implement federated learning for an ML application?**
A: Federated learning architecture: (1) server sends current model to all clients; (2) each client trains on local data for N steps; (3) clients send gradient updates (not data) to server; (4) server aggregates updates (FedAvg: weighted average of gradients); (5) server updates global model and repeats. Key challenges: communication overhead (sending model updates each round), data heterogeneity (each client has a different data distribution), and stragglers (slow clients delay aggregation). Use frameworks: TensorFlow Federated, PySyft, or Flower for implementation.

**Q: What are the privacy trade-offs of model inversion vs. membership inference attacks?**
A: Model inversion: attacker reconstructs input data from model outputs—more dangerous for models with rich outputs (face recognition, generative models). Defends against: output rounding, output perturbation, limiting access to model internals. Membership inference: attacker determines if a specific data point was in training—enables stalking (was this person's medical record used?). More practical attack on most ML systems. Defends against: differential privacy, early stopping, temperature tuning, and monitoring for unusual query patterns. Test your model's vulnerability to both attacks before deploying with sensitive training data.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
