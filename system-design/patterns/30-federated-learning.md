# Federated learning

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
- Apply differential privacy at the client level to protect individual contributions
- Use secure aggregation to prevent server from seeing individual client updates
- Implement client selection strategies to handle heterogeneous data and compute
- Use FedAvg with momentum for faster convergence
- Communicate only model deltas, not full weights, to reduce bandwidth
- Monitor for Byzantine clients and implement robust aggregation
- Test with heterogeneous data (non-IID) distributions — federated data is never IID

## Interview Q&A

**Q: What are the communication efficiency challenges in federated learning and how do you address them?**
A: Challenge: sending full model updates (1GB+ for large models) from millions of clients per round is infeasible. Solutions: gradient compression (top-k sparsification: only send the top-k% largest gradients, 99% compression with <5% quality loss), quantization of updates (8-bit gradients), local steps (clients take N gradient steps before sending updates, reducing communication frequency), and model architecture choices (smaller models with fewer parameters to communicate). In cross-device FL (mobile), communication cost often exceeds compute cost.

**Q: How do you handle data heterogeneity (non-IID data) in federated learning?**
A: Non-IID (non-independent identically distributed) data: each client has a different data distribution (one user only uses the app for cooking, another for travel). Standard FedAvg assumes IID data and can diverge with extreme heterogeneity. Solutions: FedProx (adds proximal term to prevent local model from diverging too far from global), SCAFFOLD (variance reduction technique), and personalized FL (train a global model + per-client adaptation layer). In practice: understand your data heterogeneity before choosing an aggregation algorithm, as different methods have different heterogeneity tolerances.

**Q: How do you prevent adversarial clients from poisoning a federated learning model?**
A: Byzantine attacks: malicious clients send adversarial gradients to steer the global model. Defenses: robust aggregation (coordinate-wise median instead of mean, Krum: select the update closest to the majority), anomaly detection (reject updates that deviate significantly from the mean), differential privacy (DP aggregation masks individual updates making poisoning harder), and reputation systems (clients with a history of good updates weighted more). No single defense is sufficient—combine multiple approaches for production FL systems.

**Q: When does federated learning provide meaningful privacy protection vs. just data locality?**
A: Data locality: data doesn't leave the device (good for compliance, data sovereignty). Privacy: the server and other clients can't infer the client's data from gradient updates. Gradient updates can leak significant private information—gradient inversion attacks can reconstruct training images from gradients. For strong privacy, combine FL with differential privacy (add noise to client gradients before sending). Without DP, FL provides data locality but not strong privacy protection—be precise about which property you're claiming when communicating about FL.

**Q: What evaluation methodology do you use for federated learning models when you can't centralize test data?**
A: Three approaches: (1) holdout clients—reserve 10-20% of clients who don't participate in training, evaluate on their local data after each round; (2) federated evaluation—send the global model to all clients, each evaluates locally and reports metrics (aggregated without sharing data); (3) public proxy dataset—use a publicly available dataset that approximates the distribution for centralized evaluation. Each has trade-offs: holdout clients reduces training data, federated evaluation adds communication overhead, and proxy datasets may not represent actual distribution.

## Interview Quick-Reference
| Question | What to say |
|---|---|
| "Explain?" | [Answer] |

## Related Topics
- [Related](other.md)

## Resources
- [Reference](url)
