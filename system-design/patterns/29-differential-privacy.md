# Differential Privacy

## TL;DR
Add noise to data or gradients during training. Model cannot memorize individuals (membership inference attack fails). Formally proven privacy: ε-differential privacy (smaller ε = more private).

## Core Intuition
Data = {Alice, Bob, Charlie}. Train model on data. Model memorizes Alice. Attack: "is Alice in training data?" → Yes (privacy leak). Solution: add noise so model can't memorize Alice.

## How It Works

**Differential privacy:**
- Add Gaussian noise to gradients during training
- Smaller ε = more noise = more privacy (but less accuracy)
- ε=1: good privacy. ε=10: less privacy.

| ε | Privacy | Accuracy |
|---|---------|----------|
| 0.5 | Very high | 90% |
| 1.0 | High | 92% |
| 5.0 | Medium | 94% |
| 10.0 | Low | 95% |

## Key Properties / Trade-offs
- Privacy vs accuracy: more privacy → more accuracy drop
- Complexity: requires clipping and noise addition
- Proof: differential privacy = formal proof (unlike heuristics)

## Detailed Trade-off Analysis

| Privacy Guarantee (ε) | Accuracy | Noise Level | Use Case | Regulatory OK |
|----------------------|----------|------------|----------|--------------|
| 0.1 | 88% | Very high | Academic research | Yes |
| 1.0 | 92% | High | Industry standard | Yes |
| 5.0 | 94% | Medium | Borderline | Maybe |
| 10.0 | 95% | Low | Weak guarantee | No |
| ∞ (no DP) | 96% | None | No privacy | No |

**Decision:** Regulated (healthcare, finance) → ε≤1.0. Compliant (GDPR) → ε≤5.0. Startup MVP → ε>5.0 (trade privacy for accuracy).

---

## Production Failure Scenarios

**Scenario 1: DP noise too low, membership inference attack succeeds**
- Train with noise_multiplier=0.1 (claimed ε=1.0 but actual ε=8.0). Attacker extracts training data.
- Privacy claim false.
- Prevention: Test with membership inference attack (open-source tools available). Verify ε empirically.

**Scenario 2: DP applied incorrectly, privacy guarantee invalid**
- Add noise to gradients but batch size changes mid-training. Privacy math breaks.
- Prevention: Use library correctly (TensorFlow Privacy, Opacus). Don't DIY. Verify implementation.

**Scenario 3: DP breaks fairness**
- Add noise for privacy. Fairness metrics degrade (noise disproportionately hurts minority groups).
- Privacy achieved, fairness broken.
- Prevention: Monitor fairness with DP. May need group-specific noise levels.

**Scenario 4: DP accuracy unacceptable, model unusable**
- DP reduces accuracy from 95% to 80%. Too low for production.
- Prevention: Test DP impact before implementation. Know accuracy trade-off upfront.

---

## Implementation Guidance

**Wrong:** DIY differential privacy. Implement gradient clipping + noise yourself.
**Right:** Use library (TensorFlow Privacy or Opacus). Peer-reviewed, proven correct. Easier to get right.

**Wrong:** Set noise_multiplier arbitrarily (e.g., 0.5). Hope privacy is good.
**Right:** Calculate ε budget from privacy requirements. Iterate: test accuracy, adjust noise, verify privacy.

---

## Sophisticated Interview Q&A

**Q1: ε=1.0. What does this privacy guarantee mean practically?**
A: (1) Formal: attacker with access to gradients, auxiliary data, can't determine if single person in training with confidence >68%. (2) Practical: model doesn't memorize individuals. Membership inference attack has <68% success. (3) Not perfect privacy, but provable. (4) Stronger than anonymization (no re-identification).

**Q2: Accuracy drops 5% with DP. How justify to business?**
A: (1) Quantify: 95% → 90% on what metric? (2) Business impact: 5% drop = revenue loss. (3) Cost of privacy breach: regulatory fine + reputation = $100K+. (4) Trade-off: is 5% accuracy loss < cost of breach? (5) Explore: can different models achieve 92-93% with DP? (can sometimes).

**Q3: Multiple DP implementations (TensorFlow, Opacus, CrypTen). Which to use?**
A: (1) TensorFlow Privacy: Keras API, easy integration, good for Keras/TF. (2) Opacus: PyTorch native, more flexible, better for research. (3) CrypTen: multi-party computation, for federated settings. (4) Pick based on framework (TF vs PyTorch) and use case. All are solid, peer-reviewed.

**Q4: DP noise per batch. How determine noise_multiplier?**
A: (1) Start with default (0.5). (2) Measure ε budget consumed. (3) If ε too high (weak privacy), increase noise. (4) If ε too low (high accuracy loss), decrease noise. (5) Trade privacy-accuracy. (6) Use: δ (privacy failure probability, typically 1e-5) + desired ε → compute noise_multiplier.

---

## Cost & Resource Analysis

**Differential privacy infrastructure:** TensorFlow Privacy or Opacus library. Minimal cost (open-source, integrate into training).
**Accuracy loss:** Typically 1-5%. Can mean retraining, larger models, more data. Cost: extra GPU hours, development time.
**Privacy auditing:** Membership inference tests, ε verification. 1-2 weeks engineer time = $5-10K.
**Compliance certification:** External audit for regulated domains = $10-50K.

**Cost of privacy breach (no DP):** Regulatory fine $4% revenue (GDPR). Litigation $100K+. Reputation. Easily $100K-10M+.
**ROI:** DP investment $10-100K. Prevents incident worth $100K+. Break-even year 1.

---

## Monitoring & Observability

**Key metrics:** Privacy budget ε consumed per epoch, accuracy loss percentage, noise multiplier in use, membership inference attack success rate, privacy coverage (% of training data under DP protection)

**Alerts:** ε budget exceeded (privacy guarantee violated), accuracy drops below threshold, noise_multiplier configuration incorrect, membership inference attack succeeds (privacy broken)

## Common Mistakes / Gotchas
- DIY differential privacy (easy to get wrong)
- Insufficient noise (privacy bound too loose)
- No privacy audit (claims not validated)
- Confusing ε values (different libraries use different scales)

## Best Practices
- **Use library:** TensorFlow Privacy, Opacus (PyTorch)
- **Test privacy:** membership inference attack to validate
- **Report ε value:** always report privacy budget
- **Noise scheduling:** start high noise, reduce over training
- **Batch size:** smaller batch → more privacy needed

## Code Example
```python
import opacus
from opacus.utils.batch_memory_manager import BatchMemoryManager

# Attach DP to optimizer
model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

privacy_engine = opacus.PrivacyEngine(
    model,
    batch_size=32,
    sample_size=len(dataset),
    alphas=[10, 100],
    noise_multiplier=1.0,
    max_grad_norm=1.0
)
privacy_engine.attach(optimizer)

# Train normally, DP is automatic
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        loss = model(batch_x, batch_y)
        loss.backward()
        optimizer.step()
```

## Interview Q&A
**Q: Model trained with DP (ε=1.0). Privacy guarantee?**
A: Formal guarantee: attacker with ~N/2 data samples can't determine if single person is in training set (at ε=1.0). More precisely: probability attacker guesses wrong is at least 50% + (1-e^(-1))/2 = 68%.

**Q: DP reduces accuracy 5%. Acceptable?**
A: Depends on use case. Recommendation (95% → 90%): maybe. Medical diagnosis (95% → 90%): unacceptable. Measure fairness-accuracy frontier, choose based on user needs.

## Interview Quick-Reference
| ε | Privacy Level |
|---|---|
| 0.1 | Excellent (academic research) |
| 1.0 | Good (industry standard) |
| 10.0 | Weak (privacy claim but not strong) |

## Related Topics
- [Privacy-Preserving ML](28-privacy-preserving-ml.md)
- [Data Governance](26-data-governance.md)

## Resources
- [Differential Privacy Library](https://github.com/tensorflow/privacy)
- [Opacus: PyTorch DP Library](https://opacus.ai/)
