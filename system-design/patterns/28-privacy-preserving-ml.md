# Privacy-Preserving ML

## TL;DR
Train models without exposing private data. Techniques: differential privacy (add noise), federated learning (train on edge), homomorphic encryption (compute on encrypted data). Trade latency/accuracy for privacy.

## Core Intuition
Hospital has patient data. Can't share with cloud. Solution: send model to hospital, train locally, send results back (federated). Or: add noise to data, train, model doesn't memorize individuals (differential privacy).

## How It Works

**Privacy techniques:**

1. **Differential privacy:** add noise to data/gradients
   - Model can't memorize individuals
   - Cost: accuracy drops 1-5%

2. **Federated learning:** train on decentralized data
   - Phone/hospital trains locally, sends gradient updates
   - Server aggregates, sends back
   - Data never leaves device

3. **Homomorphic encryption:** compute on encrypted data
   - Encrypt data, train model on encrypted data
   - Cost: 1000x slower

| Technique | Privacy | Accuracy | Speed |
|-----------|---------|----------|-------|
| Diff privacy | High | 95% | Fast |
| Federated | High | 95% | Medium |
| Homomorphic | Highest | 95% | Very slow |

## Key Properties / Trade-offs
- Privacy vs accuracy: enforcing privacy reduces accuracy
- Privacy vs speed: more privacy usually means slower
- Complexity: privacy-preserving ML more complex to implement

## Detailed Trade-off Analysis

| Technique | Privacy Level | Accuracy Impact | Latency | Complexity | Adoption |
|-----------|---------------|-----------------|---------|-----------|----------|
| Anonymization | Medium (re-identifiable) | 0% | <1ms | Low | High |
| Differential privacy | High | 1-5% drop | <1ms | Medium | Growing |
| Federated learning | High | 1-3% drop | 100-1000ms | High | Growing |
| Homomorphic encryption | Very high | 0% (minimal) | 1000x slower | Very high | Rare |
| Secure multi-party | High | 0% | 100-1000x slower | Very high | Rare |

**Decision:** GDPR compliant → differential privacy. Decentralized data → federated. Maximum privacy + not speed-critical → homomorphic.

---

## Production Failure Scenarios

**Scenario 1: Anonymization, user re-identified**
- Remove name/email from data. Attacker combines with public records, re-identifies user.
- Privacy claim false. Regulatory fine.
- Prevention: Differential privacy (provably private). Or: proper k-anonymity (tested with re-id attacks).

**Scenario 2: Differential privacy noise too low, privacy attack succeeds**
- Add noise_multiplier=0.1 (too low). Researcher extracts training data from gradients.
- Privacy compromised.
- Prevention: Use published libraries (TensorFlow Privacy). Follow recommendations (noise_multiplier ≥ 0.5).

**Scenario 3: Federated learning, gradient leakage**
- Hospital sends gradient update to server. Gradient intercepted, contains individual patient data.
- Privacy broken despite "federated".
- Prevention: Encrypt gradients in transit. Add differential privacy to gradients.

**Scenario 4: Privacy audit skipped, privacy claims unvalidated**
- Company claims "privacy-preserving model" but never tested. Sued by user.
- Prevention: Hire external audit firm. Verify privacy properties. Document findings.

---

## Implementation Guidance

**Wrong:** Remove identifying columns (name, email), assume data is anonymized.
**Right:** Use differential privacy (provable). Or: k-anonymity with re-identification testing. Don't claim privacy without validation.

**Wrong:** DIY differential privacy implementation.
**Right:** Use published libraries (TensorFlow Privacy, PyDP). Peer-reviewed, battle-tested.

---

## Sophisticated Interview Q&A

**Q1: User GDPR right-to-be-forgotten. Model trained on their data. Retrain?**
A: (1) Simplest: retrain without their data. (2) Cost: retraining expensive ($10K). (3) Better: use differential privacy upfront (model doesn't memorize individuals, deletion request = update bounds). (4) Alternative: certified unlearning (prove user's influence removed without retraining).

**Q2: Federated learning: accuracy drops 3% from centralized. Accept?**
A: Depends on stakes. (1) If <99% required: accept (privacy > 3% accuracy). (2) If >99% required: explore hybrid (most data centralized, sensitive data federated). (3) Increase federated rounds (more communication, better convergence). (4) Use techniques: knowledge distillation (server sends public model to improve local training).

**Q3: Homomorphic encryption, 1000x slower. Practical?**
A: (1) Batch processing only (not real-time). (2) For very sensitive data (military, top-secret). (3) Research phase, not production yet. (4) Latency requirements: if <1 second acceptable, no. If can wait hours, maybe. (5) Cost: 1000x compute = $1K per prediction. Viable only for high-value decisions.

**Q4: Differential privacy: what noise_multiplier to use?**
A: (1) Depends on privacy budget ε (epsilon). Higher ε = less privacy = less noise needed. (2) Typical: ε=1.0 (conservative), noise_multiplier=0.5-1.0. (3) ε=10 (permissive), noise_multiplier=0.1-0.3. (4) Measure: accuracy loss acceptable? Adjust noise accordingly. (5) Default recommendation: ε=1.0 for regulated domains.

---

## Cost & Resource Analysis

**Differential privacy:** Minimal overhead (add noise to gradients). Infrastructure cost: negligible.
**Federated learning:** High overhead (gradient aggregation, secure communication). Infrastructure: $5-20K/month for large-scale.
**Homomorphic encryption:** Very high overhead (1000x compute). Only for high-value applications.
**Privacy audit:** $10-50K external audit. Required for compliance (GDPR, HIPAA).

**Cost of privacy breach:** Regulatory fine $4% revenue (GDPR). Reputation damage. Loss of trust. Easily $100K-10M+.
**ROI:** Privacy protection $20-100K/year. Prevents incident worth $100K+. Break-even year 1.

---

## Monitoring & Observability

**Key metrics:** Privacy guarantee ε (epsilon) value, accuracy loss due to privacy mechanisms, federated communication rounds, gradient leakage detection (monitor for privacy attacks), privacy audit status, compliance certification

**Alerts:** Privacy guarantee violated (ε exceeded), privacy attack detected (suspicious gradient patterns), federated accuracy drops unexpectedly, privacy audit overdue, compliance certificate expires

## Common Mistakes / Gotchas
- Assuming anonymization = privacy (can be re-identified)
- Insufficient noise (privacy attack succeeds)
- No privacy audit (privacy claims unvalidated)
- Ignoring computational cost (homomorphic encryption very slow)

## Best Practices
- **Privacy audit:** hire external party to verify privacy
- **Differential privacy:** use published libraries (not DIY)
- **Federated: careful gradient aggregation (averagingalone leaks data)
- **Benchmark privacy vs accuracy:** measure trade-off

## Code Example
```python
from tensorflow_privacy import DPKerasOptimizer

# Differential privacy: add noise to gradients
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
dp_optimizer = DPKerasOptimizer(
    optimizer,
    l2_norm_clip=1.0,  # Clip gradients
    noise_multiplier=1.0  # Add Gaussian noise
)

# Train with DP optimizer
model.compile(optimizer=dp_optimizer, loss='mse')
model.fit(X, y, epochs=10)
```

## Interview Q&A
**Q: GDPR: user requests right-to-be-forgotten. Model trained on their data. What to do?**
A: (1) Remove their data from training. (2) Retrain model. Or: (3) Use differential privacy (model already doesn't memorize individuals). Or: (4) Use certified forgetting (retraining procedure that provably removes their influence).

**Q: Federated learning: train on hospital data. Hospital sues, says you stole data. How prevent?**
A: (1) Contract specifying data never leaves hospital. (2) Audit: verify no data exfiltrated. (3) Differential privacy: even if gradient intercepted, doesn't reveal individuals. (4) Encryption: data encrypted on hospital device.

## Interview Quick-Reference
| Technique | Best For |
|-----------|----------|
| Diff privacy | Quick privacy, accept slight accuracy drop |
| Federated | Decentralized data (hospitals, phones) |
| Homomorphic | Highest privacy (if speed not critical) |

## Related Topics
- [Data Governance](26-data-governance.md)
- [Differential Privacy](29-differential-privacy.md)

## Resources
- [Differential Privacy Library](https://github.com/tensorflow/privacy)
