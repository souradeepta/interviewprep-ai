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

**Decision:** GDPR compliant -> differential privacy. Decentralized data -> federated. Maximum privacy + not speed-critical -> homomorphic.

---

## Production Failure Scenarios

**1. Gradient Inversion Attack**
- **Symptom:** Federated learning client gradients are reconstructed by a researcher, revealing individual user records that were never shared.
- **Root Cause:** Raw gradients sent to the aggregation server without noise addition; high-resolution gradients contain enough signal to reconstruct inputs.
- **Detection:** Gradient similarity test -- if cosine_sim(intercepted_gradient, original_data_gradient) > 0.95, data can be reconstructed; run this test in a shadow environment before production.
- **Fix:** Add DP-SGD noise (epsilon=1.0, delta=1e-5) to gradients before aggregation; clip gradient norm to C=1.0 to bound sensitivity; verify with membership inference test post-deployment.

**2. Membership Inference Attack**
- **Symptom:** Attacker can determine with 80% accuracy whether a specific medical record was in the model's training set, enabling targeted deanonymization.
- **Root Cause:** Model overfits on rare training examples -- rare patients appear in model weights at detectable signal strength.
- **Detection:** Shadow model attack simulation on held-out records: if attack accuracy > 60% (vs 50% baseline), the model memorizes training data.
- **Fix:** Apply differential privacy at training (epsilon < 10); enforce early stopping; add L2 regularization; avoid including rare outlier records (n < 5 in a subgroup) in training data.

**3. Trusted Aggregator Compromise**
- **Symptom:** Aggregation server operator accesses individual client updates, violating the "server sees only aggregates" privacy guarantee.
- **Root Cause:** No cryptographic aggregation protocol -- server receives raw individual gradients and can read them before averaging.
- **Detection:** Audit whether Secure Aggregation (SecAgg) protocol is implemented; test by attempting to log individual client updates at the server layer.
- **Fix:** Implement SecAgg using threshold secret sharing (Bonawitz et al.); no single party can reconstruct individual client contributions; requires minimum participation threshold (e.g., 80% of selected clients) for aggregation to proceed.

**4. Re-Identification via Aggregate Statistics**
- **Symptom:** Published aggregate statistics about rare demographic groups allow re-identification of specific individuals who have a unique combination of attributes.
- **Root Cause:** Aggregates published without noise addition; groups with n < 5 members are effectively identifiable.
- **Detection:** k-anonymity check on all published aggregates (k < 5 = risky; k < 10 = caution); run before any external publication.
- **Fix:** Suppress aggregates with fewer than 10 records; add Laplace noise calibrated to sensitivity of the statistic before publication; document suppression decisions for audit.

---

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| DP-SGD compute overhead (40% more) | $2/hr x 1.4 | 100 hr/mo | $280 |
| SecAgg protocol overhead | 30% comm. overhead | -- | ~$150 bandwidth |
| Privacy accounting tooling | $0 | (open source) | $0 |
| Privacy engineer (specialist) | $250/hr | 20 hr/mo | $5,000 |
| **Total** | | | **~$5,430/month** |

The dominant cost in privacy-preserving ML is specialized talent, not infrastructure. A privacy engineer who can correctly tune epsilon budgets, run membership inference audits, and implement SecAgg correctly is rare and commands $200-300/hr. The compute overhead for DP-SGD is modest (~40% more GPU hours) and is often offset by the ability to train on datasets that would otherwise be off-limits due to data sharing restrictions. For organizations in healthcare or finance where training data access is the bottleneck, the privacy infrastructure investment pays for itself by unlocking more data: a hospital consortium training a federated model can often outperform a centralized model trained on limited consented data, even accounting for the 1-3% federated accuracy penalty.

---

## Implementation Guidance

**Wrong:** Remove identifying columns (name, email), assume data is anonymized.
**Right:** Use differential privacy (provable). Or: k-anonymity with re-identification testing. Don't claim privacy without validation.

**Wrong:** DIY differential privacy implementation.
**Right:** Use published libraries (TensorFlow Privacy, PyDP). Peer-reviewed, battle-tested.

---

## Interview Q&A

**Q: A user exercises their GDPR right-to-be-forgotten. The model was trained on their data. What do you do?**
A: Three options in order of cost: (1) If you used differential privacy during training (epsilon < 10), the model provably does not memorize individuals -- document this in your privacy policy and treat the request as satisfied at the data deletion layer; (2) If no DP, use certified unlearning (machine unlearning algorithms that provably reduce a user's influence without full retraining -- computationally cheaper); (3) If no unlearning infrastructure, retrain from scratch on the dataset minus that user. The upfront cost of implementing DP training typically pays for itself the first time you face this request at scale.

**Q: Your federated model's accuracy is 3% lower than the centralized baseline. The business wants to close this gap. What levers do you have?**
A: Four levers: (1) Increase communication rounds (more aggregation cycles = more convergence, but more latency and bandwidth cost); (2) Use knowledge distillation: the server trains a distillation model on public data augmented with client predictions, closing part of the accuracy gap without accessing private data; (3) Personalized federated learning (pFedMe or MAML-based approaches): keep a global model but allow clients to fine-tune locally for their distribution; (4) Hybrid: federate only the most privacy-sensitive data, centralize the rest. Accept that 1-3% gap is often the irreducible cost of privacy for highly non-IID data distributions.

**Q: When would you recommend homomorphic encryption over differential privacy?**
A: Homomorphic encryption is justified when you need correctness guarantees that DP cannot provide: exact computation on encrypted inputs without any accuracy loss. The practical conditions are: (1) the use case is batch inference, not real-time (1000x compute overhead is acceptable); (2) the data is extremely sensitive (military, top-secret classification); (3) the computation graph is relatively simple (HE complexity scales with circuit depth). For most production ML, DP is preferable -- it is 1000x cheaper and the 1-5% accuracy trade-off is acceptable. HE remains primarily a research and niche high-security deployment technology in 2026.

**Q: How do you validate that your differential privacy implementation is actually providing the privacy guarantee you claim?**
A: Four-step validation: (1) Use a known-good library (Opacus for PyTorch, TF Privacy for TensorFlow) rather than a custom implementation; (2) Run the privacy accountant after each training epoch to track cumulative epsilon -- alert if it exceeds your budget; (3) Run a membership inference attack simulation on held-out records: if attack accuracy > 55% for epsilon <= 1.0, the implementation is incorrect; (4) Have a privacy specialist audit the epsilon and delta values against your threat model. The common mistake is setting epsilon = 10 and claiming DP compliance -- epsilon > 10 provides negligible practical protection.

**Q: A colleague argues that removing all PII columns is sufficient privacy protection. How do you respond?**
A: Anonymization via column removal is consistently broken by linkage attacks. The canonical example: the AOL search dataset was "anonymized" but re-identified by journalists within days using public records. Netflix's anonymized rating dataset was re-identified using IMDB. The root problem is that removing obvious identifiers (name, SSN) leaves auxiliary identifiers (zip code + age + gender identifies 87% of the US population). Differential privacy solves this by providing a formal, quantifiable guarantee that is not dependent on assumptions about what an attacker knows. For any dataset that will be published, shared across organizational boundaries, or used to train a model that will be shared externally, DP or federated learning is the appropriate tool.

**Q: How do you detect and respond to a gradient inversion attack on a federated learning deployment?**
A: Detection: monitor gradient cosine similarity between client updates and a set of probe inputs -- if any client's gradient reconstructs a probe with similarity > 0.95, flag for investigation. Also monitor for clients sending unusually large gradient norms (clipping violation). Response: (1) immediately add DP-SGD noise to all gradients from that client and quarantine its updates; (2) rotate the aggregation key if using SecAgg; (3) investigate whether the attacking client is compromised or running a modified client binary; (4) if you have not already deployed SecAgg, do so immediately -- it prevents the server from seeing individual client gradients even if they are unencrypted on the wire.

**Q: Your organization needs to train a model on hospital data across 10 institutions. None of them will share raw data. Design the system.**
A: Use federated learning with three privacy layers: (1) DP-SGD at each institution (epsilon=1.0, delta=1e-5) before sending gradients; (2) SecAgg at the aggregation server so the server cannot read individual hospital gradients; (3) HTTPS with mutual TLS for transport. For aggregation: use FedAvg with a minimum participation threshold of 7/10 institutions per round to ensure statistical stability. For the non-IID data problem (each hospital has a different patient demographic): run FedProx (adds a proximal term to keep local models close to the global model). For validation: each institution holds out 10% of their data for local validation; aggregate validation metrics without exposing individual results. Expected accuracy gap vs centralized: 1-4% depending on data heterogeneity -- document this in the model card.

**Q: What are the first signs that a privacy-preserving ML deployment is failing in production?**
A: Four warning signs: (1) privacy budget exhausted faster than planned -- someone is making more queries or running more training epochs than the epsilon budget was allocated for; (2) membership inference attack accuracy rising above 55% in continuous monitoring -- model is memorizing more than expected, possibly due to distribution shift introducing rare records; (3) accuracy degrading faster than expected -- noise levels may be too high relative to the current dataset size, or the data distribution has shifted; (4) compliance audit flags -- model card doesn't have a documented epsilon value, or the epsilon was set but not validated against an actual threat model. Any of these warrants an immediate review of the privacy configuration.

---

## Monitoring & Observability

**Key metrics:** Privacy guarantee epsilon value, accuracy loss due to privacy mechanisms, federated communication rounds, gradient leakage detection (monitor for privacy attacks), privacy audit status, compliance certification

**Alerts:** Privacy guarantee violated (epsilon exceeded), privacy attack detected (suspicious gradient patterns), federated accuracy drops unexpectedly, privacy audit overdue, compliance certificate expires

## Common Mistakes / Gotchas
- Assuming anonymization = privacy (can be re-identified)
- Insufficient noise (privacy attack succeeds)
- No privacy audit (privacy claims unvalidated)
- Ignoring computational cost (homomorphic encryption very slow)

## Best Practices
- **Privacy audit:** hire external party to verify privacy
- **Differential privacy:** use published libraries (not DIY)
- **Federated:** careful gradient aggregation (averaging alone leaks data via gradient inversion)
- **Benchmark privacy vs accuracy:** measure trade-off before production deployment

## Code Example
```python
from tensorflow_privacy import DPKerasOptimizer

# Differential privacy: add noise to gradients
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
dp_optimizer = DPKerasOptimizer(
    optimizer,
    l2_norm_clip=1.0,  # Clip gradients to bound sensitivity
    noise_multiplier=1.0  # Add Gaussian noise proportional to sensitivity
)

# Train with DP optimizer
model.compile(optimizer=dp_optimizer, loss='mse')
model.fit(X, y, epochs=10)
```

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
