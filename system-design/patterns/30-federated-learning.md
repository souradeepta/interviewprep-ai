# Federated Learning

## TL;DR
Train model across decentralized devices (phones, hospitals) without centralizing data. Devices compute locally, send gradient updates, server aggregates. Privacy + collaboration.

## Core Intuition
Hospital A, B, C have patient data. Can't share data (privacy). Solution: send model to each hospital, they train locally, send gradient updates, aggregate.

## How It Works

**Federated learning workflow:**
1. Server sends model to 100 devices
2. Each device trains locally for N iterations
3. Devices send gradient updates to server
4. Server averages gradients
5. Server sends updated model back
6. Repeat

Gradient updates ≠ raw data. Privacy preserved.

| Component | Detail |
|-----------|--------|
| Device | Hospital/phone, trains locally |
| Update | Gradient update (not raw data) |
| Server | Aggregates updates |
| Privacy | Data stays on device |

## Key Properties / Trade-offs
- Privacy: data never leaves device
- Communication: gradient updates sent over network (bandwidth cost)
- Accuracy: local training on limited data might reduce accuracy
- Complexity: coordination across devices is hard

## Detailed Trade-off Analysis

| Aspect | Centralized | Federated (Sync) | Federated (Async) | Hybrid |
|--------|-----------|-----------------|-----------------|--------|
| Privacy | Low | High | High | Medium-high |
| Accuracy | 95%+ | 93-95% | 92-94% | 94-95% |
| Communication rounds | N/A | 100s (slow) | 1000s (data stale) | 100-200 (compromise) |
| Device dropout tolerance | N/A | None (waits) | Full (ignores) | Partial (timeouts) |
| Complexity | Low | High | Very high | Very high |
| Time to convergence | Days | Weeks | Weeks+ | Days-weeks |

**Decision:** Privacy critical + decentralized → federated. Data centralized OK → centralized. Want both speed + privacy → hybrid.

---

## Production Failure Scenarios

**Scenario 1: Gradient leakage attack succeeds**
- Device sends gradient update. Attacker intercepts. Reconstructs training data from gradient.
- Privacy broken despite "federated".
- Prevention: (1) Differential privacy on gradients. (2) Secure aggregation (encrypt in transit). (3) Monitor for privacy attacks.

**Scenario 2: Non-IID data causes poor global model**
- Hospital A has mostly young patients, Hospital B old patients. Local models diverge. Global model weak on both groups.
- Prevention: (1) Periodic global validation. (2) Detect distribution mismatch (KL divergence). (3) Increase communication rounds (more aggregation). (4) Accept accuracy loss as cost of privacy.

**Scenario 3: Device dropout + slow convergence**
- 30% devices offline at any time. Training rounds take 1 month (should be 1 week).
- Budget exceeded, project cancelled.
- Prevention: (1) Async aggregation (don't wait for offline). (2) Faster networks. (3) Compression (smaller updates). (4) Pilot: measure convergence before full deployment.

**Scenario 4: Communication bottleneck**
- 1000 devices, each sends 100MB gradient. Total: 100GB per round. Network saturated.
- Training stalls.
- Prevention: (1) Gradient compression (10-100x). (2) Quantization (int8). (3) Sparsification (send only non-zero gradients). (4) Tier devices (send updates from subset, aggregate, broadcast).

---

## Implementation Guidance

**Wrong:** Federated learning without privacy protection (gradients leakable).
**Right:** Federated + differential privacy + secure aggregation. Complete stack.

**Wrong:** Expect same accuracy as centralized training.
**Right:** Expect 1-5% accuracy drop. Validate on diverse device populations.

---

## Sophisticated Interview Q&A

**Q1: 100 hospitals, train model, converges in 1 month. Acceptable?**
A: Depends on use case. (1) If model improves quarterly: yes. (2) If model must update weekly: no. (3) Solutions: (a) async aggregation (faster). (b) Gradient compression (10-100x). (c) Reduce devices (sample subset). (d) Hybrid (some hospitals centralize, some federate).

**Q2: Federated accuracy 2% worse than centralized. Is the privacy gain worth it?**
A: (1) Quantify: 2% on what metric? (2) Business impact: if accuracy >90%, maybe acceptable. If accuracy 70%, unacceptable. (3) Cost of centralization: privacy breach = $100K fine. 2% accuracy = $10K impact. (4) Privacy > accuracy for regulated domains.

**Q3: Gradient update intercepted. How prevent privacy leak?**
A: (1) Differential privacy: add noise to gradient before sending. (2) Secure aggregation: encrypt gradient, server aggregates encrypted values (can't see individual gradients). (3) Secure enclaves: compute on trusted hardware. (4) Combination of all three for maximum privacy.

**Q4: Non-IID data: how detect if federated learning failing?**
A: (1) Monitor convergence: is loss decreasing? If plateaus early, distribution mismatch. (2) Fairness metrics: accuracy per device. If varies wildly, non-IID. (3) Distribution test: KL divergence between device data distributions. Threshold >1.0 = non-IID. (4) Validate on each device's test set.

---

## Cost & Resource Analysis

**Federated learning infrastructure:** TensorFlow Federated, PySyft, or custom framework. Development: 4-8 weeks = $50-100K.
**Communication cost:** Gradient transmission, encryption, aggregation. For 100 devices, ~$5K/month bandwidth.
**Privacy tools:** Differential privacy, secure aggregation libraries. Minimal cost (open-source).
**Coordination overhead:** Device management, async handling, fault tolerance. 1 FTE = $100K/year.

**Cost of centralized alternative:** Privacy breach lawsuit $100K-1M+. Compliance fine $4% revenue.
**ROI:** Federated investment $200-400K initial + $100K/year operations. Prevents incident $100K+/year. Break-even 1-2 years.

---

## Monitoring & Observability

**Key metrics:** Convergence rate (rounds to target accuracy), communication cost (bits/round), device dropout rate, non-IID divergence (KL between device distributions), global model accuracy, per-device accuracy variance, privacy budget (ε) if using DP

**Alerts:** Convergence stalled (accuracy not improving), communication bottleneck (high latency), device dropout >50%, accuracy drops after aggregation, non-IID divergence too high, privacy attack detected

## Common Mistakes / Gotchas
- Gradient updates leak data (privacy attack possible)
- Device dropout (devices go offline mid-training)
- Non-IID data (devices have different data distributions)
- Communication bottleneck (hundreds of devices × large gradients)

## Best Practices
- **Secure aggregation:** encrypt gradient updates (can't see individual updates)
- **Compression:** compress gradients before sending (reduce bandwidth)
- **Async update:** devices don't need to synchronized (handles dropout)
- **Privacy + federation:** combine differential privacy + federated learning

## Code Example
```python
# Simplified federated learning
def federated_training():
    server_model = load_model()
    
    for round in range(num_rounds):
        # Send model to devices
        device_updates = []
        
        for device in devices:
            local_model = copy(server_model)
            # Device trains locally
            local_update = device.train(local_model, local_data)
            device_updates.append(local_update)
        
        # Server aggregates
        avg_update = average(device_updates)
        server_model = apply_update(server_model, avg_update)
```

## Interview Q&A
**Q: Federated learning: 100 devices, 50% go offline. Training fails?**
A: No. Async aggregation: server aggregates updates from online devices, discards offline. Model still improves (slower, but works). Next round, different devices participate.

**Q: Gradient updates leak privacy. How to prevent?**
A: Differential privacy + secure aggregation. (1) Add noise to gradients on device. (2) Encrypt before sending. (3) Server aggregates encrypted updates (can't see individual gradients). Privacy preserved.

## Interview Quick-Reference
| Component | Purpose |
|-----------|---------|
| Device training | Privacy (data stays local) |
| Gradient update | Communication (efficient) |
| Aggregation | Model improvement |

## Related Topics
- [Privacy-Preserving ML](28-privacy-preserving-ml.md)
- [Differential Privacy](29-differential-privacy.md)

## Resources
- [Federated Learning: Collaborative ML without Centralizing Data](https://arxiv.org/abs/1602.05629)
