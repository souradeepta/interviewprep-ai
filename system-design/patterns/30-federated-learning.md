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

Gradient updates are not raw data. Privacy preserved.

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

**Decision:** Privacy critical + decentralized -> federated. Data centralized OK -> centralized. Want both speed + privacy -> hybrid.

---

## Production Failure Scenarios

**1. Straggler Clients Block Aggregation**
- **Symptom:** Synchronous FL round takes 10x longer than the median client time because 5% of clients are slow; entire training pipeline stalls waiting for them.
- **Root Cause:** Wait-for-all-clients policy in synchronous aggregation; no timeout or straggler tolerance built into the round protocol.
- **Detection:** Track per-round wall-clock time vs median client completion time; alert when round duration exceeds 3x the median client time.
- **Fix:** Switch to async aggregation (FedAsync) or implement straggler tolerance in synchronous FL -- drop the slowest 20% of clients per round and proceed; tune the minimum client participation threshold to balance convergence stability vs round speed.

**2. Client Data Heterogeneity**
- **Symptom:** Global model performs well on the majority locale but poorly on minority-locale clients; aggregate accuracy looks fine while per-client accuracy variance is high.
- **Root Cause:** Non-IID data across clients -- majority locale dominates the gradient direction during FedAvg; minority clients' gradients are consistently overridden.
- **Detection:** Log per-client validation loss after each round; flag clients where local loss diverges by more than 2 standard deviations from the global mean.
- **Fix:** Use FedProx (adds a proximal regularization term to keep local models close to the global model); alternatively, use personalized FL (pFedMe) which maintains a personalized model per client on top of the global model; increase communication rounds to give minority clients more gradient influence.

**3. Client Dropout Corrupts Round**
- **Symptom:** Model diverges after a round where 40% of selected clients dropped out mid-training; the aggregated update is heavily biased toward the 60% that completed.
- **Root Cause:** No minimum participation threshold enforced; aggregation proceeded with an insufficient number of clients to produce a statistically stable update.
- **Detection:** Track client completion rate per round; alert and abort the round if completion rate falls below the configured threshold (e.g., 70% of selected clients).
- **Fix:** Set a hard minimum participation rate (e.g., 70%); abort and retry the round with a fresh client selection if the threshold is not met; implement exponential backoff for retry to avoid thundering herd on a recovering network.

**4. Poisoning Attack**
- **Symptom:** After a federated training round, the global model starts producing systematically biased or malicious outputs -- for example, recommending harmful content for a specific trigger phrase.
- **Root Cause:** Byzantine clients sending adversarial gradients designed to steer the global model toward a backdoor behavior; FedAvg has no native defense against malicious updates.
- **Detection:** Monitor gradient norm distribution across clients per round; flag clients whose gradient norm exceeds 3 standard deviations from the mean; run consistency checks on the post-aggregation model against a trusted validation set.
- **Fix:** Deploy robust aggregation (Krum selects the update minimizing distance to neighbors; coordinate-wise median is Byzantine-robust); clip all client gradient norms to a global bound before aggregation; add anomaly detection on client update patterns; consider requiring client attestation (signed updates from verified device environments).

---

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| Central aggregation server | $0.50/hr | 720 hr | $360 |
| Communication bandwidth | $0.09/GB | 100K clients x 100MB x 10 rounds | $9,000 |
| Client compute (amortized) | -- | On-device, client-paid | $0 |
| FL framework ops | $200/hr | 10 hr/mo | $2,000 |
| **Total** | | | **~$11,360/month** |

Communication bandwidth dominates the federated learning cost structure -- at 100K clients sending 100MB updates for 10 rounds per month, bandwidth alone is $9,000/month. This is the primary lever for optimization: gradient compression (TopK sparsification, quantization to int8) can reduce update size by 10-100x, potentially cutting bandwidth costs to $90-900/month. The aggregation server itself is cheap ($360/month) because aggregation is computationally simple. The FL framework operations cost ($2,000/month) reflects the coordination complexity -- client scheduling, round management, straggler handling -- which requires dedicated engineering attention especially at scales above 10K clients. For large-scale deployments (millions of clients, as in Google Gboard), the infrastructure investment is justified by the data access it unlocks: federated training on billions of user interactions that could never be centralized.

---

## Implementation Guidance

**Wrong:** Federated learning without privacy protection (gradients leakable).
**Right:** Federated + differential privacy + secure aggregation. Complete stack.

**Wrong:** Expect same accuracy as centralized training.
**Right:** Expect 1-5% accuracy drop. Validate on diverse device populations.

---

## Interview Q&A

**Q: 100 hospitals want to train a shared diagnostic model. Convergence is taking 6 weeks. How do you speed it up?**
A: Four levers, ranked by impact: (1) async aggregation -- don't wait for all hospitals each round; aggregate as updates arrive with a staleness weight (FedAsync); (2) gradient compression -- reduce update size 10-100x using TopK sparsification or quantization to int8; (3) reduce devices per round -- instead of all 100 hospitals each round, sample a random subset of 30; convergence is only slightly slower but rounds complete 3x faster; (4) increase local epochs -- more local training steps per round means fewer total rounds, at the cost of higher non-IID divergence. Combine (1) and (2) first; they usually halve the total wall-clock time without sacrificing accuracy.

**Q: Federated accuracy is 2% lower than the centralized baseline. Is the privacy tradeoff worth it?**
A: Reframe the question in business terms: (1) what is the revenue impact of 2% accuracy degradation? (2) what is the alternative cost -- can you actually centralize this data? In healthcare, centralization may be legally impossible or would take 18+ months of compliance work; 2% is free compared to that alternative. (3) Is the 2% gap irreducible? Non-IID data heterogeneity is the main driver -- if you can homogenize the data distribution across clients (e.g., by standardizing data collection protocols), you can often close 50-70% of the gap. For regulated domains, the privacy protection typically justifies gaps up to 5%.

**Q: A gradient update from one client is intercepted in transit. What private information is exposed?**
A: With raw gradients and no protection, a gradient inversion attack can reconstruct training samples with high fidelity (see Zhu et al. 2019). The amount of information exposed depends on the model architecture and gradient resolution. Mitigation stack: (1) HTTPS with mutual TLS for transport encryption (prevents passive interception); (2) DP-SGD noise added by the client before sending (makes reconstruction computationally infeasible even if the gradient is captured); (3) SecAgg (Secure Aggregation) so the server only sees the sum of encrypted gradients, not individual client updates. Each layer addresses a different threat: TLS for network adversaries, DP for malicious servers, SecAgg for honest-but-curious servers.

**Q: How do you detect when federated learning is failing due to non-IID data distributions?**
A: Three diagnostic signals: (1) per-client validation loss variance -- if the global model's loss varies by more than 2x across clients, non-IID is causing unequal fit; (2) gradient direction divergence -- compute cosine similarity between individual client gradients and the global FedAvg gradient; low similarity (<0.3) indicates clients are pulling in very different directions; (3) KL divergence of client label distributions -- if one hospital has 90% elderly patients and another has 90% pediatric patients, their label distributions diverge significantly. When detected, switch from FedAvg to FedProx or personalized FL variants; increase the communication frequency to give the global model more signal from minority clients.

**Q: Design a federated learning system for a keyboard autocomplete model on 100M mobile devices.**
A: This is close to the actual Google Gboard design. Key decisions: (1) client selection -- randomly sample 1,000-5,000 devices per round from those that are idle, charging, and on WiFi (reduces bandwidth cost and battery impact); (2) local training -- 1-5 epochs of SGD on local typing history; (3) update compression -- TopK sparsification (send top 1% of gradient values by magnitude) + quantization to 16-bit floats; reduces update size from 50MB to <1MB; (4) aggregation -- FedAvg with secure aggregation (SecAgg) so no individual's updates are visible to Google; (5) DP -- add Gaussian noise per the DP-SGD protocol before SecAgg; (6) evaluation -- hold out 0.1% of devices as an evaluation cohort; measure next-word prediction accuracy after each aggregation round.

**Q: A federated training round produces a global model that performs worse than the previous round. How do you investigate?**
A: Check in order: (1) client dropout rate for that round -- if >30% of clients dropped, the update is statistically unstable; replay the round with the previous round's model as initialization; (2) gradient norm outliers -- were any clients sending abnormally large gradients that dominated the average? Inspect the norm distribution; (3) data distribution shift -- did any client's local dataset change significantly between rounds? (4) round size -- was this round unusually small (fewer clients than the minimum threshold)? (5) poisoning -- if all other checks are clean, investigate whether any clients sent adversarial updates; compare pre- and post-aggregation model weights for unusual spikes. Add a quality gate: only accept the aggregated model if it improves over the previous round on a trusted holdout set.

**Q: What is the minimum privacy stack for a production federated learning deployment handling sensitive medical data?**
A: Three non-negotiable layers: (1) DP-SGD at the client (epsilon <= 1.0, delta = 1e-5) -- protects against gradient inversion even if the server is compromised; (2) SecAgg at the aggregation layer -- prevents the server from seeing individual client gradients, even before noise is added; (3) HTTPS with certificate pinning for transport -- prevents man-in-the-middle interception. Optional fourth layer: Trusted Execution Environments (TEEs) on the aggregation server for additional hardware-level isolation. Document the full privacy guarantee in the system's privacy notice: "No individual patient record leaves the hospital. The aggregation server sees only the cryptographic sum of noise-added gradients. The trained model satisfies epsilon = 1.0 differential privacy."

**Q: Your federated training cost $9,000/month in bandwidth. The project budget is $3,000/month. How do you cut costs without abandoning federated learning?**
A: Three techniques for bandwidth reduction: (1) TopK gradient sparsification -- send only the top 1% of gradient values by magnitude; 100x reduction in update size with typically <1% accuracy loss; cost: $90/month; (2) gradient quantization -- compress float32 (4 bytes/param) to int8 (1 byte/param); 4x reduction; cost: $2,250/month; (3) reduce clients per round -- instead of all 100K clients, sample 10K per round; 10x bandwidth reduction; requires more rounds for convergence but overall cost is similar; (4) increase local epochs -- more local training per round reduces total rounds needed, but watch for non-IID divergence. Combining (1) and (2) alone gets you from $9,000 to under $600/month while maintaining >98% of the original model quality.

---

## Monitoring & Observability

**Key metrics:** Convergence rate (rounds to target accuracy), communication cost (bits/round), device dropout rate, non-IID divergence (KL between device distributions), global model accuracy, per-device accuracy variance, privacy budget (epsilon) if using DP

**Alerts:** Convergence stalled (accuracy not improving), communication bottleneck (high latency), device dropout >50%, accuracy drops after aggregation, non-IID divergence too high, privacy attack detected

## Common Mistakes / Gotchas
- Gradient updates leak data (privacy attack possible)
- Device dropout (devices go offline mid-training)
- Non-IID data (devices have different data distributions)
- Communication bottleneck (hundreds of devices x large gradients)

## Best Practices
- **Secure aggregation:** encrypt gradient updates (can't see individual updates)
- **Compression:** compress gradients before sending (reduce bandwidth)
- **Async update:** devices don't need to synchronize (handles dropout)
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

        # Server aggregates (FedAvg)
        avg_update = average(device_updates)
        server_model = apply_update(server_model, avg_update)
```

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
