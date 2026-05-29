# Differential Privacy

## TL;DR
Add noise to data or gradients during training. Model cannot memorize individuals (membership inference attack fails). Formally proven privacy: epsilon-differential privacy (smaller epsilon = more private).

## Core Intuition
Data = {Alice, Bob, Charlie}. Train model on data. Model memorizes Alice. Attack: "is Alice in training data?" -> Yes (privacy leak). Solution: add noise so model can't memorize Alice.

## How It Works

**Differential privacy:**
- Add Gaussian noise to gradients during training
- Smaller epsilon = more noise = more privacy (but less accuracy)
- epsilon=1: good privacy. epsilon=10: less privacy.

| epsilon | Privacy | Accuracy |
|---------|---------|----------|
| 0.5 | Very high | 90% |
| 1.0 | High | 92% |
| 5.0 | Medium | 94% |
| 10.0 | Low | 95% |

## Key Properties / Trade-offs
- Privacy vs accuracy: more privacy -> more accuracy drop
- Complexity: requires clipping and noise addition
- Proof: differential privacy = formal proof (unlike heuristics)

## Detailed Trade-off Analysis

| Privacy Guarantee (epsilon) | Accuracy | Noise Level | Use Case | Regulatory OK |
|-----------------------------|----------|------------|----------|--------------|
| 0.1 | 88% | Very high | Academic research | Yes |
| 1.0 | 92% | High | Industry standard | Yes |
| 5.0 | 94% | Medium | Borderline | Maybe |
| 10.0 | 95% | Low | Weak guarantee | No |
| infinity (no DP) | 96% | None | No privacy | No |

**Decision:** Regulated (healthcare, finance) -> epsilon <= 1.0. Compliant (GDPR) -> epsilon <= 5.0. Startup MVP -> epsilon > 5.0 (trade privacy for accuracy).

---

## Production Failure Scenarios

**1. Epsilon Budget Exhausted**
- **Symptom:** Analytics team blocked from running queries after the weekly epsilon = 10.0 budget is consumed on day 2; downstream reports fail.
- **Root Cause:** No budget planning; each ad-hoc query consumed epsilon = 0.5 without tracking cumulative spend; no alerting at budget thresholds.
- **Detection:** Track cumulative epsilon via privacy accountant (Google DP library); alert at 80% consumption; log each query's epsilon cost to a budget ledger.
- **Fix:** Plan epsilon allocation upfront by use case (e.g., reporting: epsilon = 3.0, model training: epsilon = 5.0, analytics: epsilon = 2.0); use advanced composition theorem for sequential queries to tighten the bound; reset budget weekly or monthly on a fixed schedule.

**2. Noise Destroys Minority Subgroup Signal**
- **Symptom:** DP-protected aggregate statistics for groups with n < 50 are effectively useless -- confidence intervals span the entire plausible range.
- **Root Cause:** Noise calibrated to global sensitivity of the full population; for small groups the noise-to-signal ratio is overwhelming.
- **Detection:** Compute confidence intervals on DP stats per subgroup; if CI width > 50% of the point estimate, suppress and flag the statistic.
- **Fix:** Enforce minimum group size threshold (n >= 100) before reporting any subgroup statistic; suppress smaller groups with an explanatory note in the output; if small-group reporting is required, explore local DP with per-record guarantees rather than global DP.

**3. Epsilon Chosen Without Calibration**
- **Symptom:** A model is trained with epsilon = 100 and described as "DP-compliant" in the model card -- the actual privacy protection is negligible.
- **Root Cause:** Team set epsilon based on trial-and-error accuracy optimization without understanding the epsilon scale (epsilon < 1 = strong; epsilon > 10 = weak; epsilon = 100 = essentially no protection).
- **Detection:** Audit epsilon values in all model cards; flag any epsilon > 10 for privacy team review; require documented justification for epsilon > 5.
- **Fix:** Maintain a company-wide epsilon policy: regulated domains (HIPAA, GDPR) require epsilon <= 1.0; internal analytics epsilon <= 5.0; document epsilon interpretation in every model card alongside a threat model statement.

**4. Sensitivity Underestimated**
- **Symptom:** The DP mechanism adds insufficient noise; a privacy audit reveals the actual protection is weaker than the claimed epsilon because outlier records were not clipped.
- **Root Cause:** Global sensitivity computed on a bounded training sample; unbounded outliers exist in production (e.g., a user with 10,000 transactions while the dataset assumed max 500).
- **Detection:** Compare sensitivity estimate vs realized maximum gradient norm across a production data sample; alert if realized max exceeds the assumed sensitivity by more than 10%.
- **Fix:** Clip all inputs to a bounded range before sensitivity computation; validate the clipping bound using the 99th percentile of the production distribution; re-derive the sensitivity guarantee on the clipped distribution and update the model card.

---

## Cost Model

| Resource | Unit Cost | Volume | Monthly Cost |
|----------|-----------|--------|-------------|
| DP-SGD training overhead | $2/hr x 40% overhead | 100 hr/mo | $80 |
| Privacy accountant compute | Negligible | -- | $1 |
| Analyst query system (DP noise) | $0.001/query | 10K queries/mo | $10 |
| Privacy officer review | $300/hr | 4 hr/mo | $1,200 |
| **Total** | | | **~$1,291/month** |

Differential privacy is the most cost-efficient of the privacy-preserving techniques. The infrastructure overhead (DP-SGD adds approximately 40% more training compute due to per-sample gradient computation) is modest, and the tooling is open source. The dominant cost is specialized human time: a privacy officer who can interpret epsilon guarantees, set policy, and respond to audit inquiries costs $1,200/month at 4 hours. For organizations making public claims about DP compliance (in product documentation or regulatory filings), budget an additional $10-50K annually for external privacy audit certification, which is a one-time cost that amortizes across all DP-trained models.

---

## Implementation Guidance

**Wrong:** DIY differential privacy. Implement gradient clipping + noise yourself.
**Right:** Use library (TensorFlow Privacy or Opacus). Peer-reviewed, proven correct. Easier to get right.

**Wrong:** Set noise_multiplier arbitrarily (e.g., 0.5). Hope privacy is good.
**Right:** Calculate epsilon budget from privacy requirements. Iterate: test accuracy, adjust noise, verify privacy.

---

## Interview Q&A

**Q: A colleague says epsilon = 1.0 means only 1% of privacy is lost. Correct them.**
A: This is a common misinterpretation. Epsilon = 1.0 is a mathematical bound, not a percentage. The formal guarantee is: the probability ratio of any output on a dataset including person X vs a dataset excluding person X is bounded by e^epsilon. At epsilon = 1.0, e^1 = 2.72 -- meaning an attacker's advantage from seeing the output is bounded by 2.72x. Practically: a membership inference attack (is person X in the training set?) has success probability at most (e^1)/(1 + e^1) = 73% vs 50% random guessing. Lower epsilon = tighter bound = less attacker advantage.

**Q: Accuracy drops 5% with DP. How do you justify this to the business?**
A: Quantify the trade-off in business terms: (1) what is the revenue impact of 5% accuracy degradation? (e.g., if the model drives $10M/year revenue, 5% degradation = ~$500K impact); (2) what is the expected cost of a privacy breach in this domain? (GDPR fine up to 4% of global revenue + litigation + reputation damage -- easily $1M-100M for a major breach); (3) are there architectural options to close the gap? (larger model, more training epochs, better data cleaning can often recover 2-3% of the DP accuracy penalty). Present these three numbers together; in regulated domains, the trade-off almost always favors DP.

**Q: You need to implement DP for a production PyTorch model. What library do you use and why?**
A: Opacus (Meta AI) for PyTorch. Three reasons: (1) it handles the per-sample gradient computation automatically (the most error-prone part of DP-SGD); (2) it provides a built-in privacy accountant that tracks cumulative epsilon and delta as you train; (3) it works with standard PyTorch DataLoaders with minimal code changes. For TensorFlow, use TF Privacy. Both are peer-reviewed and widely deployed in production. Never implement DP-SGD from scratch: the subtle correctness requirements (sampling without replacement, correct noise scaling with batch size) are easy to get wrong in ways that invalidate the privacy guarantee.

**Q: What is the difference between local differential privacy and global differential privacy? When would you use each?**
A: In global DP, a trusted aggregator adds noise to the aggregate result after collecting raw data; in local DP, each user adds noise to their own data before sending it. Use global DP when you have a trusted data collector (first-party analytics): stronger utility because noise is added once to the aggregate, not to each individual record. Use local DP when you cannot trust the data collector (e.g., Apple/Google collecting data from user devices): stronger privacy guarantees even against a compromised server, but at the cost of much higher noise levels (local DP typically requires 10-100x more data for the same accuracy as global DP). The practical rule: if you control the pipeline end-to-end, use global DP; if the data must leave the user's device before aggregation, use local DP.

**Q: How do you handle the epsilon budget for a model that is retrained monthly?**
A: Composition: each retraining consumes epsilon from the budget. Two approaches: (1) treat each model version independently with its own epsilon budget (simpler but allows unbounded total exposure over time); (2) use the moments accountant or zero-concentrated DP (zCDP) framework to track cumulative epsilon across all training runs on the same individuals (more conservative). For monthly retraining with the same user cohort, zCDP is the correct tool: it provides tighter composition bounds than naive epsilon addition. In practice, if individuals rotate out of the training set (e.g., only the last 12 months of data) the budget resets naturally as their records leave the training window.

**Q: A security team finds that your DP model's predictions can be used to infer whether a specific record was in the training set with 65% accuracy. You claimed epsilon = 1.0. What went wrong?**
A: 65% membership inference accuracy is higher than the theoretical bound for epsilon = 1.0 (~73% maximum, but typical DP-trained models at epsilon = 1.0 are much closer to 50%). Possible root causes to investigate: (1) sensitivity underestimation -- outlier records not clipped, violating the assumed sensitivity bound; (2) incorrect batch sampling -- the privacy accounting assumes Poisson subsampling, but if the DataLoader uses sequential (non-random) sampling, the accounting is invalid; (3) library misconfiguration -- e.g., Opacus max_grad_norm set too high, allowing individual gradients to dominate; (4) data leak outside the model -- the attacker may be exploiting a non-DP side channel (e.g., prediction confidence scores rather than class labels). Audit each in order.

**Q: How would you set up a DP-protected analytics system for a company that needs to publish monthly statistics about user behavior?**
A: Three-layer design: (1) define the epsilon budget per reporting period (e.g., epsilon = 2.0/month per user cohort); (2) implement a query gateway that intercepts all aggregate queries, adds Laplace noise calibrated to global sensitivity before returning results, and deducts from the epsilon ledger; (3) enforce minimum group size (n >= 50) before any subgroup statistic is computed, suppressing smaller groups. For the reporting layer: publish only pre-defined, pre-budgeted statistics rather than allowing ad-hoc queries (prevents budget exhaustion via exploratory analysis). Audit the epsilon ledger monthly and reject queries once the budget is 80% consumed.

**Q: When is differential privacy NOT the right tool, and what should you use instead?**
A: DP is not the right tool in four situations: (1) the data is already public or aggregated to the point where individual inference is impossible -- DP adds cost without benefit; (2) the accuracy requirement is absolute and the DP accuracy penalty is unacceptable (e.g., rare disease diagnosis requiring >99% sensitivity) -- explore federated learning with secure aggregation instead; (3) you need to publish exact statistics that must be auditable -- DP noise makes exact auditability impossible; (4) the dataset is tiny (n < 1000) -- DP noise completely drowns the signal. In cases 2 and 3, federated learning + secure aggregation provides strong practical privacy without accuracy loss. In case 4, consider whether the model should be built at all on such a small dataset.

---

## Monitoring & Observability

**Key metrics:** Privacy budget epsilon consumed per epoch, accuracy loss percentage, noise multiplier in use, membership inference attack success rate, privacy coverage (% of training data under DP protection)

**Alerts:** Epsilon budget exceeded (privacy guarantee violated), accuracy drops below threshold, noise_multiplier configuration incorrect, membership inference attack succeeds (privacy broken)

## Common Mistakes / Gotchas
- DIY differential privacy (easy to get wrong)
- Insufficient noise (privacy bound too loose)
- No privacy audit (claims not validated)
- Confusing epsilon values (different libraries use different scales)

## Best Practices
- **Use library:** TensorFlow Privacy, Opacus (PyTorch)
- **Test privacy:** membership inference attack to validate
- **Report epsilon value:** always report privacy budget
- **Noise scheduling:** start high noise, reduce over training
- **Batch size:** smaller batch -> more privacy needed

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

## Interview Quick-Reference
| epsilon | Privacy Level |
|---------|---------------|
| 0.1 | Excellent (academic research) |
| 1.0 | Good (industry standard) |
| 10.0 | Weak (privacy claim but not strong) |

## Related Topics
- [Privacy-Preserving ML](28-privacy-preserving-ml.md)
- [Data Governance](26-data-governance.md)

## Resources
- [Differential Privacy Library](https://github.com/tensorflow/privacy)
- [Opacus: PyTorch DP Library](https://opacus.ai/)
