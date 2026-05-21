# Shadow Mode

## Detailed Description

Run new model in parallel (offline) with production model. Don't use predictions, just log them. Compare: accuracy on real traffic. Zero user impact. Safest way to test.

## Core Intuition

Shadow = test in dark. New model gets same inputs as production. Makes predictions. Logged, compared to production. Users never see new model predictions. If new model perfect: switch to production. If issues: fix offline.

## How It Works

1. **Shadow phase:** both models running
   - Production model: used for actual predictions
   - Shadow model: same input, predictions logged but not used
   - Compare predictions offline

2. **Comparison:** 
   - Accuracy: does shadow match production?
   - Confidence: if shadow score > 0.95, how often does it agree with prod?
   - Performance: latency, errors

3. **Promotion:** when metrics good → shadow becomes production

| Phase | Prod | Shadow | User Impact |
|-------|------|--------|-------------|
| Baseline | v1.0 | - | v1.0 only |
| Shadow | v1.0 | v2.0 | v1.0 only (v2.0 logged) |
| Compare | v1.0 | v2.0 | v1.0 only |
| Promote | v2.0 | - | v2.0 only |

## Key Properties / Trade-offs
- Safety: test on real users without affecting them
- Cost: 2x inference (both models running)
- Time: slow (need days to confidence comparison)

## Common Mistakes / Gotchas
- Shadow model slower → bottleneck (block production)
- No comparison → run shadow, forget about it
- Different input versions → shadow sees different data than prod
- Agreement threshold too high → never promote (waiting for 100% agreement)

## Detailed Trade-off Analysis

### Shadow Mode Cost Model (1M requests/day, $5K baseline)

**Shadow infrastructure cost:**
- Production model: $5K/month
- Shadow model (duplicate, 50% traffic): $2.5K/month (run in parallel)
- Request mirroring/logging overhead: $500/month (send to shadow, log divergence)
- **Total: $8K/month** (60% overhead vs 100% blue-green)

**When shadow pays for itself:**
- Catch 1 bad model per month = prevent $10K loss
- Cost: $8K/month
- Net: saves $2K/month
- **ROI: 1.25x** (compared to 40x for canary, but safer validation)

### Deployment Strategy Decision Tree

| Scenario | Best Strategy | Reason | Time | Cost |
|----------|---------------|--------|------|------|
| **New ML model (high confidence)** | Canary | Confident from testing, fast rollout | 4h | 5% |
| **New ML model (low confidence)** | Shadow | Uncertain, validate on real traffic first | 3-7 days | 50% |
| **Payment system update (any)** | Blue-green | Zero-downtime is non-negotiable | 30 min | 100% |
| **Bug hotfix (low risk)** | Rolling | Fast, minimal risk | 15 min | 0% |
| **Risky architecture change** | Shadow first, then Canary | Validate fundamentals, then gradual rollout | 10 days | 55% total |

### Shadow vs Canary Comparison

**Use Shadow when:**
- Model quality uncertain (new algorithm, new data source, new domain)
- Willing to wait 3-7 days for validation
- Want 100% traffic exposure before production (most realistic)
- Cost not critical (50% overhead acceptable)
- Example: ML model for fraud detection (high risk of bad model)

**Use Canary when:**
- Model quality confident (tested extensively)
- Need faster rollout (4 hours acceptable)
- Cost matters (5% overhead vs 50%)
- Want to catch issues affecting real users (but limited exposure)
- Example: Improved recommendation model (incremental improvement)

**Use Blue-green when:**
- Downtime unacceptable (payment, auth systems)
- Zero-downtime requirement overrides cost
- Speed critical
- Example: Infrastructure update, critical dependency change

---

## Production Failure Scenarios

**Scenario 1: Shadow diverges from production**
- Shadow outputs different predictions (bug). Discover after 1 week.
- Fix: Compare live. Fix bug. Restart shadow.

**Scenario 2: Shadow blocks production latency**
- Shadow takes 500ms. Blocks user response.
- Fix: Async shadow (don't wait).

**Scenario 3: Stateful side effects in shadow**
- Shadow writes to database. Data corrupted.
- Fix: Shadow read-only. No writes.

---

## Implementation Guidance

**Wrong:** Shadow blocks production (waits for response).
**Right:** Async shadow (fire-and-forget).

**Wrong:** Shadow writes to DB (side effects).
**Right:** Shadow read-only (compare output only).

---

## Sophisticated Interview Q&A

**Q1: When use shadow over canary?**
A: When uncertain about model. Shadow validates on real traffic before any customer sees it.

**Q2: Shadow latency impact?**
A: Use async (don't block). Sample traffic (10%, not 100%).

**Q3: Agreement metric for promotion?**
A: >95% agreement between shadow and prod. If 95%, promote.

---

## Cost & Resource Analysis

50% compute overhead. ROI: prevent 1 bad rollout/month.

---

## Monitoring & Observability

Metrics: shadow_agreement, shadow_error_rate, shadow_latency. Alerts: agreement<95%, shadow_error>0.

## Best Practices
- **Async shadow:** shadow inference doesn't block production latency
- **Sample traffic:** shadow only 10% of traffic, not 100% (cost)
- **Agreement metrics:** track accuracy, ROC-AUC agreement, F1 correlation
- **Promotion threshold:** agree on <99% agreement → promote when hit
- **Gradual transition:** shadow for 1 week, then canary, then full

## Code Example
```python
import logging

class ShadowDeployment:
    def __init__(self, prod_model, shadow_model):
        self.prod = prod_model
        self.shadow = shadow_model
    
    def predict(self, features):
        # Production prediction (used)
        prod_pred = self.prod.predict(features)
        
        # Shadow prediction (logged, not used)
        try:
            shadow_pred = self.shadow.predict(features)
            # Log for comparison
            logging.info({
                "prod": prod_pred,
                "shadow": shadow_pred,
                "agreement": prod_pred == shadow_pred
            })
        except Exception as e:
            logging.error(f"Shadow failed: {e}")
        
        return prod_pred  # Always return production
```

## Interview Q&A

Q: Shadow mode overhead?
A: A: Run 2 models: production (used), new (logged). Double computation. Acceptable during testing, remove after switch.

Q: How long run shadow mode?
A: A: Until statistically confident new model is better. Depends on traffic and accuracy difference. 1-2 weeks typical. More if accuracy difference <1%.

Q: Compare: shadow model vs production on same data?
A: A: Yes. Shadow input = production input (same time, features). Compare predictions. If shadow is accurate 95% of time (matches production), good. If differs 20%, investigate why.

Q: New model performs worse in shadow. Acceptable?
A: A: If specific to certain cohorts (older users, rural areas), might be acceptable with explicit caveats. If worse overall, don't deploy.

Q: How do you log shadow predictions for analysis?
A: A: Write to logging system (BigQuery, Kafka). Include: timestamp, input, shadow prediction, production prediction, did they match? Post-analysis: compute accuracy, identify failure modes.

Q: Shadow model is slower than production?
A: A: Don't deploy (performance regression). Or optimize before deploy. Or accept slower for 5% of users (canary).

Q: Multiple shadow models simultaneously?
A: A: Yes. Shadow_v1, Shadow_v2 both run. Compare both to production. Pick best for production switch.

Q: Shadow traffic is different from production?
A: A: Possible. Shadow runs in off-peak (cheaper compute). Production peak traffic. Can stress test: run shadow during peak, observe (it's offline, no user impact).
## Interview Quick-Reference
| Metric | Target | Action |
|--------|--------|--------|
| Agreement | >98% | Consider promoting |
| Latency increase | <5ms | Accept |
| Errors | <0.1% | Investigate |

## Related Topics
- [Canary Deployment](12-canary-deployment.md)
- [Blue-Green](11-blue-green-deployment.md)

## Resources
- [Shadow Testing Best Practices](https://engineering.fb.com/2014/01/02/core-data/shadow-testing/)

