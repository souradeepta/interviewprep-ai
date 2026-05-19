# Shadow Mode Deployment

## TL;DR
Run new model in parallel with production model. Log predictions from both, compare accuracy offline. Only when confident (>99% agreement) promote new model to production.

## Core Intuition
Test new model on real traffic without affecting users. New model runs silently, results discarded. Compare with production model. When better, promote.

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
**Q: Shadow model takes 500ms, production takes 50ms. Issue?**
A: Shadow inference is async (doesn't block). Run in background thread, log results. Production latency unaffected.

**Q: Agreement metric: 98% of shadow predictions match production. Promote?**
A: Maybe not. Check: (1) are mismatches systematic (certain user segment)? (2) does shadow have higher accuracy on validation set? (3) which model is actually better? 98% agreement might mean both are wrong together.

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
