# Blue-Green Deployment

## TL;DR
Two identical environments (Blue=current, Green=new). Deploy new model to Green, test, then switch traffic. Instant rollback: switch back to Blue. Zero downtime.

## Core Intuition
Safe deployment: have two full systems. Test on the new one. Users still use old one. Convinced? Switch all users to new. Detect issue? Switch back. No downtime.

## How It Works

1. Blue (production): serving all traffic with model v1.0.0
2. Green (staging): deploy model v2.0.0, test thoroughly
3. Validation: run synthetic tests, smoke tests, canary with 1% traffic
4. Switch: update load balancer config to route to Green
5. Rollback (if needed): switch back to Blue (config change, instant)

| Phase | Blue | Green | Traffic |
|-------|------|-------|---------|
| Initial | v1.0.0 running | Idle | 100% → Blue |
| Deployment | v1.0.0 running | v2.0.0 running | 100% → Blue |
| Validation | v1.0.0 running | v2.0.0 tested | 100% → Blue |
| Switch | v1.0.0 idle | v2.0.0 running | 100% → Green |
| Rollback | v1.0.0 running | v2.0.0 idle | 100% → Blue |

## Key Properties / Trade-offs
- Cost: 2x infrastructure (both Blue and Green running)
- Speed: instant switch (config change)
- Safety: full rollback, no downtime
- Complexity: manage two environments

## Common Mistakes / Gotchas
- Forgetting to warm cache on Green (cold start latency)
- Different infrastructure on Blue/Green (breaks reproducibility)
- Not testing on Green thoroughly before switch
- Rollback not working (Blue degraded during switch window)

## Best Practices
- **Pre-warm Green:** run synthetic traffic to populate caches
- **Identical infrastructure:** Blue and Green must be clones
- **Test on Green:** run full test suite before switch
- **Gradual switch:** 10% → 50% → 100% rather than instant
- **Keep Blue warm:** don't turn off old environment for 30 min after switch

## Code Example
```python
import subprocess

class BlueGreenDeployment:
    def deploy(self, model_version):
        # 1. Deploy to Green
        print("Deploying to Green...")
        subprocess.run(["kubectl", "apply", "-f", "green-deployment.yaml"])
        
        # 2. Wait for Green readiness
        subprocess.run(["kubectl", "wait", "--for=condition=ready", "pod", "-l", "app=green"])
        
        # 3. Test Green
        print("Running tests on Green...")
        test_result = subprocess.run(["python", "test_model.py", "--endpoint=green.local"])
        
        if test_result.returncode != 0:
            print("Tests failed, not switching")
            return False
        
        # 4. Switch traffic to Green
        print("Switching traffic to Green...")
        subprocess.run(["kubectl", "patch", "service", "model-api", 
                       "-p", '{"spec":{"selector":{"version":"green"}}}'])
        
        # 5. Monitor
        print("Monitoring Green for 5 minutes...")
        time.sleep(300)
        
        print("Deployment complete!")
        return True
```

## Interview Q&A
**Q: Infrastructure cost doubles (2x servers). Worth it?**
A: For critical models: yes. For experimental: maybe not. Compromise: use staging environment (smaller), test there, then deploy to prod. Cost: 1.5x instead of 2x.

**Q: Deployment takes 10 minutes, traffic increases during that window. Issue?**
A: Blue might be overwhelmed if Green is still warming up (request rejected). Solution: pre-warm Green during off-peak, or use canary (small % traffic to Green) before full switch.

## Interview Quick-Reference
| Phase | Blue | Green | Traffic |
|-------|------|-------|---------|
| Initial | v1 (prod) | - | 100% |
| Deploy | v1 (prod) | v2 (test) | 100% |
| Switch | - | v2 (prod) | 100% |

## Related Topics
- [Canary Deployment](12-canary-deployment.md)
- [Model Serving](05-model-serving.md)

## Resources
- [Blue-Green Deployment Guide](https://martinfowler.com/bliki/BlueGreenDeployment.html)
