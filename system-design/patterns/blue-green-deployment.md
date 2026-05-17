# Blue-Green Deployment

## TL;DR
Two identical environments (blue, green). Traffic on blue. Deploy to green, test, switch traffic to green. Instant rollback: switch back to blue.

## Core Intuition
Two parallel systems. Keep old one ready, switch to new, easy rollback.

## How It Works
```
Blue (current): production, getting all traffic
Green (new): deploy new model, run tests, no real traffic

When ready:
  Switch load balancer: blue → green
  Now green is live, blue is backup

If issues:
  Switch back: green → blue (instant)
```

## Key Properties / Trade-offs
- Rollback: instant (switch LB)
- Cost: run 2 environments (2x cost)
- Downtime: minimal (switch is <1s)
- Testing: can test on green before switching

## Common Mistakes / Gotchas
- **Double cost:** full redundancy is expensive
- **Data sync:** if data changes during switch, blue-green diverge. Minimize DB changes during switch.
- **Not testing green:** deploy without testing → issues after switch

## Interview Quick-Reference
**Blue-green?** Two envs, instant switch, instant rollback. More expensive, safer.

## Related Topics
- [Canary Deployment](canary-deployment.md) — gradual alternative
- [Production Readiness](production-readiness.md)

## Resources
- [Blue-Green Deployment](https://martinfowler.com/bliki/BlueGreenDeployment.html)
