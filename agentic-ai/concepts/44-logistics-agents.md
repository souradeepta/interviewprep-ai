# Logistics Agents
**TL;DR:** A logistics agent recommends or executes constrained shipment actions while preserving feasibility, freshness, and an auditable human override path.

## Prerequisites
Optimization basics, event-driven systems, and agent tool-use boundaries.

## Learning Outcomes
- Model shipment assignment with capacity and time-window constraints.
- Design replanning around stale events and idempotent updates.
- Define escalation rules for infeasible or high-impact actions.

## Core Intuition
A dispatcher does not merely choose the shortest route: the truck has capacity, the dock has operating hours, the shipment may be regulated, and a missed delivery can cost more than extra mileage. A logistics agent must reason over a feasible plan and the cost of changing it. The plan is advisory until constraints and authority checks pass.

## How It Works
Represent orders, vehicles, depots, and carrier options as typed records with explicit units and timestamps. A planner proposes assignments that satisfy hard constraints such as capacity, service window, and restricted destination. Soft objectives—distance, cost, lateness risk, emissions—are optimized only inside the feasible set. On each new event, compute the affected part of the plan and compare a replan against the committed baseline. Before calling a carrier or warehouse API, validate authorization, deduplicate the command, and persist an idempotency key.
Events are imperfect observations. A missing scan does not prove that a shipment stopped moving, and two systems may report different states because their clocks or update delays differ. Store event time and ingestion time separately, assign source reliability, and expose stale state to the planner. A replan should carry the assumptions that produced it, the affected shipment identifiers, and the expected cost and service change. If those assumptions expire before execution, recompute rather than applying the obsolete route.
Replanning should preserve commitments that cannot safely change, including regulated handling requirements, dock appointments, and already dispatched vehicles. Model those as hard constraints with explicit ownership, not as large penalty weights that an optimizer may violate when costs change. Human override is also an event in the plan history: record who changed the route, what information they used, and whether the new plan was later feasible. Replay this history during incident review to distinguish bad recommendations from missing or delayed data.
Evaluate on replayed event sequences, not isolated static routes. Measure feasible-plan rate, on-time probability, cost per delivered shipment, disruption recovery time, and the percentage of proposed changes accepted by an operator. Report tail lateness and performance for rare constraints such as cold-chain handling or customs holds. A low route cost is not success if it creates infeasible schedules or excess replanning churn. Keep synthetic exercises clearly separate from carrier performance claims.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Reoptimize globally | Can find system-wide savings | Large solve time and plan churn |
| Replan affected shipments | Fast and stable | Can miss cross-route gains |
| Automate low-impact changes | Low operator load | Requires trustworthy constraints |
| Human approval for exceptions | Safer on ambiguous cases | Slower resolution |

## Implementation
This dependency-light example models a bounded policy decision. It is an executable illustration of the control boundary; it does not call an LLM or establish production quality.

```python
def feasible_load(load, capacity, window_ok):
    return load <= capacity and window_ok

assert feasible_load(8, 10, True)
assert not feasible_load(12, 10, True)
```

## Failure Modes
- Stale tracking event: compare event and ingestion times and query the authoritative source before changing a commitment.
- Infeasible route after disruption: run hard-constraint validation after every solve and keep a safe fallback plan.
- Duplicate carrier instruction: use a durable idempotency key and reconcile command state before retrying.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| How do you replan after a carrier outage? | Identify affected shipments, recompute feasible alternatives, quantify service and cost changes, then automate only within preapproved limits. Design follow-up: what state makes the plan stale? |
| How do you handle conflicting tracking events? | Retain both source and timestamps, apply an explicit source policy, and avoid treating missing data as a confirmed event. Debug follow-up: how would you find a clock skew? |
| What should be automated? | Reversible low-impact assignments under validated constraints; regulated, infeasible, or high-value exceptions go to a human. |

## Related Concepts
See [Real-time agents](50-real-time-agents.md), [agent state management](57-agent-state-management.md), [agent monitoring](30-agent-monitoring.md).

## References
- [developers.google.com](https://developers.google.com/optimization/routing)
- [www.gs1.org](https://www.gs1.org/standards/epcis)

## Further Topic Detail
## Core Intuition
A dispatcher does not merely choose the shortest route: the truck has capacity, the dock has operating hours, the shipment may be regulated, and a missed delivery can cost more than extra mileage. A logistics agent must reason over a feasible plan and the cost of changing it. The plan is advisory until constraints and authority checks pass.


## How It Works
Represent orders, vehicles, depots, and carrier options as typed records with explicit units and timestamps. A planner proposes assignments that satisfy hard constraints such as capacity, service window, and restricted destination. Soft objectives—distance, cost, lateness risk, emissions—are optimized only inside the feasible set. On each new event, compute the affected part of the plan and compare a replan against the committed baseline. Before calling a carrier or warehouse API, validate authorization, deduplicate the command, and persist an idempotency key.
Events are imperfect observations. A missing scan does not prove that a shipment stopped moving, and two systems may report different states because their clocks or update delays differ. Store event time and ingestion time separately, assign source reliability, and expose stale state to the planner. A replan should carry the assumptions that produced it, the affected shipment identifiers, and the expected cost and service change. If those assumptions expire before execution, recompute rather than applying the obsolete route.
Replanning should preserve commitments that cannot safely change, including regulated handling requirements, dock appointments, and already dispatched vehicles. Model those as hard constraints with explicit ownership, not as large penalty weights that an optimizer may violate when costs change. Human override is also an event in the plan history: record who changed the route, what information they used, and whether the new plan was later feasible. Replay this history during incident review to distinguish bad recommendations from missing or delayed data.
Evaluate on replayed event sequences, not isolated static routes. Measure feasible-plan rate, on-time probability, cost per delivered shipment, disruption recovery time, and the percentage of proposed changes accepted by an operator. Report tail lateness and performance for rare constraints such as cold-chain handling or customs holds. A low route cost is not success if it creates infeasible schedules or excess replanning churn. Keep synthetic exercises clearly separate from carrier performance claims.


## Detailed Explanation

Logistics agents optimize operations: route planning, inventory management, shipment tracking. Mechanisms: (1) constraint satisfaction (time windows, capacity), (2) optimization (minimize cost/time), (3) real-time adaptation. Advantages: cost reduction, faster delivery, better utilization. Challenges: complexity (hundreds of constraints), uncertainty (traffic, delays), dynamics (orders arrive continuously). Best for: delivery routing, warehouse optimization, inventory planning.



## Interview Q&A

**Q: How do you design a logistics agent to handle real-time supply chain disruptions?**
A: Build disruption detection into the data pipeline: monitor carrier APIs, weather feeds, and customs status in real-time. When disruption is detected, trigger an impact analysis: which shipments are affected, what are the alternative routes/carriers, what are the cost/delay trade-offs. Implement automated rerouting for low-complexity situations (one carrier down, one backup available) and escalate to human decision-makers for complex multi-variable trade-offs. Maintain pre-computed contingency plans for known high-risk routes.

**Q: What data sources does a logistics agent need and how do you handle data quality issues?**
A: Key sources: carrier tracking APIs, warehouse management systems, ERP/order management, customs and compliance databases, weather and traffic feeds, carrier capacity and rate data. Quality issues: missing tracking events, delayed updates, conflicting statuses across systems. Handle with: data validation layers (flag impossible transitions like tracking backward), reconciliation logic (resolve conflicts by source reliability ranking), staleness detection (alert when tracking hasn't updated in expected window), and anomaly detection for outlier delays.

**Q: How should a logistics agent handle situations where it cannot complete a delivery commitment?**
A: Detect the issue as early as possible (before the expected delivery window, not after). Proactively communicate to the customer with a new ETA based on best available data. Provide options where feasible (expedited shipping at cost, alternative delivery location). Trigger internal escalation for high-value shipments or repeat failures. Log the root cause for carrier performance tracking. Never wait for the customer to ask—proactive communication with context reduces inbound support contacts by 60-70%.

**Q: What are the key metrics for a logistics agent to monitor and optimize?**
A: On-time delivery rate (by carrier, route, product type), exception rate (shipments requiring intervention), time-to-resolution for exceptions, cost per shipment deviation (expediting costs), and customer-reported delivery satisfaction. Track trends: is on-time rate improving or degrading? Which carriers have the highest exception rates? Are certain origin-destination pairs systematically problematic? Use these metrics to optimize carrier selection, routing rules, and contingency protocols.

**Q: How do you handle customs and trade compliance in an automated logistics agent?**
A: Integrate with authoritative trade compliance databases (HTS codes, denied parties lists, ECCN classifications). Before any international shipment, verify: product classification, export control requirements, import duties and taxes, customs documentation completeness. Implement hard stops for denied parties or embargoed destinations—these are compliance requirements, not suggestions. Flag ambiguous classifications for human review rather than auto-processing. Maintain audit logs of all compliance checks and decisions for regulatory reporting.

**Q: What is the risk of automation bias in logistics agents and how do you mitigate it?**
A: Automation bias: operators trust and follow agent recommendations without sufficient scrutiny, even when the agent is wrong. In logistics, this can mean accepting a suboptimal routing decision, missing a compliance issue, or mishandling a high-value shipment exception. Mitigate with: requiring acknowledgment rather than auto-acceptance for high-impact decisions, displaying the agent's confidence and key factors behind recommendations, tracking human override rates (very low = automation bias risk), and regularly testing with adversarial edge cases where the agent's default recommendation is wrong.
