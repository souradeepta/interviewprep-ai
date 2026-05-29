## Process Flow (Demand Signal to Purchase Order)

```mermaid
sequenceDiagram
    participant POS as POS System
    participant Forecaster as Demand Forecaster
    participant Seasonal as Seasonal Model
    participant Optimizer as Inventory Optimizer
    participant Supplier as Supplier Agent
    participant LLM as Action Recommender
    participant Human as Supply Chain Manager
    participant ERP as ERP System

    POS->>Forecaster: Real-time sales signal
    Forecaster->>Seasonal: Decompose trend and seasonal
    Seasonal-->>Forecaster: Adjusted demand curve
    Forecaster->>Forecaster: Generate 30-day forecast per SKU
    Forecaster->>Optimizer: Forecast + uncertainty bounds
    Optimizer->>Optimizer: Compute optimal reorder quantities
    Optimizer->>Optimizer: Calculate safety stock by service level
    Optimizer->>Supplier: Candidate SKUs needing reorder
    Supplier->>Supplier: Score suppliers (lead time, cost, risk)
    Supplier->>LLM: Reorder plan + supplier scores
    LLM->>LLM: Generate human-readable recommendation
    LLM->>Human: Show recommended POs with rationale
    alt Human approves
        Human->>ERP: Approve purchase orders
        ERP-->>Supplier: PO submitted to supplier
        ERP-->>Forecaster: Record order for feedback
    else Human modifies
        Human->>LLM: Provide override rationale
        LLM->>Optimizer: Update constraints
        Optimizer->>Supplier: Revised reorder plan
        Supplier->>ERP: Revised POs
    else Human rejects
        Human->>LLM: Rejection reason
        LLM->>Forecaster: Log feedback for model improvement
    end
```

**Key Decision Points:**
1. **Seasonal Decomposition**: STL decomposition prevents over-ordering before off-season
2. **Uncertainty Bounds**: Safety stock formula uses forecast standard deviation
3. **Supplier Scoring**: Multi-criteria (lead time, cost, disruption risk) for resilience
4. **Human Approval Gate**: All POs above threshold require human sign-off
5. **Override Feedback**: Manager overrides feed back into the optimization model

**Error Paths:**
- Supplier disruption signal: escalate to dual-source recommendation
- Demand spike detected (3x normal): alert manager immediately, hold auto-order
- ERP submission failure: queue for retry, alert supply chain manager

**Optimization Points:**
- Cache daily optimization results per SKU (avoid redundant LP runs)
- Batch low-velocity SKUs into weekly optimization instead of daily
- Pre-compute seasonal adjustments for the next 90 days during off-hours
