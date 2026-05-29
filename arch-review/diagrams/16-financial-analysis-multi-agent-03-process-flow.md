## Process Flow (Analysis Request to Trade Recommendation)

```mermaid
sequenceDiagram
    participant U as User
    participant API as REST API
    participant Q as Task Queue
    participant FA as Fundamental Agent
    participant TA as Technical Agent
    participant SA as Sentiment Agent
    participant AG as Aggregator
    participant RA as Risk Assessor
    participant PO as Portfolio Optimizer

    U->>API: Submit analysis request (ticker, universe)
    API->>Q: Enqueue parallel agent tasks
    Q->>FA: Dispatch fundamental analysis
    Q->>TA: Dispatch technical analysis
    Q->>SA: Dispatch sentiment analysis
    FA-->>AG: Fundamental signals (P/E, EV/EBITDA)
    TA-->>AG: Technical signals (RSI, MACD)
    SA-->>AG: Sentiment scores (FinBERT)
    AG->>RA: Combined weighted signal view
    RA->>RA: Compute VaR and Sharpe ratio
    RA->>PO: Risk-adjusted signal + constraints
    PO->>PO: Solve optimization (cvxpy)
    PO-->>API: Optimal weights + trade recommendations
    API-->>U: Ranked trade recommendations with rationale
```

**Key Decision Points:**
1. **Parallel Dispatch**: All three agents run concurrently to minimize latency
2. **Confidence Weighting**: Aggregator weights agent signals by historical accuracy per asset class
3. **Risk Filtering**: Positions exceeding VaR or drawdown thresholds are reduced before optimization
4. **Constraint Enforcement**: Sector concentration and position size limits applied before output

**Error Paths:**
- Agent timeout (>5s) - use last cached signal with staleness flag
- Missing data for a ticker - skip that agent, note in rationale
- Optimization infeasible - relax constraints incrementally until feasible

**Optimization Points:**
- Feature caching avoids redundant indicator computation across agent runs
- Batch ticker processing amortizes model loading overhead
- Async agent dispatch with shared result collector reduces wall-clock latency
