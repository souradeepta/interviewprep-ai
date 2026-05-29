## Application Architecture (Components & Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        RestAPI["REST API\n(FastAPI)"]
        WebhookSink["Webhook Sink\n(Trade Events)"]
    end

    subgraph AgentLayer["Agent Layer"]
        FundAgent["Fundamental Agent\n(P/E, EV/EBITDA)"]
        TechAgent["Technical Agent\n(RSI, MACD, BB)"]
        SentAgent["Sentiment Agent\n(FinBERT)"]
        MacroAgent["Macro Agent\n(Rates, FX, CPI)"]
    end

    subgraph CoordLayer["Coordination Layer"]
        TaskQueue["Task Queue\n(Celery + Redis)"]
        Aggregator["Result Aggregator\n(Confidence Weighting)"]
        RiskEngine["Risk Engine\n(VaR, Sharpe)"]
    end

    subgraph OptLayer["Optimization Layer"]
        PortfolioOpt["Portfolio Optimizer\n(cvxpy)"]
        ConstraintChecker["Constraint Checker\n(Sector Limits)"]
        TradeSignalGen["Trade Signal\nGenerator"]
    end

    subgraph DataLayer["Data Layer"]
        TimescaleDB["TimescaleDB\n(Time-Series Prices)"]
        VectorStore["Vector Store\n(News Embeddings)"]
        FeatureCache["Feature Cache\n(Redis)"]
    end

    RestAPI --> TaskQueue
    WebhookSink --> TaskQueue

    TaskQueue --> FundAgent
    TaskQueue --> TechAgent
    TaskQueue --> SentAgent
    TaskQueue --> MacroAgent

    FundAgent --> Aggregator
    TechAgent --> Aggregator
    SentAgent --> Aggregator
    MacroAgent --> Aggregator

    Aggregator --> RiskEngine
    RiskEngine --> PortfolioOpt
    PortfolioOpt --> ConstraintChecker
    ConstraintChecker --> TradeSignalGen

    FundAgent --> TimescaleDB
    TechAgent --> FeatureCache
    SentAgent --> VectorStore
    TradeSignalGen --> RestAPI
```

**Layer Breakdown:**
- **Presentation**: REST API for recommendation queries and webhook sink for real-time trade events
- **Agent Layer**: Specialized agents covering fundamental, technical, sentiment, and macro signals
- **Coordination**: Celery task queue dispatching to agents; confidence-weighted aggregation
- **Optimization**: cvxpy-based portfolio optimization with sector/position constraints
- **Data Layer**: TimescaleDB for price history, vector store for news, Redis for feature caching
