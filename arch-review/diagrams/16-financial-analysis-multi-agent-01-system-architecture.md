## System Architecture (Infrastructure & Deployment)

```mermaid
graph TD
    subgraph Ingestion["Data Ingestion Layer"]
        MarketFeed["Market Data Feeds\n(Bloomberg, Reuters)"]
        AltData["Alternative Data\n(News, Social)"]
        MsgBus["Message Bus\n(Kafka)"]
    end

    subgraph AgentCluster["Agent Compute Cluster (Kubernetes)"]
        FundAgent["Fundamental Agent\nPod x 4"]
        TechAgent["Technical Agent\nPod x 4"]
        SentAgent["Sentiment Agent\nPod x 4"]
        MacroAgent["Macro Agent\nPod x 2"]
    end

    subgraph Orchestration["Orchestration Layer"]
        Aggregator["Result Aggregator\n(Weighted Merge)"]
        RiskAssessor["Risk Assessor\n(VaR, Drawdown)"]
        PortfolioOpt["Portfolio Optimizer\n(MPT + Constraints)"]
    end

    subgraph Output["Output & Storage"]
        TradeRec["Trade Recommendations\n(API)"]
        ResultStore["Result Store\n(PostgreSQL)"]
        AuditLog["Audit Log\n(S3)"]
    end

    subgraph Monitoring["Monitoring"]
        MetricsDB["Prometheus"]
        Dashboard["Grafana Dashboard"]
    end

    MarketFeed --> MsgBus
    AltData --> MsgBus
    MsgBus --> FundAgent
    MsgBus --> TechAgent
    MsgBus --> SentAgent
    MsgBus --> MacroAgent

    FundAgent --> Aggregator
    TechAgent --> Aggregator
    SentAgent --> Aggregator
    MacroAgent --> Aggregator

    Aggregator --> RiskAssessor
    RiskAssessor --> PortfolioOpt
    PortfolioOpt --> TradeRec
    PortfolioOpt --> ResultStore
    ResultStore --> AuditLog

    FundAgent --> MetricsDB
    Aggregator --> MetricsDB
    MetricsDB --> Dashboard
```

**Infrastructure Components:**
- **Data Ingestion**: Kafka message bus consuming Bloomberg, Reuters, and alternative data feeds
- **Agent Cluster**: Kubernetes-managed pods for Fundamental, Technical, Sentiment, and Macro agents
- **Orchestration**: Weighted aggregation, risk assessment (VaR), and portfolio optimization (MPT)
- **Storage**: PostgreSQL for results, S3 for immutable audit logs
- **Monitoring**: Prometheus metrics with Grafana dashboards for agent health and latency
