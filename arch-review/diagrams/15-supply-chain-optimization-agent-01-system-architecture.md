## System Architecture (Infrastructure and Deployment)

```mermaid
graph TD
    subgraph Signals["Demand Signal Ingestion"]
        SalesData["Historical Sales\n(Data Warehouse)"]
        ExternalSignals["External Signals\n(Weather, Events)"]
        POSStream["POS Stream\n(Real-time Sales)"]
    end

    subgraph Forecasting["Forecasting Cluster"]
        DemandForecaster["Demand Forecaster\n(ARIMA/ML)"]
        SeasonalDecomposer["Seasonal Decomposer\n(STL)"]
        UncertaintyEstimator["Uncertainty Estimator\n(Conformal)"]
    end

    subgraph Optimization["Optimization Services"]
        InventoryOptimizer["Inventory Optimizer\n(LP/MIP)"]
        SupplierAgent["Supplier Agent\n(Multi-source)"]
        LogisticsAgent["Logistics Agent\n(Route Optimizer)"]
    end

    subgraph Actions["Action & Approval"]
        ActionRecommender["Action Recommender\n(LLM)"]
        HumanApproval["Human Approval\n(Dashboard)"]
        POSystem["PO System\n(ERP Integration)"]
    end

    subgraph DataLayer["Data Layer"]
        Warehouse["Data Warehouse\n(Snowflake)"]
        InventoryDB["Inventory DB\n(PostgreSQL)"]
        Redis["Cache\n(Redis)"]
    end

    subgraph Monitoring["Observability"]
        Prom["Prometheus"]
        AlertSystem["Alert System\n(PagerDuty)"]
        KPIDashboard["KPI Dashboard\n(Grafana)"]
    end

    SalesData --> DemandForecaster
    POSStream --> DemandForecaster
    ExternalSignals --> DemandForecaster
    DemandForecaster --> SeasonalDecomposer
    SeasonalDecomposer --> UncertaintyEstimator
    UncertaintyEstimator --> InventoryOptimizer
    InventoryOptimizer --> SupplierAgent
    InventoryOptimizer --> LogisticsAgent
    SupplierAgent --> ActionRecommender
    LogisticsAgent --> ActionRecommender
    ActionRecommender --> HumanApproval
    HumanApproval --> POSystem
    POSystem --> InventoryDB
    DemandForecaster --> Warehouse
    InventoryOptimizer --> Redis
    InventoryOptimizer --> Prom
    ActionRecommender --> AlertSystem
    KPIDashboard --> Warehouse
```

**Infrastructure Components:**
- **Compute**: Kubernetes cluster with forecasting and optimization pods
- **Storage**: Snowflake (data warehouse), PostgreSQL (inventory records), Redis (optimization cache)
- **ML**: ARIMA/ML demand forecasters, LP/MIP inventory optimizers, multi-source supplier agents
- **Integration**: ERP for purchase order submission, human approval dashboard
- **Monitoring**: Prometheus, PagerDuty alerts, Grafana KPI dashboards
