## Application Architecture (Components and Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        DashboardAPI["Dashboard API\n(REST)"]
        WebhookReceiver["Webhook Receiver\n(POS events)"]
        ApprovalUI["Approval UI\n(Web Dashboard)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        PlanningOrchestrator["Planning Orchestrator\n(Daily batch)"]
        EventProcessor["Event Processor\n(Stream handler)"]
        ApprovalWorkflow["Approval Workflow\n(Human-in-loop)"]
    end

    subgraph ForecastingServices["Forecasting Services"]
        DemandForecaster["Demand Forecaster\n(ARIMA + XGBoost)"]
        SeasonalModel["Seasonal Model\n(STL decomposition)"]
        ExternalSignalProcessor["External Signal\nProcessor (weather)"]
    end

    subgraph OptimizationServices["Optimization Services"]
        InventoryOptimizer["Inventory Optimizer\n(LP solver)"]
        SafetyStockCalc["Safety Stock Calc\n(Service level)"]
        SupplierSelector["Supplier Selector\n(Multi-criteria)"]
    end

    subgraph ActionServices["Action Services"]
        POGenerator["PO Generator\n(ERP integration)"]
        ActionRecommender["Action Recommender\n(LLM narratives)"]
        NotificationService["Notification Service\n(Email/Slack)"]
    end

    subgraph DataServices["Data Services"]
        WarehouseClient["Warehouse Client\n(Snowflake)"]
        InventoryDBClient["Inventory DB\n(PostgreSQL ORM)"]
        CacheClient["Cache Client\n(Redis)"]
    end

    WebhookReceiver --> EventProcessor
    DashboardAPI --> PlanningOrchestrator
    EventProcessor --> DemandForecaster
    PlanningOrchestrator --> DemandForecaster
    DemandForecaster --> SeasonalModel
    ExternalSignalProcessor --> DemandForecaster
    SeasonalModel --> InventoryOptimizer
    InventoryOptimizer --> SafetyStockCalc
    SafetyStockCalc --> SupplierSelector
    SupplierSelector --> ActionRecommender
    ActionRecommender --> ApprovalWorkflow
    ApprovalWorkflow --> POGenerator
    POGenerator --> NotificationService
    DemandForecaster --> WarehouseClient
    InventoryOptimizer --> InventoryDBClient
    ActionRecommender --> CacheClient
    ApprovalUI --> ApprovalWorkflow
```

**Layer Breakdown:**
- **Presentation**: Dashboard API, POS webhook receiver, approval UI
- **Orchestration**: Daily batch planner, real-time event processor, human approval workflow
- **Forecasting Services**: ARIMA/ML demand forecasting, seasonal decomposition, external signals
- **Optimization Services**: LP-based inventory optimizer, safety stock calculator, multi-criteria supplier selection
- **Action Services**: PO generation, LLM-narrated recommendations, notifications
- **Data Services**: Snowflake warehouse, inventory database, optimization cache
