## Application Architecture (Components & Layers)

```mermaid
graph TD
    subgraph Ingestion["Event Ingestion Layer"]
        SIEMConnector["SIEM Connector\n(Kafka Consumer)"]
        AlertNormalizer["Alert Normalizer\n(CEF/LEEF Parser)"]
    end

    subgraph Classification["Classification Layer"]
        ThreatClassifier["Threat Classifier\n(XGBoost + LLM)"]
        SeverityScorer["Severity Scorer\n(CVSS-like)"]
        TriageRouter["Triage Router\n(Priority Queue)"]
    end

    subgraph AgentLayer["Investigation Agent Layer"]
        LogAnalyzer["Log Analysis Agent\n(Pattern Matching)"]
        NetworkAnalyzer["Network Analyzer\n(Graph Traversal)"]
        EndpointAnalyzer["Endpoint Analyzer\n(IOC Lookup)"]
        ThreatEnricher["Threat Enricher\n(MITRE ATT&CK)"]
    end

    subgraph PlaybookEngine["Playbook Engine"]
        PlaybookSelector["Playbook Selector\n(Rule Matcher)"]
        ActionExecutor["Action Executor\n(API Calls)"]
        RollbackManager["Rollback Manager\n(Undo Actions)"]
    end

    subgraph Notification["Notification & Reporting"]
        SOCAlert["SOC Alert\n(PagerDuty)"]
        TicketCreator["Ticket Creator\n(ServiceNow)"]
        ReportGen["Report Generator\n(PDF/HTML)"]
    end

    SIEMConnector --> AlertNormalizer
    AlertNormalizer --> ThreatClassifier
    ThreatClassifier --> SeverityScorer
    SeverityScorer --> TriageRouter

    TriageRouter --> LogAnalyzer
    TriageRouter --> NetworkAnalyzer
    TriageRouter --> EndpointAnalyzer

    LogAnalyzer --> ThreatEnricher
    NetworkAnalyzer --> ThreatEnricher
    EndpointAnalyzer --> ThreatEnricher

    ThreatEnricher --> PlaybookSelector
    PlaybookSelector --> ActionExecutor
    ActionExecutor --> RollbackManager

    ActionExecutor --> SOCAlert
    ActionExecutor --> TicketCreator
    ActionExecutor --> ReportGen
```

**Layer Breakdown:**
- **Ingestion**: Kafka consumer normalizing heterogeneous SIEM formats (CEF, LEEF)
- **Classification**: XGBoost + LLM hybrid classifier with CVSS-like severity scoring
- **Agent Layer**: Parallel investigation agents enriched with MITRE ATT&CK framework
- **Playbook Engine**: Rule-based playbook selection driving automated API-based actions with rollback
- **Notification**: SOC alerting, ticket creation, and incident reporting
