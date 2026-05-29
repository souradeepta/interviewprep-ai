## System Architecture (Infrastructure & Deployment)

```mermaid
graph TD
    subgraph DataSources["Security Data Sources"]
        SIEM["SIEM Platform\n(Splunk/Sentinel)"]
        NetFlow["Network Flow\n(NetFlow/IPFIX)"]
        EDR["EDR Platform\n(CrowdStrike)"]
    end

    subgraph DetectionLayer["Detection Layer"]
        ThreatDetector["Threat Detector\n(ML + Rules)"]
        TriageAgent["Triage Agent\n(LLM Classifier)"]
    end

    subgraph InvestigationCluster["Investigation Cluster (Kubernetes)"]
        LogAgent["Log Analysis Agent\nPod x 4"]
        NetworkAgent["Network Agent\nPod x 4"]
        EndpointAgent["Endpoint Agent\nPod x 4"]
    end

    subgraph ResponseLayer["Response Orchestration"]
        ResponseOrch["Response Orchestrator\n(Playbook Engine)"]
        Remediation["Automated Remediation\n(Isolate, Block, Patch)"]
        SOCNotify["SOC Notification\n(PagerDuty, Slack)"]
    end

    subgraph DataLayer["Data & Audit"]
        ThreatDB["Threat Intel DB\n(STIX/TAXII)"]
        IncidentLog["Incident Log\n(Elasticsearch)"]
        AuditTrail["Audit Trail\n(S3 Immutable)"]
    end

    SIEM --> ThreatDetector
    NetFlow --> ThreatDetector
    EDR --> ThreatDetector

    ThreatDetector --> TriageAgent
    TriageAgent --> LogAgent
    TriageAgent --> NetworkAgent
    TriageAgent --> EndpointAgent

    LogAgent --> ResponseOrch
    NetworkAgent --> ResponseOrch
    EndpointAgent --> ResponseOrch

    ResponseOrch --> Remediation
    ResponseOrch --> SOCNotify

    ThreatDB --> TriageAgent
    ResponseOrch --> IncidentLog
    Remediation --> AuditTrail
```

**Infrastructure Components:**
- **Data Sources**: SIEM (Splunk/Sentinel), network flow collectors, EDR telemetry
- **Detection**: ML + rule-based threat detector feeding an LLM triage classifier
- **Investigation Cluster**: Kubernetes pods for log, network, and endpoint agents running in parallel
- **Response**: Playbook-driven orchestrator executing automated remediation and SOC notifications
- **Data**: STIX/TAXII threat intel, Elasticsearch incident log, immutable S3 audit trail
