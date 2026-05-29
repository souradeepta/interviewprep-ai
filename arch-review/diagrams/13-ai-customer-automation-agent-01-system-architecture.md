## System Architecture (Infrastructure and Deployment)

```mermaid
graph TD
    subgraph Ingestion["Message Ingestion Layer"]
        Email["Email Gateway"]
        Chat["Chat Interface"]
        API["REST API Gateway"]
    end

    subgraph Processing["Processing Cluster (Kubernetes)"]
        IntentClassifier["Intent Classifier\nPod x 10"]
        Router["Ticket Router\nPod x 5"]
        AgentCore["Agent Core\nPod x 8"]
        ResponseGen["Response Generator\nPod x 6"]
    end

    subgraph Tools["Tool Services"]
        CRM["CRM Connector\n(Salesforce)"]
        Orders["Order Management\n(Internal API)"]
        KB["Knowledge Base\n(Vector Search)"]
    end

    subgraph DataLayer["Data Layer"]
        PG["PostgreSQL\n(Tickets, History)"]
        Redis["Redis Cluster\n(Session Cache)"]
        Pinecone["Pinecone\n(KB Vectors)"]
        S3["S3 Bucket\n(Attachments)"]
    end

    subgraph Monitoring["Observability"]
        Prom["Prometheus\n(Metrics)"]
        ELK["ELK Stack\n(Logs)"]
        FeedbackDB["Feedback Store\n(PostgreSQL)"]
    end

    Email --> API
    Chat --> API
    API --> IntentClassifier
    IntentClassifier --> Router
    Router --> AgentCore
    AgentCore --> CRM
    AgentCore --> Orders
    AgentCore --> KB
    KB --> Pinecone
    AgentCore --> ResponseGen
    ResponseGen --> PG
    ResponseGen --> Redis
    IntentClassifier --> Prom
    AgentCore --> ELK
    ResponseGen --> FeedbackDB
```

**Infrastructure Components:**
- **Compute**: Kubernetes cluster (auto-scaling 5-50 pods based on queue depth)
- **Storage**: PostgreSQL (tickets, conversation history), Redis (session state), Pinecone (KB embeddings), S3 (attachments)
- **Tool Integrations**: CRM (Salesforce), Order Management, Knowledge Base
- **Monitoring**: Prometheus (metrics), ELK (logs), Feedback Store (quality tracking)
- **Load Balancing**: AWS ALB with health checks and auto-scaling policies
