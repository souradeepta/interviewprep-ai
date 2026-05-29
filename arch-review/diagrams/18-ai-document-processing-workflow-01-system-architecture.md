## System Architecture (Infrastructure & Deployment)

```mermaid
graph TD
    subgraph Intake["Document Intake Layer"]
        DocIntake["Document Intake\n(S3 Upload / API)"]
        FormatRouter["Format Router\n(PDF, DOCX, Image)"]
    end

    subgraph Processing["Processing Cluster (Kubernetes)"]
        OCR["OCR Engine\n(Tesseract / AWS Textract)"]
        Classifier["Document Classifier\n(Fine-tuned BERT)"]
        ExtractionAgents["Extraction Agents\n(Named Entity + Layout)"]
    end

    subgraph Validation["Validation Layer"]
        Validator["Schema Validator\n(Pydantic)"]
        ConfidenceFilter["Confidence Filter\n(Threshold Check)"]
        HumanReview["Human Review Queue\n(Low Confidence)"]
    end

    subgraph Output["Output & Integration"]
        OutputStore["Structured Output\n(PostgreSQL + S3 JSON)"]
        Downstream["Downstream Systems\n(ERP, CRM, Webhook)"]
        NotifyService["Notification Service\n(Email / Webhook)"]
    end

    subgraph Monitoring["Monitoring"]
        MetricsSvc["Prometheus\n(Throughput, Accuracy)"]
        AlertMgr["Alert Manager"]
    end

    DocIntake --> FormatRouter
    FormatRouter --> OCR
    OCR --> Classifier
    Classifier --> ExtractionAgents
    ExtractionAgents --> Validator
    Validator --> ConfidenceFilter
    ConfidenceFilter -->|High Confidence| OutputStore
    ConfidenceFilter -->|Low Confidence| HumanReview
    HumanReview --> OutputStore
    OutputStore --> Downstream
    OutputStore --> NotifyService
    ExtractionAgents --> MetricsSvc
    MetricsSvc --> AlertMgr
```

**Infrastructure Components:**
- **Intake**: S3 upload trigger or REST API accepting PDF, DOCX, and image formats
- **Processing**: Kubernetes cluster running OCR, BERT-based classifier, and extraction agents
- **Validation**: Pydantic schema validation with confidence-based human review queue
- **Output**: Structured JSON stored in PostgreSQL and S3, forwarded to downstream systems
- **Monitoring**: Prometheus tracking throughput, OCR accuracy, and extraction confidence
