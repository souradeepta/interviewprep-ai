## Application Architecture (Components & Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        UploadAPI["Upload API\n(FastAPI)"]
        StatusAPI["Status API\n(Job Tracker)"]
        WebhookEmitter["Webhook Emitter\n(Async Notify)"]
    end

    subgraph PipelineLayer["Pipeline Layer"]
        JobQueue["Job Queue\n(Celery + Redis)"]
        OCRProcessor["OCR Processor\n(Textract Wrapper)"]
        DocClassifier["Document Classifier\n(LayoutLM)"]
    end

    subgraph ExtractionLayer["Extraction Layer"]
        InvoiceExtractor["Invoice Extractor\nAgent"]
        ContractExtractor["Contract Extractor\nAgent"]
        FormExtractor["Form Extractor\nAgent"]
        TableParser["Table Parser\n(camelot)"]
    end

    subgraph ValidationLayer["Validation Layer"]
        SchemaValidator["Schema Validator\n(Pydantic Models)"]
        BusinessRules["Business Rules\n(Domain Logic)"]
        AuditLogger["Audit Logger\n(All Decisions)"]
    end

    subgraph StorageLayer["Storage Layer"]
        DocumentDB["Document Store\n(MongoDB)"]
        StructuredDB["Structured Store\n(PostgreSQL)"]
        RawStore["Raw Archive\n(S3)"]
    end

    UploadAPI --> JobQueue
    JobQueue --> OCRProcessor
    OCRProcessor --> DocClassifier
    DocClassifier --> InvoiceExtractor
    DocClassifier --> ContractExtractor
    DocClassifier --> FormExtractor
    DocClassifier --> TableParser

    InvoiceExtractor --> SchemaValidator
    ContractExtractor --> SchemaValidator
    FormExtractor --> SchemaValidator
    TableParser --> SchemaValidator

    SchemaValidator --> BusinessRules
    BusinessRules --> AuditLogger
    BusinessRules --> StructuredDB
    AuditLogger --> DocumentDB
    OCRProcessor --> RawStore
    StructuredDB --> WebhookEmitter
    StatusAPI --> JobQueue
```

**Layer Breakdown:**
- **Presentation**: REST API for uploads, status polling, and async webhook notifications
- **Pipeline**: Celery job queue driving OCR and LayoutLM-based document classification
- **Extraction**: Specialized agents per document type (invoice, contract, form) plus table parser
- **Validation**: Pydantic schema + domain business rules with full decision audit logging
- **Storage**: MongoDB for document metadata, PostgreSQL for structured fields, S3 for raw archives
