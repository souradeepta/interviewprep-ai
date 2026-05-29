# AI Legal Document Analysis - System Architecture

```mermaid
graph TD
    subgraph Ingestion["Document Ingestion"]
        DocUpload["Document Upload<br/>(REST API)"]
        FileStore["File Store<br/>(S3)"]
    end

    subgraph Extraction["Extraction Pipeline"]
        OCREngine["OCR Engine<br/>(Textract)"]
        SectionExtractor["Section Extractor<br/>(NLP Parser)"]
    end

    subgraph Analysis["Analysis Layer"]
        WorkerPool["Analysis Workers<br/>(Parallel per Section)"]
        LLMAnalyzer["LLM Analyzer<br/>(GPT-4)"]
    end

    subgraph Scoring["Risk and Reporting"]
        RiskScorer["Risk Scorer<br/>(Weighted Model)"]
        ReportBuilder["Report Builder<br/>(Template Engine)"]
    end

    subgraph ClientPortal["Client Portal"]
        Portal["Client Portal<br/>(Web App)"]
        NotifyService["Notify Service<br/>(Email)"]
    end

    subgraph Storage["Storage and Audit"]
        ResultDB["Result DB<br/>(PostgreSQL)"]
        AuditLog["Audit Log<br/>(Immutable S3)"]
    end

    DocUpload --> FileStore
    FileStore --> OCREngine
    OCREngine --> SectionExtractor
    SectionExtractor --> WorkerPool
    WorkerPool --> LLMAnalyzer
    LLMAnalyzer --> RiskScorer
    RiskScorer --> ReportBuilder
    ReportBuilder --> Portal
    ReportBuilder --> NotifyService
    ReportBuilder --> ResultDB
    LLMAnalyzer --> AuditLog
```

**Infrastructure Components:**
- **OCR Engine**: AWS Textract for PDF/scanned document text extraction
- **Section Extractor**: NLP-based document structure parser (clauses, sections, definitions)
- **Analysis Workers**: Parallel worker pool processing independent document sections
- **LLM Analyzer**: GPT-4 with legal-domain prompts for clause interpretation and risk flagging
- **Risk Scorer**: Weighted model combining LLM outputs into document-level risk score
- **Audit Log**: Immutable log of all LLM inputs/outputs for professional liability compliance
