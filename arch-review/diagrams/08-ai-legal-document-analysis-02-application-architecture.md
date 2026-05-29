# AI Legal Document Analysis - Application Architecture

```mermaid
graph TD
    subgraph UploadLayer["Upload Layer"]
        UploadSvc["Upload Service<br/>(REST)"]
        FileValidator["File Validator<br/>(MIME, Size)"]
    end

    subgraph ParseLayer["Parse Layer"]
        OCREngine["OCR Engine<br/>(Text Extraction)"]
        DocStructurer["Document Structurer<br/>(Section Parser)"]
    end

    subgraph AnalysisLayer["Analysis Layer"]
        SectionRouter["Section Router<br/>(Dispatch)"]
        AnalysisWorkers["Analysis Workers<br/>(Parallel)"]
        RiskAggregator["Risk Aggregator<br/>(Score Merger)"]
    end

    subgraph ReportLayer["Report Layer"]
        ReportBuilder["Report Builder<br/>(Template)"]
        ExportSvc["Export Service<br/>(PDF/JSON)"]
    end

    subgraph DataLayer["Data Layer"]
        ResultDB["Result DB<br/>(PostgreSQL)"]
        AuditLog["Audit Logger<br/>(S3 Write)"]
    end

    UploadSvc --> FileValidator
    FileValidator --> OCREngine
    OCREngine --> DocStructurer
    DocStructurer --> SectionRouter
    SectionRouter --> AnalysisWorkers
    AnalysisWorkers --> RiskAggregator
    RiskAggregator --> ReportBuilder
    ReportBuilder --> ExportSvc
    ReportBuilder --> ResultDB
    AnalysisWorkers --> AuditLog
```

**Layer Breakdown:**
- **Upload Layer**: File validation (MIME type, size limit, malware scan) before processing
- **Parse Layer**: OCR text extraction, then section/clause boundary detection
- **Analysis Layer**: Parallel section analysis workers, risk score aggregation across sections
- **Report Layer**: Template-based report generation with PDF and JSON export options
- **Data Layer**: Result persistence and immutable audit log of all AI analysis decisions
