# Medical Diagnosis Assistant - System Architecture

```mermaid
graph TD
    subgraph PatientData["Patient Data Input"]
        PatientAPI["Patient API<br/>(Secure REST)"]
        EHRConnector["EHR Connector<br/>(HL7/FHIR)"]
    end

    subgraph Privacy["Privacy Layer"]
        DeIdentifier["De-identifier<br/>(PHI Stripper)"]
        AuditLogger["Audit Logger<br/>(HIPAA Compliant)"]
    end

    subgraph MLPipeline["ML Pipeline"]
        FeatureExtractor["Feature Extractor<br/>(Clinical NLP)"]
        DiagnosticModel["Diagnostic Model<br/>(Ensemble)"]
        ConfScorer["Confidence Scorer<br/>(Calibrated)"]
    end

    subgraph ReviewWorkflow["Clinician Review"]
        ReviewQueue["Review Queue<br/>(Low Confidence)"]
        ReviewInterface["Review Interface<br/>(Web UI)"]
    end

    subgraph Recommendations["Recommendation Layer"]
        RecommEngine["Recommendation Engine<br/>(Evidence-Based)"]
        OutputFormatter["Output Formatter<br/>(Clinical Report)"]
    end

    PatientAPI --> DeIdentifier
    EHRConnector --> DeIdentifier
    DeIdentifier --> AuditLogger
    DeIdentifier --> FeatureExtractor
    FeatureExtractor --> DiagnosticModel
    DiagnosticModel --> ConfScorer
    ConfScorer -->|Low confidence| ReviewQueue
    ConfScorer -->|High confidence| RecommEngine
    ReviewQueue --> ReviewInterface
    ReviewInterface --> RecommEngine
    RecommEngine --> OutputFormatter
```

**Infrastructure Components:**
- **EHR Connector**: FHIR/HL7 integration for structured patient record ingestion
- **De-identifier**: PHI removal pipeline (names, IDs, dates) to HIPAA Safe Harbor standard
- **Diagnostic Model**: Ensemble of clinical ML models (XGBoost + transformer) for differential diagnosis
- **Confidence Scorer**: Calibrated probability outputs; flags low-confidence cases for clinician review
- **Review Interface**: Clinician-facing UI for validating and overriding AI recommendations
- **Audit Logger**: Immutable log of all data accesses and AI decisions for HIPAA compliance
