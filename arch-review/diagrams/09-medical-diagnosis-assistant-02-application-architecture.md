# Medical Diagnosis Assistant - Application Architecture

```mermaid
graph TD
    subgraph InputLayer["Input Layer"]
        PatientAPI["Patient API<br/>(REST)"]
        PHIStripper["PHI Stripper<br/>(De-identification)"]
    end

    subgraph FeatureLayer["Feature Engineering"]
        FeaturePipeline["Feature Pipeline<br/>(Clinical NLP)"]
        LabNormalizer["Lab Value Normalizer<br/>(Unit Convert)"]
    end

    subgraph ModelLayer["Model Layer"]
        MLModel["ML Model<br/>(Ensemble Classifier)"]
        ThresholdChecker["Threshold Checker<br/>(Confidence Gate)"]
    end

    subgraph ReviewLayer["Review Layer"]
        ReviewInterface["Review Interface<br/>(Clinician UI)"]
        OverrideHandler["Override Handler<br/>(Clinician Input)"]
    end

    subgraph OutputLayer["Output Layer"]
        Recommender["Recommender<br/>(Evidence Links)"]
        ReportFormatter["Report Formatter<br/>(Clinical PDF)"]
        AuditLogger["Audit Logger<br/>(Immutable Log)"]
    end

    PatientAPI --> PHIStripper
    PHIStripper --> FeaturePipeline
    FeaturePipeline --> LabNormalizer
    LabNormalizer --> MLModel
    MLModel --> ThresholdChecker
    ThresholdChecker -->|Above threshold| Recommender
    ThresholdChecker -->|Below threshold| ReviewInterface
    ReviewInterface --> OverrideHandler
    OverrideHandler --> Recommender
    Recommender --> ReportFormatter
    ReportFormatter --> AuditLogger
```

**Layer Breakdown:**
- **Input Layer**: Secure API with immediate PHI de-identification before any downstream processing
- **Feature Engineering**: Clinical NLP for symptom extraction, lab value normalization to standard units
- **Model Layer**: Ensemble classifier with calibrated confidence scores and threshold gating
- **Review Layer**: Clinician-facing review UI with override capability for low-confidence predictions
- **Output Layer**: Evidence-linked recommendations, clinical PDF reports, HIPAA audit trail
