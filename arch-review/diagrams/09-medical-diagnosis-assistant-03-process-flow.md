# Medical Diagnosis Assistant - Process Flow

```mermaid
sequenceDiagram
    participant Clinician as Clinician
    participant System as Diagnosis System
    participant PHI as PHI De-identifier
    participant Features as Feature Pipeline
    participant Model as Diagnostic Model
    participant Review as Clinician Review
    participant Recommender as Recommendation Engine

    Clinician->>System: Submit patient case
    System->>PHI: De-identify patient data
    PHI-->>System: De-identified case data
    System->>Features: Extract clinical features
    Features-->>System: Feature vector
    System->>Model: Run diagnostic prediction
    Model-->>System: Diagnosis + confidence score
    alt Confidence below threshold (less than 0.75)
        System->>Review: Flag for clinician review
        Review-->>Clinician: Review request notification
        Clinician->>Review: Provide clinical judgment
        Review-->>System: Validated diagnosis
    else Confidence above threshold
        System->>Recommender: Format recommendation
    end
    Recommender-->>System: Evidence-based recommendation
    System-->>Clinician: Clinical report with recommendations
    System->>System: Write HIPAA audit log entry
```

**Key Decision Points:**
1. **PHI De-identification**: All patient data stripped of identifiers before ML processing
2. **Confidence Threshold**: Predictions below 0.75 confidence routed to clinician review queue
3. **Clinician Override**: Clinician can override AI diagnosis with their clinical judgment
4. **Audit Logging**: Every access and AI decision logged for HIPAA compliance

**Optimization Points:**
- PHI de-identification runs synchronously to ensure no raw PHI reaches ML components
- Feature extraction caches results for repeated case submissions (same patient, same visit)
- Confidence calibration uses Platt scaling to ensure confidence scores are meaningful probabilities
