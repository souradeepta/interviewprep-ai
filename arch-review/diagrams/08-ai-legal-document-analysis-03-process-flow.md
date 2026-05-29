# AI Legal Document Analysis - Process Flow

```mermaid
sequenceDiagram
    participant U as User
    participant Portal as Client Portal
    participant OCR as OCR Engine
    participant Structurer as Document Structurer
    participant Workers as Analysis Workers
    participant Aggregator as Risk Aggregator
    participant Report as Report Builder

    U->>Portal: Upload legal document
    Portal->>OCR: Extract text from document
    alt OCR failed
        OCR-->>Portal: Extraction error
        Portal-->>U: Upload error, retry
    else OCR succeeded
        OCR-->>Structurer: Raw extracted text
        Structurer-->>Workers: Parsed sections (parallel dispatch)
        par Section 1 analysis
            Workers->>Workers: LLM analyze section 1
        and Section 2 analysis
            Workers->>Workers: LLM analyze section 2
        and Section N analysis
            Workers->>Workers: LLM analyze section N
        end
        Workers-->>Aggregator: All section risk scores
        Aggregator-->>Report: Aggregated risk profile
        Report-->>Portal: Completed analysis report
        Portal-->>U: Report ready (email + portal)
    end
```

**Key Decision Points:**
1. **OCR Quality Check**: Low confidence OCR score triggers manual review request before processing
2. **Parallel Section Analysis**: Independent sections analyzed concurrently to minimize total latency
3. **Risk Aggregation**: Section-level scores combined with weights (e.g., indemnity clauses weighted higher)
4. **Report Delivery**: Both portal notification and email delivery on completion

**Optimization Points:**
- Parallel section analysis reduces wall-clock time from O(N sections) to ~O(1) with enough workers
- LLM prompt caching for common clause patterns reduces cost on similar documents
- OCR result cached to avoid re-processing if same document uploaded again
