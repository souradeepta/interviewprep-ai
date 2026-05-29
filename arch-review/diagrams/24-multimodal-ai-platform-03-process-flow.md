## Process Flow (Multi-modal Request to Unified Understanding)

```mermaid
sequenceDiagram
    participant Client as API Client
    participant Router as Request Router
    participant Cache as Request Cache
    participant Vision as Vision Encoder
    participant Text as Text Encoder
    participant Audio as Audio Encoder
    participant Fusion as Fusion Layer
    participant LLM as LLM Interpreter
    participant Formatter as Response Formatter

    Client->>Router: POST /understand {image, text, audio}
    Router->>Router: Detect present modalities
    Router->>Cache: Check content hash
    alt Cache hit
        Cache-->>Client: Return cached unified result
    else Cache miss
        Router->>Vision: Dispatch image encoding (async)
        Router->>Text: Dispatch text encoding (async)
        Router->>Audio: Dispatch audio encoding (async)
        Vision->>Vision: ViT-B image encoding (100ms)
        Text->>Text: RoBERTa text encoding (50ms)
        Audio->>Audio: Whisper-Large transcription and encoding (200ms)
        Vision-->>Fusion: 768-d image embedding
        Text-->>Fusion: 768-d text embedding
        Audio-->>Fusion: 768-d audio embedding
        Fusion->>Fusion: Project all embeddings to 768-d (normalize)
        Fusion->>Fusion: Attention-weighted combination
        alt Modality conflict detected
            Fusion->>Fusion: Resolve by confidence weighting
        end
        Fusion->>LLM: Fused 768-d multimodal embedding
        LLM->>LLM: Generate unified interpretation (LLaMA-7B)
        LLM->>Formatter: Structured JSON output
        Formatter->>Cache: Store result (content hash key)
        Formatter-->>Client: Return unified understanding JSON
    end
```

**Key Decision Points:**
1. **Parallel Dispatch**: All three encoders run simultaneously (not sequentially) to minimize latency
2. **Cache by Content**: Hash of image+text+audio content as cache key for deduplication
3. **Missing Modality**: Zero-embedding substituted for absent modalities (graceful degradation)
4. **Conflict Resolution**: When modalities disagree, attention weights favor higher-confidence modality
5. **Fusion Level**: Mid-level fusion (after encoding, before interpretation) for best accuracy/latency trade-off

**Error Paths:**
- Audio encoder timeout (>400ms): return partial result (image+text only) immediately, deliver audio async
- GPU OOM: downscale image to 224x224 and retry
- LLM interpreter failure: return raw embeddings with similarity scores, no narrative

**Optimization Points:**
- Cache individual modality embeddings separately (reuse image embedding across calls with same image)
- Batch requests arriving within 50ms window (reduces GPU overhead)
- Quantize encoders to INT8 for 4x throughput at -1% accuracy
