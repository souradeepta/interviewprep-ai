## Process Flow (Video Upload to Metadata and Captions)

```mermaid
sequenceDiagram
    participant Uploader as Content Creator
    participant API as Upload API
    participant Sampler as Frame Sampler
    participant Motion as Motion Detector
    participant YOLO as Object Detector
    participant Safety as Safety Classifier
    participant ASR as Whisper ASR
    participant LLM as LLM Narrator
    participant Store as Metadata Store

    Uploader->>API: Upload video file
    API->>Sampler: Extract frames at 1fps baseline
    Sampler->>Motion: Check motion between frames
    alt High motion detected
        Motion->>Sampler: Increase to 5fps in region
        Sampler->>YOLO: Submit dense frame batch
    else Low motion (static scene)
        Motion->>YOLO: Submit 1fps sample
    end
    YOLO->>YOLO: Detect objects and actions per frame
    YOLO->>Safety: Pass detected objects and scenes
    Safety->>Safety: Multi-model ensemble scoring
    alt Safety score above 0.99 (very high risk)
        Safety-->>Uploader: Video removed, policy violation
    else Safety score 0.90 to 0.99 (high risk)
        Safety->>Store: Flag for human review
        Store-->>Uploader: Video held for review
    else Safety score below 0.90 (safe)
        YOLO->>LLM: Keyframe objects and scene context
        API->>ASR: Extract and transcribe audio (parallel)
        ASR-->>LLM: Transcript text
        LLM->>LLM: Generate scene descriptions (keyframes only)
        LLM->>Store: Write metadata and tags
        ASR->>Store: Write captions (multi-language async)
        Store-->>Uploader: Video processed, metadata ready
    end
```

**Key Decision Points:**
1. **Adaptive Sampling**: Motion detector upgrades from 1fps to 5fps on activity bursts
2. **Safety Triage**: Three tiers (auto-remove above 0.99, human review 0.90-0.99, allow below 0.90)
3. **LLM Selectivity**: Narrator only processes scene-change keyframes, not every frame
4. **Parallel Audio**: ASR runs concurrently with frame processing to avoid sequential latency
5. **Multi-language Captions**: Generated asynchronously after primary English captions complete

**Error Paths:**
- GPU memory exceeded: reduce frame rate, drop to 1fps for remainder
- ASR failure: deliver video without captions, retry async
- LLM narration timeout: deliver basic object tags without descriptive narration

**Optimization Points:**
- Skip narration for static scenes (no scene change in last 30 seconds)
- Cache common object vocabularies per video category
- Batch translate captions to top-5 languages off-peak
