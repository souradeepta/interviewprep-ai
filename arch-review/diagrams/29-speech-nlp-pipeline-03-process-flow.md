## Process Flow (Audio Input to Voice Response)

```mermaid
sequenceDiagram
    participant User as User (Voice)
    participant VAD as VAD Filter
    participant Whisper as Whisper ASR
    participant NLU as Intent Classifier
    participant NER as NER Extractor
    participant Router as Skill Router
    participant Cache as Response Cache
    participant LLM as LLM Generator
    participant TTS as TTS Synthesizer
    participant Session as Session Manager

    User->>VAD: Speak (audio stream)
    VAD->>VAD: Detect speech start, trim silence
    VAD->>Whisper: Stream audio chunks
    Whisper->>Whisper: Partial transcripts at 100ms intervals
    Whisper->>NLU: Partial transcript (early routing)
    NLU->>NLU: Classify intent (DistilBERT, 50ms)
    alt High confidence (above 0.90) on partial
        NLU->>Router: Early route signal
        Router->>Cache: Check for cached response
    end
    Whisper->>Whisper: Complete transcript at 200ms
    Whisper->>NER: Final transcript
    NER->>NER: Extract entities (person, place, date, product)
    NER->>Router: Intent + entities + confidence
    Router->>Session: Inject conversation context
    Session-->>Router: Prior turns and slot values
    alt Intent confidence above 0.90
        alt Cache hit
            Cache-->>TTS: Cached response text
        else Cache miss
            Router->>LLM: Intent + entities + context
            LLM->>LLM: Generate response (GPT-3.5, 300ms)
            LLM-->>TTS: Response text
        end
    else Confidence below 0.90
        Router->>LLM: Clarification request
        LLM-->>TTS: Clarification question
    end
    TTS->>TTS: Neural synthesis (streaming, 150ms first word)
    TTS-->>User: Audio response stream
    TTS->>Session: Update conversation history
```

**Key Decision Points:**
1. **Early Routing**: NLU starts on partial transcripts (100ms) for faster response initiation
2. **VAD Trimming**: Voice activity detection prevents processing silence and noise
3. **Session Context**: Prior conversation turns injected into LLM prompt for multi-turn coherence
4. **Confidence Gate**: Below 0.90 triggers clarification question rather than wrong response
5. **Streaming TTS**: First audio word delivered at 150ms without waiting for full text generation

**Error Paths:**
- Whisper confidence below 0.80: ask user to repeat (not cascade to wrong intent)
- LLM timeout (above 800ms): fall back to template response for known high-frequency intents
- TTS synthesis failure: deliver text response as fallback, retry audio async

**Optimization Points:**
- Cache template responses for top-50 most common intents (zero LLM cost)
- Parallelize NER and entity linking with partial ASR output
- Use DistilBERT (50ms) not BERT (200ms) for intent classification in the critical path
