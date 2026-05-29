## System Architecture (Infrastructure and Deployment)

```mermaid
graph TD
    subgraph Ingestion["Audio Input Layer"]
        MicrophoneStream["Microphone Stream\n(WebRTC)"]
        TelephonyGateway["Telephony Gateway\n(SIP/PSTN)"]
        AudioFileAPI["Audio File API\n(REST)"]
        VADService["Voice Activity Detector\n(Silence trim)"]
    end

    subgraph ASRCluster["ASR Cluster (GPU)"]
        WhisperASR["Whisper ASR\n(Streaming mode)"]
        PunctRestorer["Punct Restorer\n(BERT-based)"]
        ConfidenceScorer["Confidence Scorer\n(WER estimate)"]
    end

    subgraph NLUCluster["NLU Cluster"]
        IntentClassifier["Intent Classifier\n(DistilBERT)"]
        NERExtractor["NER Extractor\n(SpaCy + fine-tuned)"]
        EntityLinker["Entity Linker\n(KB lookup)"]
        SkillRouter["Skill Router\n(Rules + ML)"]
    end

    subgraph ResponseLayer["Response and TTS Layer"]
        LLMGenerator["LLM Generator\n(GPT-3.5, templates)"]
        TTSSynth["TTS Synthesizer\n(Neural, streaming)"]
        SessionManager["Session Manager\n(Context, Redis)"]
    end

    subgraph DataLayer["Data Layer"]
        TranscriptStore["Transcript Store\n(PostgreSQL)"]
        ResponseCache["Response Cache\n(Redis, common intents)"]
        FeedbackStore["Feedback Store\n(Kafka)"]
    end

    subgraph Monitoring["Observability"]
        Prom["Prometheus\n(WER, latency, turn)"]
        ELK["ELK Stack\n(Conversation logs)"]
        QualityDashboard["Quality Dashboard\n(Grafana)"]
    end

    MicrophoneStream --> VADService
    TelephonyGateway --> VADService
    AudioFileAPI --> VADService
    VADService --> WhisperASR
    WhisperASR --> PunctRestorer
    PunctRestorer --> ConfidenceScorer
    ConfidenceScorer --> IntentClassifier
    IntentClassifier --> NERExtractor
    NERExtractor --> EntityLinker
    EntityLinker --> SkillRouter
    SkillRouter --> LLMGenerator
    SkillRouter --> ResponseCache
    LLMGenerator --> TTSSynth
    SessionManager --> LLMGenerator
    SessionManager --> IntentClassifier
    TTSSynth --> TranscriptStore
    TranscriptStore --> FeedbackStore
    WhisperASR --> Prom
    LLMGenerator --> ELK
```

**Infrastructure Components:**
- **Compute**: GPU cluster for Whisper ASR (200ms streaming) and NLU inference (50ms)
- **Storage**: Redis (session context, response cache), PostgreSQL (transcript store)
- **Pipeline**: Parallel ASR and NLU processing, streaming TTS for sub-500ms response
- **Monitoring**: WER tracking, end-to-end latency per turn, conversation quality dashboards
