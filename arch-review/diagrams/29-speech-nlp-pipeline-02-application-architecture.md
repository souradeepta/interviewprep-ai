## Application Architecture (Components and Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        AudioStreamAPI["Audio Stream API\n(WebRTC/REST)"]
        TextQueryAPI["Text Query API\n(REST fallback)"]
        ResponseAPI["Response API\n(Text + TTS audio)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        VoiceOrchestrator["Voice Orchestrator\n(Pipeline coordinator)"]
        SessionOrchestrator["Session Orchestrator\n(Turn management)"]
        FallbackOrchestrator["Fallback Orchestrator\n(Low confidence paths)"]
    end

    subgraph ASRServices["ASR Services"]
        VADFilter["VAD Filter\n(Silence detection)"]
        WhisperStreamSvc["Whisper Stream Svc\n(Partial transcripts)"]
        PunctRestorationSvc["Punct Restoration Svc\n(BERT)"]
        ASRConfidenceEval["ASR Confidence Eval\n(WER estimate)"]
    end

    subgraph NLUServices["NLU Services"]
        IntentClassifierSvc["Intent Classifier Svc\n(DistilBERT, 50ms)"]
        NERExtractorSvc["NER Extractor Svc\n(SpaCy)"]
        EntityLinkerSvc["Entity Linker Svc\n(KB lookup)"]
        SkillRouterSvc["Skill Router Svc\n(Rules + confidence)"]
    end

    subgraph ResponseServices["Response Generation Services"]
        TemplateEngine["Template Engine\n(Known intents, fast)"]
        LLMGeneratorSvc["LLM Generator Svc\n(GPT-3.5, complex)"]
        TTSSynthSvc["TTS Synth Svc\n(Neural, streaming)"]
    end

    subgraph DataServices["Data Services"]
        SessionCache["Session Cache\n(Redis, context)"]
        ResponseCacheClient["Response Cache Client\n(Redis, common intents)"]
        TranscriptWriter["Transcript Writer\n(PostgreSQL)"]
    end

    AudioStreamAPI --> VADFilter
    VADFilter --> WhisperStreamSvc
    WhisperStreamSvc --> PunctRestorationSvc
    PunctRestorationSvc --> ASRConfidenceEval
    ASRConfidenceEval --> IntentClassifierSvc
    IntentClassifierSvc --> NERExtractorSvc
    NERExtractorSvc --> EntityLinkerSvc
    EntityLinkerSvc --> SkillRouterSvc
    SkillRouterSvc --> ResponseCacheClient
    ResponseCacheClient --> TemplateEngine
    SkillRouterSvc --> LLMGeneratorSvc
    SessionCache --> LLMGeneratorSvc
    TemplateEngine --> TTSSynthSvc
    LLMGeneratorSvc --> TTSSynthSvc
    TTSSynthSvc --> ResponseAPI
    TTSSynthSvc --> TranscriptWriter
    SessionOrchestrator --> SessionCache
```

**Layer Breakdown:**
- **Presentation**: Streaming audio API, text fallback API, combined text+audio response API
- **Orchestration**: Voice pipeline coordinator, multi-turn session management, low-confidence fallback
- **ASR Services**: VAD filter, Whisper streaming transcription, punctuation restoration, confidence evaluation
- **NLU Services**: Intent classification (50ms), named entity recognition, knowledge base entity linking, skill routing
- **Response Services**: Fast template engine for known intents, LLM for complex queries, neural TTS synthesis
- **Data Services**: Redis session context, response cache for common intents, transcript persistence
