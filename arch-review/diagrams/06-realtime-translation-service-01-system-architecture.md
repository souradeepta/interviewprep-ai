# Realtime Translation Service - System Architecture

```mermaid
graph TD
    subgraph Inputs["Input Layer"]
        AudioInput["Audio Stream<br/>(WebRTC)"]
        TextInput["Text Input<br/>(REST/WS)"]
    end

    subgraph Processing["Processing Pipeline"]
        LangDetector["Language Detector<br/>(FastText)"]
        ASR["ASR Engine<br/>(Whisper)"]
        TranslationEngine["Translation Engine<br/>(NMT Model)"]
        PostProc["Post-Processor<br/>(Grammar Fix)"]
    end

    subgraph CacheLayer["Cache Layer"]
        TransCache["Translation Cache<br/>(Redis)"]
    end

    subgraph Quality["Quality and Monitoring"]
        QualityMonitor["Quality Monitor<br/>(BLEU Score)"]
        Prometheus["Prometheus<br/>(Latency Metrics)"]
    end

    subgraph Delivery["Delivery Layer"]
        OutputGateway["Output Gateway<br/>(WebSocket)"]
    end

    AudioInput --> ASR
    TextInput --> LangDetector
    ASR --> LangDetector
    LangDetector --> TransCache
    TransCache -->|Hit| OutputGateway
    TransCache -->|Miss| TranslationEngine
    TranslationEngine --> PostProc
    PostProc --> TransCache
    PostProc --> QualityMonitor
    PostProc --> OutputGateway
    QualityMonitor --> Prometheus
```

**Infrastructure Components:**
- **ASR Engine**: Whisper model for audio-to-text transcription (real-time streaming chunks)
- **Language Detector**: FastText language ID model, sub-5ms latency
- **Translation Engine**: Neural Machine Translation (Helsinki-NLP or NLLB-200) covering 100+ languages
- **Post-Processor**: Grammar correction and fluency improvement on translated text
- **Cache Layer**: Redis for repeated phrase translations (high hit rate on common utterances)
- **Quality Monitor**: BLEU/COMET score tracking for translation quality regression alerts
