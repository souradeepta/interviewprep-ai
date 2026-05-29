## Application Architecture (Components and Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        VideoUploadAPI["Video Upload API\n(REST, multipart)"]
        StreamIngestAPI["Stream Ingest API\n(RTMP)"]
        MetadataAPI["Metadata Query API\n(REST)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        VideoOrchestrator["Video Orchestrator\n(Adaptive sampling)"]
        ModerationOrchestrator["Moderation Orchestrator\n(Safety routing)"]
        CaptionOrchestrator["Caption Orchestrator\n(Multi-language)"]
    end

    subgraph FrameServices["Frame Processing Services"]
        FrameSampler["Frame Sampler\n(1fps to 30fps adaptive)"]
        MotionDetector["Motion Detector\n(OpenCV)"]
        ObjectDetector["Object Detector\n(YOLO v8)"]
        SceneClassifier["Scene Classifier\n(ResNet-50)"]
    end

    subgraph ContentServices["Content Understanding Services"]
        AudioExtractor["Audio Extractor\n(FFmpeg)"]
        ASRModel["ASR Model\n(Whisper)"]
        TranslationService["Translation Service\n(Multi-language)"]
        LLMNarrator["LLM Narrator\n(GPT-4V, keyframes only)"]
    end

    subgraph ModerationServices["Moderation Services"]
        SafetyClassifier["Safety Classifier\n(Ensemble)"]
        HumanReviewRouter["Human Review Router\n(Confidence-based)"]
        ActionEnforcer["Action Enforcer\n(Auto-remove above 0.99)"]
    end

    subgraph DataServices["Data Services"]
        MetadataWriter["Metadata Writer\n(PostgreSQL)"]
        CaptionWriter["Caption Writer\n(S3)"]
        TagIndexer["Tag Indexer\n(Elasticsearch)"]
    end

    VideoUploadAPI --> VideoOrchestrator
    StreamIngestAPI --> VideoOrchestrator
    VideoOrchestrator --> FrameSampler
    FrameSampler --> MotionDetector
    MotionDetector --> ObjectDetector
    ObjectDetector --> SceneClassifier
    ObjectDetector --> SafetyClassifier
    VideoOrchestrator --> AudioExtractor
    AudioExtractor --> ASRModel
    ASRModel --> TranslationService
    SceneClassifier --> LLMNarrator
    ASRModel --> LLMNarrator
    LLMNarrator --> MetadataWriter
    TranslationService --> CaptionWriter
    MetadataWriter --> TagIndexer
    SafetyClassifier --> HumanReviewRouter
    SafetyClassifier --> ActionEnforcer
    ModerationOrchestrator --> HumanReviewRouter
```

**Layer Breakdown:**
- **Presentation**: Video upload, live stream ingest, metadata query APIs
- **Orchestration**: Adaptive sampling coordinator, safety routing, multi-language caption pipeline
- **Frame Services**: Adaptive sampler (1-30fps), motion detection, YOLO object detection, scene classification
- **Content Services**: FFmpeg audio extraction, Whisper ASR, multi-language translation, selective LLM narration
- **Moderation Services**: Multi-model ensemble, confidence-based human routing, auto-enforcement
- **Data Services**: Metadata persistence, caption storage, Elasticsearch tag index
