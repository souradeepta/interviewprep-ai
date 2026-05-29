## System Architecture (Infrastructure and Deployment)

```mermaid
graph TD
    subgraph Ingestion["Video Ingestion Layer"]
        VideoUpload["Video Upload\n(REST API)"]
        LiveStream["Live Stream\n(RTMP/HLS)"]
        StorageBucket["Video Storage\n(S3)"]
    end

    subgraph FrameProcessing["Frame Processing Cluster (GPU)"]
        FrameExtractor["Frame Extractor\n(FFmpeg, 1-30fps)"]
        MotionDetector["Motion Detector\n(OpenCV)"]
        ObjectDetector["Object Detector\n(YOLO v8, GPU)"]
        SceneClassifier["Scene Classifier\n(ResNet-50)"]
    end

    subgraph ContentServices["Content Services"]
        AudioExtractor["Audio Extractor\n(FFmpeg)"]
        ASRService["ASR Service\n(Whisper, GPU)"]
        LLMNarrator["LLM Narrator\n(GPT-4 Vision)"]
    end

    subgraph ModerationLayer["Moderation Layer"]
        SafetyClassifier["Safety Classifier\n(Multi-model ensemble)"]
        ReviewQueue["Human Review Queue\n(Flagged videos)"]
        AlertSystem["Alert System\n(High risk auto-remove)"]
    end

    subgraph DataLayer["Data Layer"]
        MetadataStore["Metadata Store\n(PostgreSQL)"]
        CaptionStore["Caption Store\n(S3)"]
        TagIndex["Tag Index\n(Elasticsearch)"]
    end

    subgraph Monitoring["Observability"]
        Prom["Prometheus\n(Throughput, latency)"]
        ELK["ELK Stack\n(Moderation logs)"]
        CostTracker["Cost Tracker\n(Per-video)"]
    end

    VideoUpload --> StorageBucket
    LiveStream --> FrameExtractor
    StorageBucket --> FrameExtractor
    FrameExtractor --> MotionDetector
    MotionDetector --> ObjectDetector
    ObjectDetector --> SceneClassifier
    ObjectDetector --> SafetyClassifier
    SceneClassifier --> LLMNarrator
    StorageBucket --> AudioExtractor
    AudioExtractor --> ASRService
    ASRService --> LLMNarrator
    LLMNarrator --> MetadataStore
    LLMNarrator --> CaptionStore
    MetadataStore --> TagIndex
    SafetyClassifier --> ReviewQueue
    SafetyClassifier --> AlertSystem
    ObjectDetector --> Prom
    LLMNarrator --> CostTracker
```

**Infrastructure Components:**
- **Compute**: GPU cluster for YOLO v8 object detection, ResNet scene classification, Whisper ASR
- **Storage**: S3 (videos, captions), PostgreSQL (metadata), Elasticsearch (tag index)
- **Moderation**: Multi-model ensemble for safety classification, human review queue, auto-remove for high-confidence violations
- **Cost Optimization**: Adaptive sampling (1fps baseline, scale up on motion detection)
