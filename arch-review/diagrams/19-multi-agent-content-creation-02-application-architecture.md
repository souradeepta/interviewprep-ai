## Application Architecture (Components & Layers)

```mermaid
graph TB
    subgraph Presentation["Presentation Layer"]
        BriefAPI["Brief Intake API\n(FastAPI REST)"]
        StatusWS["Status WebSocket\n(Progress Updates)"]
        BriefValidator["Brief Validator\n(Pydantic Schema)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        PipelineOrch["Pipeline Orchestrator\n(DAG Executor)"]
        AgentRegistry["Agent Registry\n(Capability Routing)"]
        StateManager["State Manager\n(Redis Session)"]
    end

    subgraph AgentServices["Agent Services"]
        ResearchSvc["Research Service\n(Web + DB Tools)"]
        OutlineSvc["Outline Service\n(Structure Generator)"]
        WritingSvc["Writing Pool\n(Section Workers x8)"]
        EditorSvc["Editor Service\n(Coherence + Grammar)"]
        SEOSvc["SEO Service\n(Keyword Optimizer)"]
    end

    subgraph ToolLayer["Tool Layer"]
        WebSearchTool["Web Search Tool\n(Bing API Client)"]
        FactCheckTool["Fact Check Tool\n(Source Verifier)"]
        ImageGenTool["Image Gen Tool\n(DALL-E / Stable Diffusion)"]
    end

    subgraph LLMLayer["LLM Services"]
        LLMRouter["LLM Router\n(Model Selector)"]
        GPT4Client["GPT-4 Client\n(Complex Tasks)"]
        ClaudeClient["Claude Client\n(Long Documents)"]
        EmbedSvc["Embedding Service\n(Semantic Dedup)"]
    end

    subgraph Storage["Storage"]
        ContentDB["Content Store\n(PostgreSQL)"]
        DraftCache["Draft Cache\n(Redis)"]
        AssetStore["Asset Store\n(S3)"]
    end

    subgraph Publishing["Publishing"]
        CMSConnector["CMS Connector\n(WordPress / Contentful)"]
        QualityScorer["Quality Scorer\n(Flesch + SEO)"]
    end

    BriefAPI --> BriefValidator
    BriefValidator --> PipelineOrch
    StatusWS --> StateManager

    PipelineOrch --> AgentRegistry
    AgentRegistry --> ResearchSvc
    AgentRegistry --> OutlineSvc
    AgentRegistry --> WritingSvc
    AgentRegistry --> EditorSvc
    AgentRegistry --> SEOSvc

    ResearchSvc --> WebSearchTool
    ResearchSvc --> FactCheckTool
    WritingSvc --> ImageGenTool

    ResearchSvc --> LLMRouter
    OutlineSvc --> LLMRouter
    WritingSvc --> LLMRouter
    EditorSvc --> LLMRouter
    LLMRouter --> GPT4Client
    LLMRouter --> ClaudeClient
    LLMRouter --> EmbedSvc

    PipelineOrch --> StateManager
    StateManager --> DraftCache
    EditorSvc --> ContentDB
    AssetStore --> CMSConnector
    EditorSvc --> QualityScorer
    QualityScorer --> CMSConnector
```

**Layer Breakdown:**
- **Presentation**: REST API for brief intake, WebSocket for real-time pipeline status
- **Orchestration**: DAG-based pipeline with agent registry for capability routing
- **Agent Services**: Specialized agents per content creation stage
- **Tool Layer**: External tools (web search, fact check, image generation)
- **LLM Services**: Model routing between GPT-4 and Claude by task type
- **Storage**: PostgreSQL for content, Redis for drafts, S3 for media assets
- **Publishing**: Quality scoring before CMS connector push
