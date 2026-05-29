## System Architecture (Infrastructure & Deployment)

```mermaid
graph TD
    subgraph Input["Input Layer"]
        BriefAPI["Content Brief API\n(REST)"]
        TemplateStore["Template Store\n(S3)"]
    end

    subgraph AgentCluster["Agent Cluster (Kubernetes)"]
        ResearchAgent["Research Agent\nPod x 3"]
        OutlineAgent["Outline Agent\nPod x 2"]
        WritingAgents["Writing Agents\nPod x 8 (parallel sections)"]
        EditorAgent["Editor Agent\nPod x 2"]
        SEOAgent["SEO Agent\nPod x 2"]
    end

    subgraph LLMBackend["LLM Backend"]
        LLMRouter["LLM Router\n(Model Selector)"]
        LLMPool["LLM Inference Pool\n(GPT-4 / Claude)"]
        WebSearch["Web Search Tool\n(Bing API)"]
    end

    subgraph Output["Output & Distribution"]
        Publisher["Publishing Agent\n(CMS Connector)"]
        ContentDB["Content Store\n(PostgreSQL)"]
        CDNPush["CDN Push\n(CloudFront)"]
    end

    subgraph Monitoring["Monitoring"]
        QualityScorer["Quality Scorer\n(Flesch, SEO score)"]
        MetricsSvc["Prometheus"]
    end

    BriefAPI --> ResearchAgent
    TemplateStore --> OutlineAgent
    ResearchAgent --> OutlineAgent
    OutlineAgent --> WritingAgents
    WritingAgents --> EditorAgent
    EditorAgent --> SEOAgent
    SEOAgent --> Publisher

    ResearchAgent --> WebSearch
    WritingAgents --> LLMRouter
    EditorAgent --> LLMRouter
    LLMRouter --> LLMPool

    Publisher --> ContentDB
    Publisher --> CDNPush
    EditorAgent --> QualityScorer
    QualityScorer --> MetricsSvc
```

**Infrastructure Components:**
- **Input**: REST API accepting content briefs, S3 template store for style guides
- **Agent Cluster**: Kubernetes pods for each pipeline stage; Writing Agents parallelized by section
- **LLM Backend**: Model router selecting GPT-4 or Claude based on task; Bing API for web research
- **Output**: CMS-connected publishing agent pushing to PostgreSQL and CloudFront CDN
- **Quality**: Readability and SEO scoring tracked via Prometheus
