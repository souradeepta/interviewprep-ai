## Process Flow (Request Lifecycle)

```mermaid
sequenceDiagram
    participant C as Client
    participant API as Brief API
    participant Orch as Orchestrator
    participant Res as Research Agent
    participant Out as Outline Agent
    participant Wrt as Writing Pool
    participant Ed as Editor Agent
    participant SEO as SEO Agent
    participant CMS as CMS Publisher

    C->>API: POST /brief {topic, style, length}
    API->>API: Validate brief schema
    API->>Orch: Create pipeline job (job_id)
    API-->>C: 202 Accepted {job_id}

    Orch->>Res: research(topic, depth=3)
    Res->>Res: Web search x5 queries
    Res->>Res: Fact verification
    Res-->>Orch: research_report {sources, facts}

    Orch->>Out: outline(research_report, style)
    Out->>Out: Generate section structure
    Out-->>Orch: outline {sections: [intro, body x5, conclusion]}

    Orch->>Wrt: write_sections(outline, research_report)
    note over Wrt: 8 workers write sections in parallel
    Wrt-->>Orch: drafts {section_1..N}

    Orch->>Ed: edit(drafts, style_guide)
    Ed->>Ed: Coherence check
    Ed->>Ed: Grammar correction
    Ed->>Ed: Fact cross-reference
    Ed-->>Orch: edited_article {quality_score}

    alt quality_score < 0.7
        Orch->>Wrt: rewrite_low_quality_sections
        Wrt-->>Orch: revised_drafts
        Orch->>Ed: re_edit(revised_drafts)
        Ed-->>Orch: final_article
    end

    Orch->>SEO: optimize(final_article, target_keywords)
    SEO->>SEO: Keyword density check
    SEO->>SEO: Meta description generation
    SEO-->>Orch: seo_article {meta, slug}

    Orch->>CMS: publish(seo_article)
    CMS-->>Orch: published_url
    Orch-->>C: WebSocket: job_complete {url, metrics}
```

**Flow Highlights:**
- Brief validation enforces schema before pipeline starts
- Research runs first to ground all subsequent generation
- Writing Pool parallelizes section drafting for 8x throughput
- Quality gate triggers selective rewrite rather than full regeneration
- SEO optimization is the final step before CMS push
- Client receives async updates via WebSocket throughout the pipeline
