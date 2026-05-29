# Autonomous Research Agent - Application Architecture

```mermaid
graph TD
    subgraph TaskLayer["Task Layer"]
        TaskAPI["Task API<br/>(REST)"]
        PlanningModule["Planning Module<br/>(ReAct Loop)"]
    end

    subgraph ToolRegistry["Tool Registry"]
        ToolReg["Tool Registry<br/>(Tool Catalogue)"]
        WebSearchTool["Web Search Tool<br/>(Serper)"]
        CodeTool["Code Executor<br/>(Sandbox)"]
        DataTool["Data Tool<br/>(SQL Engine)"]
    end

    subgraph ContextMgmt["Context Management"]
        ContextMgr["Context Manager<br/>(Memory Buffer)"]
        Summarizer["Summarizer<br/>(Long Context)"]
    end

    subgraph SynthesisLayer["Synthesis Layer"]
        ResultParser["Result Parser<br/>(Structured)"]
        SynthEngine["Synthesis Engine<br/>(LLM)"]
        ReportWriter["Report Writer<br/>(Markdown)"]
    end

    TaskAPI --> PlanningModule
    PlanningModule --> ToolReg
    ToolReg --> WebSearchTool
    ToolReg --> CodeTool
    ToolReg --> DataTool
    WebSearchTool --> ResultParser
    CodeTool --> ResultParser
    DataTool --> ResultParser
    ResultParser --> ContextMgr
    ContextMgr --> Summarizer
    ContextMgr --> PlanningModule
    Summarizer --> SynthEngine
    SynthEngine --> ReportWriter
```

**Layer Breakdown:**
- **Task Layer**: REST API with ReAct (Reason + Act) planning loop controlling tool invocation
- **Tool Registry**: Catalogue of available tools with descriptions for LLM-based selection
- **Context Management**: Rolling memory buffer with LLM summarization for long research sessions
- **Synthesis Layer**: Structured result parsing, LLM synthesis, final Markdown report generation
