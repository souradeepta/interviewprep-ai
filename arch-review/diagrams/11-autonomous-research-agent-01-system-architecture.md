# Autonomous Research Agent - System Architecture

```mermaid
graph TD
    subgraph TaskInput["Task Input"]
        TaskAPI["Task API<br/>(REST)"]
        TaskParser["Task Parser<br/>(Goal Extractor)"]
    end

    subgraph AgentCore["Agent Core"]
        Planner["Planner<br/>(LLM Decomposer)"]
        ToolSelector["Tool Selector<br/>(Action Policy)"]
        MemoryStore["Memory Store<br/>(Context Window)"]
    end

    subgraph Tools["Tool Executors"]
        WebSearch["Web Search<br/>(Serper API)"]
        CodeExecutor["Code Executor<br/>(Sandbox)"]
        DataTool["Data Tool<br/>(SQL/CSV)"]
    end

    subgraph Synthesis["Synthesis and Output"]
        ResultParser["Result Parser<br/>(Structured Extract)"]
        Synthesizer["Synthesizer<br/>(LLM Summary)"]
        ReportGen["Report Generator<br/>(Markdown)"]
    end

    TaskAPI --> TaskParser
    TaskParser --> Planner
    Planner --> ToolSelector
    ToolSelector --> WebSearch
    ToolSelector --> CodeExecutor
    ToolSelector --> DataTool
    WebSearch --> ResultParser
    CodeExecutor --> ResultParser
    DataTool --> ResultParser
    ResultParser --> MemoryStore
    MemoryStore --> Planner
    MemoryStore --> Synthesizer
    Synthesizer --> ReportGen
```

**Infrastructure Components:**
- **Planner**: LLM-based task decomposition into ordered sub-goals and tool calls
- **Tool Selector**: Policy-based selection of tools based on sub-goal type
- **Memory Store**: Rolling context window with summarization for long research tasks
- **Web Search**: Serper/Bing API for real-time web search with result parsing
- **Code Executor**: Sandboxed Python executor (E2B or Docker) for data analysis
- **Synthesizer**: LLM-based synthesis of gathered evidence into coherent report
