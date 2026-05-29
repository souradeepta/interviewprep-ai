# Autonomous Research Agent - Process Flow

```mermaid
sequenceDiagram
    participant U as User
    participant Agent as Research Agent
    participant Planner as Planner
    participant ToolSel as Tool Selector
    participant Tools as Tool Executors
    participant Memory as Memory Store
    participant Synth as Synthesizer

    U->>Agent: Submit research task
    Agent->>Planner: Decompose task into sub-goals
    Planner-->>Agent: Ordered list of sub-goals
    loop For each sub-goal
        Agent->>ToolSel: Select tool for sub-goal
        ToolSel-->>Agent: Tool and parameters
        Agent->>Tools: Execute tool call
        Tools-->>Agent: Tool result
        Agent->>Memory: Store result in context
        Memory-->>Agent: Updated context
    end
    Agent->>Planner: Evaluate completion (all sub-goals done?)
    alt More sub-goals needed
        Planner-->>Agent: Additional sub-goals
    else Research complete
        Agent->>Synth: Compile context into report
        Synth-->>Agent: Synthesized report
        Agent-->>U: Final research report
    end
```

**Key Decision Points:**
1. **Task Decomposition**: Planner breaks research task into specific, tool-actionable sub-goals
2. **Tool Selection**: LLM selects best tool for each sub-goal (web, code, data)
3. **Memory Management**: Context window monitored; summarization triggered when nearing limit
4. **Completion Check**: Planner evaluates if gathered information sufficiently addresses task

**Optimization Points:**
- Parallel tool execution for independent sub-goals reduces total research time
- Memory summarization preserves key findings when context exceeds window limit
- Result caching prevents duplicate web searches within the same research session
