## Process Flow (Agent Execution)

```mermaid
graph TD
    START["User Request"] --> PARSE["Parse User Input<br/>(Intent + Context)"]

    PARSE --> DECOMPOSE["Decompose Goal<br/>(Sub-tasks)"]
    DECOMPOSE --> PLAN["Create Plan<br/>(Action Sequence)"]

    PLAN --> LOOP["Agent Loop Iteration"]

    LOOP --> OBSERVE["Observe Current State<br/>(Context, History)"]
    OBSERVE --> THINK["Think/Reason<br/>(Generate Next Action)"]

    THINK --> LLM_CALL["Call LLM<br/>(Chain-of-Thought)"]
    LLM_CALL --> PARSE_ACTION["Parse Action<br/>(Tool + Args)"]

    PARSE_ACTION --> VALIDATE{{"Valid<br/>Action?"}}
    VALIDATE -->|No| ERROR_HANDLING["Error Handling<br/>(Retry/Replan)"]
    ERROR_HANDLING --> LOOP

    VALIDATE -->|Yes| SELECT_TOOL["Select Tool<br/>(from Registry)"]
    SELECT_TOOL --> EXECUTE["Execute Tool<br/>(Sandboxed)"]

    EXECUTE --> CAPTURE["Capture Output<br/>(Result)"]
    CAPTURE --> UPDATE_STATE["Update State<br/>(Memory + History)"]

    UPDATE_STATE --> CHECK_DONE{{"Goal<br/>Achieved?"}}
    CHECK_DONE -->|No| LOOP
    CHECK_DONE -->|Yes| REFLECT["Reflection<br/>(Self-Critique)"]

    REFLECT --> IMPROVE{{"Should<br/>Retry?"}}
    IMPROVE -->|Yes| LOOP
    IMPROVE -->|No| FORMAT["Format Final<br/>Answer"]

    FORMAT --> EXPLAIN["Add Explanation<br/>(Reasoning Chain)"]
    EXPLAIN --> RETURN["Return to User<br/>(With Trace)"]

    RETURN --> END["Complete"]
```

**Agent Execution Characteristics:**
- **Observe**: Get current context (user request, history, state)
- **Think**: Generate reasoning and next action using LLM
- **Act**: Execute selected tool with validated arguments
- **Reflect**: Self-critique and decide if refinement needed
- **Loop**: Continue until goal achieved or max iterations reached