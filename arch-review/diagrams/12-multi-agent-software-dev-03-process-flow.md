# Multi-Agent Software Development - Process Flow

```mermaid
sequenceDiagram
    participant U as User
    participant PM as PM Agent
    participant Arch as Architect Agent
    participant Coders as Coder Agents
    participant Reviewer as Reviewer Agent
    participant TestRunner as Test Runner
    participant Deploy as Deploy Agent

    U->>PM: Feature request
    PM->>Arch: Design feature
    Arch-->>PM: Architecture plan and component specs
    PM->>Coders: Implement components (parallel)
    par Component A
        Coders->>Coders: Implement and test locally
    and Component B
        Coders->>Coders: Implement and test locally
    end
    Coders-->>Reviewer: Submit code for review
    Reviewer-->>Coders: Review feedback
    alt Changes requested
        Coders->>Coders: Apply fixes
        Coders-->>Reviewer: Resubmit
    else Review approved
        Reviewer->>TestRunner: Run full test suite
        TestRunner-->>Reviewer: Test results
        alt Tests fail
            TestRunner-->>Coders: Failure details
            Coders->>Coders: Fix failing tests
            Coders->>TestRunner: Rerun
        else Tests pass
            TestRunner->>Deploy: Trigger deployment
            Deploy-->>U: Feature deployed
        end
    end
```

**Key Decision Points:**
1. **Parallel Implementation**: Independent components coded concurrently to minimize total time
2. **Review Gate**: All code must pass LLM code review before test execution
3. **Test Failure Loop**: Test failures routed back to coder agents for targeted fixes
4. **Deployment Gate**: Deploy only triggered on clean test suite pass

**Optimization Points:**
- Component parallelism reduces wall-clock time proportional to number of independent components
- Interface contracts defined up-front allow parallel implementation without integration conflicts
- Review caching avoids re-reviewing unchanged code sections on fix iterations
