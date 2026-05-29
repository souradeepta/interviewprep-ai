# Multi-Agent Software Development - System Architecture

```mermaid
graph TD
    subgraph Orchestration["Orchestration Layer"]
        TaskMgr["Task Manager<br/>(PM Agent)"]
        ArchAgent["Architect Agent<br/>(Design)"]
    end

    subgraph CodeGeneration["Code Generation Layer"]
        CoderA["Coder Agent A<br/>(Feature)"]
        CoderB["Coder Agent B<br/>(Feature)"]
        CoderC["Coder Agent C<br/>(Feature)"]
    end

    subgraph QualityAssurance["Quality Assurance"]
        ReviewAgent["Reviewer Agent<br/>(Code Review)"]
        TestAgent["Test Agent<br/>(Test Runner)"]
    end

    subgraph Deployment["Deployment"]
        DeployAgent["Deploy Agent<br/>(CI/CD Trigger)"]
        ArtifactStore["Artifact Store<br/>(S3)"]
    end

    subgraph SharedState["Shared State"]
        CodeRepo["Code Repo<br/>(Git)"]
        TaskBoard["Task Board<br/>(State DB)"]
    end

    TaskMgr --> ArchAgent
    ArchAgent --> CoderA
    ArchAgent --> CoderB
    ArchAgent --> CoderC
    CoderA --> ReviewAgent
    CoderB --> ReviewAgent
    CoderC --> ReviewAgent
    ReviewAgent --> TestAgent
    TestAgent -->|Pass| DeployAgent
    TestAgent -->|Fail| CoderA
    DeployAgent --> ArtifactStore
    CoderA --> CodeRepo
    CoderB --> CodeRepo
    CoderC --> CodeRepo
    TaskMgr --> TaskBoard
```

**Infrastructure Components:**
- **Task Manager (PM Agent)**: Receives feature requests, creates tickets, delegates to Architect
- **Architect Agent**: Designs system structure, creates component specs, delegates to coders
- **Coder Agents (N parallel)**: Independent feature implementation workers
- **Reviewer Agent**: Code review with LLM-based quality checks (style, security, correctness)
- **Test Agent**: Runs test suite, reports failures back to coder agents for fix cycles
- **Deploy Agent**: Triggers CI/CD pipeline on test pass, manages artifact storage
