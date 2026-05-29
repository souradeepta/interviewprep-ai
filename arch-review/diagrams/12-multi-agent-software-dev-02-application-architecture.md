# Multi-Agent Software Development - Application Architecture

```mermaid
graph TD
    subgraph PMLayer["PM Layer"]
        TaskManager["Task Manager<br/>(Feature Intake)"]
        SpecWriter["Spec Writer<br/>(Requirements)"]
    end

    subgraph ArchLayer["Architecture Layer"]
        ArchDesigner["Architecture Designer<br/>(Component Plan)"]
        InterfaceDefiner["Interface Definer<br/>(API Contracts)"]
    end

    subgraph CodeLayer["Code Generation Layer"]
        CodeGenerator["Code Generator<br/>(N Workers)"]
        CodeValidator["Code Validator<br/>(Syntax Check)"]
    end

    subgraph ReviewLayer["Review and Test Layer"]
        CodeReviewer["Code Reviewer<br/>(LLM Review)"]
        TestRunner["Test Runner<br/>(pytest/jest)"]
    end

    subgraph DeployLayer["Deploy Layer"]
        DeployManager["Deploy Manager<br/>(CI/CD)"]
        VersionControl["Version Control<br/>(Git)"]
    end

    TaskManager --> SpecWriter
    SpecWriter --> ArchDesigner
    ArchDesigner --> InterfaceDefiner
    InterfaceDefiner --> CodeGenerator
    CodeGenerator --> CodeValidator
    CodeValidator --> CodeReviewer
    CodeReviewer --> TestRunner
    TestRunner -->|Pass| DeployManager
    TestRunner -->|Fail| CodeGenerator
    DeployManager --> VersionControl
```

**Layer Breakdown:**
- **PM Layer**: Feature intake, requirement specification writing, task ticket creation
- **Architecture Layer**: Component design, API contract definition, implementation guidance
- **Code Generation**: Parallel code generation workers with syntax validation before review
- **Review and Test**: LLM-based code review followed by automated test execution
- **Deploy Layer**: Git version control, CI/CD pipeline trigger on test pass
