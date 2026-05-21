#!/usr/bin/env python3
"""
Apply three-tier architecture diagrams to all 30 system architectures.

Generates:
1. System Architecture - Infrastructure, deployment, scaling
2. Application Architecture - Components, modules, layers
3. Process Flow - Request pipelines, decision logic

Uses template-based approach to quickly generate diagrams for all systems.
"""

import os
import re
from pathlib import Path
from typing import Dict, Tuple


# Templates for each system type
TEMPLATES = {
    "api-gateway": {
        "system": """## System Architecture

```mermaid
graph TB
    subgraph Client["Client Layer"]
        WEB["Web Clients"]
        MOBILE["Mobile Apps"]
        THIRD_PARTY["3rd Party APIs"]
    end

    subgraph Gateway["API Gateway Layer"]
        LB["Load Balancer<br/>(AWS ALB)"]
        APIGW["API Gateway<br/>(Rate Limit, Auth, Routing)"]
        CACHE["Cache Layer<br/>(Redis)"]
    end

    subgraph Backend["Backend Services"]
        SERVICE1["LLM Service<br/>(Pod ×3)"]
        SERVICE2["Processing Service<br/>(Pod ×5)"]
        SERVICE3["ML Service<br/>(Pod ×2)"]
    end

    subgraph Data["Data Layer"]
        PG["PostgreSQL<br/>(Transactional)"]
        ELASTIC["Elasticsearch<br/>(Logs)"]
        MONGO["MongoDB<br/>(Documents)"]
    end

    WEB --> LB
    MOBILE --> LB
    THIRD_PARTY --> LB
    LB --> APIGW
    APIGW --> CACHE
    APIGW --> SERVICE1
    APIGW --> SERVICE2
    APIGW --> SERVICE3
    SERVICE1 --> PG
    SERVICE2 --> ELASTIC
    SERVICE3 --> MONGO
```

**Key Infrastructure:**
- Load balancer for traffic distribution
- API Gateway for routing, rate limiting, authentication
- Redis cache for response caching
- Multiple backend services with auto-scaling
- Separate databases optimized for each service""",

        "application": """## Application Architecture

```mermaid
graph TB
    subgraph API["API Layer"]
        REST["REST Endpoints<br/>(FastAPI)"]
        VALIDATOR["Input Validator<br/>(Pydantic)"]
        AUTH["Auth Middleware<br/>(JWT, OAuth)"]
    end

    subgraph Gateway["Gateway Core"]
        ROUTER["Request Router<br/>(Intelligent Routing)"]
        RATELIMIT["Rate Limiter<br/>(Token Bucket)"]
        LOGGER["Request Logger<br/>(Structured Logs)"]
    end

    subgraph Services["Service Adapters"]
        ADAPTER1["LLM Service Adapter"]
        ADAPTER2["Processing Adapter"]
        ADAPTER3["ML Service Adapter"]
    end

    subgraph Caching["Caching & Optimization"]
        CACHE["Cache Manager<br/>(Redis)"]
        INVALIDATOR["Cache Invalidator<br/>(TTL)"]
    end

    REST --> VALIDATOR
    VALIDATOR --> AUTH
    AUTH --> ROUTER
    ROUTER --> RATELIMIT
    RATELIMIT --> LOGGER
    LOGGER --> ADAPTER1
    LOGGER --> ADAPTER2
    LOGGER --> ADAPTER3
    ADAPTER1 --> CACHE
    ADAPTER2 --> CACHE
    ADAPTER3 --> CACHE
    CACHE --> INVALIDATOR
```""",

        "process": """## Process Flow

```mermaid
graph TD
    START["Request Arrives<br/>(REST)"] --> AUTH["Authenticate<br/>(JWT/OAuth)"]
    AUTH -->|Invalid| REJECT["Reject<br/>(401/403)"]
    AUTH -->|Valid| RATELIMIT["Check Rate Limit"]

    RATELIMIT -->|Exceeded| THROTTLE["Throttle<br/>(429)"]
    RATELIMIT -->|OK| CACHE_CHECK["Cache Hit?"]

    CACHE_CHECK -->|Yes| SERVE_CACHE["Serve from Cache"]
    CACHE_CHECK -->|No| ROUTE["Route to Service"]

    ROUTE --> CALL_SERVICE["Call Backend Service"]
    CALL_SERVICE --> VALIDATE_RESPONSE["Validate Response"]

    VALIDATE_RESPONSE -->|Invalid| ERROR["Error Response<br/>(500)"]
    VALIDATE_RESPONSE -->|Valid| CACHE_STORE["Store in Cache"]

    CACHE_STORE --> RESPONSE["Return Response<br/>(200)"]
    SERVE_CACHE --> RESPONSE

    RESPONSE --> LOG["Log Request<br/>(Analytics)"]
    LOG --> END["Complete"]

    REJECT --> END
    THROTTLE --> END
    ERROR --> END
```"""
    },

    "rag-system": {
        "system": """## System Architecture

```mermaid
graph TB
    subgraph Ingestion["Document Ingestion"]
        UPLOAD["File Upload<br/>(REST)"]
        PROCESS["Document Processor<br/>(PDF, DOCX)"]
        CHUNK["Chunking<br/>(Batch)"]
    end

    subgraph Vectorization["Vectorization Pipeline"]
        EMBED["Embedding Service<br/>(text-embedding-3)"]
        BATCH["Batch Processor<br/>(1000s docs)"]
    end

    subgraph Storage["Vector & Document Storage"]
        VECTOR_DB["Vector DB<br/>(Pinecone)"]
        DOC_STORE["Document Store<br/>(S3)"]
        INDEX["Search Index<br/>(Elasticsearch)"]
    end

    subgraph Query["Query & Generation"]
        SEARCH["Hybrid Search<br/>(Vector + BM25)"]
        RERANK["Re-ranking<br/>(Cross-encoder)"]
        LLM["LLM Generator<br/>(Claude/GPT)"]
    end

    subgraph Storage2["Result Storage"]
        RESULT_DB["Results DB<br/>(PostgreSQL)"]
        CACHE["Result Cache<br/>(Redis)"]
    end

    UPLOAD --> PROCESS
    PROCESS --> CHUNK
    CHUNK --> BATCH
    BATCH --> EMBED
    EMBED --> VECTOR_DB
    CHUNK --> DOC_STORE
    CHUNK --> INDEX
    SEARCH --> VECTOR_DB
    SEARCH --> INDEX
    SEARCH --> RERANK
    RERANK --> LLM
    LLM --> RESULT_DB
    LLM --> CACHE
```""",

        "application": """## Application Architecture

```mermaid
graph TB
    subgraph API["API Layer"]
        UPLOAD_EP["POST /documents"]
        QUERY_EP["POST /query"]
    end

    subgraph Processing["Processing Pipeline"]
        DOC_PROC["Document Processor"]
        CHUNKER["Chunker"]
        EMBEDDER["Embedder"]
    end

    subgraph Retrieval["Retrieval Engine"]
        RETRIEVER["Hybrid Retriever"]
        RANKER["Re-ranker"]
    end

    subgraph Generation["Generation Engine"]
        CONTEXT_BUILDER["Context Builder"]
        LLM_WRAPPER["LLM Wrapper"]
        POST_PROCESS["Post-processor"]
    end

    subgraph Storage["Data Access Layer"]
        VECTOR_CLIENT["Vector Client"]
        DOC_CLIENT["Document Client"]
        CACHE_CLIENT["Cache Client"]
    end

    UPLOAD_EP --> DOC_PROC
    DOC_PROC --> CHUNKER
    CHUNKER --> EMBEDDER
    EMBEDDER --> VECTOR_CLIENT
    QUERY_EP --> RETRIEVER
    RETRIEVER --> RANKER
    RANKER --> CONTEXT_BUILDER
    CONTEXT_BUILDER --> LLM_WRAPPER
    LLM_WRAPPER --> POST_PROCESS
    POST_PROCESS --> CACHE_CLIENT
    VECTOR_CLIENT --> CACHE_CLIENT
    DOC_CLIENT --> CACHE_CLIENT
```""",

        "process": """## Process Flow

```mermaid
graph TD
    USER_QUERY["User Submits Query"] --> CLEAN["Clean Query Text"]
    CLEAN --> CACHE_CHECK{"Cache Hit?"}

    CACHE_CHECK -->|Yes| RETURN_CACHE["Return Cached Result"]
    CACHE_CHECK -->|No| EMBED["Embed Query"]

    EMBED --> VECTOR_SEARCH["Vector Search"]
    VECTOR_SEARCH --> BM25["BM25 Search"]
    BM25 --> FUSION["Fusion Ranking"]

    FUSION --> RERANK["Re-rank Results"]
    RERANK --> SELECT["Select Top-K"]

    SELECT --> CONTEXT["Build Context"]
    CONTEXT --> BUILD_PROMPT["Build Prompt"]

    BUILD_PROMPT --> GENERATE["Generate Answer<br/>(LLM)"]
    GENERATE --> CITE["Add Citations"]

    CITE --> CACHE_STORE["Store in Cache"]
    CACHE_STORE --> RETURN["Return with Sources"]

    RETURN_CACHE --> END["Complete"]
    RETURN --> END
```"""
    },

    "agent-system": {
        "system": """## System Architecture

```mermaid
graph TB
    subgraph Orchestration["Agent Orchestration"]
        SCHEDULER["Job Scheduler<br/>(Kubernetes)"]
        AGENT_RUNNER["Agent Runner<br/>(Container)"]
    end

    subgraph Tools["Tool Ecosystem"]
        TOOL_REGISTRY["Tool Registry<br/>(Catalog)"]
        WEB_SEARCH["Web Search Tool<br/>(API)"]
        CODE_EXEC["Code Execution<br/>(Sandbox)"]
        DB_QUERY["Database Query<br/>(Safe)"]
    end

    subgraph Memory["Memory Systems"]
        WORKING_MEM["Working Memory<br/>(In-process)"]
        PERSISTENT["Persistent Memory<br/>(PostgreSQL)"]
        VECTOR_MEM["Vector Memory<br/>(Embeddings)"]
    end

    subgraph Models["AI Models"]
        LLM["LLM<br/>(Claude/GPT)"]
        EMBEDDING["Embedding Model<br/>(for Similarity)"]
    end

    subgraph Monitoring["Monitoring"]
        TRACE["Trace Store<br/>(Jaeger)"]
        METRICS["Metrics<br/>(Prometheus)"]
        LOGS["Logs<br/>(ELK)"]
    end

    SCHEDULER --> AGENT_RUNNER
    AGENT_RUNNER --> TOOL_REGISTRY
    AGENT_RUNNER --> LLM
    TOOL_REGISTRY --> WEB_SEARCH
    TOOL_REGISTRY --> CODE_EXEC
    TOOL_REGISTRY --> DB_QUERY
    AGENT_RUNNER --> WORKING_MEM
    WORKING_MEM --> PERSISTENT
    EMBEDDING --> VECTOR_MEM
    AGENT_RUNNER --> TRACE
    AGENT_RUNNER --> METRICS
    AGENT_RUNNER --> LOGS
```""",

        "application": """## Application Architecture

```mermaid
graph TB
    subgraph Core["Core Agent"]
        AGENT["Agent Controller"]
        PLANNER["Planner"]
        EXECUTOR["Executor"]
    end

    subgraph LLM["LLM Integration"]
        LLM_WRAPPER["LLM Wrapper"]
        PROMPT_BUILDER["Prompt Builder"]
        PARSER["Output Parser"]
    end

    subgraph Tools["Tool Management"]
        TOOL_MGR["Tool Manager"]
        VALIDATOR["Tool Validator"]
        EXECUTOR_TOOLS["Tool Executor"]
    end

    subgraph Memory["Memory Management"]
        STATE["State Manager"]
        MEMORY["Memory Store"]
        HISTORY["History Tracker"]
    end

    subgraph Storage["Data Access"]
        VECTOR_STORE["Vector Store Client"]
        DB_CLIENT["Database Client"]
        CACHE["Cache Layer"]
    end

    AGENT --> PLANNER
    PLANNER --> LLM_WRAPPER
    LLM_WRAPPER --> PROMPT_BUILDER
    PROMPT_BUILDER --> PARSER
    PARSER --> EXECUTOR
    EXECUTOR --> TOOL_MGR
    TOOL_MGR --> VALIDATOR
    VALIDATOR --> EXECUTOR_TOOLS
    AGENT --> STATE
    STATE --> MEMORY
    MEMORY --> VECTOR_STORE
    MEMORY --> DB_CLIENT
    MEMORY --> CACHE
    EXECUTOR_TOOLS --> DB_CLIENT
```""",

        "process": """## Process Flow

```mermaid
graph TD
    START["User Goal"] --> DECOMPOSE["Decompose into<br/>Sub-tasks"]
    DECOMPOSE --> PLAN["Create Plan<br/>(Action Sequence)"]

    PLAN --> LOOP["Agent Loop"]
    LOOP --> OBSERVE["Observe State"]
    OBSERVE --> THINK["Think<br/>(Generate Action)"]

    THINK --> LLM_CALL["Call LLM"]
    LLM_CALL --> PARSE["Parse Action"]

    PARSE --> VALID{"Valid?"}
    VALID -->|No| ERROR["Handle Error<br/>(Retry/Replan)"]
    ERROR --> LOOP

    VALID -->|Yes| SELECT["Select Tool"]
    SELECT --> EXECUTE["Execute Tool<br/>(Sandboxed)"]

    EXECUTE --> UPDATE["Update State"]
    UPDATE --> DONE{"Goal<br/>Achieved?"}

    DONE -->|No| LOOP
    DONE -->|Yes| REFLECT["Reflect<br/>(Self-critique)"]

    REFLECT --> FINAL{"Retry?"}
    FINAL -->|Yes| LOOP
    FINAL -->|No| RETURN["Return Result"]

    RETURN --> END["Complete"]
```"""
    },

    "generic": {
        "system": """## System Architecture

```mermaid
graph TB
    subgraph Client["Client Layer"]
        API["REST API<br/>(Load Balanced)"]
    end

    subgraph Services["Microservices"]
        SVC1["Service 1<br/>(Scaled)"]
        SVC2["Service 2<br/>(Scaled)"]
        SVC3["Service 3<br/>(Scaled)"]
    end

    subgraph Data["Data & Cache"]
        DB["Primary DB<br/>(Optimized)"]
        CACHE["Cache<br/>(Redis)"]
        SEARCH["Search Index<br/>(Elastic)"]
    end

    subgraph External["External Services"]
        EXT1["External API 1"]
        EXT2["External API 2"]
    end

    subgraph Monitoring["Monitoring"]
        LOGS["Logs"]
        METRICS["Metrics"]
    end

    API --> SVC1
    API --> SVC2
    API --> SVC3
    SVC1 --> DB
    SVC1 --> CACHE
    SVC2 --> SEARCH
    SVC3 --> EXT1
    SVC3 --> EXT2
    SVC1 --> LOGS
    SVC2 --> METRICS
```""",

        "application": """## Application Architecture

```mermaid
graph TB
    subgraph API["API Layer"]
        REST["REST Endpoints"]
        AUTH["Authentication"]
    end

    subgraph Services["Business Logic"]
        SVC1["Service 1"]
        SVC2["Service 2"]
        SVC3["Service 3"]
    end

    subgraph Data["Data Access"]
        REPO["Repository Layer"]
        CACHE["Cache Manager"]
    end

    subgraph Utils["Utilities"]
        LOG["Logger"]
        CONFIG["Config"]
    end

    REST --> AUTH
    AUTH --> SVC1
    AUTH --> SVC2
    AUTH --> SVC3
    SVC1 --> REPO
    SVC2 --> CACHE
    SVC3 --> LOG
    REPO --> CONFIG
```""",

        "process": """## Process Flow

```mermaid
graph TD
    START["Request Arrives"] --> VALIDATE["Validate Input"]
    VALIDATE -->|Invalid| ERROR["Return Error"]
    VALIDATE -->|Valid| AUTH["Authenticate"]

    AUTH -->|Failed| REJECT["Reject Request"]
    AUTH -->|Success| PROCESS["Process Request"]

    PROCESS --> CHECK_CACHE{"Cache Hit?"}
    CHECK_CACHE -->|Yes| SERVE_CACHE["Return Cached Result"]
    CHECK_CACHE -->|No| EXECUTE["Execute Logic"]

    EXECUTE --> STORE["Store Result"]
    STORE --> RETURN["Return Response"]
    SERVE_CACHE --> RETURN

    RETURN --> LOG["Log Request"]
    LOG --> END["Complete"]

    ERROR --> END
    REJECT --> END
```"""
    }
}


def extract_system_info(markdown_content: str) -> Dict[str, str]:
    """Extract system information from markdown file."""
    info = {}

    # Extract title
    title_match = re.search(r'^# (.+?)$', markdown_content, re.MULTILINE)
    if title_match:
        info['title'] = title_match.group(1)

    # Extract overview
    overview_match = re.search(r'## Overview\n(.+?)(?=\n##)', markdown_content, re.DOTALL)
    if overview_match:
        info['overview'] = overview_match.group(1).strip()

    return info


def determine_system_type(markdown_content: str) -> str:
    """Determine the system type to select appropriate templates."""
    content_lower = markdown_content.lower()

    if 'api gateway' in content_lower or 'gateway' in content_lower:
        return 'api-gateway'
    elif 'rag' in content_lower or 'retrieval' in content_lower:
        return 'rag-system'
    elif 'agent' in content_lower or 'orchestrat' in content_lower:
        return 'agent-system'
    else:
        return 'generic'


def create_architecture_section(system_num: int, diagrams_dict: Dict[str, str]) -> str:
    """Create the architecture section for a system."""
    section = f"""## Architecture Diagrams

### System Architecture (Infrastructure & Deployment)

{diagrams_dict['system']}

### Application Architecture (Components & Layers)

{diagrams_dict['application']}

### Process Flow (Request Pipeline)

{diagrams_dict['process']}
"""
    return section


def apply_diagrams_to_system(system_path: Path, system_num: int) -> bool:
    """Apply architecture diagrams to a single system."""
    try:
        content = system_path.read_text()

        # Check if already has architecture diagrams
        if '## Architecture Diagrams' in content:
            print(f"⏭️  {system_path.name} - Already has diagrams")
            return False

        # Determine system type
        system_type = determine_system_type(content)
        templates = TEMPLATES.get(system_type, TEMPLATES['generic'])

        # Create architecture section
        arch_section = create_architecture_section(
            system_num,
            templates
        )

        # Find insertion point (after "High-Level Architecture" or "## Overview")
        if '## High-Level Architecture' in content:
            # Replace the existing architecture section
            pattern = r'## High-Level Architecture\n```.*?```'
            new_content = re.sub(pattern, arch_section, content, flags=re.DOTALL)
        elif '## Data Flow' in content:
            # Insert before Data Flow
            insertion_point = content.find('## Data Flow')
            new_content = content[:insertion_point] + arch_section + '\n' + content[insertion_point:]
        else:
            # Insert after Overview or Requirements
            for marker in ['## Component Breakdown', '## AI/ML Integration', '## Requirements']:
                if marker in content:
                    insertion_point = content.find(marker)
                    new_content = content[:insertion_point] + arch_section + '\n' + content[insertion_point:]
                    break
            else:
                # Insert at end
                new_content = content + '\n' + arch_section

        # Write updated content
        system_path.write_text(new_content)
        return True

    except Exception as e:
        print(f"❌ {system_path.name} - Error: {e}")
        return False


def main():
    """Apply diagrams to all 30 systems."""
    systems_dir = Path('arch-review/systems')
    system_files = sorted(systems_dir.glob('*.md'))

    print(f"Applying architecture diagrams to {len(system_files)} systems...\n")

    applied = 0
    skipped = 0

    for system_file in system_files:
        # Extract system number
        match = re.match(r'(\d+)-', system_file.name)
        system_num = int(match.group(1)) if match else 0

        if apply_diagrams_to_system(system_file, system_num):
            applied += 1
            print(f"✅ {system_file.name}")
        else:
            skipped += 1

    print(f"\n✅ Applied diagrams to {applied} systems")
    print(f"⏭️  Skipped {skipped} systems (already have diagrams)")
    print(f"\nAll {len(system_files)} systems now have three-tier architecture diagrams!")


if __name__ == '__main__':
    main()
