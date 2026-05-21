# Architecture Diagrams Library

This directory contains three types of architecture diagrams for each system:

1. **System Architecture** - Infrastructure, deployment, cloud services, scaling
2. **Application Architecture** - Components, modules, layers, internal structure
3. **Process Flow** - Request pipelines, business workflows, decision flows

## Systems with Diagrams

### Customer Service Platform
- [System Architecture](01-customer-service-01-system-architecture.md)
- [Application Architecture](01-customer-service-02-application-architecture.md)
- [Process Flow](01-customer-service-03-process-flow.md)

### RAG Document QA System
- [System Architecture](02-rag-system-01-system-architecture.md)
- [Application Architecture](02-rag-system-02-application-architecture.md)
- [Process Flow](02-rag-system-03-process-flow.md)

### AI Code Review Agent
- [System Architecture](05-code-review-agent-01-system-architecture.md)
- [Application Architecture](05-code-review-agent-02-application-architecture.md)
- [Process Flow](05-code-review-agent-03-process-flow.md)

## How to Use

### Integrate into Architecture Files
Include diagrams in your system architecture markdown:

```markdown
## System Architecture

[System-level infrastructure diagram]

## Application Architecture

[Component-level internal structure diagram]

## Process Flow

[Request pipeline and decision flow diagram]
```

### Create New Diagrams
Run the generator script:
```bash
python3 scripts/generate_architecture_diagrams.py
```

## Diagram Types Explained

### System Architecture
Shows infrastructure, deployment, and operational concerns:
- Kubernetes clusters, auto-scaling
- Databases, caching layers
- External APIs and services
- Monitoring and observability tools
- CDN, load balancers, geographic distribution

### Application Architecture
Shows component design and internal structure:
- API layers, handlers
- Service layers and modules
- Data access patterns
- Internal dependencies
- Configuration and utilities

### Process Flow
Shows request pipelines and decision workflows:
- Request → processing → response flow
- Decision points (if/then branches)
- Error handling paths
- Optimization points (caching, deduplication)
- Async operations

## Benefits

- **Clarity**: Different levels of detail for different audiences
- **Scalability**: Shows how system scales with load
- **Operations**: Shows what happens at deployment
- **Debugging**: Shows where errors can occur
- **Design**: Shows architectural patterns and trade-offs
