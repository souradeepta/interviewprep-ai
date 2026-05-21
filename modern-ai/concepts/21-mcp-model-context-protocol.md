# MCP (Model Context Protocol)

## Detailed Explanation

MCP (Model Context Protocol) is a critical modern technique in AI engineering. Anthropic protocol for LLM resource integration. This represents the practical state-of-the-art in how production AI systems are built and connected today. Understanding this technique is essential for building scalable, reliable AI systems that integrate seamlessly with external resources and services. The key insight is that MCP (Model Context Protocol) bridges the gap between LLMs and external systems, enabling agents to access tools, memory, and resources in a standardized way.

## Core Intuition

Think of MCP (Model Context Protocol) as the standardized language that lets LLMs talk to the rest of your infrastructure. Instead of each model needing custom integrations, you define once and use everywhere.

## How It Works

1. Define your resources, tools, or memory requirements
2. Implement the MCP (Model Context Protocol) protocol or use an SDK
3. Connect to your LLM or agent framework
4. Handle requests and responses through the standard interface
5. Scale across multiple models and deployments
6. Monitor and optimize the connections

```mermaid
graph TD
    A[LLM/Agent] --> B[MCP (Model Context Protocol)]
    B --> C[Resources/Tools]
    B --> D[Memory/Storage]
    B --> E[External Services]
    C --> F[Results]
    D --> F
    E --> F
```

## Architecture / Trade-offs

MCP can be deployed in different integration patterns, each with distinct trade-offs. The choice depends on your architecture, team expertise, and standardization goals.

### Integration Approach Comparison

| Aspect | Direct API Integration | MCP Wrapper Layer | MCP Gateway |
|--------|------------------------|-------------------|------------|
| Setup Complexity | Low (native code) | Medium (adapter code) | High (separate service) |
| Protocol Standardization | None (custom per API) | Full (single protocol) | Full + centralized |
| Tool Discovery | Manual (hardcoded) | Automatic (schema-driven) | Automatic (registry) |
| Versioning Control | Per implementation | Independent from apps | Decoupled completely |
| Deployment Overhead | Minimal (in-process) | Low (added layer) | Medium (network calls) |
| Scaling to 100+ tools | Difficult (coupling) | Moderate (manageable) | Easy (centralized) |

**Direct API**: Fastest to start, couples application logic to tool details. Breaks when tools change. Best for POCs with 1-2 stable tools.

**MCP Wrapper**: Adds standardization layer without operational overhead. Good balance for teams adopting MCP in existing apps. Requires wrapper maintenance.

**MCP Gateway**: Enterprise pattern for multi-team, multi-model deployments. Centralizes tool management and versioning. Adds network latency (50-200ms per call) but enables organization-wide governance.

### Protocol Complexity vs Flexibility

| Scenario | Approach | Rationale |
|----------|----------|-----------|
| Single LLM, 2-3 tools | Direct API | Standardization overhead not justified |
| Multi-LLM, 5-10 tools | MCP Wrapper | Benefits outweigh wrapper cost |
| Organization-wide (20+ teams) | MCP Gateway | Centralization essential, latency acceptable |
| Cost-sensitive inference | Direct API | Every network hop matters |
| Rapid tool iteration | MCP Pattern | Schema-driven discovery reduces friction |

## Design Challenges

MCP abstracts tool integration but introduces its own challenges in production:

- **Protocol Versioning & Compatibility**: When you deploy an MCP server v2.0 but clients still expect v1.x, silent failures occur. Tools may partially respond or drop fields. Requires explicit version negotiation, semantic versioning, and graceful degradation. Best practice: always include API version in protocol handshake; maintain backward compatibility for at least one version.

- **Resource Discovery**: In direct API integration, you hardcode endpoints. With MCP, tools advertise what they offer via schema. If schemas are incomplete or incorrect (missing parameters, wrong types), agents hallucinate tool usage. Discovery becomes a contract that must be tested and versioned. Requires comprehensive schema validation and integration tests.

- **Error Handling Across Diverse Tools**: MCP clients must handle errors from tools they've never seen before. A database tool returns timeout, an API returns rate-limit, a file tool returns permission denied. Standardizing error codes across independent tool providers is nearly impossible. Requires middleware that maps diverse errors to canonical MCP error types with retry logic.

- **Latency & Cascading Failures**: In monolithic systems, internal calls are fast. MCP adds network boundaries. When a tool is slow or unreachable, agents must decide: retry, fallback, or fail? Without proper timeouts and circuit breakers, one slow tool stalls an entire agent. Requires structured observability and latency budgets per tool.

- **Security & Multi-Tenancy**: MCP servers may serve multiple agents/organizations. A payment tool must not expose customer data across tenants. Authentication/authorization must be MCP-level, not just network-level. Requires careful isolation, audit logging, and token management.

## Interview Q&A

**Q: What problems does MCP solve that function calling doesn't?**
A: Function calling couples your LLM to specific tools defined at deploy time. MCP decouples: tools are discovered at runtime, schemas are versioned independently, and the same agent works with different tool sets. Example: function calling requires you to rebuild+redeploy the model config. MCP means you restart the agent server, tools auto-discover via MCP.

**Q: How do you discover what tools are available via MCP without hardcoding?**
A: MCP servers advertise their capabilities via a discovery mechanism (tools list, schemas). Clients query this at startup or on-demand to learn what's available. The agent then dynamically adapts. This breaks the static function-calling model where tools are baked into prompts. Requires middleware to convert discovery responses into usable tool descriptions for the LLM.

**Q: When would you NOT use MCP?**
A: Single-model deployments with 2-3 stable tools where direct function calling is simpler. MCP adds protocol overhead (network hops, schema negotiation). For latency-critical applications (<50ms RTT requirement), MCP gateway adds unacceptable overhead. For deeply integrated systems where tool and model are co-developed, tight coupling via function calling may be faster.

**Q: How do you handle versioning when tools change but agents must keep working?**
A: MCP requires explicit version tags in schemas. When a tool changes, old clients continue calling v1 (if server maintains it), new clients use v2. Requires schema migration strategy: maintain 2 versions, deprecate old one after grace period. Without this, you force all clients to upgrade atomically—defeating MCP's decoupling benefit.

**Q: What's the difference between tool discovery via MCP vs static function definitions?**
A: Static (function calling): Model learns tools during training/fine-tuning, tools are immutable at runtime, adding a tool requires model rebuild. Dynamic (MCP): Agent queries schemas at runtime, agent adapts behavior based on available tools, adding a tool is service deployment. MCP wins for flexibility; function calling wins for performance and control.

**Q: How do you debug a failing tool call through MCP?**
A: Check three layers: (1) MCP protocol (is request valid JSON-RPC? Schema matches?), (2) Tool implementation (does tool exist? correct parameters?), (3) Error mapping (did MCP correctly translate tool error to LLM-readable format?). Enable logging at protocol boundaries. Test tools independently first with curl/client, then in agent context.

## Best Practices

- Use official SDKs when available (don't reinvent the wheel)
- Version your protocol implementations and clients independently
- Implement proper error handling for all resource types
- Monitor connection latency and resource availability
- Test with multiple LLM models to ensure compatibility
- Document your resource schemas clearly for other developers
- Plan for scaling: MCP (Model Context Protocol) should work with thousands of resources

## Common Pitfalls

- **Tools don't implement MCP spec correctly**: A tool claims to support MCP but returns JSON with wrong structure, missing required fields, or non-standard error formats. The agent silently misinterprets the response or crashes. Result: tool works in isolation but fails in agent loop. Fix: Use strict schema validation, test tool responses against MCP spec before production.

- **Protocol version mismatch causes silent failures**: Client v1.0 talks to server v2.0. Both parse JSON successfully but interpret fields differently. Tool returns new v2 field that v1 client ignores. Agent doesn't get critical information, makes wrong decision. Result: hard to debug—no error, just wrong behavior. Fix: Include version in every request/response, implement version negotiation, log version mismatches.

- **No error handling on tool failure**: Agent calls tool, tool times out or returns error, agent continues with null/empty result, makes nonsensical follow-up calls. Users see incoherent agent behavior. Fix: Implement timeout per tool (not global), map all errors to canonical MCP error types, fail fast with user notification.

- **Assuming tools are always available**: Hardcode assumption that calculator tool exists in schema. Deploy agent, tool is temporarily down, agent crashes at first calculation. Result: production outage. Fix: Check tool availability before routing request, implement fallback strategies (use different tool, simplify task), monitor tool health separately.

- **Infinite recursion in tool chains**: Agent calls Tool A, which calls Tool B (via MCP), which calls Agent to refine query, which calls Tool A again. Creates circular dependency. May not manifest until tools are live. Fix: Implement call depth limits (max 3-4 levels), track tool call history, prevent agents calling themselves recursively.

## Code Examples

### Example 1: Basic Implementation

```python
# Basic MCP (Model Context Protocol) pattern
class Resource:
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def execute(self, params):
        return {'name': self.name, 'result': params}

# Define resources
calculator = Resource('calculator', 'Basic math operations')
memory = Resource('memory', 'Agent memory storage')

# Execute
result = calculator.execute({'operation': 'add', 'a': 5, 'b': 3})
print(result)
```

### Example 2: Production with Error Handling

```python
import logging
from typing import Dict, Any
import time

logger = logging.getLogger(__name__)

class ManagedResource:
    def __init__(self, name: str, timeout: int = 30):
        self.name = name
        self.timeout = timeout
        self.available = True
    
    def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f'Executing {self.name}: {request}')
            start = time.time()
            
            # Check availability
            if not self.available:
                return {'error': 'Resource unavailable'}
            
            # Execute with timeout
            result = self._do_execute(request)
            latency = time.time() - start
            
            logger.info(f'Completed in {latency:.2f}s')
            return {'success': True, 'result': result, 'latency': latency}
            
        except Exception as e:
            logger.error(f'Error: {e}')
            return {'error': str(e)}
    
    def _do_execute(self, request):
        # Your implementation here
        return request

# Usage
resource = ManagedResource('api-gateway', timeout=5)
response = resource.execute({'endpoint': '/data', 'query': 'test'})
print(response)
```

## Related Concepts

- [Agentic Testing Harness](./03-agentic-testing-harness.md)
- [Persistent AI Memory](./04-persistent-ai-memory.md)
- [LLMOps](./18-llmops.md)
- [AI Gateway & Routing](./19-ai-gateway-routing.md)
