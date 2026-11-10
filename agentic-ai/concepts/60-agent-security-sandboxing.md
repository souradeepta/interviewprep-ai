# Agent Security and Sandboxing
**TL;DR:** Agent security limits what untrusted model output can cause through capability controls, isolated execution, and auditable policy enforcement.

## Prerequisites
Tool calling, operating-system permissions, and threat modeling.

## Learning Outcomes
- Identify prompt injection and confused-deputy paths.
- Apply least privilege to tools and execution environments.
- Test containment using adversarial inputs and side-effect assertions.

## Core Intuition
A sandbox is a locked workbench with only the tools needed for one job. Telling an agent “do not access other files” is a sign on the wall; operating-system permissions are the lock. Safety instructions help behavior, but containment must be enforced outside the model.

## How It Works
The operating flow identifies the task state, selects a bounded strategy, gathers only the evidence needed for that decision, validates the result, and records the outcome. Threat modeling must include indirect instructions inside retrieved content and actions taken by downstream tools. A model may be persuaded to send data to an allowed browser or database endpoint even when shell access is blocked, so policy checks need user, resource, and operation context. Separate identities for read and write tools, scope permissions to a task, and record authorization decisions. A sandbox limits damage after a mistake; it does not decide whether the requested action was appropriate.
Start from assets and trust boundaries: secrets, user data, source code, external services, and the agent’s tool gateway. Treat retrieved documents, emails, web pages, and code as untrusted data that may contain instructions. Separate data from control decisions, but assume prompt separation can fail. Give each tool a narrow schema and authorization check, scope credentials to one operation, and prevent arbitrary URL, shell, and filesystem access. For code execution, use disposable containers with resource limits, no ambient secrets, restricted network egress, and timeouts. Validate outputs before any privileged action.
Threat modeling must include indirect instructions inside retrieved content and actions taken by downstream tools. A model may be persuaded to send data to an allowed browser or database endpoint even when shell access is blocked, so policy checks need user, resource, and operation context. Separate identities for read and write tools, scope permissions to a task, and record authorization decisions. A sandbox limits damage after a mistake; it does not decide whether the requested action was appropriate.
Security tests should assert both the response and the absence of side effects. Use canary secrets, malicious retrieved instructions, path traversal inputs, oversized payloads, and attempts to access unauthorized tools. Measure blocked-action rate, false-denial rate on allowed workflows, secret exposure, and time to revoke a compromised capability. Review traces for sensitive content and define retention. A sandbox’s isolation claims depend on its runtime configuration; test the deployed image and kernel boundary, not only a local mock.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Prompt-only restriction | Easy to add | Cannot enforce against injection |
| Tool allowlist | Limits available actions | Needs correct per-user authorization |
| Container sandbox | Constrains execution | Kernel and configuration risk |
| Human approval gate | Controls high-impact action | Adds delay and reviewer burden |

## Implementation
This dependency-light example models a bounded policy decision. It is an executable illustration of the control boundary; it does not call an LLM or establish production quality.

```python
def authorize(tool, allowed, user_scope):
    return tool in allowed and user_scope.get(tool, False)

assert authorize('read_ticket', {'read_ticket', 'refund'}, {'read_ticket': True})
assert not authorize('refund', {'read_ticket', 'refund'}, {'read_ticket': True})
```

## Failure Modes
- Retrieved prompt injection triggers a tool: enforce authorization at the gateway and test malicious documents.
- Credential leaks into model context: use short-lived scoped credentials outside prompts and inspect traces.
- Sandbox escapes via network or files: deny by default, restrict egress, and test deployed isolation.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| Is prompt injection solved by a system prompt? | No; use untrusted-input handling plus external capability and authorization controls. Design follow-up: what does the tool gateway verify? |
| What belongs in a code sandbox? | Only task inputs and minimal dependencies, with bounded CPU, memory, time, filesystem, and network access. Debug follow-up: how do you test egress denial? |
| How do you contain a compromised agent? | Revoke scoped credentials, stop active jobs, preserve audit evidence, and replay the incident path. |

## Related Concepts
See [Tool use](03-tool-use.md), [safety alignment](36-safety-alignment.md), [agent deployment](56-agent-deployment-patterns.md).

## References
- [owasp.org](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [csrc.nist.gov](https://csrc.nist.gov/pubs/sp/800/207/final)

## Further Topic Detail
## Detailed Explanation

Agent security sandboxing constrains the actions agents can take, preventing them from accessing sensitive data, making dangerous API calls, or performing unauthorized operations. Unlike model-only systems where the model's output is reviewed before taking action, agents autonomously call tools, creating security risks if not carefully controlled. Sandboxing creates isolation: agents operate in constrained environments where their tool access is limited to safe operations.

Sandboxing approaches include: (1) Capability restrictions (agents can only access specific APIs), (2) Resource limits (rate limiting, timeouts, cost caps), (3) Input/output filtering (scrubbing sensitive data from agent inputs/outputs), (4) Approval workflows (requiring human approval for certain actions), (5) Execution containers (running agent code in isolated VMs), (6) Monitoring (detecting suspicious patterns). The key tension is between safety (more restrictions) and usefulness (more capabilities). A heavily sandboxed agent is safe but limited; an unrestricted agent is useful but dangerous.

Agent security sandboxing is crucial as agents gain more capabilities and operate in higher-stakes domains. Understanding it requires security thinking and appreciation for threat models—what could go wrong if agents misbehave, and what controls prevent those failures.


## Core Intuition

Giving someone access to your bank account is risky. A security sandbox is like giving them access to a limited, monitored account: they can perform safe operations, but can't drain all funds or access sensitive information. Each agent should work in such a 'sandbox' appropriate to their trustworthiness and the risks they could cause.


## How It Works

1. Sandboxing: isolated execution environment, limits resource access
2. Containerization: Docker containers with resource limits (CPU, memory, disk)
3. Code execution: execute agent-generated code in sandbox, not main process
4. Timeouts: kill execution if exceeds time limit (prevent infinite loops)
5. Permissions: restrict file system access, network access, capability access
6. Monitoring: log all system calls, detect malicious behavior
7. Auditing: track what code executed, who triggered it, what resources used




## Interview Q&A


**Q: Why is sandboxing necessary for agents?**
A: Risk: agent-generated code might have bugs (infinite loop, crash), or be malicious (steal credentials, delete files). Sandbox isolates: bug doesn't crash main system, malicious code has limited damage. Essential for untrusted agent code.

**Q: What are levels of sandboxing?**
A: Process isolation: separate OS process (some isolation). Container: Docker with resource limits (good isolation). VM: full virtual machine (complete isolation, expensive). Choose based on risk (untrusted code → VM, trusted code → process).

**Q: How do you handle code that needs external resources?**
A: Restricted APIs: agent calls whitelisted functions (not arbitrary code). Examples: read_file('/data/...') allowed, read_file('/etc/passwd') blocked. Requires careful API design to be both safe and useful.

**Q: What is the difference between code execution and code generation?**
A: Generation: agent writes code, human reviews, human executes (safe). Execution: agent writes and runs code (risky but faster). For agents: auto-execute only whitelisted operations (math, data processing), require approval for external operations.

**Q: How do you prevent agents from accessing credentials?**
A: Secrets management: don't pass credentials in prompts. Use secret vault (AWS Secrets Manager, HashiCorp Vault). Agent calls API with permission token (not full credentials). Audit: log which secrets accessed by whom. Rotate regularly.
