# Agent Debugging

## TL;DR
Agents fail: trace steps, inspect tool calls, validate outputs. Tools: logging, stepping through execution, replaying sessions.

## Core Intuition
Agent is black box. Debug by: trace execution, see what happened, why it failed.

## How It Works
**Tools:**
- **Logging:** log every step, tool call, result
- **Replay:** given same input, same random seed, can replay
- **Inspection:** step through execution
- **Traces:** visualize agent loop

**Example:**
```
Session ID: xyz
Step 1: Tool="search", Params={...}, Result=...
Step 2: Tool="analyze", Params={...}, Result=ERROR
Step 3: Agent retried with different params
...
Session failed: Tool 2 always failed
```

## Common Mistakes / Gotchas
- **No logs:** can't debug without history
- **Non-deterministic:** random sampling → can't replay
- **PII in logs:** log carefully, don't expose sensitive data

## Interview Quick-Reference
**Debug agent?** Log steps, replay execution, inspect tool calls, trace decisions.

## Related Topics
- [Error Recovery](error-recovery.md)
- [Agent Testing](agent-testing.md)

## Resources
- [Debugging ML Systems](https://arxiv.org/abs/2202.02771)
