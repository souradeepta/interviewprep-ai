# Error Recovery

## TL;DR
Agents fail: timeouts, invalid outputs, hallucinations. Recovery strategies: retry, fallback, validation, ask clarification. Must handle gracefully.

## Core Intuition
Humans recover from errors: ask for help, try differently, validate facts. Agents need explicit error handling.

## How It Works
```
try:
  result = execute_tool(action)
except ToolError as e:
  if e.type == "timeout":
    retry(exponential_backoff)
  elif e.type == "invalid_params":
    fix_and_retry()
  elif e.type == "hallucination":
    validate_against_kb()
  else:
    fallback_action()
```

**Recovery strategies:**
- **Retry:** network error, transient failures
- **Fallback:** use alternative tool or simplified approach
- **Validation:** check result against KB before using
- **Ask clarification:** tell LLM result was invalid, ask to fix
- **Reset:** clear state and restart

## Common Mistakes / Gotchas
- **No error types:** generic "failed" → can't recover properly
- **Infinite retries:** no backoff → hammers system. Use exponential backoff.
- **Trusting hallucinations:** don't validate → bad decisions. Always validate.

## Interview Quick-Reference
**Error recovery?** Retry, fallback, validation, ask for clarification, reset. Set max_attempts.

## Related Topics
- [Agent Loops](agent-loops.md) — loop handles retries
- [Safety & Alignment](safety-alignment.md) — validation is safety check

## Resources
- [Robust AI Systems](https://openai.com/research/safety/)
