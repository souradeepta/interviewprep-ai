# Agent Loops

## TL;DR
Core loop: observe → think → act → observe. Repeat until goal or max steps. Typically 1-10 iterations. Enables iterative problem-solving.

## Core Intuition
Loop lets agents refine approach: try action, observe result, adjust. Like humans iteratively solving problems.

## How It Works
```
while not_done and step < max_steps:
  obs = environment.observe()
  thought = llm.think(obs, memory)
  action = llm.choose_action(thought, tools)
  result = execute(action)
  memory.update(action, result)
  step += 1
```

**Iteration example:**
```
Step 1: Think "need to search for flights"
        Act: search_flights(NYC-LA)
        Observe: found 5 options
        
Step 2: Think "need to check prices"
        Act: get_prices(options)
        Observe: prices 400-600
        
Step 3: Think "user budget $500, pick option 2"
        Act: book_flight(option_2)
        Observe: booked successfully
```

## Key Properties / Trade-offs
- Iterations: more steps = better decisions but higher latency/cost
- Max steps: prevents infinite loops (set to 5-10 typically)
- Early stopping: stop if goal reached

## Common Mistakes / Gotchas
- **Infinite loops:** agent gets stuck. Set max_steps, detect loops.
- **No early exit:** continue even after goal. Check goal after each step.
- **Cost explosion:** many steps = many LLM calls. Monitor cost.

## Interview Quick-Reference
**Agent loop?** Iterate: observe, think, act, observe. Until goal or max steps.

## Related Topics
- [What Is an Agent](what-is-an-agent.md) — core concept
- [Planning & Reasoning](planning-reasoning.md) — thinking component
- [Error Recovery](error-recovery.md) — handling failures in loop

## Resources
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
