# Skill Composition

## TL;DR
Build complex agent behaviors by composing simple skills. Skills: search, summarize, analyze, plan. Combine via orchestration: sequence, conditional, loops.

## Core Intuition
Skills are building blocks. Compose them to build complex behaviors without rewriting everything.

## How It Works
**Skills:**
- Search: query information
- Summarize: condense text
- Analyze: extract insights
- Plan: break into steps
- Execute: perform action

**Composition:**
```
Task: "Analyze trend and recommend action"
  ↓
Step 1: Skill.search("trend data")
Step 2: Skill.analyze(results)
Step 3: Skill.summarize(analysis)
Step 4: Skill.plan("recommendation")
Step 5: Skill.execute(recommendation)
```

**Types:**
- Sequential: one after another
- Conditional: if-then-else
- Loop: repeat until condition

## Interview Quick-Reference
**Skill composition?** Build complex behaviors by combining simple skills via orchestration.

## Related Topics
- [Tool Use](tool-use.md) — skills use tools
- [Planning & Reasoning](planning-reasoning.md) — orchestration strategies

## Resources
- [LangChain](https://python.langchain.com/) — skill composition framework
