# Agentic AI Roadmap

## Who This Is For
Engineers who want to build, evaluate, and deploy AI agents — from simple tool-calling loops
to complex multi-agent systems. Assumes basic LLM familiarity (complete LLM Roadmap Phase 1 first).

---

## Phase 1 — Foundations (Beginner)
**Goal:** Understand what agents are, build a basic agent loop, use tools and memory.
**Estimated time:** 1–2 weeks at 10 hrs/week

- [ ] [What Is an Agent](../agentic-ai/concepts/what-is-an-agent.md)
- [ ] [Tool Use](../agentic-ai/concepts/tool-use.md)
- [ ] [Memory Types](../agentic-ai/concepts/memory-types.md)
- [ ] Implement: [Basic Agent Loop](../agentic-ai/implementations/basic-agent-loop.ipynb)
- [ ] Implement: [Tool Calling Agent](../agentic-ai/implementations/tool-calling-agent.ipynb)
- [ ] Practice: [Agentic Theory Questions](../agentic-ai/interview-prep/agentic-theory-questions.md) — Q1–Q10

**Phase 1 exit check:**
- Can you build a ReAct agent from scratch using only raw API calls (no framework)?
- Can you explain the difference between in-context and external memory?

---

## Phase 2 — Core Depth (Intermediate)
**Goal:** Planning/reasoning strategies, multi-agent patterns, RAG agents.
**Estimated time:** 2–3 weeks at 10 hrs/week

- [ ] [Planning & Reasoning](../agentic-ai/concepts/planning-reasoning.md)
- [ ] [Multi-Agent Systems](../agentic-ai/concepts/multi-agent-systems.md)
- [ ] Implement: [RAG Agent](../agentic-ai/implementations/rag-agent.ipynb)
- [ ] Implement: [Multi-Agent Workflow](../agentic-ai/implementations/multi-agent-workflow.ipynb)
- [ ] Implement: [LangGraph Agent](../agentic-ai/implementations/langgraph-agent.ipynb)
- [ ] Implement: [Memory Agent](../agentic-ai/implementations/memory-agent.ipynb)
- [ ] Practice: [Agentic Theory Questions](../agentic-ai/interview-prep/agentic-theory-questions.md) — Q11–Q25

**Phase 2 exit check:**
- Can you implement a multi-agent workflow where one agent routes tasks to specialist agents?
- Can you explain Tree of Thought vs. ReAct vs. MCTS for planning?

---

## Phase 3 — Advanced + Production (Advanced)
**Goal:** Agent evals, safety, production deployment, system design.
**Estimated time:** 1–2 weeks at 10 hrs/week

- [ ] [Agent Evals](../agentic-ai/concepts/agent-evals.md)
- [ ] [Safety & Alignment](../agentic-ai/concepts/safety-alignment.md)
- [ ] [System Design — Agentic System Design](../agentic-ai/system-design/agentic-system-design.md)
- [ ] [System Design — Multi-Agent Orchestration](../agentic-ai/system-design/multi-agent-orchestration.md)
- [ ] [System Design — Production Agents](../agentic-ai/system-design/production-agents.md)
- [ ] Practice: [Agentic System Design Questions](../agentic-ai/interview-prep/agentic-system-design-questions.md)

**Phase 3 exit check:**
- Can you design an eval harness for a customer support agent?
- Can you design a multi-agent orchestration system with human-in-the-loop checkpoints?

---

## Interview Readiness Checklist
- [ ] Built a ReAct agent from raw API (no framework)
- [ ] Built a multi-agent workflow with at least 2 agents
- [ ] Can explain 3 planning/reasoning strategies with trade-offs
- [ ] Completed 20+ agentic theory questions in simulation format
- [ ] Completed one full agentic system design mock

---

## Suggested Weekly Schedule

| Week | Focus | Files |
|------|-------|-------|
| 1 | What is an agent + tool use + basic loop | what-is-an-agent.md, tool-use.md, basic-agent-loop.ipynb, tool-calling-agent.ipynb |
| 2 | Memory + RAG agent + planning | memory-types.md, planning-reasoning.md, rag-agent.ipynb, memory-agent.ipynb |
| 3 | Multi-agent + LangGraph | multi-agent-systems.md, multi-agent-workflow.ipynb, langgraph-agent.ipynb |
| 4 | Evals + safety + system design | agent-evals.md, safety-alignment.md, system design files, system design Qs |
