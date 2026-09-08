# Agentic AI Roadmap

## Who This Is For
Engineers who want to build, evaluate, and deploy AI agents — from simple tool-calling loops
to complex multi-agent systems. Assumes basic LLM familiarity (complete LLM Roadmap Phase 1 first).

---

## Phase 1 — Foundations (Beginner)
**Goal:** Understand what agents are, build a basic agent loop, use tools and memory.
**Estimated time:** 1–2 weeks at 10 hrs/week

- [ ] [What Is an Agent](../agentic-ai/concepts/01-what-is-an-agent.md)
- [ ] [Tool Use](../agentic-ai/concepts/03-tool-use.md)
- [ ] [Memory Types](../agentic-ai/concepts/11-memory-types.md)
- [ ] Implement: [Basic Agent Loop](../agentic-ai/implementations/01-what-is-an-agent.py)
- [ ] Implement: [Tool Calling Agent](../agentic-ai/implementations/04-tool-calling.py)
- [ ] Practice: [Shared ML Interview Prep](../ml/interview-prep/README.md) — Q1–Q10

**Phase 1 exit check:**
- Can you build a ReAct agent from scratch using only raw API calls (no framework)?
- Can you explain the difference between in-context and external memory?

---

## Phase 2 — Core Depth (Intermediate)
**Goal:** Planning/reasoning strategies, multi-agent patterns, RAG agents.
**Estimated time:** 2–3 weeks at 10 hrs/week

- [ ] [Planning & Reasoning](../agentic-ai/concepts/07-planning-reasoning.md)
- [ ] [Multi-Agent Systems](../agentic-ai/concepts/14-multi-agent-systems.md)
- [ ] Implement: [RAG Agent](../agentic-ai/implementations/25-retrieval-augmented-generation.py)
- [ ] Implement: [Multi-Agent Workflow](../agentic-ai/implementations/14-multi-agent-systems.py)
- [ ] Implement: [LangGraph Agent](../agentic-ai/implementations/53-langchain-frameworks.py)
- [ ] Implement: [Memory Agent](../agentic-ai/implementations/12-agent-memory-management.py)
- [ ] Practice: [Shared ML Interview Prep](../ml/interview-prep/README.md) — Q11–Q25

**Phase 2 exit check:**
- Can you implement a multi-agent workflow where one agent routes tasks to specialist agents?
- Can you explain Tree of Thought vs. ReAct vs. MCTS for planning?

---

## Phase 3 — Advanced + Production (Advanced)
**Goal:** Agent evals, safety, production deployment, system design.
**Estimated time:** 1–2 weeks at 10 hrs/week

- [ ] [Agent Evals](../agentic-ai/concepts/29-agent-evals.md)
- [ ] [Safety & Alignment](../agentic-ai/concepts/36-safety-alignment.md)
- [ ] [Agent Deployment Patterns](../agentic-ai/concepts/56-agent-deployment-patterns.md)
- [ ] [Multi-Agent Systems](../agentic-ai/concepts/14-multi-agent-systems.md)
- [ ] [Production Agents](../agentic-ai/concepts/56-agent-deployment-patterns.md)
- [ ] Practice: [System Design Framework](../system-design/interview-prep/system-design-framework.md)
- [ ] Practice: [Agent Interview Questions](../agentic-ai/interview-prep/agent-interview-questions.md) — Q1–Q20
- [ ] Practice: [Agent Coding Exercises](../agentic-ai/interview-prep/agent-coding-exercises.md)

**Phase 3 exit check:**
- Can you design an eval harness for a customer support agent?
- Can you design a multi-agent orchestration system with human-in-the-loop checkpoints?

---

## Interview Readiness Checklist
- [ ] Built a ReAct agent from raw API (no framework)
- [ ] Built a multi-agent workflow with at least 2 agents
- [ ] Can explain 3 planning/reasoning strategies with trade-offs
- [ ] Completed 20 agent interview questions in simulation format
- [ ] Completed one full agentic system design mock

---

## Suggested Weekly Schedule

| Week | Focus | Files |
|------|-------|-------|
| 1 | What is an agent + tool use + basic loop | what-is-an-agent.md, tool-use.md, basic-agent-loop.ipynb, tool-calling-agent.ipynb |
| 2 | Memory + RAG agent + planning | memory-types.md, planning-reasoning.md, rag-agent.ipynb, memory-agent.ipynb |
| 3 | Multi-agent + LangGraph | multi-agent-systems.md, multi-agent-workflow.ipynb, langgraph-agent.ipynb |
| 4 | Evals + safety + system design | agent-evals.md, safety-alignment.md, system design files, system design Qs |
