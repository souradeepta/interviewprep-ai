import os

# Concept 5: web-agents
web_agents = """# Web Agents

## Detailed Explanation

Web agents interact with websites through browser automation or API calls. Core mechanisms: (1) browser automation—control browser to click, type, navigate (Selenium, Puppeteer), (2) API calls—direct HTTP requests to web services, (3) parsing—extract data from HTML (BeautifulSoup, CSS selectors). Advantages: automate repetitive web tasks (data scraping, form filling), access web services not designed for automation, test web applications. Challenges: website changes break automation (selectors become invalid), JavaScript rendering (agent must wait for page to load), handling authentication (cookies, sessions), rate limiting (don't overwhelm server), legal/ethical concerns (respect robots.txt, terms of service). Best for: data collection (public data from websites), workflow automation (fill forms across multiple sites), testing (automated testing of web apps), content aggregation (collect articles from multiple sources).

## Core Intuition

Imagine hiring someone to browse websites and fill forms for you. They open browser, navigate to site, reads instructions on page, clicks buttons, fills forms. Web agents are this—automated web browsing and interaction.

## How It Works

Web agents operate through: select element → perform action → wait for response → extract data:

1. **Navigation** — Open website or call API
2. **Element Selection** — Find button/input using CSS selector, XPath, or API endpoint
3. **Action** — Click button, type text, submit form, or call API
4. **Wait** — Wait for page load or response
5. **Extraction** — Parse HTML/JSON response, extract needed data
6. **Validation** — Verify action succeeded (page changed, API returned 200)
7. **Next Step** — Repeat or finish

## Architecture / Trade-offs

**Automation Style:**
- **Browser automation** — Full browser control, handles JavaScript, slower
- **API calls** — Direct HTTP, faster, requires API documentation
- **Hybrid** — Use API when available, fallback to browser when needed

**Timing:**
- **Synchronous** — Wait for each action to complete before next
- **Asynchronous** — Fire actions, handle responses async, faster but complex

**Reliability:**
- **Exact** — Wait for specific element (fails if missing), precise
- **Robust** — Wait for state change (timeout if too long), forgiving

## Best Practices

1. **Respect Terms of Service** — Check if automation is allowed
2. **Respect Rate Limits** — Don't overwhelm server; add delays
3. **Use APIs When Available** — Faster and more reliable than browser automation
4. **Explicit Waits** — Wait for elements to appear, don't use sleep()
5. **Error Handling** — Handle missing elements, timeouts, network errors
6. **Logging** — Log actions and failures for debugging
7. **Headless Mode** — Run browser without UI (faster)
8. **Cookies/Sessions** — Persist authentication across requests

## Code Examples

### Example 1: Browser Automation

```python
from selenium import webdriver
from selenium.webdriver.common.by import By

def automate_login(username, password):
    driver = webdriver.Chrome()
    try:
        driver.get("https://example.com/login")
        driver.find_element(By.NAME, "username").send_keys(username)
        driver.find_element(By.NAME, "password").send_keys(password)
        driver.find_element(By.ID, "submit").click()
        driver.implicitly_wait(10)
        return driver.current_url == "https://example.com/dashboard"
    finally:
        driver.quit()
```

### Example 2: API-Based Web Agent

```python
import requests

def agent_api_call(endpoint, params):
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API error: {response.status_code}")

data = agent_api_call("https://api.example.com/search", {"q": "python"})
```

### Example 3: Hybrid Approach

```python
import httpx

class WebAgent:
    def __init__(self, base_url):
        self.client = httpx.Client(base_url=base_url)

    def get_data(self, endpoint):
        '''Try API first, fallback to scraping.'''
        try:
            response = self.client.get(f"/api/{endpoint}")
            return response.json()
        except Exception:
            # Fallback: scrape HTML
            return self._scrape(endpoint)

    def _scrape(self, endpoint):
        response = self.client.get(f"/{endpoint}")
        # Parse HTML and extract data
        return {}
```

## Related Concepts

- **Agent Loops** — Web agent loop: request, parse, extract, repeat
- **Error Recovery** — Handle failed requests, retries
- **Tool Use** — Web APIs as tools
- **Observability** — Monitor agent requests and responses
"""

# Concept 6: coding-agents
coding_agents = """# Coding Agents

## Detailed Explanation

Coding agents generate and execute code autonomously. Capabilities: (1) code generation—write Python/JavaScript/etc., (2) execution—run code with sandboxed environment, (3) testing—verify code works, (4) debugging—fix errors, (5) iteration—refine code based on test results. Advantages: automate programming tasks (generate boilerplate, fix bugs), assist developers (suggest improvements), educational (teach through examples). Challenges: hallucinated code (generated code looks right but doesn't work), security (executing code from LLM is risky), sandbox limitations (can't do everything, network access restricted). Best for: code generation from specs, automated testing, code review assistance, learning programming, data transformation scripts.

## Core Intuition

A pair programmer who can write, test, and refine code. You describe what you want ("parse JSON and find entries with value >100"), they write code, run it, see if it works, fix if needed, deliver working code.

## How It Works

Coding agents operate through: generate → test → validate → iterate:

1. **Specification** — User provides spec ("function that sorts list")
2. **Generation** — Agent generates code
3. **Execution** — Run code in sandbox
4. **Testing** — Agent writes tests or checks output
5. **Validation** — Verify code meets spec
6. **Iteration** — If failed, debug and regenerate

## Architecture / Trade-offs

**Language:**
- **Python** — Easiest to generate and execute
- **JavaScript** — Web-based, but execution more complex
- **Multi-language** — Flexible, but harder to validate

**Execution Environment:**
- **Sandboxed** — Safe (can't harm system), limited (no file access)
- **Unrestricted** — Powerful, but dangerous if agent misbehaves

**Validation:**
- **Syntax** — Check code is valid Python/JavaScript
- **Type** — Check types match (mypy, TypeScript)
- **Test** — Run tests, verify output

## Best Practices

1. **Sandboxed Execution** — Never run untrusted code outside sandbox
2. **Code Review** — Human reviews generated code before production
3. **Test-Driven** — Agent generates tests with code
4. **Type Hints** — Use types to constrain what agent can generate
5. **Error Feedback** — If code fails, show error to agent for iteration
6. **Guardrails** — Restrict what code can do (no network, no exec)
7. **Versioning** — Track generated code versions

## Code Examples

### Example 1: Simple Code Generator

```python
def generate_parse_json_function(spec):
    code = '''
import json

def parse_json(data):
    """''' + spec + '''"""
    obj = json.loads(data)
    return obj
'''
    return code

spec = "Parse JSON string and return Python dict"
code = generate_parse_json_function(spec)
exec(code)
```

### Example 2: Code with Execution

```python
import subprocess

def execute_code(code):
    '''Run code in subprocess (sandboxed).'''
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            timeout=5
        )
        return result.stdout.decode()
    except subprocess.TimeoutExpired:
        return "Execution timeout"

code = "print(sum([1, 2, 3, 4, 5]))"
result = execute_code(code)
```

### Example 3: Iterative Refinement

```python
class CodingAgent:
    def __init__(self, max_iterations=3):
        self.max_iterations = max_iterations

    def generate_and_test(self, spec, test_cases):
        for iteration in range(self.max_iterations):
            code = self._generate(spec)
            passed = self._test(code, test_cases)

            if passed:
                return code

            spec += "\\n# Previous attempt failed. Revise."

        return None

    def _generate(self, spec):
        # Mock: in practice, use LLM
        return f"# Generated from: {spec}\\ndef solution(x): return x"

    def _test(self, code, test_cases):
        # Run code with test cases
        return True  # Mock
```

## Related Concepts

- **Error Recovery** — Fixing code errors through iteration
- **Agent Loops** — Generate-test-refine loop
- **Tool Use** — Code execution as a tool
- **Observability** — Monitoring code execution
"""

# Concept 7: human-agent-collaboration
human_agent_collab = """# Human-Agent Collaboration

## Detailed Explanation

Human-agent collaboration combines human judgment with agent capabilities. Patterns: (1) human specifies goal, agent executes, human reviews, (2) agent suggests action, human approves, (3) agent handles routine, human handles exceptions, (4) agent gathers info, human decides. Advantages: agents handle volume, humans handle judgment; better than agent alone (safer) or human alone (slower). Challenges: keeping human in loop doesn't slow things down, ensuring human actually reviews (vs rubber-stamping), managing async back-and-forth (I ask agent, agent waits for me, I decide). Best for: high-stakes decisions (medical, legal), novel situations (agent unsure), user preferences (agent can't decide for user). Critical: design for human workflow—what does human actually do? Can they review in 30 seconds or needs 30 minutes?

## Core Intuition

Pair programming but human is expert, agent is assistant. Agent does research, proposes solution, human validates and improves. Neither alone is better than together.

## How It Works

Human-agent loops operate through: agent action → human review → human feedback → agent learns/adjusts:

1. **Agent Acts** — Agent takes action or proposes solution
2. **Human Reviews** — Human inspects decision/output
3. **Feedback** — Human provides feedback (approve, modify, reject)
4. **Agent Learns** — Agent updates understanding based on feedback
5. **Iteration** — Repeat until human satisfied

## Architecture / Trade-offs

**Synchronicity:**
- **Synchronous** — Agent waits for human approval (safe, slow)
- **Asynchronous** — Agent acts, human reviews later (fast, risky)
- **Hybrid** — Agent acts on low-risk items, waits on high-risk

**Oversight:**
- **All** — Human reviews every decision (safest, slowest)
- **Sampling** — Human reviews 10% (faster, higher risk)
- **Exception** — Human only reviews anomalies (best tradeoff)

## Best Practices

1. **Clear Responsibility** — Explicit: which decisions are agent's, which are human's
2. **Easy Feedback** — Make it easy for human to approve/reject (one click)
3. **Explain Reasoning** — Agent explains why it decided/suggested (builds trust)
4. **Async Default** — Agent acts, human reviews async (don't block)
5. **Escalation** — Agent flags uncertain decisions for human
6. **Human Feedback Loop** — Agent learns from human feedback
7. **Transparent Disagreement** — If agent disagrees with human, log it
8. **Skill Development** — Humans learn agent's capabilities over time

## Code Examples

### Example 1: Human Approval Loop

```python
class HumanAgentLoop:
    def __init__(self):
        self.pending_decisions = []

    def agent_proposes(self, decision, confidence):
        '''Agent proposes decision.'''
        if confidence > 0.95:
            self._execute(decision)  # Auto-execute high-confidence
        else:
            self.pending_decisions.append(decision)

    def human_reviews(self):
        '''Return pending decisions for human.'''
        return self.pending_decisions

    def human_feedback(self, decision_id, approved):
        '''Human provides feedback.'''
        if approved:
            self._execute(self.pending_decisions[decision_id])
        else:
            # Log rejection, agent can learn from it
            pass

    def _execute(self, decision):
        print(f"Executing: {decision}")
```

### Example 2: Async Collaboration

```python
import asyncio

class AsyncCollaborationAgent:
    def __init__(self):
        self.pending_review = []
        self.human_feedback = {}

    async def agent_work(self, task):
        result = await self._process(task)
        self.pending_review.append({"task": task, "result": result})
        return result

    async def human_review(self):
        '''Non-blocking human review.'''
        return self.pending_review

    def human_provide_feedback(self, task_id, feedback):
        '''Human provides async feedback.'''
        self.human_feedback[task_id] = feedback
        # Agent can learn from feedback

    async def _process(self, task):
        await asyncio.sleep(0.1)  # Simulate processing
        return f"Result of {task}"
```

### Example 3: Escalation-Based Collaboration

```python
class EscalatingCollaborativeAgent:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.decisions = []

    def decide(self, task, confidence):
        if confidence >= self.threshold:
            self._execute(task)
            self.decisions.append({"task": task, "type": "auto", "confidence": confidence})
        else:
            self._escalate_to_human(task, confidence)
            self.decisions.append({"task": task, "type": "escalate", "confidence": confidence})

    def _execute(self, task):
        print(f"✓ Auto-executed: {task}")

    def _escalate_to_human(self, task, confidence):
        print(f"⚠ Need human review ({confidence:.0%} confidence): {task}")
```

## Related Concepts

- **Autonomous Agents** — Agents operating without human
- **Human-in-the-Loop** — Keeping human involved
- **Safety Alignment** — Ensuring safety through collaboration
- **Error Recovery** — Humans fixing agent mistakes
- **Observability** — Making agent decisions transparent
"""

# Concept 8: agent-prompt-engineering
agent_prompts = """# Agent Prompt Engineering

## Detailed Explanation

Prompt engineering for agents is designing prompts that guide agent behavior. Key techniques: (1) role definition—"You are a helpful assistant", (2) context—provide background information, (3) instruction—explicit steps to follow, (4) examples—few-shot learning, (5) constraints—"Only use these tools", (6) feedback—guide agent toward correct behavior. Difference from user prompt engineering: agent prompts are long-lived (reused across many queries), stable (changing often breaks things), and optimized for specific behaviors (not just generating text). Challenges: brittleness (small prompt changes cause big behavior changes), testing (hard to measure if prompt change helps), scaling (more complex agent = longer prompt = more latency). Best for: steering agent toward specific behavior (be formal vs casual), constraining agent (only output JSON), teaching agent patterns (examples of good behavior).

## Core Intuition

Instructions to a team member. Clear instructions = better work. Vague instructions = disappointing results. Agent prompts are like detailed job description + playbook of how to do the job well.

## How It Works

Agent prompts operate through: system prompt (baseline behavior) + instructions (task-specific) + examples (few-shot) + constraints:

1. **System Prompt** — Define agent role and general behavior
2. **Task Instruction** — What to do
3. **Context** — Relevant information
4. **Examples** — Few-shot examples of desired behavior
5. **Constraints** — What agent cannot do
6. **Evaluation** — How to know if succeeded

## Best Practices

1. **Clear Role** — "You are a customer support agent"
2. **Explicit Constraints** — "Only approve orders <$1000"
3. **Examples** — Show 2-3 good examples of agent behavior
4. **Output Format** — Specify JSON, XML, or text format
5. **Fallback** — "If unsure, ask human"
6. **Testing** — A/B test prompt changes
7. **Versioning** — Track prompt versions
8. **Iteration** — Refine based on results

## Code Examples

### Example 1: System Prompt

```python
SYSTEM_PROMPT = """
You are a helpful customer support agent. Your role:
1. Answer customer questions about products
2. Help resolve issues
3. Escalate complex issues to humans

Constraints:
- Only recommend products in stock
- Don't make promises about delivery dates
- Always be polite

If unsure, ask customer for clarification.
"""
```

### Example 2: Few-Shot Examples

```python
EXAMPLES = [
    {
        "query": "What are your payment methods?",
        "response": "We accept credit cards (Visa, Mastercard, Amex), PayPal, and bank transfers."
    },
    {
        "query": "This product is broken!",
        "response": "I'm sorry to hear that! Let me help. Can you describe the issue? We'll arrange a replacement or refund."
    }
]
```

### Example 3: Output-Constrained Prompt

```python
OUTPUT_PROMPT = """
Respond in JSON format:
{
    "action": "approve" | "reject" | "escalate",
    "reason": "string",
    "confidence": 0-1
}
"""
```

## Related Concepts

- **Agent Loops** — Prompts guide agent behavior in loops
- **Tool Use** — Prompts teach agent to use tools
- **Safety Alignment** — Prompts enforce safety rules
- **Reflection** — Agents can reflect on prompts
"""

# Create 4 new concepts
concepts = {
    "web-agents": web_agents,
    "coding-agents": coding_agents,
    "human-agent-collaboration": human_agent_collab,
    "agent-prompt-engineering": agent_prompts
}

for name, content in concepts.items():
    path = f"/home/sbisw/github/interviewprep-ml/agentic-ai/concepts/{name}.md"
    with open(path, "w") as f:
        f.write(content)
    print(f"✓ Created {name}")

