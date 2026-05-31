---
title: "Tree of Thoughts: Deliberate Problem Solving with Language Models"
authors: "Yao, Yu, Zhao, et al."
year: 2023
venue: "ICLR"
arxiv: "https://arxiv.org/abs/2305.10601"
domain: "agents"
difficulty: "advanced"
interview_frequency: "high"
related_concepts:
  - agentic-ai/concepts/XX-planning
  - agentic-ai/concepts/XX-search-algorithms
---

# Tree of Thoughts: Deliberate Problem Solving with Language Models

## Paper Overview

**Title:** Tree of Thoughts: Deliberate Problem Solving with Language Models

**Authors:** Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, et al. (Google DeepMind)

**Published:** ICLR 2023 | [arXiv](https://arxiv.org/abs/2305.10601)

**Citation:** 3,000+ (emerging standard for complex reasoning)

Chain-of-Thought asks models to reason step-by-step, but follows a single path: each thought leads to one action, which leads to one observation. This greedy approach works for simple problems but fails for complex tasks where you need to *explore alternatives*, *backtrack*, or *prune bad paths*. Tree of Thoughts (ToT) generalizes CoT from a **linear chain** to a **tree of possible thoughts**, enabling models to consider multiple reasoning paths, evaluate them, and pursue the most promising branches. For problems where greedy search fails (like 24-game, code generation, or planning), ToT dramatically improves success rates by enabling deliberate exploration.

**Why this matters for interviews:** ToT is becoming standard for hard problems. Interviews ask: "How would you solve this if the model keeps failing?", "What if you had multiple options?", "How do you prune bad branches?". ToT provides the conceptual framework.

---

## Core Contribution

### The Problem: Single Greedy Path Fails

Chain-of-Thought generates one step at a time, committing to that step forever. For complex problems, this fails:

- **24-game:** Given four numbers (e.g., 3, 8, 3, 8), find arithmetic operations to make 24. CoT tries one sequence, hits a dead end, can't backtrack.
- **Code generation:** Model generates one line of code, if wrong, can't reconsider. Needs to explore alternatives.
- **Planning:** Model plans one step at a time. If the step doesn't work, entire plan fails.

**Why greedy fails:**
1. Some problems have multiple valid solution paths
2. Early wrong choices lead to unsolvable states
3. No mechanism to backtrack or reconsider
4. Model can't explore "what if" scenarios

### The Solution: Explore Multiple Paths

Instead of one linear chain, generate multiple possible next thoughts at each step, evaluate them, and explore promising branches:

```
Initial State: [3, 8, 3, 8], Target: 24
        |
    ├── Thought 1: "3 + 8 = 11"
    │   └── State: [11, 3, 8], next steps...
    │
    ├── Thought 2: "3 * 8 = 24" ← GOAL REACHED!
    │   └── SOLUTION FOUND
    │
    └── Thought 3: "8 - 3 = 5"
        └── State: [5, 3, 8], next steps...
```

By exploring multiple branches (Thought 1, 2, 3), the model finds the solution via Thought 2.

### Key Innovation: Structured Exploration

Tree of Thoughts formulates problem-solving as **search**:

```
1. Decompose problem into states (intermediate problem states)
2. Generate multiple candidate thoughts at each state
3. Evaluate thoughts (are they promising? leading toward goal?)
4. Select best thoughts to explore further
5. Prune bad branches (don't explore low-scoring paths)
6. Repeat until goal found or budget exhausted
```

This is not random sampling—it's **structured search** guided by heuristics.

---

## Key Ideas & Algorithm

### Tree-of-Thoughts Algorithm

**Step 1: Decompose Problem**
- Break problem into states: `State = (problem, decisions_so_far)`
- Example (24-game): `State = ([3, 8, 3, 8], [decisions])` → `([11, 3, 8], [3+8])`

**Step 2: Generate Candidates**
- At each state, model generates K possible next thoughts
- Thoughts are atomic steps: "3 + 8", "8 - 3", "3 * 8", etc.

**Step 3: Evaluate Thoughts**
- Score each thought (is it promising?)
- Methods: classifier, value function, or LLM evaluation
- Example: "Does 3 + 8 = 11 get us closer to 24?"

**Step 4: Select and Explore**
- Choose top-B thoughts (beam search)
- For each, recursively expand to next level

**Step 5: Termination**
- If goal found: return solution
- If max depth reached: return best path found
- If all branches pruned: backtrack

### Tree Search Strategies

| Strategy | How | Best For | Trade-off |
|----------|-----|----------|-----------|
| **Breadth-First** | Explore all nodes at depth D before depth D+1 | Finding solution quickly | Memory-intensive |
| **Depth-First** | Explore one branch fully, backtrack | Limited memory | May miss better paths |
| **Beam Search** | Keep top-B nodes per level | Balanced exploration | Risk of pruning good branches |
| **A* Search** | Use heuristic to guide exploration | Efficient search | Requires good heuristic |

### Scoring Functions

**How to evaluate whether a thought is promising?**

**Method 1: Classifier**
```
thought_score = classifier(state, thought) 
→ outputs confidence that thought leads to solution
```

**Method 2: Value Function**
```
thought_score = value_function(state + thought)
→ estimates how close to goal
```

**Method 3: LLM Self-Evaluation**
```
Prompt: "Is this thought helpful? Rate 1-10."
thought_score = LLM(state, thought)
```

**Tradeoff:**
- Classifier: Fast, learned from data
- Value function: Principled, learns heuristic
- LLM evaluation: Flexible, but slow and expensive

### Example: 24-Game with Tree of Thoughts

```
Tree Depth 1:
  3 + 8 = 11  (score: 7/10, somewhat close to 24)
  3 * 8 = 24  (score: 10/10, GOAL!)
  8 / 3 = 2.67 (score: 3/10, not helpful)
  
→ Keep top-2 (3+8, 3*8)

From "3*8=24":
  Found goal! Return [3*8]

From "3+8=11", State=[11,3,8]:
Tree Depth 2:
  11 + 3 = 14  (score: 5/10)
  11 * 3 = 33  (score: 2/10)
  11 + 8 = 19  (score: 6/10)
  
Continue exploring...
```

### Search Budget and Trade-offs

**Budget constraints:**
- Generate K candidates per node
- Keep B best candidates (beam width)
- Search depth D
- Total nodes evaluated: ≈ B^D

**Cost analysis:**
- Low B, low D: Fast but may miss solutions
- High B, high D: Accurate but expensive
- Typical: B=3-5, D=3-5, K=5-10

| Config | Nodes | Cost | Success |
|--------|-------|------|---------|
| Greedy (K=1, D=1) | 1 | Very low | 10% |
| Light (B=2, D=3, K=3) | ~8 | Low | 40% |
| Medium (B=3, D=4, K=5) | ~81 | Medium | 70% |
| Heavy (B=5, D=5, K=10) | ~3125 | High | 85% |

---

## Architecture & Trade-offs

### Tree of Thoughts vs. Alternatives

| Method | Explores Multiple Paths | Backtracks | Heuristic-Guided | Cost |
|--------|-------------------------|------------|------------------|------|
| **CoT** | No (single chain) | No | No | Very low |
| **Self-Consistency** | Yes (multiple chains) | No | No | Medium |
| **Tree of Thoughts** | Yes (tree structure) | Yes | Yes | High |
| **Reinforcement Learning** | Yes (exploration) | Implicit | Implicit | Very high |

### Pruning Strategies

**Problem:** Exploring entire tree is expensive (exponential growth)

**Solution: Prune bad branches**

**Strategy 1: Beam Search**
```
Keep top-B candidates per level
Discard bottom (K-B) candidates
```

**Pros:** Simple, effective
**Cons:** May prune good branch if heuristic is wrong

**Strategy 2: Adaptive Pruning**
```
If (node_score < threshold):
    prune this branch
Else:
    expand further
```

**Pros:** Flexible, can adjust threshold
**Cons:** Requires good threshold tuning

**Strategy 3: Early Stopping**
```
If (goal found):
    stop expansion
Else if (depth > max_depth):
    stop expansion
Else if (budget exhausted):
    stop expansion
```

### Depth vs Breadth Trade-off

**Shallow, Broad Tree:**
```
        Root
        / | \
       A  B  C
      / \ / \ / \
```
- Explores many first options (high breadth)
- Doesn't think deeply (low depth)
- Good for finding quick solutions

**Deep, Narrow Tree:**
```
        Root
        |
        A
        |
        B
        |
        C
```
- Commits to early choices (low breadth)
- Thinks deeply about each path (high depth)
- Good for verification

**Balanced Tree:**
```
        Root
       / | \
      A  B  C
      |  |  |
      A1 B1 C1
```
- Explores multiple options at each level
- Commits to best options (medium breadth)
- Thinks moderately deeply (medium depth)

---

## Interview Q&A

**Q: Why is Tree of Thoughts better than just generating multiple CoT chains (self-consistency)?**

A: Self-consistency generates N independent chains and votes on answer—no structure. Tree of Thoughts generates K options at each step, evaluates and prunes, then expands only promising branches. ToT is more efficient: for 24-game, self-consistency tries 100 random chains and succeeds 4% of the time; ToT with beam search succeeds 74% of the time with fewer evaluations. ToT exploits problem structure; self-consistency is brute force.

**Q: How do you design the scoring function in Tree of Thoughts?**

A: Several options. Simplest: use an LLM to rate thoughts ("Rate if this helps 1-10"). But this is expensive (N ratings per node). Better: learn a lightweight classifier on examples: does this thought lead toward solution? For standardized problems (24-game, code), you can use domain-specific heuristics (is number closer to target?). The key: scoring function must be fast to evaluate many options quickly.

**Q: What's the budget allocation strategy? When do you increase beam width vs. depth?**

A: General heuristic: allocate budget proportionally to problem hardness. For easy problems: low B, low D (e.g., B=2, D=2). For hard problems: higher B or D depending on structure. If the problem requires deep planning (many sequential decisions), increase D. If the problem has many options per step, increase B. In practice, measure: if success rate improves with wider beam, increase B; if it helps to go deeper, increase D.

**Q: Can Tree of Thoughts guarantee finding the solution?**

A: Only if you explore the entire tree (exhaustive search). With pruning, you sacrifice completeness for efficiency. Good heuristics make pruning work: avoid cutting branches that lead to solutions. Bad heuristics fail. The art is designing heuristics that prune useless branches but keep promising ones. This is why ToT works best on problems where you can quickly evaluate if a path is promising.

**Q: How does Tree of Thoughts compare to traditional search algorithms like A*?**

A: Very similar! ToT is essentially A* where the heuristic is learned by the LLM. A* uses domain-specific heuristics (e.g., Manhattan distance for pathfinding); ToT uses LLM-based heuristics. ToT's advantage: flexible, works on any problem the LLM understands. A* advantage: optimal with admissible heuristic. In practice, ToT is practical for problems where traditional A* fails (e.g., abstract reasoning, code generation).

**Q: In production, how do you avoid ToT becoming too expensive?**

A: Multiple strategies: (1) Use smaller models for evaluation (use GPT-3.5 to evaluate, GPT-4 to generate only final answer). (2) Cache evaluations (same thought scored twice → use cached score). (3) Use lighter scoring (e.g., yes/no classifier instead of 10-point scale). (4) Limit depth/breadth (B=2-3, D=3-4 is usually sufficient). (5) Only use ToT for hard problems—use CoT for easy ones.

**Q: How do you know if ToT is working? What metrics should you monitor?**

A: Track: (1) Success rate (does ToT solve the problem?). (2) Nodes explored (how many thought evaluations?). (3) Average path length (how many steps to solution?). (4) Early termination rate (how often do you find solution before max depth?). In practice, ToT should improve success rate 3-10x over CoT, at cost of 5-20x more evaluation. If this trade-off isn't there, your heuristic is bad.

---

## Best Practices

- **Design problems as state spaces first:** Before implementing ToT, formalize your problem as state transitions. What's a state? What's a thought? What's the goal? Clear problem decomposition makes ToT effective.

- **Scoring function is critical:** Spend time on good heuristics. A 10% improvement in scoring accuracy can 2x your success rate. Use validation data to tune scoring thresholds.

- **Start with moderate budget:** B=3-5, D=3-5. Measure success rate. If still low, increase budget; if too slow, decrease. Find your sweet spot.

- **Use domain-specific heuristics when possible:** Generic scoring is slow. If you know problem structure (24-game: "numbers closer to 24 are better"), use it. Much faster than LLM evaluation.

- **Monitor for infinite loops:** Even with pruning, greedy expansion can loop. Add visited-state tracking: if you reach same state twice, stop expanding.

- **Combine with CoT inside nodes:** Don't just score—have the model reason about why it's promising. Multi-turn evaluation is better than single-shot scoring.

- **Cache observations:** If same thought evaluated twice (different branches), cache the result. Avoid redundant evaluations.

- **Visualize tree during debugging:** Print the explored tree. You'll spot issues (e.g., heuristic cutting good branches) quickly.

---

## Common Pitfalls

- **Mistake: Poor heuristic that prunes solutions.** If your scoring function ranks the correct path low, ToT prunes it and fails. Validate your heuristic on known solutions first.

- **Mistake: No early termination.** If you find a valid solution at depth 2, keep exploring deeper. Wastes computation. Add goal-check: if solution found, return immediately.

- **Mistake: Unlimited breadth.** Generating K=100 options per node is too expensive. Stick to K=3-10, select top-B (B=2-5). Exponential growth kills performance.

- **Mistake: Greedy beam search cuts off all progress.** If B=1 (only keep 1 best option), you've reverted to greedy. Need B≥2 to allow backtracking.

- **Mistake: Scoring function doesn't correlate with success.** If scoring 0/10 and 9/10 paths have same success rate, scoring is useless. Validate that scores correlate with outcome.

- **Mistake: No visited-state tracking.** In some problems, different thought sequences lead to same state. Explore both: wasteful. Track visited states, reuse evaluations.

- **Mistake: Unbounded tree explosion.** Without max-depth or max-nodes limits, tree grows exponentially. Always set limits.

---

## Code Examples

### Example 1: Basic Tree of Thoughts for 24-Game

```python
import heapq
from typing import List, Tuple, Set

class TreeOfThoughtsAgent:
    """Solve 24-game using tree search."""
    
    def __init__(self, max_depth: int = 5, beam_width: int = 3):
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.visited = set()
    
    def solve(self, numbers: List[float], target: float = 24) -> Tuple[bool, List[str]]:
        """Find arithmetic sequence to make target."""
        initial_state = (tuple(sorted(numbers)), [])
        
        # Priority queue: (score, depth, state, path)
        queue = [(-self.score_state(numbers, target), 0, initial_state, [])]
        
        while queue:
            neg_score, depth, (nums, _), path = heapq.heappop(queue)
            
            # Goal check
            if len(nums) == 1 and abs(nums[0] - target) < 1e-6:
                return True, path
            
            # Depth limit
            if depth >= self.max_depth:
                continue
            
            # State visited?
            state_key = tuple(sorted(nums))
            if state_key in self.visited:
                continue
            self.visited.add(state_key)
            
            # Generate candidates
            candidates = self.generate_thoughts(nums, target)
            
            # Select top beam_width
            top_candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:self.beam_width]
            
            for score, new_nums, thought in top_candidates:
                new_state = (tuple(sorted(new_nums)), [])
                new_path = path + [thought]
                new_score = -self.score_state(new_nums, target)
                
                heapq.heappush(queue, (new_score, depth + 1, new_state, new_path))
        
        return False, []
    
    def generate_thoughts(self, numbers: List[float], target: float) -> List[Tuple[float, List[float], str]]:
        """Generate candidate operations."""
        candidates = []
        
        # Try all pairs of numbers
        for i in range(len(numbers)):
            for j in range(len(numbers)):
                if i == j:
                    continue
                
                a, b = numbers[i], numbers[j]
                remaining = [numbers[k] for k in range(len(numbers)) if k != i and k != j]
                
                # Try operations
                operations = [
                    (a + b, f"{a}+{b}={a+b}"),
                    (a - b, f"{a}-{b}={a-b}"),
                    (a * b, f"{a}*{b}={a*b}"),
                ]
                if b != 0:
                    operations.append((a / b, f"{a}/{b}={a/b:.2f}"))
                
                for result, thought in operations:
                    new_nums = remaining + [result]
                    score = self.score_state(new_nums, target)
                    candidates.append((score, new_nums, thought))
        
        return candidates
    
    def score_state(self, numbers: List[float], target: float) -> float:
        """Heuristic score: how close is closest number to target?"""
        if not numbers:
            return -float('inf')
        closest = min(abs(n - target) for n in numbers)
        return 1.0 / (1.0 + closest)  # Score in [0, 1]

# Usage
agent = TreeOfThoughtsAgent(max_depth=5, beam_width=3)
numbers = [3, 8, 3, 8]
success, path = agent.solve(numbers)
print(f"Success: {success}")
print(f"Path: {path}")
```

### Example 2: Beam Search with Scoring

```python
class BeamSearchToT:
    """General-purpose tree search with beam width."""
    
    def __init__(self, beam_width: int = 3, max_depth: int = 5):
        self.beam_width = beam_width
        self.max_depth = max_depth
    
    def search(self, initial_state, goal_fn, expand_fn, score_fn):
        """
        General tree search.
        - initial_state: starting point
        - goal_fn(state): checks if state is goal
        - expand_fn(state): returns list of (new_state, action)
        - score_fn(state): returns float (higher=better)
        """
        current_beam = [(initial_state, [])]
        
        for depth in range(self.max_depth):
            next_beam = []
            
            for state, path in current_beam:
                # Goal check
                if goal_fn(state):
                    return True, path
                
                # Expand state
                for next_state, action in expand_fn(state):
                    next_beam.append((next_state, path + [action]))
            
            # Score and select top-K
            next_beam = sorted(
                next_beam,
                key=lambda x: score_fn(x[0]),
                reverse=True
            )[:self.beam_width]
            
            if not next_beam:
                break
            
            current_beam = next_beam
        
        # Return best path found
        if current_beam:
            return False, current_beam[0][1]
        return False, []
```

### Example 3: Adaptive Pruning with Thresholds

```python
class AdaptiveTreeSearch:
    """Tree search with dynamic pruning threshold."""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
    
    def search(self, initial_state, goal_fn, expand_fn, score_fn, 
               min_score: float = 0.3):
        """
        Search with adaptive pruning.
        - min_score: prune if score < min_score
        """
        queue = [(initial_state, [], 0)]
        best_solution = None
        
        while queue:
            state, path, depth = queue.pop(0)
            
            # Goal check
            if goal_fn(state):
                return True, path
            
            # Depth limit
            if depth >= self.max_depth:
                continue
            
            # Expand
            candidates = expand_fn(state)
            
            for next_state, action in candidates:
                score = score_fn(next_state)
                
                # Pruning
                if score < min_score:
                    continue  # Prune low-scoring branches
                
                queue.append((next_state, path + [action], depth + 1))
        
        return False, best_solution or []
```

---

## Related Concepts

- [Chain-of-Thought Prompting](./01-chain-of-thought.md) — Linear reasoning (base case for ToT)
- [ReAct: Synergizing Reasoning and Acting](./02-react.md) — Reasoning with tool use
- [agentic-ai/concepts/XX-planning](../../agentic-ai/concepts/XX-planning.md) — Multi-step planning
- [agentic-ai/concepts/XX-search-algorithms](../../agentic-ai/concepts/XX-search-algorithms.md) — Classical search methods

---

**Last Updated:** 2026-05-31
