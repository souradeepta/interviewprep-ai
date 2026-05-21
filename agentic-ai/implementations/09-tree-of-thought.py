"""
Auto-generated from 09-tree-of-thought.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Tree of Thought Reasoning
# Learning objectives:
# - Understand how ToT explores multiple solution paths
# - Implement branch generation and evaluation
# ======================================================================

import anthropic
from typing import List, Dict, Tuple
import json

client = anthropic.Anthropic()
print("✓ Setup complete")


# ======================================================================
# ## Level 1: Generate and Evaluate Branches
# ======================================================================

def generate_branches(client, problem: str, num_branches: int = 3) -> List[str]:
    """Generate multiple solution approaches"""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Problem: {problem}\n\nSuggest {num_branches} different approaches. List only approaches, one per line."
        }]
    )
    approaches = response.content[0].text.strip().split('\n')
    return approaches[:num_branches]

def evaluate_branch(client, problem: str, approach: str) -> float:
    """Score a branch's promise (0-10)"""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": f"Problem: {problem}\nApproach: {approach}\n\nRate promise (0-10). Answer with just number."
        }]
    )
    try:
        return float(response.content[0].text.strip())
    except:
        return 5.0

# Example
problem = "How to optimize Python code for speed?"
branches = generate_branches(client, problem, num_branches=3)
print(f"Branches for: {problem}")
for i, branch in enumerate(branches, 1):
    score = evaluate_branch(client, problem, branch)
    print(f"  {i}. {branch[:50]}... (score: {score}/10)")


# ======================================================================
# ## Level 2: Tree Exploration with Pruning
# ======================================================================

class TreeOfThoughtSolver:
    """Tree of Thought with branch pruning"""
    
    def __init__(self, client, branching_factor: int = 3, prune_to: int = 2, max_depth: int = 3):
        self.client = client
        self.branching_factor = branching_factor
        self.prune_to = prune_to
        self.max_depth = max_depth
        self.explored_count = 0
    
    def solve(self, problem: str) -> str:
        solutions = []
        
        def dfs(state: str, depth: int) -> None:
            if depth >= self.max_depth:
                solutions.append(state)
                return
            
            # Generate branches
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": f"Current: {state}\n\nNext {self.branching_factor} steps?"
                }]
            )
            branches = response.content[0].text.strip().split('\n')[:self.branching_factor]
            
            # Evaluate and prune
            scored = []
            for branch in branches:
                try:
                    score_resp = self.client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=20,
                        messages=[{"role": "user", "content": f"Rate: {branch} (0-10)"}]
                    )
                    score = float(score_resp.content[0].text.strip())
                    scored.append((score, branch))
                except:
                    scored.append((5.0, branch))
            
            scored.sort(reverse=True)
            
            # Explore top branches
            for score, branch in scored[:self.prune_to]:
                self.explored_count += 1
                new_state = f"{state} → {branch[:30]}..."
                dfs(new_state, depth + 1)
        
        dfs(problem, 0)
        return solutions[0] if solutions else "No solution found"

# Example
solver = TreeOfThoughtSolver(client, branching_factor=3, prune_to=2, max_depth=2)
# result = solver.solve("How to solve a complex math problem?")
# print(f"Solution: {result}")
# print(f"Explored {solver.explored_count} branches")
print("ToT solver ready (skipping demo to save API calls)")


# ======================================================================
# ## Level 3: Hybrid ToT (Low Cost)
# ======================================================================

class HybridToT:
    """ToT with batch evaluation for cost efficiency"""
    
    def __init__(self, client, max_depth: int = 2):
        self.client = client
        self.max_depth = max_depth
    
    def evaluate_branches_batch(self, state: str, branches: List[str]) -> Dict[str, float]:
        """Evaluate all branches in one call"""
        branches_text = "\n".join([f"{i+1}. {b}" for i, b in enumerate(branches)])
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"State: {state}\n\nRate branches (0-10):\n{branches_text}\n\nReply: 1: X, 2: Y, 3: Z"
            }]
        )
        
        # Parse scores
        scores = {}
        for line in response.content[0].text.strip().split('\n'):
            try:
                parts = line.split(':')
                idx = int(parts[0].strip()) - 1
                score = float(parts[1].strip())
                if idx < len(branches):
                    scores[branches[idx]] = score
            except:
                pass
        
        return scores
    
    def solve(self, problem: str) -> Tuple[str, int]:
        """Solve with batched evaluation"""
        api_calls = 0
        
        # Generate initial branches
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=150,
            messages=[{"role": "user", "content": f"Problem: {problem}\n\n3 approaches?"}]
        )
        api_calls += 1
        branches = response.content[0].text.strip().split('\n')[:3]
        
        # Batch evaluate
        scores = self.evaluate_branches_batch(problem, branches)
        api_calls += 1
        
        # Keep top branch
        best_branch = max(scores.items(), key=lambda x: x[1])[0] if scores else branches[0]
        
        return best_branch, api_calls

# Example
hybrid = HybridToT(client)
# solution, calls = hybrid.solve("Design a recommendation system")
# print(f"Solution: {solution}")
# print(f"API calls: {calls} (vs ~9 for naive ToT)")
print("Hybrid ToT ready")


# ======================================================================
# ## Key Takeaways
# 1. **ToT explores alternatives** — Multiple branches per step vs single linear path
# 2. **Pruning controls cost** — Keep top 2, discard rest. Exponential explosion prevented
# 3. **Batch evaluation saves API calls** — Evaluate 3 branches in 1 call vs 3 separate calls
# ======================================================================
