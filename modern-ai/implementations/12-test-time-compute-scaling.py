"""
Auto-generated from 12-test-time-compute-scaling.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Test-Time Compute Scaling
# ## Learning Objectives
# 1. Understand how test-time compute (reasoning) improves model performance
# 2. Implement best-of-N, beam search, and iterative refinement strategies
# 3. Measure quality-vs-compute trade-offs and find optimal budgets
# 4. Apply test-time scaling to real generation tasks
# ======================================================================

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Device setup for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'CUDA available: {torch.cuda.is_available()}')


# ======================================================================
# ## Level 1: Basic Test-Time Compute (Best-of-N Sampling)
# Implement best-of-N sampling and beam search from scratch with explicit compute budget tracking.
# ======================================================================

# Level 1: Basic best-of-N sampling with mock scoring
@dataclass
class GenerationResult:
    text: str
    score: float
    tokens: int
    compute_cost: float

def mock_language_model(prompt: str, max_tokens: int = 50) -> str:
    """
    Mock LLM that generates text deterministically based on seed.
    Returns: Generated text
    """
    np.random.seed(hash(prompt) % 2**32)
    tokens = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 
              'hello', 'world', 'ai', 'is', 'great', 'learning', 'model']
    generated = ' '.join(np.random.choice(tokens, size=np.random.randint(5, max_tokens)))
    return f"{prompt} {generated}"

def mock_scorer(text: str) -> float:
    """
    Mock quality scorer. Returns score in [0, 1].
    Scores based on text length and word variety.
    """
    words = text.lower().split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    length_bonus = min(len(words) / 50, 1.0)  # Prefer longer text
    return 0.6 * unique_ratio + 0.4 * length_bonus

def best_of_n_sampling(prompt: str, n: int, max_tokens: int = 50) -> GenerationResult:
    """
    Best-of-N sampling: generate N candidates, pick the best.
    Compute cost: O(N) - proportional to number of samples.
    """
    candidates = []
    total_cost = 0.0
    
    for i in range(n):
        # Generate one candidate
        text = mock_language_model(prompt, max_tokens)
        score = mock_scorer(text)
        candidates.append((text, score))
        # Mock compute cost: 1 unit per generation
        total_cost += 1.0
    
    # Select best
    best_text, best_score = max(candidates, key=lambda x: x[1])
    token_count = len(best_text.split())
    
    return GenerationResult(text=best_text, score=best_score, tokens=token_count, 
                           compute_cost=total_cost)

def beam_search(prompt: str, beam_width: int, max_tokens: int = 50) -> GenerationResult:
    """
    Beam search: maintain top-K candidates throughout generation.
    More efficient than best-of-N for large budgets.
    """
    # Simplified: maintain beam_width candidates
    candidates = []
    total_cost = 0.0
    
    # Generate beam_width candidates
    for i in range(beam_width):
        text = mock_language_model(prompt, max_tokens)
        score = mock_scorer(text)
        candidates.append((text, score))
        total_cost += 1.0
    
    # For each new token position, expand and prune
    for step in range(2):  # Simplified: 2 expansion steps
        expanded = []
        for text, score in candidates:
            # Expand with variations
            new_text = text + ' extended'
            new_score = mock_scorer(new_text)
            expanded.append((new_text, new_score))
            total_cost += 0.5  # Smaller cost for expansion
        
        # Keep top beam_width
        candidates = sorted(expanded, key=lambda x: x[1], reverse=True)[:beam_width]
    
    best_text, best_score = candidates[0]
    token_count = len(best_text.split())
    
    return GenerationResult(text=best_text, score=best_score, tokens=token_count,
                           compute_cost=total_cost)

# Test both strategies
prompt = "The future of AI is"
print(f'Prompt: "{prompt}"\n')

# Best-of-N with N=5
result_bon = best_of_n_sampling(prompt, n=5)
print(f'✅ Best-of-N (N=5):')
print(f'  Text: {result_bon.text[:80]}...')
print(f'  Score: {result_bon.score:.3f}')
print(f'  Compute cost: {result_bon.compute_cost:.1f}')

# Beam search with beam_width=3
result_beam = beam_search(prompt, beam_width=3)
print(f'\n✅ Beam Search (width=3):')
print(f'  Text: {result_beam.text[:80]}...')
print(f'  Score: {result_beam.score:.3f}')
print(f'  Compute cost: {result_beam.compute_cost:.1f}')


# ======================================================================
# ## Level 2: Advanced Test-Time Compute with Iterative Refinement
# Add iterative refinement with explicit compute budget tracking, early stopping, and quality vs cost trade-offs.
# ======================================================================

class IterativeRefinementEngine:
    """
    Test-time compute via iterative refinement.
    Generate → Critique → Refine → Score, repeat until budget exhausted.
    """
    def __init__(self, budget: float = 10.0, improvement_threshold: float = 0.01):
        self.budget = budget
        self.remaining_budget = budget
        self.improvement_threshold = improvement_threshold
        self.history = []
    
    def generate(self, prompt: str) -> str:
        """Generate initial text"""
        return mock_language_model(prompt, max_tokens=50)
    
    def critique(self, text: str) -> Dict[str, float]:
        """Rule-based critique: identify weaknesses"""
        # Mock critique with scores for different aspects
        words = text.lower().split()
        word_count = len(words)
        
        critique = {
            'clarity': min(word_count / 30, 1.0),  # Longer = clearer (mock)
            'diversity': len(set(words)) / max(word_count, 1),  # Word variety
            'relevance': 0.7 + 0.3 * np.sin(hash(text) % 100 / 100),  # Random
        }
        critique['overall'] = np.mean(list(critique.values()))
        return critique
    
    def refine(self, text: str, critique: Dict) -> str:
        """Refine based on critique"""
        # Add tokens addressing weaknesses
        refinements = []
        if critique['clarity'] < 0.6:
            refinements.append('clearly')
        if critique['diversity'] < 0.5:
            refinements.append('precisely')
        
        refined = text + ' ' + ' '.join(refinements) if refinements else text + ' continued'
        return refined
    
    def refine_iteratively(self, prompt: str) -> GenerationResult:
        """Iteratively refine with budget tracking"""
        # Initial generation
        text = self.generate(prompt)
        current_score = mock_scorer(text)
        self.history = [(text, current_score, 1.0)]  # (text, score, cumulative_cost)
        total_cost = 1.0
        
        iteration = 0
        while total_cost < self.budget and iteration < 20:
            # Critique
            critique = self.critique(text)
            critique_cost = 0.3  # Critique is cheaper than generation
            total_cost += critique_cost
            
            # Refine
            text = self.refine(text, critique)
            refine_cost = 0.7  # Refinement (cheap, no new generation)
            total_cost += refine_cost
            
            # Re-score
            new_score = mock_scorer(text)
            improvement = new_score - current_score
            
            self.history.append((text, new_score, total_cost))
            
            # Early stopping if improvement plateaus
            if improvement < self.improvement_threshold and total_cost > self.budget * 0.5:
                break
            
            current_score = new_score
            iteration += 1
        
        token_count = len(text.split())
        return GenerationResult(text=text, score=current_score, tokens=token_count,
                               compute_cost=total_cost)

# Test iterative refinement with different budgets
prompt = "The future of AI is"
budgets = [5.0, 10.0, 15.0, 20.0]
results = []

for budget in budgets:
    engine = IterativeRefinementEngine(budget=budget, improvement_threshold=0.01)
    result = engine.refine_iteratively(prompt)
    results.append((budget, result))
    
    print(f'\n✅ Budget: {budget:.1f}')
    print(f'  Final score: {result.score:.3f}')
    print(f'  Compute cost: {result.compute_cost:.2f}')
    print(f'  Iterations: {len(engine.history) - 1}')
    print(f'  Text: {result.text[:70]}...')

print(f'\n✅ Test-time compute scaling complete')


# ======================================================================
# ## Real-World Example 1: GPT-2 Text Generation with Best-of-N
# ======================================================================

# Real-world: Use transformers library GPT-2 with best-of-N sampling
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    def gpt2_generate_one(prompt: str, max_length: int = 50) -> str:
        """Generate one sequence from GPT-2"""
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,  # Add some randomness
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text
    
    def gpt2_scorer(text: str) -> float:
        """Score GPT-2 output: perplexity-based (higher = better)"""
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs[0]
            # Convert loss to score (lower loss = higher score)
            score = 1.0 / (1.0 + loss.item())
        
        return score
    
    def gpt2_best_of_n(prompt: str, n: int = 3) -> Tuple:
        """Best-of-N with real GPT-2"""
        candidates = []
        total_time = 0
        
        for i in range(n):
            start = time.time()
            text = gpt2_generate_one(prompt)
            score = gpt2_scorer(text)
            elapsed = time.time() - start
            
            candidates.append((text, score))
            total_time += elapsed
        
        best_text, best_score = max(candidates, key=lambda x: x[1])
        return best_text, best_score, total_time, candidates
    
    # Test with different N values
    prompt = "The future of artificial intelligence is"
    ns = [1, 2, 4]
    gpt2_results = []
    
    for n in ns:
        print(f'\nGenerating {n} candidates...')
        best_text, best_score, total_time, candidates = gpt2_best_of_n(prompt, n=n)
        
        print(f'✅ Best-of-{n}:')
        print(f'  Best score: {best_score:.4f}')
        print(f'  Total time: {total_time:.2f}s')
        print(f'  Output: {best_text[:80]}...')
        
        gpt2_results.append({
            'n': n,
            'score': best_score,
            'time': total_time,
            'text': best_text
        })
    
    print(f'\n✅ GPT-2 best-of-N testing complete')

except Exception as e:
    print(f'Note: GPT-2 example requires transformers library: {e}')
    print('Using mock scorer instead')
    gpt2_results = []


# ======================================================================
# ## Real-World Example 2: Beam Search with Compute Budget
# ======================================================================

class BudgetedBeamSearch:
    """
    Beam search with explicit compute budget (tokens or time).
    Stops generation when budget exhausted or target reached.
    """
    def __init__(self, budget_tokens: int = 500, beam_width: int = 4):
        self.budget_tokens = budget_tokens
        self.beam_width = beam_width
    
    def search(self, prompt: str) -> Dict:
        """Perform beam search with budget"""
        # Simplified beam search simulation
        candidates = []  # (text, score, tokens_used)
        total_tokens = 0
        
        # Initial generation for each beam
        for b in range(self.beam_width):
            text = mock_language_model(prompt, max_tokens=30)
            score = mock_scorer(text)
            tokens = len(text.split())
            candidates.append((text, score, tokens))
            total_tokens += tokens
        
        # Iteratively expand
        iteration = 0
        while total_tokens < self.budget_tokens and iteration < 5:
            expanded = []
            
            for text, score, tokens in candidates:
                # Generate continuations (simplified)
                new_text = text + ' extended'
                new_score = mock_scorer(new_text)
                new_tokens = tokens + 1  # 1 additional token
                
                if total_tokens + new_tokens <= self.budget_tokens:
                    expanded.append((new_text, new_score, new_tokens))
                    total_tokens += 1
            
            # Keep top beam_width
            candidates = sorted(expanded, key=lambda x: x[1], reverse=True)[:self.beam_width]
            iteration += 1
            
            if not candidates:
                break
        
        # Return best
        best_text, best_score, best_tokens = max(candidates, key=lambda x: x[1])
        
        return {
            'text': best_text,
            'score': best_score,
            'tokens_used': total_tokens,
            'budget': self.budget_tokens,
            'efficiency': best_score / (total_tokens + 1)
        }

# Test budgeted beam search
prompt = "The future of AI is"
budgets = [200, 400, 800]

beam_results = []
for budget in budgets:
    searcher = BudgetedBeamSearch(budget_tokens=budget, beam_width=4)
    result = searcher.search(prompt)
    beam_results.append(result)
    
    print(f'\n✅ Budget: {budget} tokens')
    print(f'  Score: {result["score"]:.3f}')
    print(f'  Tokens used: {result["tokens_used"]}')
    print(f'  Efficiency: {result["efficiency"]:.4f}')
    print(f'  Text: {result["text"][:70]}...')

print(f'\n✅ Budgeted beam search complete')


# ======================================================================
# ## Real-World Example 3: Chain-of-Thought with Budget Allocation
# ======================================================================

class ChainOfThoughtReasoner:
    """
    Generate reasoning chain with explicit budget allocation.
    Allocate budget to: thinking → exploration → verification
    """
    def __init__(self, total_budget: float = 20.0):
        self.total_budget = total_budget
        # Budget allocation: thinking (50%), exploration (30%), verification (20%)
        self.thinking_budget = 0.5 * total_budget
        self.exploration_budget = 0.3 * total_budget
        self.verification_budget = 0.2 * total_budget
    
    def thinking_phase(self, prompt: str) -> Dict:
        """Generate reasoning/thinking"""
        # Multiple thinking chains
        chains = []
        budget_per_chain = self.thinking_budget / 3
        
        for i in range(3):
            thought = f"Thinking path {i+1}: {mock_language_model(prompt, max_tokens=20)}"
            chains.append({
                'thought': thought,
                'cost': budget_per_chain
            })
        
        return {'chains': chains, 'total_cost': self.thinking_budget}
    
    def exploration_phase(self, thinking: Dict) -> Dict:
        """Explore variations based on thinking"""
        explorations = []
        budget_per_exploration = self.exploration_budget / 2
        
        for i in range(2):
            exploration = f"Exploration {i+1}: extending from thinking"
            explorations.append({
                'exploration': exploration,
                'cost': budget_per_exploration
            })
        
        return {'explorations': explorations, 'total_cost': self.exploration_budget}
    
    def verification_phase(self, explorations: Dict) -> Dict:
        """Verify and score final answer"""
        final_answer = "The answer is: high quality reasoning"
        verification_score = 0.85  # Mock score
        
        return {
            'answer': final_answer,
            'score': verification_score,
            'cost': self.verification_budget
        }
    
    def reason(self, prompt: str) -> Dict:
        """Full reasoning chain"""
        thinking = self.thinking_phase(prompt)
        exploration = self.exploration_phase(thinking)
        verification = self.verification_phase(exploration)
        
        return {
            'prompt': prompt,
            'thinking': thinking,
            'exploration': exploration,
            'verification': verification,
            'total_cost': (thinking['total_cost'] + exploration['total_cost'] + 
                           verification['cost'])
        }

# Test chain-of-thought with different budgets
cot_results = []
budgets = [10.0, 15.0, 20.0, 30.0]

for budget in budgets:
    prompt = "What is the best approach to scaling LLMs?"
    reasoner = ChainOfThoughtReasoner(total_budget=budget)
    result = reasoner.reason(prompt)
    
    print(f'\n✅ Budget: {budget:.1f}')
    print(f'  Thinking budget: {reasoner.thinking_budget:.1f}')
    print(f'  Exploration budget: {reasoner.exploration_budget:.1f}')
    print(f'  Verification budget: {reasoner.verification_budget:.1f}')
    print(f'  Final score: {result["verification"]["score"]:.3f}')
    print(f'  Answer: {result["verification"]["answer"]}')
    
    cot_results.append({
        'budget': budget,
        'score': result['verification']['score'],
        'cost': result['total_cost']
    })

print(f'\n✅ Chain-of-thought reasoning complete')


# ======================================================================
# ## Comparison: Quality vs Compute Cost
# ======================================================================

# Create quality vs compute visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Iterative refinement - score vs cost
if results:
    budgets_list = [b for b, _ in results]
    scores_list = [r.score for _, r in results]
    
    axes[0, 0].plot(budgets_list, scores_list, 'o-', linewidth=2, markersize=8, color='#2ca02c')
    axes[0, 0].fill_between(budgets_list, scores_list, alpha=0.2, color='#2ca02c')
    axes[0, 0].set_xlabel('Compute Budget', fontsize=10)
    axes[0, 0].set_ylabel('Quality Score', fontsize=10)
    axes[0, 0].set_title('Iterative Refinement: Score vs Budget', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Best-of-N comparison
if gpt2_results:
    ns = [r['n'] for r in gpt2_results]
    scores = [r['score'] for r in gpt2_results]
    times = [r['time'] for r in gpt2_results]
    
    ax2_2 = axes[0, 1]
    ax2_2_twin = ax2_2.twinx()
    
    ax2_2.plot(ns, scores, 'o-', linewidth=2, markersize=8, color='#1f77b4', label='Score')
    ax2_2_twin.plot(ns, times, 's--', linewidth=2, markersize=8, color='#ff7f0e', label='Time')
    
    ax2_2.set_xlabel('N (number of samples)', fontsize=10)
    ax2_2.set_ylabel('Quality Score', fontsize=10, color='#1f77b4')
    ax2_2_twin.set_ylabel('Time (seconds)', fontsize=10, color='#ff7f0e')
    ax2_2.set_title('Best-of-N: Quality vs Compute Time', fontsize=11, fontweight='bold')
    ax2_2.grid(True, alpha=0.3)
    ax2_2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2_2_twin.tick_params(axis='y', labelcolor='#ff7f0e')
else:
    axes[0, 1].text(0.5, 0.5, 'GPT-2 results not available\n(requires transformers)', 
                    ha='center', va='center', fontsize=10)

# Plot 3: Beam search efficiency
if beam_results:
    budgets_beam = [r['budget'] for r in beam_results]
    scores_beam = [r['score'] for r in beam_results]
    efficiency = [r['efficiency'] for r in beam_results]
    
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    ax3.bar(range(len(beam_results)), scores_beam, alpha=0.7, color='#2ca02c', label='Score')
    ax3_twin.plot(range(len(beam_results)), efficiency, 'ro-', linewidth=2, markersize=8, label='Efficiency')
    
    ax3.set_xlabel('Budget (tokens)', fontsize=10)
    ax3.set_ylabel('Quality Score', fontsize=10)
    ax3_twin.set_ylabel('Efficiency', fontsize=10)
    ax3.set_xticks(range(len(beam_results)))
    ax3.set_xticklabels([str(b) for b in budgets_beam])
    ax3.set_title('Beam Search: Quality and Efficiency', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Budget allocation (chain-of-thought)
if cot_results:
    budget_vals = [r['budget'] for r in cot_results]
    score_vals = [r['score'] for r in cot_results]
    
    axes[1, 1].scatter(budget_vals, score_vals, s=200, alpha=0.6, color='#d62728')
    axes[1, 1].plot(budget_vals, score_vals, '--', linewidth=1.5, color='#d62728')
    axes[1, 1].set_xlabel('Total Budget', fontsize=10)
    axes[1, 1].set_ylabel('Quality Score', fontsize=10)
    axes[1, 1].set_title('Chain-of-Thought: Score vs Budget', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Annotate points
    for b, s in zip(budget_vals, score_vals):
        axes[1, 1].annotate(f'{b:.0f}', (b, s), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('/tmp/test_time_compute_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print('✅ Comparison visualization saved')


# ======================================================================
# ## Key Takeaways
# ### Core Concept
# Test-time compute scaling (like o1 models) allocates extra computation during inference to improve quality. Key insight: spending more time reasoning at test time can dramatically improve answers without retraining.
# ### Strategies and Trade-offs
# | Strategy | Quality Gain | Compute Cost | Best For |
# |----------|---|---|---|
# | Best-of-N | Moderate (30-50%) | Linear (O(N)) | Quick quality boost |
# | Beam Search | High (50-80%) | Sub-linear | Structured problems |
# | Iterative Refinement | High (40-70%) | Controllable | Polishing answers |
# | Chain-of-Thought | Very High (50-100%) | Moderate | Complex reasoning |
# ### When to Use Each Approach
# - **Best-of-N:** Simple tasks, small N (3-5), quick inference needed
# - **Beam Search:** Constrained generation (code, SQL), deterministic problems
# - **Iterative Refinement:** Output improvement (fixing errors, polishing)
# - **Chain-of-Thought:** Math, logic, multi-step reasoning
# ### Common Pitfalls
# - **Diminishing returns:** Quality improvement plateaus after modest budget increase (e.g., 2-5x compute)
# - **Selection bias:** Choosing best from bad candidates doesn't help. Requires good baseline.
# - **Budget wastefulness:** Allocating budget evenly across phases is suboptimal. Profile and rebalance.
# ### Practical Guidance
# - Start with best-of-N (N=3-5) for quick wins
# - Measure quality-vs-compute Pareto frontier for your task
# - Combine multiple strategies: best-of-N at start, refinement at end
# - For production: cache intermediate reasoning to avoid recomputation
# ======================================================================

# ======================================================================
# ## Exercises
# 1. **Optimize Budget Allocation:** In the chain-of-thought example, adjust thinking/exploration/verification split (e.g., 60/20/20 vs 40/40/20) and measure quality changes.
# 2. **Hybrid Strategy:** Combine best-of-N (sample 3 initial candidates) + iterative refinement on the best. Measure total quality vs pure best-of-5.
# 3. **Measure Diminishing Returns:** Plot quality (y-axis) vs cumulative budget (x-axis) and find where marginal gains drop below 1% per unit budget.
# ======================================================================
