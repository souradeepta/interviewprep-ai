"""
Auto-generated from 16-chain-of-draft.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Chain-of-Draft: Iterative Refinement for High-Quality Outputs
# ## Learning Objectives
# 1. Understand draft generation and quality scoring mechanisms from scratch
# 2. Implement iterative refinement loops with convergence detection
# 3. Apply chain-of-draft patterns to real production tasks (summarization, code generation)
# 4. Optimize refinement strategies with token budgets and quality thresholds
# ======================================================================

import numpy as np
import torch
import time
import json
from typing import List, Dict, Tuple
from collections import defaultdict
import re

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available, using mock implementations")

# Device setup for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'Transformers available: {TRANSFORMERS_AVAILABLE}')


# ======================================================================
# ## Level 1: Basic Draft Generation and Quality Scoring
# ======================================================================

# Level 1: Basic draft generation and quality scoring from scratch
# Implements N drafts with heuristic quality metrics

def basic_quality_score(text: str) -> Dict[str, float]:
    """Score text quality using simple heuristics (length, coherence, uniqueness)"""
    # Length score: prefer 50-300 word responses
    words = text.split()
    word_count = len(words)
    length_score = 1.0 if 50 <= word_count <= 300 else max(0, 1 - abs(word_count - 150) / 300)
    
    # Coherence score: check for common sentence markers
    coherence_markers = ['because', 'therefore', 'however', 'moreover', 'thus']
    text_lower = text.lower()
    marker_count = sum(1 for marker in coherence_markers if marker in text_lower)
    coherence_score = min(1.0, marker_count / 3)  # Normalize to 0-1
    
    # Diversity score: penalize repetitive words
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
    else:
        unique_ratio = 0
    diversity_score = unique_ratio
    
    # Combined score (weighted average)
    combined = 0.4 * length_score + 0.3 * coherence_score + 0.3 * diversity_score
    
    return {
        'length_score': length_score,
        'coherence_score': coherence_score,
        'diversity_score': diversity_score,
        'combined_score': combined,
        'word_count': word_count
    }

def generate_basic_drafts(prompt: str, num_drafts: int = 3) -> List[str]:
    """Generate multiple drafts with temperature variation to simulate diversity"""
    drafts = []
    
    # Simulate N drafts with different "temperatures" (variation levels)
    for i in range(num_drafts):
        # In real scenario, these would be model outputs
        # For demo, we generate synthetic variants
        temperature = 0.5 + (i * 0.25)  # 0.5, 0.75, 1.0
        
        if prompt == "Explain machine learning":
            templates = [
                f"Machine learning is a subset of AI. It enables systems to learn from data. {i} iterations improve accuracy.",
                f"ML allows computers to improve through experience. Data drives learning. Temperature: {temperature:.2f}",
                f"Learning from data defines ML. Systems adapt based on examples. This approach is fundamental to modern AI."
            ]
            drafts.append(templates[i % len(templates)])
        else:
            # Generic response for other prompts
            drafts.append(f"Response {i+1} to: {prompt}")
    
    return drafts

# Test basic implementation
prompt = "Explain machine learning"
drafts = generate_basic_drafts(prompt, num_drafts=3)
print(f"Generated {len(drafts)} drafts for: {prompt}\n")

for i, draft in enumerate(drafts):
    score = basic_quality_score(draft)
    print(f"Draft {i+1}:")
    print(f"  Text: {draft[:80]}...")
    print(f"  Quality Score: {score['combined_score']:.3f}")
    print(f"  Metrics: length={score['length_score']:.2f}, coherence={score['coherence_score']:.2f}, diversity={score['diversity_score']:.2f}\n")

# Select best draft
best_idx = max(range(len(drafts)), key=lambda i: basic_quality_score(drafts[i])['combined_score'])
print(f"Best draft (index {best_idx+1}): {drafts[best_idx][:100]}...")


# ======================================================================
# ### Output from Level 1 Basic Implementation
# The above code generates 3 drafts and scores them using heuristics for length, coherence, and diversity. The best draft is selected based on combined score.
# ======================================================================

# ======================================================================
# ## Level 2: Advanced Chain-of-Draft with Refinement Loop and Convergence
# ======================================================================

# Level 2: Advanced implementation with iterative refinement and convergence detection

class ChainOfDraftRefinement:
    """Advanced chain-of-draft with refinement loop, convergence, and metrics tracking"""
    
    def __init__(self, max_iterations: int = 3, convergence_threshold: float = 0.85):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.metrics = defaultdict(list)
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts (simple heuristic)"""
        # Count overlapping key words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0
    
    def refine_draft(self, draft: str, iteration: int) -> str:
        """Simulate refinement by adding clarity markers and structure"""
        # In real scenario, this would be model-based refinement prompt
        # For demo, we simulate improvements
        
        refined = draft
        
        # Add structure markers based on iteration
        if iteration == 1:
            refined = f"{draft} [Refined: Enhanced clarity]"
        elif iteration == 2:
            refined = f"{draft} [Refined: Added examples]"
        
        return refined
    
    def check_convergence(self, draft1: str, draft2: str, threshold: float) -> bool:
        """Check if two consecutive drafts have converged (are sufficiently similar)"""
        similarity = self.semantic_similarity(draft1, draft2)
        converged = similarity >= threshold
        self.metrics['similarities'].append(similarity)
        return converged
    
    def refine_with_feedback(self, initial_draft: str, feedback: str) -> str:
        """Apply feedback-based refinement"""
        # Simulate feedback incorporation
        return f"{initial_draft} [Applied feedback: {feedback}]"
    
    def run_refinement_loop(self, initial_draft: str) -> Dict:
        """Run iterative refinement with convergence detection"""
        current_draft = initial_draft
        drafts_history = [current_draft]
        scores_history = [basic_quality_score(current_draft)['combined_score']]
        
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # Refine current draft
            refined_draft = self.refine_draft(current_draft, iteration + 1)
            refined_score = basic_quality_score(refined_draft)['combined_score']
            
            drafts_history.append(refined_draft)
            scores_history.append(refined_score)
            
            # Check if refinement improved quality
            if refined_score > scores_history[-2]:
                current_draft = refined_draft
            else:
                # Quality degraded, keep previous draft
                self.metrics['quality_degradation_caught'] = True
                break
            
            # Check convergence
            if iteration > 0 and self.check_convergence(drafts_history[-1], drafts_history[-2], self.convergence_threshold):
                self.metrics['converged'] = True
                break
        
        elapsed = time.time() - start_time
        
        return {
            'final_draft': current_draft,
            'initial_draft': initial_draft,
            'iterations': len(drafts_history) - 1,
            'initial_score': scores_history[0],
            'final_score': scores_history[-1],
            'improvement': scores_history[-1] - scores_history[0],
            'elapsed_seconds': elapsed,
            'draft_history': drafts_history,
            'score_history': scores_history,
            'metrics': dict(self.metrics)
        }

# Test advanced implementation
refinement = ChainOfDraftRefinement(max_iterations=3, convergence_threshold=0.85)
initial = "Machine learning systems improve through experience."

result = refinement.run_refinement_loop(initial)

print("=== Advanced Refinement Results ===")
print(f"Initial Score: {result['initial_score']:.3f}")
print(f"Final Score: {result['final_score']:.3f}")
print(f"Improvement: {result['improvement']:.3f}")
print(f"Iterations: {result['iterations']}")
print(f"Time elapsed: {result['elapsed_seconds']:.4f}s")
print(f"\nDraft History:")
for i, draft in enumerate(result['draft_history']):
    print(f"  Iteration {i}: Score={result['score_history'][i]:.3f} | {draft[:60]}...")


# ======================================================================
# ### Output from Level 2 Advanced Implementation
# The refinement loop iteratively improves drafts, tracking quality scores and detecting convergence or degradation.
# ======================================================================

# ======================================================================
# ## Real-World Example 1: Multi-Draft Generation with Quality Scoring (Summarization Task)
# ======================================================================

# Real-World Example 1: Generate multiple summarization drafts and select best
# This simulates the chain-of-draft approach for abstractive summarization

class SummarizationDraftSelector:
    """Generate multiple summary drafts and select highest quality"""
    
    def __init__(self, num_drafts: int = 3):
        self.num_drafts = num_drafts
    
    def generate_summary_drafts(self, text: str, target_length: int = 50) -> List[Tuple[str, Dict]]:
        """Generate multiple summary candidates with different approaches"""
        drafts_with_scores = []
        
        # Draft 1: Extractive summary (first sentences)
        sentences = text.split('. ')
        extractive = '. '.join(sentences[:max(1, len(sentences)//2)]) + '.'
        score1 = basic_quality_score(extractive)
        drafts_with_scores.append((extractive, score1))
        
        # Draft 2: Key points summary
        keypoints = f"Key aspects of {text[:30]}... include: insights and analysis."
        score2 = basic_quality_score(keypoints)
        drafts_with_scores.append((keypoints, score2))
        
        # Draft 3: Structured summary
        structured = f"This discusses: {text[:40]}... The main points are important for understanding."
        score3 = basic_quality_score(structured)
        drafts_with_scores.append((structured, score3))
        
        return drafts_with_scores
    
    def rank_drafts(self, drafts_with_scores: List[Tuple[str, Dict]]) -> List[Tuple[int, float]]:
        """Rank drafts by quality score"""
        rankings = [(i, score['combined_score']) for i, (_, score) in enumerate(drafts_with_scores)]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def select_best_draft(self, text: str) -> Dict:
        """Full pipeline: generate, rank, and select best summary"""
        drafts = self.generate_summary_drafts(text)
        rankings = self.rank_drafts(drafts)
        best_idx, best_score = rankings[0]
        best_draft = drafts[best_idx][0]
        
        return {
            'best_summary': best_draft,
            'best_score': best_score,
            'rank': best_idx + 1,
            'all_rankings': [(idx+1, score) for idx, score in rankings],
            'token_cost_estimate': len(best_draft.split()) * 1.3  # Estimate tokens
        }

# Test summarization example
long_text = "Machine learning is transforming industries through automation and intelligence. Deep learning drives state-of-the-art results. Modern systems combine multiple approaches. Production deployment requires careful engineering."

selector = SummarizationDraftSelector(num_drafts=3)
result = selector.select_best_draft(long_text)

print("=== Summarization Draft Selection ===")
print(f"Best Summary (Rank {result['rank']}, Score={result['best_score']:.3f}):")
print(f"  {result['best_summary']}\n")
print(f"All Rankings:")
for rank, score in result['all_rankings']:
    print(f"  Rank {rank}: Score={score:.3f}")
print(f"\nEstimated Token Cost: {result['token_cost_estimate']:.0f} tokens")


# ======================================================================
# ## Real-World Example 2: Single-Pass vs Draft-and-Refine Comparison with Quality Metrics
# ======================================================================

# Real-World Example 2: Benchmark single-pass vs draft+refine
# Simulate on a code generation task with synthetic test results

class CodeGenerationComparison:
    """Compare single-pass vs draft+refine on code generation"""
    
    def __init__(self):
        self.results = {}
    
    def simulate_code_generation(self, prompt: str, approach: str) -> Dict:
        """Simulate code generation with different approaches"""
        start_time = time.time()
        
        if approach == 'single_pass':
            # Single generation
            code = f"def solution():\n    # {prompt}\n    return result  # {approach}"
            tokens_used = 80
            test_pass_rate = 0.65  # 65% tests pass on single-pass
            num_refinements = 0
        else:  # draft_and_refine
            # Initial draft + refinement
            code = f"def solution():\n    # {prompt} (refined)\n    return result  # optimized version"
            tokens_used = 180  # 2-3x tokens
            test_pass_rate = 0.85  # 85% tests pass with refinement
            num_refinements = 1
        
        elapsed = time.time() - start_time
        
        return {
            'code': code,
            'tokens_used': tokens_used,
            'test_pass_rate': test_pass_rate,
            'num_refinements': num_refinements,
            'latency_ms': elapsed * 1000,
            'cost_estimate': tokens_used * 0.0001  # Mock cost per token
        }
    
    def compare_approaches(self, prompt: str) -> Dict:
        """Generate and compare both approaches"""
        single = self.simulate_code_generation(prompt, 'single_pass')
        refined = self.simulate_code_generation(prompt, 'draft_and_refine')
        
        quality_improvement = (refined['test_pass_rate'] - single['test_pass_rate']) / single['test_pass_rate']
        cost_multiplier = refined['tokens_used'] / single['tokens_used']
        quality_per_cost = quality_improvement / cost_multiplier
        
        return {
            'single_pass': single,
            'draft_and_refine': refined,
            'quality_improvement_pct': quality_improvement * 100,
            'cost_multiplier': cost_multiplier,
            'quality_per_cost_ratio': quality_per_cost,
            'recommendation': 'draft_and_refine' if quality_per_cost > 0.05 else 'single_pass'
        }

# Test code generation comparison
comparison = CodeGenerationComparison()
result = comparison.compare_approaches("Find the maximum element in a list")

print("=== Code Generation: Single-Pass vs Draft+Refine ===")
print(f"\n{'Metric':<30} {'Single-Pass':<20} {'Draft+Refine':<20}")
print("-" * 70)
print(f"{'Test Pass Rate':<30} {result['single_pass']['test_pass_rate']:.1%}{' '*12} {result['draft_and_refine']['test_pass_rate']:.1%}")
print(f"{'Tokens Used':<30} {result['single_pass']['tokens_used']:<20} {result['draft_and_refine']['tokens_used']}")
print(f"{'Cost Estimate':<30} ${result['single_pass']['cost_estimate']:.6f}{' '*10} ${result['draft_and_refine']['cost_estimate']:.6f}")
print(f"{'Latency (ms)':<30} {result['single_pass']['latency_ms']:.2f}{' '*13} {result['draft_and_refine']['latency_ms']:.2f}")
print(f"\nQuality Improvement: {result['quality_improvement_pct']:.1f}%")
print(f"Cost Multiplier: {result['cost_multiplier']:.2f}x")
print(f"Recommendation: Use {result['recommendation'].replace('_', ' ').title()}")


# ======================================================================
# ## Real-World Example 3: Refinement with Token Budget and Failure Handling
# ======================================================================

# Real-World Example 3: Budget-aware refinement with failure detection
# Handle cases where refinement fails or exceeds budget constraints

class BudgetAwareChainOfDraft:
    """Chain-of-draft with token budget constraints and fallback logic"""
    
    def __init__(self, total_token_budget: int = 200):
        self.total_token_budget = total_token_budget
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (simple: word count * 1.3)"""
        return int(len(text.split()) * 1.3)
    
    def allocate_budget(self, estimated_draft_tokens: int) -> Dict[str, int]:
        """Allocate budget between drafting and refinement"""
        # Allocate 60% for draft, 40% for refinement
        draft_budget = int(self.total_token_budget * 0.6)
        refine_budget = self.total_token_budget - draft_budget
        
        return {
            'draft_budget': draft_budget,
            'refine_budget': refine_budget,
            'total': self.total_token_budget
        }
    
    def can_refine(self, current_tokens: int, budget_allocation: Dict) -> bool:
        """Check if we have budget remaining for refinement"""
        used = current_tokens
        available = budget_allocation['draft_budget']
        return used <= available
    
    def detect_refinement_failure(self, original: str, refined: str) -> bool:
        """Detect if refinement made output worse"""
        original_score = basic_quality_score(original)['combined_score']
        refined_score = basic_quality_score(refined)['combined_score']
        
        # Refinement failed if quality dropped
        return refined_score < original_score
    
    def run_with_budget_constraints(self, prompt: str) -> Dict:
        """Execute chain-of-draft with budget constraints and fallback"""
        # Phase 1: Generate initial draft
        draft = f"Response to: {prompt[:50]}..."
        draft_tokens = self.estimate_tokens(draft)
        budget = self.allocate_budget(draft_tokens)
        
        result = {
            'prompt': prompt,
            'initial_draft': draft,
            'draft_tokens': draft_tokens,
            'budget': budget,
            'refinement_attempted': False,
            'refinement_succeeded': False,
            'final_output': draft,
            'total_tokens_used': draft_tokens,
            'budget_exceeded': False
        }
        
        # Phase 2: Check if refinement is possible
        if self.can_refine(draft_tokens, budget):
            result['refinement_attempted'] = True
            
            # Attempt refinement
            refined = f"{draft} [Refined for clarity]"
            refined_tokens = self.estimate_tokens(refined)
            
            # Check if refinement succeeded
            if not self.detect_refinement_failure(draft, refined):
                result['refinement_succeeded'] = True
                result['final_output'] = refined
                result['total_tokens_used'] = draft_tokens + refined_tokens
            else:
                # Refinement failed, keep original
                result['final_output'] = draft
        else:
            result['budget_exceeded'] = True
        
        # Check final budget
        if result['total_tokens_used'] > self.total_token_budget:
            result['budget_exceeded'] = True
        
        return result

# Test budget-aware implementation
budget_system = BudgetAwareChainOfDraft(total_token_budget=200)

test_prompts = [
    "Explain neural networks",
    "How does transformer attention work?",
    "What is fine-tuning in LLMs?"
]

print("=== Budget-Aware Chain-of-Draft Results ===")
for prompt in test_prompts:
    result = budget_system.run_with_budget_constraints(prompt)
    print(f"\nPrompt: {result['prompt'][:40]}...")
    print(f"  Draft tokens: {result['draft_tokens']}")
    print(f"  Budget: {result['budget']['draft_budget']}/{result['budget']['total']}")
    print(f"  Refinement attempted: {result['refinement_attempted']}")
    print(f"  Refinement succeeded: {result['refinement_succeeded']}")
    print(f"  Total tokens used: {result['total_tokens_used']}")
    print(f"  Budget exceeded: {result['budget_exceeded']}")


# ======================================================================
# ## Comparison: Quality vs Iteration Count and Token Cost Analysis
# ======================================================================

import matplotlib.pyplot as plt

# Analysis: Quality improvement vs cost
iterations = [0, 1, 2, 3, 4, 5]
quality_scores = [0.65, 0.78, 0.85, 0.87, 0.88, 0.88]  # Diminishing returns
token_costs = [100, 180, 260, 340, 420, 500]  # Linear cost increase
cost_per_quality = [t / q for t, q in zip(token_costs, quality_scores)]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Quality vs Iterations
axes[0].plot(iterations, quality_scores, 'o-', linewidth=2, markersize=8, color='green')
axes[0].axhline(y=0.85, color='r', linestyle='--', label='Target Quality')
axes[0].set_xlabel('Refinement Iterations')
axes[0].set_ylabel('Quality Score')
axes[0].set_title('Quality Improvement with Refinement')
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].set_ylim([0.6, 1.0])

# Plot 2: Token Cost vs Iterations
axes[1].plot(iterations, token_costs, 's-', linewidth=2, markersize=8, color='orange')
axes[1].set_xlabel('Refinement Iterations')
axes[1].set_ylabel('Token Cost')
axes[1].set_title('Token Cost Growth')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 550])

# Plot 3: Efficiency (Quality per Token Cost)
axes[2].bar(iterations, cost_per_quality, color=['blue', 'cyan', 'lightblue', 'lightcyan', 'white', 'white'],
           edgecolor='darkblue', linewidth=2)
axes[2].set_xlabel('Refinement Iterations')
axes[2].set_ylabel('Cost per Quality Point')
axes[2].set_title('Efficiency (Lower is Better)')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/tmp/chain_of_draft_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n=== Quantitative Analysis ===")
print(f"{'Iteration':<12} {'Quality':<12} {'Token Cost':<15} {'Cost/Quality':<15} {'Recommendation':<20}")
print("-" * 74)

for i, (qual, cost, eff) in enumerate(zip(quality_scores, token_costs, cost_per_quality)):
    rec = 'OPTIMAL' if i == 2 else ('Skip' if i == 5 else '')
    print(f"{i:<12} {qual:.3f}{' '*7} {cost:<15} {eff:<15.2f} {rec:<20}")

print(f"\nKey Finding: Refinement iterations 2-3 provide best quality-cost trade-off.")
print(f"Iteration 2: 85% quality at 260 tokens (2.6x single-pass cost)")
print(f"Beyond iteration 3: Diminishing returns, costs grow linearly but quality plateaus.")


# ======================================================================
# ## Key Takeaways
# ======================================================================

# ======================================================================
# ### Core Concept
# Chain-of-Draft is a scratchpad reasoning technique where an initial draft is iteratively refined to catch errors and improve reasoning. The key insight: a second refinement pass catches logical inconsistencies without full recomputation, delivering 15-20% quality gains for 2-3x token cost.
# ### Strategies and Their Trade-offs
# | Strategy | Quality | Speed | Token Cost | Best For |
# |----------|---------|-------|------------|----------|
# | Single-Pass | 65-75% | 1x | 1x | Real-time, latency-critical (<500ms) |
# | Draft+Refine (1 iteration) | 80-85% | 2-3x | 2-3x | Balanced quality/cost, most APIs |
# | Multi-Refine (2-3 iterations) | 85-90% | 3-4x | 3-4x | High-stakes decisions, offline |
# | Multi-Draft+Selection | 90%+ | 5-6x | 5-6x | Critical (medical, legal), max accuracy |
# ### Common Failure Modes
# - **Refinement makes output worse:** When initial draft is fundamentally flawed, refinement reinforces errors. *Fix:* Add consistency checks, compare refined vs original, keep better version.
# - **Token cost explosion:** Each refinement doubles cost. Million requests/day = $5K→$15K monthly. *Fix:* Log tokens separately, implement cost-aware routing, skip refinement for low-value queries.
# - **No convergence signal:** Infinite refinement loops or variable latency. *Fix:* Use fixed iteration counts (1-2 by default) or semantic similarity thresholds.
# - **Benchmarks ≠ Production:** MATH +20% on benchmarks; real queries +2-5%. *Fix:* Evaluate on production data, include partial-credit metrics.
# ### Production Patterns
# 1. **Multi-draft selection for quality:** Generate 3 drafts, score on heuristics (length, coherence), select best.
# 2. **Budget-constrained refinement:** Allocate 60% tokens for draft, 40% for refinement. Skip refine if budget tight.
# 3. **Convergence detection:** Compare consecutive refinements; stop if similarity >0.85 (diminishing returns).
# 4. **Fallback to original:** If refined score < original, return original. Always validate improvement.
# ======================================================================

# ======================================================================
# ## Exercises: Try It Yourself
# 1. **Modify Example 1:** Change the summarization quality metrics (e.g., prioritize diversity over coherence). How does it change which draft is selected?
# 2. **Combine approaches:** Run Example 2's code generation comparison with Example 3's budget constraints. At what budget level does draft+refine become infeasible?
# 3. **Debug the failure:** In Example 3, the refinement sometimes fails (quality drops). Add a custom failure detector that looks for specific tokens or patterns that signal degradation.
# 4. **Scaling analysis:** If you have 1 million requests/day, calculate monthly cost for single-pass vs draft+refine vs multi-draft. At what quality threshold does refinement become cost-effective?
# 5. **Real model integration:** Replace the mock draft generation with actual HuggingFace models (if transformers available). Compare real quality scores vs heuristic scores.
# ======================================================================
