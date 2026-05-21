"""
Auto-generated from 15-reflexion-self-critique.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Reflexion and Self-Critique
# ## Learning Objectives
# 1. Implement critique loop: generate → critique → score → refine
# 2. Build rule-based and model-based critics
# 3. Apply to summarization, QA with quality metrics
# 4. Measure improvement per iteration and convergence
# ======================================================================

import numpy as np
import torch
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# ======================================================================
# ## Level 1: Basic Critique Loop
# Implement generate → critique → score → refine from scratch.
# ======================================================================

class BasicCritiqueLoop:
    def generate(self, prompt: str) -> str:
        '''Generate initial output'''
        return f'Initial response to: {prompt[:30]}...'
    
    def critique(self, output: str) -> Dict[str, float]:
        '''Rule-based critique'''
        length = len(output.split())
        clarity_score = min(length / 50, 1.0)  # Longer = clearer
        relevance_score = 0.7 + 0.3 * np.sin(hash(output) % 100 / 100)
        conciseness_score = max(0, 1 - length / 200)  # Not too long
        
        return {
            'clarity': clarity_score,
            'relevance': relevance_score,
            'conciseness': conciseness_score,
            'overall': np.mean([clarity_score, relevance_score, conciseness_score])
        }
    
    def refine(self, output: str, critique: Dict) -> str:
        '''Refine based on critique scores'''
        refinements = []
        if critique['clarity'] < 0.5:
            refinements.append('clearly')
        if critique['relevance'] < 0.6:
            refinements.append('precisely')
        if critique['conciseness'] < 0.4:
            refinements.append('briefly')
        
        refined = output + ' ' + ' '.join(refinements) if refinements else output + ' [refined]'
        return refined
    
    def run_loop(self, prompt: str, max_iterations: int = 5) -> Dict:
        output = self.generate(prompt)
        history = [(output, 0.0)]
        
        for i in range(max_iterations):
            critique = self.critique(output)
            output = self.refine(output, critique)
            history.append((output, critique['overall']))
            
            if i > 0 and history[-1][1] - history[-2][1] < 0.01:
                break
        
        scores = [h[1] for h in history]
        return {
            'final_output': output,
            'iterations': len(history),
            'final_score': scores[-1],
            'improvement': scores[-1] - scores[0],
            'score_history': scores
        }

loop = BasicCritiqueLoop()
result = loop.run_loop('How to scale LLMs?')

print(f'✅ Critique loop results:')
print(f'  Iterations: {result["iterations"]}')
print(f'  Initial score: {result["score_history"][0]:.3f}')
print(f'  Final score: {result["final_score"]:.3f}')
print(f'  Improvement: {result["improvement"]:.3f}')


# ======================================================================
# ## Level 2: Advanced Critique with Iterative Refinement
# ======================================================================

class AdvancedCritic:
    def __init__(self):
        self.critique_history = []
    
    def model_based_critique(self, output: str, task: str) -> Dict:
        '''Simulate model-based critic'''
        words = output.lower().split()
        task_relevance = len([w for w in words if len(w) > 3]) / max(len(words), 1)
        coherence = 1 - (len(set(words)) / max(len(words), 1)) * 0.3  # Penalize repetition
        length_score = 1 if 10 < len(words) < 100 else max(0, 1 - abs(len(words) - 50) / 100)
        
        return {
            'task_relevance': task_relevance,
            'coherence': coherence,
            'length': length_score,
            'overall': np.mean([task_relevance, coherence, length_score])
        }
    
    def iterative_refinement(self, prompt: str, task: str, budget: int = 10) -> Dict:
        output = f'Response to {task}: {prompt[:30]}...'
        budget_used = 1
        best_output = output
        best_score = 0
        iterations = []
        
        while budget_used < budget:
            critique = self.model_based_critique(output, task)
            
            if critique['overall'] > best_score:
                best_score = critique['overall']
                best_output = output
            
            iterations.append({
                'output': output,
                'score': critique['overall'],
                'budget_used': budget_used
            })
            
            # Refine
            if critique['task_relevance'] < 0.5:
                output += ' [task-focused]'
            if critique['coherence'] < 0.6:
                output += ' [coherent]'
            if critique['length'] < 0.7:
                output += ' [balanced-length]'
            
            budget_used += 1
        
        return {
            'best_output': best_output,
            'best_score': best_score,
            'iterations': len(iterations),
            'history': iterations
        }

critic = AdvancedCritic()
result = critic.iterative_refinement('What is LLM scaling?', 'qa', budget=8)

print(f'✅ Advanced critique results:')
print(f'  Iterations: {result["iterations"]}')
print(f'  Best score: {result["best_score"]:.3f}')
scores = [h['score'] for h in result['history']]
print(f'  Score progression: {[f"{s:.2f}" for s in scores][:5]}...')


# ======================================================================
# ## Real-World Example 1: Summarization Task
# ======================================================================

class SummarizationCritic:
    def evaluate_summary(self, summary: str, original_length: int) -> Dict:
        words = summary.split()
        compression = len(words) / max(original_length, 1)
        
        return {
            'compression_ratio': compression,
            'coverage': min(1.0, compression + 0.3),
            'conciseness': 1 - compression,
            'overall': np.mean([
                1 - compression + 0.3,  # Prefer compressed
                min(1.0, compression + 0.3)
            ])
        }
    
    def refine_summary(self, summary: str, critique: Dict) -> str:
        if critique['compression_ratio'] > 0.5:
            # Too long, make more concise
            words = summary.split()
            summary = ' '.join(words[:int(len(words) * 0.7)])
        elif critique['compression_ratio'] < 0.2:
            # Too short, add details
            summary += ' [more details needed]'
        return summary
    
    def run_summarization(self, text: str, max_iter: int = 5) -> Dict:
        initial_length = len(text.split())
        summary = ' '.join(text.split()[:int(len(text.split()) * 0.5)])  # Initial 50%
        
        history = []
        for _ in range(max_iter):
            critique = self.evaluate_summary(summary, initial_length)
            history.append({'summary': summary, 'score': critique['overall']})
            summary = self.refine_summary(summary, critique)
        
        return {
            'final_summary': summary,
            'final_score': history[-1]['score'] if history else 0,
            'scores': [h['score'] for h in history]
        }

summ_critic = SummarizationCritic()
text = 'Machine learning is transforming industries. It enables systems to learn from data. ' * 5
result = summ_critic.run_summarization(text, max_iter=5)

print(f'✅ Summarization critique:')
print(f'  Initial score: {result["scores"][0]:.3f}')
print(f'  Final score: {result["final_score"]:.3f}')
print(f'  Scores: {[f"{s:.2f}" for s in result["scores"]]}')


# ======================================================================
# ## Real-World Example 2: QA Task with Metrics
# ======================================================================

class QACritic:
    def evaluate_answer(self, answer: str, question: str) -> Dict:
        # Simple metrics
        answer_words = answer.lower().split()
        question_words = question.lower().split()
        
        # Relevance: overlap with question
        overlap = len(set(answer_words) & set(question_words)) / max(len(set(question_words)), 1)
        
        # Completeness: answer length
        completeness = min(len(answer_words) / 30, 1.0)
        
        # Confidence: no hedging words
        hedges = ['maybe', 'perhaps', 'might', 'could']
        confidence = 1 - len([w for w in answer_words if w in hedges]) / max(len(answer_words), 1)
        
        return {
            'relevance': overlap,
            'completeness': completeness,
            'confidence': confidence,
            'overall': np.mean([overlap, completeness, confidence])
        }
    
    def run_qa_refinement(self, question: str, initial_answer: str, max_iter: int = 4) -> Dict:
        answer = initial_answer
        history = []
        
        for i in range(max_iter):
            metrics = self.evaluate_answer(answer, question)
            history.append({'iteration': i, 'score': metrics['overall'], 'metrics': metrics})
            
            # Refine based on weaknesses
            if metrics['relevance'] < 0.5:
                answer += ' This directly addresses the question.'
            if metrics['completeness'] < 0.5:
                answer += ' Additionally,'
            if metrics['confidence'] < 0.6:
                answer = answer.replace('might', 'will').replace('could', 'will')
        
        return {'history': history, 'final_answer': answer}

qa_critic = QACritic()
question = 'How do transformers work?'
answer = 'Transformers use attention'
result = qa_critic.run_qa_refinement(question, answer)

print(f'✅ QA refinement:')
for h in result['history']:
    print(f'  Iteration {h["iteration"]}: score={h["score"]:.3f}')


# ======================================================================
# ## Real-World Example 3: Convergence and Improvement Measurement
# ======================================================================

class ConvergenceMeasurer:
    def measure_convergence(self, scores: List[float]) -> Dict:
        improvements = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        avg_improvement = np.mean(improvements)
        
        # Find convergence iteration
        convergence_iter = None
        threshold = 0.01
        for i, imp in enumerate(improvements):
            if imp < threshold:
                convergence_iter = i
                break
        
        return {
            'total_improvement': scores[-1] - scores[0],
            'avg_improvement_per_iter': avg_improvement,
            'convergence_iteration': convergence_iter,
            'final_score': scores[-1]
        }

measurer = ConvergenceMeasurer()
scores = [0.5, 0.62, 0.71, 0.76, 0.78, 0.79, 0.79]
convergence = measurer.measure_convergence(scores)

print(f'✅ Convergence analysis:')
print(f'  Total improvement: {convergence["total_improvement"]:.3f}')
print(f'  Avg improvement/iter: {convergence["avg_improvement_per_iter"]:.3f}')
print(f'  Converged at iteration: {convergence["convergence_iteration"]}')
print(f'  Final score: {convergence["final_score"]:.3f}')


# ======================================================================
# ## Comparison: Improvement Curves
# ======================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: QA improvement
qa_scores = result['history']
iters = [h['iteration'] for h in qa_scores]
scores = [h['score'] for h in qa_scores]

ax1.plot(iters, scores, 'o-', linewidth=2, markersize=8, color='#1f77b4')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Quality Score')
ax1.set_title('QA Self-Critique: Quality Improvement')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Plot 2: Convergence
ax2.bar(range(len(scores)), scores, color=['#ff7f0e' if i < len(scores)-1 else '#2ca02c' for i in range(len(scores))], alpha=0.8)
ax2.axhline(y=scores[-1], color='red', linestyle='--', alpha=0.5, label='Converged')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Score')
ax2.set_title('Convergence Pattern')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()

plt.tight_layout()
plt.savefig('/tmp/reflexion_improvement.png', dpi=100, bbox_inches='tight')
plt.show()
print('✅ Saved visualization')


# ======================================================================
# ## Key Takeaways
# ### Core Concept
# Self-critique enables iterative improvement without retraining. Loop: generate → critique → refine → repeat.
# ### Critic Types
# | Type | Cost | Accuracy | Best For |
# |------|---|---|---|
# | Rule-based | Cheap | 70% | Length, format checks |
# | Model-based | Expensive | 85%+ | Semantic quality |
# | Hybrid | Moderate | 80%+ | Balance of quality/cost |
# ### Convergence Patterns
# - Quality typically plateaus after 3-5 iterations
# - Diminishing returns: biggest gains in first iteration
# - Stop when improvement < 1% to save compute
# ### Common Pitfalls
# - **Critic feedback is bad:** Validate critic on gold labels
# - **No stopping condition:** Add convergence detection
# - **Infinite loops:** Set max iterations and timeouts
# - **Single critique:** Ensemble multiple critics for robustness
# ======================================================================
