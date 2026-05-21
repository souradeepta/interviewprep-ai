"""
Auto-generated from 08-constitutional-ai-rlaif.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Constitutional AI & RLAIF
# ## Learning Objectives
# 1. Understand rule-based critique systems and reward scoring
# 2. Implement constitutional principles as feedback mechanisms
# 3. Build iterative refinement loops with RLAIF (RL from AI Feedback)
# 4. Apply constitutional AI to improve model outputs
# ======================================================================

import numpy as np
import torch
import time
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from collections import Counter

# Device setup
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ======================================================================
# ## Level 1: Basic Rule-Based Critique System
# ======================================================================

class BasicConstitutionalCritique:
    """Rule-based critique using constitutional principles"""
    
    def __init__(self):
        # Constitutional principles (Anthropic CAI)
        self.principles = {
            'helpful': 'Is the response helpful and directly answers the question?',
            'harmless': 'Does the response avoid harmful, illegal, or unethical content?',
            'honest': 'Is the response truthful and not deceptive?',
        }
    
    def critique_response(self, prompt: str, response: str) -> Dict:
        """
        Critique response on constitutional principles.
        Rules: score each principle 0-1.
        """
        scores = {}
        violations = []
        
        # Helpfulness: check length, relevance, completeness
        helpful_score = min(1.0, len(response.split()) / 50)  # Longer = more helpful
        if prompt.lower() in response.lower():
            helpful_score = min(1.0, helpful_score + 0.3)  # Bonus for addressing prompt
        scores['helpful'] = helpful_score
        
        # Harmlessness: check for red flags
        harmful_keywords = ['kill', 'hate', 'illegal', 'dangerous', 'exploit']
        harmless_score = 1.0
        for keyword in harmful_keywords:
            if keyword.lower() in response.lower():
                harmless_score -= 0.25
                violations.append(f"Contains '{keyword}'")
        scores['harmless'] = max(0.0, harmless_score)
        
        # Honesty: check for uncertainty markers
        certainty_markers = ['I think', 'probably', 'might', 'uncertain']
        honest_score = 1.0
        for marker in certainty_markers:
            count = response.lower().count(marker.lower())
            if count > 2:
                honest_score = max(0.3, honest_score - 0.15 * count)
        scores['honest'] = honest_score
        
        # Aggregate score
        aggregate = np.mean(list(scores.values()))
        
        return {
            'scores': scores,
            'aggregate_score': aggregate,
            'violations': violations,
            'critique_summary': self.summarize_critique(scores)
        }
    
    def summarize_critique(self, scores: Dict) -> str:
        """Create critique summary"""
        if all(s > 0.7 for s in scores.values()):
            return "Response is high quality across all principles."
        else:
            weakest = min(scores.keys(), key=lambda k: scores[k])
            return f"Response needs improvement in {weakest}."

# Test basic critique
critic = BasicConstitutionalCritique()
test_responses = [
    ("What is attention?", "Attention is a neural network mechanism that weights different input elements differently."),
    ("How to learn?", "Study hard."),
    ("Tell me a fact", "The capital of France is Paris. This is well-established geographical fact."),
]

for prompt, response in test_responses:
    result = critic.critique_response(prompt, response)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Scores: {result['scores']}")
    print(f"Aggregate: {result['aggregate_score']:.2f}")
    print()


# Visualize critique scores
principles = list(critic.principles.keys())
scores_list = [result['scores'] for _, response in test_responses 
              for result in [critic.critique_response("", response)]]

fig, ax = plt.subplots(figsize=(10, 5))
for i, scores in enumerate(scores_list):
    ax.plot(principles, [scores[p] for p in principles], marker='o', label=f'Sample {i+1}')

ax.set_ylabel('Score')
ax.set_title('Constitutional Principle Scores')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim([0, 1])
plt.tight_layout()
plt.show()


# ======================================================================
# ## Level 2: Advanced RLAIF with Iterative Refinement
# ======================================================================

class AdvancedRLAIF:
    """RL from AI Feedback with iterative refinement"""
    
    def __init__(self):
        self.critic = BasicConstitutionalCritique()
        self.history = []
    
    def get_critique_feedback(self, prompt: str, response: str) -> Dict:
        """Get detailed critique and feedback"""
        critique = self.critic.critique_response(prompt, response)
        
        # Generate improvement suggestions
        suggestions = []
        if critique['scores']['helpful'] < 0.6:
            suggestions.append("Expand the response with more detail.")
        if critique['scores']['harmless'] < 0.7:
            suggestions.append("Remove potentially harmful content.")
        if critique['scores']['honest'] < 0.7:
            suggestions.append("Be more confident in your statements.")
        
        return {
            **critique,
            'suggestions': suggestions
        }
    
    def refine_response(self, prompt: str, response: str, iteration: int = 1) -> Dict:
        """
        Iteratively refine response based on feedback.
        In practice: this would be done by the model.
        Here: simulate improvements.
        """
        feedback = self.get_critique_feedback(prompt, response)
        
        # Simulate refinement (add more tokens, confidence)
        refined = response
        if feedback['scores']['helpful'] < 0.7:
            refined += " For more context, this concept is fundamental to modern NLP."
        if feedback['scores']['honest'] < 0.7:
            # Remove uncertainty markers
            refined = refined.replace("might", "").replace("I think", "")
        
        refinement_result = self.critic.critique_response(prompt, refined)
        
        return {
            'iteration': iteration,
            'original_score': feedback['aggregate_score'],
            'refined_response': refined,
            'refined_score': refinement_result['aggregate_score'],
            'improvement': refinement_result['aggregate_score'] - feedback['aggregate_score'],
            'original_feedback': feedback
        }
    
    def iterative_refinement_loop(self, prompt: str, initial_response: str, 
                                   max_iterations: int = 3) -> List[Dict]:
        """Run iterative refinement until convergence"""
        history = []
        current_response = initial_response
        
        for i in range(max_iterations):
            result = self.refine_response(prompt, current_response, iteration=i)
            history.append(result)
            current_response = result['refined_response']
            
            # Stop if improvement is minimal
            if result['improvement'] < 0.01:
                break
        
        return history

# Test iterative refinement
rlaif = AdvancedRLAIF()
prompt = "What is gradient descent?"
initial = "It minimizes loss."

refinement_history = rlaif.iterative_refinement_loop(prompt, initial, max_iterations=3)

print("Iterative Refinement:")
print(f"Prompt: {prompt}")
print(f"Initial: {initial}")
for i, step in enumerate(refinement_history):
    print(f"\nIteration {i+1}:")
    print(f"  Original Score: {step['original_score']:.3f}")
    print(f"  Refined Score: {step['refined_score']:.3f}")
    print(f"  Improvement: {step['improvement']:+.3f}")
    print(f"  Refined: {step['refined_response'][:80]}...")


# Visualize improvement over iterations
scores = [step['original_score'] for step in refinement_history]
refined_scores = [step['refined_score'] for step in refinement_history]
iterations = list(range(1, len(scores) + 1))

plt.figure(figsize=(10, 5))
plt.plot(iterations, scores, 'o-', label='Original Score', linewidth=2, markersize=8)
plt.plot(iterations, refined_scores, 's-', label='Refined Score', linewidth=2, markersize=8)
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('Score Improvement Through Iterative Refinement')
plt.legend()
plt.grid(alpha=0.3)
plt.ylim([0, 1])
plt.tight_layout()
plt.show()


# ======================================================================
# ## Real-World Example 1: Constitutional Critique on Model Outputs
# ======================================================================

class ModelOutputCritiquer:
    """Critique outputs from language models"""
    
    def __init__(self):
        self.critic = BasicConstitutionalCritique()
    
    def evaluate_model_outputs(self, prompts: List[str], 
                               responses: List[str]) -> Dict:
        """Batch evaluation of model outputs"""
        results = []
        
        for prompt, response in zip(prompts, responses):
            critique = self.critic.critique_response(prompt, response)
            results.append({
                'prompt': prompt,
                'response': response,
                'scores': critique['scores'],
                'aggregate': critique['aggregate_score']
            })
        
        return {
            'evaluations': results,
            'avg_helpful': np.mean([r['scores']['helpful'] for r in results]),
            'avg_harmless': np.mean([r['scores']['harmless'] for r in results]),
            'avg_honest': np.mean([r['scores']['honest'] for r in results]),
            'overall_avg': np.mean([r['aggregate'] for r in results])
        }

# Test model output critique
test_prompts = [
    "What is machine learning?",
    "How do neural networks work?",
    "Explain attention mechanisms",
]
test_responses = [
    "Machine learning is when computers learn from data without explicit programming.",
    "Neural networks have neurons organized in layers that process information.",
    "Attention mechanisms let models weight different parts of input differently.",
]

critic_eval = ModelOutputCritiquer()
eval_result = critic_eval.evaluate_model_outputs(test_prompts, test_responses)

print("Model Output Evaluation:")
for eval in eval_result['evaluations']:
    print(f"Q: {eval['prompt']}")
    print(f"A: {eval['response']}")
    print(f"Score: {eval['aggregate']:.2f}\n")


# ======================================================================
# ## Real-World Example 2: Preference Ranking with Constitutional Principles
# ======================================================================

class PreferenceRanker:
    """Compare outputs and rank by constitutional principles"""
    
    def __init__(self):
        self.critic = BasicConstitutionalCritique()
    
    def rank_responses(self, prompt: str, responses: List[str]) -> List[Dict]:
        """Rank multiple responses by constitutional scores"""
        evaluations = []
        
        for i, response in enumerate(responses):
            critique = self.critic.critique_response(prompt, response)
            evaluations.append({
                'rank_idx': i,
                'response': response,
                'scores': critique['scores'],
                'aggregate': critique['aggregate_score']
            })
        
        # Sort by aggregate score
        evaluations.sort(key=lambda x: x['aggregate'], reverse=True)
        
        # Add ranks
        for i, eval in enumerate(evaluations):
            eval['final_rank'] = i + 1
        
        return evaluations

# Test preference ranking
ranker = PreferenceRanker()
prompt = "What is a transformer model?"
candidates = [
    "It's a type of neural network architecture.",
    "Transformers are neural networks using self-attention to process sequences in parallel, enabling faster training and better capture of long-range dependencies compared to RNNs.",
    "Transformers use attention mechanisms.",
]

rankings = ranker.rank_responses(prompt, candidates)

print(f"Ranking responses to: {prompt}\n")
for ranking in rankings:
    print(f"Rank {ranking['final_rank']}: Score {ranking['aggregate']:.2f}")
    print(f"Response: {ranking['response']}")
    print(f"Scores: {ranking['scores']}\n")


# ======================================================================
# ## Real-World Example 3: Feedback-Driven Model Improvement
# ======================================================================

class FeedbackLoopTrainer:
    """Use constitutional feedback to guide model training"""
    
    def __init__(self):
        self.critic = BasicConstitutionalCritique()
        self.feedback_history = []
    
    def collect_feedback_batch(self, prompts: List[str], 
                              responses: List[str]) -> Dict:
        """Collect feedback on a batch of responses"""
        batch_feedback = []
        
        for prompt, response in zip(prompts, responses):
            critique = self.critic.critique_response(prompt, response)
            
            batch_feedback.append({
                'prompt': prompt,
                'response': response,
                'helpful': critique['scores']['helpful'],
                'harmless': critique['scores']['harmless'],
                'honest': critique['scores']['honest'],
                'reward': critique['aggregate_score']  # Reward signal for RL
            })
        
        return {
            'feedback': batch_feedback,
            'avg_reward': np.mean([f['reward'] for f in batch_feedback]),
            'reward_std': np.std([f['reward'] for f in batch_feedback])
        }
    
    def training_step(self, prompts: List[str], responses: List[str]):
        """Simulate a training step with constitutional feedback"""
        feedback_batch = self.collect_feedback_batch(prompts, responses)
        self.feedback_history.append(feedback_batch)
        return feedback_batch

# Simulate training iterations
trainer = FeedbackLoopTrainer()

training_prompts = [
    "What is deep learning?",
    "Explain backpropagation",
    "How do embeddings work?"
]

training_responses_iter1 = [
    "Deep learning uses neural networks.",
    "It updates weights using gradients.",
    "Embeddings map words to vectors."
]

training_responses_iter2 = [
    "Deep learning is a subset of ML that uses neural networks with multiple layers to learn hierarchical representations from data.",
    "Backpropagation computes gradients using chain rule to update network weights via gradient descent, minimizing loss.",
    "Embeddings map discrete tokens to continuous vector space, capturing semantic relationships for downstream tasks."
]

print("Training Iteration 1:")
feedback1 = trainer.training_step(training_prompts, training_responses_iter1)
print(f"Average Reward: {feedback1['avg_reward']:.3f}")

print("\nTraining Iteration 2 (improved):")
feedback2 = trainer.training_step(training_prompts, training_responses_iter2)
print(f"Average Reward: {feedback2['avg_reward']:.3f}")

print(f"\nReward Improvement: {feedback2['avg_reward'] - feedback1['avg_reward']:+.3f}")


# ======================================================================
# ## Comparison: Score Evolution Over Training
# ======================================================================

# Simulate multiple training steps
all_rewards = []
for iteration in range(5):
    # Simulate improving responses
    quality_factor = 0.6 + (iteration * 0.08)  # Gradual improvement
    mock_rewards = [quality_factor + np.random.normal(0, 0.05) for _ in range(len(training_prompts))]
    all_rewards.append(np.mean(mock_rewards))

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Reward curve
iterations = list(range(len(all_rewards)))
ax1.plot(iterations, all_rewards, 'o-', linewidth=2, markersize=8, color='steelblue')
ax1.fill_between(iterations, all_rewards, alpha=0.3, color='steelblue')
ax1.set_xlabel('Training Iteration')
ax1.set_ylabel('Average Reward')
ax1.set_title('Constitutional AI Reward Signal Over Time')
ax1.grid(alpha=0.3)
ax1.set_ylim([0.5, 1.0])

# Principle breakdown
principles = ['helpful', 'harmless', 'honest']
final_scores = [0.85, 0.82, 0.88]  # Example final scores
ax2.barh(principles, final_scores, color=['steelblue', 'coral', 'green'], alpha=0.7)
ax2.set_xlabel('Score')
ax2.set_title('Constitutional Principle Scores (Final)')
ax2.set_xlim([0, 1])
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Final Average Reward: {all_rewards[-1]:.3f}")


# ======================================================================
# ## Key Takeaways
# **Core Idea:** Constitutional AI uses rule-based critiques and reward signals from AI feedback to align models with desired behaviors without human-in-the-loop rating.
# **Approaches:**
# | Method | Speed | Coverage | Quality |
# |--------|-------|----------|----------|
# | Rule-based | Very Fast | Limited | Moderate |
# | Learned Critic | Medium | Broad | High |
# | Constitutional CAI | Fast | Broad | High |
# | Iterative Refinement | Medium | Targeted | Very High |
# **Common Issues:**
# - **Gaming:** Model optimizes for critique score rather than true quality. Fix: Ensure principles cover all important dimensions.
# - **Rigid Rules:** Hard thresholds miss nuance. Fix: Use learned critics with confidence scores.
# - **Circular Feedback:** AI feedback reinforces AI biases. Fix: Mix human feedback, diverse critics.
# **Related Concepts:**
# - [RLHF](./XX) – Human feedback variant
# - [Synthetic Data](./07-synthetic-data-generation.ipynb) – Generate training data with feedback
# - [RAFT](./09-raft-retrieval-augmented-finetuning.ipynb) – Fine-tune with feedback
# ======================================================================

# ======================================================================
# ## Try It Yourself
# 1. **Modify principles:** Add new constitutional principles and see how they affect rankings.
# 2. **Compare refinement strategies:** Run iterative loops with different improvement strategies.
# 3. **Analyze feedback distribution:** Collect feedback on diverse prompts and analyze principle coverage.
# ======================================================================
