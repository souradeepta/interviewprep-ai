"""
Auto-generated from 01-llm-evaluation-harness.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # LLM Evaluation Harness
# ## Learning Objectives
# 1. Understand how to compute BLEU, ROUGE, and other evaluation metrics from scratch
# 2. Build a production evaluation harness with batch processing and timing
# 3. Apply evaluation metrics to real LLM outputs and datasets
# 4. Design regression detection systems that flag model performance degradation
# ======================================================================

import numpy as np
import torch
import time
import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

# Device setup for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ======================================================================
# ## Level 1: Basic Metric Computation from Scratch
# ======================================================================

# Level 1: Implement BLEU and ROUGE from scratch using only numpy
def tokenize_simple(text: str) -> List[str]:
    """Simple word-level tokenization"""
    return text.lower().split()

def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from token list"""
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] += 1
    return ngrams

def compute_bleu_1gram(reference: str, hypothesis: str) -> float:
    """Compute BLEU-1 (1-gram precision)"""
    ref_tokens = tokenize_simple(reference)
    hyp_tokens = tokenize_simple(hypothesis)
    
    if len(hyp_tokens) == 0:
        return 0.0
    
    ref_ngrams = get_ngrams(ref_tokens, 1)
    hyp_ngrams = get_ngrams(hyp_tokens, 1)
    
    # Clipped precision: count matches, not exceeding reference counts
    matches = sum(min(hyp_ngrams[ngram], ref_ngrams[ngram]) 
                   for ngram in hyp_ngrams)
    precision = matches / len(hyp_tokens)
    return precision

def compute_rouge_1(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-1 (1-gram recall)"""
    ref_tokens = tokenize_simple(reference)
    hyp_tokens = tokenize_simple(hypothesis)
    
    if len(ref_tokens) == 0:
        return 0.0
    
    ref_ngrams = get_ngrams(ref_tokens, 1)
    hyp_ngrams = get_ngrams(hyp_tokens, 1)
    
    # Recall: what fraction of reference is covered by hypothesis
    matches = sum(min(hyp_ngrams[ngram], ref_ngrams[ngram]) 
                   for ngram in ref_ngrams if ngram in hyp_ngrams)
    recall = matches / len(ref_tokens)
    return recall

# Test with simple examples
ref = "the quick brown fox jumps over the lazy dog"
hyp1 = "the quick brown fox jumps over the lazy dog"
hyp2 = "a fast brown fox jumps over lazy dog"
hyp3 = "the quick fox"

print(f"Perfect match BLEU-1: {compute_bleu_1gram(ref, hyp1):.3f}")
print(f"Partial match BLEU-1: {compute_bleu_1gram(ref, hyp2):.3f}")
print(f"Short text BLEU-1: {compute_bleu_1gram(ref, hyp3):.3f}")
print()
print(f"Perfect match ROUGE-1: {compute_rouge_1(ref, hyp1):.3f}")
print(f"Partial match ROUGE-1: {compute_rouge_1(ref, hyp2):.3f}")
print(f"Short text ROUGE-1: {compute_rouge_1(ref, hyp3):.3f}")


# ======================================================================
# ## Level 2: Advanced Evaluation Harness with Multiple Metrics
# ======================================================================

class AdvancedEvaluationHarness:
    """Production evaluation harness with multiple metrics, batch processing, and timing"""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.metrics_history = []
        self.timing = {}
    
    def compute_bleu_multi_ngram(self, reference: str, hypothesis: str, max_n: int = 4) -> float:
        """Compute corpus BLEU with multiple n-grams and geometric mean"""
        ref_tokens = tokenize_simple(reference)
        hyp_tokens = tokenize_simple(hypothesis)
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        precisions = []
        for n in range(1, min(max_n + 1, len(hyp_tokens) + 1)):
            ref_ngrams = get_ngrams(ref_tokens, n)
            hyp_ngrams = get_ngrams(hyp_tokens, n)
            
            matches = sum(min(hyp_ngrams[ngram], ref_ngrams.get(ngram, 0)) 
                          for ngram in hyp_ngrams)
            total = len(hyp_tokens) - n + 1
            if total > 0:
                precisions.append(matches / total)
            else:
                precisions.append(0.0)
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            bleu = np.exp(np.mean(np.log(precisions)))
        else:
            bleu = 0.0
        
        return bleu
    
    def compute_f1_score(self, reference: str, hypothesis: str) -> float:
        """Compute F1 score (harmonic mean of precision and recall)"""
        precision = compute_bleu_1gram(reference, hypothesis)
        recall = compute_rouge_1(reference, hypothesis)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def evaluate_batch(self, references: List[str], hypotheses: List[str]) -> Dict:
        """Evaluate a batch of reference-hypothesis pairs"""
        start_time = time.time()
        
        bleu_scores = []
        rouge_scores = []
        f1_scores = []
        
        try:
            for ref, hyp in zip(references, hypotheses):
                bleu = self.compute_bleu_multi_ngram(ref, hyp)
                rouge = compute_rouge_1(ref, hyp)
                f1 = self.compute_f1_score(ref, hyp)
                
                bleu_scores.append(bleu)
                rouge_scores.append(rouge)
                f1_scores.append(f1)
            
            elapsed = time.time() - start_time
            
            results = {
                'bleu_mean': np.mean(bleu_scores),
                'bleu_std': np.std(bleu_scores),
                'rouge_mean': np.mean(rouge_scores),
                'rouge_std': np.std(rouge_scores),
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'batch_size': len(references),
                'elapsed_seconds': elapsed,
                'throughput_per_second': len(references) / elapsed if elapsed > 0 else 0
            }
            
            self.metrics_history.append(results)
            return results
        
        except Exception as e:
            print(f'❌ Evaluation error: {str(e)}')
            return {'status': 'error', 'message': str(e)}

# Test the harness
harness = AdvancedEvaluationHarness(batch_size=32)

# Create test dataset
test_references = [
    "machine learning requires large amounts of data",
    "neural networks are inspired by biological brains",
    "deep learning has revolutionized computer vision",
    "transformers are the foundation of modern nlp"
]

test_hypotheses = [
    "machine learning needs a lot of data",
    "neural networks come from biology",
    "deep learning changed computer vision",
    "transformers build modern language models"
]

results = harness.evaluate_batch(test_references, test_hypotheses)
print("Evaluation Results:")
for key, value in results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")


# ======================================================================
# ## Real-World Example 1: Evaluating QA System on Custom Dataset
# ======================================================================

# Example 1: Build a QA dataset and evaluate a simple rule-based baseline
class SimpleQABaseline:
    """Mock LLM that generates answers based on question patterns"""
    
    def __init__(self):
        self.knowledge_base = {
            'what is machine learning': 'Machine learning is a field of study that teaches computers to learn from data',
            'how does gradient descent work': 'Gradient descent updates weights by moving in direction of steepest descent',
            'what are transformers': 'Transformers are neural network architectures based on self-attention mechanisms',
        }
    
    def answer(self, question: str) -> str:
        """Return answer from knowledge base or generic fallback"""
        q_lower = question.lower()
        for q_key, answer in self.knowledge_base.items():
            if q_key in q_lower:
                return answer
        return "I am not sure about that question"

# Create QA dataset with reference answers
qa_dataset = [
    {
        'question': 'What is machine learning?',
        'reference': 'Machine learning is an area of study enabling computers to learn from data without being explicitly programmed'
    },
    {
        'question': 'How does gradient descent work?',
        'reference': 'Gradient descent iteratively updates parameters by computing gradients and moving in the direction of steepest descent'
    },
    {
        'question': 'What are transformers?',
        'reference': 'Transformers are deep neural networks that use self-attention to process input sequences in parallel'
    },
    {
        'question': 'Explain overfitting',
        'reference': 'Overfitting occurs when a model learns training data patterns including noise, reducing generalization to new data'
    },
    {
        'question': 'What is regularization?',
        'reference': 'Regularization is a technique that constrains model complexity to prevent overfitting by penalizing large weights'
    }
]

# Generate answers and evaluate
baseline = SimpleQABaseline()
harness = AdvancedEvaluationHarness()

references = [item['reference'] for item in qa_dataset]
hypotheses = [baseline.answer(item['question']) for item in qa_dataset]

qa_results = harness.evaluate_batch(references, hypotheses)

print("QA System Evaluation on 5 Questions:")
print(f"  BLEU score: {qa_results['bleu_mean']:.4f} (±{qa_results['bleu_std']:.4f})")
print(f"  ROUGE score: {qa_results['rouge_mean']:.4f} (±{qa_results['rouge_std']:.4f})")
print(f"  F1 score: {qa_results['f1_mean']:.4f} (±{qa_results['f1_std']:.4f})")
print(f"  Throughput: {qa_results['throughput_per_second']:.1f} samples/sec")


# ======================================================================
# ## Real-World Example 2: Standardized Benchmark Evaluation (LAMBADA-style)
# ======================================================================

# Example 2: Simulate LAMBADA-style evaluation (predict last word in passages)
class LAMBADAEvaluator:
    """Evaluator for word prediction tasks (like LAMBADA benchmark)"""
    
    def __init__(self):
        # Mock LAMBADA passages: context and target word
        self.passages = [
            {
                'context': 'The cat sat on the',
                'target': 'mat',
                'model_pred': 'mat'  # Perfect prediction
            },
            {
                'context': 'She put the book on the',
                'target': 'shelf',
                'model_pred': 'table'  # Wrong prediction
            },
            {
                'context': 'The capital of France is',
                'target': 'Paris',
                'model_pred': 'Paris'  # Correct
            },
            {
                'context': 'The largest planet in our solar system is',
                'target': 'Jupiter',
                'model_pred': 'Saturn'  # Wrong
            },
            {
                'context': 'The chemical symbol for gold is',
                'target': 'Au',
                'model_pred': 'Au'  # Correct
            }
        ]
    
    def evaluate(self) -> Dict:
        """Compute accuracy and BLEU metrics on predictions"""
        exact_matches = 0
        bleu_scores = []
        
        for passage in self.passages:
            target = passage['target']
            pred = passage['model_pred']
            
            # Exact match accuracy
            if target.lower() == pred.lower():
                exact_matches += 1
            
            # BLEU score for each prediction
            bleu = compute_bleu_1gram(target, pred)
            bleu_scores.append(bleu)
        
        return {
            'accuracy': exact_matches / len(self.passages),
            'bleu_mean': np.mean(bleu_scores),
            'bleu_scores': bleu_scores,
            'total_samples': len(self.passages)
        }

evaluator = LAMBADAEvaluator()
benchmark_results = evaluator.evaluate()

print("LAMBADA-style Benchmark Results:")
print(f"  Exact Match Accuracy: {benchmark_results['accuracy']:.1%}")
print(f"  Average BLEU: {benchmark_results['bleu_mean']:.4f}")
print(f"  Total Samples: {benchmark_results['total_samples']}")
print(f"  Per-sample BLEU: {[f'{score:.3f}' for score in benchmark_results['bleu_scores']]}")


# ======================================================================
# ## Real-World Example 3: Regression Detection Harness
# ======================================================================

# Example 3: Build regression detection system to flag performance degradation
class RegressionDetectionHarness:
    """Detects model performance degradation between versions"""
    
    def __init__(self, baseline_threshold: float = 0.95, alert_threshold: float = 0.9):
        self.baseline_threshold = baseline_threshold  # 95% of baseline = OK
        self.alert_threshold = alert_threshold        # 90% of baseline = ALERT
        self.baseline_metrics = None
        self.current_metrics = None
    
    def set_baseline(self, references: List[str], hypotheses: List[str]):
        """Set baseline model metrics"""
        harness = AdvancedEvaluationHarness()
        self.baseline_metrics = harness.evaluate_batch(references, hypotheses)
        print(f"✓ Baseline BLEU set to {self.baseline_metrics['bleu_mean']:.4f}")
    
    def check_regression(self, references: List[str], hypotheses: List[str]) -> Dict:
        """Check if current model regressed vs baseline"""
        if self.baseline_metrics is None:
            return {'status': 'error', 'message': 'Baseline not set'}
        
        harness = AdvancedEvaluationHarness()
        self.current_metrics = harness.evaluate_batch(references, hypotheses)
        
        baseline_bleu = self.baseline_metrics['bleu_mean']
        current_bleu = self.current_metrics['bleu_mean']
        
        ratio = current_bleu / baseline_bleu if baseline_bleu > 0 else 0
        
        status = 'OK'
        if ratio < self.alert_threshold:
            status = '🚨 CRITICAL REGRESSION'
        elif ratio < self.baseline_threshold:
            status = '⚠️  WARNING: Performance Below Threshold'
        
        return {
            'status': status,
            'baseline_bleu': baseline_bleu,
            'current_bleu': current_bleu,
            'ratio': ratio,
            'delta_bleu': current_bleu - baseline_bleu,
            'passed': ratio >= self.baseline_threshold
        }

# Create and test regression detector
detector = RegressionDetectionHarness(baseline_threshold=0.95, alert_threshold=0.90)

# Baseline model (good performance)
baseline_refs = test_references
baseline_hyps = [
    "machine learning requires large amounts of data",
    "neural networks are inspired by brains",
    "deep learning changed computer vision",
    "transformers form the foundation of nlp"
]

detector.set_baseline(baseline_refs, baseline_hyps)

# New model version 1 (slight degradation)
v1_hyps = [
    "machine learning needs data",
    "neural networks are models",
    "deep learning is important",
    "transformers work with text"
]

regression_check_v1 = detector.check_regression(baseline_refs, v1_hyps)
print("\nVersion 1 Regression Check:")
print(f"  {regression_check_v1['status']}")
print(f"  Baseline BLEU: {regression_check_v1['baseline_bleu']:.4f}")
print(f"  Current BLEU: {regression_check_v1['current_bleu']:.4f}")
print(f"  Ratio: {regression_check_v1['ratio']:.2%} (threshold: 95%)")
print(f"  Delta: {regression_check_v1['delta_bleu']:+.4f}")


# ======================================================================
# ## Comparison: Metric Trade-offs and Benchmarking
# ======================================================================

# Benchmark different metrics and their properties
import matplotlib.pyplot as plt

# Create diverse test cases
test_cases = [
    {'ref': 'the quick brown fox', 'hyp': 'the quick brown fox', 'name': 'Perfect match'},
    {'ref': 'the quick brown fox', 'hyp': 'the quick fox', 'name': 'Missing word'},
    {'ref': 'the quick brown fox', 'hyp': 'the quick brown', 'name': 'Truncated'},
    {'ref': 'the quick brown fox', 'hyp': 'quick brown fox the', 'name': 'Word order'},
    {'ref': 'the quick brown fox', 'hyp': 'a slow brown fox', 'name': 'Partial match'},
    {'ref': 'the quick brown fox', 'hyp': 'dog', 'name': 'Mostly wrong'},
]

harness = AdvancedEvaluationHarness()
results_data = []

for test in test_cases:
    refs = [test['ref']]
    hyps = [test['hyp']]
    result = harness.evaluate_batch(refs, hyps)
    results_data.append({
        'name': test['name'],
        'bleu': result['bleu_mean'],
        'rouge': result['rouge_mean'],
        'f1': result['f1_mean']
    })

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
names = [r['name'] for r in results_data]
bleu_vals = [r['bleu'] for r in results_data]
rouge_vals = [r['rouge'] for r in results_data]
f1_vals = [r['f1'] for r in results_data]

for ax, vals, title in zip(axes, [bleu_vals, rouge_vals, f1_vals], ['BLEU', 'ROUGE', 'F1']):
    ax.bar(range(len(vals)), vals, color=['green' if v > 0.8 else 'orange' if v > 0.5 else 'red' for v in vals])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score')
    ax.set_title(f'{title} Scores')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_metrics_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nMetric Comparison Table:")
print(f"{'Test Case':<20} {'BLEU':<8} {'ROUGE':<8} {'F1':<8}")
print("-" * 44)
for r in results_data:
    print(f"{r['name']:<20} {r['bleu']:<8.3f} {r['rouge']:<8.3f} {r['f1']:<8.3f}")


# ======================================================================
# ## Key Takeaways
# ### Core Concept
# LLM evaluation requires multiple metrics because no single metric captures all aspects of output quality. BLEU measures precision (how much generated text matches references), ROUGE measures recall (how much reference content appears in output), and F1 balances both.
# ### Evaluation Metrics Comparison
# | Metric | Strength | Weakness | Best For |
# |--------|----------|----------|----------|
# | BLEU | Matches reference word choices | Penalizes synonyms unfairly | Machine translation |
# | ROUGE | Captures important content words | Ignores word order | Summarization |
# | F1 | Balances precision/recall | Equally weights both | General-purpose QA |
# | Exact Match | No ambiguity | Too strict for NLG | Factoid QA |
# | Semantic Similarity | Captures meaning | Requires embeddings | Open-ended generation |
# ### Common Failure Modes
# - **Metric optimization without human validation**: BLEU-gaming via surface copying loses semantic meaning. Always validate with human judges.
# - **Single metric reliance**: BLEU alone misses errors in style/tone. Use 3+ metrics and correlate with human ratings.
# - **Ignoring reference diversity**: Multiple valid answers exist (e.g., "cat" vs "feline"). Use multi-reference evaluation or semantic similarity.
# - **Batch size too small**: High variance in metrics with <20 samples. Always batch evaluate with representative samples.
# - **Regression detection without baselines**: Comparing versions requires tracked baseline metrics and statistical significance testing.
# ### Related Concepts
# - [LLM Fine-tuning](./09-raft-retrieval-augmented-finetuning.ipynb) — Evaluation drives training objectives
# - [Synthetic Data Generation](./07-synthetic-data-generation.ipynb) — Evaluation metrics validate synthetic data quality
# - [Structured Generation](./14-structured-generation.ipynb) — Evaluation ensures output format compliance
# ======================================================================

# ======================================================================
# ## Try It Yourself
# 1. **Create your own QA dataset**: Add 5 more question-answer pairs to `qa_dataset` and evaluate. Notice how BLEU/ROUGE vary with answer similarity.
# 2. **Implement n-gram precision for all n**: Extend `compute_bleu_multi_ngram` to separately return precision for 1,2,3,4-grams. Plot how each n-gram precision decreases for lower-quality matches.
# 3. **Test metric sensitivity**: Create variants of references (spelling errors, synonyms, reordered words) and see which metrics are robust vs sensitive.
# 4. **Regression test your own model**: Generate 10 outputs from any model you have access to, compute baseline metrics, then perturb the model slightly and detect regression.
# ======================================================================
