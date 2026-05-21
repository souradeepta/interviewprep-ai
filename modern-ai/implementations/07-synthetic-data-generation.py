"""
Auto-generated from 07-synthetic-data-generation.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Synthetic Data Generation
# ## Learning Objectives
# 1. Understand self-instruct and template-based data generation pipelines
# 2. Implement quality filtering and diversity metrics from scratch
# ======================================================================

import numpy as np
import torch
import time
import json
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from collections import Counter

# Device setup for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ======================================================================
# ## Level 1: Basic Self-Instruct Data Generation
# Implement template-based instruction generation with simple response simulation.
# ======================================================================

# Level 1: Template-based instruction generation from scratch

class BasicSelfInstruct:
    """Generate instruction-response pairs using templates"""
    
    def __init__(self):
        # Instruction templates
        self.templates = [
            "Explain how to {task}.",
            "What are the steps to {task}?",
            "Describe the process of {task}.",
            "How would you {task}?",
            "Provide a guide for {task}.",
        ]
        
        # Task pool
        self.tasks = [
            "build a neural network",
            "train a language model",
            "implement tokenization",
            "optimize inference",
            "debug gradient issues",
            "parallelize training",
        ]
    
    def generate_instructions(self, n_samples: int = 5) -> List[Dict]:
        """Generate instruction-response pairs"""
        pairs = []
        
        for i in range(n_samples):
            # Select template and task
            template = self.templates[i % len(self.templates)]
            task = self.tasks[i % len(self.tasks)]
            
            # Generate instruction
            instruction = template.format(task=task)
            
            # Simulate response (in practice: use a language model)
            response = self.simulate_response(task)
            
            pairs.append({
                'instruction': instruction,
                'response': response,
                'task_category': self.categorize_task(task)
            })
        
        return pairs
    
    def simulate_response(self, task: str) -> str:
        """Simulate response based on task (mock)"""
        responses = {
            "build a neural network": "To build a neural network, define layers (embedding, linear, attention), initialize weights, and implement forward pass.",
            "train a language model": "Training requires: dataset preparation, tokenization, batch creation, forward pass, loss computation, backpropagation, and weight updates.",
            "implement tokenization": "Tokenization converts text to tokens using vocab. Use byte-pair encoding or wordpiece. Map tokens to IDs for model input.",
            "optimize inference": "Use quantization to reduce model size, batch requests together, cache embeddings, and use int8/fp16 precision.",
            "debug gradient issues": "Check for NaN gradients, verify loss function, ensure input normalization, and use gradient clipping.",
            "parallelize training": "Use data parallelism (distribute batches) or model parallelism (distribute layers). Sync gradients across devices."
        }
        return responses.get(task, f"Response for {task}")
    
    def categorize_task(self, task: str) -> str:
        """Categorize task by type"""
        if "build" in task or "implement" in task:
            return "implementation"
        elif "train" in task or "optimize" in task:
            return "training"
        elif "debug" in task:
            return "debugging"
        else:
            return "general"

# Test basic generation
generator = BasicSelfInstruct()
pairs = generator.generate_instructions(n_samples=6)

print("Generated Instruction-Response Pairs:")
print("=" * 60)
for i, pair in enumerate(pairs, 1):
    print(f"\n[{i}] Category: {pair['task_category']}")
    print(f"Instruction: {pair['instruction']}")
    print(f"Response: {pair['response'][:80]}...")


# Visualize generated data distribution
categories = [p['task_category'] for p in pairs]
cat_counts = Counter(categories)

plt.figure(figsize=(10, 4))
plt.bar(cat_counts.keys(), cat_counts.values(), color='steelblue')
plt.xlabel('Task Category')
plt.ylabel('Count')
plt.title('Distribution of Generated Instruction Categories (Basic)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Generated {len(pairs)} instruction-response pairs")


# ======================================================================
# ## Level 2: Advanced Generation with Quality Filtering
# Add heuristic quality classifiers, diversity metrics, and batch generation with error handling.
# ======================================================================

# Level 2: Advanced generation with quality filtering and diversity

class AdvancedSyntheticGenerator:
    """Generate synthetic data with quality control and diversity metrics"""
    
    def __init__(self, quality_threshold=0.6):
        self.quality_threshold = quality_threshold
        self.templates = [
            "Explain: {topic}",
            "Q: What is {topic}? A:",
            "Implement {topic}",
            "Compare {topic} and {other}",
            "Debug {topic}",
        ]
        self.topics = [
            "attention mechanisms",
            "gradient descent",
            "embeddings",
            "fine-tuning",
            "inference optimization",
        ]
    
    def compute_quality_score(self, instruction: str, response: str) -> float:
        """
        Compute quality score based on:
        1. Length (not too short, not too long)
        2. Keyword presence (relevant terms)
        3. Response-instruction coherence
        """
        scores = []
        
        # Length score (prefer 50-300 char responses)
        resp_len = len(response)
        if 50 <= resp_len <= 300:
            scores.append(1.0)
        elif 30 <= resp_len <= 500:
            scores.append(0.7)
        else:
            scores.append(0.3)
        
        # Coherence score: how much vocabulary overlaps
        inst_words = set(w.lower() for w in instruction.split())
        resp_words = set(w.lower() for w in response.split())
        overlap = len(inst_words & resp_words) / (len(inst_words | resp_words) + 1e-8)
        coherence_score = min(1.0, overlap * 2)  # Boost overlap slightly
        scores.append(coherence_score)
        
        # Keyword diversity: avoid too many repeated words
        word_freq = Counter(w.lower() for w in response.split())
        max_freq = max(word_freq.values()) if word_freq else 1
        diversity_score = 1.0 - (max_freq / (len(response.split()) + 1e-8)) * 0.5
        scores.append(max(0.3, diversity_score))
        
        return np.mean(scores)
    
    def compute_diversity_metrics(self, data: List[Dict]) -> Dict:
        """Compute diversity across generated samples"""
        instructions = [d['instruction'] for d in data]
        responses = [d['response'] for d in data]
        
        # Lexical diversity: unique words / total words
        all_inst_words = set()
        all_inst_total = 0
        for inst in instructions:
            words = set(w.lower() for w in inst.split())
            all_inst_words.update(words)
            all_inst_total += len(inst.split())
        
        inst_diversity = len(all_inst_words) / (all_inst_total + 1e-8)
        
        # Response diversity: unique words
        all_resp_words = set()
        all_resp_total = 0
        for resp in responses:
            words = set(w.lower() for w in resp.split())
            all_resp_words.update(words)
            all_resp_total += len(resp.split())
        
        resp_diversity = len(all_resp_words) / (all_resp_total + 1e-8)
        
        # Instruction-response similarity
        similarities = []
        for d in data:
            inst_words = set(w.lower() for w in d['instruction'].split())
            resp_words = set(w.lower() for w in d['response'].split())
            overlap = len(inst_words & resp_words) / (len(inst_words | resp_words) + 1e-8)
            similarities.append(overlap)
        
        return {
            'instruction_diversity': inst_diversity,
            'response_diversity': resp_diversity,
            'avg_similarity': np.mean(similarities),
            'similarity_std': np.std(similarities)
        }
    
    def generate_with_filtering(self, n_samples: int = 20) -> Dict:
        """Generate with quality filtering"""
        all_pairs = []
        quality_scores = []
        
        # Generate candidates
        for i in range(n_samples):
            template = self.templates[i % len(self.templates)]
            topic = self.topics[i % len(self.topics)]
            
            instruction = template.format(topic=topic, other="transformers")
            response = self.mock_response(topic)
            
            quality = self.compute_quality_score(instruction, response)
            quality_scores.append(quality)
            
            all_pairs.append({
                'instruction': instruction,
                'response': response,
                'quality_score': quality
            })
        
        # Filter by quality
        filtered = [p for p in all_pairs if p['quality_score'] >= self.quality_threshold]
        
        # Compute diversity
        diversity = self.compute_diversity_metrics(filtered)
        
        return {
            'generated': len(all_pairs),
            'filtered': len(filtered),
            'pass_rate': len(filtered) / len(all_pairs),
            'quality_scores': quality_scores,
            'diversity_metrics': diversity,
            'data': filtered
        }
    
    def mock_response(self, topic: str) -> str:
        """Mock response based on topic"""
        responses = {
            "attention mechanisms": "Attention computes weighted sum of values based on query-key similarity. Allows models to focus on relevant parts.",
            "gradient descent": "Iteratively update parameters in direction of negative gradient. Step size (learning rate) controls convergence.",
            "embeddings": "Dense vector representations of discrete tokens. Capture semantic relationships between words.",
            "fine-tuning": "Adapt pretrained model to downstream task. Update all weights with task-specific data.",
            "inference optimization": "Use quantization, pruning, and batching to reduce latency. Trade-off between speed and accuracy.",
        }
        return responses.get(topic, f"Response about {topic}")

# Test advanced generation
adv_gen = AdvancedSyntheticGenerator(quality_threshold=0.5)
result = adv_gen.generate_with_filtering(n_samples=25)

print(f"Generated: {result['generated']} candidates")
print(f"Filtered: {result['filtered']} high-quality samples")
print(f"Pass rate: {result['pass_rate']:.1%}")
print(f"\nDiversity Metrics:")
for key, val in result['diversity_metrics'].items():
    print(f"  {key}: {val:.3f}")


# Visualize quality filtering
quality_scores = result['quality_scores']
threshold = adv_gen.quality_threshold

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Quality score distribution
ax1.hist(quality_scores, bins=10, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
ax1.set_xlabel('Quality Score')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Quality Scores')
ax1.legend()
ax1.grid(alpha=0.3)

# Pass/fail breakdown
passed = len([s for s in quality_scores if s >= threshold])
failed = len(quality_scores) - passed
ax2.bar(['Passed', 'Failed'], [passed, failed], color=['green', 'red'], alpha=0.7)
ax2.set_ylabel('Count')
ax2.set_title(f'Quality Filter Results (Threshold: {threshold})')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Samples passing quality filter: {passed}/{len(quality_scores)}")


# ======================================================================
# ## Real-World Example 1: Instruction Generation via Prompt Engineering
# Generate instruction-response pairs using prompt engineering and templates.
# ======================================================================

# Real-World Example 1: Instruction generation with templates

class InstructionGenerator:
    """Generate instructions via prompt templates"""
    
    def __init__(self):
        self.instructions_bank = []
    
    def generate_seed_instructions(self) -> List[str]:
        """Create seed instructions to bootstrap generation"""
        return [
            "Write code to implement",
            "Explain the concept of",
            "Compare and contrast",
            "Design a system for",
            "Debug the issue with",
        ]
    
    def expand_instruction(self, seed: str, topic: str) -> str:
        """Expand seed instruction with topic"""
        # Simple expansion (in practice: use LM)
        return f"{seed} {topic}."
    
    def generate_responses(self, instructions: List[str]) -> List[Dict]:
        """Generate responses for instructions"""
        pairs = []
        topics = [
            "backpropagation",
            "self-attention",
            "loss functions",
            "batch normalization",
            "dropout regularization",
        ]
        
        for i, inst in enumerate(instructions):
            topic = topics[i % len(topics)]
            expanded = self.expand_instruction(inst, topic)
            response = self.generate_response_text(expanded, topic)
            
            pairs.append({
                'instruction': expanded,
                'response': response,
                'tokens': len(expanded.split()) + len(response.split())
            })
        
        return pairs
    
    def generate_response_text(self, instruction: str, topic: str) -> str:
        """Generate response text (mock)"""
        responses = {
            "backpropagation": "Backpropagation computes gradients via chain rule. Initialize output gradient, propagate backward through layers, accumulate parameter gradients, update weights.",
            "self-attention": "Self-attention: Q=WqX, K=WkX, V=WvX, attention(Q,K,V)=softmax(QK^T/√d)V. Allows parallel processing, captures long-range dependencies.",
            "loss functions": "Cross-entropy for classification: -Σ p(i)log(q(i)). MSE for regression: Σ(y-ŷ)². Choose based on task and output distribution.",
            "batch normalization": "Normalize activations to mean 0, variance 1 per batch. Reduces internal covariate shift, allows higher learning rates, acts as regularization.",
            "dropout regularization": "Randomly zero activations during training (p=0.5). Reduces co-adaptation, improves generalization. No dropout at test time.",
        }
        return responses.get(topic, f"This explains {topic} with depth and clarity.")

# Test instruction generation
gen = InstructionGenerator()
seeds = gen.generate_seed_instructions()
pairs = gen.generate_responses(seeds)

print("Generated Instruction-Response Pairs:")
print("=" * 70)
for i, pair in enumerate(pairs, 1):
    print(f"\n[{i}] Tokens: {pair['tokens']}")
    print(f"Instruction: {pair['instruction'][:60]}...")
    print(f"Response: {pair['response'][:80]}...")


# ======================================================================
# ## Real-World Example 2: Magpie-Style Quality Filtering
# Implement heuristic classifiers to filter low-quality generated data.
# ======================================================================

# Real-World Example 2: Quality filtering pipeline

class QualityClassifier:
    """Magpie-style filtering using heuristics"""
    
    def __init__(self):
        self.low_quality_indicators = [
            (lambda t: len(t.split()) < 3, "too_short"),
            (lambda t: len(t.split()) > 100, "too_long"),
            (lambda t: t.count('?') > 2, "too_many_questions"),
            (lambda t: t.count('!') > 2, "too_many_exclamations"),
            (lambda t: "[" in t or "{" in t, "contains_placeholders"),
        ]
    
    def is_high_quality(self, instruction: str, response: str) -> Tuple[bool, List[str]]:
        """Check if sample passes quality filters"""
        issues = []
        
        # Check instruction quality
        for checker, issue_type in self.low_quality_indicators:
            if checker(instruction):
                issues.append(f"instruction_{issue_type}")
        
        # Check response quality
        for checker, issue_type in self.low_quality_indicators:
            if checker(response):
                issues.append(f"response_{issue_type}")
        
        # Additional checks
        if len(instruction.split()) < 2:
            issues.append("instruction_too_simple")
        
        if response.count('.') == 0 and response.count('?') == 0:
            issues.append("response_no_punctuation")
        
        return len(issues) == 0, issues
    
    def filter_batch(self, data: List[Dict]) -> Dict:
        """Filter batch of generated data"""
        passed = []
        failed = []
        issue_counts = Counter()
        
        for item in data:
            is_good, issues = self.is_high_quality(
                item['instruction'], 
                item['response']
            )
            
            if is_good:
                passed.append(item)
            else:
                failed.append({
                    **item,
                    'rejection_reasons': issues
                })
                issue_counts.update(issues)
        
        return {
            'total': len(data),
            'passed': len(passed),
            'failed': len(failed),
            'pass_rate': len(passed) / len(data) if data else 0,
            'quality_data': passed,
            'rejected_data': failed,
            'failure_reasons': dict(issue_counts)
        }

# Test quality filtering
classifier = QualityClassifier()

test_data = [
    {'instruction': 'Explain attention', 'response': 'Attention mechanisms allow models to focus on relevant parts of the input.'},
    {'instruction': 'X', 'response': 'Too short instruction'},
    {'instruction': 'What is machine learning?', 'response': 'ML is a field of AI'},
    {'instruction': 'How to [PLACEHOLDER]?', 'response': 'This has placeholders'},
    {'instruction': 'Implement a transformer model', 'response': 'Use PyTorch: nn.TransformerEncoder, nn.TransformerDecoder, attention layers.'},
]

filter_result = classifier.filter_batch(test_data)

print(f"Filtering Results:")
print(f"  Total: {filter_result['total']}")
print(f"  Passed: {filter_result['passed']}")
print(f"  Failed: {filter_result['failed']}")
print(f"  Pass rate: {filter_result['pass_rate']:.1%}")
print(f"\nFailure Reasons:")
for reason, count in filter_result['failure_reasons'].items():
    print(f"  {reason}: {count}")


# ======================================================================
# ## Real-World Example 3: Domain-Specific Synthetic Data (Code Generation)
# Create domain-specific synthetic data using prompt templates for code generation.
# ======================================================================

# Real-World Example 3: Code generation synthetic data

class CodeSyntheticGenerator:
    """Generate synthetic code problems and solutions"""
    
    def __init__(self):
        self.programming_domains = [
            "array manipulation",
            "string processing",
            "graph algorithms",
            "dynamic programming",
            "sorting and searching",
        ]
        self.difficulty_levels = ["easy", "medium", "hard"]
    
    def generate_code_instruction(self, domain: str, difficulty: str) -> str:
        """Generate a programming problem statement"""
        templates = [
            f"Write a function to solve {domain} problem (difficulty: {difficulty}).",
            f"Implement a solution for {domain} (level: {difficulty}).",
            f"Code a {difficulty} {domain} problem.",
        ]
        return templates[hash((domain, difficulty)) % len(templates)]
    
    def generate_code_solution(self, domain: str) -> str:
        """Generate a code solution (mock)"""
        solutions = {
            "array manipulation": '''def reverse_array(arr):\n    left, right = 0, len(arr) - 1\n    while left < right:\n        arr[left], arr[right] = arr[right], arr[left]\n        left += 1\n        right -= 1\n    return arr''',
            "string processing": '''def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]''',
            "graph algorithms": '''def bfs(graph, start):\n    visited = set()\n    queue = [start]\n    while queue:\n        node = queue.pop(0)\n        if node not in visited:\n            visited.add(node)\n            queue.extend(graph[node])\n    return visited''',
            "dynamic programming": '''def fibonacci(n):\n    if n <= 1: return n\n    dp = [0] * (n + 1)\n    dp[1] = 1\n    for i in range(2, n + 1):\n        dp[i] = dp[i-1] + dp[i-2]\n    return dp[n]''',
            "sorting and searching": '''def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1''',
        }
        return solutions.get(domain, f"# Solution for {domain}")
    
    def generate_batch(self, batch_size: int = 10) -> List[Dict]:
        """Generate a batch of code problems"""
        batch = []
        
        for i in range(batch_size):
            domain = self.programming_domains[i % len(self.programming_domains)]
            difficulty = self.difficulty_levels[i % len(self.difficulty_levels)]
            
            instruction = self.generate_code_instruction(domain, difficulty)
            code = self.generate_code_solution(domain)
            
            batch.append({
                'domain': domain,
                'difficulty': difficulty,
                'instruction': instruction,
                'code_solution': code,
                'code_length': len(code.split('\n'))
            })
        
        return batch

# Test code generation
code_gen = CodeSyntheticGenerator()
code_batch = code_gen.generate_batch(batch_size=10)

print("Generated Code Problems:")
print("=" * 70)
for i, item in enumerate(code_batch[:3], 1):
    print(f"\n[{i}] Domain: {item['domain']} | Difficulty: {item['difficulty']}")
    print(f"Problem: {item['instruction']}")
    print(f"Code lines: {item['code_length']}")


# ======================================================================
# ## Comparison: Quality Distribution Before and After Filtering
# Analyze how filtering affects data quality distribution.
# ======================================================================

# Comparison: Quality distribution before/after filtering

# Generate unfiltered data
n_generate = 100
adv_gen_comp = AdvancedSyntheticGenerator(quality_threshold=0.0)
unfiltered_result = adv_gen_comp.generate_with_filtering(n_samples=n_generate)

# Generate filtered data
adv_gen_filt = AdvancedSyntheticGenerator(quality_threshold=0.6)
filtered_result = adv_gen_filt.generate_with_filtering(n_samples=n_generate)

# Extract quality scores
unfiltered_scores = unfiltered_result['quality_scores']
filtered_scores = [d['quality_score'] for d in filtered_result['data']]

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Distribution comparison
ax1.hist(unfiltered_scores, bins=15, alpha=0.6, label='Unfiltered', color='steelblue', edgecolor='black')
ax1.hist(filtered_scores, bins=15, alpha=0.6, label='Filtered (≥0.6)', color='green', edgecolor='black')
ax1.axvline(0.6, color='red', linestyle='--', linewidth=2, label='Threshold')
ax1.set_xlabel('Quality Score')
ax1.set_ylabel('Frequency')
ax1.set_title('Quality Score Distribution')
ax1.legend()
ax1.grid(alpha=0.3)

# Summary statistics
stats_data = {
    'Mean': [np.mean(unfiltered_scores), np.mean(filtered_scores)],
    'Median': [np.median(unfiltered_scores), np.median(filtered_scores)],
    'Std': [np.std(unfiltered_scores), np.std(filtered_scores)],
}

x = np.arange(len(stats_data))
width = 0.35

for i, (metric, values) in enumerate(stats_data.items()):
    ax2.bar(i - width/2, values[0], width, label='Unfiltered' if i == 0 else '', color='steelblue', alpha=0.7)
    ax2.bar(i + width/2, values[1], width, label='Filtered' if i == 0 else '', color='green', alpha=0.7)

ax2.set_ylabel('Score')
ax2.set_title('Quality Metrics Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(stats_data.keys())
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("Quality Filtering Summary:")
print(f"Unfiltered - Mean: {np.mean(unfiltered_scores):.3f}, Median: {np.median(unfiltered_scores):.3f}, Std: {np.std(unfiltered_scores):.3f}")
print(f"Filtered   - Mean: {np.mean(filtered_scores):.3f}, Median: {np.median(filtered_scores):.3f}, Std: {np.std(filtered_scores):.3f}")
print(f"Filtering removed {len(unfiltered_scores) - len(filtered_scores)}/{len(unfiltered_scores)} samples")


# ======================================================================
# ## Key Takeaways
# **Core Idea:** Synthetic data generation bootstraps training data via templates, seed instructions, and language models, with quality filtering to maintain data quality.
# **Approaches and When to Use:**
# ======================================================================

# ======================================================================
# ## Try It Yourself
# 1. **Vary template diversity:** Modify BasicSelfInstruct.templates and observe how diversity metrics change.
# 2. **Adjust quality threshold:** Run AdvancedSyntheticGenerator with different thresholds (0.4, 0.6, 0.8) and compare filtering results.
# 3. **Add domain-specific filters:** Extend QualityClassifier with checks for your domain (e.g., code syntax validation).
# ======================================================================
