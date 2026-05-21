"""
Auto-generated from 10-model-merging.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Model Merging: Combining Fine-Tuned Adapters
# ## Learning Objectives
# 1. Implement SLERP and Task Arithmetic merging from scratch using numpy
# 2. Merge multiple fine-tuned LoRA adapters into a single model
# 3. Measure task performance trade-offs when merging multiple models
# 4. Build interpolation strategies to balance multi-task performance
# ======================================================================

import numpy as np
import torch
import time
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ======================================================================
# ## Level 1: Basic Merging (SLERP and Task Arithmetic)
# ======================================================================

class BasicModelMerging:
    """Implement SLERP and Task Arithmetic merging"""
    
    @staticmethod
    def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical Linear Interpolation between vectors.
        Smooth interpolation on sphere, preserves magnitude.
        """
        # Normalize vectors
        v0_norm = v0 / (np.linalg.norm(v0) + 1e-8)
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        
        # Compute angle between vectors
        cos_theta = np.dot(v0_norm, v1_norm)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Clamp for numerical stability
        theta = np.arccos(cos_theta)
        
        # Handle case where vectors are parallel
        if abs(theta) < 1e-6:
            return v0 * (1 - t) + v1 * t
        
        # SLERP formula
        sin_theta = np.sin(theta)
        w0 = np.sin((1 - t) * theta) / sin_theta
        w1 = np.sin(t * theta) / sin_theta
        
        return w0 * v0 + w1 * v1
    
    @staticmethod
    def task_arithmetic(base: np.ndarray, task_models: List[np.ndarray],
                       weights: List[float]) -> np.ndarray:
        """
        Task Arithmetic: merge adapters as task vectors.
        merged = base + sum(weights[i] * (task[i] - base))
        """
        # Compute task vectors (differences from base)
        task_vectors = [task - base for task in task_models]
        
        # Weighted sum of task vectors
        merged_task_vector = np.zeros_like(base)
        for weight, task_vec in zip(weights, task_vectors):
            merged_task_vector += weight * task_vec
        
        # Add to base
        merged = base + merged_task_vector
        return merged
    
    def test_merging(self):
        """Test merging strategies"""
        # Create mock weight matrices
        np.random.seed(42)
        base = np.random.randn(100)
        task1 = base + np.random.randn(100) * 0.1  # Small deviation
        task2 = base + np.random.randn(100) * 0.1
        
        # Test SLERP
        merged_slerp = self.slerp(task1, task2, t=0.5)
        
        # Test Task Arithmetic
        merged_ta = self.task_arithmetic(base, [task1, task2], [0.5, 0.5])
        
        return {
            'base': base,
            'task1': task1,
            'task2': task2,
            'merged_slerp': merged_slerp,
            'merged_ta': merged_ta
        }

# Test basic merging
merger = BasicModelMerging()
results = merger.test_merging()

# Compute distances
dist_slerp_to_base = np.linalg.norm(results['merged_slerp'] - results['base'])
dist_ta_to_base = np.linalg.norm(results['merged_ta'] - results['base'])

print(f"Merging Results:")
print(f"Distance from base (SLERP): {dist_slerp_to_base:.3f}")
print(f"Distance from base (Task Arithmetic): {dist_ta_to_base:.3f}")


# Visualize merging
t_values = np.linspace(0, 1, 50)
slerp_distances = []
ta_distances = []

for t in t_values:
    merged_s = merger.slerp(results['task1'], results['task2'], t)
    merged_t = merger.task_arithmetic(results['base'], 
                                     [results['task1'], results['task2']], 
                                     [t, 1-t])
    slerp_distances.append(np.linalg.norm(merged_s))
    ta_distances.append(np.linalg.norm(merged_t))

plt.figure(figsize=(10, 5))
plt.plot(t_values, slerp_distances, 'o-', label='SLERP', alpha=0.7, markersize=4)
plt.plot(t_values, ta_distances, 's-', label='Task Arithmetic', alpha=0.7, markersize=4)
plt.xlabel('Interpolation Ratio (t)')
plt.ylabel('Merged Model Norm')
plt.title('Merging Strategy Comparison')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ======================================================================
# ## Level 2: Advanced Multi-Model Merging with Validation
# ======================================================================

class AdvancedModelMerging:
    """Merge multiple fine-tuned models and measure task performance"""
    
    def __init__(self):
        np.random.seed(42)
        # Create base model
        self.base_model = np.random.randn(200)
        
        # Create task-specific models
        self.task_models = [
            self.base_model + np.random.randn(200) * 0.15,  # Task 1
            self.base_model + np.random.randn(200) * 0.15,  # Task 2
            self.base_model + np.random.randn(200) * 0.15,  # Task 3
        ]
        
        # Task-specific ground truth weights
        self.task_ground_truth = [
            np.array([1.0, 0.0, 0.0]),  # Task 1 best
            np.array([0.0, 1.0, 0.0]),  # Task 2 best
            np.array([0.0, 0.0, 1.0]),  # Task 3 best
        ]
    
    def task_arithmetic_merge(self, weights: List[float]) -> np.ndarray:
        """Merge models using Task Arithmetic"""
        task_vectors = [m - self.base_model for m in self.task_models]
        merged_vector = np.zeros_like(self.base_model)
        
        for w, tv in zip(weights, task_vectors):
            merged_vector += w * tv
        
        return self.base_model + merged_vector
    
    def measure_task_performance(self, merged_model: np.ndarray) -> List[float]:
        """
        Measure performance on each task.
        Similarity to task-specific model.
        """
        performances = []
        
        for task_model in self.task_models:
            # Cosine similarity as proxy for performance
            sim = np.dot(merged_model, task_model) / (
                np.linalg.norm(merged_model) * np.linalg.norm(task_model) + 1e-8
            )
            performances.append(max(0, sim))  # Clamp to [0, 1]
        
        return performances
    
    def sweep_merge_ratios(self) -> List[Dict]:
        """Sweep interpolation ratios and measure trade-offs"""
        results = []
        
        # Equal weight merge
        for t in np.linspace(0, 1, 11):
            weights = [t, (1-t)/2, (1-t)/2]  # Task 1 emphasis
            merged = self.task_arithmetic_merge(weights)
            perf = self.measure_task_performance(merged)
            
            results.append({
                'weights': weights,
                'task_perf': perf,
                'avg_perf': np.mean(perf)
            })
        
        return results

# Test advanced merging
adv_merger = AdvancedModelMerging()
sweep_results = adv_merger.sweep_merge_ratios()

print("Task Performance Sweep:")
for i, result in enumerate(sweep_results[::2]):  # Print every other
    print(f"Weights: {[f'{w:.1f}' for w in result['weights']]}")
    print(f"  Task Performance: {[f'{p:.2f}' for p in result['task_perf']]}")
    print(f"  Average: {result['avg_perf']:.2f}")


# Visualize performance trade-off
tasks_perf = [r['task_perf'] for r in sweep_results]
task_names = ['Task 1', 'Task 2', 'Task 3']
weights = [r['weights'][0] for r in sweep_results]

plt.figure(figsize=(10, 5))
for i, task_name in enumerate(task_names):
    task_scores = [p[i] for p in tasks_perf]
    plt.plot(weights, task_scores, 'o-', label=task_name, linewidth=2, markersize=6)

plt.xlabel('Task 1 Weight (merging emphasis)')
plt.ylabel('Performance')
plt.title('Task Performance Trade-off in Model Merging')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ======================================================================
# ## Real-World Example 1: Fine-tune GPT2 on Two Tasks, Merge with SLERP
# ======================================================================

class GPT2TaskMerging:
    """Mock GPT2 merging simulation"""
    
    def __init__(self):
        # Simulate GPT2 weights
        np.random.seed(42)
        self.base_weights = {}
        for layer in range(3):
            self.base_weights[f'layer_{layer}'] = np.random.randn(100, 100) * 0.01
    
    def finetune_task(self, task_name: str, task_data_size: int) -> Dict:
        """Mock fine-tuning: add task-specific noise"""
        task_weights = {}
        
        for key, base_w in self.base_weights.items():
            # Task-specific fine-tuning adds small perturbations
            perturbation = np.random.randn(*base_w.shape) * (0.01 * np.sqrt(task_data_size))
            task_weights[key] = base_w + perturbation
        
        return task_weights
    
    def merge_models(self, model1: Dict, model2: Dict, t: float = 0.5) -> Dict:
        """Merge two fine-tuned models"""
        merged = {}
        
        for key in model1.keys():
            # Simple linear interpolation for now
            merged[key] = (1 - t) * model1[key] + t * model2[key]
        
        return merged
    
    def evaluate_on_task(self, weights: Dict, task_id: int) -> float:
        """Mock evaluation"""
        # Simulate task-specific accuracy
        task_similarity = 0.7 + (task_id * 0.05)
        return task_similarity

# Test GPT2 merging
gpt2_merger = GPT2TaskMerging()

# Fine-tune on two tasks
task1_weights = gpt2_merger.finetune_task('Classification', task_data_size=1000)
task2_weights = gpt2_merger.finetune_task('QA', task_data_size=1000)

# Merge with different ratios
merge_results = []
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    merged = gpt2_merger.merge_models(task1_weights, task2_weights, t=t)
    perf1 = gpt2_merger.evaluate_on_task(merged, task_id=0)
    perf2 = gpt2_merger.evaluate_on_task(merged, task_id=1)
    
    merge_results.append({
        'ratio': t,
        'task1_perf': perf1,
        'task2_perf': perf2,
        'avg_perf': (perf1 + perf2) / 2
    })

print("GPT2 Task Merging:")
for result in merge_results:
    print(f"Task1 weight={result['ratio']:.2f}: "
          f"Task1={result['task1_perf']:.3f}, "
          f"Task2={result['task2_perf']:.3f}, "
          f"Avg={result['avg_perf']:.3f}")


# ======================================================================
# ## Real-World Example 2: DARE Merging Strategy
# ======================================================================

class DAREMerging:
    """DARE: Drop And REscale merging"""
    
    @staticmethod
    def dare_merge(base: np.ndarray, task_adapters: List[np.ndarray],
                  weights: List[float], drop_rate: float = 0.5) -> np.ndarray:
        """
        DARE merging: drop elements of task vectors randomly, then rescale.
        This reduces redundancy when merging multiple adapters.
        """
        merged = base.copy()
        
        for weight, adapter in zip(weights, task_adapters):
            # Compute task vector
            task_vector = adapter - base
            
            # Drop random elements
            mask = np.random.rand(*task_vector.shape) > drop_rate
            dropped_vector = task_vector * mask
            
            # Rescale to account for dropped elements
            rescale = 1.0 / (1.0 - drop_rate + 1e-8)
            rescaled_vector = dropped_vector * rescale
            
            # Add to merged model
            merged += weight * rescaled_vector
        
        return merged
    
    def compare_merge_strategies(self):
        """Compare standard merging vs DARE"""
        np.random.seed(42)
        base = np.random.randn(100)
        adapters = [base + np.random.randn(100) * 0.2 for _ in range(3)]
        weights = [0.33, 0.33, 0.34]
        
        # Standard merging
        merged_std = base + sum(w * (a - base) for w, a in zip(weights, adapters))
        
        # DARE merging
        merged_dare = self.dare_merge(base, adapters, weights, drop_rate=0.5)
        
        return {
            'std_norm': np.linalg.norm(merged_std),
            'dare_norm': np.linalg.norm(merged_dare),
            'std_model': merged_std,
            'dare_model': merged_dare
        }

dare = DAREMerging()
dare_result = dare.compare_merge_strategies()

print(f"Merging Strategy Comparison:")
print(f"Standard Merging Norm: {dare_result['std_norm']:.3f}")
print(f"DARE Merging Norm: {dare_result['dare_norm']:.3f}")


# ======================================================================
# ## Real-World Example 3: Three-Model Merging and Trade-off Analysis
# ======================================================================

class TripleModelMerging:
    """Merge three fine-tuned models optimally"""
    
    def __init__(self):
        np.random.seed(42)
        self.base = np.random.randn(150)
        self.models = [
            self.base + np.random.randn(150) * 0.15,
            self.base + np.random.randn(150) * 0.15,
            self.base + np.random.randn(150) * 0.15,
        ]
    
    def merge_with_weights(self, weights: Tuple[float, float, float]) -> np.ndarray:
        """Task arithmetic with specified weights"""
        merged = self.base.copy()
        for w, model in zip(weights, self.models):
            merged += w * (model - self.base)
        return merged
    
    def evaluate_merged_model(self, merged: np.ndarray) -> Tuple[float, float, float]:
        """Evaluate on three tasks"""
        perf = []
        for model in self.models:
            # Cosine similarity
            sim = np.dot(merged, model) / (np.linalg.norm(merged) * np.linalg.norm(model) + 1e-8)
            perf.append(max(0.5, sim))  # Min 0.5
        return tuple(perf)
    
    def find_optimal_merge(self, num_samples: int = 50) -> Dict:
        """Find merge weights that maximize some metric"""
        best_result = None
        best_score = -np.inf
        results = []
        
        # Grid search over weight combinations
        for i in range(num_samples):
            # Random weights summing to 1
            w = np.random.dirichlet([1, 1, 1])
            merged = self.merge_with_weights(tuple(w))
            perf = self.evaluate_merged_model(merged)
            
            # Metric: balance between tasks
            avg_perf = np.mean(perf)
            min_perf = np.min(perf)  # Don't let any task fail
            score = avg_perf * 0.7 + min_perf * 0.3  # Weighted metric
            
            results.append({
                'weights': w,
                'perf': perf,
                'avg': avg_perf,
                'min': min_perf,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_result = results[-1]
        
        return {
            'best': best_result,
            'all_results': results
        }

# Find optimal merge
triple_merger = TripleModelMerging()
optimal = triple_merger.find_optimal_merge(num_samples=100)

print(f"Optimal Merge Weights: {optimal['best']['weights']}")
print(f"Task Performance: {[f'{p:.2f}' for p in optimal['best']['perf']]}")
print(f"Average: {optimal['best']['avg']:.3f}, Min: {optimal['best']['min']:.3f}")


# ======================================================================
# ## Comparison: Task Accuracy Heatmap and Trade-offs
# ======================================================================

# Create heatmap of performance across merge ratios
ratios = np.linspace(0, 1, 15)
all_results = []

for t1 in np.linspace(0, 1, 8):
    for t2 in np.linspace(0, 1 - t1, 8):
        t3 = 1 - t1 - t2
        merged = triple_merger.merge_with_weights((t1, t2, t3))
        perf = triple_merger.evaluate_merged_model(merged)
        all_results.append({
            'w1': t1, 'w2': t2, 'w3': t3,
            'p1': perf[0], 'p2': perf[1], 'p3': perf[2]
        })

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for task_idx, ax in enumerate(axes):
    perfs = [r[f'p{task_idx+1}'] for r in all_results]
    weights_1 = [r['w1'] for r in all_results]
    
    scatter = ax.scatter(weights_1, perfs, c=perfs, cmap='viridis', s=100, alpha=0.6)
    ax.set_xlabel('Model 1 Weight')
    ax.set_ylabel(f'Task {task_idx+1} Performance')
    ax.set_title(f'Task {task_idx+1} Performance vs Merge Weights')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()

print(f"Analyzed {len(all_results)} merge configurations")


# ======================================================================
# ## Key Takeaways
# **Core Idea:** Model merging combines multiple fine-tuned adapters into a single model, enabling multi-task capabilities without training separate models.
# **Methods:**
# | Method | Speed | Quality | Stability |
# |--------|-------|---------|----------|
# | Linear Interp | Very Fast | Moderate | High |
# | SLERP | Fast | Moderate-High | High |
# | Task Arithmetic | Fast | High | Medium |
# | DARE | Medium | Very High | Medium |
# **Trade-offs:**
# - **Task Coverage:** More weight on one task → worse on others
# - **Stability:** Merging dissimilar tasks harder
# - **Performance:** Individual task performance < single-task model
# **When to Merge:**
# - Multi-task serving (cost efficiency)
# - Limited storage/memory
# - Similar fine-tuned models
# - Avoid: Very different tasks, strict accuracy requirements
# **Related:**
# - [LoRA](./XX) – Efficient fine-tuning for merging
# - [RAFT](./09-raft-retrieval-augmented-finetuning.ipynb) – Fine-tuning strategy
# - [Quantization](./XX) – Compress after merging
# ======================================================================

# ======================================================================
# ## Try It Yourself
# 1. Experiment with different merge weights and measure task performance
# 2. Implement SLERP on your model weights
# 3. Compare merge quality with and without DARE dropping
# ======================================================================
