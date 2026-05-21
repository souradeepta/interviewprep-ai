"""
Auto-generated from 08-hyperparameter-optimization.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Hyperparameter Optimization: Finding the Best Model Configuration
# ## Learning Objectives
# - Understand search strategies: grid, random, Bayesian
# - Implement hyperparameter optimization efficiently
# ======================================================================

# ======================================================================
# ## Basic Implementation: Grid Search
# ======================================================================

import itertools
from typing import Dict, List, Tuple

class GridSearch:
    """Basic grid search over hyperparameter space"""
    
    def __init__(self, param_grid: Dict[str, list]):
        self.param_grid = param_grid
        self.results = []
    
    def run(self):
        """Try all combinations"""
        # Generate all combinations
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        total_combos = 1
        for v in values:
            total_combos *= len(v)
        
        print(f"Grid search: {total_combos} combinations")
        
        trial_num = 0
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            
            # Simulate training
            accuracy = self._simulate_model(params)
            
            self.results.append({
                'params': params,
                'accuracy': accuracy,
            })
            
            trial_num += 1
            if trial_num % 5 == 0:
                print(f"  Trial {trial_num}: {params} → acc={accuracy:.3f}")
    
    def _simulate_model(self, params: Dict) -> float:
        """Simulate training (in reality: train model)"""
        # Simple formula: accuracy improves with lr, batch_size
        lr = params.get('learning_rate', 0.01)
        bs = params.get('batch_size', 32)
        acc = 0.85 + 0.05 * (lr / 0.1) - 0.02 * (1 - (bs / 128))
        return min(0.99, max(0.7, acc))  # Clamp to [0.7, 0.99]
    
    def get_best(self) -> Dict:
        """Return best result"""
        return max(self.results, key=lambda x: x['accuracy'])

# Usage
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
}

search = GridSearch(param_grid)
search.run()

best = search.get_best()
print(f"\n✓ Best: {best['params']} → accuracy={best['accuracy']:.4f}")


# ======================================================================
# ## Advanced Implementation: Bayesian Optimization
# ======================================================================

import random
from typing import Callable

class BayesianOptimization:
    """Simplified Bayesian optimization"""
    
    def __init__(self, objective: Callable, param_bounds: Dict):
        self.objective = objective
        self.param_bounds = param_bounds
        self.trials = []
    
    def run(self, num_trials: int = 20):
        """Run Bayesian optimization"""
        print(f"Bayesian optimization: {num_trials} trials")
        
        for trial in range(num_trials):
            # Sample parameters (simplified: random initially, then refine)
            params = self._sample_params()
            
            # Evaluate
            score = self.objective(params)
            
            self.trials.append({'params': params, 'score': score})
            
            if trial % 5 == 0:
                best_so_far = max(self.trials, key=lambda x: x['score'])
                print(f"  Trial {trial}: {params} → score={score:.4f} (best: {best_so_far['score']:.4f})")
    
    def _sample_params(self) -> Dict:
        """Sample parameters (simplified)"""
        return {
            'learning_rate': random.uniform(0.001, 0.1),
            'batch_size': random.choice([32, 64, 128, 256]),
            'dropout': random.uniform(0.0, 0.5),
        }
    
    def get_best(self) -> Dict:
        return max(self.trials, key=lambda x: x['score'])

# Objective function
def objective(params: Dict) -> float:
    lr = params['learning_rate']
    bs = params['batch_size']
    dropout = params['dropout']
    # Accuracy: favors lr ~0.01, bs ~64, dropout ~0.2
    acc = 0.95 - (lr - 0.01)**2 * 1000 - (bs - 64)**2 / 1000 - (dropout - 0.2)**2 * 5
    return min(0.99, max(0.8, acc))

# Run
opt = BayesianOptimization(objective, {})
opt.run(num_trials=20)

best = opt.get_best()
print(f"\n✓ Best: {best['params']}")
print(f"  Score: {best['score']:.4f}")


# ======================================================================
# ## Real-World Example 1: Netflix Recommendation HPO
# ======================================================================

import pandas as pd
import numpy as np

def netflix_hpo_search():
    """Bayesian optimization for 1000+ hyperparameter trials"""

    print("NETFLIX: Hyperparameter Optimization Search")
    print("=" * 60)

    # Simulate Bayesian optimization trials
    np.random.seed(42)
    trials = pd.DataFrame({
        'trial': range(1, 51),
        'learning_rate': np.random.loguniform(0.0001, 0.1, 50),
        'batch_size': np.random.choice([32, 64, 128, 256], 50),
        'dropout': np.random.uniform(0.1, 0.5, 50),
        'accuracy': np.random.uniform(0.88, 0.96, 50),
    })

    print(f"\nTRIAL SUMMARY:")
    print(f"  Total trials: {len(trials)}")
    print(f"  Duration: 10 GPU-hours per trial = 500 GPU-hours total")
    print(f"  Parallelism: 10 GPUs = 50 hours wall time")
    print(f"  Search space size: ~1B combinations (sampled 50)")

    # Best trials
    best_5 = trials.nlargest(5, 'accuracy')
    print(f"\nTOP 5 TRIALS (By Accuracy):")
    print(best_5[['trial', 'learning_rate', 'batch_size', 'accuracy']].to_string(index=False))

    best = trials.loc[trials['accuracy'].idxmax()]
    print(f"\n\nBEST CONFIGURATION:")
    print(f"  Learning rate: {best['learning_rate']:.4f}")
    print(f"  Batch size: {best['batch_size']:.0f}")
    print(f"  Dropout: {best['dropout']:.2f}")
    print(f"  Accuracy: {best['accuracy']:.4f}")

    print(f"\nPARAMETER IMPORTANCE:")
    print(f"  learning_rate: HIGH (critical)")
    print(f"  batch_size: MEDIUM (affects convergence)")
    print(f"  dropout: LOW (minimal impact)")

netflix_hpo_search()



# ======================================================================
# ## Real-World Example 2: Stripe Fraud Model HPO
# ======================================================================

import numpy as np

def uber_early_stopping():
    """Early stopping to save compute during hyperparameter search"""

    print("UBER: Early Stopping in HPO")
    print("=" * 60)

    # Simulate two models: one trains well, one converges slowly
    epochs = 100

    # Model A: Good configuration (converges by epoch 15)
    model_a_loss = 1.0 - np.cumsum(np.random.uniform(0.005, 0.015, epochs))

    # Model B: Bad configuration (no improvement after epoch 10)
    model_b_loss = 1.0 - np.cumsum([0.02] + [0.001] * 99)  # minimal improvement

    print("\nEARLY STOPPING SIMULATION:")
    print(f"  Patience: 10 epochs (no improvement = stop)")
    print(f"  Full training: 100 epochs per model")
    print()

    # Model A with early stopping
    print("Model A (Good Config):")
    print(f"  Training epochs: 100 (completes)")
    print(f"  Final loss: {model_a_loss[-1]:.3f}")
    print(f"  Early stop at: N/A (keeps improving)")
    print()

    # Model B with early stopping
    patience = 10
    stopped_at = None
    for i in range(patience, len(model_b_loss)):
        if np.mean(np.diff(model_b_loss[i-patience:i])) < 0.0001:
            stopped_at = i
            break

    print("Model B (Bad Config):")
    print(f"  Training epochs: {stopped_at or 100}")
    print(f"  Final loss: {model_b_loss[stopped_at-1]:.3f}")
    print(f"  Early stop at: epoch {stopped_at} (no improvement for {patience} epochs)")
    print()

    print("SAVINGS:")
    print(f"  Model A: no savings (good config)")
    print(f"  Model B: {100-stopped_at} epochs saved = {(100-stopped_at)/100*100:.0f}% compute saved")
    print(f"  Across 50 trials: 50% of models get early stopped")
    print(f"  Total: 500 GPU-hours → ~200 GPU-hours")

uber_early_stopping()



# ======================================================================
# ## Real-World Example 3: Uber ETA Model HPO
# ======================================================================

def uber_hpo():
    print("Uber: ETA Model Hyperparameter Optimization")
    print()
    print("1. Optimization challenge:")
    print("   - Accuracy vs Latency trade-off")
    print("   - Deeper model = better accuracy, slower inference")
    print()
    print("2. Search space:")
    print("   - Tree depth: [3, 5, 7, 9, 11]")
    print("   - Learning rate: [0.01, 0.05, 0.1]")
    print("   - Subsample: [0.5, 0.7, 1.0]")
    print()
    print("3. Constraints:")
    print("   - Latency: must be <50ms per prediction")
    print("   - Accuracy: minimize MAE (mean absolute error)")
    print()
    print("4. Results:")
    print("   - Optimal depth: 7 (accuracy 85%, latency 45ms)")
    print("   - Depth 9: accuracy 86%, latency 65ms (too slow!)")
    print("   - Trade-off: 1% accuracy vs 15ms latency (chose depth 7)")
    print()
    print("5. Impact:")
    print("   - 2-3% ETA prediction improvement")
    print("   - Better user experience (more accurate predictions)")

uber_hpo()


# ======================================================================
# ## Interview Case Study: HPO for Fraud Detection
# ======================================================================

print("CASE STUDY: HYPERPARAMETER OPTIMIZATION FOR FRAUD DETECTION")
print()
print("SCENARIO:")
print("  Train XGBoost fraud detector")
print("  Uncertain about hyperparameters")
print("  Need: highest accuracy in <1 hour")
print()

print("SOLUTION:")
print()
print("1. Define search space:")
print("   max_depth: [4, 6, 8, 10]")
print("   learning_rate: [0.01, 0.05, 0.1]")
print("   subsample: [0.5, 0.7, 1.0]")
print("   colsample_bytree: [0.5, 0.7, 1.0]")
print("   Total: 4 × 3 × 3 × 3 = 108 combinations")
print()

print("2. Choose strategy:")
print("   - Grid search: 108 trials (easy to parallelize)")
print("   - 2 GPU-hours per trial = 216 GPU-hours total")
print("   - Parallelize on 16 GPUs = 14 hours (too long)")
print()
print("   Better: Random search with early stopping")
print("   - Sample 50 random configs (not all 108)")
print("   - Early stopping: kill trials if not improving by epoch 10")
print("   - Saves 50-80% of GPU time")
print()

print("3. Execute:")
print("   - Run 50 trials on 16 GPUs in parallel")
print("   - Wall time: ~2 hours (50 × 2h / 16 GPUs)")
print()

print("4. Analyze results:")
print("   - Best: max_depth=8, lr=0.05, subsample=0.7, cs=0.8 → acc=0.96")
print("   - Plot importance: max_depth > lr > subsample > cs")
print()

print("5. Retrain and deploy:")
print("   - Use best config on full training data")
print("   - Evaluate on test set: acc=0.95 (validation was 0.96)")
print("   - Deploy to production")
print()

print("STRONG ANSWER:")
print("  'Define search space with key parameters.")
print("  Random search with 50 trials (samples broadly).")
print("  Early stopping: kill unpromising trials (saves 70% compute).")
print("  Parallelize across GPUs (50 trials / 16 GPUs = 2 hours).")
print("  Analyze: which parameters matter (plot importance).")
print("  Retrain with best config on full data.")
print("  Typically finds 1-2% accuracy improvement.'")


# ======================================================================
# ## Key Takeaways
# **Search Strategies:**
# - Grid Search: exhaustive (all combinations), slow for large spaces
# - Random Search: samples randomly, faster, often finds good results
# ======================================================================
