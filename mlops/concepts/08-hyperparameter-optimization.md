# Hyperparameter Optimization: Finding the Best Model Configuration

## Comprehensive Overview

Hyperparameter optimization (HPO) is the systematic search for model configuration parameters (learning rate, batch size, tree depth, regularization) that maximize performance. Without optimization, teams make arbitrary choices: "learning rate is 0.01" (why?), "batch size is 64" (why not 32?). With optimization, they search parameter space systematically, finding configurations that achieve 1-2% higher accuracy—sometimes worth millions in business impact.

The cost of poor hyperparameter choices is real. A model trained with learning_rate=0.001 achieves 90% accuracy. With learning_rate=0.01, the same model achieves 94%. That 4% difference comes from one parameter, not from new code. Yet teams often overlook this—they retrain from scratch, build features, refactor code. With HPO, they try a few learning rates first (1 hour), find the best, then do other work.

Modern HPO uses intelligent search: grid search (try all combinations), random search (sample randomly), Bayesian optimization (learn from past trials, propose promising parameters), and evolutionary algorithms. Tools like Optuna, Ray Tune, and Hyperband automate this: specify search space, let them find best configuration. Many support distributed search: train 100 models in parallel on 100 machines.

The operational challenge is search scale: 10 parameters × 10 values each = 10B combinations. Can't try them all. Solutions: dimensionality reduction (focus on important parameters), early stopping (kill unpromising runs), parallelization (search across machines), and smart sampling (Bayesian optimization learns which regions to explore).

## How It Works

### Search Strategy Comparison

```
Grid Search:
  Parameters: learning_rate ∈ [0.001, 0.01, 0.1], batch_size ∈ [32, 64, 128]
  Combinations: 3 × 3 = 9
  Trials: 9 models trained
  Best: learning_rate=0.01, batch_size=64 → accuracy=0.95
  Cost: 9 GPU-hours

Random Search:
  Sample 20 random combinations from same space
  Trials: 20 models trained
  Often finds better result than grid (explores more space)
  Cost: 20 GPU-hours
  
Bayesian Optimization:
  Trial 1: learning_rate=0.005, batch_size=50 → accuracy=0.91
  Trial 2: learning_rate=0.02, batch_size=100 → accuracy=0.93 (good)
  Trial 3: learning_rate=0.015, batch_size=90 → accuracy=0.94 (better)
  ...refine search around promising regions...
  Trial 50: learning_rate=0.01, batch_size=64 → accuracy=0.96
  Cost: 50 GPU-hours, but finds better result
  
Early Stopping:
  Trial runs for 100 epochs normally
  If validation loss not improving by epoch 10 → stop, free GPU
  Saves 90% compute for bad configs
```

### HPO Workflow

```
Define Search Space:
  learning_rate: [0.0001, 0.1]
  batch_size: [16, 256]
  dropout: [0.0, 0.5]
  weight_decay: [0.0, 0.01]
    ↓
Select Search Strategy:
  Grid: try all (exhaustive)
  Random: sample randomly (broad exploration)
  Bayesian: learn from trials (efficient)
    ↓
Initialize Study:
  Optuna Study with objective function
  Goal: maximize validation accuracy
    ↓
Run Trials (Parallel):
  Trial 1: lr=0.001, bs=32, drop=0.1, wd=0.0 → val_acc=0.88
  Trial 2: lr=0.01, bs=64, drop=0.3, wd=0.005 → val_acc=0.92
  Trial 3: lr=0.005, bs=128, drop=0.2, wd=0.001 → val_acc=0.91
  Trial 4-100: ...continue trials...
    ↓
Analyze Results:
  Best: Trial 47: lr=0.01, bs=64, drop=0.3, wd=0.005 → val_acc=0.95
  Plot importance: learning_rate most important, batch_size second
  Remove: weight_decay has no effect
    ↓
Retrain on Best Config:
  Use full dataset (not just validation split)
  Final model: accuracy=0.96 (test set)
```

## Tool Comparisons

| Tool | Approach | Strengths | Weaknesses | Best For |
|------|----------|-----------|-----------|----------|
| **Optuna** | Bayesian, open-source | Simple API, strong Bayesian optimization, integrates with PyTorch | Less distributed scaling, smaller community | Small-medium teams, quick iteration |
| **Ray Tune** | Distributed, ML-focused | Excellent parallelization (100s of trials), integrates with Pytorch Lightning | Steeper learning curve, more infrastructure | Large-scale search, distributed teams |
| **Hyperband** | Multi-fidelity, efficient | Early stopping built-in, efficient resource use | Newer tool, smaller ecosystem | Cost-sensitive optimization |
| **Keras Tuner** | Integration, TensorFlow-native | Simple for Keras/TF models, easy to use | Limited to TensorFlow ecosystem | TensorFlow teams, Keras models |
| **Scikit-Optimize** | Bayesian, light-weight | Simple, good for sklearn models | Limited to sklearn, less sophisticated | Sklearn teams, classical ML |

**Decision Framework:**
- **Small team, quick:** Optuna (simplicity)
- **Large-scale, distributed:** Ray Tune (parallelization)
- **Keras/TensorFlow:** Keras Tuner (native integration)
- **Sklearn models:** Scikit-Optimize
- **Cost-optimized:** Hyperband (early stopping)

## Interview Q&A

**Q: You're training a model and unsure about hyperparameters. How do you approach optimization?**

A: (1) Define search space: which parameters matter (learning_rate, batch_size, regularization)? (2) Choose strategy: start with random search (explores broadly), move to Bayesian for refinement. (3) Run trials: ideally in parallel (100 configs × 1 GPU-hour = 100 GPU-hours, parallelized across 10 GPUs = 10 hours). (4) Analyze: plot importance (which parameters matter?), remove non-impactful. (5) Retrain: use best config on full data, evaluate on test set.

**Q: Hyperparameter search is slow. 10 trials/day, but you need 100 trials. How do you speed it up?**

A: (1) Parallelize: run 10 trials in parallel instead of serial (10 GPUs). (2) Early stopping: kill unpromising runs by epoch 10 (saves 90% compute). (3) Reduce dataset: validate on subset during search, full dataset for final retrain. (4) Focus parameters: which matter most? Optimize those first. (5) Multi-fidelity: Hyperband uses cheap approximations (fewer epochs) for initial filtering.

**Q: You found optimal learning_rate=0.01 on small data. Will it generalize to production (larger data)?**

A: Maybe not. Learning rate often depends on batch size: larger batches need higher learning rates (gradient noise lower). On small data: learning_rate=0.01 works. On production (10x more data, 10x bigger batches): might need learning_rate=0.05. Solution: (1) Optimize on representative data (similar scale to production). (2) Fine-tune on production data before deployment. (3) Monitor: if training curve looks off in production, re-optimize.

**Q: How do you handle categorical hyperparameters (model_type ∈ [xgboost, lightgbm, neural_net])?**

A: Bayesian optimization handles categorical parameters. Define search space: model_type as categorical choice. Run trials: each trial trains different model type. Analyze: accuracy by model type. However, may need nested optimization: (1) Compare model types (3 trials). (2) For best model_type, optimize hyperparameters (20 trials). Total: 23 trials, not 3×20=60.

**Q: You want to share optimal hyperparameters across teams. How do you document it?**

A: (1) Registry: central place for hyperparameters (database, wiki, config file). (2) Context: document what data/task these optimal for. (3) Domain: learning_rate=0.01 optimal for NLP tasks, might differ for vision. (4) Validation: show metrics (accuracy on this task with these hyperparams). (5) Evolution: as data/task change, hyperparams may need re-optimization.

## Best Practices

1. **Start Simple:** Grid or random search first (easy to parallelize, understand results). Move to Bayesian if needed.

2. **Define Range Wisely:** learning_rate ∈ [0.0001, 0.1] (log scale). Don't waste on obviously bad ranges.

3. **Validate on Holdout:** Use validation set during search. Test set only at the end.

4. **Parallelize:** Run multiple trials simultaneously. Cost: same total GPU-hours, time reduced by parallelism.

5. **Early Stopping:** Kill unpromising runs. Saves 50-90% compute.

6. **Parameter Importance:** Plot which parameters matter. Often 1-2 matter, others don't.

7. **Report All Results:** Show distribution of accuracies, not just best. Understand search difficulty.

8. **Reproducibility:** Fix random seed. Same search should yield same best hyperparameters.

## Common Pitfalls

1. **Searching Too Many Parameters:** Explosion of combinations. Focus on known important ones.

2. **Search on Test Set:** Optimizing on test data leads to overfitting to test set, poor generalization.

3. **Single Run:** Hyperparameter search has randomness. Run multiple times, report mean ± std.

4. **Ignoring Context:** Learned learning_rate on small data may not work on production scale.

5. **No Time Budget:** Searching indefinitely. Set: "200 GPU-hours max" and stop there.

6. **Missing Analysis:** Know which parameters matter. Reporting "best=0.95" without explaining why.

## Real-World Examples

### Netflix: Recommendation Model HPO

Netflix optimizes 100+ recommendation models:
- Search space: 5-10 important parameters per model
- Strategy: Bayesian optimization (learns from past models)
- Parallelization: 100 trials × 2 GPU-hours = 200 GPU-hours, parallel on 50 GPUs = 4 hours
- Early stopping: validates on 1M user sample, full validation on best
- Result: 1-2% accuracy improvement over default params (worth millions in engagement)

### Stripe: Fraud Model HPO

Stripe optimizes fraud models with cost constraints:
- Goal: high precision (minimize false declines) + high recall (catch fraud)
- Trade-off: changing threshold changes both
- Search: regularization strength (C ∈ [0.001, 100])
- Validation: monthly re-optimization (patterns change)
- Result: achieves 98% precision, 92% recall (improved fraud economics)

### Uber: Pricing Model HPO

Uber optimizes pricing model hyperparameters:
- Parameters: 3 key ones (regularization, depth, learning_rate)
- Scale: distributed search across 100 machines
- Cost: optimize for accuracy + latency (<50ms inference)
- Trade-off: deeper models (better accuracy, slower). Search finds sweet spot.

## Sample Interview Questions

1. "Design a hyperparameter optimization system for your team."

2. "100 models, each needs HPO. How do you prioritize?"

3. "Optimal hyperparameters changed when data grew 10x. Why? How do you reoptimize?"

## Interview Case Study

**Scenario:** You're training XGBoost models for fraud detection. Hyperparameter choices affect accuracy significantly. Design an HPO strategy.

**Solution Walkthrough:**

1. **Define Search Space:**
   ```
   max_depth: [2, 3, 4, 5, 6, 8, 10]
   learning_rate: [0.001, 0.01, 0.05, 0.1]
   subsample: [0.5, 0.7, 1.0]
   colsample_bytree: [0.5, 0.7, 1.0]
   ```

2. **Strategy:**
   - Grid search: 7 × 4 × 3 × 3 = 252 combinations (too many)
   - Random search: sample 50 random configs
   - Better: Bayesian optimization with 50 trials

3. **Parallelization:**
   ```
   Trials: 50
   GPU-hours per trial: 2
   Total: 100 GPU-hours
   Parallelism: run 10 trials in parallel on 10 machines
   Wall time: 10 hours (not 50 hours)
   ```

4. **Early Stopping:**
   ```
   Train for 100 boosting rounds normally
   If validation metric not improving for 10 rounds → stop
   Saves 90% for bad configs
   Actual: 100 GPU-hours with ES = 30-50 GPU-hours
   ```

5. **Analysis:**
   - Plot importance: max_depth > learning_rate > subsample
   - Remove: colsample_bytree has no effect
   - Best config: max_depth=6, learning_rate=0.05, subsample=0.7

6. **Retrain & Validate:**
   - Retrain on full training set (not validation split)
   - Evaluate on held-out test set
   - Compare with baseline (default hyperparams)
   - Document: "optimal hyperparams achieve 0.95 accuracy vs 0.91 baseline"

**Strong vs Weak Answers:**

Strong: "Define search space (important parameters only). Use Bayesian optimization with 50 trials. Parallelize across GPUs (50 trials = 10 hours on 10 GPUs). Early stopping: kill bad configs by epoch 10. Analyze importance: which parameters matter. Retrain on best config, validate on test set. Result: typically 1-2% accuracy improvement."

Weak: "Try learning_rate=0.1, 0.01, 0.001." (Only 3 values, no systematic approach, no parallelization, no analysis)

---

## Related Concepts

- **Concept 05:** Experiment Tracking — Logging HPO trials
- **Concept 08:** Hyperparameter Optimization — This concept
- **Concept 06:** Model Versioning — Storing best models from HPO

## Resources

- Optuna: https://optuna.org/
- Ray Tune: https://docs.ray.io/en/latest/tune/
- Hyperband: https://arxiv.org/abs/1603.06393
- Keras Tuner: https://keras.io/keras_tuner/
