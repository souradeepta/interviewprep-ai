"""
Auto-generated from 18-llmops.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # LLMOps: Experiment Tracking and Version Control for Language Models
# ## Learning Objectives
# 1. Implement a mini experiment tracker with prompt versioning and hash-based change tracking
# 2. Build metrics logging and performance dashboards for prompt experiments
# 3. Analyze experiment correlation and identify which changes improved or degraded performance
# 4. Design rollback and regression detection mechanisms for production LLM systems
# ======================================================================

import numpy as np
import hashlib
import json
import time
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict
import copy

# Device setup
np.random.seed(42)
print('LLMOps Experiment Tracking System')
print(f'Timestamp: {datetime.now().isoformat()}')


# ======================================================================
# ## Level 1: Basic Experiment Tracker with Prompt Versioning
# ======================================================================

# Level 1: Basic prompt versioning and experiment tracking from scratch

class ExperimentTracker:
    """Basic experiment tracker with prompt versioning and metrics logging"""
    
    def __init__(self):
        self.experiments = []
        self.prompts = {}
        self.change_log = []
    
    def hash_prompt(self, prompt: str) -> str:
        """Generate hash for prompt versioning and reproducibility"""
        return hashlib.sha256(prompt.encode()).hexdigest()[:8]
    
    def register_prompt(self, name: str, prompt: str) -> Dict:
        """Register a new prompt version"""
        prompt_hash = self.hash_prompt(prompt)
        version = {
            'name': name,
            'hash': prompt_hash,
            'prompt': prompt,
            'timestamp': datetime.now().isoformat(),
            'length': len(prompt)
        }
        self.prompts[prompt_hash] = version
        self.change_log.append(f"Registered {name}: {prompt_hash}")
        return version
    
    def log_experiment(self, prompt_hash: str, metrics: Dict[str, float]) -> Dict:
        """Log experiment results with metrics"""
        experiment = {
            'prompt_hash': prompt_hash,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'experiment_id': len(self.experiments) + 1
        }
        self.experiments.append(experiment)
        return experiment
    
    def get_prompt_performance(self, prompt_hash: str) -> Dict:
        """Get aggregated statistics for a prompt version"""
        exp_for_prompt = [e for e in self.experiments if e['prompt_hash'] == prompt_hash]
        
        if not exp_for_prompt:
            return {'status': 'no_experiments'}
        
        all_metrics = {}
        for exp in exp_for_prompt:
            for metric_name, value in exp['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        stats = {}
        for metric_name, values in all_metrics.items():
            stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'min': np.min(values),
                'count': len(values)
            }
        
        return stats

# Test basic tracking
tracker = ExperimentTracker()

print("=== Basic Experiment Tracking ===")

# Register prompts
prompt_v1 = "Summarize the following text in one sentence:"
prompt_v2 = "Create a concise one-sentence summary of the text below:"
prompt_v3 = "Summarize in a single sentence, focusing on key insights:"

v1 = tracker.register_prompt('summarization_v1', prompt_v1)
v2 = tracker.register_prompt('summarization_v2', prompt_v2)
v3 = tracker.register_prompt('summarization_v3', prompt_v3)

print(f"\\nRegistered {len(tracker.prompts)} prompt versions")
for hash_id, version in tracker.prompts.items():
    print(f"  {version['name']}: {hash_id}")

# Log experiments
print("\\nLogging experiment results...")
tracker.log_experiment(v1['hash'], {'accuracy': 0.72, 'latency_ms': 450, 'cost': 0.008})
tracker.log_experiment(v2['hash'], {'accuracy': 0.78, 'latency_ms': 420, 'cost': 0.009})
tracker.log_experiment(v3['hash'], {'accuracy': 0.81, 'latency_ms': 480, 'cost': 0.010})
tracker.log_experiment(v2['hash'], {'accuracy': 0.79, 'latency_ms': 415, 'cost': 0.009})

print(f"Total experiments logged: {len(tracker.experiments)}")

# Get performance summary
print("\\n=== Performance by Prompt Version ===")
for hash_id, version in tracker.prompts.items():
    performance = tracker.get_prompt_performance(hash_id)
    print(f"\\n{version['name']} ({hash_id}):")
    for metric, stats in performance.items():
        if isinstance(stats, dict) and 'mean' in stats:
            print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")


# ======================================================================
# ### Output: Prompt Versions Tracked with Metrics
# ======================================================================

# ======================================================================
# ## Level 2: Advanced Experiment Analysis with Correlation and Regression Detection
# ======================================================================

# Level 2: Advanced analysis, correlation, rollback capability (80-100 lines)

class AdvancedExperimentAnalyzer:
    """Advanced analysis: correlation, regression detection, rollback"""
    
    def __init__(self):
        self.experiments = []
        self.baseline_metrics = None
    
    def add_experiment(self, exp_id: int, changes: List[str], metrics: Dict[str, float]):
        """Add experiment with description of changes"""
        exp = {
            'id': exp_id,
            'changes': changes,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.experiments.append(exp)
    
    def set_baseline(self, baseline_metrics: Dict[str, float]):
        """Set baseline metrics for comparison"""
        self.baseline_metrics = baseline_metrics
    
    def compute_metric_deltas(self):
        """Compute impact of each experiment vs baseline"""
        if self.baseline_metrics is None:
            return None
        
        deltas = []
        for exp in self.experiments:
            delta = {}
            for metric_name, value in exp['metrics'].items():
                baseline_val = self.baseline_metrics.get(metric_name, 0)
                if baseline_val != 0:
                    pct_change = (value - baseline_val) / baseline_val * 100
                else:
                    pct_change = (value - baseline_val) * 100
                delta[metric_name] = {'value': value, 'pct_change': pct_change}
            exp['deltas'] = delta
            deltas.append({'id': exp['id'], 'changes': exp['changes'], 'deltas': delta})
        
        return deltas
    
    def detect_regression(self, metric: str, threshold: float = -5.0) -> List[int]:
        """Detect experiments that regressed on a metric"""
        regressions = []
        for exp in self.experiments:
            if 'deltas' in exp and metric in exp['deltas']:
                pct_change = exp['deltas'][metric]['pct_change']
                if pct_change < threshold:
                    regressions.append(exp['id'])
        return regressions
    
    def get_rollback_info(self, target_exp_id: int) -> Dict:
        """Get information for rolling back to a previous experiment"""
        target_exp = next((e for e in self.experiments if e['id'] == target_exp_id), None)
        if not target_exp:
            return {'status': 'not_found'}
        
        rollback_ids = [e['id'] for e in self.experiments if e['id'] > target_exp_id]
        
        return {
            'rollback_target': target_exp_id,
            'metrics_at_rollback': target_exp['metrics'],
            'experiments_to_revert': rollback_ids,
            'reason': 'Regression detected'
        }

# Test advanced analyzer
print("\\n=== Advanced Experiment Analysis ===")
analyzer = AdvancedExperimentAnalyzer()

baseline = {'accuracy': 0.75, 'latency_ms': 500, 'cost': 0.010}
analyzer.set_baseline(baseline)
print(f"Baseline: {baseline}")

analyzer.add_experiment(1, ['added_system_prompt'], {'accuracy': 0.78, 'latency_ms': 520, 'cost': 0.011})
analyzer.add_experiment(2, ['improved_examples'], {'accuracy': 0.82, 'latency_ms': 490, 'cost': 0.010})
analyzer.add_experiment(3, ['verbose_instructions'], {'accuracy': 0.71, 'latency_ms': 580, 'cost': 0.012})
analyzer.add_experiment(4, ['chain_of_thought'], {'accuracy': 0.85, 'latency_ms': 650, 'cost': 0.015})

deltas = analyzer.compute_metric_deltas()
print("\\n=== Experiment Impact Analysis ===")
for delta_info in deltas:
    exp_id = delta_info['id']
    print(f"\\nExp {exp_id} - Changes: {delta_info['changes']}")
    for metric, change_info in delta_info['deltas'].items():
        pct = change_info['pct_change']
        direction = '↑' if pct > 0 else '↓'
        print(f"  {metric}: {change_info['value']:.3f} ({direction} {abs(pct):+.1f}%)")

regressions = analyzer.detect_regression('accuracy', threshold=-5.0)
print(f"\\n=== Regression Detection ===")
print(f"Regressions (accuracy < -5%): {regressions}")


# ======================================================================
# ### Output: Experiment Correlation and Regression Analysis
# ======================================================================

# ======================================================================
# ## Real-World Example 1: A/B Testing with Prompt Variants
# ======================================================================

# Real-World Example 1: A/B testing different prompt versions (50-60 lines)

class ABTestRunner:
    """Run A/B tests comparing prompt variants"""
    
    def simulate_prompt_performance(self, prompt_id: str, num_samples: int = 100) -> Dict:
        """Simulate running prompt and collecting metrics"""
        np.random.seed(hash(prompt_id) % 2**32)
        
        base_accuracy = 0.75
        if 'detailed' in prompt_id:
            base_accuracy += 0.08
        if 'examples' in prompt_id:
            base_accuracy += 0.05
        
        accuracies = np.random.normal(base_accuracy, 0.05, num_samples)
        accuracies = np.clip(accuracies, 0, 1)
        latencies = np.random.lognormal(6, 0.3, num_samples)
        
        return {
            'prompt_id': prompt_id,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'mean_latency_ms': np.mean(latencies),
            'samples': num_samples
        }
    
    def run_ab_test(self, variant_a: str, variant_b: str) -> Dict:
        """Run A/B test with statistical significance"""
        results_a = self.simulate_prompt_performance(variant_a, num_samples=200)
        results_b = self.simulate_prompt_performance(variant_b, num_samples=200)
        
        diff = results_b['mean_accuracy'] - results_a['mean_accuracy']
        se = np.sqrt(results_a['std_accuracy']**2 + results_b['std_accuracy']**2) / np.sqrt(100)
        z_score = diff / se if se > 0 else 0
        p_value = 0.05 if abs(z_score) > 1.96 else 0.1
        winner = 'B' if diff > 0 else 'A'
        
        return {
            'variant_a': results_a,
            'variant_b': results_b,
            'accuracy_diff': diff,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05,
            'winner': f'Variant {winner}',
            'confidence': 'High' if p_value < 0.05 else 'Low'
        }

print("\\n=== A/B Test: Comparing Prompt Variants ===")
ab_tester = ABTestRunner()

result = ab_tester.run_ab_test(
    'simple_prompt',
    'detailed_prompt_with_examples'
)

print(f"Variant A (Baseline): {result['variant_a']['mean_accuracy']:.3f} ± {result['variant_a']['std_accuracy']:.3f}")
print(f"Variant B (Experimental): {result['variant_b']['mean_accuracy']:.3f} ± {result['variant_b']['std_accuracy']:.3f}")
print(f"\\nAccuracy Difference: {result['accuracy_diff']:+.3f}")
print(f"Winner: {result['winner']} (Confidence: {result['confidence']})")


# ======================================================================
# ## Real-World Example 2: Experiment Correlation Analysis
# ======================================================================

# Real-World Example 2: Identify which changes improved which metrics (50-60 lines)

class CorrelationAnalyzer:
    """Analyze which changes correlate with metric improvements"""
    
    def __init__(self):
        self.experiments = []
        self.changes = defaultdict(list)
    
    def add_experiment(self, changes_dict: Dict[str, bool], metrics: Dict[str, float]):
        """Add experiment with feature flags"""
        exp = {'changes': changes_dict, 'metrics': metrics}
        self.experiments.append(exp)
        for change, enabled in changes_dict.items():
            if enabled:
                self.changes[change].append(metrics)
    
    def compute_change_impact(self) -> Dict[str, Dict[str, float]]:
        """Compute average metric impact of each change"""
        all_metrics = self.experiments[0]['metrics'].keys()
        
        impact = {}
        for change, metric_lists in self.changes.items():
            impact[change] = {}
            for metric in all_metrics:
                values = [m[metric] for m in metric_lists if metric in m]
                if values:
                    impact[change][metric] = np.mean(values)
        
        return impact

print("\\n=== Experiment Correlation Analysis ===")
corr_analyzer = CorrelationAnalyzer()

exp_configs = [
    ({'system_prompt': True, 'few_shot': False, 'cot': False}, {'acc': 0.76, 'latency': 450}),
    ({'system_prompt': True, 'few_shot': True, 'cot': False}, {'acc': 0.82, 'latency': 480}),
    ({'system_prompt': False, 'few_shot': True, 'cot': False}, {'acc': 0.78, 'latency': 420}),
    ({'system_prompt': True, 'few_shot': True, 'cot': True}, {'acc': 0.88, 'latency': 620}),
]

for changes, metrics in exp_configs:
    corr_analyzer.add_experiment(changes, metrics)

impact = corr_analyzer.compute_change_impact()

print(f"\\n{'Change':<20} {'Avg Accuracy':<15} {'Avg Latency':<15}")
print("-" * 50)
for change, metrics in sorted(impact.items()):
    acc_str = f"{metrics.get('acc', 0):.3f}" if 'acc' in metrics else "—"
    lat_str = f"{metrics.get('latency', 0):.0f}" if 'latency' in metrics else "—"
    print(f"{change:<20} {acc_str:<15} {lat_str:<15}")


# ======================================================================
# ## Real-World Example 3: Rollback and Version Control
# ======================================================================

# Real-World Example 3: Version control and rollback mechanism (50-60 lines)

class VersionControlSystem:
    """Simple version control for prompts with rollback"""
    
    def __init__(self):
        self.versions = []
        self.current_version = None
    
    def create_version(self, prompt: str, changes: List[str], metrics: Dict) -> Dict:
        """Create a new prompt version"""
        version_num = len(self.versions) + 1
        version = {
            'version': version_num,
            'prompt': prompt,
            'changes': changes,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'is_deployed': False
        }
        self.versions.append(version)
        return version
    
    def deploy_version(self, version_num: int):
        """Deploy a version to production"""
        if self.current_version is not None:
            self.versions[self.current_version - 1]['is_deployed'] = False
        
        if version_num <= len(self.versions):
            self.versions[version_num - 1]['is_deployed'] = True
            self.current_version = version_num
            return {'status': 'deployed', 'version': version_num}
        return {'status': 'error'}
    
    def rollback(self, steps: int = 1):
        """Rollback to previous version"""
        if self.current_version is None:
            return {'status': 'error'}
        
        target = self.current_version - steps
        return self.deploy_version(target) if target >= 1 else {'status': 'error'}
    
    def get_version_diff(self, v1: int, v2: int) -> Dict:
        """Show diff between versions"""
        ver1 = self.versions[v1 - 1] if v1 <= len(self.versions) else None
        ver2 = self.versions[v2 - 1] if v2 <= len(self.versions) else None
        
        if not ver1 or not ver2:
            return {'status': 'error'}
        
        metric_diffs = {}
        for metric in ver2['metrics']:
            if metric in ver1['metrics']:
                diff = ver2['metrics'][metric] - ver1['metrics'][metric]
                metric_diffs[metric] = {'v1': ver1['metrics'][metric], 'v2': ver2['metrics'][metric], 'diff': diff}
        
        return {'from_version': v1, 'to_version': v2, 'metric_differences': metric_diffs}

print("\\n=== Version Control and Rollback ===")
vcs = VersionControlSystem()

v1 = vcs.create_version('Summarize: ', ['initial'], {'accuracy': 0.75, 'latency_ms': 400})
v2 = vcs.create_version('Summarize in one sentence: ', ['added_constraint'], {'accuracy': 0.78, 'latency_ms': 410})
v3 = vcs.create_version('Create a one-sentence summary: ', ['improved_instructions'], {'accuracy': 0.83, 'latency_ms': 420})

vcs.deploy_version(3)
print(f"Deployed v3 to production")

diff = vcs.get_version_diff(1, 3)
print(f"\\nDiff v1 → v3:")
for metric, change in diff['metric_differences'].items():
    pct = (change['diff'] / change['v1'] * 100) if change['v1'] != 0 else 0
    print(f"  {metric}: {change['v1']:.3f} → {change['v2']:.3f} ({pct:+.1f}%)")

print(f"\\nRolling back...")
vcs.rollback(steps=1)
print(f"Current version: v{vcs.current_version}")


# ======================================================================
# ## Comparison: Experiment Dashboard
# ======================================================================

import matplotlib.pyplot as plt

# Simulate experiment history
experiments = [
    {'version': 1, 'accuracy': 0.75, 'latency': 400, 'cost': 0.008},
    {'version': 2, 'accuracy': 0.76, 'latency': 420, 'cost': 0.009},
    {'version': 3, 'accuracy': 0.81, 'latency': 450, 'cost': 0.010},
    {'version': 4, 'accuracy': 0.83, 'latency': 480, 'cost': 0.011},
    {'version': 5, 'accuracy': 0.79, 'latency': 520, 'cost': 0.013},
    {'version': 6, 'accuracy': 0.85, 'latency': 500, 'cost': 0.012},
]

versions = [e['version'] for e in experiments]
accuracies = [e['accuracy'] for e in experiments]
latencies = [e['latency'] for e in experiments]
costs = [e['cost'] for e in experiments]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(versions, accuracies, 'o-', linewidth=2, markersize=8, color='green')
axes[0, 0].axhline(y=0.75, color='r', linestyle='--', alpha=0.5)
axes[0, 0].set_title('Accuracy Over Versions')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(versions, latencies, 's-', linewidth=2, markersize=8, color='orange')
axes[0, 1].set_title('Latency Trend')
axes[0, 1].set_ylabel('Latency (ms)')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].bar(versions, costs, color='steelblue', alpha=0.7)
axes[1, 0].set_title('Cost Tracking')
axes[1, 0].set_ylabel('Cost per Request ($)')
axes[1, 0].grid(True, alpha=0.3, axis='y')

axes[1, 1].scatter(latencies, accuracies, s=200, alpha=0.6, c=versions, cmap='viridis')
for i, v in enumerate(versions):
    axes[1, 1].annotate(f'v{v}', (latencies[i], accuracies[i]), fontsize=9)
axes[1, 1].set_title('Quality vs Speed Trade-off')
axes[1, 1].set_xlabel('Latency (ms)')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/llmops_dashboard.png', dpi=100, bbox_inches='tight')
plt.show()

print("\\n=== Experiment Performance Summary ===")
print(f"Best version: v{versions[accuracies.index(max(accuracies))]} (Accuracy: {max(accuracies):.3f})")


# ======================================================================
# ## Key Takeaways
# ======================================================================

# ======================================================================
# ### Core Concept
# LLMOps is systematic experiment tracking: prompt versioning (hashing), metrics logging, experiment correlation, regression detection, and automated rollback. Essential for production LLM systems with frequent prompt updates.
# ### Production Patterns
# 1. **Prompt Versioning:** Hash prompts (SHA256) for reproducibility and change tracking
# 2. **Track Metrics:** Accuracy, latency (p50/p95/p99), token cost per request
# 3. **A/B Testing:** Statistical significance (p < 0.05) required before deploy
# 4. **Regression Detection:** Automatic alerts for >5% metric degradation
# 5. **Version Control:** Enable quick rollback (single command)
# ======================================================================

# ======================================================================
# ## Exercises: Try It Yourself
# 1. **Multi-metric A/B testing:** Track accuracy, latency, and cost. Implement composite decision rule.
# 2. **Correlation analysis:** Add 8+ feature combinations. Identify optimal combo.
# 3. **Version history:** Create 10 versions with simulated metrics. Plot quality over time.
# 4. **Automated alerts:** Set up regression detection for >5% accuracy drop.
# ======================================================================
