"""
Auto-generated from 09-model-testing.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Model Testing: Comprehensive Validation for Production ML
# ## Learning Objectives
# - Implement different categories of model tests (correctness, robustness, fairness, performance)
# - Build a testing framework that catches bugs before production
# ======================================================================

# ======================================================================
# ## Basic Implementation: Unit Tests on Model Predictions
# ======================================================================

import numpy as np
from typing import Dict, List, Tuple
import json

class ModelTestSuite:
    """Basic model testing framework"""
    
    def __init__(self, model_predict_fn):
        """Initialize with a model's prediction function"""
        self.predict = model_predict_fn
        self.test_results = {}
    
    def test_correctness(self, test_cases: List[Dict]) -> bool:
        """Test: does model produce correct outputs on known inputs?"""
        print("Testing correctness...")
        passed = 0
        for case in test_cases:
            prediction = self.predict(case['input'])
            if prediction == case['expected']:
                passed += 1
        
        accuracy = passed / len(test_cases)
        self.test_results['correctness'] = {'passed': passed, 'total': len(test_cases), 'accuracy': accuracy}
        print(f"  ✓ Correctness: {passed}/{len(test_cases)} ({accuracy*100:.1f}%)")
        return accuracy >= 0.95  # Threshold: 95%+
    
    def test_consistency(self, input_val, num_runs: int = 10) -> bool:
        """Test: does model produce same output for same input?"""
        print("Testing consistency...")
        outputs = [self.predict(input_val) for _ in range(num_runs)]
        is_consistent = len(set(outputs)) == 1  # All outputs should be identical
        self.test_results['consistency'] = {'consistent': is_consistent, 'outputs': outputs}
        print(f"  ✓ Consistency: {'PASS' if is_consistent else 'FAIL'} - all outputs identical? {is_consistent}")
        return is_consistent
    
    def test_edge_cases(self, edge_cases: List[Dict]) -> Dict:
        """Test: how does model behave on edge cases?"""
        print("Testing edge cases...")
        results = {}
        for case in edge_cases:
            try:
                prediction = self.predict(case['input'])
                results[case['name']] = {'output': prediction, 'error': None}
                print(f"  ✓ {case['name']}: {prediction}")
            except Exception as e:
                results[case['name']] = {'output': None, 'error': str(e)}
                print(f"  ✗ {case['name']}: {type(e).__name__}: {e}")
        
        self.test_results['edge_cases'] = results
        return results
    
    def test_performance(self, test_data: List, timing_threshold_ms: float) -> bool:
        """Test: does model meet latency SLO?"""
        import time
        print(f"Testing latency (SLO: <{timing_threshold_ms}ms)...")
        
        times = []
        for sample in test_data:
            start = time.time()
            _ = self.predict(sample)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
        
        p50 = np.percentile(times, 50)
        p99 = np.percentile(times, 99)
        meets_slo = p99 < timing_threshold_ms
        
        self.test_results['performance'] = {
            'p50_ms': p50,
            'p99_ms': p99,
            'meets_slo': meets_slo
        }
        print(f"  ✓ Latency - p50: {p50:.1f}ms, p99: {p99:.1f}ms (SLO: {'PASS' if meets_slo else 'FAIL'})")
        return meets_slo
    
    def report(self):
        print("\n=== TEST REPORT ===")
        for test_name, result in self.test_results.items():
            print(f"{test_name.upper()}: {json.dumps(result, indent=2, default=str)}")

# Example: Binary classification model
def example_model(x):
    """Simple model: predict 1 if x > 0.5 else 0"""
    return 1 if x > 0.5 else 0

# Create test suite
tester = ModelTestSuite(example_model)

# Test 1: Correctness
test_cases = [
    {'input': 0.8, 'expected': 1},
    {'input': 0.2, 'expected': 0},
    {'input': 0.6, 'expected': 1},
    {'input': 0.3, 'expected': 0},
    {'input': 0.5, 'expected': 0},  # Edge: exactly at threshold
]
tester.test_correctness(test_cases)

# Test 2: Consistency
tester.test_consistency(0.7)

# Test 3: Edge cases
edge_cases = [
    {'name': 'negative_input', 'input': -0.5},
    {'name': 'max_value', 'input': 1.0},
    {'name': 'min_value', 'input': 0.0},
    {'name': 'threshold', 'input': 0.5},
]
tester.test_edge_cases(edge_cases)

# Test 4: Performance
test_data = [np.random.rand() for _ in range(100)]
tester.test_performance(test_data, timing_threshold_ms=1.0)

tester.report()


# ======================================================================
# ## Advanced Implementation: Comprehensive Testing Framework
# ======================================================================

from scipy.stats import ks_2samp
from collections import defaultdict

class ProductionModelTestSuite:
    """Production-grade testing with fairness, robustness, and monitoring"""
    
    def __init__(self, model_predict_fn, dataset_name: str = "default"):
        self.predict = model_predict_fn
        self.dataset_name = dataset_name
        self.results = {}
        self.train_distribution = None
    
    def test_correctness_stratified(self, test_data: List[Dict], stratify_by_key: str) -> Dict:
        """Correctness test stratified by subgroup (catches fairness issues)"""
        print(f"\nTesting correctness (stratified by {stratify_by_key})...")
        results = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for case in test_data:
            group = case.get(stratify_by_key, 'unknown')
            prediction = self.predict(case['features'])
            if prediction == case['label']:
                results[group]['correct'] += 1
            results[group]['total'] += 1
        
        # Compute per-group accuracy
        accuracies = {}
        for group, counts in results.items():
            acc = counts['correct'] / counts['total']
            accuracies[group] = acc
            print(f"  {group}: {counts['correct']}/{counts['total']} ({acc*100:.1f}%)")
        
        # Check for fairness (max accuracy gap < 5%)
        max_gap = max(accuracies.values()) - min(accuracies.values())
        is_fair = max_gap < 0.05
        print(f"  Fairness gap: {max_gap*100:.1f}% ({'PASS' if is_fair else 'FAIL'} - threshold: <5%)")
        
        self.results['correctness_stratified'] = {'accuracies': accuracies, 'fair': is_fair, 'gap': max_gap}
        return results
    
    def test_robustness(self, test_data: List[Dict], noise_level: float = 0.1) -> Dict:
        """Robustness test: how does model handle noisy/corrupted inputs?"""
        print(f"\nTesting robustness (noise level: {noise_level})...")
        
        clean_correct = 0
        noisy_correct = 0
        
        for case in test_data:
            # Test on clean input
            clean_pred = self.predict(case['features'])
            if clean_pred == case['label']:
                clean_correct += 1
            
            # Test on noisy input (add random noise)
            noisy_features = np.array(case['features']) + np.random.normal(0, noise_level, len(case['features']))
            noisy_pred = self.predict(noisy_features.tolist())
            if noisy_pred == case['label']:
                noisy_correct += 1
        
        clean_acc = clean_correct / len(test_data)
        noisy_acc = noisy_correct / len(test_data)
        robustness_drop = clean_acc - noisy_acc
        
        acceptable = robustness_drop < 0.1  # Tolerance: <10% accuracy drop
        print(f"  Clean accuracy: {clean_acc*100:.1f}%")
        print(f"  Noisy accuracy: {noisy_acc*100:.1f}%")
        print(f"  Robustness drop: {robustness_drop*100:.1f}% ({'PASS' if acceptable else 'FAIL'})")
        
        self.results['robustness'] = {'clean_acc': clean_acc, 'noisy_acc': noisy_acc, 'drop': robustness_drop}
        return self.results['robustness']
    
    def test_distribution_shift(self, train_dist: np.ndarray, production_dist: np.ndarray, feature_name: str) -> Dict:
        """Detect input distribution shift using KS test"""
        print(f"\nTesting distribution shift on {feature_name}...")
        
        statistic, p_value = ks_2samp(train_dist, production_dist)
        
        has_shift = p_value < 0.05  # Statistical significance
        print(f"  KS statistic: {statistic:.3f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Distribution shift: {'YES' if has_shift else 'NO'}")
        
        self.results['distribution_shift'] = {'statistic': statistic, 'p_value': p_value, 'has_shift': has_shift}
        return self.results['distribution_shift']
    
    def test_output_consistency(self, test_data: List, num_inferences: int = 10) -> bool:
        """Test: same input → same output across multiple runs?"""
        print(f"\nTesting output consistency across {num_inferences} inferences...")
        
        # Use first test sample
        sample = test_data[0]['features'] if isinstance(test_data[0], dict) else test_data[0]
        outputs = [self.predict(sample) for _ in range(num_inferences)]
        
        is_consistent = len(set(outputs)) == 1
        print(f"  Outputs: {outputs[:5]}..." if len(outputs) > 5 else f"  Outputs: {outputs}")
        print(f"  Consistent: {'PASS' if is_consistent else 'FAIL'}")
        
        self.results['output_consistency'] = {'consistent': is_consistent}
        return is_consistent
    
    def report(self):
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL TEST REPORT")
        print("="*60)
        for test_name, result in self.results.items():
            print(f"\n{test_name.upper()}:")
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

# Example: More complex model (simulated)
def fraud_detection_model(features):
    """Simulated fraud detection: features = [amount, age, is_international]"""
    amount, age, is_intl = features
    # Simple heuristic: high amount + international = likely fraud
    fraud_score = (amount / 1000) * 0.7 + (is_intl * 0.5) - (age / 50) * 0.2
    return 1 if fraud_score > 0.5 else 0

# Create test suite
tester = ProductionModelTestSuite(fraud_detection_model, "fraud_detection")

# Generate test data
np.random.seed(42)
test_data = []
for i in range(100):
    amount = np.random.uniform(10, 5000)
    age = np.random.randint(18, 80)
    is_intl = np.random.choice([0, 1])
    # Create labels (imperfect, matches heuristic 85% of time)
    expected = fraud_detection_model([amount, age, is_intl])
    noise = np.random.choice([0, 1], p=[0.85, 0.15])
    label = expected if noise == 0 else 1 - expected
    geography = 'US' if is_intl == 0 else 'International'
    test_data.append({
        'features': [amount, age, is_intl],
        'label': label,
        'geography': geography
    })

# Run tests
tester.test_correctness_stratified(test_data, stratify_by_key='geography')
tester.test_robustness(test_data, noise_level=0.2)

# Distribution shift test
train_amounts = np.random.uniform(10, 2000, 500)  # Training had smaller transactions
prod_amounts = np.random.uniform(100, 5000, 500)  # Production has larger
tester.test_distribution_shift(train_amounts, prod_amounts, "transaction_amount")

tester.test_output_consistency(test_data[:10])
tester.report()


# ======================================================================
# ## Real-World Example 1: Stripe Fraud Detection Model Testing
# ======================================================================

import pandas as pd
import numpy as np

def netflix_fairness_testing():
    """Test recommendation model fairness across regions"""

    print("NETFLIX: Fairness Testing")
    print("=" * 60)

    np.random.seed(42)

    regions = ['US', 'EU', 'Asia', 'LATAM']

    results = pd.DataFrame({
        'region': regions,
        'precision': [0.92, 0.90, 0.88, 0.85],
        'recall': [0.85, 0.82, 0.79, 0.76],
        'users': ['100M', '50M', '80M', '20M'],
    })

    print("\nACCURACY BY REGION:")
    print(results.to_string(index=False))

    print("\n\nFAIRNESS ANALYSIS:")
    precision_delta = results['precision'].max() - results['precision'].min()
    recall_delta = results['recall'].max() - results['recall'].min()

    print(f"  Precision variance: {precision_delta:.3f} ({precision_delta*100:.1f}% difference)")
    print(f"  Recall variance: {recall_delta:.3f} ({recall_delta*100:.1f}% difference)")
    print()

    print("ISSUES FOUND:")
    print("  ⚠ Precision 7% lower in LATAM (85% vs 92% in US)")
    print("  ⚠ Recall 9% lower in LATAM (76% vs 85% in US)")
    print("  Root cause: LATAM data 10x smaller, different content patterns")
    print()

    print("REMEDIATION:")
    print("  1. Collect more LATAM data (current: 50M events, target: 500M)")
    print("  2. Add region-specific features (local content popularity)")
    print("  3. Retrain on balanced dataset")
    print("  4. Target: <2% variance across regions (fairness threshold)")

netflix_fairness_testing()



# ======================================================================
# ## Real-World Example 2: Netflix Recommendation Fairness Testing
# ======================================================================

import pandas as pd
import numpy as np

def stripe_adversarial_testing():
    """Test fraud model robustness against evasion attacks"""

    print("STRIPE: Adversarial Testing - Fraud Model Robustness")
    print("=" * 60)

    # Baseline model
    clean_transactions = np.random.lognormal(4.5, 1.2, 1000)
    fraud_transactions = np.random.lognormal(5.0, 1.5, 1000)

    # Model baseline accuracy
    clean_pred = (clean_transactions < 100).sum()
    fraud_pred = (fraud_transactions > 100).sum()
    baseline_acc = (clean_pred + fraud_pred) / 2000

    print(f"\nBASELINE MODEL:")
    print(f"  Clean transaction detection: {clean_pred/1000*100:.1f}%")
    print(f"  Fraud transaction detection: {fraud_pred/1000*100:.1f}%")
    print(f"  Overall accuracy: {baseline_acc*100:.1f}%")

    print(f"\n\nADVERSARIAL ATTACK 1: Tiny Transactions (Evasion)")
    print(f"  Attack: Use tiny amounts to evade detection")
    adversarial_amounts = np.random.uniform(0.1, 5, 1000)
    evasion_detect = (adversarial_amounts < 100).sum()
    print(f"  Model detects: {evasion_detect/1000*100:.1f}% (rule-based fallback)")

    print(f"\n\nADVERSARIAL ATTACK 2: Velocity (Rapid-Fire)")
    print(f"  Attack: 50 transactions in 1 minute from single user")
    print(f"  Model detects: 95% (velocity rule catches most)")
    print(f"  Evasion rate: 5% (some get through)")

    print(f"\n\nADVERSARIAL ATTACK 3: Account Takeover")
    print(f"  Attack: Legitimate account, fraudulent use")
    print(f"  Model detects: 60% (harder to catch)")
    print(f"  Reason: Legitimate user history masks fraud signals")

    print(f"\n\nMITIGATION:")
    print(f"  Add velocity rules (50+ txns/min = flag)")
    print(f"  Add geographic anomalies (different country = verify)")
    print(f"  Improve behavioral models (user patterns)")
    print(f"  Target: >90% evasion resistance")

stripe_adversarial_testing()



# ======================================================================
# ## Real-World Example 3: Uber ETA Model Latency Testing
# ======================================================================

def uber_latency_testing():
    print("UBER: ETA Model Latency Testing in Production\n")
    
    print("1. Requirements:")
    print("   - ETA model serves 100K+ requests/second globally")
    print("   - Latency SLO: p99 <200ms (user waiting in app)")
    print("   - New model: deeper neural network, better accuracy")
    print("   - Risk: will latency still meet SLO?")
    print()
    
    print("2. Load test setup:")
    print("   - Simulate production traffic patterns:")
    print("     - 80% single predictions (user requests ETA)")
    print("     - 15% batch predictions (background jobs)")
    print("     - 5% spike traffic (rush hour)")
    print()
    
    print("3. Latency test results:")
    print()
    
    print("   Baseline model (existing):")
    print("     - p50: 80ms")
    print("     - p99: 150ms")
    print("     - Throughput: 10K req/sec on 1 GPU")
    print()
    
    print("   New model (deeper network):")
    print("     - p50: 120ms")
    print("     - p99: 200ms (at SLO limit!)")
    print("     - Throughput: 6K req/sec on 1 GPU")
    print()
    
    print("   Decision: Model meets SLO but no buffer")
    print("   Action: Optimize with model quantization")
    print()
    
    print("4. Optimization and retest:")
    print("   - Use INT8 quantization (no accuracy loss)")
    print("   - Retested results:")
    print("     - p50: 95ms")
    print("     - p99: 170ms")
    print("     - Throughput: 12K req/sec on 1 GPU (improvement!)")
    print()
    
    print("5. Memory test:")
    print("   - GPU memory available: 16GB")
    print("   - Model size: 500MB")
    print("   - Batch size: 128 (feasible)")
    print("   - Under load: 8GB used (safe)")
    print()
    
    print("6. Shadow test (3 days):")
    print("   - Model runs on 1% of traffic, results logged")
    print("   - Latency p99: 168ms (matches lab test)")
    print("   - No memory issues")
    print()
    
    print("7. Canary deployment:")
    print("   - Roll out to 5% of traffic")
    print("   - Monitor: latency, accuracy, error rate")
    print("   - After 24h: metrics stable, no alerts")
    print("   - Expand to 50% traffic")
    print("   - After 7 days: full deployment")
    print()
    
    print("✓ New model deployed successfully, accuracy +2%, latency within SLO")

uber_latency_testing()


# ======================================================================
# ## Interview Case Study: DoorDash Recommendation Testing
# ======================================================================

print("CASE STUDY: DOORDASH RECOMMENDATION TESTING")
print()
print("SCENARIO:")
print("  Build recommendation system: 50M users, 500K restaurants")
print("  Must recommend restaurants for users to order from")
print("  Previous model: only top-rated restaurants recommended (boring, unfair to new restaurants)")
print("  New model: balance quality + discovery")
print()

print("SOLUTION: Design comprehensive test plan")
print()
print("1. UNIT TESTS (Model correctness)")
print()
print("   Test 1: Does model rank user's favorite restaurant high?")
print("     - User history: 50 orders from Restaurant A")
print("     - Model should rank A in top 5")
print("     - Test on 10K users with clear favorites")
print("     - Result: 97% rank favorite in top 5 ✓")
print()
print("   Test 2: Cold-start user handling")
print("     - New user (0 orders)")
print("     - Can model recommend diverse restaurants?")
print("     - Test: 5 recommended restaurants should span 3+ cuisines")
print("     - Result: 92% have diversity ✓")
print()
print("   Test 3: Consistency")
print("     - Same user, same timestamp → same recommendations?")
print("     - Result: 100% consistent ✓")
print()

print("2. INTEGRATION TESTS (Model + Feature Store + Data)")
print()
print("   Test 4: Feature freshness")
print("     - User's recent orders should be in feature store")
print("     - Check: latest order from <1h ago")
print("     - Result: 99.8% of users have fresh features ✓")
print()
print("   Test 5: Restaurant availability")
print("     - Don't recommend closed/unavailable restaurants")
print("     - Check: all recommendations are open now")
print("     - Result: 99.9% recommendations are available ✓")
print()

print("3. FAIRNESS TESTS (Performance across subgroups)")
print()
print("   Test 6: Geography fairness")
print("     - Does recommendation quality vary by city?")
print("     - Metric: % top-rated restaurants recommended")
print("     - SF: 85%, NY: 83%, LA: 84% (gap <2%) ✓")
print()
print("   Test 7: Restaurant age fairness")
print("     - New restaurants <6 months old: % recommended")
print("     - Old restaurants: 70% recommended")
print("     - New restaurants: 45% recommended (gap 25%) ✗")
print("     - Action: Model needs adjustment to boost discovery")
print()

print("4. PERFORMANCE TESTS (Latency & Throughput)")
print()
print("   Test 8: Latency SLO")
print("     - User opens app → recommendations in <500ms")
print("     - p50: 250ms")
print("     - p99: 480ms (SLO: <500ms) ✓")
print()
print("   Test 9: Throughput")
print("     - Peak traffic: 500K requests/min")
print("     - Latency at peak: p99 520ms (exceeds SLO!) ✗")
print("     - Action: Add caching + load balancing")
print()

print("5. PRODUCTION TESTS (Shadow + Canary)")
print()
print("   Test 10: Shadow test (1 week)")
print("     - New model runs on 100% of traffic")
print("     - Predictions logged but not shown")
print("     - Compared to old model: click-through rate same, diversity +20%")
print("     - No edge cases found ✓")
print()
print("   Test 11: Canary test (3 days)")
print("     - 5% of users see new recommendations")
print("     - Monitor: CTR, conversion rate, user complaints")
print("     - Results: CTR +2%, no complaints ✓")
print()
print("   Test 12: A/B test (7 days)")
print("     - 25% new model, 75% baseline")
print("     - Metrics:")
print("       - Order volume: +3% (statistically significant)")
print("       - User satisfaction: 4.6/5 → 4.7/5")
print("       - Discovery: % ordering from new restaurants +8%")
print("     - Decision: DEPLOY ✓")
print()

print("STRONG ANSWER:")
print()
print("'Testing strategy has 4 layers:")
print()
print("1. Unit tests: model correctness on known inputs (favorites, cold-start)")
print()
print("2. Integration tests: with real data (feature freshness, availability)")
print()
print("3. Fairness tests: performance across geographies and restaurant age")
print()
print("4. Production tests: shadow (1 week on real traffic, no user impact)")
print("   → canary (5% users, monitor) → A/B (25% users, measure business impact)")
print()
print("Each layer catches different bugs. Deployment only after all pass.'")


# ======================================================================
# ## Key Takeaways
# **Model testing is multi-layered:** data → model → integration → E2E (shadow → canary → A/B → production).
# **Test categories matter:**
# ======================================================================
