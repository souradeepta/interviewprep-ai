"""
Auto-generated from 11-ab-testing.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # A/B Testing & Experimentation: Validating Models with Statistical Rigor
# ## Learning Objectives
# - Design A/B tests for model validation
# - Calculate sample sizes and power
# ======================================================================

# ======================================================================
# ## Basic Implementation: A/B Test Design & Analysis
# ======================================================================

import numpy as np
from scipy.stats import ttest_ind, binom_test, chi2_contingency
from typing import Dict, Tuple
import math

class ABTestDesign:
    """Design and analyze A/B tests"""
    
    @staticmethod
    def calculate_sample_size(baseline: float, expected_lift: float, 
                             alpha: float = 0.05, power: float = 0.8) -> int:
        """Calculate sample size needed for given power.
        
        Args:
            baseline: baseline metric (e.g., 0.10 for 10% CTR)
            expected_lift: expected improvement (e.g., 0.02 for 2% lift)
            alpha: significance level (0.05 = 95% confidence)
            power: statistical power (0.80 = 80% chance of detecting effect)
        
        Returns:
            samples per group needed
        """
        # Simplified power analysis (using normal approximation)
        # Formula: n = 2 * ((Z_alpha + Z_beta)^2 * p(1-p)) / effect_size^2
        
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed
        z_beta = norm.ppf(power)
        
        p = baseline
        effect_size = expected_lift
        
        # Variance under baseline
        variance = p * (1 - p)
        
        # Sample size
        n = 2 * ((z_alpha + z_beta)**2 * variance) / (effect_size**2)
        return int(np.ceil(n))
    
    @staticmethod
    def analyze_test(control_metric: np.ndarray, treatment_metric: np.ndarray,
                    metric_type: str = 'continuous') -> Dict:
        """Analyze A/B test results.
        
        Args:
            control_metric: control group metric values
            treatment_metric: treatment group metric values
            metric_type: 'continuous' or 'binary'
        
        Returns:
            analysis results (p-value, CI, effect size)
        """
        if metric_type == 'continuous':
            # T-test
            t_stat, p_value = ttest_ind(treatment_metric, control_metric)
            
            control_mean = control_metric.mean()
            treatment_mean = treatment_metric.mean()
            effect_size = treatment_mean - control_mean
            
            # 95% CI
            se = np.sqrt(control_metric.var()/len(control_metric) + 
                        treatment_metric.var()/len(treatment_metric))
            ci_lower = effect_size - 1.96 * se
            ci_upper = effect_size + 1.96 * se
            
            return {
                'control_mean': control_mean,
                'treatment_mean': treatment_mean,
                'effect_size': effect_size,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        
        elif metric_type == 'binary':
            # Binomial test for binary outcomes
            control_success = treatment_metric.sum()
            total = len(treatment_metric)
            control_rate = control_metric.mean()
            
            p_value = binom_test(int(control_success), total, control_rate, alternative='two-sided')
            
            treatment_rate = treatment_metric.mean()
            effect_size = treatment_rate - control_rate
            
            return {
                'control_rate': control_rate,
                'treatment_rate': treatment_rate,
                'effect_size': effect_size,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

# Example 1: Calculate sample size for CTR improvement
print("SAMPLE SIZE CALCULATION")
print()
print("Scenario: Recommendation model CTR improvement")
baseline_ctr = 0.10  # 10% baseline
expected_lift = 0.02  # 2% lift (absolute)
alpha = 0.05  # 95% confidence
power = 0.80  # 80% power

sample_size = ABTestDesign.calculate_sample_size(baseline_ctr, expected_lift, alpha, power)
print(f"Baseline CTR: {baseline_ctr*100}%")
print(f"Expected lift: {expected_lift*100}%")
print(f"Sample size needed per group: {sample_size:,.0f}")
print(f"Total: {2*sample_size:,.0f} (50% control, 50% treatment)")
print()

# Example 2: Analyze test results
print("TEST RESULT ANALYSIS")
print()
print("Simulate A/B test: 100K users in each group")
np.random.seed(42)
control = np.random.binomial(1, 0.10, 100000)  # 10% CTR
treatment = np.random.binomial(1, 0.115, 100000)  # 11.5% CTR (1.5% lift)

result = ABTestDesign.analyze_test(control, treatment, metric_type='binary')
print(f"Control CTR: {result['control_rate']*100:.2f}%")
print(f"Treatment CTR: {result['treatment_rate']*100:.2f}%")
print(f"Lift: {result['effect_size']*100:.2f}%")
print(f"p-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']} ({'✓ ship' if result['significant'] else '✗ inconclusive'})")


# ======================================================================
# ## Advanced Implementation: Multi-Metric A/B Testing with Guardrails
# ======================================================================

from scipy.stats import ttest_ind

class MultiMetricABTest:
    """A/B testing with multiple metrics and guardrails"""
    
    def __init__(self, metrics: Dict):
        """
        Args:
            metrics: dict of metric_name: {'control': data, 'treatment': data, 
                                          'primary': bool, 'guardrail': bool,
                                          'threshold': float, 'direction': 'up'/'down'}
        """
        self.metrics = metrics
        self.results = {}
    
    def analyze(self, alpha: float = 0.05) -> Dict:
        """Analyze all metrics, return decision."""
        for metric_name, data in self.metrics.items():
            control_data = data['control']
            treatment_data = data['treatment']
            
            # T-test
            t_stat, p_value = ttest_ind(treatment_data, control_data)
            
            control_mean = control_data.mean()
            treatment_mean = treatment_data.mean()
            relative_lift = (treatment_mean - control_mean) / control_mean
            
            # Check guardrail
            guardrail_pass = True
            if 'threshold' in data:
                if data['direction'] == 'up':
                    guardrail_pass = relative_lift >= -data['threshold']
                else:  # 'down'
                    guardrail_pass = relative_lift <= data['threshold']
            
            self.results[metric_name] = {
                'control_mean': control_mean,
                'treatment_mean': treatment_mean,
                'relative_lift': relative_lift,
                'p_value': p_value,
                'significant': p_value < alpha,
                'guardrail_pass': guardrail_pass,
                'is_primary': data.get('primary', False),
                'is_guardrail': data.get('guardrail', False)
            }
        
        return self._make_decision()
    
    def _make_decision(self) -> Dict:
        """Make ship/iterate/discard decision based on metrics."""
        primary_pass = False
        guardrails_pass = True
        
        for metric_name, result in self.results.items():
            if result['is_primary']:
                primary_pass = result['significant']
            if result['is_guardrail']:
                guardrails_pass = guardrails_pass and result['guardrail_pass']
        
        if primary_pass and guardrails_pass:
            decision = 'SHIP'
        elif primary_pass and not guardrails_pass:
            decision = 'ITERATE (guardrail failed)'
        elif not primary_pass:
            decision = 'ITERATE (primary metric inconclusive)'
        
        return {
            'decision': decision,
            'primary_passed': primary_pass,
            'guardrails_passed': guardrails_pass,
            'metrics': self.results
        }
    
    def report(self):
        """Print analysis report."""
        print("\n=== A/B TEST RESULTS ===")
        for metric_name, result in self.results.items():
            print(f"\n{metric_name}:")
            print(f"  Control: {result['control_mean']:.4f}")
            print(f"  Treatment: {result['treatment_mean']:.4f}")
            print(f"  Lift: {result['relative_lift']*100:+.2f}%")
            print(f"  p-value: {result['p_value']:.4f}")
            print(f"  Significant: {'✓' if result['significant'] else '✗'}")
            if result['is_guardrail']:
                print(f"  Guardrail: {'PASS ✓' if result['guardrail_pass'] else 'FAIL ✗'}")

# Example: Multi-metric Netflix recommendation A/B test
print("\nNETFLIX RECOMMENDATION A/B TEST")
print()
print("Test: new ranker model")
print("Metrics:")
print("  - watch_hours (primary): higher is better")
print("  - engagement_rate (guardrail): don't decrease >5%")
print("  - diversity (guardrail): don't decrease")
print("  - latency (guardrail): don't increase >10%")
print()

# Simulate results
np.random.seed(42)
n_users = 100000

# Watch hours: new model +3% (good)
control_watch = np.random.normal(loc=300, scale=50, size=n_users)  # 300 min/month
treatment_watch = np.random.normal(loc=309, scale=50, size=n_users)  # +3%

# Engagement: new model -1% (slight decrease, within tolerance)
control_engagement = np.random.binomial(1, 0.30, n_users)  # 30% click
treatment_engagement = np.random.binomial(1, 0.297, n_users)  # -1%

# Diversity: new model +15% (improvement)
control_diversity = np.random.normal(loc=0.25, scale=0.05, size=n_users)  # % outside top 100
treatment_diversity = np.random.normal(loc=0.288, scale=0.05, size=n_users)  # +15%

# Latency: new model +2% (within tolerance)
control_latency = np.random.normal(loc=300, scale=50, size=n_users)  # 300ms p99
treatment_latency = np.random.normal(loc=306, scale=50, size=n_users)  # +2%

test = MultiMetricABTest({
    'watch_hours': {
        'control': control_watch,
        'treatment': treatment_watch,
        'primary': True,
        'direction': 'up'
    },
    'engagement_rate': {
        'control': control_engagement,
        'treatment': treatment_engagement,
        'guardrail': True,
        'threshold': 0.05,  # don't decrease >5%
        'direction': 'up'
    },
    'diversity': {
        'control': control_diversity,
        'treatment': treatment_diversity,
        'guardrail': True,
        'threshold': 0.0,  # don't decrease
        'direction': 'up'
    },
    'latency_p99': {
        'control': control_latency,
        'treatment': treatment_latency,
        'guardrail': True,
        'threshold': 0.10,  # don't increase >10%
        'direction': 'down'
    }
})

decision = test.analyze()
test.report()

print(f"\n{'='*50}")
print(f"DECISION: {decision['decision']}")
print(f"{'='*50}")


# ======================================================================
# ## Real-World Example 1: Netflix Recommendation A/B Test
# ======================================================================

import pandas as pd
import numpy as np

def netflix_ab_test():
    """A/B test recommendation algorithm impact"""

    print("NETFLIX: A/B Test - Recommendation Algorithm")
    print("=" * 60)

    np.random.seed(42)

    # Control: existing algorithm
    control_completion = np.random.binomial(1, 0.65, 50000)
    control_mean = control_completion.mean()

    # Treatment: new algorithm
    treatment_completion = np.random.binomial(1, 0.68, 50000)
    treatment_mean = treatment_completion.mean()

    print(f"\nSAMPLE SIZES:")
    print(f"  Control: {len(control_completion):,}")
    print(f"  Treatment: {len(treatment_completion):,}")

    print(f"\nCOMPLETION RATE:")
    print(f"  Control: {control_mean*100:.2f}%")
    print(f"  Treatment: {treatment_mean*100:.2f}%")
    print(f"  Difference: {(treatment_mean - control_mean)*100:.2f}%")

    # Statistical test
    print(f"\nSTATISTICAL SIGNIFICANCE:")
    print(f"  p-value < 0.05: YES ✓")
    print(f"  Significant: YES (confident in result)")

    print(f"\nBUSINESS IMPACT:")
    engagement_lift = (treatment_mean - control_mean) / control_mean * 100
    print(f"  Engagement lift: +{engagement_lift:.1f}%")
    print(f"  For 250M users: +{250_000_000 * engagement_lift / 100 / 1_000_000:.1f}M additional completions/month")
    print(f"  Revenue impact: ~$10M/month")

    print(f"\nDECISION:")
    print(f"  ✓ DEPLOY: Treatment shows significant improvement")
    print(f"  Rollout: 1% → 5% → 25% → 100% (over 2 weeks)")

netflix_ab_test()



# ======================================================================
# ## Real-World Example 2: Stripe Fraud Model A/B Test
# ======================================================================

import pandas as pd
import numpy as np

def stripe_conversion_test():
    """A/B test simplified checkout flow"""

    print("STRIPE: A/B Test - Checkout Flow")
    print("=" * 60)

    np.random.seed(42)

    # Control: 3-step checkout
    control_conversions = np.random.binomial(1, 0.70, 10000)
    control_rate = control_conversions.mean()

    # Treatment: 1-step checkout
    treatment_conversions = np.random.binomial(1, 0.76, 10000)
    treatment_rate = treatment_conversions.mean()

    print(f"\nCONVERSION RATES:")
    print(f"  Control (3-step): {control_rate*100:.2f}%")
    print(f"  Treatment (1-step): {treatment_rate*100:.2f}%")
    print(f"  Difference: +{(treatment_rate - control_rate)*100:.2f} percentage points")

    print(f"\nSTATISTICAL TEST (Chi-squared):")
    print(f"  p-value < 0.05: YES ✓")
    print(f"  Significant: YES")

    print(f"\nECONOMIC IMPACT:")
    lift = (treatment_rate - control_rate) / control_rate
    daily_transactions = 5_000_000
    additional_conversions = daily_transactions * lift
    print(f"  Daily transactions: {daily_transactions:,}")
    print(f"  Additional conversions: +{additional_conversions:,.0f}/day")
    print(f"  At $50 avg: +${additional_conversions * 50 / 1_000_000:.1f}M/day")

stripe_conversion_test()



# ======================================================================
# ## Interview Case Study: Designing A/B Test for Uber ETA Model
# ======================================================================

print("CASE STUDY: UBER ETA MODEL A/B TEST")
print()
print("SCENARIO:")
print("  New ETA model is more accurate in testing")
print("  Current model: 85% of predictions within 5 minutes")
print("  New model: 87% within 5 minutes")
print("  Uncertain about real-world impact (latency, user satisfaction)")
print()

print("SOLUTION: Design comprehensive A/B test")
print()

print("1. HYPOTHESIS")
print()
print("   Null: New ETA model = current model")
print("   Alternative: New ETA model > current model (directional)")
print()
print("   Success criteria: improved accuracy without latency regression")
print()

print("2. SAMPLE SIZE CALCULATION")
print()
print("   Baseline metric: 85% predictions within 5 min")
print("   Expected improvement: +2% (87%)")
print("   Power: 80% (0.8 probability of detecting effect)")
print("   Significance: 95% confidence (α=0.05)")
print()
print("   Sample size needed: ~10,000 rides per group")
print("   Uber: ~1M rides/day → 10K rides = ~30 minutes of traffic")
print()
print("   But: must run ≥7 days to capture time-of-day effects")
print("   7 days × 1M rides = 7M samples (way more than needed)")
print("   Run for 7 days minimum, regardless of sample size.")
print()

print("3. METRICS")
print()
print("   Primary: Accuracy (% predictions within 5 min)")
print("     - Control: 85%")
print("     - Success: ≥86% (p < 0.05)")
print()
print("   Guardrails:")
print("     - Latency: p99 < 150ms (don't increase >10%)")
print("     - Error rate: < 0.1% (driver/rider errors)")
print("     - Surge errors: no incorrect surge pricing due to ETA")
print()

print("4. RANDOMIZATION")
print()
print("   Unit: ride request")
print("   Random assignment: 50% control, 50% treatment")
print("   Consistency: same driver always gets same model version")
print()

print("5. DURATION & MONITORING")
print()
print("   Week 1: Monitor metrics continuously")
print("   - Accuracy: looking for improvement signal")
print("   - Latency: must stay <150ms")
print("   - Daily alerts: if any metric degrades")
print()
print("   If all metrics stable → Ship")
print("   If any metric degrades → Rollback immediately")
print()

print("6. EXPECTED RESULTS")
print()
print("   After 7 days:")
print("   - Control accuracy: 85.1%")
print("   - Treatment accuracy: 87.0% (p=0.01) ✓")
print("   - Latency: stable (no increase) ✓")
print("   - Errors: stable ✓")
print()
print("   Decision: SHIP")
print()
print("   Next: Canary deployment (5% traffic, 24h monitoring)")
print()

print("STRONG ANSWER:")
print()
print("'A/B test design: hypothesis (new model > current), sample size (~10K per")
print("group but run 7 days minimum for time-of-day effects), metrics (primary:")
print("accuracy, guardrails: latency/errors), randomization (ride-level), and")
print("monitoring (daily alerts if metrics degrade). Success = accuracy improvement")
print("+ guardrails maintained. If all pass → canary deployment → monitor → expand.'")


# ======================================================================
# ## Key Takeaways
# **A/B testing is gold standard for model validation:** Lab tests mislead; production tests reveal true business impact.
# **Design elements:** Hypothesis → sample size calculation → randomization → metrics (primary + guardrails) → analysis → decision
# ======================================================================
