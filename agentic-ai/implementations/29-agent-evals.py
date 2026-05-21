"""
Auto-generated from 29-agent-evals.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Agent Evals
# Learning objectives:
# - Understand evaluation frameworks: metrics, test design, human review
# - Implement automated and human-based evaluation
# ======================================================================

import os
import json
import time
from anthropic import Anthropic
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

print("Setup complete. Ready for agent evaluation!")


# ======================================================================
# ## Level 1: Basic Evaluation
# Simple success/fail checks on test cases.
# ======================================================================

class BasicEvaluator:
    """Simple pass/fail evaluation without metrics."""
    def __init__(self):
        self.client = Anthropic()
    
    def evaluate(self, task: str, success_criterion: callable) -> bool:
        """Run agent and check if it succeeds."""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": task}]
        )
        
        result = response.content[0].text
        return success_criterion(result)
    
    def run_test_suite(self, test_cases):
        """Run all tests and count passes."""
        passed = 0
        total = len(test_cases)
        
        for test in test_cases:
            success = self.evaluate(test["task"], test["check"])
            status = "✓" if success else "✗"
            print(f"{status} {test['name']}")
            if success:
                passed += 1
        
        print(f"\nResult: {passed}/{total} passed ({100*passed/total:.0f}%)")
        return {"passed": passed, "total": total, "success_rate": passed/total}

# Test cases
test_cases = [
    {"name": "Math question", "task": "What is 2+2?", "check": lambda r: "4" in r},
    {"name": "Reasoning", "task": "Is a dog a mammal?", "check": lambda r: "yes" in r.lower()},
    {"name": "Creative", "task": "Write a haiku about AI.", "check": lambda r: len(r) > 20}
]

evaluator = BasicEvaluator()
result = evaluator.run_test_suite(test_cases)


# ======================================================================
# ## Level 2: Advanced Evaluation with Metrics
# Track latency, cost, runs per test, and stochastic distributions.
# ======================================================================

class AdvancedEvaluator:
    """Evaluate with detailed metrics tracking."""
    def __init__(self):
        self.client = Anthropic()
        self.metrics = defaultdict(list)
    
    def evaluate_with_metrics(self, test: dict, runs: int = 3) -> dict:
        """Run test multiple times and collect metrics."""
        results = {"successes": 0, "latencies": [], "costs": []}
        
        for run in range(runs):
            start = time.time()
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=256,
                messages=[{"role": "user", "content": test["task"]}]
            )
            
            elapsed = time.time() - start
            success = test["check"](response.content[0].text)
            
            # Track metrics
            results["successes"] += int(success)
            results["latencies"].append(elapsed)
            
            # Estimate cost
            input_cost = response.usage.input_tokens * 0.003 / 1000
            output_cost = response.usage.output_tokens * 0.006 / 1000
            results["costs"].append(input_cost + output_cost)
        
        # Aggregate
        results["success_rate"] = results["successes"] / runs
        results["avg_latency"] = sum(results["latencies"]) / runs
        results["total_cost"] = sum(results["costs"])
        
        return results
    
    def evaluate_suite(self, test_cases, runs=2):
        """Evaluate suite and aggregate metrics."""
        all_results = {}
        total_cost = 0
        total_tests = 0
        total_success = 0
        
        for test in test_cases:
            print(f"\nEvaluating: {test['name']} ({runs} runs)")
            metrics = self.evaluate_with_metrics(test, runs=runs)
            all_results[test["name"]] = metrics
            
            print(f"  Success rate: {metrics['success_rate']:.0%}")
            print(f"  Avg latency: {metrics['avg_latency']:.2f}s")
            print(f"  Total cost: ${metrics['total_cost']:.4f}")
            
            total_cost += metrics["total_cost"]
            total_tests += runs
            total_success += metrics["successes"]
        
        print(f"\n=== Summary ===")
        print(f"Overall success rate: {total_success/total_tests:.0%}")
        print(f"Total cost: ${total_cost:.3f}")
        print(f"Cost per test: ${total_cost/total_tests:.4f}")
        
        return all_results

# Test
evaluator = AdvancedEvaluator()
metrics = evaluator.evaluate_suite(test_cases, runs=2)


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: Automated Metrics with Tool Usage Tracking
# ======================================================================

class AutomatedMetricsEvaluator:
    """Track task completion, constraints, tool accuracy."""
    def __init__(self):
        self.client = Anthropic()
        self.tool_definitions = {
            "search_flights": {"params": ["from", "to", "date", "max_price"]},
            "book_flight": {"params": ["flight_id", "passenger_name"]}
        }
    
    def evaluate_test(self, test_case: dict) -> dict:
        """Evaluate with multiple metrics."""
        # Run agent
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": test_case["task"]}]
        )
        
        result_text = response.content[0].text
        
        # Check metrics
        metrics = {
            "task_completion": 1 if test_case["check"](result_text) else 0,
            "constraints_met": sum(
                1 for constraint in test_case.get("constraints", []) 
                if constraint.lower() in result_text.lower()
            ) / max(1, len(test_case.get("constraints", []))),
            "response_length": len(result_text),
            "cost": (response.usage.input_tokens * 0.003 + 
                    response.usage.output_tokens * 0.006) / 1000
        }
        
        return metrics
    
    def evaluate_suite(self, test_cases):
        """Evaluate with aggregated metrics."""
        results = []
        
        for test in test_cases:
            print(f"Evaluating: {test['name']}")
            metrics = self.evaluate_test(test)
            results.append({"test": test["name"], "metrics": metrics})
            print(f"  Task completion: {metrics['task_completion']}")
            print(f"  Constraints met: {metrics['constraints_met']:.0%}")
            print(f"  Cost: ${metrics['cost']:.4f}")
        
        # Aggregate
        avg_completion = sum(r["metrics"]["task_completion"] for r in results) / len(results)
        avg_constraints = sum(r["metrics"]["constraints_met"] for r in results) / len(results)
        total_cost = sum(r["metrics"]["cost"] for r in results)
        
        print(f"\n=== Aggregated ===")
        print(f"Avg task completion: {avg_completion:.0%}")
        print(f"Avg constraints met: {avg_constraints:.0%}")
        print(f"Total cost: ${total_cost:.3f}")
        
        return results

# Test with constraints
test_cases = [
    {
        "name": "Flight booking",
        "task": "Book a flight from NYC to LA on Jan 15 for under $500",
        "constraints": ["NYC", "LA", "Jan 15", "500"],
        "check": lambda r: "booked" in r.lower() or "booking" in r.lower()
    }
]

evaluator = AutomatedMetricsEvaluator()
results = evaluator.evaluate_suite(test_cases)


# ======================================================================
# ### Example 2: Human Rubric-Based Evaluation
# ======================================================================

class RubricEvaluator:
    """Score responses using weighted evaluation rubric."""
    def __init__(self):
        self.client = Anthropic()
        self.rubric = {
            "correctness": {"weight": 0.5, "description": "Did agent reach correct goal?"},
            "efficiency": {"weight": 0.25, "description": "Was path efficient?"},
            "reasoning": {"weight": 0.15, "description": "Was reasoning transparent?"},
            "safety": {"weight": 0.1, "description": "Were there safety issues?"}
        }
    
    def score_response(self, response_text: str, rubric_items: dict) -> dict:
        """Score response against rubric items."""
        scores = {}
        
        # Automated checks
        for item, check in rubric_items.items():
            if item in self.rubric:
                scores[item] = 1 if check(response_text) else 0
        
        return scores
    
    def compute_weighted_score(self, scores: dict) -> float:
        """Compute final weighted score (1-5)."""
        weighted_sum = sum(
            scores.get(item, 0) * self.rubric[item]["weight"] 
            for item in self.rubric
        )
        return weighted_sum * 5  # Scale to 1-5
    
    def evaluate_test(self, test_case: dict) -> dict:
        """Run test and apply rubric."""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": test_case["task"]}]
        )
        
        result_text = response.content[0].text
        
        # Apply rubric checks
        rubric_checks = {
            "correctness": test_case.get("check", lambda x: True),
            "efficiency": lambda r: len(r) < 200,
            "reasoning": lambda r: "because" in r.lower() or "therefore" in r.lower(),
            "safety": lambda r: "harmful" not in r.lower()
        }
        
        scores = self.score_response(result_text, rubric_checks)
        final_score = self.compute_weighted_score(scores)
        
        return {"scores": scores, "final_score": final_score, "response": result_text[:100]}
    
    def evaluate_suite(self, test_cases):
        """Evaluate suite with rubric scoring."""
        results = []
        
        for test in test_cases:
            print(f"\nRubric evaluation: {test['name']}")
            eval_result = self.evaluate_test(test)
            results.append(eval_result)
            
            print(f"  Scores: {eval_result['scores']}")
            print(f"  Final score: {eval_result['final_score']:.1f}/5")
        
        avg_final = sum(r["final_score"] for r in results) / len(results)
        print(f"\nAverage rubric score: {avg_final:.1f}/5")
        
        return results

# Test
evaluator = RubricEvaluator()
test = {"name": "Question", "task": "Why is the sky blue?", "check": lambda r: "light" in r.lower()}
results = evaluator.evaluate_suite([test])


# ======================================================================
# ### Example 3: Production Framework with Regression Detection
# ======================================================================

class ProductionEvalFramework:
    """Production eval with monitoring, cost tracking, regression detection."""
    def __init__(self, baseline_success_rate=0.85, regression_threshold=0.05):
        self.client = Anthropic()
        self.baseline_success_rate = baseline_success_rate
        self.regression_threshold = regression_threshold
        self.eval_history = []
        self.cost_tracker = 0
    
    def run_test(self, test: dict) -> bool:
        """Run single test."""
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=256,
                messages=[{"role": "user", "content": test["task"]}]
            )
            
            # Track cost
            cost = (response.usage.input_tokens * 0.003 + 
                   response.usage.output_tokens * 0.006) / 1000
            self.cost_tracker += cost
            
            return test["check"](response.content[0].text)
        
        except Exception as e:
            print(f"  Error running test: {str(e)[:30]}")
            return False
    
    def evaluate_suite(self, test_cases: list) -> dict:
        """Run full suite with regression detection."""
        print(f"Running {len(test_cases)} tests...")
        passed = 0
        
        for test in test_cases:
            success = self.run_test(test)
            passed += int(success)
            status = "✓" if success else "✗"
            print(f"  {status} {test['name']}")
        
        success_rate = passed / len(test_cases)
        
        # Check for regression
        regression = success_rate < self.baseline_success_rate - self.regression_threshold
        
        result = {
            "passed": passed,
            "total": len(test_cases),
            "success_rate": success_rate,
            "regression_detected": regression,
            "total_cost": self.cost_tracker,
            "cost_per_test": self.cost_tracker / len(test_cases)
        }
        
        self.eval_history.append(result)
        
        # Report
        print(f"\n=== Results ===")
        print(f"Success rate: {success_rate:.0%}")
        print(f"Baseline: {self.baseline_success_rate:.0%}")
        print(f"Regression: {'YES ⚠️' if regression else 'No'}")
        print(f"Cost per test: ${result['cost_per_test']:.4f}")
        
        return result

# Test
framework = ProductionEvalFramework(baseline_success_rate=0.70)
test_suite = [
    {"name": "Test 1", "task": "Hello", "check": lambda r: len(r) > 0},
    {"name": "Test 2", "task": "2+2=?", "check": lambda r: "4" in r},
]
result = framework.evaluate_suite(test_suite)


# ======================================================================
# ## Key Takeaways
# 1. **Evaluate multiple times.** Agents are stochastic. Run each test 3-5 times, report mean ± std dev, not single runs.
# 2. **Use automated metrics for scale.** Task completion, tool accuracy, latency, cost run on 1000s of examples instantly. Cheap baseline.
# ======================================================================
