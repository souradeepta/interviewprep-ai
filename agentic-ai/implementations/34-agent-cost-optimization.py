"""
Auto-generated from 34-agent-cost-optimization.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Agent Cost Optimization
# Learning objectives:
# - Implement caching to reduce API calls
# - Route tasks to appropriate models by cost/quality
# ======================================================================

import os
import json
import hashlib
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

print("Setup complete. Ready for cost optimization!")


# ======================================================================
# ## Level 1: Basic Cost Tracking
# Log and measure cost per API call.
# ======================================================================

client = Anthropic()

class BasicCostTracker:
    def __init__(self):
        self.costs = []
    
    def call_api(self, query: str) -> dict:
        """Call API and track cost."""
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": query}]
        )
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        
        # Cost calculation (Claude 3.5 Sonnet pricing)
        cost = (input_tokens * 0.003 + output_tokens * 0.006) / 1000
        
        self.costs.append({"query": query[:30], "tokens": total_tokens, "cost": cost})
        
        return {"response": response.content[0].text, "cost": cost, "tokens": total_tokens}
    
    def report(self) -> dict:
        """Cost summary."""
        if not self.costs:
            return {}
        
        total_cost = sum(c["cost"] for c in self.costs)
        total_tokens = sum(c["tokens"] for c in self.costs)
        avg_cost = total_cost / len(self.costs)
        
        return {
            "total_calls": len(self.costs),
            "total_cost": f"${total_cost:.4f}",
            "total_tokens": total_tokens,
            "avg_cost_per_call": f"${avg_cost:.4f}"
        }

# Test
tracker = BasicCostTracker()
for i in range(2):
    result = tracker.call_api(f"Question {i+1}: 2+2?")
    print(f"Query {i+1}: Cost ${result['cost']:.4f}, Tokens {result['tokens']}")

print(f"\nReport: {json.dumps(tracker.report(), indent=2)}")


# ======================================================================
# ## Level 2: Caching and Smart Routing
# Reduce costs with caching and model selection.
# ======================================================================

class SmartCostOptimizer:
    def __init__(self):
        self.cache = {}
        self.metrics = {"cache_hits": 0, "cache_misses": 0, "total_cost": 0}
    
    def hash_query(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()
    
    def classify_complexity(self, query: str) -> str:
        """Classify if query is simple or complex."""
        complex_words = ["analyze", "compare", "evaluate", "research", "detailed"]
        return "complex" if any(w in query.lower() for w in complex_words) else "simple"
    
    def call_api(self, query: str) -> dict:
        query_hash = self.hash_query(query)
        
        # Check cache
        if query_hash in self.cache:
            self.metrics["cache_hits"] += 1
            print(f"[CACHE HIT] {query[:30]}...")
            return {"response": self.cache[query_hash], "cached": True, "cost": 0}
        
        # Cache miss
        self.metrics["cache_misses"] += 1
        complexity = self.classify_complexity(query)
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": query}]
        )
        
        result = response.content[0].text
        cost = (response.usage.input_tokens * 0.003 + response.usage.output_tokens * 0.006) / 1000
        
        self.cache[query_hash] = result
        self.metrics["total_cost"] += cost
        
        print(f"[API CALL] {complexity} query, cost ${cost:.4f}")
        return {"response": result, "cached": False, "cost": cost}
    
    def get_savings(self) -> dict:
        total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        hit_rate = self.metrics["cache_hits"] / total if total > 0 else 0
        
        return {
            "cache_hit_rate": f"{hit_rate:.0%}",
            "total_cost": f"${self.metrics['total_cost']:.4f}",
            "queries": total
        }

# Test
optimizer = SmartCostOptimizer()
queries = ["What is 2+2?", "What is 2+2?", "Analyze quantum computing."]

for q in queries:
    optimizer.call_api(q)

print(f"\nSavings: {json.dumps(optimizer.get_savings(), indent=2)}")


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: Batch Processing
# ======================================================================

class BatchProcessor:
    """Process multiple items in single batch."""
    
    def process_batch(self, items: list) -> dict:
        """Process all items in one call."""
        batch_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)])
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"Summarize each:\n{batch_text}"
            }]
        )
        
        cost = (response.usage.input_tokens * 0.003 + response.usage.output_tokens * 0.006) / 1000
        
        return {
            "method": "batch",
            "items_processed": len(items),
            "cost": cost,
            "cost_per_item": cost / len(items)
        }

# Test
processor = BatchProcessor()
items = ["AI trends", "ML models", "Data science"]
result = processor.process_batch(items)
print(f"Batch result: {json.dumps({k: f'{v:.4f}' if isinstance(v, float) else v for k, v in result.items()}, indent=2)}")


# ======================================================================
# ### Example 2: Prompt Optimization
# ======================================================================

class PromptOptimizer:
    """Optimize prompts to reduce token count."""
    
    def measure_cost(self, prompt: str) -> dict:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}]
        )
        
        cost = (response.usage.input_tokens * 0.003 + response.usage.output_tokens * 0.006) / 1000
        return {"tokens": response.usage.input_tokens, "cost": cost}
    
    def compare_prompts(self):
        """Compare verbose vs concise prompts."""
        verbose = "Please provide a detailed and comprehensive analysis of quantum computing, including its definition, principles, applications, and implications for the future."
        concise = "Summarize quantum computing."
        
        print("Measuring prompt efficiency...\n")
        print(f"Verbose ({len(verbose.split())} words):")
        verbose_cost = self.measure_cost(verbose)
        print(f"  Input tokens: {verbose_cost['tokens']}, Cost: ${verbose_cost['cost']:.4f}")
        
        print(f"\nConcise ({len(concise.split())} words):")
        concise_cost = self.measure_cost(concise)
        print(f"  Input tokens: {concise_cost['tokens']}, Cost: ${concise_cost['cost']:.4f}")
        
        savings = (verbose_cost['cost'] - concise_cost['cost']) / verbose_cost['cost'] * 100
        print(f"\nSavings: {savings:.0f}%")

# Test
optimizer = PromptOptimizer()
optimizer.compare_prompts()


# ======================================================================
# ### Example 3: Budget Monitoring
# ======================================================================

class BudgetMonitor:
    """Monitor costs against budget."""
    
    def __init__(self, daily_budget: float):
        self.daily_budget = daily_budget
        self.spent = 0
        self.alerts = []
    
    def log_cost(self, cost: float) -> dict:
        """Log cost and check budget."""
        self.spent += cost
        
        # Alert at 80% and 100% of budget
        if self.spent > self.daily_budget * 0.8 and self.spent - cost <= self.daily_budget * 0.8:
            self.alerts.append("⚠️  WARNING: 80% of daily budget used")
        
        if self.spent > self.daily_budget:
            self.alerts.append(f"🚨 ALERT: Budget exceeded ${self.spent:.2f} > ${self.daily_budget:.2f}")
        
        remaining = self.daily_budget - self.spent
        
        return {
            "spent": f"${self.spent:.2f}",
            "budget": f"${self.daily_budget:.2f}",
            "remaining": f"${max(0, remaining):.2f}",
            "percent_used": f"{min(100, self.spent/self.daily_budget*100):.0f}%",
            "alerts": self.alerts
        }

# Test
monitor = BudgetMonitor(daily_budget=1.00)  # $1 per day
for cost in [0.30, 0.35, 0.40]:  # 3 calls
    status = monitor.log_cost(cost)
    print(f"After ${cost:.2f} cost:")
    print(f"  Remaining: {status['remaining']}, Used: {status['percent_used']}")
    if status['alerts']:
        for alert in status['alerts']:
            print(f"  {alert}")


# ======================================================================
# ## Key Takeaways
# 1. **Measure first.** Log every API call with cost. Can't optimize what you don't measure.
# 2. **Cache aggressively.** Identical queries produce identical results. 50% cache hit rate = 50% cost reduction.
# ======================================================================
