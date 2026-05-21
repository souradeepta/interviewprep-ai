"""
Auto-generated from 19-ai-gateway-routing.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # AI Gateway Routing: Cost and Latency Optimization
# ## Learning Objectives
# 1. Implement a routing engine with cost/latency scoring and fallback logic
# 2. Build adaptive routing strategies based on request characteristics
# 3. Simulate model availability and failure handling in production
# 4. Optimize routing decisions with cost-benefit analysis
# ======================================================================

import numpy as np
import time
from typing import Dict, List, Tuple
from collections import defaultdict
from enum import Enum

np.random.seed(42)
print('AI Gateway Routing System')


# ======================================================================
# ## Level 1: Basic Routing with Cost/Latency Scoring
# ======================================================================

# Level 1: Basic routing engine from scratch (40-50 lines)

class RouterModel:
    """Model definition for routing decisions"""
    def __init__(self, name: str, latency_ms: float, cost_per_request: float, accuracy: float):
        self.name = name
        self.latency_ms = latency_ms
        self.cost_per_request = cost_per_request
        self.accuracy = accuracy
        self.available = True
    
    def compute_score(self, latency_weight=0.3, cost_weight=0.5, accuracy_weight=0.2) -> float:
        """Compute routing score (lower is better for latency/cost)"""
        # Normalize metrics to 0-1 scale
        latency_norm = min(1.0, self.latency_ms / 1000)  # Normalize by 1s
        cost_norm = min(1.0, self.cost_per_request / 0.1)  # Normalize by $0.1
        accuracy_norm = self.accuracy  # Already 0-1
        
        # Weighted score (lower latency/cost is better, higher accuracy is better)
        score = (latency_weight * latency_norm + 
                 cost_weight * cost_norm - 
                 accuracy_weight * accuracy_norm)
        return score

class BasicRouter:
    """Basic routing engine with cost/latency scoring"""
    
    def __init__(self):
        self.models = {}
        self.routing_stats = defaultdict(int)
    
    def register_model(self, model: RouterModel):
        """Register a model for routing"""
        self.models[model.name] = model
    
    def route_request(self, request_type: str, latency_budget_ms: int = 500) -> Tuple[str, Dict]:
        """Route request to best available model"""
        available = [m for m in self.models.values() if m.available and m.latency_ms <= latency_budget_ms]
        
        if not available:
            return None, {'status': 'no_available_model'}
        
        # Select model with best score
        best_model = min(available, key=lambda m: m.compute_score())
        self.routing_stats[best_model.name] += 1
        
        return best_model.name, {
            'model': best_model.name,
            'latency_ms': best_model.latency_ms,
            'cost': best_model.cost_per_request,
            'accuracy': best_model.accuracy
        }

# Test basic router
print("=== Basic Routing Engine ===")
router = BasicRouter()

model_a = RouterModel('gpt4-turbo', latency_ms=800, cost_per_request=0.02, accuracy=0.95)
model_b = RouterModel('gpt35-fast', latency_ms=150, cost_per_request=0.005, accuracy=0.85)
model_c = RouterModel('llama2-cheap', latency_ms=300, cost_per_request=0.002, accuracy=0.80)

for model in [model_a, model_b, model_c]:
    router.register_model(model)
    print(f"Registered {model.name}: {model.latency_ms}ms, ${model.cost_per_request:.4f}, {model.accuracy:.2%} accuracy")

print("\\n=== Routing Decisions ===")
for i in range(5):
    model_name, route_info = router.route_request('query', latency_budget_ms=500)
    print(f"Request {i+1}: {model_name} (latency={route_info.get('latency_ms')}ms, cost=${route_info.get('cost'):.4f})")

print(f"\\nRouting distribution: {dict(router.routing_stats)}")


# ======================================================================
# ### Output: Basic routing selects models based on cost/latency trade-off
# ======================================================================

# ======================================================================
# ## Level 2: Advanced Routing with Adaptive Strategies and Failure Handling
# ======================================================================

# Level 2: Advanced routing with failover, buffering, cost optimization (80-100 lines)

class AdvancedRouter:
    """Advanced routing with failover, buffering, and cost optimization"""
    
    def __init__(self, cost_per_million_tokens: float = 10.0):
        self.models = {}
        self.request_queue = []
        self.cost_per_million = cost_per_million_tokens
        self.metrics = {'routed': 0, 'failed': 0, 'retried': 0}
        self.model_health = defaultdict(lambda: {'healthy': True, 'error_count': 0})
    
    def register_model(self, name: str, config: Dict):
        """Register model with configuration"""
        self.models[name] = config
    
    def check_model_health(self, model_name: str) -> bool:
        """Simulate health check"""
        health = self.model_health[model_name]
        return health['healthy'] and health['error_count'] < 5
    
    def compute_dynamic_score(self, model_name: str, request_complexity: float) -> float:
        """Compute score adapting to request complexity"""
        config = self.models[model_name]
        
        # For complex requests, prefer accurate models
        # For simple requests, prefer fast/cheap models
        if request_complexity > 0.7:
            return config['accuracy'] - 0.2 * config['latency_ms'] / 1000
        else:
            return -config['cost'] - 0.1 * config['latency_ms'] / 1000 + 0.3 * config['accuracy']
    
    def route_with_fallback(self, request_type: str, request_complexity: float = 0.5) -> Dict:
        """Route with fallback chain"""
        candidates = [(name, self.compute_dynamic_score(name, request_complexity)) 
                     for name in self.models.keys() if self.check_model_health(name)]
        
        if not candidates:
            self.metrics['failed'] += 1
            return {'status': 'all_models_down', 'model': None}
        
        # Sort by score and try primary, then fallback
        candidates.sort(key=lambda x: x[1], reverse=True)
        primary_model = candidates[0][0]
        fallback_model = candidates[1][0] if len(candidates) > 1 else None
        
        # Simulate primary failure (10% chance)
        if np.random.random() < 0.1:
            if fallback_model:
                self.metrics['retried'] += 1
                return {'model': fallback_model, 'status': 'fallback', 'primary_failed': primary_model}
            else:
                self.metrics['failed'] += 1
                return {'status': 'primary_failed_no_fallback'}
        
        self.metrics['routed'] += 1
        return {'model': primary_model, 'status': 'success', 'fallback': fallback_model}

# Test advanced router
print("\\n=== Advanced Router with Failover ===")
adv_router = AdvancedRouter()

adv_router.register_model('gpt4', {'latency_ms': 800, 'cost': 0.02, 'accuracy': 0.95})
adv_router.register_model('gpt35', {'latency_ms': 150, 'cost': 0.005, 'accuracy': 0.85})
adv_router.register_model('llama', {'latency_ms': 300, 'cost': 0.002, 'accuracy': 0.80})

print("\\nRouting 100 requests with different complexities...")
for i in range(100):
    complexity = np.random.random()  # Random complexity 0-1
    result = adv_router.route_with_fallback('query', complexity)

print(f"\\nRouting Metrics:")
print(f"  Successfully routed: {adv_router.metrics['routed']}")
print(f"  Failed: {adv_router.metrics['failed']}")
print(f"  Retried (fallback): {adv_router.metrics['retried']}")


# ======================================================================
# ### Output: Advanced routing handles failures with fallback chains
# ======================================================================

# ======================================================================
# ## Real-World Example 1: Routing Based on Request Type
# ======================================================================

# Real-World Example 1: Route different request types to appropriate models (50-60 lines)

class RequestTypeRouter:
    """Route requests to models based on request type"""
    
    def __init__(self):
        self.models = {
            'fast': {'latency': 100, 'cost': 0.002, 'accuracy': 0.75},
            'balanced': {'latency': 300, 'cost': 0.01, 'accuracy': 0.88},
            'accurate': {'latency': 800, 'cost': 0.05, 'accuracy': 0.98}
        }
        self.request_type_to_model = {
            'search': 'fast',  # Fast autocomplete
            'qa': 'balanced',  # Balanced for Q&A
            'analysis': 'accurate',  # Accurate for data analysis
            'summary': 'balanced'  # Balanced for summaries
        }
    
    def route(self, request_type: str) -> Dict:
        """Route based on request type"""
        model_class = self.request_type_to_model.get(request_type, 'balanced')
        model_config = self.models[model_class]
        
        return {
            'request_type': request_type,
            'routed_to': model_class,
            'latency_ms': model_config['latency'],
            'cost': model_config['cost'],
            'accuracy': model_config['accuracy']
        }

print("\\n=== Request Type-Based Routing ===")
type_router = RequestTypeRouter()

request_types = ['search', 'qa', 'analysis', 'summary']
for req_type in request_types:
    result = type_router.route(req_type)
    print(f"{req_type.upper():10} → {result['routed_to']:10} (latency={result['latency_ms']:4}ms, cost=${result['cost']:.4f})")


# ======================================================================
# ## Real-World Example 2: Cost-Optimized Routing with Dynamic Pricing
# ======================================================================

# Real-World Example 2: Cost optimization with dynamic pricing simulation (50-60 lines)

class CostOptimizedRouter:
    """Route with cost optimization based on load"""
    
    def __init__(self, monthly_budget: float = 1000.0):
        self.monthly_budget = monthly_budget
        self.spent_today = 0
        self.requests_today = 0
        self.models = {
            'expensive': {'cost_per_req': 0.05, 'accuracy': 0.98, 'latency': 800},
            'balanced': {'cost_per_req': 0.01, 'accuracy': 0.88, 'latency': 300},
            'cheap': {'cost_per_req': 0.002, 'accuracy': 0.75, 'latency': 100}
        }
    
    def compute_budget_remaining_pct(self) -> float:
        """Compute remaining budget percentage"""
        daily_budget = self.monthly_budget / 30
        return (daily_budget - self.spent_today) / daily_budget
    
    def select_model(self, request_priority: str = 'normal') -> Tuple[str, Dict]:
        """Select model based on budget constraints and request priority"""
        budget_pct = self.compute_budget_remaining_pct()
        
        if request_priority == 'high':
            # Always use accurate model for high-priority
            model_choice = 'expensive'
        elif budget_pct > 0.5:
            # Plenty of budget: use balanced
            model_choice = 'balanced'
        elif budget_pct > 0.2:
            # Running low: use cheap model
            model_choice = 'cheap'
        else:
            # Critical: use cheapest
            model_choice = 'cheap'
        
        config = self.models[model_choice]
        self.spent_today += config['cost_per_req']
        self.requests_today += 1
        
        return model_choice, config

print("\\n=== Cost-Optimized Routing ===")
cost_router = CostOptimizedRouter(monthly_budget=1000)

print(f"Daily budget: ${1000/30:.2f}")
print("\\nRouting decisions based on budget remaining:")

for i in range(5):
    model, config = cost_router.select_model('normal')
    budget_pct = cost_router.compute_budget_remaining_pct()
    print(f"Request {i+1}: Budget={budget_pct:>5.1%}, Selected={model:10}, Cost=${config['cost_per_req']:.4f}, Accuracy={config['accuracy']:.2%}")


# ======================================================================
# ## Real-World Example 3: Failover Scenarios
# ======================================================================

# Real-World Example 3: Test failover when primary model is down (50-60 lines)

class FailoverRouter:
    """Route with explicit failover testing"""
    
    def __init__(self):
        self.models = {
            'primary': {'status': 'up', 'accuracy': 0.95},
            'secondary': {'status': 'up', 'accuracy': 0.88},
            'tertiary': {'status': 'up', 'accuracy': 0.80}
        }
        self.failover_chain = ['primary', 'secondary', 'tertiary']
        self.stats = defaultdict(int)
    
    def mark_model_down(self, model_name: str):
        """Simulate model going down"""
        self.models[model_name]['status'] = 'down'
    
    def mark_model_up(self, model_name: str):
        """Simulate model recovery"""
        self.models[model_name]['status'] = 'up'
    
    def route_with_explicit_failover(self) -> Dict:
        """Try failover chain in order"""
        for model_name in self.failover_chain:
            if self.models[model_name]['status'] == 'up':
                self.stats[model_name] += 1
                return {'model': model_name, 'accuracy': self.models[model_name]['accuracy']}
        
        self.stats['all_down'] += 1
        return {'model': None, 'status': 'all_models_unavailable'}

print("\\n=== Failover Scenario Testing ===")
failover = FailoverRouter()

print("Scenario 1: All models up")
for _ in range(3):
    result = failover.route_with_explicit_failover()
print(f"Routed to primary 3 times")

print("\\nScenario 2: Primary down, failover to secondary")
failover.mark_model_down('primary')
for _ in range(3):
    result = failover.route_with_explicit_failover()
print(f"Routed to secondary 3 times")

print("\\nScenario 3: Primary and secondary down, failover to tertiary")
failover.mark_model_down('secondary')
for _ in range(2):
    result = failover.route_with_explicit_failover()
print(f"Routed to tertiary 2 times")

print(f"\\nFailover Chain Statistics:")
for model, count in sorted(failover.stats.items()):
    print(f"  {model}: {count}")


# ======================================================================
# ## Comparison: Cost/Latency Trade-off Matrix
# ======================================================================

import matplotlib.pyplot as plt

# Model characteristics
models_data = {
    'GPT-4 Turbo': {'latency': 800, 'cost': 0.05, 'accuracy': 0.98},
    'GPT-3.5': {'latency': 150, 'cost': 0.008, 'accuracy': 0.85},
    'Llama-2-70b': {'latency': 400, 'cost': 0.01, 'accuracy': 0.88},
    'Mistral-7b': {'latency': 250, 'cost': 0.003, 'accuracy': 0.82},
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Cost vs Latency
for model_name, metrics in models_data.items():
    axes[0].scatter(metrics['latency'], metrics['cost'], s=300, alpha=0.6, label=model_name)
    axes[0].annotate(model_name.split()[0], (metrics['latency'], metrics['cost']), fontsize=9, ha='center')

axes[0].set_xlabel('Latency (ms)')
axes[0].set_ylabel('Cost per Request ($)')
axes[0].set_title('Cost vs Latency Trade-off')
axes[0].grid(True, alpha=0.3)

# Plot 2: Accuracy vs Cost
for model_name, metrics in models_data.items():
    axes[1].scatter(metrics['cost'], metrics['accuracy'], s=300, alpha=0.6)
    axes[1].annotate(model_name.split()[0], (metrics['cost'], metrics['accuracy']), fontsize=9, ha='center')

axes[1].set_xlabel('Cost per Request ($)')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy vs Cost')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/gateway_routing_tradeoffs.png', dpi=100, bbox_inches='tight')
plt.show()

print("\\n=== Routing Decision Matrix ===")
print(f"{'Model':<20} {'Latency':<12} {'Cost':<12} {'Accuracy':<12} {'Use Case'}")
print("-" * 70)
for model_name, metrics in sorted(models_data.items(), key=lambda x: x[1]['cost']):
    print(f"{model_name:<20} {metrics['latency']:<12}ms ${metrics['cost']:<11.4f} {metrics['accuracy']:<12.1%} ", end="")
    if metrics['cost'] < 0.01:
        print("Low-latency queries")
    elif metrics['accuracy'] > 0.9:
        print("High-accuracy tasks")
    else:
        print("Balanced workflows")


# ======================================================================
# ## Key Takeaways
# ======================================================================

# ======================================================================
# ### Core Concept
# AI Gateway routing intelligently directs requests to models based on cost, latency, accuracy, and availability. Essential for systems with multiple model endpoints to optimize quality, cost, and performance.
# ### Routing Strategies
# | Strategy | Latency | Cost | Accuracy | Use Case |
# |----------|---------|------|----------|----------|
# | Fast-Only | <200ms | $0.005 | 80-85% | Real-time search, autocomplete |
# | Balanced | 200-400ms | $0.01 | 85-90% | Q&A, general queries |
# | Accurate | 500-1000ms | $0.05 | 95%+ | Analysis, complex reasoning |
# | Adaptive | Varies | Varies | Varies | Dynamic based on load/budget |
# ### Production Patterns
# 1. **Cost-based routing:** Route cheaper models during high load or budget constraints
# 2. **Request-type routing:** Different models for different task types
# 3. **Failover chains:** Primary → secondary → tertiary models
# 4. **Health checks:** Monitor model availability, skip unhealthy models
# 5. **Budget-aware routing:** Throttle expensive models when budget is tight
# ======================================================================

# ======================================================================
# ## Exercises: Try It Yourself
# 1. **Multi-criteria optimization:** Modify routing to balance all 3 dimensions (cost, latency, accuracy).
# 2. **Request-type mapping:** Add 10+ request types with optimal model assignment.
# 3. **Cost simulation:** Track daily spending across 1 week with varying loads.
# 4. **Failover chains:** Test cascading failures (3+ models down sequentially).
# ======================================================================
