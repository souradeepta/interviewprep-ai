"""
Auto-generated from 20-agent-routing.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Agent Routing
# Learning objectives:
# - Understand routing strategies: rule-based, semantic, load-aware, hierarchical
# - Implement classification and agent selection
# ======================================================================

import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv
from typing import Dict, Tuple

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

print("Setup complete. Ready for routing!")


# ======================================================================
# ## Level 1: Rule-Based Routing
# Simple keyword-based classification.
# ======================================================================

class RuleBasedRouter:
    """Route queries based on keyword rules"""
    def __init__(self):
        self.agents = {
            "billing": "Handles invoices, payments, refunds",
            "technical": "Handles bugs, errors, crashes",
            "general": "General information and fallback"
        }
        self.client = Anthropic()
    
    def classify(self, query: str) -> str:
        """Classify query type using rules"""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ["invoice", "payment", "billing", "refund"]):
            return "billing"
        elif any(w in query_lower for w in ["error", "bug", "crash", "broken"]):
            return "technical"
        return "general"
    
    def route(self, query: str) -> Dict:
        agent_type = self.classify(query)
        print(f"\n[Rule Router] → {agent_type} agent")
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"[You are a {agent_type} specialist]\n{query}"
            }]
        )
        
        return {
            "agent_type": agent_type,
            "confidence": "100%",
            "response": response.content[0].text
        }

# Test
router = RuleBasedRouter()
result = router.route("I was charged twice for my subscription")
print(f"Response: {result['response'][:80]}...")


# ======================================================================
# ## Level 2: Semantic Routing with Confidence
# LLM classifies queries with confidence scores.
# ======================================================================

class SemanticRouter:
    """Route using LLM classification with confidence"""
    def __init__(self):
        self.agents = ["billing", "technical", "sales", "general"]
        self.client = Anthropic()
        self.fallback_agent = "general"
    
    def classify(self, query: str) -> Tuple[str, int]:
        """Use LLM to classify"""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"""Classify this query: {query}
                
Categories: billing, technical, sales, general

Response format:
CATEGORY: [pick one]
CONFIDENCE: [0-100]"""
            }]
        )
        
        text = response.content[0].text
        category = "general"
        confidence = 50
        
        for line in text.split("\n"):
            if "CATEGORY:" in line:
                cat = line.split(":")[-1].strip().lower()
                if cat in self.agents:
                    category = cat
            elif "CONFIDENCE:" in line:
                try:
                    confidence = int(line.split(":")[-1].strip().rstrip("%"))
                except:
                    pass
        
        return category, confidence
    
    def route(self, query: str) -> Dict:
        agent_type, confidence = self.classify(query)
        
        # Fallback for low confidence
        if confidence < 70:
            print(f"\n[Semantic Router] Low confidence ({confidence}%), using fallback")
            agent_type = self.fallback_agent
        else:
            print(f"\n[Semantic Router] → {agent_type} (confidence: {confidence}%)")
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"[{agent_type.title()} Specialist]\n{query}"
            }]
        )
        
        return {
            "agent_type": agent_type,
            "confidence": f"{confidence}%",
            "response": response.content[0].text
        }

# Test
router = SemanticRouter()
result = router.route("My payment didn't process but I was charged anyway")
print(f"Response: {result['response'][:80]}...")


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: Load-Aware Routing
# ======================================================================

class LoadAwareRouter:
    """Route to least-busy agent of appropriate type"""
    def __init__(self):
        self.agents = {
            "billing_1": {"type": "billing", "load": 2, "busy": False},
            "billing_2": {"type": "billing", "load": 5, "busy": False},
            "technical_1": {"type": "technical", "load": 3, "busy": False},
            "general_1": {"type": "general", "load": 1, "busy": False}
        }
        self.client = Anthropic()
    
    def classify(self, query: str) -> str:
        if any(w in query.lower() for w in ["billing", "payment", "invoice"]):
            return "billing"
        return "general"
    
    def select_agent(self, agent_type: str) -> str:
        """Pick least-busy agent of type"""
        candidates = [
            (name, info) for name, info in self.agents.items()
            if info["type"] == agent_type and not info["busy"]
        ]
        
        if not candidates:
            # All busy, pick least loaded
            candidates = [
                (name, info) for name, info in self.agents.items()
                if info["type"] == agent_type
            ]
        
        best = min(candidates, key=lambda x: x[1]["load"])
        return best[0]
    
    def route(self, query: str) -> Dict:
        agent_type = self.classify(query)
        agent_name = self.select_agent(agent_type)
        load = self.agents[agent_name]["load"]
        
        print(f"\n[Load Router] → {agent_name} (load: {load})")
        
        # Increment load
        self.agents[agent_name]["load"] += 1
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"[Agent: {agent_name}]\n{query}"
            }]
        )
        
        # Decrement load after processing
        self.agents[agent_name]["load"] -= 1
        
        return {
            "agent": agent_name,
            "load": load,
            "response": response.content[0].text
        }

# Test: multiple requests
router = LoadAwareRouter()
print("\nBefore routing:")
print(json.dumps({k: v["load"] for k, v in router.agents.items()}, indent=2))

for i in range(2):
    result = router.route(f"I have a billing question #{i+1}")

print("\nAfter routing:")
print(json.dumps({k: v["load"] for k, v in router.agents.items()}, indent=2))


# ======================================================================
# ### Example 2: Hierarchical Routing with Teams
# ======================================================================

class HierarchicalRouter:
    """Route to teams, which delegate internally"""
    def __init__(self):
        self.teams = {
            "customer_service": ["billing_agent", "general_agent"],
            "engineering": ["technical_agent", "infrastructure_agent"]
        }
        self.client = Anthropic()
    
    def route_to_team(self, query: str) -> str:
        """Decide which team"""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": f"Should this go to customer_service or engineering team?\n{query}\nRespond with just the team name."
            }]
        )
        
        text = response.content[0].text.lower()
        if "engineering" in text:
            return "engineering"
        return "customer_service"
    
    def route(self, query: str) -> Dict:
        team = self.route_to_team(query)
        print(f"\n[Hierarchical Router] → {team} team")
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"[Team: {team}]\nHandle this request: {query}"
            }]
        )
        
        return {
            "team": team,
            "response": response.content[0].text
        }

# Test
router = HierarchicalRouter()
result = router.route("The server is running out of disk space")
print(f"Response: {result['response'][:100]}...")


# ======================================================================
# ### Example 3: Hybrid Routing (Rules + Semantic Fallback)
# ======================================================================

class HybridRouter:
    """Fast rules first, semantic for edge cases"""
    def __init__(self):
        self.client = Anthropic()
        self.routing_stats = {"rules_matched": 0, "semantic_needed": 0}
    
    def try_rule_match(self, query: str) -> str or None:
        """Try rule-based classification first (fast)"""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ["invoice", "billing", "payment"]):
            return "billing"
        elif any(w in query_lower for w in ["error", "bug", "crash"]):
            return "technical"
        
        return None  # No rule match
    
    def semantic_classify(self, query: str) -> str:
        """Fallback to semantic classification (slower)"""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": f"Classify as: billing, technical, or general.\nQuery: {query}\nAnswer with one word."
            }]
        )
        
        text = response.content[0].text.lower()
        for category in ["billing", "technical", "general"]:
            if category in text:
                return category
        return "general"
    
    def route(self, query: str) -> Dict:
        # Try rules first (1ms)
        agent_type = self.try_rule_match(query)
        
        if agent_type:
            print(f"\n[Hybrid] Rule match → {agent_type}")
            self.routing_stats["rules_matched"] += 1
        else:
            # Fallback to semantic (100ms)
            agent_type = self.semantic_classify(query)
            print(f"\n[Hybrid] Semantic fallback → {agent_type}")
            self.routing_stats["semantic_needed"] += 1
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"[{agent_type.title()} Agent]\n{query}"
            }]
        )
        
        return {
            "agent_type": agent_type,
            "response": response.content[0].text
        }

# Test hybrid
router = HybridRouter()
queries = [
    "I was charged twice",  # Rules match
    "My invoice doesn't match my memory"  # Semantic needed
]

for q in queries:
    router.route(q)

print(f"\nStats: {json.dumps(router.routing_stats, indent=2)}")


# ======================================================================
# ## Key Takeaways
# 1. **Start with rules, add semantic fallback.** Rules are fast (1ms), work for 80% of cases. Semantic for edge cases.
# 2. **Use confidence scores.** If classification confidence <70%, escalate or try fallback agent.
# ======================================================================
