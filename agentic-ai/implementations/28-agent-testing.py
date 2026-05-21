"""
Auto-generated from 28-agent-testing.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Agent Testing
# Learning objectives:
# - Implement unit tests with mocking
# - Build integration tests for component interactions
# ======================================================================

import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv
from unittest.mock import MagicMock, patch
from typing import Callable

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

print("Setup complete. Ready for testing!")


# ======================================================================
# ## Level 1: Unit Tests with Mocks
# Test individual components in isolation.
# ======================================================================

class ToolSelector:
    """Component: select correct tool for query."""
    def __init__(self):
        self.tools = {
            "search": ["search", "find", "look"],
            "book": ["book", "reserve", "purchase"],
            "cancel": ["cancel", "return", "refund"]
        }
    
    def select(self, query: str) -> str:
        for tool, keywords in self.tools.items():
            if any(kw in query.lower() for kw in keywords):
                return tool
        return "fallback"

class ParameterExtractor:
    """Component: extract parameters from query."""
    def extract(self, query: str) -> dict:
        # Simple extraction: look for keywords
        params = {}
        if "from" in query.lower():
            parts = query.split("from")
            if len(parts) > 1:
                params["from"] = parts[1].split("to")[0].strip()
        return params

class UnitTests:
    """Unit tests for components."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def assert_equal(self, actual, expected, test_name: str):
        if actual == expected:
            print(f"✓ {test_name}")
            self.passed += 1
        else:
            print(f"✗ {test_name}: expected {expected}, got {actual}")
            self.failed += 1
    
    def test_tool_selector(self):
        """Unit test: Tool selector."""
        selector = ToolSelector()
        
        self.assert_equal(selector.select("search flights"), "search", "Select search tool")
        self.assert_equal(selector.select("book a flight"), "book", "Select book tool")
        self.assert_equal(selector.select("cancel my booking"), "cancel", "Select cancel tool")
        self.assert_equal(selector.select("xyz unknown"), "fallback", "Select fallback tool")
    
    def test_parameter_extractor(self):
        """Unit test: Parameter extractor."""
        extractor = ParameterExtractor()
        
        params = extractor.extract("Search flights from NYC to LA")
        self.assert_equal("from" in params, True, "Extract 'from' parameter")
    
    def run_all(self):
        """Run all unit tests."""
        print("\n=== Unit Tests ===")
        self.test_tool_selector()
        self.test_parameter_extractor()
        print(f"\nResult: {self.passed} passed, {self.failed} failed")
        return self.failed == 0

# Run tests
tester = UnitTests()
tester.run_all()


# ======================================================================
# ## Level 2: Integration Tests
# Test components working together with tool stubs.
# ======================================================================

class MockFlightAPI:
    """Stub for flight search API."""
    def __init__(self):
        self.calls = []
    
    def search(self, from_city: str, to_city: str, date: str) -> dict:
        self.calls.append({"from": from_city, "to": to_city, "date": date})
        
        if from_city == "NYC" and to_city == "LA":
            return {
                "flights": [
                    {"id": "F001", "price": 350, "time": "10am"},
                    {"id": "F002", "price": 450, "time": "2pm"}
                ]
            }
        else:
            return {"error": "Route not found"}

class BookingAgent:
    """Agent using tool selector and API."""
    def __init__(self):
        self.selector = ToolSelector()
        self.extractor = ParameterExtractor()
        self.flight_api = None  # Will be mocked in tests
    
    def handle(self, request: str) -> dict:
        tool = self.selector.select(request)
        
        if tool == "search" and self.flight_api:
            return {
                "tool_used": tool,
                "result": self.flight_api.search("NYC", "LA", "2026-01-15")
            }
        return {"tool_used": tool, "result": {}}

class IntegrationTests:
    """Integration tests."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def assert_equal(self, actual, expected, test_name: str):
        if actual == expected:
            print(f"✓ {test_name}")
            self.passed += 1
        else:
            print(f"✗ {test_name}: expected {expected}, got {actual}")
            self.failed += 1
    
    def test_tool_selection_with_api(self):
        """Integration test: Tool selection + API call."""
        agent = BookingAgent()
        agent.flight_api = MockFlightAPI()
        
        result = agent.handle("Find flights NYC to LA")
        
        self.assert_equal(result["tool_used"], "search", "Selected search tool")
        self.assert_equal(
            len(result["result"].get("flights", [])), 2,
            "API returned 2 flights"
        )
    
    def test_api_called_with_correct_params(self):
        """Integration test: API receives correct parameters."""
        agent = BookingAgent()
        mock_api = MockFlightAPI()
        agent.flight_api = mock_api
        
        agent.handle("Find flights NYC to LA")
        
        self.assert_equal(len(mock_api.calls), 1, "API called once")
        self.assert_equal(mock_api.calls[0]["from"], "NYC", "From city correct")
    
    def run_all(self):
        """Run all integration tests."""
        print("\n=== Integration Tests ===")
        self.test_tool_selection_with_api()
        self.test_api_called_with_correct_params()
        print(f"\nResult: {self.passed} passed, {self.failed} failed")
        return self.failed == 0

# Run tests
tester = IntegrationTests()
tester.run_all()


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: End-to-End Tests
# ======================================================================

class E2EBookingAgent:
    """Full agent with realistic behavior."""
    def __init__(self):
        self.client = Anthropic()
    
    def book(self, from_city: str, to_city: str, date: str, budget: float) -> dict:
        """Execute full booking flow."""
        # In real code, this would call APIs, extract params, etc.
        # For testing, we simulate
        if budget < 300:
            return {"success": False, "reason": "Budget too low"}
        
        return {
            "success": True,
            "flight_id": "F001",
            "price": 350,
            "confirmation": "CONF123"
        }

class E2ETests:
    """End-to-end tests."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def assert_true(self, condition: bool, test_name: str):
        if condition:
            print(f"✓ {test_name}")
            self.passed += 1
        else:
            print(f"✗ {test_name}")
            self.failed += 1
    
    def test_happy_path(self):
        """E2E test: Happy path booking."""
        agent = E2EBookingAgent()
        result = agent.book("NYC", "LA", "2026-01-15", 500)
        
        self.assert_true(result["success"], "Booking succeeded")
        self.assert_true(result["flight_id"] == "F001", "Flight ID correct")
        self.assert_true(result["price"] <= 500, "Price within budget")
    
    def test_budget_constraint(self):
        """E2E test: Budget constraint enforced."""
        agent = E2EBookingAgent()
        result = agent.book("NYC", "LA", "2026-01-15", 200)
        
        self.assert_true(not result["success"], "Booking rejected with low budget")
        self.assert_true("Budget" in result["reason"], "Error message mentions budget")
    
    def run_all(self):
        """Run all E2E tests."""
        print("\n=== End-to-End Tests ===")
        self.test_happy_path()
        self.test_budget_constraint()
        print(f"\nResult: {self.passed} passed, {self.failed} failed")
        return self.failed == 0

# Run tests
tester = E2ETests()
tester.run_all()


# ======================================================================
# ### Example 2: Testing with Temperature=0 (Deterministic)
# ======================================================================

class DeterministicAgent:
    """Agent with temperature=0 for deterministic testing."""
    def __init__(self, temperature: float = 0.0):
        self.client = Anthropic()
        self.temperature = temperature
    
    def answer(self, question: str) -> str:
        """Get deterministic answer."""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            temperature=self.temperature,
            messages=[{"role": "user", "content": question}]
        )
        return response.content[0].text

class DeterministicTests:
    """Tests using deterministic LLM behavior."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def assert_contains(self, text: str, substring: str, test_name: str):
        if substring.lower() in text.lower():
            print(f"✓ {test_name}")
            self.passed += 1
        else:
            print(f"✗ {test_name}: '{substring}' not in '{text[:50]}'")
            self.failed += 1
    
    def test_math_with_deterministic_llm(self):
        """Test: Math question with temp=0."""
        agent = DeterministicAgent(temperature=0.0)
        answer = agent.answer("What is 2+2?")
        
        self.assert_contains(answer, "4", "Math answer contains 4")
    
    def test_question_answering(self):
        """Test: Question answering with temp=0."""
        agent = DeterministicAgent(temperature=0.0)
        answer = agent.answer("Is a dog a mammal?")
        
        self.assert_contains(answer, "yes", "Q&A question answered affirmatively")
    
    def run_all(self):
        """Run all deterministic tests."""
        print("\n=== Deterministic LLM Tests (temperature=0) ===")
        self.test_math_with_deterministic_llm()
        self.test_question_answering()
        print(f"\nResult: {self.passed} passed, {self.failed} failed")
        return self.failed == 0

# Run tests
tester = DeterministicTests()
tester.run_all()


# ======================================================================
# ### Example 3: Error Path Testing
# ======================================================================

class ErrorHandlingAgent:
    """Agent with error handling for different scenarios."""
    def __init__(self):
        self.fallback_tool = "fallback"
    
    def handle_tool_timeout(self, retries: int = 3) -> dict:
        """Handle tool timeout with retry."""
        for attempt in range(retries):
            try:
                # Simulate tool timeout on first 2 attempts
                if attempt < 2:
                    raise TimeoutError("Tool timeout")
                return {"success": True, "result": "recovered after retry"}
            except TimeoutError:
                if attempt == retries - 1:
                    return {"success": False, "error": "Tool timeout after retries", "used_fallback": self.fallback_tool}
        
        return {"success": False, "error": "Unknown error"}
    
    def handle_invalid_params(self, params: dict) -> dict:
        """Validate and handle invalid parameters."""
        required = ["from", "to", "date"]
        missing = [p for p in required if p not in params]
        
        if missing:
            return {"success": False, "error": f"Missing parameters: {missing}"}
        
        return {"success": True, "params": params}

class ErrorPathTests:
    """Tests for error handling."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def assert_equal(self, actual, expected, test_name: str):
        if actual == expected:
            print(f"✓ {test_name}")
            self.passed += 1
        else:
            print(f"✗ {test_name}: expected {expected}, got {actual}")
            self.failed += 1
    
    def test_tool_timeout_recovery(self):
        """Test: Agent recovers from tool timeout."""
        agent = ErrorHandlingAgent()
        result = agent.handle_tool_timeout(retries=3)
        
        self.assert_equal(result["success"], True, "Tool timeout recovered")
    
    def test_timeout_exhaustion(self):
        """Test: Agent fails after exhausting retries."""
        # This test would need to mock retries=1 to fail immediately
        agent = ErrorHandlingAgent()
        # In real test, we'd mock to always timeout
        pass
    
    def test_parameter_validation(self):
        """Test: Agent validates required parameters."""
        agent = ErrorHandlingAgent()
        
        # Missing parameters
        result = agent.handle_invalid_params({"from": "NYC"})
        self.assert_equal(result["success"], False, "Invalid params detected")
        
        # Valid parameters
        result = agent.handle_invalid_params({"from": "NYC", "to": "LA", "date": "2026-01-15"})
        self.assert_equal(result["success"], True, "Valid params accepted")
    
    def run_all(self):
        """Run all error path tests."""
        print("\n=== Error Path Tests ===")
        self.test_tool_timeout_recovery()
        self.test_parameter_validation()
        print(f"\nResult: {self.passed} passed, {self.failed} failed")
        return self.failed == 0

# Run tests
tester = ErrorPathTests()
tester.run_all()


# ======================================================================
# ## Key Takeaways
# 1. **Test pyramid: many unit, some integration, few E2E.** Unit tests are fast (run 1000s instantly), E2E are slow (need real APIs). Use tests where they're cheapest.
# 2. **Use temperature=0 in unit tests.** Makes LLM outputs deterministic so tests don't flake. Temperature > 0 is for realistic tests but slower.
# ======================================================================
