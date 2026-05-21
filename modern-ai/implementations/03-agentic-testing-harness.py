"""
Auto-generated from 03-agentic-testing-harness.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Agentic Testing Harness
# ## Learning Objectives
# 1. Understand agent-environment interaction patterns and observe-plan-act loops
# 2. Implement a mock agent executor with tool registry and assertion frameworks
# ======================================================================

import numpy as np
import torch
import time
import json
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import matplotlib.pyplot as plt

# Device setup for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ======================================================================
# ## Level 1: Basic Mock Agent with Tool Registry
# ======================================================================

# Level 1: Simple agent with observe-plan-act loop and tool registry
class ToolRegistry:
    """Registry for available tools an agent can use"""
    
    def __init__(self):
        self.tools = {}
    
    def register(self, name: str, func: Callable, description: str = ''):
        """Register a tool"""
        self.tools[name] = {
            'func': func,
            'description': description
        }
    
    def execute(self, tool_name: str, *args, **kwargs):
        """Execute a registered tool"""
        if tool_name not in self.tools:
            raise ValueError(f'Tool not found: {tool_name}')
        return self.tools[tool_name]['func'](*args, **kwargs)
    
    def list_tools(self):
        """List available tools"""
        return {name: self.tools[name]['description'] for name in self.tools}

class BasicMockAgent:
    """Simple agent with observe-plan-act loop"""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.trace = []  # Track all steps
        self.state = {}
    
    def observe(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Observe current state from environment"""
        observation = {
            'timestamp': time.time(),
            'environment_state': environment.copy(),
            'agent_memory': self.state.copy()
        }
        self.trace.append(('observe', observation))
        return observation
    
    def plan(self, observation: Dict[str, Any]) -> str:
        """Plan next action based on observation"""
        # Simplified planning logic
        if 'goal' in observation['environment_state']:
            plan = 'search'
        else:
            plan = 'idle'
        self.trace.append(('plan', plan))
        return plan
    
    def act(self, action: str, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action using tools"""
        try:
            if action == 'search':
                result = self.registry.execute('search', environment.get('query', ''))
            elif action == 'analyze':
                result = self.registry.execute('analyze', environment.get('data', {}))
            else:
                result = {'status': 'idle'}
            
            self.trace.append(('act', {'action': action, 'result': result}))
            return result
        except Exception as e:
            error = {'error': str(e), 'action': action}
            self.trace.append(('act_error', error))
            return error
    
    def run_cycle(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one observe-plan-act cycle"""
        obs = self.observe(environment)
        action = self.plan(obs)
        result = self.act(action, environment)
        return {'observation': obs, 'action': action, 'result': result}

# Set up tools
registry = ToolRegistry()
registry.register('search', lambda q: {'results': [f'Result for {q}'], 'count': 1}, 'Search for information')
registry.register('analyze', lambda d: {'summary': 'Analyzed data'}, 'Analyze provided data')

# Test agent
agent = BasicMockAgent(registry)
env = {'goal': 'find information', 'query': 'machine learning'}

result = agent.run_cycle(env)
print("Agent Observe-Plan-Act Cycle:")
print(f"  Action taken: {result['action']}")
print(f"  Result: {result['result']}")
print(f"  Trace length: {len(agent.trace)} steps")


# ======================================================================
# ## Level 2: Advanced Agent with Async Execution, Timeouts, and Retry Logic
# ======================================================================

class AdvancedAgentExecutor:
    """Production agent executor with async tools, timeouts, and retry logic"""
    
    def __init__(self, max_retries: int = 3, timeout_seconds: float = 5.0):
        self.registry = ToolRegistry()
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.execution_history = []
        self.failure_count = 0
    
    def simulate_tool_execution(self, tool_name: str, *args, timeout: float = None) -> Dict[str, Any]:
        """Simulate tool execution with potential failure"""
        # Randomly fail some tools to test error handling
        fail_rate = 0.2  # 20% failure rate
        if np.random.random() < fail_rate:
            raise RuntimeError(f'Tool {tool_name} timed out or failed')
        
        # Simulate work
        time.sleep(np.random.uniform(0.1, 0.3))
        return {'status': 'success', 'tool': tool_name, 'output': f'Result from {tool_name}'}
    
    def execute_with_retry(self, tool_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute tool with retry logic and timeout handling"""
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.time()
                result = self.simulate_tool_execution(tool_name, *args, timeout=self.timeout_seconds)
                elapsed = time.time() - start
                
                execution_record = {
                    'tool': tool_name,
                    'attempt': attempt,
                    'status': 'success',
                    'elapsed': elapsed,
                    'result': result
                }
                self.execution_history.append(execution_record)
                return result
            
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = 2 ** (attempt - 1)  # Exponential backoff
                    time.sleep(min(wait_time, 1.0))  # Cap at 1 second
        
        # All retries failed
        self.failure_count += 1
        execution_record = {
            'tool': tool_name,
            'attempt': self.max_retries,
            'status': 'failed',
            'error': str(last_error)
        }
        self.execution_history.append(execution_record)
        return {'status': 'failed', 'error': str(last_error), 'tool': tool_name}
    
    def test_agent_recovery(self, num_steps: int = 5) -> Dict[str, Any]:
        """Test agent's ability to recover from tool failures"""
        successful = 0
        failed = 0
        
        for step in range(num_steps):
            tool = np.random.choice(['search', 'synthesize', 'verify'])
            result = self.execute_with_retry(tool)
            
            if result.get('status') == 'success':
                successful += 1
            else:
                failed += 1
        
        return {
            'total_steps': num_steps,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / num_steps,
            'execution_history': self.execution_history
        }

# Test advanced executor
executor = AdvancedAgentExecutor(max_retries=3, timeout_seconds=5.0)
recovery_test = executor.test_agent_recovery(num_steps=8)

print("\nAdvanced Agent Recovery Test:")
print(f"  Total tool executions: {recovery_test['total_steps']}")
print(f"  Successful: {recovery_test['successful']}")
print(f"  Failed: {recovery_test['failed']}")
print(f"  Success rate: {recovery_test['success_rate']:.1%}")


# ======================================================================
# ## Real-World Example 1: Multi-Step Research Agent
# ======================================================================

# Example 1: Multi-step agent (research -> synthesize -> summarize)
class ResearchAgent:
    """Agent that performs multi-step research task"""
    
    def __init__(self):
        self.registry = ToolRegistry()
        self._setup_tools()
        self.steps = []
    
    def _setup_tools(self):
        """Register mock research tools"""
        self.registry.register(
            'wikipedia_lookup',
            lambda query: {
                'source': 'Wikipedia',
                'query': query,
                'content': f'Information about {query}: Definition, history, applications...'
            },
            'Look up information on Wikipedia'
        )
        
        self.registry.register(
            'synthesize_information',
            lambda data: {
                'synthesis': f'Combined knowledge: {len(data)} sources integrated',
                'confidence': 0.85
            },
            'Combine multiple sources into coherent explanation'
        )
        
        self.registry.register(
            'generate_summary',
            lambda content: {
                'summary': f'Key points from {len(content)} characters of content',
                'length': 'concise'
            },
            'Generate summary from content'
        )
    
    def execute_research_workflow(self, topic: str) -> Dict[str, Any]:
        """Execute research -> synthesize -> summarize workflow"""
        results = {}
        
        # Step 1: Research
        print(f"Step 1: Researching '{topic}'...")
        research_result = self.registry.execute('wikipedia_lookup', topic)
        results['research'] = research_result
        self.steps.append(('research', research_result))
        
        # Step 2: Synthesize
        print(f"Step 2: Synthesizing information...")
        synthesis_result = self.registry.execute('synthesize_information', [research_result])
        results['synthesis'] = synthesis_result
        self.steps.append(('synthesize', synthesis_result))
        
        # Step 3: Summarize
        print(f"Step 3: Generating summary...")
        summary_result = self.registry.execute('generate_summary', research_result['content'])
        results['summary'] = summary_result
        self.steps.append(('summarize', summary_result))
        
        return results

# Run research workflow
agent = ResearchAgent()
workflow_result = agent.execute_research_workflow('Machine Learning')

print("\nWorkflow Results:")
print(f"  Research confidence: {workflow_result['research']['source']}")
print(f"  Synthesis confidence: {workflow_result['synthesis']['confidence']:.0%}")
print(f"  Summary length: {workflow_result['summary']['length']}")
print(f"  Total steps executed: {len(agent.steps)}")


# ======================================================================
# ## Real-World Example 2: Agent Error Handling and Recovery
# ======================================================================

# Example 2: Test agent behavior when tools fail
class ResilientAgent:
    """Agent that gracefully handles and recovers from tool failures"""
    
    def __init__(self):
        self.registry = ToolRegistry()
        self.failure_scenarios = []
        self.recovery_actions = []
    
    def simulate_flaky_tool(self, success_rate: float = 0.7) -> Dict[str, Any]:
        """Simulate a tool that fails sometimes"""
        if np.random.random() < success_rate:
            return {'status': 'success', 'data': 'Tool produced output'}
        else:
            raise RuntimeError('Tool execution failed - possibly network error')
    
    def handle_tool_failure(self, tool_name: str, error: Exception) -> Dict[str, Any]:
        """Handle tool failure with appropriate recovery strategy"""
        error_msg = str(error)
        
        # Log failure
        self.failure_scenarios.append({
            'tool': tool_name,
            'error': error_msg,
            'timestamp': time.time()
        })
        
        # Determine recovery action
        if 'network' in error_msg.lower():
            recovery = 'retry_with_backoff'
        elif 'timeout' in error_msg.lower():
            recovery = 'increase_timeout'
        elif 'invalid' in error_msg.lower():
            recovery = 'validate_input'
        else:
            recovery = 'fallback_strategy'
        
        self.recovery_actions.append(recovery)
        return {'error': error_msg, 'recovery_action': recovery}
    
    def execute_with_fallback(self, tool_name: str, fallback_data: Dict = None) -> Dict[str, Any]:
        """Execute tool with fallback if primary fails"""
        try:
            return self.simulate_flaky_tool(success_rate=0.6)
        except Exception as e:
            recovery = self.handle_tool_failure(tool_name, e)
            
            # Use fallback data if available
            if fallback_data:
                return {'status': 'recovered_with_fallback', 'data': fallback_data}
            else:
                return {'status': 'failed', 'error': str(e)}
    
    def run_error_scenario_tests(self, num_trials: int = 10) -> Dict[str, Any]:
        """Run multiple tool executions to test error handling"""
        successful = 0
        recovered = 0
        failed = 0
        
        for i in range(num_trials):
            result = self.execute_with_fallback('tool', fallback_data={'mock': 'data'})
            
            if result.get('status') == 'success':
                successful += 1
            elif result.get('status') == 'recovered_with_fallback':
                recovered += 1
            else:
                failed += 1
        
        return {
            'total_trials': num_trials,
            'successful_executions': successful,
            'recovered_with_fallback': recovered,
            'unrecovered_failures': failed,
            'total_failures': len(self.failure_scenarios),
            'recovery_strategy_distribution': dict(Counter(self.recovery_actions))
        }

# Test resilient agent
resilient_agent = ResilientAgent()
error_test = resilient_agent.run_error_scenario_tests(num_trials=15)

print("\nAgent Error Handling Test Results:")
print(f"  Successful executions: {error_test['successful_executions']}")
print(f"  Recovered with fallback: {error_test['recovered_with_fallback']}")
print(f"  Unrecovered failures: {error_test['unrecovered_failures']}")
print(f"  Total failures encountered: {error_test['total_failures']}")
print(f"  Recovery strategies used: {error_test['recovery_strategy_distribution']}")


# ======================================================================
# ## Real-World Example 3: Trace-Based Test Harness with Assertions
# ======================================================================

# Example 3: Trace-based testing with step validation
class TraceBasedTestHarness:
    """Test agent behavior by validating execution traces against expected behavior"""
    
    def __init__(self):
        self.tests = []
        self.assertions = []
    
    def assert_step_exists(self, trace: List[Tuple], step_name: str) -> bool:
        """Assert that specific step was executed"""
        result = any(t[0] == step_name for t in trace)
        self.assertions.append({'type': 'step_exists', 'step': step_name, 'passed': result})
        return result
    
    def assert_step_order(self, trace: List[Tuple], expected_order: List[str]) -> bool:
        """Assert that steps occurred in expected order"""
        trace_steps = [t[0] for t in trace]
        
        # Check if expected order appears in actual trace
        result = True
        for i, expected_step in enumerate(expected_order):
            if i < len(trace_steps) and trace_steps[i] == expected_step:
                continue
            else:
                result = False
                break
        
        self.assertions.append({'type': 'step_order', 'expected': expected_order, 'passed': result})
        return result
    
    def assert_no_errors_in_trace(self, trace: List[Tuple]) -> bool:
        """Assert that no error steps occurred"""
        has_errors = any('error' in t[0].lower() for t in trace)
        result = not has_errors
        self.assertions.append({'type': 'no_errors', 'passed': result})
        return result
    
    def test_happy_path(self) -> Dict[str, Any]:
        """Test agent executing happy path successfully"""
        # Simulate agent trace for happy path
        happy_trace = [
            ('observe', {}),
            ('plan', 'search'),
            ('act', {'action': 'search'}),
            ('observe', {}),
            ('plan', 'analyze'),
            ('act', {'action': 'analyze'}),
        ]
        
        # Run assertions
        tests = {
            'observe_called': self.assert_step_exists(happy_trace, 'observe'),
            'correct_order': self.assert_step_order(happy_trace, ['observe', 'plan', 'act']),
            'no_errors': self.assert_no_errors_in_trace(happy_trace)
        }
        
        return {'happy_path': tests, 'trace_length': len(happy_trace)}
    
    def test_error_scenario(self) -> Dict[str, Any]:
        """Test agent executing with errors and recovery"""
        # Simulate agent trace with error recovery
        error_trace = [
            ('observe', {}),
            ('plan', 'search'),
            ('act_error', {'error': 'Network timeout'}),
            ('plan_recovery', 'retry'),
            ('act', {'action': 'search', 'retry': 1}),
        ]
        
        # Run assertions
        tests = {
            'error_detected': any('error' in t[0].lower() for t in error_trace),
            'recovery_attempted': any('recovery' in t[0].lower() for t in error_trace),
            'ended_with_action': error_trace[-1][0] == 'act'
        }
        
        return {'error_scenario': tests, 'trace_length': len(error_trace)}
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run all test scenarios"""
        happy_path = self.test_happy_path()
        error_scenario = self.test_error_scenario()
        
        total_assertions = sum(1 for a in self.assertions)
        passed_assertions = sum(1 for a in self.assertions if a['passed'])
        
        return {
            'happy_path_results': happy_path,
            'error_scenario_results': error_scenario,
            'total_assertions': total_assertions,
            'passed_assertions': passed_assertions,
            'pass_rate': passed_assertions / total_assertions if total_assertions > 0 else 0
        }

# Run test harness
harness = TraceBasedTestHarness()
test_results = harness.run_full_test_suite()

print("\nTrace-Based Test Results:")
print(f"Happy path: {test_results['happy_path_results']['happy_path']}")
print(f"Error scenario: {test_results['error_scenario_results']['error_scenario']}")
print(f"Total assertions: {test_results['total_assertions']}")
print(f"Passed: {test_results['passed_assertions']} ({test_results['pass_rate']:.0%})")


# ======================================================================
# ## Comparison: Test Coverage Matrix
# ======================================================================

# Test coverage matrix across scenarios
import matplotlib.pyplot as plt

# Coverage data: coverage percentage for different test scenarios
test_scenarios = [
    'Happy Path',
    'Tool Timeout',
    'Tool Failure',
    'Invalid Input',
    'Resource Exhaustion',
    'Concurrent Calls'
]

# Pass/fail rates for different test types
test_results_matrix = {
    'Functionality': [1.0, 0.95, 0.85, 0.90, 0.70, 0.75],
    'Error Handling': [1.0, 0.90, 1.0, 0.85, 0.80, 0.70],
    'Recovery': [1.0, 0.85, 0.90, 0.80, 0.60, 0.65],
    'Performance': [1.0, 0.75, 0.70, 0.95, 0.50, 0.55]
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap of coverage
data = np.array([test_results_matrix[test_type] for test_type in test_results_matrix.keys()])
im = ax1.imshow(data, cmap='RdYlGn', vmin=0, vmax=1)
ax1.set_xticks(range(len(test_scenarios)))
ax1.set_yticks(range(len(test_results_matrix)))
ax1.set_xticklabels(test_scenarios, rotation=45, ha='right')
ax1.set_yticklabels(test_results_matrix.keys())
ax1.set_title('Test Coverage Matrix (Pass Rate)')

# Add percentages
for i in range(len(test_results_matrix)):
    for j in range(len(test_scenarios)):
        text = ax1.text(j, i, f'{data[i, j]:.0%}',
                       ha="center", va="center", color="black", fontsize=10)

plt.colorbar(im, ax=ax1, label='Pass Rate')

# Summary coverage by scenario
scenario_coverage = np.mean(data, axis=0)
colors = ['green' if c >= 0.8 else 'orange' if c >= 0.7 else 'red' for c in scenario_coverage]
ax2.bar(range(len(test_scenarios)), scenario_coverage, color=colors)
ax2.set_xticks(range(len(test_scenarios)))
ax2.set_xticklabels(test_scenarios, rotation=45, ha='right')
ax2.set_ylim([0, 1.1])
ax2.set_ylabel('Coverage')
ax2.set_title('Overall Coverage by Scenario')
ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target (80%)')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

for i, v in enumerate(scenario_coverage):
    ax2.text(i, v + 0.05, f'{v:.0%}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('agentic_test_coverage_matrix.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nTest Coverage by Scenario:")
for scenario, cov in zip(test_scenarios, scenario_coverage):
    status = '✓' if cov >= 0.8 else '⚠️' if cov >= 0.7 else '✗'
    print(f"  {status} {scenario:<25} {cov:.0%}")


# ======================================================================
# ## Key Takeaways
# ### Core Concept
# Agent testing validates the observe-plan-act loop by systematically testing tool execution, error handling, and multi-step reasoning. Trace-based testing allows inspection of every decision step, making failures observable and reproducible.
# ======================================================================

# ======================================================================
# ## Try It Yourself
# 1. **Add custom assertions**: Extend `TraceBasedTestHarness` with assertions for tool output validation and resource usage.
# 2. **Test concurrent agent execution**: Run multiple agents in parallel and verify they don't interfere or corrupt shared state.
# ======================================================================
