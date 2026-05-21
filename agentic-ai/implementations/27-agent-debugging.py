"""
Auto-generated from 27-agent-debugging.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Agent Debugging
# Learning objectives:
# - Implement structured logging with session IDs
# - Create replay mechanisms for reproducible debugging
# ======================================================================

import os
import json
import uuid
from anthropic import Anthropic
from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

print("Setup complete. Ready for debugging!")


# ======================================================================
# ## Level 1: Structured Logging with Session ID
# Basic structured logging to enable later debugging.
# ======================================================================

class DebuggedAgent:
    """Agent with structured logging."""
    def __init__(self, log_file="agent.jsonl"):
        self.client = Anthropic()
        self.session_id = str(uuid.uuid4())[:8]
        self.step = 0
        self.log_file = log_file
    
    def log_step(self, component_type: str, inputs: dict, outputs: dict, status: str = "success"):
        """Log with structured format."""
        self.step += 1
        entry = {
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "step": self.step,
            "component_type": component_type,
            "inputs": inputs,
            "outputs": outputs,
            "status": status
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        return entry
    
    def execute(self, task: str) -> dict:
        """Execute with logging."""
        # Step 1: LLM reasoning
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{"role": "user", "content": task}]
        )
        
        result = response.content[0].text
        
        # Log the execution
        self.log_step(
            "llm_call",
            {"prompt": task},
            {"response": result[:100], "tokens": response.usage.input_tokens},
            "success"
        )
        
        return {"session_id": self.session_id, "result": result}

# Test
agent = DebuggedAgent()
result = agent.execute("What is 2+2?")
print(f"Session: {result['session_id']}")
print(f"Result: {result['result'][:50]}...")


# ======================================================================
# ## Level 2: Replay Mechanism
# Deterministic replay using logged LLM outputs.
# ======================================================================

class ReplayableAgent:
    """Agent supporting deterministic replay."""
    def __init__(self, log_file="agent.jsonl"):
        self.log_file = log_file
        self.session_logs = {}
    
    def load_session(self, session_id: str) -> list:
        """Load all logs for a session."""
        logs = []
        with open(self.log_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("session_id") == session_id:
                    logs.append(entry)
        
        self.session_logs[session_id] = logs
        return logs
    
    def replay_session(self, session_id: str) -> dict:
        """Replay logged session step-by-step."""
        logs = self.load_session(session_id)
        
        print(f"\nReplaying session {session_id} ({len(logs)} steps):")
        
        summary = {"session_id": session_id, "steps": [], "errors": []}
        
        for log in logs:
            step = log["step"]
            component = log["component_type"]
            status = log["status"]
            status_icon = "✓" if status == "success" else "✗"
            
            print(f"  {status_icon} Step {step}: {component}")
            
            if status == "error":
                summary["errors"].append({"step": step, "component": component, "output": log["outputs"]})
            
            summary["steps"].append({"step": step, "component": component, "status": status})
        
        return summary

# Test
replayer = ReplayableAgent()
if os.path.exists("agent.jsonl"):
    summary = replayer.replay_session(agent.session_id)
    print(f"\nSession summary: {len(summary['steps'])} steps, {len(summary['errors'])} errors")


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: Execution Tracing with Root Cause Analysis
# ======================================================================

class ExecutionTracer:
    """Trace agent execution and identify root cause."""
    def __init__(self):
        self.trace = []
    
    def record_step(self, step_num: int, component: str, status: str, details: dict = None):
        """Record execution step."""
        entry = {
            "step": step_num,
            "component": component,
            "status": status,
            "details": details or {}
        }
        self.trace.append(entry)
    
    def visualize_trace(self) -> str:
        """Visualize execution trace."""
        lines = ["\nExecution Trace:"]
        for entry in self.trace:
            icon = "✓" if entry["status"] == "success" else "✗"
            lines.append(f"  {icon} Step {entry['step']}: {entry['component']}")
            if entry["status"] == "error":
                lines.append(f"     Error: {entry['details'].get('error', 'Unknown')}")
        
        return "\n".join(lines)
    
    def analyze_failure(self) -> dict:
        """Analyze what went wrong."""
        for i, entry in enumerate(self.trace):
            if entry["status"] == "error":
                return {
                    "failure_step": entry["step"],
                    "failed_component": entry["component"],
                    "prior_steps": [e for e in self.trace[:i]],
                    "error": entry["details"].get("error"),
                    "root_cause": self._infer_cause(entry)
                }
        
        return {"status": "No failures found"}
    
    def _infer_cause(self, failed_entry: dict) -> str:
        """Infer root cause from component type."""
        component = failed_entry["component"]
        error = failed_entry["details"].get("error", "")
        
        if component == "tool_call":
            return "Tool failed or returned error"
        elif component == "llm_reasoning":
            return "LLM reasoning error"
        elif component == "parameter_extraction":
            return "Failed to extract parameters correctly"
        else:
            return "Unknown component failure"

# Test
tracer = ExecutionTracer()
tracer.record_step(1, "llm_reasoning", "success", {"reasoning": "Understood task"})
tracer.record_step(2, "tool_selection", "success", {"selected_tool": "search"})
tracer.record_step(3, "tool_call", "error", {"error": "Invalid parameters"})

print(tracer.visualize_trace())
analysis = tracer.analyze_failure()
print(f"\nAnalysis: {json.dumps(analysis, indent=2)}")


# ======================================================================
# ### Example 2: Log Search and Filtering
# ======================================================================

class LogSearcher:
    """Search and filter logs for debugging."""
    def __init__(self, log_file="agent.jsonl"):
        self.log_file = log_file
        self.logs = []
        self._load_all_logs()
    
    def _load_all_logs(self):
        """Load all logs from file."""
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        self.logs.append(json.loads(line))
                    except:
                        pass
        except FileNotFoundError:
            pass
    
    def find_by_session(self, session_id: str) -> list:
        """Find logs by session ID."""
        return [l for l in self.logs if l.get("session_id") == session_id]
    
    def find_errors(self) -> list:
        """Find all error logs."""
        return [l for l in self.logs if l.get("status") == "error"]
    
    def find_by_component(self, component: str) -> list:
        """Find logs by component type."""
        return [l for l in self.logs if l.get("component_type") == component]
    
    def get_report(self) -> dict:
        """Generate search report."""
        errors = self.find_errors()
        return {
            "total_logs": len(self.logs),
            "errors": len(errors),
            "unique_sessions": len(set(l.get("session_id") for l in self.logs)),
            "error_rate": f"{100*len(errors)/max(1, len(self.logs)):.1f}%",
            "error_components": list(set(e.get("component_type") for e in errors))
        }

# Test
if os.path.exists("agent.jsonl") and os.path.getsize("agent.jsonl") > 0:
    searcher = LogSearcher()
    report = searcher.get_report()
    print(f"Log report: {json.dumps(report, indent=2)}")
else:
    print("No logs yet. Run the agent first.")


# ======================================================================
# ### Example 3: Interactive Debugging Console
# ======================================================================

class DebuggingConsole:
    """Interactive console for debugging agent issues."""
    def __init__(self, log_file="agent.jsonl"):
        self.log_file = log_file
        self.searcher = LogSearcher(log_file)
        self.selected_session = None
    
    def select_session(self, session_id: str):
        """Select session for detailed inspection."""
        logs = self.searcher.find_by_session(session_id)
        if not logs:
            print(f"Session {session_id} not found")
            return
        
        self.selected_session = session_id
        print(f"\nSelected session {session_id}:")
        print(f"  Steps: {len(logs)}")
        
        errors = [l for l in logs if l.get("status") == "error"]
        if errors:
            print(f"  Errors: {len(errors)}")
            for error in errors:
                print(f"    Step {error['step']}: {error['component_type']}")
        else:
            print(f"  Status: All steps successful")
    
    def inspect_step(self, step_num: int) -> dict:
        """Inspect a specific step."""
        if not self.selected_session:
            print("No session selected")
            return {}
        
        logs = self.searcher.find_by_session(self.selected_session)
        step_log = next((l for l in logs if l["step"] == step_num), None)
        
        if not step_log:
            print(f"Step {step_num} not found")
            return {}
        
        print(f"\nStep {step_num} Details:")
        print(f"  Component: {step_log['component_type']}")
        print(f"  Status: {step_log['status']}")
        print(f"  Inputs: {json.dumps(step_log.get('inputs', {}), indent=4)}")
        print(f"  Outputs: {json.dumps(step_log.get('outputs', {}), indent=4)}")
        
        return step_log
    
    def show_available_sessions(self):
        """Show all available sessions for selection."""
        sessions = set(l.get("session_id") for l in self.searcher.logs)
        print(f"\nAvailable sessions: {len(sessions)}")
        for session in list(sessions)[:10]:  # Show first 10
            print(f"  - {session}")

# Test
if os.path.exists("agent.jsonl") and os.path.getsize("agent.jsonl") > 0:
    console = DebuggingConsole()
    console.show_available_sessions()
else:
    print("No debug logs available yet.")


# ======================================================================
# ## Key Takeaways
# 1. **Structured logging is essential.** Log every step with session ID, timestamp, component, inputs, outputs. Without logs, you can't debug.
# 2. **Store actual LLM responses.** Don't just log tokens. Store the response text so you can see what the model actually said.
# ======================================================================
