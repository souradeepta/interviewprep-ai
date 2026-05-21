"""
Auto-generated from 24-reflection-and-self-improvement.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Reflection and Self-Improvement
# Objectives: Learning from failures, root cause analysis, lesson extraction, strategy updates, iteration with improvement
# ======================================================================

# Level 1: Basic Reflection Loop

class BasicReflectingAgent:
    def __init__(self):
        self.lessons = []
        self.attempt = 0
        self.max_attempts = 3

    def execute(self, task):
        '''Execute task, return success/failure.'''
        self.attempt += 1
        # Mock: deterministic failure first attempt, success second
        return self.attempt >= 2

    def reflect(self, task, failed=True):
        '''Extract lesson from failure.'''
        if failed:
            lesson = f"Lesson {len(self.lessons)}: Tried {task} and failed; next time use alternative approach"
            self.lessons.append(lesson)
            return True
        return False

    def try_with_reflection(self, task):
        '''Attempt task, reflect on failure, retry.'''
        print(f"Task: {task}\n")
        for i in range(self.max_attempts):
            success = self.execute(task)
            print(f"  Attempt {i+1}: {'✓' if success else '✗'}")

            if success:
                print(f"  Success after {i+1} attempts\n")
                return True

            if i < self.max_attempts - 1:
                self.reflect(task, failed=True)
                print(f"  Learned: {self.lessons[-1]}")

        print(f"  Failed after {self.max_attempts} attempts\n")
        return False

# Test
agent = BasicReflectingAgent()
agent.try_with_reflection("database_query")



# Level 2: Structured Reflection with Root Cause Analysis

class StructuredReflectingAgent:
    def __init__(self):
        self.lessons = {}  # condition -> action mapping
        self.confidence_scores = {}

    def execute_with_context(self, task, context=None):
        '''Execute with optional context.'''
        if context and "use_cache" in context:
            success = True  # Cache works
        else:
            success = hash(task) % 2 == 0  # Random
        return success

    def analyze_failure(self, task, error_type):
        '''Structured root cause analysis.'''
        analysis = {
            "timeout": {"cause": "Tool slow", "action": "use_cache_or_alternative"},
            "wrong_output": {"cause": "Validation issue", "action": "add_output_validation"},
            "parameter_error": {"cause": "Wrong format", "action": "validate_parameters_first"},
        }

        if error_type in analysis:
            return analysis[error_type]
        return {"cause": "Unknown", "action": "try_different_approach"}

    def reflect_and_update(self, task, error_type, confidence=0.8):
        '''Reflect on failure and update strategy.'''
        analysis = self.analyze_failure(task, error_type)

        # Create lesson key from task and error
        key = f"{task}:{error_type}"
        self.lessons[key] = analysis["action"]
        self.confidence_scores[key] = confidence

        print(f"  Root cause: {analysis['cause']}")
        print(f"  New action: {analysis['action']} (confidence: {confidence:.0%})")

    def attempt_with_strategy(self, task):
        '''Attempt using learned strategies.'''
        print(f"Task: {task}")

        # Check if we have a learned strategy
        for lesson_key, action in self.lessons.items():
            if task in lesson_key:
                print(f"  Applying learned strategy: {action}")
                context = {"use_cache": True} if "cache" in action else None
                success = self.execute_with_context(task, context)
                if success:
                    print(f"  ✓ Success using learned strategy\n")
                    return True

        # No learned strategy; try default
        success = self.execute_with_context(task)
        if not success:
            self.reflect_and_update(task, "timeout")
            print()

        return success

# Test
agent = StructuredReflectingAgent()
agent.attempt_with_strategy("database_query")
agent.attempt_with_strategy("database_query")



# Example 1: Confidence-Based Reflection

class ConfidenceAwareAgent:
    def __init__(self):
        self.lessons = []

    def analyze_with_evidence(self, task, error, evidence_list):
        '''Reflect only when confident.'''
        confidence = min(len(evidence_list) / 3, 1.0)

        if confidence < 0.5:
            print(f"  Low confidence ({confidence:.0%}); need more evidence")
            return False

        lesson = {
            "task": task,
            "error": error,
            "confidence": confidence,
            "evidence": evidence_list
        }

        self.lessons.append(lesson)
        print(f"  Learned with {confidence:.0%} confidence")
        return True

# Example
agent = ConfidenceAwareAgent()
print("Example 1 - Confidence-Based Reflection:\n")

# Weak evidence
agent.analyze_with_evidence("task1", "timeout", ["saw timeout once"])

# Strong evidence
agent.analyze_with_evidence("task2", "timeout", ["saw timeout 3 times", "different tool worked", "cache fixed it"])
print()



# Example 2: Iteration with Bounded Attempts

class BoundedReflectionAgent:
    def __init__(self, max_iterations=3):
        self.lessons = []
        self.max_iterations = max_iterations
        self.iteration = 0

    def try_with_bounded_reflection(self, task):
        '''Try with reflection, bounded iterations.'''
        print(f"Task: {task}")

        for i in range(self.max_iterations):
            self.iteration += 1
            success = hash(task + str(i)) % 3 == 0  # Success on iteration 3

            status = "✓" if success else "✗"
            print(f"  Iteration {i+1}: {status}")

            if success:
                print(f"Success after {i+1} iterations\n")
                return True

            if i < self.max_iterations - 1:
                lesson = f"Strategy {i+1} failed; try different approach"
                self.lessons.append(lesson)

        print(f"Exhausted {self.max_iterations} iterations; escalate to human\n")
        return False

# Example
print("Example 2 - Bounded Iterations:\n")
agent = BoundedReflectionAgent(max_iterations=3)
agent.try_with_bounded_reflection("complex_task")



# Example 3: Lesson Validation and Audit

from datetime import datetime, timedelta

class AuditingAgent:
    def __init__(self):
        self.lessons = []

    def add_lesson(self, condition, action, valid_days=30):
        '''Add lesson with expiry.'''
        lesson = {
            "condition": condition,
            "action": action,
            "created": datetime.now(),
            "valid_until": datetime.now() + timedelta(days=valid_days),
            "validation_count": 0,
            "success_count": 0
        }
        self.lessons.append(lesson)
        print(f"  Added lesson: {condition} -> {action}")

    def get_valid_lessons(self):
        '''Get non-expired lessons.'''
        now = datetime.now()
        return [l for l in self.lessons if now < l["valid_until"]]

    def validate_lesson(self, lesson, success):
        '''Test lesson and update validation stats.'''
        lesson["validation_count"] += 1
        if success:
            lesson["success_count"] += 1

        success_rate = lesson["success_count"] / max(lesson["validation_count"], 1)

        if success_rate < 0.5:
            self.lessons.remove(lesson)
            print(f"  Removed lesson: success rate {success_rate:.0%} too low")
        else:
            print(f"  Validated: {lesson['action']} ({success_rate:.0%} success rate)")

# Example
print("\nExample 3 - Lesson Validation:\n")
agent = AuditingAgent()
agent.add_lesson("tool_timeout", "use_cache", valid_days=30)

lesson = agent.lessons[0]
agent.validate_lesson(lesson, success=True)
agent.validate_lesson(lesson, success=True)
agent.validate_lesson(lesson, success=False)
agent.validate_lesson(lesson, success=True)



# ======================================================================
# ## Key Takeaways
# **Reflection Pattern:**
# 1. Execute task
# 2. Detect failure
# ======================================================================
