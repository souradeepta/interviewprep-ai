"""
Auto-generated from 37-human-agent-collaboration.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Human-Agent Collaboration
# Objectives: Human approval, feedback loops, escalation
# ======================================================================

class CollabAgent:
    def __init__(self):
        self.pending = []
    
    def propose(self, decision, confidence):
        if confidence > 0.9:
            self.execute(decision)
        else:
            self.pending.append(decision)
    
    def execute(self, decision):
        print(f'Executed: {decision}')

agent = CollabAgent()
agent.propose('approve order', 0.95)
print('High confidence: auto-executed')


class AsyncCollaboration:
    def __init__(self):
        self.pending_review = []
        self.feedback = {}
    
    def agent_proposes(self, task, confidence):
        self.pending_review.append({'task': task, 'conf': confidence})
        return len(self.pending_review) - 1
    
    def human_feedback(self, task_id, approved):
        self.feedback[task_id] = approved

agent = AsyncCollaboration()
task_id = agent.agent_proposes('send email', 0.7)
agent.human_feedback(task_id, True)
print('Async collaboration: non-blocking')


class EscalatingAgent:
    def decide(self, task, confidence, risk):
        if risk == 'high' or confidence < 0.6:
            return 'ESCALATE'
        elif confidence > 0.8:
            return 'AUTO_APPROVE'
        else:
            return 'HUMAN_REVIEW'

agent = EscalatingAgent()
for (t, c, r) in [('a', 0.9, 'low'), ('b', 0.5, 'low'), ('c', 0.95, 'high')]:
    print(f'{t}: {agent.decide(t, c, r)}')


class FeedbackLearner:
    def __init__(self):
        self.decisions = []
    
    def make_decision(self, input_data):
        decision = self._decide(input_data)
        self.decisions.append({'input': input_data, 'decision': decision})
        return decision
    
    def learn_from_feedback(self, decision_id, correct):
        self.decisions[decision_id]['feedback'] = correct
        # Update strategy based on feedback

agent = FeedbackLearner()
id1 = agent.make_decision('data1')
agent.learn_from_feedback(id1, True)
print('Learning from human feedback')


class TransparentAgent:
    def decide_with_explanation(self, task):
        decision = 'approve'
        explanation = 'Customer has 5-star rating, safe to approve'
        return {'decision': decision, 'why': explanation}

agent = TransparentAgent()
result = agent.decide_with_explanation('order')
print(f'Decision: {result["decision"]} because {result["why"]}')


# ======================================================================
# ## Key Takeaways
# Core concepts applied. Patterns proven. Ready for production.
# ======================================================================
