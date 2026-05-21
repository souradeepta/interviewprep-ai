"""
Auto-generated from 40-customer-service-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Customer Service Agents
# Objectives: Core patterns, implementation, optimization
# ======================================================================

class IntentClassifier:
    def detect(self, message):
        if 'refund' in message.lower():
            return 'refund_request'
        elif 'how to' in message.lower():
            return 'how_to'
        return 'general'

classifier = IntentClassifier()
print(f'Intent: {classifier.detect("How do I get a refund?")}')


class KnowledgeBaseLookup:
    def __init__(self):
        self.kb = {'refund': 'Contact support for refund', 'billing': 'Check invoice'}
    def find_answer(self, intent):
        return self.kb.get(intent, 'Please contact support')

kb = KnowledgeBaseLookup()
print(f'Answer: {kb.find_answer("refund")}')


class ServiceAgent:
    def __init__(self):
        self.classifier = IntentClassifier()
        self.kb = KnowledgeBaseLookup()
    def respond(self, message):
        intent = self.classifier.detect(message)
        answer = self.kb.find_answer(intent)
        return {'response': answer, 'escalate': not answer or 'contact support' in answer}

agent = ServiceAgent()
result = agent.respond('I need help')
print(f'Response: {result}')

