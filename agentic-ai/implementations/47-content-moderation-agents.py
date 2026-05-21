"""
Auto-generated from 47-content-moderation-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Content Moderation Agents
# Objectives: Core patterns, implementation, optimization
# ======================================================================

class ToxicityDetector:
    def score(self, text):
        toxicity = 0
        toxic_words = ['hate', 'kill', 'attack']
        for word in toxic_words:
            if word in text.lower():
                toxicity += 0.3
        return min(toxicity, 1.0)

detector = ToxicityDetector()
print(f'Toxicity: {detector.score("I hate this")}')


class ContentModerator:
    def __init__(self, threshold=0.5):
        self.detector = ToxicityDetector()
        self.threshold = threshold
    def classify(self, text):
        score = self.detector.score(text)
        if score > self.threshold:
            return 'FLAG'
        return 'ALLOW'

moderator = ContentModerator()
print(f'Classification: {moderator.classify("I hate this")}')


class ModerationSystem:
    def process(self, content):
        moderator = ContentModerator()
        result = moderator.classify(content)
        if result == 'FLAG':
            return {'action': 'review', 'priority': 'high'}
        return {'action': 'publish'}

system = ModerationSystem()
print(f'Action: {system.process("Great content")}')

