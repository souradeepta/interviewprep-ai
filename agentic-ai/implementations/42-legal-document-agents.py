"""
Auto-generated from 42-legal-document-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Legal Document Agents
# Objectives: Core patterns, implementation, optimization
# ======================================================================

class ClauseExtractor:
    def extract(self, text):
        clauses = []
        if 'payment' in text.lower():
            clauses.append('Payment clause')
        if 'termination' in text.lower():
            clauses.append('Termination clause')
        return clauses

extractor = ClauseExtractor()
text = 'Payment terms: net 30. Termination: either party may terminate.'
print(f'Clauses: {extractor.extract(text)}')


class RiskAssessor:
    def assess(self, clause):
        risk = 0
        if 'unlimited liability' in clause.lower():
            risk = 0.9
        elif 'indemnity' in clause.lower():
            risk = 0.6
        return risk

assessor = RiskAssessor()
risk = assessor.assess('Unlimited liability clause')
print(f'Risk: {risk}')


class LegalAnalyzer:
    def analyze_contract(self, text):
        extractor = ClauseExtractor()
        assessor = RiskAssessor()
        clauses = extractor.extract(text)
        risks = [{'clause': c, 'risk': assessor.assess(c)} for c in clauses]
        return {'clauses': clauses, 'risks': [r for r in risks if r['risk'] > 0.5]}

analyzer = LegalAnalyzer()
result = analyzer.analyze_contract('Unlimited liability in contract')
print(f'Analysis: {result}')

