"""
Auto-generated from 41-medical-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Medical Agents
# Objectives: Core patterns, implementation, optimization
# ======================================================================

class DiagnosisHelper:
    def __init__(self):
        self.known_symptoms = {'fever': ['flu', 'covid', 'cold']}
    def suggest(self, symptom):
        return self.known_symptoms.get(symptom, [])

helper = DiagnosisHelper()
print(f'Possible diagnoses: {helper.suggest("fever")}')


class CriticalMonitor:
    def flag_critical(self, vitals):
        alerts = []
        if vitals['hr'] > 120:
            alerts.append('Tachycardia')
        if vitals['temp'] > 39:
            alerts.append('High fever')
        return alerts

monitor = CriticalMonitor()
alerts = monitor.flag_critical({'hr': 130, 'temp': 38.5})
print(f'Alerts: {alerts}')


class TreatmentRecommender:
    def recommend(self, diagnosis):
        guidelines = {'flu': ['rest', 'fluids'], 'covid': ['isolate', 'monitor']}
        return guidelines.get(diagnosis, [])

rec = TreatmentRecommender()
plan = rec.recommend('flu')
print(f'Treatment: {plan}')

