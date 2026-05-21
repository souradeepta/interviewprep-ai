"""
Auto-generated from 08-react-reasoning-acting.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # ReAct: Reasoning + Acting
# Objectives: Thought-action-observation cycles, tool integration, loop control
# ======================================================================

class ReactAgent:
    def __init__(self, max_steps=10):
        self.max_steps = max_steps
        self.trace = []
    
    def react(self, question: str):
        for step in range(self.max_steps):
            # Thought
            thought = self.generate_thought(question, self.trace)
            print(f'Step {step+1}. Thought: {thought}')
            
            if 'Final Answer' in thought:
                return thought
            
            # Action
            action = self.parse_action(thought)
            obs = self.execute_action(action)
            print(f'  Action: {action}')
            print(f'  Observation: {obs}')
            
            self.trace.append({'thought': thought, 'action': action, 'obs': obs})
        
        return 'Max steps reached'
    
    def generate_thought(self, question, trace):
        # Simulate: would call LLM in real scenario
        if len(trace) == 0:
            return f'Thought: I need to gather info about {question}'
        elif len(trace) == 1:
            return f'Thought: Based on results, I can now {"search more" if len(trace) < 2 else "decide"}'
        else:
            return f'Final Answer: Synthesized from {len(trace)} steps'
    
    def parse_action(self, thought):
        if 'search' in thought.lower():
            return 'search_web(query)'
        elif 'info' in thought.lower():
            return 'lookup_fact(topic)'
        else:
            return 'final_response()'
    
    def execute_action(self, action):
        return f'Result from {action}'

agent = ReactAgent()
result = agent.react('What is ReAct?')
print(f'\nFinal: {result}')

