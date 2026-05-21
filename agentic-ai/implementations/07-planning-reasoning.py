"""
Auto-generated from 07-planning-reasoning.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Planning Agents
# Objectives: Goal decomposition, hierarchical planning, replanning, contingencies
# ======================================================================

class PlanningAgent:
    def decompose(self, goal: str) -> list:
        mapping = {
            'book_flight': ['search_flights', 'check_budget', 'book', 'confirm'],
            'search_flights': ['get_dates', 'get_budget', 'query_api'],
            'book': ['select_flight', 'enter_payment', 'confirm_booking']
        }
        return mapping.get(goal, [goal])
    
    def get_dependencies(self) -> dict:
        return {
            'book': ['search_flights', 'check_budget'],
            'confirm_booking': ['select_flight', 'enter_payment'],
            'query_api': ['get_dates', 'get_budget']
        }
    
    def order_plan(self, goal: str) -> list:
        subgoals = self.decompose(goal)
        deps = self.get_dependencies()
        ordered, visited = [], set()
        
        def visit(g):
            if g in visited: return
            for d in deps.get(g, []):
                visit(d)
            visited.add(g)
            ordered.append(g)
        
        for g in subgoals:
            visit(g)
        return ordered
    
    def plan(self, goal: str) -> list:
        return self.order_plan(goal)
    
    def execute_with_replan(self, goal: str):
        plan = self.plan(goal)
        for step in plan:
            try:
                self.execute_step(step)
            except Exception as e:
                print(f'{step} failed: {e}, replanning...')
                plan = self.plan(goal)
        return 'done'
    
    def execute_step(self, step: str):
        print(f'Executing: {step}')
        return True

agent = PlanningAgent()
plan = agent.plan('book_flight')
print('Plan:', plan)

