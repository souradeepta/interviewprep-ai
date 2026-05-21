"""
Auto-generated from 44-logistics-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Logistics Agents
# Objectives: Core patterns, implementation, optimization
# ======================================================================

class RouteOptimizer:
    def plan_route(self, orders, vehicle_capacity):
        routes = []
        current_load = 0
        current_route = []
        for order in orders:
            if current_load + order['size'] <= vehicle_capacity:
                current_route.append(order)
                current_load += order['size']
            else:
                routes.append(current_route)
                current_route = [order]
                current_load = order['size']
        routes.append(current_route)
        return routes

opt = RouteOptimizer()
orders = [{'id': 1, 'size': 10}, {'id': 2, 'size': 15}]
routes = opt.plan_route(orders, 30)
print(f'Routes: {len(routes)}')


class ConstraintChecker:
    def check_feasibility(self, route, constraints):
        if 'time_window' in constraints:
            return route[0].get('time') <= constraints['time_window']
        if 'weight_limit' in constraints:
            total = sum(item['weight'] for item in route)
            return total <= constraints['weight_limit']
        return True

checker = ConstraintChecker()
route = [{'time': 9, 'weight': 5}]
feasible = checker.check_feasibility(route, {'time_window': 12})
print(f'Feasible: {feasible}')


class LogisticsAgent:
    def plan_and_validate(self, orders):
        optimizer = RouteOptimizer()
        routes = optimizer.plan_route(orders, 50)
        checker = ConstraintChecker()
        valid_routes = [r for r in routes if checker.check_feasibility(r, {})]
        return valid_routes

agent = LogisticsAgent()
print('Logistics planning complete')

