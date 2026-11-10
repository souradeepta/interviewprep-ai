"""Estimate serving capacity, monthly inference cost and an explicit budget decision."""
def capacity_plan(requests_per_second, capacity_per_worker_rps, *, target_utilization=.7, headroom=1.3):
    if requests_per_second<0 or capacity_per_worker_rps<=0 or not 0<target_utilization<=1 or headroom<1:
        raise ValueError("invalid capacity input")
    required=requests_per_second*headroom/(capacity_per_worker_rps*target_utilization)
    return {"workers":int(required) if required==int(required) else int(required)+1,
            "target_utilization":target_utilization,"headroom":headroom}

def monthly_cost(requests_per_day, cost_per_1000, *, days=30, fixed_cost=0):
    if min(requests_per_day,cost_per_1000,days,fixed_cost)<0: raise ValueError("cost inputs must be nonnegative")
    return requests_per_day*days/1000*cost_per_1000+fixed_cost

def budget_decision(projected_cost, budget, *, reserve_fraction=.1):
    if min(projected_cost,budget)<0 or not 0<=reserve_fraction<1: raise ValueError("invalid budget")
    limit=budget*(1-reserve_fraction)
    return {"decision":"within_reserve" if projected_cost<=limit else "review_capacity_or_cost",
            "spend_limit":limit,"projected_cost":projected_cost}

if __name__ == "__main__":
    plan=capacity_plan(100,25); cost=monthly_cost(100000,0.8,fixed_cost=500)
    print(plan,cost,budget_decision(cost,3000))
