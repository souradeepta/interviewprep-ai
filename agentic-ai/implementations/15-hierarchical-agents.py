"""
Auto-generated from 15-hierarchical-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Hierarchical Agents
# Objectives: Manager-delegate-aggregate pattern, parallel execution, fault tolerance, multi-level hierarchies
# ======================================================================

import asyncio
from typing import List, Dict, Optional
import time

# Level 1: Basic Hierarchical Agent with Simple Decomposition

class SimpleWorker:
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
    
    async def solve(self, task: str) -> Dict:
        await asyncio.sleep(0.3)  # Simulate work
        return {
            'worker': self.name,
            'specialty': self.specialty,
            'task': task[:30] + '...' if len(task) > 30 else task,
            'result': f'Analysis from {self.specialty} perspective'
        }

class SimpleManager:
    def __init__(self, workers: List[SimpleWorker]):
        self.workers = {w.specialty: w for w in workers}
    
    async def run(self, goal: str):
        print(f'Manager: Received goal: {goal}')
        
        # Delegate to all workers in parallel
        tasks = [w.solve(goal) for w in self.workers.values()]
        results = await asyncio.gather(*tasks)
        
        print(f'\nManager: Aggregating {len(results)} results')
        for r in results:
            print(f"  - {r['worker']} ({r['specialty']}): {r['result']}")
        
        return results

# Test Level 1
async def test_level1():
    workers = [
        SimpleWorker('Alice', 'data'),
        SimpleWorker('Bob', 'engineering'),
        SimpleWorker('Carol', 'strategy')
    ]
    manager = SimpleManager(workers)
    await manager.run('Design a recommendation system')

await test_level1()


# Level 2: Advanced Manager with Fault Tolerance and Routing

class RobustWorker:
    def __init__(self, name: str, specialty: str, failure_rate: float = 0.0):
        self.name = name
        self.specialty = specialty
        self.failure_rate = failure_rate
    
    async def solve(self, task: str, timeout: int = 2) -> Dict:
        import random
        
        # Simulate random failures
        if random.random() < self.failure_rate:
            raise Exception(f'{self.name} encountered error')
        
        await asyncio.sleep(0.5)
        return {
            'worker': self.name,
            'specialty': self.specialty,
            'status': 'success',
            'result': f'Solution from {self.specialty}'
        }

class RobustManager:
    def __init__(self, workers: List[RobustWorker], max_retries: int = 2):
        self.workers = {w.specialty: w for w in workers}
        self.max_retries = max_retries
        self.task_routing = {
            'analyze': 'data',
            'implement': 'engineering',
            'strategy': 'strategy',
            'design': 'engineering'
        }
    
    def decompose_task(self, goal: str) -> List[Dict]:
        """Map task keywords to worker specialties."""
        subtasks = []
        keywords = goal.lower().split()
        
        specialties = set()
        for kw in keywords:
            if kw in self.task_routing:
                specialties.add(self.task_routing[kw])
        
        if not specialties:
            specialties = set(self.workers.keys())
        
        return [{'specialty': s, 'task': goal} for s in specialties]
    
    async def call_worker_with_retry(self, specialty: str, task: str) -> Optional[Dict]:
        """Call worker with retry logic."""
        for attempt in range(self.max_retries):
            try:
                worker = self.workers[specialty]
                result = await asyncio.wait_for(worker.solve(task), timeout=2)
                return result
            except asyncio.TimeoutError:
                print(f'  ⏱️  {specialty} timeout (attempt {attempt + 1})')
            except Exception as e:
                print(f'  ❌ {specialty} error: {str(e)[:40]}')
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(0.2)
        
        return None
    
    async def run(self, goal: str):
        print(f'Manager: Decomposing "{goal}"')
        subtasks = self.decompose_task(goal)
        print(f'  → {len(subtasks)} subtasks identified')
        
        # Dispatch in parallel
        tasks = [
            self.call_worker_with_retry(s['specialty'], s['task'])
            for s in subtasks
        ]
        results = await asyncio.gather(*tasks)
        
        # Aggregate
        successful = [r for r in results if r is not None]
        print(f'Manager: Aggregated {len(successful)}/{len(subtasks)} results')
        
        return successful

# Test Level 2
async def test_level2():
    workers = [
        RobustWorker('Alice', 'data', failure_rate=0.2),
        RobustWorker('Bob', 'engineering', failure_rate=0.1),
        RobustWorker('Carol', 'strategy', failure_rate=0.0)
    ]
    manager = RobustManager(workers, max_retries=2)
    await manager.run('Implement and design a new analytics strategy')

await test_level2()


# Example 1: Multi-Level Hierarchy (CEO → Managers → Workers)

class DepartmentManager:
    def __init__(self, dept_name: str, workers: List[RobustWorker]):
        self.dept_name = dept_name
        self.workers = {w.specialty: w for w in workers}
    
    async def handle_dept_task(self, task: str) -> Dict:
        await asyncio.sleep(0.3)
        return {
            'department': self.dept_name,
            'task': task[:25],
            'output': f'Report from {self.dept_name}'
        }

class CEOAgent:
    def __init__(self):
        self.eng_dept = DepartmentManager('Engineering', [])
        self.data_dept = DepartmentManager('Data', [])
        self.product_dept = DepartmentManager('Product', [])
    
    async def execute_strategy(self, goal: str):
        print(f'CEO: Strategy = "{goal}"')
        
        # Delegate to all departments in parallel
        tasks = [
            self.eng_dept.handle_dept_task(goal),
            self.data_dept.handle_dept_task(goal),
            self.product_dept.handle_dept_task(goal)
        ]
        results = await asyncio.gather(*tasks)
        
        print(f'CEO: Received reports from {len(results)} departments')
        for r in results:
            print(f"  - {r['department']}: {r['output']}")

# Test Example 1
async def test_example1():
    ceo = CEOAgent()
    await ceo.execute_strategy('Build next-generation platform')

await test_example1()


# Example 2: Load-Balanced Worker Assignment

class LoadBalancedManager:
    def __init__(self, workers: List[RobustWorker]):
        self.workers = workers
        self.worker_load = {w.name: 0 for w in workers}
    
    def get_least_loaded_worker(self, specialty: str) -> RobustWorker:
        """Route task to worker with lowest load."""
        candidates = [w for w in self.workers if w.specialty == specialty]
        if not candidates:
            return self.workers[0]
        return min(candidates, key=lambda w: self.worker_load[w.name])
    
    async def assign_task(self, worker: RobustWorker, task: str):
        self.worker_load[worker.name] += 1
        print(f'  → Assigned to {worker.name} (load: {self.worker_load[worker.name]})')
        
        result = await worker.solve(task)
        self.worker_load[worker.name] -= 1
        return result
    
    async def dispatch_subtasks(self, goal: str):
        """Dispatch similar subtasks to least-loaded workers."""
        print(f'Manager: Load-balancing dispatch for "{goal}"')
        
        # Create multiple similar subtasks
        subtasks = [f'{goal} (variant {i})' for i in range(3)]
        
        tasks = []
        for subtask in subtasks:
            worker = self.get_least_loaded_worker('data')
            tasks.append(self.assign_task(worker, subtask))
        
        results = await asyncio.gather(*tasks)
        print(f'Completed {len(results)} tasks')
        return results

# Test Example 2
async def test_example2():
    workers = [
        RobustWorker('Alice', 'data'),
        RobustWorker('Bob', 'data'),
        RobustWorker('Carol', 'data')
    ]
    manager = LoadBalancedManager(workers)
    await manager.dispatch_subtasks('Analyze user cohort')

await test_example2()


# Example 3: Specialized Workers with Domain Expertise

class SpecializedAnalyst(RobustWorker):
    def __init__(self):
        super().__init__('DataScientist', 'data')
    
    async def analyze(self, dataset: str) -> Dict:
        await asyncio.sleep(0.4)
        return {
            'worker': self.name,
            'analysis': f'Statistical insights from {dataset}',
            'confidence': 0.95
        }

class SpecializedEngineer(RobustWorker):
    def __init__(self):
        super().__init__('SoftwareEngineer', 'engineering')
    
    async def design(self, system: str) -> Dict:
        await asyncio.sleep(0.3)
        return {
            'worker': self.name,
            'architecture': f'Scalable design for {system}',
            'components': 5
        }

class SpecializedStrategist(RobustWorker):
    def __init__(self):
        super().__init__('ProductManager', 'strategy')
    
    async def strategize(self, problem: str) -> Dict:
        await asyncio.sleep(0.35)
        return {
            'worker': self.name,
            'strategy': f'Go-to-market plan for {problem}',
            'phases': 3
        }

class SpecializedManager:
    def __init__(self):
        self.specialists = {
            'data': SpecializedAnalyst(),
            'engineering': SpecializedEngineer(),
            'strategy': SpecializedStrategist()
        }
    
    async def coordinate(self, goal: str):
        print(f'Manager: Coordinating specialists for "{goal}"')
        
        tasks = [
            self.specialists['data'].analyze(goal),
            self.specialists['engineering'].design(goal),
            self.specialists['strategy'].strategize(goal)
        ]
        
        results = await asyncio.gather(*tasks)
        
        print(f'\nIntegrated Results:')
        for r in results:
            print(f"  {r['worker']}:")
            for key, val in r.items():
                if key != 'worker':
                    print(f"    - {key}: {val}")

# Test Example 3
async def test_example3():
    manager = SpecializedManager()
    await manager.coordinate('Real-time recommendation engine')

await test_example3()


# ======================================================================
# ## Key Takeaways
# **Manager-Delegate Pattern:**
# 1. Manager decomposes goal into subtasks
# 2. Identifies required worker specialties
# ======================================================================
