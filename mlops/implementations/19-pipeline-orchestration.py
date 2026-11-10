"""Small deterministic DAG runner with idempotent run keys and backfill bounds."""
from collections import defaultdict,deque

def topological_order(tasks, dependencies):
    tasks=list(tasks); incoming={t:set(dependencies.get(t,())) for t in tasks}
    if any(d not in incoming for ds in incoming.values() for d in ds): raise ValueError("dependency is not a task")
    ready=deque(t for t in tasks if not incoming[t]); order=[]
    while ready:
        task=ready.popleft(); order.append(task)
        for child in tasks:
            if task in incoming[child]:
                incoming[child].remove(task)
                if not incoming[child]: ready.append(child)
    if len(order)!=len(tasks): raise ValueError("pipeline contains a cycle")
    return order

def run_partition(partition,tasks,dependencies,completed=None):
    """Return newly completed (task, partition) keys; already completed keys are idempotent."""
    done=set(completed or ()); order=topological_order(tasks,dependencies); emitted=[]
    for task in order:
        key=(task,partition)
        if key not in done:
            if any((dep,partition) not in done for dep in dependencies.get(task,())):
                raise RuntimeError("dependency did not complete")
            done.add(key); emitted.append(key)
    return emitted

def backfill_partitions(start,end,max_partitions):
    if end<start: raise ValueError("end precedes start")
    parts=list(range(start,end+1))
    if len(parts)>max_partitions: raise ValueError("backfill exceeds bounded partition count")
    return parts

if __name__ == "__main__":
    tasks=["extract","validate","train","publish"]; deps={"validate":["extract"],"train":["validate"],"publish":["train"]}
    print(run_partition(42,tasks,deps),backfill_partitions(40,42,3))
