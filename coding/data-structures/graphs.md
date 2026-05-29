# Graphs

A graph is a set of vertices (nodes) connected by edges. Edges can be directed or undirected, weighted or unweighted. Graphs generalize trees: a tree is a connected acyclic undirected graph. DFS and BFS together cover the vast majority of graph interview problems; topological sort and union-find handle scheduling and grouping problems.

## Core Patterns

| Pattern | When to use | Time | Space |
|---|---|---|---|
| Adjacency list build | Store graph from edge list | O(V+E) | O(V+E) |
| DFS (recursive) | Connectivity, cycle detection, flood fill | O(V+E) | O(V) visited + O(V) stack |
| DFS (iterative) | Same as recursive but avoids Python recursion limit | O(V+E) | O(V) |
| BFS shortest path | Unweighted shortest path, level exploration | O(V+E) | O(V) |
| Topological sort (Kahn's) | Dependency ordering, cycle detection in DAG | O(V+E) | O(V) |
| Connected components | Count or label disconnected parts | O(V+E) | O(V) |
| Bipartite check | 2-color graph → detect odd cycle | O(V+E) | O(V) |

## Python Implementations

```python
from collections import defaultdict, deque
from typing import Optional


def build_adj_list(
    n: int, edges: list[tuple[int, int]], directed: bool = False
) -> dict[int, list[int]]:
    """Build adjacency list from an edge list.

    Args:
        n: number of vertices (0-indexed).
        edges: list of (u, v) pairs.
        directed: if True, add u→v only; else add both directions.
    Returns:
        dict mapping each vertex to its list of neighbors.
    """
    graph: dict[int, list[int]] = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        if not directed:
            graph[v].append(u)
    return dict(graph)


def dfs(
    graph: dict[int, list[int]], start: int
) -> list[int]:
    """Iterative DFS.  Returns visited nodes in discovery order.
    Time: O(V+E)  Space: O(V)
    """
    visited: set[int] = set()
    order: list[int] = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                stack.append(neighbor)
    return order


def bfs(
    graph: dict[int, list[int]], start: int
) -> dict[int, int]:
    """BFS from start.  Returns dict of {node: shortest_distance}.
    Time: O(V+E)  Space: O(V)
    """
    dist: dict[int, int] = {start: 0}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in dist:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    return dist


def has_cycle_undirected(
    graph: dict[int, list[int]], n: int
) -> bool:
    """Detect cycle in undirected graph via DFS with parent tracking.
    Time: O(V+E)  Space: O(V)
    """
    visited: set[int] = set()

    def dfs_cycle(node: int, parent: int) -> bool:
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs_cycle(neighbor, node):
                    return True
            elif neighbor != parent:  # back edge → cycle
                return True
        return False

    for v in range(n):
        if v not in visited:
            if dfs_cycle(v, -1):
                return True
    return False


def topological_sort(
    n: int, prerequisites: list[tuple[int, int]]
) -> list[int]:
    """Kahn's algorithm: BFS-based topological sort.

    Returns a valid ordering, or [] if a cycle exists (no valid ordering).
    Time: O(V+E)  Space: O(V)
    """
    in_degree = [0] * n
    graph: dict[int, list[int]] = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue: deque[int] = deque(v for v in range(n) if in_degree[v] == 0)
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return order if len(order) == n else []  # [] means cycle detected


def num_connected_components(
    n: int, edges: list[tuple[int, int]]
) -> int:
    """Count connected components via DFS.
    Time: O(V+E)  Space: O(V+E)
    """
    graph = build_adj_list(n, edges, directed=False)
    visited: set[int] = set()
    count = 0
    for v in range(n):
        if v not in visited:
            # DFS to mark entire component
            stack = [v]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                stack.extend(graph.get(node, []))
            count += 1
    return count
```

## Complexity Summary

| Operation | Time | Space | Notes |
|---|---|---|---|
| Build adjacency list | O(V+E) | O(V+E) | Edge list → adj list |
| DFS | O(V+E) | O(V) | Visited set + stack/recursion |
| BFS | O(V+E) | O(V) | Distance dict + queue |
| Topological sort (Kahn's) | O(V+E) | O(V) | Returns [] if cycle |
| Cycle detection (undirected) | O(V+E) | O(V) | Parent tracking in DFS |
| Connected components | O(V+E) | O(V+E) | One DFS pass per component |
| Bipartite check | O(V+E) | O(V) | 2-coloring via BFS/DFS |

## Interview Recognition Template

- Relationships, dependencies, networks → graph problem; build adjacency list first.
- "Shortest path" on unweighted graph → BFS (not DFS).
- "Can finish all courses?", "task ordering" → topological sort; check for cycle.
- "Number of islands", "connected components", "flood fill" → DFS/BFS + visited set.
- Cycle in undirected graph → DFS with parent; cycle in directed graph → DFS with 3-color.
- Bipartite / 2-coloring → BFS, assign alternating colors, detect conflict.
- Avoid re-visiting: always maintain a `visited` set before enqueueing.

## Worked Examples

### 200. Number of Islands

**Problem:** Count the number of distinct islands (connected groups of '1') in a 2-D grid.

```python
def numIslands(grid: list[list[str]]) -> int:
    """DFS flood-fill: mark visited cells as '0'.
    Time: O(m*n)  Space: O(m*n) recursion stack worst case.
    """
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r: int, c: int) -> None:
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        grid[r][c] = '0'   # mark visited in-place
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1
    return count
```

### 207. Course Schedule

**Problem:** Given n courses and a list of prerequisites, determine if you can finish all courses.

```python
from collections import defaultdict as _dd, deque as _dq

def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    """Kahn's topological sort.  If all nodes processed → no cycle.
    Time: O(V+E)  Space: O(V+E)
    """
    in_degree = [0] * numCourses
    graph: dict[int, list[int]] = _dd(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue: _dq[int] = _dq(v for v in range(numCourses) if in_degree[v] == 0)
    processed = 0
    while queue:
        node = queue.popleft()
        processed += 1
        for nxt in graph[node]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
    return processed == numCourses
```

### 133. Clone Graph

**Problem:** Deep clone an undirected connected graph.

```python
from typing import Dict

class _Node:
    def __init__(self, val: int = 0, neighbors: Optional[list] = None):
        self.val = val
        self.neighbors: list[_Node] = neighbors or []

def cloneGraph(node: Optional[_Node]) -> Optional[_Node]:
    """BFS clone.  Map original → clone to handle cycles.
    Time: O(V+E)  Space: O(V)
    """
    if not node:
        return None
    cloned: Dict[_Node, _Node] = {}
    queue: deque[_Node] = deque([node])
    cloned[node] = _Node(node.val)
    while queue:
        curr = queue.popleft()
        for neighbor in curr.neighbors:
            if neighbor not in cloned:
                cloned[neighbor] = _Node(neighbor.val)
                queue.append(neighbor)
            cloned[curr].neighbors.append(cloned[neighbor])
    return cloned[node]
```

### 323. Number of Connected Components in an Undirected Graph

**Problem:** Given n nodes and a list of edges, return the number of connected components.

```python
def countComponents(n: int, edges: list[list[int]]) -> int:
    """BFS per unvisited node.
    Time: O(V+E)  Space: O(V+E)
    """
    graph: dict[int, list[int]] = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    visited: set[int] = set()
    count = 0
    for v in range(n):
        if v not in visited:
            queue: deque[int] = deque([v])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                queue.extend(graph[node])
            count += 1
    return count
```

### 785. Is Graph Bipartite?

**Problem:** Determine whether a graph can be colored with two colors such that no adjacent nodes share a color.

```python
def isBipartite(graph_adj: list[list[int]]) -> bool:
    """BFS 2-coloring.  Conflict on same-color adjacent nodes → not bipartite.
    Time: O(V+E)  Space: O(V)
    """
    n = len(graph_adj)
    color = [-1] * n  # -1 = uncolored
    for start in range(n):
        if color[start] != -1:
            continue
        queue: deque[int] = deque([start])
        color[start] = 0
        while queue:
            node = queue.popleft()
            for neighbor in graph_adj[node]:
                if color[neighbor] == -1:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
    return True
```

## Related Topics

- [Trees](trees.md) — [Union-Find](union-find.md) — [Heaps](heaps.md)
