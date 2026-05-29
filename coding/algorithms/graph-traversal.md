# Graph Traversal

## TL;DR
Graph traversal visits all reachable nodes from a source. **BFS** (queue) finds shortest paths in unweighted graphs. **DFS** (stack/recursion) explores deeply, useful for cycle detection, topological sort, and connected components. **Dijkstra** (min-heap) finds shortest paths in weighted graphs. **Topological sort** (Kahn's algorithm or DFS) orders tasks respecting dependencies. All run in O(V+E).

## Core Concepts

**Graph representations:**

| Representation | Space | Edge lookup | Add edge | When to use |
|---|---|---|---|---|
| Adjacency list | O(V+E) | O(degree) | O(1) | Sparse graphs (most problems) |
| Adjacency matrix | O(V²) | O(1) | O(1) | Dense graphs, quick connectivity check |
| Edge list | O(E) | O(E) | O(1) | Algorithms that process edges (Kruskal) |

**Algorithm selection guide:**

| Goal | Algorithm | Time | Space |
|---|---|---|---|
| Shortest path (unweighted) | BFS | O(V+E) | O(V) |
| Shortest path (weighted, non-neg) | Dijkstra | O((V+E) log V) | O(V) |
| Shortest path (negative weights) | Bellman-Ford | O(VE) | O(V) |
| Shortest all pairs | Floyd-Warshall | O(V³) | O(V²) |
| Detect cycle (directed) | DFS + coloring | O(V+E) | O(V) |
| Topological order | Kahn's (BFS) | O(V+E) | O(V) |
| Connected components | DFS/Union-Find | O(V+E) | O(V) |
| Minimum spanning tree | Prim's or Kruskal's | O(E log V) | O(V) |

**BFS vs DFS:**
- BFS: level-by-level, guaranteed shortest path (unweighted), queue
- DFS: depth-first, detects cycles, topological sort, smaller memory for deep graphs, stack/recursion

## Implementations

```python
from typing import List, Dict, Set, Optional, Tuple
from collections import deque, defaultdict
import heapq


# --- BFS: shortest path in unweighted graph ---
def bfs_shortest_path(
    graph: Dict[int, List[int]], start: int, end: int
) -> int:
    """
    Return shortest path length from start to end.
    -1 if unreachable. O(V+E).
    """
    if start == end:
        return 0
    visited: Set[int] = {start}
    queue: deque[Tuple[int, int]] = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor == end:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1   # unreachable


# --- DFS: iterative (pre-order) ---
def dfs_iterative(
    graph: Dict[int, List[int]], start: int
) -> List[int]:
    """Iterative DFS using explicit stack. O(V+E)."""
    visited: Set[int] = set()
    stack = [start]
    order = []
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


# --- Cycle detection in directed graph (DFS with 3-color) ---
def has_cycle_directed(graph: Dict[int, List[int]], n: int) -> bool:
    """
    0=unvisited, 1=in-progress, 2=done.
    Back edge (1→1) indicates cycle.
    O(V+E).
    """
    color = [0] * n

    def dfs(node: int) -> bool:
        color[node] = 1   # mark in-progress
        for neighbor in graph.get(node, []):
            if color[neighbor] == 1:
                return True    # back edge → cycle
            if color[neighbor] == 0 and dfs(neighbor):
                return True
        color[node] = 2   # mark done
        return False

    return any(dfs(i) for i in range(n) if color[i] == 0)


# --- Dijkstra: shortest path in weighted graph ---
def dijkstra(
    graph: Dict[int, List[Tuple[int, int]]], source: int, n: int
) -> List[float]:
    """
    graph[u] = [(v, weight), ...]. Returns dist[] from source.
    O((V+E) log V) with min-heap.
    """
    dist = [float("inf")] * n
    dist[source] = 0
    heap: List[Tuple[float, int]] = [(0.0, source)]  # (distance, node)
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue    # stale entry in heap, skip
        for v, w in graph.get(u, []):
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    return dist


# --- Topological Sort: Kahn's Algorithm (BFS-based) ---
def topological_sort_kahn(
    n: int, edges: List[Tuple[int, int]]
) -> List[int]:
    """
    LC 210 Course Schedule II.
    Return topological order, or [] if cycle exists.
    O(V+E).
    """
    in_degree = [0] * n
    graph: Dict[int, List[int]] = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue: deque[int] = deque(i for i in range(n) if in_degree[i] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == n else []   # [] → cycle detected


# --- BFS word ladder (LC 127) ---
def ladder_length(begin_word: str, end_word: str, word_list: List[str]) -> int:
    """
    Shortest transformation sequence length.
    BFS on implicit graph: edges between words differing by 1 char.
    O(N * L^2) where N=word count, L=word length.
    """
    word_set = set(word_list)
    if end_word not in word_set:
        return 0
    queue: deque[Tuple[str, int]] = deque([(begin_word, 1)])
    visited: Set[str] = {begin_word}
    while queue:
        word, steps = queue.popleft()
        for i in range(len(word)):
            for ch in "abcdefghijklmnopqrstuvwxyz":
                next_word = word[:i] + ch + word[i+1:]
                if next_word == end_word:
                    return steps + 1
                if next_word in word_set and next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, steps + 1))
    return 0
```

## Key Problems

| Problem | Algorithm | Time | Space |
|---|---|---|---|
| LC 127 Word Ladder | BFS implicit graph | O(N * L²) | O(N * L) |
| LC 743 Network Delay Time | Dijkstra | O((V+E) log V) | O(V+E) |
| LC 787 Cheapest Flights K Stops | Modified Dijkstra/Bellman-Ford | O(K * E) | O(V) |
| LC 207 Course Schedule | Topological sort / cycle detect | O(V+E) | O(V+E) |
| LC 210 Course Schedule II | Kahn's topo sort | O(V+E) | O(V+E) |
| LC 200 Number of Islands | DFS/BFS connected components | O(m*n) | O(m*n) |
| LC 417 Pacific Atlantic Water Flow | Reverse BFS from edges | O(m*n) | O(m*n) |

## Common Mistakes / Gotchas
- **Forgetting visited set in BFS:** causes infinite loops in cyclic graphs; add to visited when enqueuing, not when dequeuing
- **Stale entries in Dijkstra:** heap may contain outdated distances; check `if d > dist[u]: continue` to skip
- **Directed vs undirected:** add both u→v and v→u for undirected graphs; cycle detection differs between directed (3-color) and undirected (parent tracking)
- **Topological sort cycle detection:** if `len(order) != n` after Kahn's, a cycle exists
- **In-degree initialization:** count all edges, not just unique neighbors

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "BFS vs DFS?" | BFS: shortest path (unweighted), level-by-level, queue. DFS: deep exploration, cycle detection, topo sort, stack/recursion. BFS uses more memory for wide graphs; DFS for deep. |
| "When Dijkstra vs BFS?" | BFS for unweighted (all edge weights equal). Dijkstra for non-negative weighted edges. Bellman-Ford if negative weights. |
| "How does Kahn's algorithm detect cycles?" | If a cycle exists, nodes in the cycle never reach in-degree 0 → never enter queue → not included in output → `len(order) < n`. |
| "Dijkstra complexity?" | O((V+E) log V) with binary min-heap. The log V factor comes from heap operations (each edge relaxation is a heap push). |

## Practice Resources
- LeetCode: 127, 200, 207, 210, 417, 743, 787, 1091
- Master BFS + Dijkstra + Kahn's — they cover ~80% of graph interview questions

## Related Topics
- [Trees](../data-structures/trees.md) — trees are acyclic connected graphs; tree traversal is graph traversal without visited set
- [Graphs](../data-structures/graphs.md) — graph representations and properties
- [Union-Find](../data-structures/union-find.md) — alternative for connected components
- [Dynamic Programming](dynamic-programming.md) — DP on DAGs (topological order defines computation order)
