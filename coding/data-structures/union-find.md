# Union-Find (Disjoint Set Union)

Union-Find (also called Disjoint Set Union, DSU) is a data structure that tracks a partition of n elements into disjoint sets. With path compression and union by rank, both `find` and `union` run in O(α(n)) ≈ O(1) amortized — nearly constant for any practical n. It is the canonical tool for dynamic connectivity problems.

## Core Patterns

| Pattern | When to use | Time | Space |
|---|---|---|---|
| Connected components | Count or label disconnected groups | O(E * α(n)) | O(V) |
| Cycle detection (undirected) | Edge creates cycle iff both endpoints in same set | O(E * α(n)) | O(V) |
| Dynamic connectivity | Add edges; query whether two nodes are connected | O(α(n)) per op | O(V) |
| Grouping / merging accounts | Merge sets based on shared elements | O(n * α(n)) | O(n) |
| Minimum spanning tree (Kruskal) | Add edges in weight order, skip same-component edges | O(E log E) | O(V) |

## Python Implementations

```python
class UnionFind:
    """Disjoint Set Union with path compression and union by rank.

    Path compression flattens the tree during find so future finds
    are faster.  Union by rank attaches smaller trees under larger ones
    to keep height logarithmic.  Together they give O(α(n)) per op.
    """

    def __init__(self, n: int) -> None:
        """Initialize n singletons.  Each node is its own parent."""
        self.parent: list[int] = list(range(n))
        self.rank: list[int] = [0] * n
        self._num_components: int = n

    def find(self, x: int) -> int:
        """Return the representative (root) of x's set.

        Path compression: on the way back up, point every node
        directly to the root.  O(α(n)) amortized.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compress
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Merge the sets containing x and y.

        Returns True if x and y were in different sets (a new merge
        happened), False if they were already connected.
        Union by rank attaches smaller-rank tree under larger-rank root.
        O(α(n)) amortized.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False   # already in the same set
        # Attach smaller-rank tree under larger-rank root
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self._num_components -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Return True if x and y are in the same set.  O(α(n))"""
        return self.find(x) == self.find(y)

    @property
    def num_components(self) -> int:
        """Current number of disjoint sets."""
        return self._num_components


# ── String-keyed variant ───────────────────────────────────────────────────

class UnionFindStr:
    """Union-Find for arbitrary hashable keys (strings, tuples, etc.)."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        """Find root with path compression.  Auto-initialises x."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> bool:
        """Merge sets; return True if a new merge occurred."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def connected(self, x: str, y: str) -> bool:
        return self.find(x) == self.find(y)
```

## Complexity Summary

| Operation | Time (with optimizations) | Space | Notes |
|---|---|---|---|
| Initialize n elements | O(n) | O(n) | parent + rank arrays |
| find (path compression) | O(α(n)) amortized | O(1) | Nearly O(1) for all n |
| union (by rank) | O(α(n)) amortized | O(1) | Merges two sets |
| connected | O(α(n)) amortized | O(1) | Two finds |
| E union/find operations | O(E * α(n)) | O(V) | α(n) < 5 for all practical n |

α(n) is the inverse Ackermann function — effectively constant (≤ 5) for any n you'll encounter.

## Interview Recognition Template

- "Number of connected components" that changes dynamically → Union-Find.
- "Detect cycle in undirected graph" without DFS → Union-Find (union returns False on same-set edge).
- "Merge accounts / groups" by shared element → Union-Find with string keys.
- "Are X and Y connected after merges?" → Union-Find connected query.
- BFS/DFS vs Union-Find: BFS/DFS is one-time traversal; Union-Find supports incremental online merges.

## Worked Examples

### 547. Number of Provinces

**Problem:** Given an n×n adjacency matrix, find the number of directly or indirectly connected groups.

```python
def findCircleNum(isConnected: list[list[int]]) -> int:
    """Union-Find: union all directly connected cities.
    Time: O(n^2 * α(n))  Space: O(n)
    """
    n = len(isConnected)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if isConnected[i][j] == 1:
                uf.union(i, j)
    return uf.num_components
```

### 684. Redundant Connection

**Problem:** Find the edge that, when removed, leaves the remaining graph a tree (acyclic + connected).

```python
def findRedundantConnection(edges: list[list[int]]) -> list[int]:
    """Process edges in order; the first edge whose endpoints are already
    connected forms the redundant edge.
    Time: O(E * α(V))  Space: O(V)
    """
    n = len(edges)
    uf = UnionFind(n + 1)   # 1-indexed nodes
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]   # already connected: this edge is redundant
    return []
```

### 721. Accounts Merge

**Problem:** Merge accounts that share at least one email address.

```python
from collections import defaultdict as _dd

def accountsMerge(accounts: list[list[str]]) -> list[list[str]]:
    """Union-Find on email strings.  Map each email to its account owner.
    Time: O(N * m * α(N))  Space: O(N * m) where m = emails per account.
    """
    uf = UnionFindStr()
    email_to_name: dict[str, str] = {}

    for account in accounts:
        name = account[0]
        first_email = account[1]
        for email in account[1:]:
            uf.union(first_email, email)
            email_to_name[email] = name

    # Group emails by their root representative
    groups: dict[str, list[str]] = _dd(list)
    for email in email_to_name:
        root = uf.find(email)
        groups[root].append(email)

    result: list[list[str]] = []
    for root, emails in groups.items():
        name = email_to_name[root]
        result.append([name] + sorted(emails))
    return result
```

## Related Topics

- [Graphs](graphs.md) — [Hash Tables](hash-tables.md) — [Trees](trees.md)
