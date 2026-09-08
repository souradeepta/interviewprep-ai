# Matrix and Grid Traversal

Grid problems combine coordinate validation with BFS, DFS, dynamic
programming, or in-place state updates. Write the direction vectors once and
centralize bounds checks so edge cases do not dominate the implementation.

```python
from collections import deque


def shortest_open_path(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start[0], start[1], 0)])
    seen = {start}
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while queue:
        row, col, distance = queue.popleft()
        if (row, col) == goal:
            return distance
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                    grid[nr][nc] == 0 and (nr, nc) not in seen):
                seen.add((nr, nc))
                queue.append((nr, nc, distance + 1))
    return -1
```

Use BFS for an unweighted shortest path, DFS for connected-component counts,
multi-source BFS for simultaneous spread, and DP when the path count or cost
depends on a prior state. For in-place marking, either mutate safely, use a
separate `seen` set, or encode temporary states only when the value domain
allows it.

Clarify diagonal movement, obstacles, whether revisiting is allowed, and
whether the grid is rectangular. Practice: LeetCode 48, 54, 62, 73, 130,
200, 286, 490, 542, 994, 1091.
