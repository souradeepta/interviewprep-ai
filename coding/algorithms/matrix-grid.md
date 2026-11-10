# Matrix and Grid Traversal

Grid problems combine coordinate validation with BFS, DFS, dynamic programming, or in-place state updates. Write the direction vectors once and centralize bounds checks so edge cases do not dominate the implementation. Treat a cell as a graph vertex and legal moves as edges; then choose the graph algorithm from the question’s objective.

## Recognition criteria

Use BFS for a shortest path when every move has equal cost; the first time a node is removed from the queue, its distance is minimal. Use DFS or BFS to count connected components or flood-fill a region. Use multi-source BFS when many starting cells spread at the same speed, as with simultaneous infection or fire. Use dynamic programming when paths only move in an acyclic direction, such as right/down, and the task asks for path counts or minimum cost. Use Dijkstra when moves have nonnegative but unequal weights. Confirm whether diagonal movement, wraparound, and revisiting are allowed.

Centralize shape and bounds assumptions. Many interview grids are rectangular and non-empty, but production inputs may be ragged or empty. If rows have different lengths, validate each neighbor against its own row. A coordinate is `(row, column)`; keeping that order consistent avoids transposition bugs. A separate `seen` set is clear and safe when input values cannot be mutated.

```python
from collections import deque


def shortest_open_path(grid, start, goal):
    if not grid or not grid[0]:
        return -1
    rows, cols = len(grid), len(grid[0])
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < rows and 0 <= sc < cols):
        return -1
    if not (0 <= gr < rows and 0 <= gc < cols):
        return -1
    if grid[sr][sc] != 0 or grid[gr][gc] != 0:
        return -1
    queue = deque([(sr, sc, 0)])
    seen = {(sr, sc)}
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while queue:
        row, col, distance = queue.popleft()
        if (row, col) == (gr, gc):
            return distance
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] == 0 and (nr, nc) not in seen):
                seen.add((nr, nc))
                queue.append((nr, nc, distance + 1))
    return -1
```

Mark nodes seen when enqueuing, not when removing them. Otherwise the same cell may enter the queue through several parents, multiplying work and complicating distance reasoning. For a rectangular grid with V = rows×columns cells and at most four edges per cell, BFS is O(V) time and O(V) space.

## Worked trace: shortest path

Consider `[[0,0,1],[1,0,0],[0,0,0]]`, start `(0,0)`, goal `(2,2)`. Initially the queue contains `(0,0,0)`. The only open unseen neighbor is `(0,1)`, so it receives distance 1. From `(0,1)`, BFS discovers `(1,1)` at distance 2. Then `(1,1)` discovers `(1,2)` and `(2,1)` at distance 3. The goal `(2,2)` is next reached at distance 4. Queue order explores all distance-d nodes before distance-(d+1) nodes, which is why the returned distance is shortest.

## Complete problem 1: number of islands

**Prompt:** Count 4-connected groups of land cells in a binary grid. **Approach:** Scan cells; when unvisited land is found, increment the component count and flood-fill its neighbors using DFS or BFS, marking each land cell once. **Example:** In `[[1,1,0],[0,1,0],[1,0,1]]`, the upper cluster and the two isolated bottom cells yield 3 islands. **Correctness:** Each flood-fill visits exactly the cells reachable from its seed; because all reachable land is marked, a later scan cannot count that component again. **Complexity:** O(rows×columns) time and O(rows×columns) worst-case memory. If input mutation is permitted, mark in place; otherwise use `seen`.

## Complete problem 2: distance to nearest zero

**Prompt:** For each cell in a binary matrix, return distance to the nearest zero using four-direction movement. **Approach:** Initialize a queue with every zero at distance 0, mark all of them visited, and run one BFS. The first discovered distance for each cell is its nearest-zero distance. This multi-source BFS avoids running a separate traversal from each cell. **Example:** For `[[0,1,1],[1,1,1]]`, the result is `[[0,1,2],[1,2,3]]`. **Correctness:** All zero sources expand in nondecreasing distance layers; the first source reaching a cell has minimum distance. **Complexity:** O(rows×columns) time and space. Define behavior when the grid contains no zero: reject it or return an explicit sentinel rather than leaving distances undefined.

## In-place updates and dynamic programming

In-place flood fill can save auxiliary storage, but it is safe only when the marker cannot be confused with an original value. If a grid stores arbitrary integers, reserve a separate visited structure or use a tagged representation. When one update affects a neighborhood, consider whether later reads should observe earlier writes; use a copy or two-phase update if transitions are simultaneous.

For right/down path counts, `dp[r][c] = dp[r-1][c] + dp[r][c-1]` for open cells, with an appropriate start condition. This is not a shortest-path BFS problem because the movement graph is acyclic and only the accumulated count matters. A one-row DP can reduce memory to O(columns). Obstacles reset the cell count to zero. For minimum cost, use `min` plus the cell cost; clarify whether the start cell cost is included.

## Common failure cases and diagnostics

- **Out-of-bounds neighbor:** Accessing a cell before validating both coordinates causes edge crashes. Test all four corners and a one-row grid; keep the bounds check adjacent to the access.
- **Marking too late:** Adding `seen` only after dequeue can enqueue duplicates. Mark at discovery and test a cell reachable from multiple parents.
- **Wrong connectivity:** Diagonal adjacency changes the component count and shortest paths. Encode the direction set explicitly and test a diagonal-only contact.
- **Assuming a nonempty rectangle:** Empty or ragged rows break `grid[0]` and fixed-width checks. Either validate and reject, or implement row-specific bounds.
- **BFS on weighted edges:** BFS minimizes number of moves, not total cost when weights differ. Use 0-1 BFS for weights in {0,1} or Dijkstra for nonnegative weights; test a short expensive path versus longer cheap path.
- **Incorrect simultaneous transition:** Updating cells in scan order can let a spread propagate multiple steps in one round. Use layer boundaries or a separate next-state buffer.
- **Recursive depth overflow:** Recursive DFS can exceed language stack limits on a large component. Use an explicit stack or queue for unbounded grids.

## Complexity and selection summary

For a grid with R rows and C columns, a constant-neighbor traversal usually costs O(RC) time and O(RC) worst-case memory. BFS is the default for unweighted shortest distances; DFS/BFS both find connected regions; multi-source BFS finds distance from the nearest source; topological DP solves one-direction acyclic path problems; Dijkstra handles nonnegative weighted movement. Output storage may itself be O(RC), so count it separately from auxiliary memory when comparing approaches.

## Practice list

Practice: LeetCode 48 (rotate image), 54 (spiral order), 62 (unique paths), 73 (set matrix zeroes), 130 (surrounded regions), 200 (number of islands), 286 (walls and gates), 490 (rolling maze), 542 (01 matrix), 994 (rotting oranges), and 1091 (shortest path in binary matrix). For each, write the movement set, visited invariant, and empty/unreachable behavior before implementation.
