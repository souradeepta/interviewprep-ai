# Backtracking

## TL;DR
Backtracking is systematic enumeration with pruning. Build a candidate solution incrementally; when a partial candidate can't lead to a valid solution, backtrack (undo the last choice) and try the next option. Used for subsets, permutations, combinations, and constraint satisfaction (N-queens, Sudoku). Time complexity is exponential in the worst case, but pruning makes it practical.

## Core Concepts

**Core template:**
```
backtrack(state):
    if is_complete(state):
        record result
        return
    for each valid choice:
        make choice
        backtrack(updated state)
        undo choice  ← this is the "backtrack" step
```

**Key variants:**

| Problem Type | Start index | Reuse elements | Skip duplicates |
|---|---|---|---|
| Subsets | `i+1` (move forward) | No | Sort + skip if `i>start and arr[i]==arr[i-1]` |
| Permutations | 0 (all positions) | No (use `visited` set) | Sort + skip if same value as previous unused |
| Combination Sum | `i` (allow reuse) | Yes (unbounded) | No |
| Combination Sum II | `i+1` | No | Sort + skip duplicates |
| N-Queens | Next row | — | Check column/diagonal sets |

**Pruning:** add early termination when partial state already violates constraints. Critical for performance — without pruning, backtracking degenerates to brute force.

**Complexity:**
- Subsets of n elements: O(2^n) solutions × O(n) to copy each = O(n * 2^n)
- Permutations of n elements: O(n!) solutions × O(n) copy = O(n * n!)
- Pruning reduces constants significantly

## Implementations

```python
from typing import List


# --- Subsets (LC 78) ---
def subsets(nums: List[int]) -> List[List[int]]:
    """All subsets (power set). No duplicates in input here. O(n * 2^n)."""
    result: List[List[int]] = []

    def backtrack(start: int, path: List[int]) -> None:
        result.append(path[:])   # record current subset (even empty)
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()           # undo choice

    backtrack(0, [])
    return result


# --- Permutations (LC 46) ---
def permutations(nums: List[int]) -> List[List[int]]:
    """All permutations of distinct integers. O(n * n!)."""
    result: List[List[int]] = []
    visited = [False] * len(nums)

    def backtrack(path: List[int]) -> None:
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if visited[i]:
                continue
            visited[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            visited[i] = False

    backtrack([])
    return result


# --- Combination Sum (LC 39, reuse allowed) ---
def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    Find all combos summing to target. Can reuse elements.
    Prune: skip branch if remaining < 0.
    """
    result: List[List[int]] = []
    candidates.sort()   # enables early termination

    def backtrack(start: int, path: List[int], remaining: int) -> None:
        if remaining == 0:
            result.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break           # pruning: sorted → no further candidates can work
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])  # i (not i+1) → allow reuse
            path.pop()

    backtrack(0, [], target)
    return result


# --- N-Queens (LC 51) ---
def solve_n_queens(n: int) -> List[List[str]]:
    """
    Place n queens on n×n board so no two attack each other.
    State: which columns and diagonals are occupied.
    O(n!) upper bound; much less with pruning.
    """
    result: List[List[str]] = []
    cols: set[int] = set()
    diag1: set[int] = set()   # row - col (same for \\ diagonal)
    diag2: set[int] = set()   # row + col (same for / diagonal)

    def backtrack(row: int, board: List[List[str]]) -> None:
        if row == n:
            result.append(["".join(r) for r in board])
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue       # pruning: this placement is attacked
            # Place queen
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            board[row][col] = "Q"
            backtrack(row + 1, board)
            # Undo placement
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
            board[row][col] = "."

    board = [["." for _ in range(n)] for _ in range(n)]
    backtrack(0, board)
    return result


# --- General backtracking template ---
def backtrack_template(candidates: List[int], target: int) -> List[List[int]]:
    """Template — customize is_complete, is_valid, and how to generate choices."""
    result: List[List[int]] = []

    def backtrack(start: int, path: List[int]) -> None:
        if is_complete(path, target):   # replace with actual completion check
            result.append(path[:])
            return
        for i in range(start, len(candidates)):
            if not is_valid(i, path):   # replace with actual validity check
                continue
            path.append(candidates[i])
            backtrack(i + 1, path)      # i+1 for no-reuse; i for reuse allowed
            path.pop()

    backtrack(0, [])
    return result

def is_complete(path: list, target: int) -> bool:
    return sum(path) == target

def is_valid(i: int, path: list) -> bool:
    return True
```

## Key Problems

| Problem | Pattern | Time | Space |
|---|---|---|---|
| LC 78 Subsets | No-duplicate subsets | O(n * 2^n) | O(n) |
| LC 90 Subsets II | Subsets with duplicates → sort + skip | O(n * 2^n) | O(n) |
| LC 46 Permutations | visited array | O(n * n!) | O(n) |
| LC 47 Permutations II | Permutations with duplicates | O(n * n!) | O(n) |
| LC 39 Combination Sum | Reuse allowed, prune by sum | Varies | O(target/min) |
| LC 40 Combination Sum II | No reuse, skip duplicates | Varies | O(n) |
| LC 51 N-Queens | Row-by-row, col/diag sets | O(n!) | O(n) |
| LC 37 Sudoku Solver | Cell-by-cell, row/col/box sets | O(9^81) pruned | O(1) |

## Common Mistakes / Gotchas
- **Not copying path:** `result.append(path)` appends a reference — path will be empty at end; use `path[:]` or `list(path)`
- **Missing undo step:** every `path.append(x)` must have a paired `path.pop()` (or undo for sets/boards)
- **Start index for reuse:** `backtrack(i, ...)` allows reusing `candidates[i]`; `backtrack(i+1, ...)` does not
- **Duplicate handling:** sort input first; skip `i > start and nums[i] == nums[i-1]` — this skips only at the same level, not different recursive depths
- **Pruning placement:** prune before the recursive call, not inside — early return saves recursive overhead

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "When backtracking vs DP?" | Backtracking: enumerate all valid solutions or can't express as overlapping subproblems. DP: count/optimize, subproblems overlap. If you need all results → backtracking; if you need min/max/count → likely DP. |
| "How to handle duplicates?" | Sort the input. At each level, skip element i if `i > start and nums[i] == nums[i-1]`. The `i > start` check is crucial — it only skips within the same recursive level. |
| "What makes backtracking exponential?" | At each position we have up to n choices, and we have n positions → n^n paths in worst case. Pruning reduces constants. Subsets is 2^n, permutations is n!. |
| "How to optimize N-Queens?" | Use sets for columns, diagonals (row-col, row+col) instead of scanning board → O(1) validity check per placement. Bit manipulation further reduces constant. |

## Practice Resources
- LeetCode: 78, 90, 46, 47, 39, 40, 51, 52, 37, 131
- Build intuition: do Subsets → Permutations → Combination Sum → N-Queens in order

## Related Topics
- [Dynamic Programming](dynamic-programming.md) — DP when subproblems overlap; backtracking when all solutions needed
- [Recursion](recursion.md) — backtracking is structured recursion with undo
- [Graph Traversal](graph-traversal.md) — DFS with backtracking for path finding
