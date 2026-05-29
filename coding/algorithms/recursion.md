# Recursion

## TL;DR
Recursion solves a problem by defining it in terms of smaller versions of itself. Three ingredients: **base case** (stop condition), **recursive case** (reduce problem), **trust the recursion** (assume the recursive call returns the correct answer for the smaller input). Memoization converts naive exponential recursion to polynomial. Many recursive solutions convert to iterative using an explicit stack.

## Core Concepts

**The three laws of recursion:**
1. Must have a base case
2. Must move toward the base case (reduce problem size)
3. Must call itself with the reduced problem

**Recursion types:**

| Type | Shape | Example |
|---|---|---|
| Linear | One recursive call | factorial, linked list traversal |
| Binary | Two recursive calls | binary tree traversal, merge sort |
| Multiple | Many recursive calls | combinatorics, backtracking |
| Tail | Last operation is recursive call | tail-recursive factorial (optimizable) |
| Mutual | A calls B, B calls A | even/odd checkers |

**Memoization:** cache results of pure recursive calls. Converts O(2^n) to O(n) for Fibonacci. Use `@functools.lru_cache` or explicit dict.

**Recursive → Iterative:** replace call stack with explicit stack data structure. Pre-order DFS becomes: push root, pop and process, push children.

**Key insight:** "Trust the recursion" — when writing a recursive function, assume `f(smaller)` works correctly and build `f(n)` from it. Don't mentally trace through all calls.

**Complexity analysis:**
- Write recurrence: `T(n) = aT(n/b) + O(n^d)`
- Apply master theorem or draw recursion tree

## Implementations

```python
from typing import List, Optional
from functools import lru_cache
import sys

# Increase for deep recursion (use iterative for production)
sys.setrecursionlimit(10000)


# --- Fibonacci: naive vs memoized ---
def fib_naive(n: int) -> int:
    """O(2^n) time — exponential due to overlapping subproblems."""
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


@lru_cache(maxsize=None)
def fib_memo(n: int) -> int:
    """O(n) time, O(n) space — memoization eliminates redundant calls."""
    if n <= 1:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)


def fib_iterative(n: int) -> int:
    """O(n) time, O(1) space — DP bottom-up (no recursion overhead)."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# --- Fast power: Pow(x, n) (LC 50) ---
def my_pow(x: float, n: int) -> float:
    """
    O(log n) time. Divide and conquer: x^n = x^(n//2) * x^(n//2) [* x if odd].
    Trust recursion: assume pow(x, n//2) is correct.
    """
    if n == 0:
        return 1.0
    if n < 0:
        return 1.0 / my_pow(x, -n)
    half = my_pow(x, n // 2)
    if n % 2 == 0:
        return half * half
    else:
        return half * half * x


# --- Flatten Nested List (LC 341) ---
def flatten_nested(nested: list) -> List[int]:
    """Recursively flatten arbitrarily nested list. O(n) where n = total elements."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_nested(item))   # recurse on nested list
        else:
            result.append(item)
    return result


# --- Recursive DFS (tree) ---
class TreeNode:
    def __init__(self, val: int = 0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_recursive(root: Optional[TreeNode]) -> List[int]:
    """Recursive inorder traversal. O(n) time, O(h) space (h = height)."""
    if root is None:
        return []
    return inorder_recursive(root.left) + [root.val] + inorder_recursive(root.right)


# --- Recursive → Iterative (pre-order DFS with explicit stack) ---
def preorder_iterative(root: Optional[TreeNode]) -> List[int]:
    """
    Equivalent to recursive pre-order but uses explicit stack.
    Good for very deep trees to avoid stack overflow.
    """
    if root is None:
        return []
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)   # push right first (LIFO → left processed first)
        if node.left:
            stack.append(node.left)
    return result


# --- Helper function pattern (accumulator) ---
def path_sum(root: Optional[TreeNode], target: int) -> bool:
    """
    LC 112. Does root-to-leaf path with sum=target exist?
    Helper function carries accumulated sum.
    """
    def helper(node: Optional[TreeNode], remaining: int) -> bool:
        if node is None:
            return False
        if node.left is None and node.right is None:  # leaf node
            return remaining == node.val
        return (helper(node.left, remaining - node.val) or
                helper(node.right, remaining - node.val))

    return helper(root, target)


# --- Recursion with return value accumulation ---
def max_depth(root: Optional[TreeNode]) -> int:
    """
    Max depth of binary tree. Trust recursion: max_depth(left) gives correct depth.
    T(n) = 2T(n/2) + O(1) → O(n) by master theorem.
    """
    if root is None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

## Key Problems

| Problem | Recursion type | Time | Space |
|---|---|---|---|
| LC 509 Fibonacci | Binary (naive) / Linear (memo) | O(2^n) / O(n) | O(n) |
| LC 50 Pow(x, n) | Binary D&C | O(log n) | O(log n) |
| LC 341 Flatten Nested Iterator | Linear per level | O(n) | O(depth) |
| LC 104 Max Depth Binary Tree | Binary | O(n) | O(h) |
| LC 112 Path Sum | Binary | O(n) | O(h) |
| LC 21 Merge Two Sorted Lists | Linear | O(m+n) | O(m+n) |
| LC 24 Swap Nodes in Pairs | Linear | O(n) | O(n) |

## Common Mistakes / Gotchas
- **Missing base case:** infinite recursion → stack overflow; always define termination condition first
- **Not reducing problem size:** recursive call on same or larger input → infinite recursion; verify n decreases
- **Python recursion limit:** default is 1000 frames; deep trees or large n require `sys.setrecursionlimit` or iterative conversion
- **Memoization on mutable args:** `@lru_cache` only works on hashable arguments; convert lists to tuples
- **Concatenating lists in recursion:** `left + [val] + right` creates O(n²) total work for tree traversal; use `extend` or pass accumulator list

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "Recursive vs iterative?" | Recursive: cleaner code, easier to reason. Iterative: no stack overflow risk, better constant. Always prefer iterative for production when depth is large. |
| "How to memoize?" | Decorate with `@functools.lru_cache(maxsize=None)` for top-down. Or use a dict: `if n in cache: return cache[n]`. |
| "Recursion to iteration?" | Replace call stack with explicit stack data structure. Mimic how the call stack would unwind. Pre-order DFS is straightforward; in-order needs a pointer approach. |
| "Complexity of tree recursion?" | T(n) = 2T(n/2) + O(1) → O(n) by master theorem (a=2, b=2, d=0; log_2(2)=1 > 0). Most balanced tree recursions are O(n). |

## Practice Resources
- LeetCode: 509 (Fibonacci variants), 50 (Pow), 104, 112, 226 (Invert Tree), 341
- Classic exercises: Tower of Hanoi, generate all parentheses (LC 22), recursion to iteration

## Related Topics
- [Dynamic Programming](dynamic-programming.md) — DP is memoized recursion with overlapping subproblems
- [Backtracking](backtracking.md) — recursion with undo for enumeration
- [Divide & Conquer](divide-conquer.md) — recursion that splits into independent subproblems
- [Trees](../data-structures/trees.md) — most tree algorithms are recursive
