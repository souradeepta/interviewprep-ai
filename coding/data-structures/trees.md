# Trees

Hierarchical node-based structures where each node has at most one parent and zero or more children. Binary trees are the dominant interview form. Search in a balanced BST is O(log n); worst-case (degenerate) is O(n). DFS and BFS together solve almost every tree problem.

## Core Patterns

| Pattern | When to use | Time | Space |
|---|---|---|---|
| DFS pre-order | Serialize, copy, print parent before children | O(n) | O(h) |
| DFS in-order | Sorted traversal of BST | O(n) | O(h) |
| DFS post-order | Delete, evaluate expressions, return child info | O(n) | O(h) |
| BFS level-order | Level grouping, shortest path in tree, zigzag | O(n) | O(w) max width |
| Path sum | Root-to-leaf sums, backtracking on path | O(n) | O(h) |
| Lowest Common Ancestor | Find meeting point of two paths | O(n) | O(h) |
| BST validation | Range-based invariant propagation | O(n) | O(h) |
| Diameter / max-depth | Post-order: return height, track global max | O(n) | O(h) |

h = tree height (O(log n) balanced, O(n) worst).

## Python Implementations

```python
from __future__ import annotations
from collections import deque
from typing import Optional


class TreeNode:
    """Binary tree node."""

    def __init__(
        self,
        val: int = 0,
        left: Optional[TreeNode] = None,
        right: Optional[TreeNode] = None,
    ) -> None:
        self.val = val
        self.left = left
        self.right = right


# ── DFS Traversals ─────────────────────────────────────────────────────────

def inorder_recursive(root: Optional[TreeNode]) -> list[int]:
    """In-order traversal (left → root → right) — recursive.
    Time: O(n)  Space: O(h) call stack.
    """
    result: list[int] = []

    def dfs(node: Optional[TreeNode]) -> None:
        if not node:
            return
        dfs(node.left)
        result.append(node.val)
        dfs(node.right)

    dfs(root)
    return result


def inorder_iterative(root: Optional[TreeNode]) -> list[int]:
    """In-order traversal — iterative with explicit stack.
    Time: O(n)  Space: O(h)
    """
    result: list[int] = []
    stack: list[TreeNode] = []
    curr = root
    while curr or stack:
        while curr:               # go as far left as possible
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()        # visit node
        result.append(curr.val)
        curr = curr.right         # move to right subtree
    return result


# ── BFS ────────────────────────────────────────────────────────────────────

def level_order(root: Optional[TreeNode]) -> list[list[int]]:
    """BFS level-order traversal.  Returns list of levels.
    Time: O(n)  Space: O(w) where w is max width.
    """
    if not root:
        return []
    result: list[list[int]] = []
    queue: deque[TreeNode] = deque([root])
    while queue:
        level_size = len(queue)
        level: list[int] = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result


# ── Classic Tree Problems ──────────────────────────────────────────────────

def max_depth(root: Optional[TreeNode]) -> int:
    """Maximum depth (number of nodes along longest root-to-leaf path).
    Time: O(n)  Space: O(h)
    """
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """Validate BST using propagated min/max bounds.
    Time: O(n)  Space: O(h)
    """
    def check(
        node: Optional[TreeNode], lo: float, hi: float
    ) -> bool:
        if not node:
            return True
        if not (lo < node.val < hi):
            return False
        return (
            check(node.left, lo, node.val)
            and check(node.right, node.val, hi)
        )

    return check(root, float('-inf'), float('inf'))


def lowest_common_ancestor(
    root: Optional[TreeNode],
    p: TreeNode,
    q: TreeNode,
) -> Optional[TreeNode]:
    """LCA for a general binary tree (not necessarily BST).

    If the current node is p or q, return it.  If both subtrees
    return a non-None value, current node is the LCA.
    Time: O(n)  Space: O(h)
    """
    if not root or root is p or root is q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right:
        return root    # p and q are in different subtrees
    return left or right


def has_path_sum(root: Optional[TreeNode], target_sum: int) -> bool:
    """Return True if any root-to-leaf path sums to target_sum.
    Time: O(n)  Space: O(h)
    """
    if not root:
        return False
    if not root.left and not root.right:      # leaf
        return root.val == target_sum
    remaining = target_sum - root.val
    return (
        has_path_sum(root.left, remaining)
        or has_path_sum(root.right, remaining)
    )
```

## Complexity Summary

| Operation | Balanced BST | Unbalanced BST | Notes |
|---|---|---|---|
| Search | O(log n) | O(n) | Height-dependent |
| Insert | O(log n) | O(n) | Height-dependent |
| Delete | O(log n) | O(n) | Height-dependent |
| In-order traversal | O(n) | O(n) | Always linear |
| Level-order (BFS) | O(n) | O(n) | Always linear |
| Height (max depth) | O(n) | O(n) | Must visit all nodes |
| Space (call stack) | O(log n) | O(n) | Recursion depth = height |

## Interview Recognition Template

- Parent-child hierarchy, recursive sub-structure → tree problem, DFS.
- Need sorted order from BST → in-order traversal.
- Group by level, shortest path, zigzag → BFS.
- "Depth", "height", "diameter", "balanced" → post-order DFS returning height.
- "Path sum", "root-to-leaf" → DFS backtracking with running sum.
- "Two nodes' ancestor" → LCA (post-order: return node if found, merge at split).
- "Is it a valid BST?" → propagate (min, max) bounds downward.

## Worked Examples

### 104. Maximum Depth of Binary Tree

**Problem:** Find the maximum depth of a binary tree.

```python
def maxDepth(root: Optional[TreeNode]) -> int:
    """Post-order: depth = 1 + max(left_depth, right_depth).
    Time: O(n)  Space: O(h)
    """
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

### 102. Binary Tree Level Order Traversal

**Problem:** Return all node values grouped by level.

```python
from collections import deque as _deque

def levelOrder(root: Optional[TreeNode]) -> list[list[int]]:
    """BFS: drain one level per outer iteration.
    Time: O(n)  Space: O(w) where w = max width.
    """
    if not root:
        return []
    result: list[list[int]] = []
    q: _deque[TreeNode] = _deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        result.append(level)
    return result
```

### 236. Lowest Common Ancestor of a Binary Tree

**Problem:** Given a binary tree and two nodes p and q, find their LCA.

```python
def lowestCommonAncestor(
    root: Optional[TreeNode], p: TreeNode, q: TreeNode
) -> Optional[TreeNode]:
    """Post-order DFS.  Node is LCA when both subtrees return non-None.
    Time: O(n)  Space: O(h)
    """
    if not root or root is p or root is q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right:
        return root
    return left or right
```

### 98. Validate Binary Search Tree

**Problem:** Determine if a binary tree is a valid BST.

```python
def isValidBST(root: Optional[TreeNode]) -> bool:
    """Propagate (lo, hi) bounds; every node must satisfy lo < val < hi.
    Time: O(n)  Space: O(h)
    """
    def validate(node: Optional[TreeNode], lo: float, hi: float) -> bool:
        if not node:
            return True
        if not (lo < node.val < hi):
            return False
        return (
            validate(node.left, lo, node.val)
            and validate(node.right, node.val, hi)
        )

    return validate(root, float('-inf'), float('inf'))
```

### 112. Path Sum

**Problem:** Does any root-to-leaf path sum to targetSum?

```python
def hasPathSum(root: Optional[TreeNode], targetSum: int) -> bool:
    """Subtract node value as we descend; check at leaf.
    Time: O(n)  Space: O(h)
    """
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == targetSum
    rem = targetSum - root.val
    return hasPathSum(root.left, rem) or hasPathSum(root.right, rem)
```

## Related Topics

- [Graphs](graphs.md) — [Stacks and Queues](stacks-queues.md) — [Tries](tries.md)
