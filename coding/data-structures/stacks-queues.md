# Stacks and Queues

Stack (LIFO) and queue (FIFO) are the workhorses of traversal-order problems. A stack handles nested/sequential structure where you need the most-recently-seen element; a queue handles breadth-first or FIFO ordering. Both support O(1) push/pop/enqueue/dequeue. The monotonic stack and deque extensions solve a large class of harder problems.

## Core Patterns

| Pattern | When to use | Time | Space |
|---|---|---|---|
| Stack: matching brackets | Balanced parentheses, nested structure | O(n) | O(n) |
| Monotonic stack | Next greater/smaller element, histogram area | O(n) amortized | O(n) |
| Min/max stack | Track min or max alongside normal stack ops | O(1) push/pop | O(n) |
| Queue via two stacks | Implement queue with only stack primitives | O(1) amortized | O(n) |
| Deque sliding window | Sliding window maximum or minimum | O(n) | O(k) |
| BFS queue | Level-order traversal, shortest path unweighted | O(V+E) | O(V) |

## Python Implementations

```python
from collections import deque
from typing import Optional


class Stack:
    """LIFO stack backed by a Python list.  All operations O(1)."""

    def __init__(self) -> None:
        self._data: list[int] = []

    def push(self, val: int) -> None:
        self._data.append(val)

    def pop(self) -> int:
        if not self._data:
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self) -> int:
        if not self._data:
            raise IndexError("peek at empty stack")
        return self._data[-1]

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def __len__(self) -> int:
        return len(self._data)


class Queue:
    """FIFO queue backed by collections.deque.  O(1) enqueue and dequeue."""

    def __init__(self) -> None:
        self._data: deque[int] = deque()

    def enqueue(self, val: int) -> None:
        self._data.append(val)

    def dequeue(self) -> int:
        if not self._data:
            raise IndexError("dequeue from empty queue")
        return self._data.popleft()

    def front(self) -> int:
        if not self._data:
            raise IndexError("front of empty queue")
        return self._data[0]

    def is_empty(self) -> bool:
        return len(self._data) == 0


class MinStack:
    """Stack that returns the current minimum in O(1).

    A parallel 'min_stack' tracks the running minimum for each
    corresponding level of the main stack.
    """

    def __init__(self) -> None:
        self._stack: list[int] = []
        self._min_stack: list[int] = []

    def push(self, val: int) -> None:
        self._stack.append(val)
        # Track minimum: push current min if stack was non-empty
        if self._min_stack:
            self._min_stack.append(min(val, self._min_stack[-1]))
        else:
            self._min_stack.append(val)

    def pop(self) -> None:
        self._stack.pop()
        self._min_stack.pop()

    def top(self) -> int:
        return self._stack[-1]

    def get_min(self) -> int:
        return self._min_stack[-1]


def monotonic_next_greater(nums: list[int]) -> list[int]:
    """For each element, find the next greater element to the right.

    Uses a monotonic decreasing stack (indices). When a larger element
    is seen, all smaller elements on the stack have found their answer.
    Time: O(n)  Space: O(n)
    """
    n = len(nums)
    result = [-1] * n
    stack: list[int] = []  # stores indices
    for i, val in enumerate(nums):
        # Pop all indices whose value is smaller than current
        while stack and nums[stack[-1]] < val:
            idx = stack.pop()
            result[idx] = val
        stack.append(i)
    return result


def sliding_window_maximum(nums: list[int], k: int) -> list[int]:
    """Return max value in every sliding window of size k.

    Deque stores indices in decreasing order of nums value.
    Front of deque is always the index of the window maximum.
    Time: O(n)  Space: O(k)
    """
    dq: deque[int] = deque()   # indices, decreasing by value
    result: list[int] = []
    for i, val in enumerate(nums):
        # Remove indices outside current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        # Maintain monotonic decreasing deque
        while dq and nums[dq[-1]] < val:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
```

## Complexity Summary

| Operation | Stack (list) | Queue (deque) | MinStack | Monotonic Stack |
|---|---|---|---|---|
| Push / Enqueue | O(1) | O(1) | O(1) | O(1) amortized |
| Pop / Dequeue | O(1) | O(1) | O(1) | O(1) amortized |
| Peek / Front | O(1) | O(1) | O(1) | O(1) |
| Get minimum | N/A | N/A | O(1) | N/A |
| Space | O(n) | O(n) | O(n) | O(n) |

## Interview Recognition Template

- Nested or balanced structure (brackets, HTML) → stack.
- Level-by-level traversal, shortest path on unweighted graph → queue (BFS).
- "Next greater element", "histogram largest rectangle", "temperatures" → monotonic stack.
- Sliding window maximum or minimum → deque.
- Need O(1) get_min alongside push/pop → MinStack (parallel min stack).
- Implement queue with stacks → two-stack trick (inbox/outbox).

## Worked Examples

### 20. Valid Parentheses

**Problem:** Given a string of brackets, return true if it is valid (every opener has a matching closer in correct order).

```python
def isValid(s: str) -> bool:
    """Push openers; on closer, check top matches.
    Time: O(n)  Space: O(n)
    """
    stack: list[str] = []
    match = {')': '(', '}': '{', ']': '['}
    for ch in s:
        if ch in match:
            if not stack or stack[-1] != match[ch]:
                return False
            stack.pop()
        else:
            stack.append(ch)
    return len(stack) == 0
```

### 496. Next Greater Element I

**Problem:** For each element in nums1, find the next greater element in nums2.

```python
def nextGreaterElement(nums1: list[int], nums2: list[int]) -> list[int]:
    """Build next-greater map for nums2 using monotonic stack.
    Time: O(m+n)  Space: O(n)
    """
    next_greater: dict[int, int] = {}
    stack: list[int] = []
    for val in nums2:
        while stack and stack[-1] < val:
            next_greater[stack.pop()] = val
        stack.append(val)
    # Remaining elements in stack have no next greater
    return [next_greater.get(x, -1) for x in nums1]
```

### 155. Min Stack

**Problem:** Design a stack that supports push, pop, top, and getMin in O(1).

```python
class MinStackSolution:
    """Parallel min_stack tracks minimum at each stack level.
    All operations O(1).
    """

    def __init__(self) -> None:
        self.stack: list[int] = []
        self.min_stack: list[int] = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        cur_min = val if not self.min_stack else min(val, self.min_stack[-1])
        self.min_stack.append(cur_min)

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```

### 239. Sliding Window Maximum

**Problem:** Return the maximum of each sliding window of size k.

```python
from collections import deque as _deque

def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    """Monotonic decreasing deque of indices.
    Front always holds the index of the current window max.
    Time: O(n)  Space: O(k)
    """
    dq: _deque[int] = _deque()
    result: list[int] = []
    for i, val in enumerate(nums):
        # Evict indices outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        # Maintain decreasing invariant
        while dq and nums[dq[-1]] < val:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
```

## Related Topics

- [Trees](trees.md) — [Graphs](graphs.md) — [Arrays and Strings](arrays-strings.md)
