# Linked Lists

Sequential node-based structure where each node holds a value and a pointer to the next node. O(1) insert at a known pointer, O(n) access and search. Mastery of fast/slow pointers, in-place reversal, and cycle detection unlocks the majority of linked-list interview problems.

## Core Patterns

| Pattern | When to use | Time | Space |
|---|---|---|---|
| Fast/slow pointers | Detect cycle, find middle, kth from end | O(n) | O(1) |
| Reverse in-place | Reverse full list or sub-list | O(n) | O(1) |
| Merge sorted lists | Combine two sorted linked lists | O(m+n) | O(1) |
| Dummy head node | Simplify edge cases when head may change | O(n) | O(1) |
| Two-pointer distance | Find kth node from end | O(n) | O(1) |

## Python Implementations

```python
from __future__ import annotations
from typing import Optional


class ListNode:
    """Singly linked list node."""

    def __init__(self, val: int = 0, next: Optional[ListNode] = None) -> None:
        self.val = val
        self.next = next


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """Reverse a singly linked list in-place.

    Uses the prev/curr/next triple-pointer pattern.
    Time: O(n)  Space: O(1)
    """
    prev: Optional[ListNode] = None
    curr = head
    while curr:
        nxt = curr.next   # save next before overwriting
        curr.next = prev  # reverse the pointer
        prev = curr       # advance prev
        curr = nxt        # advance curr
    return prev           # prev is the new head


def has_cycle(head: Optional[ListNode]) -> bool:
    """Detect a cycle using Floyd's two-pointer algorithm.

    slow moves 1 step, fast moves 2 steps.  If there is a cycle
    they will eventually meet; if not, fast reaches None first.
    Time: O(n)  Space: O(1)
    """
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next          # type: ignore[assignment]
        fast = fast.next.next     # type: ignore[union-attr]
        if slow is fast:
            return True
    return False


def find_middle(head: Optional[ListNode]) -> Optional[ListNode]:
    """Return the middle node (second middle for even-length lists).

    Time: O(n)  Space: O(1)
    """
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next       # type: ignore[assignment]
        fast = fast.next.next  # type: ignore[union-attr]
    return slow


def merge_sorted_lists(
    l1: Optional[ListNode], l2: Optional[ListNode]
) -> Optional[ListNode]:
    """Merge two sorted singly linked lists into one sorted list.

    Uses a dummy head to avoid special-casing the first node.
    Time: O(m+n)  Space: O(1)
    """
    dummy = ListNode(0)
    tail = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next  # type: ignore[assignment]
    tail.next = l1 if l1 else l2  # attach remaining
    return dummy.next


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """Remove the n-th node from the end in one pass.

    Lead the fast pointer n+1 steps ahead so that when fast
    reaches None, slow.next is the node to remove.
    Time: O(L)  Space: O(1)
    """
    dummy = ListNode(0, head)
    slow: ListNode = dummy
    fast: ListNode = dummy
    for _ in range(n + 1):          # advance fast n+1 steps
        fast = fast.next            # type: ignore[assignment]
    while fast:
        slow = slow.next            # type: ignore[assignment]
        fast = fast.next            # type: ignore[assignment]
    slow.next = slow.next.next      # type: ignore[union-attr]
    return dummy.next
```

## Complexity Summary

| Operation | Time | Space | Notes |
|---|---|---|---|
| Access by index | O(n) | O(1) | Must traverse from head |
| Insert at known pointer | O(1) | O(1) | Update next pointers only |
| Delete at known pointer | O(1) | O(1) | Need predecessor pointer |
| Search | O(n) | O(1) | Linear scan |
| Reverse | O(n) | O(1) | Three-pointer in-place |
| Detect cycle | O(n) | O(1) | Floyd's algorithm |
| Find middle | O(n) | O(1) | Fast/slow pointers |

## Interview Recognition Template

- Two pointers on a list + cycle/meeting point → fast/slow pointers (Floyd's).
- "Reverse" + "in-place" → prev/curr/next triple pattern.
- Result head may change (delete head, reverse) → use dummy head node.
- Find kth from end, remove nth from end → two-pointer distance gap of k.
- "Merge" + "sorted" → dummy head + compare-and-advance.

## Worked Examples

### 206. Reverse Linked List

**Problem:** Reverse a singly linked list.

```python
def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:
    """Iterative reversal with prev/curr/next pattern.
    Time: O(n)  Space: O(1)
    """
    prev: Optional[ListNode] = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
```

### 141. Linked List Cycle

**Problem:** Detect whether a linked list has a cycle.

```python
def hasCycle(head: Optional[ListNode]) -> bool:
    """Floyd's cycle detection: slow 1 step, fast 2 steps.
    If they meet, there is a cycle.
    Time: O(n)  Space: O(1)
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next          # type: ignore[assignment]
        fast = fast.next.next     # type: ignore[union-attr]
        if slow is fast:
            return True
    return False
```

### 876. Middle of the Linked List

**Problem:** Return the middle node of a linked list.

```python
def middleNode(head: Optional[ListNode]) -> Optional[ListNode]:
    """Slow pointer reaches middle when fast pointer reaches end.
    Time: O(n)  Space: O(1)
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next       # type: ignore[assignment]
        fast = fast.next.next  # type: ignore[union-attr]
    return slow
```

### 21. Merge Two Sorted Lists

**Problem:** Merge two sorted linked lists and return the sorted merged list.

```python
def mergeTwoLists(
    list1: Optional[ListNode], list2: Optional[ListNode]
) -> Optional[ListNode]:
    """Dummy head avoids special-casing an empty prefix.
    Time: O(m+n)  Space: O(1)
    """
    dummy = ListNode(0)
    cur = dummy
    while list1 and list2:
        if list1.val <= list2.val:
            cur.next = list1
            list1 = list1.next
        else:
            cur.next = list2
            list2 = list2.next
        cur = cur.next   # type: ignore[assignment]
    cur.next = list1 or list2
    return dummy.next
```

### 19. Remove Nth Node From End of List

**Problem:** Remove the n-th node from the end in one pass.

```python
def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """Gap trick: fast is n+1 ahead of slow so slow lands on predecessor.
    Time: O(L)  Space: O(1)
    """
    dummy = ListNode(0, head)
    slow: ListNode = dummy
    fast: ListNode = dummy
    for _ in range(n + 1):
        fast = fast.next     # type: ignore[assignment]
    while fast:
        slow = slow.next     # type: ignore[assignment]
        fast = fast.next     # type: ignore[assignment]
    slow.next = slow.next.next  # type: ignore[union-attr]
    return dummy.next
```

## Related Topics

- [Arrays and Strings](arrays-strings.md) — [Trees](trees.md) — [Two-Pointer Technique](../algorithms/two-pointers.md)
