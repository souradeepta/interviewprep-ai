# Greedy Algorithms

## TL;DR
Greedy algorithms make the locally optimal choice at each step, hoping it leads to a global optimum. They work when the **greedy choice property** holds: a local optimum at each step leads to a global optimum. Prove correctness with an **exchange argument**: assume an optimal solution differs from greedy, show you can swap to match greedy without losing optimality. Faster than DP but only correct for specific problem structures.

## Core Concepts

**Greedy choice property:** At each step, a locally optimal choice exists that is part of some globally optimal solution. If you can prove this (usually by exchange argument), greedy is correct.

**Exchange argument:** Suppose optimal solution O differs from greedy G at step k. Show that swapping O's choice at step k with G's choice produces a solution at least as good. By induction, G is globally optimal.

**Common greedy patterns:**

| Pattern | Key insight | Example problems |
|---|---|---|
| Interval scheduling | Sort by end time, pick earliest-ending non-overlapping | LC 435, LC 253 |
| Activity selection | Greedy by end time always optimal | Classic scheduling |
| Huffman coding | Build tree bottom-up by frequency | Compression |
| Jump game | Track farthest reachable index | LC 55, LC 45 |
| Task scheduling | Fill CPU slots greedily by frequency | LC 621 |
| Assign resources | Match greedily in sorted order | LC 455 |

**When greedy FAILS:** When local optimal doesn't lead to global optimal. Example: coin change with non-standard denominations (need DP). Greedy works for US coins (25, 10, 5, 1) but fails for denominations like (1, 3, 4) with target 6 (greedy: 4+1+1=3 coins; optimal: 3+3=2 coins).

## Implementations

```python
from typing import List
import heapq
from collections import Counter


# --- Interval Scheduling (non-overlapping intervals, LC 435) ---
def erase_overlap_intervals(intervals: List[List[int]]) -> int:
    """
    Min intervals to remove so no overlap.
    Greedy: sort by end time, keep interval if it doesn't overlap with last kept.
    """
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])   # sort by end time — key insight
    count = 0          # intervals to remove
    last_end = intervals[0][1]
    for i in range(1, len(intervals)):
        if intervals[i][0] < last_end:    # overlap → remove current
            count += 1
        else:
            last_end = intervals[i][1]    # no overlap → keep, update last_end
    return count


# --- Jump Game (LC 55) ---
def can_jump(nums: List[int]) -> bool:
    """
    Can you reach last index? Greedy: track farthest reachable index.
    At each position, if it's unreachable, return False.
    O(n) time, O(1) space.
    """
    farthest = 0
    for i in range(len(nums)):
        if i > farthest:
            return False         # position i is unreachable
        farthest = max(farthest, i + nums[i])
    return True


# --- Meeting Rooms II (LC 253) ---
def min_meeting_rooms(intervals: List[List[int]]) -> int:
    """
    Min rooms needed for all meetings.
    Greedy: use heap to track end times of ongoing meetings.
    Sort by start; if earliest-ending room is free, reuse it.
    O(n log n) time, O(n) space.
    """
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[0])   # sort by start time
    heap: List[int] = []   # min-heap of end times
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heappop(heap)   # room is free, reuse it
        heapq.heappush(heap, end)
    return len(heap)   # rooms still occupied = rooms needed


# --- Task Scheduler (LC 621) ---
def least_interval(tasks: List[str], n: int) -> int:
    """
    Min CPU intervals to finish all tasks with cooldown n between same tasks.
    Greedy: always schedule the most frequent remaining task.
    Formula: max((max_count - 1) * (n + 1) + tasks_with_max_count, len(tasks))
    O(m) where m = number of tasks.
    """
    counts = Counter(tasks)
    max_count = max(counts.values())
    tasks_with_max = sum(1 for c in counts.values() if c == max_count)
    # Idle slots formula: fill by most frequent, remaining tasks fill idle slots
    slots_needed = (max_count - 1) * (n + 1) + tasks_with_max
    return max(slots_needed, len(tasks))


# --- Assign Cookies (LC 455) ---
def find_content_children(greed: List[int], size: List[int]) -> int:
    """
    Give each child at most one cookie. Child i needs cookie >= greed[i].
    Greedy: sort both; assign smallest sufficient cookie to each child.
    O(n log n) time, O(1) space.
    """
    greed.sort()
    size.sort()
    child = cookie = 0
    while child < len(greed) and cookie < len(size):
        if size[cookie] >= greed[child]:
            child += 1   # child satisfied, move to next child
        cookie += 1      # always advance cookie (used or too small)
    return child


# --- Minimum number of arrows to burst balloons (LC 452) ---
def find_min_arrows(points: List[List[int]]) -> int:
    """
    Shoot vertical arrows to burst all balloons (intervals on x-axis).
    Greedy: sort by end, shoot at end of first balloon — bursts all overlapping.
    O(n log n).
    """
    points.sort(key=lambda x: x[1])
    arrows = 1
    arrow_pos = points[0][1]
    for start, end in points[1:]:
        if start > arrow_pos:         # balloon not burst by current arrow
            arrows += 1
            arrow_pos = end
    return arrows
```

## Key Problems

| Problem | Greedy Strategy | Time | Space |
|---|---|---|---|
| LC 455 Assign Cookies | Sort both, match smallest sufficient | O(n log n) | O(1) |
| LC 55 Jump Game | Track farthest reachable | O(n) | O(1) |
| LC 253 Meeting Rooms II | Sort by start + min-heap of ends | O(n log n) | O(n) |
| LC 435 Non-overlapping Intervals | Sort by end, keep non-overlapping | O(n log n) | O(1) |
| LC 621 Task Scheduler | Frequency formula | O(m) | O(1) |
| LC 452 Min Arrows Balloons | Sort by end, shoot at end | O(n log n) | O(1) |
| LC 45 Jump Game II | BFS-level jumps | O(n) | O(1) |

## Common Mistakes / Gotchas
- **Wrong sort key:** interval scheduling requires sort by **end** time, not start — sorting by start leads to incorrect greedy
- **Greedy without proof:** always verify with exchange argument; test on examples like non-standard coin denominations
- **Meeting rooms vs non-overlapping:** meeting rooms minimizes rooms needed (count overlaps); non-overlapping minimizes removed intervals (different greedy)
- **Off-by-one in task scheduler:** formula is `(max_count - 1) * (n + 1) + tasks_with_max_count`; edge case when tasks fill all slots exactly → answer is just `len(tasks)`
- **Reusing cookie vs skipping:** in assign cookies, always advance the cookie pointer regardless; only advance child if cookie satisfied

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "How do you prove greedy is correct?" | Exchange argument: assume optimal differs from greedy at some step, show swapping to greedy choice gives an equally good or better solution. By induction, greedy is optimal. |
| "When does greedy fail?" | When the greedy choice at one step prevents reaching the global optimum. Example: coin change with denominations (1, 3, 4) and target 6. Need DP. |
| "Interval scheduling vs interval partitioning?" | Scheduling (max non-overlapping intervals): sort by end time. Partitioning (min rooms): sort by start time + min-heap of end times. |
| "Jump Game greedy intuition?" | At each position, update the farthest you can reach. If you're ever at a position beyond farthest, you're stuck. You never need to simulate exact jumps — just track reachability. |

## Practice Resources
- LeetCode: 455, 55, 45, 253, 435, 621, 452, 134, 763
- Key insight problems: 45 Jump Game II (greedy BFS), 763 Partition Labels (greedy with last occurrence)

## Related Topics
- [Dynamic Programming](dynamic-programming.md) — DP when greedy fails; interval DP for more complex scheduling
- [Sorting](sorting.md) — most greedy algorithms start with a sort
- [Heaps](../data-structures/heaps.md) — min-heap used in meeting rooms and task scheduling
