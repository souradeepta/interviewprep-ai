# Intervals and Sweep Line

Interval questions are usually about ordering events, merging overlap, or
allocating a scarce resource. Sort by the left endpoint for coverage and
merging; sort start/end events separately when the question asks for peak
concurrency or capacity.

## Merge overlapping intervals

```python
def merge(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0][:]]
    for left, right in intervals[1:]:
        if left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged
```

Clarify whether touching intervals such as `[1, 2]` and `[2, 3]` count as
overlap. Change `<=` to `<` when they do not.

## Sweep line and event ordering

For maximum concurrent meetings, add `+1` at each start and `-1` at each end.
When endpoints touch, process an end before a start if the resource is
reusable at that instant; process the start first if the intervals are
closed and overlap at the boundary.

```python
def peak_concurrency(intervals):
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    active = peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        peak = max(peak, active)
    return peak
```

Practice: LeetCode 56, 57, 252, 253, 435, 452, 986, 1094, 1851.
