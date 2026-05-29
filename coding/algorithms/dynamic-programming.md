# Dynamic Programming

## TL;DR
Dynamic programming solves problems with **overlapping subproblems** and **optimal substructure** by storing intermediate results. Framework: define state → write transition → set base case → choose iteration order. DP trades space for time: O(n²) → O(n) or O(2^n) → O(n²) by memoization. Recognize DP when you see "min/max cost with choices" or "count number of ways."

## Core Concepts

**Two implementation styles:**
- **Top-down (memoization):** recursive with cache; easier to think about; starts from answer
- **Bottom-up (tabulation):** iterative; fills table from base cases up; often faster (no recursion overhead)

**Framework: State → Transition → Base Case → Order**
1. **State:** what parameters fully characterize a subproblem? (index, remaining budget, etc.)
2. **Transition:** how does `dp[i]` relate to smaller subproblems?
3. **Base case:** smallest subproblem with known answer
4. **Order:** fill table so dependencies are computed first

**Common DP categories:**

| Category | Examples | State | Transition |
|----------|----------|-------|------------|
| 1D prefix | Climbing Stairs, Coin Change | dp[i] = best for first i | dp[i] = f(dp[i-1], dp[i-2], ...) |
| 2D grid/string | Edit Distance, LCS | dp[i][j] = best for i,j | dp[i][j] = f(dp[i-1][j], dp[i][j-1], ...) |
| Interval | Burst Balloons, Matrix Chain | dp[l][r] = best for range [l,r] | dp[l][r] = max over split points |
| 0/1 Knapsack | 0/1 Knapsack, Subset Sum | dp[i][w] = best using items 0..i with capacity w | include or exclude item i |
| Unbounded Knapsack | Coin Change, Unbounded Knapsack | dp[w] = best with capacity w | can reuse items |

**Recognition patterns:**
- "Number of ways" → DP (count)
- "Min/max cost with choices" → DP (optimization)
- "Can we achieve X" → DP (feasibility)
- Recurrence visible in brute force → DP
- Overlapping recursive calls → memoize

## Implementations

```python
from typing import List
from functools import lru_cache

# --- 1D DP: Climbing Stairs (LC 70) ---
def climb_stairs(n: int) -> int:
    """Number of ways to reach step n using 1 or 2 steps. Fibonacci variant."""
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


# --- Unbounded Knapsack: Coin Change (LC 322) ---
def coin_change(coins: List[int], amount: int) -> int:
    """
    Minimum coins to make amount. Unbounded knapsack.
    dp[i] = min coins needed for amount i.
    """
    INF = float("inf")
    dp = [INF] * (amount + 1)
    dp[0] = 0    # base case: 0 coins for amount 0
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin and dp[i - coin] != INF:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != INF else -1


# --- 2D DP: Longest Common Subsequence (LC 1143) ---
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    dp[i][j] = LCS of text1[:i] and text2[:j].
    Transition: if chars match → dp[i-1][j-1]+1; else max(dp[i-1][j], dp[i][j-1]).
    O(m*n) time and space.
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# --- 0/1 Knapsack ---
def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Max value with items that can each be used at most once.
    dp[i][w] = max value using items 0..i with weight limit w.
    Space optimization: iterate w backward → O(capacity) space.
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):  # reverse to avoid using item twice
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]


# --- LIS: Longest Increasing Subsequence (LC 300) ---
def length_of_lis(nums: List[int]) -> int:
    """
    dp[i] = length of LIS ending at index i.
    O(n^2) DP. (O(n log n) version uses patience sorting / binary search.)
    """
    if not nums:
        return 0
    n = len(nums)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


# --- 2D DP: Edit Distance (LC 72) ---
def min_distance(word1: str, word2: str) -> int:
    """
    Levenshtein distance: minimum insert/delete/replace operations.
    dp[i][j] = edit distance between word1[:i] and word2[:j].
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Base cases: convert to/from empty string
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]              # no operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],       # delete from word1
                    dp[i][j - 1],       # insert into word1
                    dp[i - 1][j - 1],   # replace
                )
    return dp[m][n]


# --- Top-down memoization example ---
@lru_cache(maxsize=None)
def fib_memo(n: int) -> int:
    """Memoized fibonacci: O(n) time, O(n) space."""
    if n <= 1:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)
```

## Key Problems

| Problem | Category | Time | Space |
|---|---|---|---|
| LC 70 Climbing Stairs | 1D DP | O(n) | O(1) |
| LC 322 Coin Change | Unbounded knapsack | O(n * amount) | O(amount) |
| LC 1143 LCS | 2D string DP | O(m*n) | O(m*n) |
| LC 300 LIS | 1D DP | O(n²) or O(n log n) | O(n) |
| LC 72 Edit Distance | 2D string DP | O(m*n) | O(m*n) |
| LC 416 Partition Equal Subset | 0/1 knapsack | O(n * sum) | O(sum) |
| LC 198 House Robber | 1D DP | O(n) | O(1) |

## Common Mistakes / Gotchas
- **Wrong state definition:** state must fully capture the subproblem — forgetting a dimension leads to incorrect transitions
- **Wrong iteration order:** for 0/1 knapsack, iterating weight forward allows reuse (unbounded); backward prevents it
- **Off-by-one in base cases:** dp[0] often represents "empty" (0 items, amount 0) — don't skip it
- **Max/min initialization:** use 0 for count DP; use `float("inf")` for min-cost DP; use `-float("inf")` for max-profit DP
- **Top-down recursion limit:** Python default recursion limit is 1000; use `sys.setrecursionlimit` or convert to bottom-up for large n

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "How do you recognize DP?" | Overlapping subproblems + optimal substructure. If brute force is exponential recursion with repeated calls → DP. Keywords: 'min/max', 'number of ways', 'can we achieve'. |
| "Top-down vs bottom-up?" | Top-down: write recursion naturally + memoize; easier to think. Bottom-up: iterate in order; better constant factor, no stack overflow risk. |
| "Space optimization?" | If dp[i] only depends on dp[i-1] (and dp[i-2]), use two variables instead of O(n) array. 2D DP often compresses to O(min(m,n)) with rolling array. |
| "LCS vs edit distance?" | LCS finds longest common subsequence (chars that appear in order in both). Edit distance allows insert/delete/replace to transform one string to another. Related but different DP tables. |

## Practice Resources
- LeetCode: 70, 198, 322, 300, 1143, 72, 416, 139, 309, 312
- Key insight: master coin change + LCS first; they cover unbounded knapsack and 2D string DP patterns

## Related Topics
- [Recursion](recursion.md) — top-down DP is memoized recursion
- [Backtracking](backtracking.md) — when DP is not applicable (need to enumerate all, not just count)
- [Divide & Conquer](divide-conquer.md) — D&C differs from DP in that subproblems don't overlap
