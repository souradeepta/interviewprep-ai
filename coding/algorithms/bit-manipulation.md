# Bit Manipulation

## TL;DR
Bit manipulation operates directly on binary representations of integers for O(1) space and very fast operations. Core tricks: XOR cancels pairs, `n & (n-1)` removes the lowest set bit, `n & (-n)` isolates the lowest set bit, left/right shifts multiply/divide by 2. Used for toggling, masking, counting set bits, enumeration of subsets, and finding unique elements.

## Core Concepts

**Bitwise operators:**

| Operator | Symbol | Effect | Example |
|---|---|---|---|
| AND | `&` | 1 only if both 1 | 5 & 3 = 1 (101 & 011 = 001) |
| OR | `\|` | 1 if either 1 | 5 \| 3 = 7 (101 \| 011 = 111) |
| XOR | `^` | 1 if different | 5 ^ 3 = 6 (101 ^ 011 = 110) |
| NOT | `~` | flip all bits | ~5 = -6 (two's complement) |
| Left shift | `<<` | multiply by 2^k | 5 << 1 = 10 |
| Right shift | `>>` | divide by 2^k (floor) | 5 >> 1 = 2 |

**Essential bit tricks (memorize):**

| Trick | Expression | Effect |
|---|---|---|
| Remove lowest set bit | `n & (n - 1)` | Clears the rightmost 1 bit |
| Isolate lowest set bit | `n & (-n)` | Keeps only the rightmost 1 bit |
| Check if power of 2 | `n > 0 and (n & (n-1)) == 0` | True iff n is a power of 2 |
| Toggle kth bit | `n ^ (1 << k)` | Flip bit k |
| Set kth bit | `n \| (1 << k)` | Force bit k to 1 |
| Clear kth bit | `n & ~(1 << k)` | Force bit k to 0 |
| Check kth bit | `(n >> k) & 1` | 1 if bit k is set |
| XOR identity | `n ^ n = 0; n ^ 0 = n` | Pairs cancel out |

**XOR properties:**
- Commutative: `a ^ b = b ^ a`
- Associative: `(a ^ b) ^ c = a ^ (b ^ c)`
- Self-inverse: `a ^ a = 0`
- Identity: `a ^ 0 = a`
- Consequence: XOR all elements in an array → duplicate elements cancel → returns element appearing odd number of times

## Implementations

```python
from typing import List


# --- Count set bits (popcount) ---
def count_bits_single(n: int) -> int:
    """Count number of 1 bits in n. O(number of set bits)."""
    count = 0
    while n:
        n &= (n - 1)   # removes lowest set bit each iteration
        count += 1
    return count

# Python built-in:
# bin(n).count('1')  or  n.bit_count()  (Python 3.10+)


# --- LC 338: Counting Bits (DP + bit trick) ---
def count_bits_range(n: int) -> List[int]:
    """
    Return array where ans[i] = number of 1 bits in i, for i in [0, n].
    DP: bits(i) = bits(i >> 1) + (i & 1). O(n) time and space.
    """
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i >> 1] + (i & 1)
    return dp


# --- LC 136: Single Number (XOR pairs cancel) ---
def single_number(nums: List[int]) -> int:
    """
    Every element appears twice except one. XOR all → duplicates cancel → unique remains.
    O(n) time, O(1) space.
    """
    result = 0
    for n in nums:
        result ^= n
    return result


# --- LC 137: Single Number II (appears 3 times, one appears once) ---
def single_number_ii(nums: List[int]) -> int:
    """
    Count bits across all numbers. Bit appears 3k or 3k+1 times.
    Mod 3 of each bit position gives the unique number's bits.
    O(32n) = O(n) time, O(1) space.
    """
    result = 0
    for bit in range(32):
        total = sum((n >> bit) & 1 for n in nums)
        if total % 3:   # this bit is set in the unique element
            result |= (1 << bit)
    # Handle negative numbers in Python (two's complement)
    if result >= (1 << 31):
        result -= (1 << 32)
    return result


# --- LC 190: Reverse Bits ---
def reverse_bits(n: int) -> int:
    """
    Reverse all 32 bits of unsigned integer n. O(32) = O(1).
    Take bit from LSB of n, put into MSB of result.
    """
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


# --- Check if n is power of 2 ---
def is_power_of_two(n: int) -> bool:
    """
    Power of 2 has exactly one set bit. n & (n-1) removes it → 0.
    O(1).
    """
    return n > 0 and (n & (n - 1)) == 0


# --- Enumerate all subsets using bitmask ---
def enumerate_subsets(nums: List[int]) -> List[List[int]]:
    """
    Generate all 2^n subsets using bitmask from 0 to 2^n - 1.
    Bit k set in mask → include nums[k] in subset.
    O(n * 2^n) time, O(2^n) space for output.
    """
    n = len(nums)
    result = []
    for mask in range(1 << n):   # 0 to 2^n - 1
        subset = []
        for bit in range(n):
            if mask & (1 << bit):
                subset.append(nums[bit])
        result.append(subset)
    return result


# --- Find two unique numbers (LC 260: Single Number III) ---
def single_number_iii(nums: List[int]) -> List[int]:
    """
    Two numbers appear once, rest appear twice.
    XOR all → a ^ b. Find any differing bit. Split by that bit.
    XOR each group → find a and b. O(n), O(1).
    """
    xor_ab = 0
    for n in nums:
        xor_ab ^= n          # = a ^ b

    # Find rightmost differing bit between a and b
    diff_bit = xor_ab & (-xor_ab)   # isolate lowest set bit

    a, b = 0, 0
    for n in nums:
        if n & diff_bit:
            a ^= n    # group where bit is set
        else:
            b ^= n    # group where bit is not set
    return [a, b]
```

## Key Problems

| Problem | Technique | Time | Space |
|---|---|---|---|
| LC 136 Single Number | XOR all elements | O(n) | O(1) |
| LC 137 Single Number II | Count bits mod 3 | O(n) | O(1) |
| LC 260 Single Number III | XOR + split by diff bit | O(n) | O(1) |
| LC 338 Counting Bits | DP: `dp[i] = dp[i>>1] + (i&1)` | O(n) | O(n) |
| LC 190 Reverse Bits | Shift + OR 32 times | O(1) | O(1) |
| LC 191 Number of 1 Bits | `n & (n-1)` loop | O(k) | O(1) |
| LC 78 Subsets (bitmask) | Enumerate 0 to 2^n | O(n * 2^n) | O(2^n) |
| LC 231 Power of Two | `n & (n-1) == 0` | O(1) | O(1) |

## Common Mistakes / Gotchas
- **Python integers are arbitrary precision:** `~n` gives `-(n+1)` (not a 32-bit flip); for 32-bit operations, mask with `& 0xFFFFFFFF`
- **Signed vs unsigned in Python:** Python has no unsigned int; when reversing bits or handling negative numbers, account for 32-bit two's complement manually
- **XOR chain of three:** `a ^ a ^ b = b` works because `a ^ a = 0` and `0 ^ b = b`; order doesn't matter (commutative)
- **Subset enumeration size:** `1 << n` for n=32 is 4 billion iterations — only use for n ≤ 20 or so
- **Bit shift precedence:** `1 << k + 1` evaluates as `1 << (k + 1)` not `(1 << k) + 1`; use parentheses

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "How to find single number with XOR?" | XOR all elements: duplicates cancel to 0, single remains. Works because a^a=0, a^0=a, and XOR is commutative/associative. |
| "Count set bits efficiently?" | Repeatedly clear lowest set bit: `n &= (n-1)` and count. Runs in O(k) where k is number of set bits. Or use `bin(n).count('1')`. |
| "Enumerate subsets via bitmask?" | For n elements, iterate mask from 0 to 2^n - 1. Bit k set in mask → include element k. O(n * 2^n) total work. |
| "Power of 2 check?" | `n > 0 and (n & (n-1)) == 0`. Power of 2 has exactly one set bit; subtracting 1 flips all lower bits; AND gives 0. |

## Practice Resources
- LeetCode: 136, 137, 190, 191, 231, 338, 260, 389, 421
- Key insight: master XOR tricks (136, 260) and bit DP (338) — they appear frequently

## Related Topics
- [Backtracking](backtracking.md) — bitmask subset enumeration is an alternative to recursive subset generation
- [Dynamic Programming](dynamic-programming.md) — bitmask DP for TSP and set-cover problems
- [Hash Tables](../data-structures/hash-tables.md) — XOR-based hashing uses bit manipulation
