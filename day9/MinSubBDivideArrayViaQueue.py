from collections import deque

# def minSumByAndSubarrays(nums, andValues):
#     n = len(nums)
#     m = len(andValues)
#     # dp[i] = minimum cost to use first i numbers
#     dp = [float('inf')] * (n + 1)
#     dp[0] = 0  # Start with 0 cost
#     # For each target AND value we need to match
#     for target in andValues:
#         new_dp = [float('inf')] * (n + 1)
#         # For each position in nums
#         for end in range(n):
#             # Use deque to store all possible AND values ending at this position
#             # Each item is (start_position, and_value)
#             possible_ands = deque()
#             # Calculate AND for all subarrays ending at 'end'
#             current_and = nums[end]
#             for start in range(end, -1, -1):
#                 if start < end: current_and &= nums[start]
#                 # Only keep if it could still match our target
#                 if current_and >= target: possible_ands.append((start, current_and))
#                 else: break  # AND will only get smaller, so stop
#             # Check if any subarray matches our target
#             for start_pos, and_val in possible_ands:
#                 if and_val == target:
#                     # We can use this subarray! Update our DP
#                     cost = dp[start_pos] + nums[end]
#                     new_dp[end + 1] = min(new_dp[end + 1], cost)
#         dp = new_dp
#     # Return result or -1 if impossible
#     return dp[n] if dp[n] != float('inf') else -1

# def minSumByAndSubarrays(nums, andValues):
#       n,m = len(nums), len(andValues)
#       if m>n:return -1
#       grid = [ [float('inf')] * (m+1) for _ in range(n+1) ]
#       grid[0][0] = 0
#       for i in range(1,n+1):
#             for j in range(1,m+1):
#                   and_accum = nums[i-1]
#                   for k in range(i-1,-1,-1):
#                         if k<i-1: and_accum &= nums[k]
#                         if and_accum == andValues[j-1]:grid[i][j] = min(grid[i][j],grid[k][j-1]+nums[i-1])
#       return grid[n][m] if grid[n][m] != float('inf') else -1

def minSumByAndSubarrays(nums, andValues):
      n,m = len(nums), len(andValues)
      index = 0
      que = deque()
      for target in andValues:
            found = False
            value = nums[index]
            for sub in range(index,n):
                  value &= nums[sub]
                  if value == target:
                      que.append(nums[sub])
                      index = sub+1
                      found=True
                      break
            if not found: return -1
      return sum(que)
      

# Test the function
print(minSumByAndSubarrays([4,6],[4]))
print("Test 1:", minSumByAndSubarrays([1,4,3,3,2], [0,3,3,2]))  # Should output: 12
print("Test 2:", minSumByAndSubarrays([2,3,5,7,7,7,5], [0,7,5]))  # Should output: 17
print("Test 3:", minSumByAndSubarrays([1,2,3,4], [2]))  # Should output: -1

'''
Helping students understand this problem involves breaking it down into digestible pieces with intuitive analogies, visual flow, and hands-on exercises. Here's how you can guide them:

---

## 🧭 Step-by-Step Teaching Guide

### 1. 🧠 Introduce Bitwise AND with Visuals
- Use binary examples:
  - Show `5 & 3` → `101 & 011 = 001` → result is `1`
- Explain that the `&` operation returns a number where *only* the bits that are `1` in **both** numbers are kept.

### 2. 🧩 Build Concept of Subarray Partitioning
- Start with the idea of cutting `nums` into **contiguous groups**, like dividing students into seated rows.
- Each group must produce a **target value** using the bitwise AND of all elements in that group.
- Example: `nums = [1, 4]` → `1 & 4 = 0`, so this subarray satisfies target `0`.

### 3. 🧮 Highlight How “Value” is Defined
- Clarify that the **value of a subarray** is its **last element**.
- Reinforce this by re-annotating examples with arrows or highlights:
  ```
  [1,4] → value: 4
  [3] → value: 3
  ```

### 4. 🪜 Explain the Goal Mechanically
- The challenge is to:
  1. Divide the array into `m` subarrays.
  2. For each subarray, check:  
     `bitwise AND of subarray == andValues[i]`
  3. Minimize the **sum of the subarray values**.
  4. If it’s not possible to meet the condition, return `-1`.

### 5. 🛠️ Coding and Debugging Practice
- Encourage students to write a brute-force approach first:
  - Try all partitions (backtracking).
  - For each, compute AND and check against `andValues`.
- Then discuss how to **optimize** using dynamic programming:
  - `dp[i][j]`: minimum sum to fulfill first `j` values using first `i` elements.
  - Show how AND accumulates from left to right:
    ```
    running_and = nums[start]
    for end in range(start+1, n):
        running_and &= nums[end]
        if running_and == andValues[i]:
            update dp
    ```

### 6. 🧪 Use Examples Actively
Try walking through this:

```python
nums = [2, 3, 5, 7, 7, 7, 5]
andValues = [0, 7, 5]
```

- First partition? Try `[2, 3, 5] → 2 & 3 & 5 = 0 ✅`
- Second? Try `[7, 7, 7] → all 7s, AND = 7 ✅`
- Third? `[5] → value = 5 ✅`
- Total sum = `5 + 7 + 5 = 17`

Let them sketch it visually, like boxes with bitwise AND chains.

---

## ✏️ Exercises
1. Ask students to write a function to compute AND of any subarray.
2. Give small inputs (e.g., `nums = [6, 5, 4], andValues = [4, 4]`) and ask them to manually try all partitions.
3. Challenge them to find which partitions minimize value sum.

---

## 🎓 Teaching Tips
- Reinforce that **bitwise AND shrinks**—values only stay same or drop.
- Build intuition: “Can we delay ending a subarray to get smaller value?”
- Encourage drawing binary numbers side-by-side.
- Let students test `isValid(start, end, target)` helpers for debugging.

---

Would you like me to prepare a classroom-ready worksheet or visual slides for this concept?

'''


# from collections import deque
# def minSumByAndSubarrays(nums, andValues):
#     n = len(nums)
#     m = len(andValues)
#     # Initialize dp: dp[i] = minimum sum if we consider first i elements of nums
#     dp = [float('inf')] * (n + 1)
#     dp[0] = 0  # Base case: no elements used, cost is 0
#     for i in range(m):
#         new_dp = [float('inf')] * (n + 1)
#         # We use a deque to store tuples (start_index, current_and)
#         for j in range(i, n):
#             cur_and = nums[j]
#             for k in range(j, i - 1, -1):
#                 cur_and &= nums[k]
#                 if cur_and < andValues[i]:  # Early exit if AND too small
#                     break
#                 if cur_and == andValues[i]:
#                     new_dp[j + 1] = min(new_dp[j + 1], dp[k] + nums[j])
#         dp = new_dp
#     return dp[n] if dp[n] != float('inf') else -1
# # Example 1
# print(minSumByAndSubarrays([1,4,3,3,2], [0,3,3,2]))  # Output: 12

# # Example 2
# print(minSumByAndSubarrays([2,3,5,7,7,7,5], [0,7,5]))  # Output: 17

# # Example 3
# print(minSumByAndSubarrays([1,2,3,4], [2]))  # Output: -1
