'''
Number of Steps to Reduce a Number to Zero
premium lock icon
Companies
Hint
Given an integer num, return the number of steps to reduce it to zero.

In one step, if the current number is even, you have to divide it by 2, otherwise, you have to subtract 1 from it.


Example 1:

Input: num = 14
Output: 6
Explanation: 
Step 1) 14 is even; divide by 2 and obtain 7. 
Step 2) 7 is odd; subtract 1 and obtain 6.
Step 3) 6 is even; divide by 2 and obtain 3. 
Step 4) 3 is odd; subtract 1 and obtain 2. 
Step 5) 2 is even; divide by 2 and obtain 1. 
Step 6) 1 is odd; subtract 1 and obtain 0.
Example 2:

Input: num = 8
Output: 4
Explanation: 
Step 1) 8 is even; divide by 2 and obtain 4. 
Step 2) 4 is even; divide by 2 and obtain 2. 
Step 3) 2 is even; divide by 2 and obtain 1. 
Step 4) 1 is odd; subtract 1 and obtain 0.
Example 3:

Input: num = 123
Output: 12
 

Constraints:

0 <= num <= 106

'''

n = int(input())
steps = 0
while n > 0:
    if n % 2 == 0:
        n = n // 2
    else:
        n = n - 1
    steps += 1
print(steps)


# or 

def numberOfSteps(num):
    if num == 0:
        return 0
    # If even, divide by 2; if odd, subtract 1
    if num % 2 == 0:
        return 1 + numberOfSteps(num // 2)
    else:
        return 1 + numberOfSteps(num - 1)


# Afternoon Session: 2D DP Problems
# Grid Path Count (m x n)
m, n = 3, 3
dp = [[1]*n for _ in range(m)]
for i in range(1, m):
    for j in range(1, n):
        dp[i][j] = dp[i-1][j] + dp[i][j-1]
print(dp[-1][-1])  # Output: 6


# Minimum Path Sum
grid = [[1,3,1],[1,5,1],[4,2,1]]
m, n = len(grid), len(grid[0])
for i in range(1, m):
    grid[i][0] += grid[i-1][0]
for j in range(1, n):
    grid[0][j] += grid[0][j-1]
for i in range(1, m):
    for j in range(1, n):
        grid[i][j] += min(grid[i-1][j], grid[i][j-1])
print(grid[-1][-1])  # Output: 7

'''
Absolutely! Let’s walk through each of the two problems—**Grid Path Count** and **Minimum Path Sum**—step by step, using one test case from each to really understand the mechanics behind them. 👣✨

---

## 🧭 Problem 1: Grid Path Count (Unique Paths)

### 🧪 Test Case: `m = 3`, `n = 3`
We want to find how many unique paths exist from the top-left corner to the bottom-right corner in a `3×3` grid, where you can only move **right** or **down**.

### 💡 Step-by-Step Breakdown

1. **Initialize DP Table**:  
   Every cell in the first row and first column can only be reached by one path (either keep going right or keep going down).
   ```python
   dp = [
       [1, 1, 1],
       [1, 0, 0],
       [1, 0, 0]
   ]
   ```

2. **Fill the DP Table**:
   For each cell not in the first row or column, the number of paths to reach it is the **sum of paths from the top and left cell**.

   - `dp[1][1] = dp[0][1] + dp[1][0] = 1 + 1 = 2`  
   - `dp[1][2] = dp[0][2] + dp[1][1] = 1 + 2 = 3`  
   - `dp[2][1] = dp[1][1] + dp[2][0] = 2 + 1 = 3`  
   - `dp[2][2] = dp[1][2] + dp[2][1] = 3 + 3 = 6`

   Final DP Table:
   ```python
   dp = [
       [1, 1, 1],
       [1, 2, 3],
       [1, 3, 6]
   ]
   ```

3. **Result**:  
   The bottom-right cell `dp[2][2] = 6` → ✅ 6 unique paths.

---

## 💸 Problem 2: Minimum Path Sum

### 🧪 Test Case: `grid = [[1,3,1],[1,5,1],[4,2,1]]`
Find the path from top-left to bottom-right with the **minimum sum**, moving only right or down.

### 💡 Step-by-Step Breakdown

1. **Initialize First Column and First Row**:
   - First column:
     - `grid[1][0] = 1 + 1 = 2`
     - `grid[2][0] = 2 + 4 = 6`
   - First row:
     - `grid[0][1] = 1 + 3 = 4`
     - `grid[0][2] = 4 + 1 = 5`

   Grid now looks like:
   ```python
   [
     [1, 4, 5],
     [2, 5, 1],
     [6, 2, 1]
   ]
   ```

2. **Fill the Rest of the Grid**:
   - `grid[1][1] = min(4, 2) + 5 = 2 + 5 = 7`
   - `grid[1][2] = min(5, 7) + 1 = 5 + 1 = 6`
   - `grid[2][1] = min(7, 6) + 2 = 6 + 2 = 8`
   - `grid[2][2] = min(6, 8) + 1 = 6 + 1 = 7`

   Final Grid:
   ```python
   [
     [1, 4, 5],
     [2, 7, 6],
     [6, 8, 7]
   ]
   ```

3. **Result**:  
   Bottom-right cell `grid[2][2] = 7` → ✅ Minimum path sum is **7**

---

Would you like me to diagram the grid visually next, or convert these into recursive solutions with memoization? I'm here for it! 🔁🧠📊




Unique Paths
(Same logic as Grid Path Count — identical code applies)

Edit Distance (Levenshtein)
word1 = "horse"
word2 = "ros"
m, n = len(word1), len(word2)

dp = [[0]*(n+1) for _ in range(m+1)]
for i in range(m+1):
    dp[i][0] = i
for j in range(n+1):
    dp[0][j] = j

for i in range(1, m+1):
    for j in range(1, n+1):
        if word1[i-1] == word2[j-1]:
            dp[i][j] = dp[i-1][j-1]
        else:
            dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

print(dp[m][n])  # Output: 3









Exactly! The Checkers game problem is **not** based on dynamic programming (DP). Instead, it revolves around **graph traversal techniques**, particularly **Depth-First Search (DFS)** or **Breadth-First Search (BFS)**.

Here's why:

### ❌ Why It's Not DP
- DP is great for problems where **subproblems overlap** and you build a solution bottom-up (like counting paths or minimizing cost).
- The Checkers game doesn't involve solving a global problem with reusable subproblems. Each move is localized around connected components of `1`s.
- The game's turn-based nature and **greedy strategy** for grabbing the largest cluster make DP an ill fit.

### ✅ What It Really Is
- Think of the matrix of `1`s as a grid of **nodes in a graph**, with edges between adjacent nodes.
- Capturing cells = finding **connected regions**.
- You simulate each player’s turn by:
  - Scanning the grid to find all regions of connected `1`s.
  - Sorting these regions by size (since the players play optimally).
  - Alternating turns to pick the largest remaining group.

This falls squarely under **grid-based connected component analysis**, similar to problems like:
- Number of Islands
- Connected colored regions in images
- Territory capture in strategic games

If you’d like, I can walk you through how to implement it or show how the regions are extracted in code. Suresh deserves a fair shot at those cells! 😄🎯

'''