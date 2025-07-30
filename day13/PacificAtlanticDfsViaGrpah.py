class Solution:
    def pacificAtlantic(self, heights):
        if not heights: return []
        
        m, n = len(heights), len(heights[0])
        pacific = [[False] * n for _ in range(m)]
        atlantic = [[False] * n for _ in range(m)]

        def dfs(r, c, visited, prev_height):
            if (r < 0 or r >= m or c < 0 or c >= n):
                return
            if visited[r][c] or heights[r][c] < prev_height:
                return
            visited[r][c] = True
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                dfs(r + dr, c + dc, visited, heights[r][c])

        # Start DFS from Pacific edges
        for i in range(m):
            print("The below is for",i, 0, pacific, heights[i][0])
            dfs(i, 0, pacific, heights[i][0])
            print(pacific)
            dfs(i, n - 1, atlantic, heights[i][n - 1])
        for j in range(n):
            dfs(0, j, pacific, heights[0][j])
            dfs(m - 1, j, atlantic, heights[m - 1][j])

        result = []
        for r in range(m):
            for c in range(n):
                if pacific[r][c] and atlantic[r][c]:
                    result.append([r, c])
        return result
heights = [[1,2,2,3,5],
           [3,2,3,4,4],
           [2,4,5,3,1],
           [6,7,1,4,5],
           [5,1,1,2,4]]

sol = Solution()
print(sol.pacificAtlantic(heights))
# Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
'''
Absolutely! Here's a detailed dry run table showing how cells are marked as reachable by both the **Pacific** and **Atlantic** oceans using DFS traversal:

---

### 🌊 Problem Overview

- Water can flow from a cell to its neighbor **only if** the neighbor's height is _equal or greater_.
- Pacific touches the **top and left edges**, Atlantic touches the **bottom and right edges**.
- Goal: find cells that can reach **both** oceans.

---

---

You're speaking my language, S — let's break it down the way a problem architect would. Here's a full dry run of the `pacificAtlantic` algorithm using the matrix you provided, with visual snapshots to trace how Pacific and Atlantic reachability evolves step by step.

---

## 🌊 Input Matrix (Heights)
current >= height => True
all row of 0th column are True since pacific start left
0th row
then 0,0 -> 1 is default pacific
then 0,1 -> 2 is greater than 0,0->1 so True
then 0,2 -> 2 is greater than 0,0->1;0,1->2 so True
then 0,3 -> 3 is greater than 0,0->1;0,1->2;0,2->2 so True
then 0,4 -> 5 is greater than 0,0->1;0,1->2;0,2->2;0,3 -> 3 so True
1th row
then 1,0 -> 3 is default pacific
then 1,1 -> 2 is greater not than 1,0->3, but top's 0,2->2 so True
then 1,2 -> 2 is greater than 0,1->2;1,1->2 so True
then 1,3 -> 4 is greater than 1,0->3;1,1->2;1,2->3 so True
then 1,4 -> 4 is greater than 1,0->3;1,1->2;1,2->3;1,3 -> 4 so True
2nd row
then 2,0 -> 2 is default pacific
then 2,1 -> 4 is greater 2,0->2 so True
then 2,2 -> 5 is greater than 2,0->2;2,1->4 so True
then 2,3 -> 3 is greater not than 2,2->5;1,3->4 so False
then 2,4 -> 1 is greater not than 2,3->3;1,4->4 so False
3rd row
then 3,0 -> 6 is default pacific
then 3,1 -> 7 is greater 3,0->6;2,1->4 so True
then 3,2 -> 1 is greater not than 3,1->7;2,2->5 so False
then 3,3 -> 4 is too isolated from Pacific borders by height barriers because 2,3->False, 3,2-> False
then 3,4 -> 5 is too isolated from Pacific borders by height barriers
4th row
then 4,0 -> 5 is default pacific
then 4,1 -> 1 is greater not 4,0->5 so False
then 4,2 -> 1 is blocked because 3,2->False; 4-1->False so False
then 4,3 -> 2 is isolated
then 4,4 -> 4 is isolated
| Row\Col | 0 | 1 | 2 | 3 | 4 |
|--------:|--:|--:|--:|--:|--:|
| **0**   | 1 | 2 | 2 | 3 | 5 |
| **1**   | 3 | 2 | 3 | 4 | 4 |
| **2**   | 2 | 4 | 5 | 3 | 1 |
| **3**   | 6 | 7 | 1 | 4 | 5 |
| **4**   | 5 | 1 | 1 | 2 | 4 |

---

## 🔁 Dry Run Tables: Pacific and Atlantic Reachability


| Step | DFS Start | Ocean     | Cell (r, c) | Condition Met                | Visited Matrix Updated | Notes                          |
|------|-----------|-----------|-------------|------------------------------|-------------------------|---------------------------------|
| 1    | (0, 0)     | Pacific   | (0, 1)      | 2 ≥ 1                        | `pacific[0][1] = True` | Water flows right              |
| 2    | (0, 1)     | Pacific   | (0, 2)      | 2 ≥ 2                        | `pacific[0][2] = True` | Continue right                 |
| 3    | (0, 2)     | Pacific   | (0, 3)      | 3 ≥ 2                        | `pacific[0][3] = True` | Move toward (0,4)              |
| 4    | (0, 3)     | Pacific   | (0, 4)      | 5 ≥ 3                        | `pacific[0][4] = True` | Top-right corner is reachable |
| ...  | ...        | Atlantic  | (4, 4)      | 4 ≥ 2                        | `atlantic[4][4] = True`| Start from bottom-right        |
| ...  | (4, 4)     | Atlantic  | (3, 4)      | 5 ≥ 4                        | `atlantic[3][4] = True`| Water flows upward             |
| ...  | (3, 4)     | Atlantic  | (2, 4)      | 1 < 5                        | Blocked                | Cannot move further           |
| ...  | (3, 0)     | Pacific+Atlantic | — | 6 ≥ —                | Both reachable         | Found intersection             |


We'll mark `P` for Pacific-reachable and `A` for Atlantic-reachable. A cell may have both flags if reachable by both.

### 📘 Step 1: Pacific DFS (Top and Left Edges)

| Row\Col | 0 | 1 | 2 | 3 | 4 |
|--------:|--:|--:|--:|--:|--:|
| **0**   | P | P | P | P | P |
| **1**   | P |   | P | P |   |
| **2**   | P | P | P |   |   |
| **3**   | P | P |   |   |   |
| **4**   | P |   |   |   |   |

### 🌊 Step 2: Atlantic DFS (Bottom and Right Edges)

| Row\Col | 0 | 1 | 2 | 3 | 4 |
|--------:|--:|--:|--:|--:|--:|
| **0**   |   |   |   |   | A |
| **1**   |   |   |   | A | A |
| **2**   |   |   | A |   |   |
| **3**   | A | A |   | A | A |
| **4**   | A |   |   |   | A |

---

## ✅ Final Cells Reachable by Both Oceans

Now we check where **both Pacific and Atlantic flags** are `True`:

| Row\Col | Status |
|--------:|:-------|
| **0,4** | ✅ P & A |
| **1,3** | ✅ P & A |
| **1,4** | ✅ P & A |
| **2,2** | ✅ P & A |
| **3,0** | ✅ P & A |
| **3,1** | ✅ P & A |
| **4,0** | ✅ P & A |

📢 Final Result:
```python
[[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
```

---

## 💡 Thought Process Recap

- **DFS from edges** ensures we only traverse cells with valid downhill water flow.
- Instead of simulating flow from every cell to both oceans, we simulate *from oceans to cells*, then find intersections.
- Efficient and elegant: scales linearly with grid size.

---

Want me to package this up as a HackerRank-style problem with constraints and test cases next? Or maybe even add a BFS flavor to the solution for variation?


'''