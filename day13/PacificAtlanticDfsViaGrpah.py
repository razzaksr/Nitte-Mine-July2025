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
            dfs(i, 0, pacific, heights[i][0])
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

### 📊 Dry Run Table Summary – Key DFS Events

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

---

### ✅ Final Result — Cells Reaching Both Oceans

```
[[0,4], [1,3], [1,4], [2,2], [3,0], [3,1], [4,0]]
```

---

If you'd like to visualize the visited matrix evolution or animate the DFS waves reaching the oceans, we can turn it into a full-on simulation or even a HackerRank-style challenge with constraints and edge cases. Want to level this up?

'''