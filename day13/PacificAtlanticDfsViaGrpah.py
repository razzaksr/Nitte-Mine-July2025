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