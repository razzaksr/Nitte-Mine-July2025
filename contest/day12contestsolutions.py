# 1. Surrounded Regions
class Solution:
    def solve(self, board):
        if not board or not board[0]:
            return

        m, n = len(board), len(board[0])

        def dfs(r, c):
            if r < 0 or r >= m or c < 0 or c >= n or board[r][c] != 'O':
                return
            board[r][c] = 'S'  # Mark as Safe
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                dfs(r + dr, c + dc)

        # Start DFS from border 'O's
        for i in range(m):
            dfs(i, 0)
            dfs(i, n - 1)
        for j in range(n):
            dfs(0, j)
            dfs(m - 1, j)

        # Flip surrounded 'O's to 'X', and safe 'S' back to 'O'
        for r in range(m):
            for c in range(n):
                if board[r][c] == 'O':
                    board[r][c] = 'X'
                elif board[r][c] == 'S':
                    board[r][c] = 'O'
board = [["X","X","X","X"],
         ["X","O","O","X"],
         ["X","X","O","X"],
         ["X","O","X","X"]]

sol = Solution()
sol.solve(board)
# print(board)
# Output:
# [
#  ["X","X","X","X"],
#  ["X","X","X","X"],
#  ["X","X","X","X"],
#  ["X","O","X","X"]
# ]

# 2. Course Schedule
class Solution:
    def canFinish(self, numCourses, prerequisites):
        from collections import defaultdict

        graph = defaultdict(list)
        for a, b in prerequisites:
            graph[b].append(a)

        visited = set()
        done = set()

        def dfs(course):
            if course in visited:
                return False  # cycle detected
            if course in done:
                return True   # already checked

            visited.add(course)
            for neighbor in graph[course]:
                if not dfs(neighbor):
                    return False
            visited.remove(course)
            done.add(course)
            return True

        for course in range(numCourses):
            if not dfs(course):
                return False
        return True

numCourses = 2
prerequisites = [[1, 0]]
sol = Solution()
# print(sol.canFinish(numCourses, prerequisites))  # Output: True

numCourses = 2
prerequisites = [[1, 0], [0, 1]]
sol = Solution()
# print(sol.canFinish(numCourses, prerequisites))  # Output: False

# Number of Islands
class Solution:
    def numIslands(self, grid):
        if not grid: return 0

        m, n = len(grid), len(grid[0])
        count = 0

        def dfs(r, c):
            if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != '1': return
            grid[r][c] = '0'  # Mark visited
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                dfs(r + dr, c + dc)

        for r in range(m):
            for c in range(n):
                if grid[r][c] == '1':
                    dfs(r, c)
                    count += 1

        return count
grid = [
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"]
]

sol = Solution()
print(sol.numIslands(grid))  # Output: 3