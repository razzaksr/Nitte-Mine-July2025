# Solution 1: Gaming Tournament Prize Pool Calculator
# Maximum coins collectible from top-left to any bottom row cell

def solve_challenge_1():
    # Read input
    m, n = map(int, input().split())
    grid = []
    for _ in range(m):
        row = list(map(int, input().split()))
        grid.append(row)
    
    # Initialize DP table
    dp = [[float('-inf')] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    
    # Fill first row
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    # Fill the DP table
    for i in range(1, m):
        for j in range(n):
            # From above
            dp[i][j] = dp[i-1][j] + grid[i][j]
            
            # From left
            if j > 0:
                dp[i][j] = max(dp[i][j], dp[i][j-1] + grid[i][j])
            
            # From diagonal (up-left)
            if j > 0:
                dp[i][j] = max(dp[i][j], dp[i-1][j-1] + grid[i][j])
    
    # Find maximum value in the last row
    result = max(dp[m-1])
    print(result)

# Solution 2: Construction Site Safety Grid Planner
# Count safe paths from top-left to bottom-right avoiding hazards

def solve_challenge_2():
    # Read input
    m, n = map(int, input().split())
    grid = []
    for _ in range(m):
        row = list(map(int, input().split()))
        grid.append(row)
    
    # If start or end is blocked, no paths possible
    if grid[0][0] == 1 or grid[m-1][n-1] == 1:
        print(0)
        return
    
    # Initialize DP table
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    
    # Fill first row
    for j in range(1, n):
        if grid[0][j] == 0:
            dp[0][j] = dp[0][j-1]
        else:
            dp[0][j] = 0
    
    # Fill first column
    for i in range(1, m):
        if grid[i][0] == 0:
            dp[i][0] = dp[i-1][0]
        else:
            dp[i][0] = 0
    
    # Fill the rest of the DP table
    for i in range(1, m):
        for j in range(1, n):
            if grid[i][j] == 0:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
            else:
                dp[i][j] = 0
    
    print(dp[m-1][n-1])

# Solution 3: Data Center Server Rack Configuration
# Count ways to place a×b rectangle in m×n grid (including rotations)

def solve_challenge_3():
    # Read input
    m, n, a, b = map(int, input().split())
    
    total_ways = 0
    
    # Count placements for a×b orientation
    if a <= m and b <= n:
        ways_horizontal = (m - a + 1) * (n - b + 1)
        total_ways += ways_horizontal
    
    # Count placements for b×a orientation (if different from a×b)
    if a != b and b <= m and a <= n:
        ways_vertical = (m - b + 1) * (n - a + 1)
        total_ways += ways_vertical
    
    print(total_ways)

# Main execution
if __name__ == "__main__":
    # Uncomment the challenge you want to run:
    
    # Challenge 1: Gaming Tournament Prize Pool Calculator
    # solve_challenge_1()
    
    # Challenge 2: Construction Site Safety Grid Planner
    # solve_challenge_2()
    
    # Challenge 3: Data Center Server Rack Configuration
    # solve_challenge_3()
    
    pass

# Test cases for verification:

# Challenge 1 Test Cases:
# Input: 3 3
#        1 2 3
#        4 5 6
#        7 8 9
# Expected Output: 21

# Input: 2 3
#        1 -2 3
#        4 5 -1
# Expected Output: 8

# Challenge 2 Test Cases:
# Input: 3 3
#        0 0 0
#        0 1 0
#        0 0 0
# Expected Output: 2

# Input: 2 2
#        0 0
#        0 0
# Expected Output: 2

# Challenge 3 Test Cases:
# Input: 3 3 2 2
# Expected Output: 4

# Input: 4 5 2 3
# Expected Output: 12