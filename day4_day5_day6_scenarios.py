"""
DSA Training Program - Day 4, Day 5 & Day 6 Real-Time Scenario Problems
Advanced Recursion, Memoization & Dynamic Programming Solutions

Real-world scenario-based problems with practical applications
"""

'''
DAY 5: Dynamic Programming Fundamentals
1. Stock Trading Analysis System 📈

Real-world Application: Financial trading, investment algorithms
Problem: Find maximum profit from buying and selling stocks once
Techniques: Kadane's algorithm variant, DP approach
Features: Tracks optimal buy/sell days, multiple solution approaches

2. Social Network Growth Analysis 📱

Real-world Application: Social media analytics, business intelligence
Problem: Find Longest Increasing Subsequence in user growth data
Techniques: Classic DP LIS, optimized binary search approach
Features: Predicts sustainable growth patterns, tracks actual growth sequence

3. Mobile Game Level Progression 🎮

Real-world Application: Game development, user engagement optimization
Problem: Find maximum reward path through game levels
Techniques: Jump Game variants, path reconstruction
Features: Reward optimization, obstacle detection, optimal path finding

4. Delivery Route Optimization 🚚

Real-world Application: GPS navigation, logistics optimization
Problem: Find minimum cost path in delivery grid (2D DP)
Techniques: Grid path problems, unique path counting
Features: Minimum cost calculation, route counting, actual path reconstruction

5. Message Decoding System 🔐

Real-world Application: Cryptography, security systems
Problem: Count ways to decode numeric string to letters
Techniques: Decode Ways DP, space optimization
Features: All possible decodings, optimized space complexity

Complete Coverage Now:
✅ Day 4: Advanced Recursion & Memoization (4 scenarios)
✅ Day 5: Dynamic Programming Fundamentals (5 scenarios) - NEW!
✅ Day 6: Advanced Dynamic Programming (4 scenarios)
Day 5 Key Features:
1D DP Problems:

Stock Trading: Maximum Subarray variant
Network Growth: Longest Increasing Subsequence
Game Levels: Jump Game with rewards
Message Decoding: Decode Ways problem

2D DP Introduction:

Delivery Routes: Grid path problems
Minimum Path Sum: Cost optimization
Unique Paths: Path counting

Multiple Solution Approaches:

Basic DP solutions
Space-optimized versions
Alternative algorithms (binary search for LIS)
Path reconstruction techniques


'''


from typing import List, Dict, Tuple
from functools import lru_cache


# ===============================
# DAY 4: ADVANCED RECURSION & INTRO TO DYNAMIC PROGRAMMING
# ===============================

class ATMCashDispenser:
    """
    Real-Time Scenario: ATM Cash Dispensing System
    
    Problem: An ATM has denominations [2000, 500, 200, 100, 50, 20, 10, 5, 1]
    A customer wants to withdraw amount X. Find the minimum number of notes
    required to dispense the exact amount.
    
    Real-world application: Banking systems, cash management
    """
    
    def __init__(self):
        self.denominations = [2000, 500, 200, 100, 50, 20, 10, 5, 1]
        self.memo = {}
    
    def min_notes_recursive(self, amount: int) -> int:
        """Basic recursive solution - inefficient"""
        if amount == 0:
            return 0
        if amount < 0:
            return float('inf')
        
        min_notes = float('inf')
        for denomination in self.denominations:
            if denomination <= amount:
                result = 1 + self.min_notes_recursive(amount - denomination)
                min_notes = min(min_notes, result)
        
        return min_notes
    
    def min_notes_memoized(self, amount: int) -> int:
        """Optimized solution with memoization"""
        if amount == 0:
            return 0
        if amount < 0:
            return float('inf')
        
        if amount in self.memo:
            return self.memo[amount]
        
        min_notes = float('inf')
        for denomination in self.denominations:
            if denomination <= amount:
                result = 1 + self.min_notes_memoized(amount - denomination)
                min_notes = min(min_notes, result)
        
        self.memo[amount] = min_notes
        return min_notes
    
    def get_note_breakdown(self, amount: int) -> Dict[int, int]:
        """Get actual breakdown of notes dispensed"""
        breakdown = {}
        remaining = amount
        
        for denomination in self.denominations:
            if remaining >= denomination:
                count = remaining // denomination
                breakdown[denomination] = count
                remaining -= count * denomination
        
        return breakdown


class TowerOfHanoiRobot:
    """
    Real-Time Scenario: Robotic Warehouse Management
    
    Problem: A robotic arm needs to move a stack of containers from one 
    position to another using an intermediate position. Only one container 
    can be moved at a time, and larger containers cannot be placed on smaller ones.
    
    Real-world application: Warehouse automation, robotic systems
    """
    
    def __init__(self):
        self.moves = []
        self.total_moves = 0
    
    def move_containers(self, n: int, source: str, destination: str, auxiliary: str) -> List[str]:
        """
        Move n containers from source to destination using auxiliary position
        Returns list of moves for the robot to execute
        """
        self.moves = []
        self.total_moves = 0
        self._hanoi_recursive(n, source, destination, auxiliary)
        return self.moves
    
    def _hanoi_recursive(self, n: int, source: str, destination: str, auxiliary: str):
        """Recursive solution for Tower of Hanoi"""
        if n == 1:
            move = f"Move container from {source} to {destination}"
            self.moves.append(move)
            self.total_moves += 1
            return
        
        # Move n-1 containers from source to auxiliary
        self._hanoi_recursive(n-1, source, auxiliary, destination)
        
        # Move the largest container from source to destination
        move = f"Move container from {source} to {destination}"
        self.moves.append(move)
        self.total_moves += 1
        
        # Move n-1 containers from auxiliary to destination
        self._hanoi_recursive(n-1, auxiliary, destination, source)
    
    def get_minimum_moves(self, n: int) -> int:
        """Calculate minimum moves required without generating the sequence"""
        return 2**n - 1


class StaircaseDesignOptimizer:
    """
    Real-Time Scenario: Building Construction Planning
    
    Problem: An architect needs to design a staircase with n steps. 
    A person can climb either 1 or 2 steps at a time. How many different 
    ways can the staircase be climbed? This helps in designing optimal 
    staircase layouts for emergency evacuation planning.
    
    Real-world application: Architecture, emergency planning, accessibility design
    """
    
    def __init__(self):
        self.memo = {}
    
    def climbing_ways_recursive(self, n: int) -> int:
        """Basic recursive solution - exponential time complexity"""
        if n <= 1:
            return 1
        if n == 2:
            return 2
        
        return self.climbing_ways_recursive(n-1) + self.climbing_ways_recursive(n-2)
    
    def climbing_ways_memoized(self, n: int) -> int:
        """Optimized solution with memoization"""
        if n <= 1:
            return 1
        if n == 2:
            return 2
        
        if n in self.memo:
            return self.memo[n]
        
        result = self.climbing_ways_memoized(n-1) + self.climbing_ways_memoized(n-2)
        self.memo[n] = result
        return result
    
    def climbing_ways_dp(self, n: int) -> int:
        """Dynamic Programming solution - bottom-up approach"""
        if n <= 1:
            return 1
        if n == 2:
            return 2
        
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1
        dp[2] = 2
        
        for i in range(3, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[n]


class HomeSecuritySystem:
    """
    Real-Time Scenario: Smart Home Security Optimization
    
    Problem: A smart home security system needs to determine the maximum 
    value of items that can be monitored without triggering adjacent sensors 
    (which would cause interference). Houses are arranged in a line, and 
    adjacent houses cannot be monitored simultaneously.
    
    Real-world application: IoT security systems, resource optimization
    """
    
    def __init__(self):
        self.memo = {}
    
    def max_monitored_value_recursive(self, house_values: List[int], index: int = 0) -> int:
        """Basic recursive solution"""
        if index >= len(house_values):
            return 0
        
        # Choice 1: Monitor current house + skip next
        monitor_current = house_values[index] + self.max_monitored_value_recursive(house_values, index + 2)
        
        # Choice 2: Skip current house
        skip_current = self.max_monitored_value_recursive(house_values, index + 1)
        
        return max(monitor_current, skip_current)
    
    def max_monitored_value_memoized(self, house_values: List[int]) -> int:
        """Optimized solution with memoization"""
        self.memo = {}
        return self._rob_memo(house_values, 0)
    
    def _rob_memo(self, house_values: List[int], index: int) -> int:
        if index >= len(house_values):
            return 0
        
        if index in self.memo:
            return self.memo[index]
        
        monitor_current = house_values[index] + self._rob_memo(house_values, index + 2)
        skip_current = self._rob_memo(house_values, index + 1)
        
        self.memo[index] = max(monitor_current, skip_current)
        return self.memo[index]


# ===============================
# DAY 5: DYNAMIC PROGRAMMING FUNDAMENTALS
# ===============================

class StockTradingAnalyzer:
    """
    Real-Time Scenario: Stock Market Trading Algorithm
    
    Problem: Find the maximum profit from buying and selling stocks.
    Given an array of stock prices, find the maximum profit that can be 
    achieved by buying and selling at most once. This is applied in 
    algorithmic trading systems.
    
    Real-world application: Financial trading, investment algorithms
    """
    
    def __init__(self):
        self.buy_day = 0
        self.sell_day = 0
    
    def max_profit_kadane(self, prices: List[int]) -> int:
        """
        Maximum Subarray Sum approach (Kadane's algorithm variant)
        Convert to profit differences array
        """
        if len(prices) < 2:
            return 0
        
        max_profit = 0
        current_profit = 0
        
        for i in range(1, len(prices)):
            profit_today = prices[i] - prices[i-1]
            current_profit = max(profit_today, current_profit + profit_today)
            max_profit = max(max_profit, current_profit)
        
        return max_profit
    
    def max_profit_dp(self, prices: List[int]) -> int:
        """
        Dynamic Programming approach
        Track minimum price and maximum profit
        """
        if len(prices) < 2:
            return 0
        
        min_price = prices[0]
        max_profit = 0
        
        for i in range(1, len(prices)):
            if prices[i] < min_price:
                min_price = prices[i]
                self.buy_day = i
            
            current_profit = prices[i] - min_price
            if current_profit > max_profit:
                max_profit = current_profit
                self.sell_day = i
        
        return max_profit
    
    def get_trading_days(self) -> tuple:
        """Get optimal buy and sell days"""
        return (self.buy_day, self.sell_day)


class NetworkGrowthPredictor:
    """
    Real-Time Scenario: Social Network Growth Analysis
    
    Problem: Find the Longest Increasing Subsequence in user growth data.
    This helps predict sustainable growth patterns and identify periods
    of consistent user acquisition.
    
    Real-world application: Social media analytics, business intelligence
    """
    
    def __init__(self):
        self.lis_sequence = []
    
    def predict_growth_pattern_dp(self, daily_users: List[int]) -> int:
        """
        Find longest increasing subsequence using DP
        Represents longest period of consistent growth
        """
        if not daily_users:
            return 0
        
        n = len(daily_users)
        dp = [1] * n
        parent = [-1] * n
        
        for i in range(1, n):
            for j in range(i):
                if daily_users[j] < daily_users[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
        
        # Find the index with maximum LIS length
        max_length = max(dp)
        max_index = dp.index(max_length)
        
        # Reconstruct the LIS
        self.lis_sequence = []
        current = max_index
        while current != -1:
            self.lis_sequence.append(daily_users[current])
            current = parent[current]
        
        self.lis_sequence.reverse()
        return max_length
    
    def predict_growth_pattern_optimized(self, daily_users: List[int]) -> int:
        """
        Optimized LIS using binary search (O(n log n))
        """
        if not daily_users:
            return 0
        
        from bisect import bisect_left
        
        tails = []
        
        for user_count in daily_users:
            pos = bisect_left(tails, user_count)
            if pos == len(tails):
                tails.append(user_count)
            else:
                tails[pos] = user_count
        
        return len(tails)
    
    def get_growth_sequence(self) -> List[int]:
        """Get the actual growth sequence"""
        return self.lis_sequence


class GameLevelOptimizer:
    """
    Real-Time Scenario: Mobile Game Level Progression
    
    Problem: A player can jump 1 or 2 levels at a time. Each level has
    a certain reward. Find the maximum reward path to reach the final level.
    This is used in game design to balance difficulty and rewards.
    
    Real-world application: Game development, user engagement optimization
    """
    
    def __init__(self):
        self.optimal_path = []
    
    def max_reward_path(self, level_rewards: List[int]) -> int:
        """
        Find maximum reward path using DP
        Similar to Jump Game with rewards
        """
        if not level_rewards:
            return 0
        
        n = len(level_rewards)
        if n == 1:
            return level_rewards[0]
        
        # dp[i] = maximum reward to reach level i
        dp = [0] * n
        dp[0] = level_rewards[0]
        dp[1] = max(level_rewards[0], level_rewards[1])
        
        for i in range(2, n):
            # Either come from previous level or jump from 2 levels back
            dp[i] = max(dp[i-1], dp[i-2] + level_rewards[i])
        
        return dp[n-1]
    
    def can_reach_final_level(self, level_obstacles: List[int]) -> bool:
        """
        Check if player can reach final level
        Jump Game variant: 0 = obstacle, positive = max jump distance
        """
        if not level_obstacles:
            return True
        
        n = len(level_obstacles)
        max_reach = 0
        
        for i in range(n):
            if i > max_reach:
                return False
            max_reach = max(max_reach, i + level_obstacles[i])
            if max_reach >= n - 1:
                return True
        
        return max_reach >= n - 1
    
    def find_optimal_path(self, level_rewards: List[int]) -> List[int]:
        """
        Find the actual path taken for maximum reward
        """
        if not level_rewards:
            return []
        
        n = len(level_rewards)
        if n == 1:
            return [0]
        
        dp = [0] * n
        path = [0] * n
        
        dp[0] = level_rewards[0]
        dp[1] = max(level_rewards[0], level_rewards[1])
        path[1] = 1 if level_rewards[1] > level_rewards[0] else 0
        
        for i in range(2, n):
            if dp[i-1] > dp[i-2] + level_rewards[i]:
                dp[i] = dp[i-1]
                path[i] = 0  # Didn't take current level
            else:
                dp[i] = dp[i-2] + level_rewards[i]
                path[i] = 1  # Took current level
        
        # Reconstruct path
        optimal_path = []
        i = n - 1
        while i >= 0:
            if path[i] == 1:
                optimal_path.append(i)
                i -= 2
            else:
                i -= 1
        
        return list(reversed(optimal_path))


class DeliveryRouteOptimizer:
    """
    Real-Time Scenario: Delivery Route Optimization
    
    Problem: Find the minimum cost path from top-left to bottom-right
    in a delivery grid. Each cell represents the fuel cost for that area.
    This is used in logistics and navigation systems.
    
    Real-world application: GPS navigation, logistics optimization, delivery services
    """
    
    def __init__(self):
        self.optimal_route = []
    
    def find_minimum_delivery_cost(self, cost_grid: List[List[int]]) -> int:
        """
        Find minimum cost path using 2D DP
        Can only move right or down
        """
        if not cost_grid or not cost_grid[0]:
            return 0
        
        rows, cols = len(cost_grid), len(cost_grid[0])
        dp = [[0] * cols for _ in range(rows)]
        
        # Initialize first cell
        dp[0][0] = cost_grid[0][0]
        
        # Initialize first row (can only come from left)
        for j in range(1, cols):
            dp[0][j] = dp[0][j-1] + cost_grid[0][j]
        
        # Initialize first column (can only come from top)
        for i in range(1, rows):
            dp[i][0] = dp[i-1][0] + cost_grid[i][0]
        
        # Fill the rest of the table
        for i in range(1, rows):
            for j in range(1, cols):
                dp[i][j] = cost_grid[i][j] + min(dp[i-1][j], dp[i][j-1])
        
        return dp[rows-1][cols-1]
    
    def count_delivery_routes(self, rows: int, cols: int) -> int:
        """
        Count unique paths from top-left to bottom-right
        Can only move right or down
        """
        if rows <= 0 or cols <= 0:
            return 0
        
        dp = [[1] * cols for _ in range(rows)]
        
        for i in range(1, rows):
            for j in range(1, cols):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[rows-1][cols-1]
    
    def find_optimal_route(self, cost_grid: List[List[int]]) -> List[tuple]:
        """
        Find the actual route taken for minimum cost
        """
        if not cost_grid or not cost_grid[0]:
            return []
        
        rows, cols = len(cost_grid), len(cost_grid[0])
        dp = [[0] * cols for _ in range(rows)]
        
        # Fill DP table (same as minimum cost)
        dp[0][0] = cost_grid[0][0]
        
        for j in range(1, cols):
            dp[0][j] = dp[0][j-1] + cost_grid[0][j]
        
        for i in range(1, rows):
            dp[i][0] = dp[i-1][0] + cost_grid[i][0]
        
        for i in range(1, rows):
            for j in range(1, cols):
                dp[i][j] = cost_grid[i][j] + min(dp[i-1][j], dp[i][j-1])
        
        # Backtrack to find path
        route = []
        i, j = rows - 1, cols - 1
        
        while i > 0 or j > 0:
            route.append((i, j))
            
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                if dp[i-1][j] < dp[i][j-1]:
                    i -= 1
                else:
                    j -= 1
        
        route.append((0, 0))
        return list(reversed(route))


class CryptographyDecoder:
    """
    Real-Time Scenario: Message Decoding System
    
    Problem: Decode a numeric string where 'A'=1, 'B'=2, ..., 'Z'=26.
    Find the total number of ways to decode the string. This is used
    in cryptography and security systems.
    
    Real-world application: Cryptography, security systems, data encoding
    """
    
    def __init__(self):
        self.decode_ways_memo = {}
    
    def count_decode_ways(self, encoded_message: str) -> int:
        """
        Count number of ways to decode the message using DP
        """
        if not encoded_message:
            return 0
        
        n = len(encoded_message)
        dp = [0] * (n + 1)
        dp[0] = 1  # Empty string has 1 way to decode
        dp[1] = 1 if encoded_message[0] != '0' else 0
        
        for i in range(2, n + 1):
            # Single digit decoding
            if encoded_message[i-1] != '0':
                dp[i] += dp[i-1]
            
            # Two digit decoding
            two_digit = int(encoded_message[i-2:i])
            if 10 <= two_digit <= 26:
                dp[i] += dp[i-2]
        
        return dp[n]
    
    def count_decode_ways_optimized(self, encoded_message: str) -> int:
        """
        Space-optimized version using only two variables
        """
        if not encoded_message:
            return 0
        
        n = len(encoded_message)
        if n == 0:
            return 0
        
        prev2 = 1  # dp[i-2]
        prev1 = 1 if encoded_message[0] != '0' else 0  # dp[i-1]
        
        for i in range(2, n + 1):
            current = 0
            
            # Single digit
            if encoded_message[i-1] != '0':
                current += prev1
            
            # Two digit
            two_digit = int(encoded_message[i-2:i])
            if 10 <= two_digit <= 26:
                current += prev2
            
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    def get_all_decodings(self, encoded_message: str) -> List[str]:
        """
        Get all possible decoded messages
        """
        if not encoded_message:
            return []
        
        def backtrack(index: int, current_decode: str) -> List[str]:
            if index == len(encoded_message):
                return [current_decode]
            
            result = []
            
            # Single digit
            if encoded_message[index] != '0':
                single_digit = int(encoded_message[index])
                char = chr(ord('A') + single_digit - 1)
                result.extend(backtrack(index + 1, current_decode + char))
            
            # Two digit
            if index + 1 < len(encoded_message):
                two_digit = int(encoded_message[index:index+2])
                if 10 <= two_digit <= 26:
                    char = chr(ord('A') + two_digit - 1)
                    result.extend(backtrack(index + 2, current_decode + char))
            
            return result
        
        return backtrack(0, "")


# ===============================
# DAY 6: ADVANCED DYNAMIC PROGRAMMING
# ===============================

class DNASequenceAnalyzer:
    """
    Real-Time Scenario: Bioinformatics - DNA Sequence Analysis
    
    Problem: Find the Longest Common Subsequence (LCS) between two DNA sequences.
    This helps in identifying genetic similarities, evolutionary relationships,
    and mutations in genetic research.
    
    Real-world application: Genetics, medical research, evolutionary biology
    """
    
    def __init__(self):
        self.memo = {}
    
    def find_lcs_length(self, dna1: str, dna2: str) -> int:
        """Find length of Longest Common Subsequence"""
        m, n = len(dna1), len(dna2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if dna1[i-1] == dna2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def find_lcs_sequence(self, dna1: str, dna2: str) -> str:
        """Find the actual LCS sequence"""
        m, n = len(dna1), len(dna2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if dna1[i-1] == dna2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Backtrack to find the sequence
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if dna1[i-1] == dna2[j-1]:
                lcs.append(dna1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
        
        return ''.join(reversed(lcs))
    
    def calculate_similarity_percentage(self, dna1: str, dna2: str) -> float:
        """Calculate similarity percentage between two DNA sequences"""
        lcs_length = self.find_lcs_length(dna1, dna2)
        max_length = max(len(dna1), len(dna2))
        return (lcs_length / max_length) * 100 if max_length > 0 else 0


class CorporateKnapsackOptimizer:
    """
    Real-Time Scenario: Corporate Resource Allocation
    
    Problem: A company has a limited budget and needs to select projects 
    that maximize profit. Each project has a cost and expected profit.
    This is a classic 0/1 Knapsack problem applied to business scenarios.
    
    Real-world application: Project management, investment decisions, resource allocation
    """
    
    def __init__(self):
        self.selected_projects = []
    
    def maximize_profit(self, projects: List[Tuple[str, int, int]], budget: int) -> int:
        """
        Maximize profit within budget constraint
        projects: List of (project_name, cost, profit)
        budget: Available budget
        """
        n = len(projects)
        # Extract costs and profits
        costs = [project[1] for project in projects]
        profits = [project[2] for project in projects]
        
        # DP table: dp[i][w] = maximum profit using first i projects with budget w
        dp = [[0] * (budget + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(1, budget + 1):
                # Don't include current project
                dp[i][w] = dp[i-1][w]
                
                # Include current project if budget allows
                if costs[i-1] <= w:
                    dp[i][w] = max(dp[i][w], dp[i-1][w - costs[i-1]] + profits[i-1])
        
        # Find selected projects
        self._find_selected_projects(dp, projects, costs, budget)
        
        return dp[n][budget]
    
    def _find_selected_projects(self, dp: List[List[int]], projects: List[Tuple[str, int, int]], 
                               costs: List[int], budget: int):
        """Find which projects were selected"""
        self.selected_projects = []
        n = len(projects)
        w = budget
        
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                self.selected_projects.append(projects[i-1])
                w -= costs[i-1]
        
        self.selected_projects.reverse()
    
    def get_selected_projects(self) -> List[Tuple[str, int, int]]:
        """Get list of selected projects"""
        return self.selected_projects


class DocumentEditingSystem:
    """
    Real-Time Scenario: Document Version Control & Auto-Correction
    
    Problem: Calculate the minimum number of operations (insertions, deletions, 
    substitutions) needed to transform one document version to another.
    This is used in version control systems and auto-correction features.
    
    Real-world application: Text editors, version control, spell checkers
    """
    
    def __init__(self):
        self.operations = []
    
    def calculate_edit_distance(self, text1: str, text2: str) -> int:
        """Calculate minimum edit distance between two texts"""
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i  # Delete all characters
        for j in range(n + 1):
            dp[0][j] = j  # Insert all characters
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # No operation needed
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # Delete
                        dp[i][j-1],      # Insert
                        dp[i-1][j-1]     # Substitute
                    )
        
        return dp[m][n]
    
    def get_edit_operations(self, text1: str, text2: str) -> List[str]:
        """Get detailed list of edit operations"""
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table (same as above)
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        # Backtrack to find operations
        operations = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and text1[i-1] == text2[j-1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                operations.append(f"Substitute '{text1[i-1]}' with '{text2[j-1]}' at position {i-1}")
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                operations.append(f"Delete '{text1[i-1]}' at position {i-1}")
                i -= 1
            else:
                operations.append(f"Insert '{text2[j-1]}' at position {i}")
                j -= 1
        
        return list(reversed(operations))


class CryptocurrencyTradingBot:
    """
    Real-Time Scenario: Cryptocurrency Trading Strategy
    
    Problem: Find all palindromic subsequences in price patterns to identify 
    potential trading opportunities. Palindromic patterns often indicate 
    market reversals or consolidation phases.
    
    Real-world application: Algorithmic trading, pattern recognition, financial analysis
    """
    
    def __init__(self):
        self.palindromic_patterns = []
    
    def count_palindromic_subsequences(self, price_sequence: str) -> int:
        """Count all palindromic subsequences in price pattern"""
        n = len(price_sequence)
        if n == 0:
            return 0
        
        # dp[i][j] = number of palindromic subsequences in sequence[i:j+1]
        dp = [[0] * n for _ in range(n)]
        
        # Every single character is a palindrome
        for i in range(n):
            dp[i][i] = 1
        
        # Fill for length 2 to n
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                if price_sequence[i] == price_sequence[j]:
                    dp[i][j] = dp[i+1][j] + dp[i][j-1] + 1
                else:
                    dp[i][j] = dp[i+1][j] + dp[i][j-1] - dp[i+1][j-1]
        
        return dp[0][n-1]
    
    def find_longest_palindromic_subsequence(self, price_sequence: str) -> str:
        """Find the longest palindromic subsequence"""
        n = len(price_sequence)
        if n == 0:
            return ""
        
        dp = [[0] * n for _ in range(n)]
        
        # Every single character is a palindrome of length 1
        for i in range(n):
            dp[i][i] = 1
        
        # Fill the table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                if price_sequence[i] == price_sequence[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        
        # Reconstruct the longest palindromic subsequence
        return self._reconstruct_lps(price_sequence, dp, 0, n-1)
    
    def _reconstruct_lps(self, s: str, dp: List[List[int]], i: int, j: int) -> str:
        """Reconstruct the longest palindromic subsequence"""
        if i > j:
            return ""
        if i == j:
            return s[i]
        
        if s[i] == s[j]:
            return s[i] + self._reconstruct_lps(s, dp, i+1, j-1) + s[j]
        elif dp[i+1][j] > dp[i][j-1]:
            return self._reconstruct_lps(s, dp, i+1, j)
        else:
            return self._reconstruct_lps(s, dp, i, j-1)


# ===============================
# TESTING AND EXAMPLES
# ===============================

def test_day4_scenarios():
    """Test Day 4 scenario-based problems"""
    print("=" * 60)
    print("DAY 4 TESTING: ADVANCED RECURSION & MEMOIZATION")
    print("=" * 60)
    
    # Test ATM Cash Dispenser
    print("\n1. ATM Cash Dispenser System")
    print("-" * 40)
    atm = ATMCashDispenser()
    amount = 2750
    print(f"Amount to withdraw: ₹{amount}")
    print(f"Minimum notes (recursive): {atm.min_notes_recursive(amount)}")
    print(f"Minimum notes (memoized): {atm.min_notes_memoized(amount)}")
    print(f"Note breakdown: {atm.get_note_breakdown(amount)}")
    
    # Test Tower of Hanoi Robot
    print("\n2. Robotic Warehouse Container Management")
    print("-" * 40)
    robot = TowerOfHanoiRobot()
    containers = 3
    moves = robot.move_containers(containers, "Position A", "Position C", "Position B")
    print(f"Moving {containers} containers:")
    for i, move in enumerate(moves[:5], 1):  # Show first 5 moves
        print(f"  Step {i}: {move}")
    print(f"Total moves required: {robot.total_moves}")
    
    # Test Staircase Design
    print("\n3. Staircase Design Optimizer")
    print("-" * 40)
    staircase = StaircaseDesignOptimizer()
    steps = 10
    print(f"Staircase with {steps} steps:")
    print(f"Ways to climb (recursive): {staircase.climbing_ways_recursive(steps)}")
    print(f"Ways to climb (memoized): {staircase.climbing_ways_memoized(steps)}")
    print(f"Ways to climb (DP): {staircase.climbing_ways_dp(steps)}")
    
    # Test Home Security System
    print("\n4. Smart Home Security System")
    print("-" * 40)
    security = HomeSecuritySystem()
    house_values = [100, 200, 300, 400, 500]
    print(f"House monitoring values: {house_values}")
    print(f"Maximum monitoring value (recursive): {security.max_monitored_value_recursive(house_values)}")
    print(f"Maximum monitoring value (memoized): {security.max_monitored_value_memoized(house_values)}")


def test_day6_scenarios():
    """Test Day 6 scenario-based problems"""
    print("\n" + "=" * 60)
    print("DAY 6 TESTING: ADVANCED DYNAMIC PROGRAMMING")
    print("=" * 60)
    
    # Test DNA Sequence Analyzer
    print("\n1. DNA Sequence Analysis System")
    print("-" * 40)
    dna_analyzer = DNASequenceAnalyzer()
    dna1 = "AGGTAB"
    dna2 = "GXTXAYB"
    print(f"DNA Sequence 1: {dna1}")
    print(f"DNA Sequence 2: {dna2}")
    print(f"LCS Length: {dna_analyzer.find_lcs_length(dna1, dna2)}")
    print(f"LCS Sequence: {dna_analyzer.find_lcs_sequence(dna1, dna2)}")
    print(f"Similarity: {dna_analyzer.calculate_similarity_percentage(dna1, dna2):.1f}%")
    
    # Test Corporate Knapsack Optimizer
    print("\n2. Corporate Resource Allocation System")
    print("-" * 40)
    optimizer = CorporateKnapsackOptimizer()
    projects = [
        ("AI Development", 10000, 15000),
        ("Mobile App", 8000, 12000),
        ("Website Redesign", 5000, 8000),
        ("Data Analytics", 12000, 18000),
        ("Cloud Migration", 15000, 20000)
    ]
    budget = 20000
    print(f"Available Budget: ₹{budget}")
    print("Available Projects:")
    for name, cost, profit in projects:
        print(f"  {name}: Cost ₹{cost}, Profit ₹{profit}")
    
    max_profit = optimizer.maximize_profit(projects, budget)
    print(f"\nMaximum Profit: ₹{max_profit}")
    print("Selected Projects:")
    for name, cost, profit in optimizer.get_selected_projects():
        print(f"  {name}: Cost ₹{cost}, Profit ₹{profit}")
    
    # Test Document Editing System
    print("\n3. Document Version Control System")
    print("-" * 40)
    editor = DocumentEditingSystem()
    text1 = "algorithm"
    text2 = "altruistic"
    print(f"Original text: '{text1}'")
    print(f"Target text: '{text2}'")
    print(f"Edit distance: {editor.calculate_edit_distance(text1, text2)}")
    print("Edit operations:")
    for op in editor.get_edit_operations(text1, text2):
        print(f"  {op}")
    
    # Test Cryptocurrency Trading Bot
    print("\n4. Cryptocurrency Trading Pattern Analysis")
    print("-" * 40)
    trading_bot = CryptocurrencyTradingBot()
    price_pattern = "HLHLHHL"  # H=High, L=Low
    print(f"Price pattern: {price_pattern}")
    print(f"Palindromic patterns count: {trading_bot.count_palindromic_subsequences(price_pattern)}")
    print(f"Longest palindromic pattern: {trading_bot.find_longest_palindromic_subsequence(price_pattern)}")


if __name__ == "__main__":
    test_day4_scenarios()
    test_day6_scenarios()
