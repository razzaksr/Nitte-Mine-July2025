"""
DSA Training Program - Day 4 & Day 6 Real-Time Scenario Problems
Advanced Recursion, Memoization & Dynamic Programming Solutions

Real-world scenario-based problems with practical applications
"""


'''
DAY 4: Advanced Recursion & Memoization
1. ATM Cash Dispenser System 🏧

Real-world Application: Banking systems, cash management
Problem: Minimize number of currency notes for withdrawal
Techniques: Recursive solution, memoization optimization
Denominations: [2000, 500, 200, 100, 50, 20, 10, 5, 1]

2. Tower of Hanoi Robot 🤖

Real-world Application: Warehouse automation, robotic systems
Problem: Move containers using robotic arm with constraints
Techniques: Classic recursion, move sequence generation
Output: Step-by-step robot instructions

3. Staircase Design Optimizer 🏗️

Real-world Application: Architecture, emergency planning
Problem: Calculate climbing ways for staircase design
Techniques: Fibonacci-like recursion, memoization, DP
Applications: Accessibility design, evacuation planning

4. Smart Home Security System 🏠

Real-world Application: IoT security, resource optimization
Problem: Maximize monitoring value without adjacent interference
Techniques: House robber problem variant, recursion + memoization

DAY 6: Advanced Dynamic Programming
1. DNA Sequence Analyzer 🧬

Real-world Application: Bioinformatics, genetic research
Problem: Find Longest Common Subsequence in DNA sequences
Techniques: 2D DP, sequence reconstruction
Output: Similarity percentage, actual LCS sequence

2. Corporate Resource Allocation 💼

Real-world Application: Project management, investment decisions
Problem: Maximize profit within budget constraints
Techniques: 0/1 Knapsack problem, project selection
Output: Selected projects with cost-benefit analysis

3. Document Version Control System 📄

Real-world Application: Text editors, version control systems
Problem: Calculate minimum edit operations between documents
Techniques: Edit distance (Levenshtein), operation tracking
Output: Detailed edit operations list

4. Cryptocurrency Trading Bot 📈

Real-world Application: Algorithmic trading, pattern recognition
Problem: Find palindromic patterns in price sequences
Techniques: Palindromic subsequence DP, pattern analysis
Output: Trading signals based on palindromic patterns
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
