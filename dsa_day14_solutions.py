"""
DSA Training Program - Day 14 Complete Solutions
Problem Solving & DSA Training - Final Assessment

This module contains well-structured solutions for Day 14 topics:
1. Trie (Prefix Tree) Implementation
2. Word Search II
3. Mixed problems from all previous topics
"""

import heapq
from collections import defaultdict, deque
from typing import List, Optional, Dict, Set


class TrieNode:
    """Node class for Trie data structure"""
    
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word = None  # Store the word for Word Search II


class Trie:
    """
    Trie (Prefix Tree) implementation
    Supports insert, search, and startsWith operations
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """Insert a word into the trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.word = word
    
    def search(self, word: str) -> bool:
        """Search for a word in the trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with the given prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


class WordSearchII:
    """Solution for Word Search II problem using Trie"""
    
    def __init__(self):
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.result = []
    
    def find_words(self, board: List[List[str]], words: List[str]) -> List[str]:
        """Find all words that can be constructed from letters in the board"""
        if not board or not board[0] or not words:
            return []
        
        # Build trie from words
        trie = Trie()
        for word in words:
            trie.insert(word)
        
        self.result = []
        rows, cols = len(board), len(board[0])
        
        # Start DFS from each cell
        for i in range(rows):
            for j in range(cols):
                self.dfs_word_search(board, i, j, trie.root, rows, cols)
        
        return self.result
    
    def dfs_word_search(self, board: List[List[str]], row: int, col: int, 
                       node: TrieNode, rows: int, cols: int) -> None:
        """DFS to find words in the board"""
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return
        
        char = board[row][col]
        if char not in node.children:
            return
        
        node = node.children[char]
        
        # Found a word
        if node.is_end_of_word:
            self.result.append(node.word)
            node.is_end_of_word = False  # Avoid duplicates
        
        # Mark as visited
        board[row][col] = '#'
        
        # Explore all directions
        for dr, dc in self.directions:
            self.dfs_word_search(board, row + dr, col + dc, node, rows, cols)
        
        # Backtrack
        board[row][col] = char


class DynamicProgrammingSolutions:
    """Solutions for Dynamic Programming problems"""
    
    def coin_change(self, coins: List[int], amount: int) -> int:
        """Coin Change problem using DP"""
        if amount == 0:
            return 0
        
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1
    
    def longest_increasing_subsequence(self, nums: List[int]) -> int:
        """Longest Increasing Subsequence using DP"""
        if not nums:
            return 0
        
        dp = [1] * len(nums)
        
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    def knapsack_01(self, weights: List[int], values: List[int], capacity: int) -> int:
        """0/1 Knapsack problem using DP"""
        n = len(weights)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(
                        dp[i-1][w],
                        dp[i-1][w - weights[i-1]] + values[i-1]
                    )
                else:
                    dp[i][w] = dp[i-1][w]
        
        return dp[n][capacity]
    
    def edit_distance(self, word1: str, word2: str) -> int:
        """Edit Distance (Levenshtein Distance) using DP"""
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # deletion
                        dp[i][j-1],      # insertion
                        dp[i-1][j-1]     # substitution
                    )
        
        return dp[m][n]


class GraphAlgorithms:
    """Solutions for Graph problems"""
    
    def __init__(self):
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def number_of_islands(self, grid: List[List[str]]) -> int:
        """Count number of islands using DFS"""
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        islands = 0
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '1':
                    self.dfs_island(grid, i, j, rows, cols)
                    islands += 1
        
        return islands
    
    def dfs_island(self, grid: List[List[str]], row: int, col: int, 
                   rows: int, cols: int) -> None:
        """DFS to mark connected land cells"""
        if (row < 0 or row >= rows or col < 0 or col >= cols or 
            grid[row][col] != '1'):
            return
        
        grid[row][col] = '0'  # Mark as visited
        
        for dr, dc in self.directions:
            self.dfs_island(grid, row + dr, col + dc, rows, cols)
    
    def course_schedule(self, num_courses: int, prerequisites: List[List[int]]) -> bool:
        """Check if all courses can be finished (cycle detection)"""
        graph = defaultdict(list)
        in_degree = [0] * num_courses
        
        # Build graph and calculate in-degrees
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        # Topological sort using BFS
        queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
        completed = 0
        
        while queue:
            course = queue.popleft()
            completed += 1
            
            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        return completed == num_courses
    
    def dijkstra_shortest_path(self, graph: Dict[int, List[tuple]], 
                              start: int, end: int) -> int:
        """Dijkstra's algorithm for shortest path"""
        distances = defaultdict(lambda: float('inf'))
        distances[start] = 0
        pq = [(0, start)]
        
        while pq:
            curr_dist, node = heapq.heappop(pq)
            
            if node == end:
                return curr_dist
            
            if curr_dist > distances[node]:
                continue
            
            for neighbor, weight in graph[node]:
                distance = curr_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
        
        return distances[end] if distances[end] != float('inf') else -1


class LinkedListSolutions:
    """Solutions for Linked List problems"""
    
    def reverse_linked_list(self, head: Optional['ListNode']) -> Optional['ListNode']:
        """Reverse a linked list iteratively"""
        prev = None
        current = head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        return prev
    
    def has_cycle(self, head: Optional['ListNode']) -> bool:
        """Detect cycle in linked list using Floyd's algorithm"""
        if not head or not head.next:
            return False
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        
        return False
    
    def merge_two_sorted_lists(self, l1: Optional['ListNode'], 
                             l2: Optional['ListNode']) -> Optional['ListNode']:
        """Merge two sorted linked lists"""
        dummy = ListNode(0)
        current = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next
        
        # Attach remaining nodes
        current.next = l1 or l2
        
        return dummy.next


class TreeSolutions:
    """Solutions for Tree problems"""
    
    def max_depth(self, root: Optional['TreeNode']) -> int:
        """Maximum depth of binary tree"""
        if not root:
            return 0
        
        return 1 + max(self.max_depth(root.left), self.max_depth(root.right))
    
    def is_valid_bst(self, root: Optional['TreeNode']) -> bool:
        """Validate if tree is a valid BST"""
        return self.validate_bst(root, float('-inf'), float('inf'))
    
    def validate_bst(self, node: Optional['TreeNode'], 
                    min_val: float, max_val: float) -> bool:
        """Helper function to validate BST"""
        if not node:
            return True
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        return (self.validate_bst(node.left, min_val, node.val) and 
                self.validate_bst(node.right, node.val, max_val))
    
    def level_order_traversal(self, root: Optional['TreeNode']) -> List[List[int]]:
        """Level order traversal of binary tree"""
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            current_level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(current_level)
        
        return result


class ArrayAndStringSolutions:
    """Solutions for Array and String problems"""
    
    def three_sum(self, nums: List[int]) -> List[List[int]]:
        """Find all unique triplets that sum to zero"""
        nums.sort()
        result = []
        
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            
            left, right = i + 1, len(nums) - 1
            
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                
                if total == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif total < 0:
                    left += 1
                else:
                    right -= 1
        
        return result
    
    def max_area(self, height: List[int]) -> int:
        """Container with most water"""
        left, right = 0, len(height) - 1
        max_water = 0
        
        while left < right:
            width = right - left
            water = min(height[left], height[right]) * width
            max_water = max(max_water, water)
            
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return max_water
    
    def longest_substring_without_repeating(self, s: str) -> int:
        """Longest substring without repeating characters"""
        char_map = {}
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            if s[right] in char_map and char_map[s[right]] >= left:
                left = char_map[s[right]] + 1
            
            char_map[s[right]] = right
            max_length = max(max_length, right - left + 1)
        
        return max_length


class BitManipulationSolutions:
    """Solutions for Bit Manipulation problems"""
    
    def single_number(self, nums: List[int]) -> int:
        """Find single number using XOR"""
        result = 0
        for num in nums:
            result ^= num
        return result
    
    def counting_bits(self, n: int) -> List[int]:
        """Count number of 1s in binary representation"""
        result = [0] * (n + 1)
        
        for i in range(1, n + 1):
            result[i] = result[i >> 1] + (i & 1)
        
        return result
    
    def is_power_of_two(self, n: int) -> bool:
        """Check if number is power of two"""
        return n > 0 and (n & (n - 1)) == 0


# Helper classes for LinkedList and Tree problems
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Example usage and test cases
def run_day14_examples():
    """Test examples for Day 14 solutions"""
    
    # Test Trie
    print("=== Trie Implementation Test ===")
    trie = Trie()
    trie.insert("apple")
    trie.insert("app")
    print(f"Search 'app': {trie.search('app')}")  # True
    print(f"Search 'apple': {trie.search('apple')}")  # True
    print(f"Starts with 'app': {trie.starts_with('app')}")  # True
    
    # Test Word Search II
    print("\n=== Word Search II Test ===")
    ws = WordSearchII()
    board = [['o','a','a','n'],['e','t','a','e'],['i','h','k','r'],['i','f','l','v']]
    words = ["oath","pea","eat","rain"]
    result = ws.find_words(board, words)
    print(f"Found words: {result}")
    
    # Test Dynamic Programming
    print("\n=== Dynamic Programming Test ===")
    dp = DynamicProgrammingSolutions()
    coins = [1, 2, 5]
    amount = 11
    print(f"Coin change for {amount}: {dp.coin_change(coins, amount)}")
    
    # Test Graph Algorithms
    print("\n=== Graph Algorithms Test ===")
    graph_algo = GraphAlgorithms()
    grid = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]
    print(f"Number of islands: {graph_algo.number_of_islands(grid)}")
    
    # Test Array problems
    print("\n=== Array Problems Test ===")
    array_solutions = ArrayAndStringSolutions()
    nums = [-1, 0, 1, 2, -1, -4]
    print(f"Three sum: {array_solutions.three_sum(nums)}")
    
    print("\n=== All Day 14 Solutions Ready! ===")


if __name__ == "__main__":
    run_day14_examples()
