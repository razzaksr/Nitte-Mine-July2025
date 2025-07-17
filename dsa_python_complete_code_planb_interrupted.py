"""
Complete DSA Python Code Collection - 14 Days Training Program
Problem Solving & Data Structures and Algorithms in Python
"""

from collections import deque, defaultdict, Counter
import heapq
from typing import List, Optional, Dict, Set, Tuple
import sys

# # =============================================================================
# # DAYS 1-3: ARRAYS, STRINGS, BASIC RECURSION (Already Covered)
# # =============================================================================

# class ArrayProblems:
#     """Day 1-2: Array and String Problems"""
    
#     def two_sum(self, nums: List[int], target: int) -> List[int]:
#         """Find two numbers that add up to target"""
#         num_map = {}
#         for i, num in enumerate(nums):
#             complement = target - num
#             if complement in num_map:
#                 return [num_map[complement], i]
#             num_map[num] = i
#         return []
    
#     def find_second_max(self, nums: List[int]) -> int:
#         """Find second maximum element"""
#         if len(nums) < 2:
#             return -1
#         first = second = float('-inf')
#         for num in nums:
#             if num > first:
#                 second = first
#                 first = num
#             elif num > second and num != first:
#                 second = num
#         return second if second != float('-inf') else -1
    
#     def dutch_flag_sort(self, nums: List[int]) -> None:
#         """Sort array of 0s, 1s, 2s (Dutch Flag Algorithm)"""
#         low = mid = 0
#         high = len(nums) - 1
        
#         while mid <= high:
#             if nums[mid] == 0:
#                 nums[low], nums[mid] = nums[mid], nums[low]
#                 low += 1
#                 mid += 1
#             elif nums[mid] == 1:
#                 mid += 1
#             else:  # nums[mid] == 2
#                 nums[mid], nums[high] = nums[high], nums[mid]
#                 high -= 1
    
#     def move_zeros_to_end(self, nums: List[int]) -> None:
#         """Move all zeros to end while maintaining order"""
#         write_index = 0
#         for read_index in range(len(nums)):
#             if nums[read_index] != 0:
#                 nums[write_index] = nums[read_index]
#                 write_index += 1
        
#         while write_index < len(nums):
#             nums[write_index] = 0
#             write_index += 1
    
#     def kadane_algorithm(self, nums: List[int]) -> int:
#         """Find maximum subarray sum"""
#         max_sum = current_sum = nums[0]
#         for i in range(1, len(nums)):
#             current_sum = max(nums[i], current_sum + nums[i])
#             max_sum = max(max_sum, current_sum)
#         return max_sum
    
#     def find_pivot_index(self, nums: List[int]) -> int:
#         """Find pivot index where left sum equals right sum"""
#         total_sum = sum(nums)
#         left_sum = 0
#         for i, num in enumerate(nums):
#             if left_sum == total_sum - left_sum - num:
#                 return i
#             left_sum += num
#         return -1
    
#     def max_profit_stock(self, prices: List[int]) -> int:
#         """Best time to buy and sell stock"""
#         if not prices:
#             return 0
#         min_price = prices[0]
#         max_profit = 0
#         for price in prices[1:]:
#             max_profit = max(max_profit, price - min_price)
#             min_price = min(min_price, price)
#         return max_profit
    
#     def insert_interval(self, intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
#         """Insert new interval and merge overlapping ones"""
#         result = []
#         i = 0
#         n = len(intervals)
        
#         # Add all intervals before new_interval
#         while i < n and intervals[i][1] < new_interval[0]:
#             result.append(intervals[i])
#             i += 1
        
#         # Merge overlapping intervals
#         while i < n and intervals[i][0] <= new_interval[1]:
#             new_interval[0] = min(new_interval[0], intervals[i][0])
#             new_interval[1] = max(new_interval[1], intervals[i][1])
#             i += 1
        
#         result.append(new_interval)
        
#         # Add remaining intervals
#         while i < n:
#             result.append(intervals[i])
#             i += 1
        
#         return result
    
#     def is_twisted_prime(self, n: int) -> bool:
#         """Check if number is twisted prime"""
#         if not self.is_prime(n):
#             return False
#         reversed_n = int(str(n)[::-1])
#         return self.is_prime(reversed_n)
    
#     def is_prime(self, n: int) -> bool:
#         """Check if number is prime"""
#         if n < 2:
#             return False
#         for i in range(2, int(n**0.5) + 1):
#             if n % i == 0:
#                 return False
#         return True
    
#     def find_primes_in_range(self, start: int, end: int) -> List[int]:
#         """Find all primes in given range"""
#         primes = []
#         for num in range(start, end + 1):
#             if self.is_prime(num):
#                 primes.append(num)
#         return primes
    
#     def find_distinct_values(self, nums: List[int]) -> List[int]:
#         """Find distinct values in list"""
#         return list(set(nums))
    
#     def find_missing_min(self, nums: List[int]) -> int:
#         """Find missing minimum positive integer"""
#         nums_set = set(nums)
#         min_positive = 1
#         while min_positive in nums_set:
#             min_positive += 1
#         return min_positive

# class StringProblems:
#     """Day 2-3: String Problems"""
    
#     def group_anagrams(self, strs: List[str]) -> List[List[str]]:
#         """Group anagrams together"""
#         anagram_map = defaultdict(list)
#         for s in strs:
#             sorted_str = ''.join(sorted(s))
#             anagram_map[sorted_str].append(s)
#         return list(anagram_map.values())
    
#     def solve_x_o_game(self, board: List[List[str]]) -> None:
#         """Solve X and O game (capture surrounded regions)"""
#         if not board:
#             return
        
#         rows, cols = len(board), len(board[0])
        
#         def dfs(r, c):
#             if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != 'O':
#                 return
#             board[r][c] = 'T'  # Temporary mark
#             dfs(r + 1, c)
#             dfs(r - 1, c)
#             dfs(r, c + 1)
#             dfs(r, c - 1)
        
#         # Mark boundary-connected O's
#         for i in range(rows):
#             dfs(i, 0)
#             dfs(i, cols - 1)
#         for j in range(cols):
#             dfs(0, j)
#             dfs(rows - 1, j)
        
#         # Convert remaining O's to X's and T's back to O's
#         for i in range(rows):
#             for j in range(cols):
#                 if board[i][j] == 'O':
#                     board[i][j] = 'X'
#                 elif board[i][j] == 'T':
#                     board[i][j] = 'O'
    
#     def is_subsequence(self, s: str, t: str) -> bool:
#         """Check if s is subsequence of t"""
#         i = j = 0
#         while i < len(s) and j < len(t):
#             if s[i] == t[j]:
#                 i += 1
#             j += 1
#         return i == len(s)
    
#     def validate_ipv4(self, ip: str) -> bool:
#         """Validate IPv4 address"""
#         parts = ip.split('.')
#         if len(parts) != 4:
#             return False
        
#         for part in parts:
#             if not part or len(part) > 3:
#                 return False
#             if part[0] == '0' and len(part) > 1:
#                 return False
#             if not part.isdigit():
#                 return False
#             if int(part) > 255:
#                 return False
#         return True

# class BasicRecursion:
#     """Day 3: Basic Recursion Problems"""
    
#     def sum_list_recursive(self, nums: List[int]) -> int:
#         """Find sum of list using recursion"""
#         if not nums:
#             return 0
#         return nums[0] + self.sum_list_recursive(nums[1:])
    
#     def max_list_recursive(self, nums: List[int]) -> int:
#         """Find maximum of list using recursion"""
#         if len(nums) == 1:
#             return nums[0]
#         return max(nums[0], self.max_list_recursive(nums[1:]))
    
#     def prefix_sum_recursive(self, nums: List[int], index: int = 0) -> List[int]:
#         """Find prefix sum using recursion"""
#         if index == 0:
#             return [nums[0]] if nums else []
#         prev_prefix = self.prefix_sum_recursive(nums, index - 1)
#         return prev_prefix + [prev_prefix[-1] + nums[index]]
    
#     def coin_change_recursive(self, coins: List[int], amount: int) -> int:
#         """Coin change using divide and conquer (inefficient)"""
#         if amount == 0:
#             return 0
#         if amount < 0:
#             return -1
        
#         min_coins = float('inf')
#         for coin in coins:
#             result = self.coin_change_recursive(coins, amount - coin)
#             if result != -1:
#                 min_coins = min(min_coins, result + 1)
        
#         return min_coins if min_coins != float('inf') else -1
    
#     def inversion_count(self, arr: List[int]) -> int:
#         """Count inversions using recursion"""
#         def merge_and_count(arr, temp, left, mid, right):
#             i, j, k = left, mid + 1, left
#             inv_count = 0
            
#             while i <= mid and j <= right:
#                 if arr[i] <= arr[j]:
#                     temp[k] = arr[i]
#                     i += 1
#                 else:
#                     temp[k] = arr[j]
#                     inv_count += (mid - i + 1)
#                     j += 1
#                 k += 1
            
#             while i <= mid:
#                 temp[k] = arr[i]
#                 i += 1
#                 k += 1
            
#             while j <= right:
#                 temp[k] = arr[j]
#                 j += 1
#                 k += 1
            
#             for i in range(left, right + 1):
#                 arr[i] = temp[i]
            
#             return inv_count
        
#         def merge_sort_and_count(arr, temp, left, right):
#             inv_count = 0
#             if left < right:
#                 mid = (left + right) // 2
#                 inv_count += merge_sort_and_count(arr, temp, left, mid)
#                 inv_count += merge_sort_and_count(arr, temp, mid + 1, right)
#                 inv_count += merge_and_count(arr, temp, left, mid, right)
#             return inv_count
        
#         temp = [0] * len(arr)
#         return merge_sort_and_count(arr.copy(), temp, 0, len(arr) - 1)

# =============================================================================
# DAYS 4-6: DYNAMIC PROGRAMMING
# =============================================================================

class DynamicProgramming:
    """Days 4-6: Dynamic Programming Problems"""
    
    def __init__(self):
        self.memo = {}
    
    def coin_change_memo(self, coins: List[int], amount: int) -> int:
        """Coin change with memoization"""
        memo = {}
        
        def dp(remaining):
            if remaining == 0:
                return 0
            if remaining < 0:
                return -1
            if remaining in memo:
                return memo[remaining]
            
            min_coins = float('inf')
            for coin in coins:
                result = dp(remaining - coin)
                if result != -1:
                    min_coins = min(min_coins, result + 1)
            
            memo[remaining] = min_coins if min_coins != float('inf') else -1
            return memo[remaining]
        
        return dp(amount)
    
    def fibonacci_memo(self, n: int) -> int:
        """Fibonacci with memoization"""
        memo = {}
        
        def fib(n):
            if n <= 1:
                return n
            if n in memo:
                return memo[n]
            memo[n] = fib(n - 1) + fib(n - 2)
            return memo[n]
        
        return fib(n)
    
    def tower_of_hanoi(self, n: int, source: str, destination: str, auxiliary: str) -> List[str]:
        """Tower of Hanoi solution"""
        moves = []
        
        def hanoi(n, src, dest, aux):
            if n == 1:
                moves.append(f"Move disk 1 from {src} to {dest}")
                return
            hanoi(n - 1, src, aux, dest)
            moves.append(f"Move disk {n} from {src} to {dest}")
            hanoi(n - 1, aux, dest, src)
        
        hanoi(n, source, destination, auxiliary)
        return moves
    
    def climb_stairs(self, n: int) -> int:
        """Number of ways to climb stairs"""
        if n <= 2:
            return n
        
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        
        return dp[n]
    
    def house_robber(self, nums: List[int]) -> int:
        """House robber problem"""
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        prev2 = nums[0]
        prev1 = max(nums[0], nums[1])
        
        for i in range(2, len(nums)):
            current = max(prev1, prev2 + nums[i])
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    def longest_increasing_subsequence(self, nums: List[int]) -> int:
        """Length of longest increasing subsequence"""
        if not nums:
            return 0
        
        dp = [1] * len(nums)
        
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    def jump_game(self, nums: List[int]) -> bool:
        """Can reach last index"""
        max_reach = 0
        for i in range(len(nums)):
            if i > max_reach:
                return False
            max_reach = max(max_reach, i + nums[i])
        return True
    
    def decode_ways(self, s: str) -> int:
        """Number of ways to decode string"""
        if not s or s[0] == '0':
            return 0
        
        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1
        
        for i in range(2, n + 1):
            if s[i - 1] != '0':
                dp[i] += dp[i - 1]
            
            two_digit = int(s[i - 2:i])
            if 10 <= two_digit <= 26:
                dp[i] += dp[i - 2]
        
        return dp[n]
    
    def unique_paths(self, m: int, n: int) -> int:
        """Number of unique paths in grid"""
        dp = [[1] * n for _ in range(m)]
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        
        return dp[m - 1][n - 1]
    
    def min_path_sum(self, grid: List[List[int]]) -> int:
        """Minimum path sum in grid"""
        m, n = len(grid), len(grid[0])
        
        for i in range(1, m):
            grid[i][0] += grid[i - 1][0]
        
        for j in range(1, n):
            grid[0][j] += grid[0][j - 1]
        
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        
        return grid[m - 1][n - 1]
    
    def edit_distance(self, word1: str, word2: str) -> int:
        """Minimum edit distance between two strings"""
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        
        return dp[m][n]
    
    def longest_common_subsequence(self, text1: str, text2: str) -> int:
        """Length of longest common subsequence"""
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def knapsack_01(self, weights: List[int], values: List[int], capacity: int) -> int:
        """0/1 Knapsack problem"""
        n = len(weights)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
                else:
                    dp[i][w] = dp[i - 1][w]
        
        return dp[n][capacity]
    
    def palindromic_substrings(self, s: str) -> int:
        """Count palindromic substrings"""
        count = 0
        
        def expand_around_center(left, right):
            nonlocal count
            while left >= 0 and right < len(s) and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1
        
        for i in range(len(s)):
            expand_around_center(i, i)      # Odd length
            expand_around_center(i, i + 1)  # Even length
        
        return count

# =============================================================================
# DAYS 7-8: STACKS, QUEUES, LINKED LISTS
# =============================================================================

class StackImplementation:
    """Day 7: Stack Implementation and Problems"""
    
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

class StackProblems:
    """Stack-based problems"""
    
    def is_valid_parentheses(self, s: str) -> bool:
        """Check if parentheses are valid"""
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        
        for char in s:
            if char in mapping:
                if not stack or stack.pop() != mapping[char]:
                    return False
            else:
                stack.append(char)
        
        return not stack
    
    def next_greater_element(self, nums: List[int]) -> List[int]:
        """Find next greater element for each element"""
        result = [-1] * len(nums)
        stack = []
        
        for i in range(len(nums)):
            while stack and nums[stack[-1]] < nums[i]:
                result[stack.pop()] = nums[i]
            stack.append(i)
        
        return result
    
    def largest_rectangle_histogram(self, heights: List[int]) -> int:
        """Find largest rectangle in histogram"""
        stack = []
        max_area = 0
        
        for i in range(len(heights)):
            while stack and heights[stack[-1]] > heights[i]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(i)
        
        while stack:
            h = heights[stack.pop()]
            w = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, h * w)
        
        return max_area
    
    def evaluate_rpn(self, tokens: List[str]) -> int:
        """Evaluate Reverse Polish Notation"""
        stack = []
        
        for token in tokens:
            if token in ['+', '-', '*', '/']:
                b = stack.pop()
                a = stack.pop()
                if token == '+':
                    stack.append(a + b)
                elif token == '-':
                    stack.append(a - b)
                elif token == '*':
                    stack.append(a * b)
                elif token == '/':
                    stack.append(int(a / b))
            else:
                stack.append(int(token))
        
        return stack[0]

class QueueImplementation:
    """Day 7: Queue Implementation"""
    
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        return None
    
    def front(self):
        if not self.is_empty():
            return self.items[0]
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

class CircularQueue:
    """Circular Queue Implementation"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue = [0] * capacity
        self.front = 0
        self.size = 0
    
    def enqueue(self, value: int) -> bool:
        if self.is_full():
            return False
        rear = (self.front + self.size) % self.capacity
        self.queue[rear] = value
        self.size += 1
        return True
    
    def dequeue(self) -> bool:
        if self.is_empty():
            return False
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return True
    
    def get_front(self) -> int:
        return -1 if self.is_empty() else self.queue[self.front]
    
    def get_rear(self) -> int:
        if self.is_empty():
            return -1
        rear = (self.front + self.size - 1) % self.capacity
        return self.queue[rear]
    
    def is_empty(self) -> bool:
        return self.size == 0
    
    def is_full(self) -> bool:
        return self.size == self.capacity

class QueueProblems:
    """Queue-based problems"""
    
    def sliding_window_maximum(self, nums: List[int], k: int) -> List[int]:
        """Find maximum in each sliding window"""
        if not nums or k == 0:
            return []
        
        dq = deque()
        result = []
        
        for i in range(len(nums)):
            # Remove elements outside window
            while dq and dq[0] < i - k + 1:
                dq.popleft()
            
            # Remove smaller elements
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()
            
            dq.append(i)
            
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result
    
    def first_non_repeating_char(self, stream: str) -> List[str]:
        """Find first non-repeating character in stream"""
        char_count = {}
        queue = deque()
        result = []
        
        for char in stream:
            char_count[char] = char_count.get(char, 0) + 1
            queue.append(char)
            
            while queue and char_count[queue[0]] > 1:
                queue.popleft()
            
            result.append(queue[0] if queue else '#')
        
        return result

class ListNode:
    """Linked List Node"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedListProblems:
    """Day 8: Linked List Problems"""
    
    def reverse_linked_list(self, head: ListNode) -> ListNode:
        """Reverse a linked list"""
        prev = None
        current = head
        
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        return prev
    
    def merge_two_sorted_lists(self, l1: ListNode, l2: ListNode) -> ListNode:
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
        
        current.next = l1 or l2
        return dummy.next
    
    def remove_nth_from_end(self, head: ListNode, n: int) -> ListNode:
        """Remove nth node from end"""
        dummy = ListNode(0)
        dummy.next = head
        fast = slow = dummy
        
        for _ in range(n + 1):
            fast = fast.next
        
        while fast:
            fast = fast.next
            slow = slow.next
        
        slow.next = slow.next.next
        return dummy.next
    
    def has_cycle(self, head: ListNode) -> bool:
        """Detect cycle in linked list (Floyd's algorithm)"""
        if not head or not head.next:
            return False
        
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        
        return False
    
    def find_intersection(self, headA: ListNode, headB: ListNode) -> ListNode:
        """Find intersection of two linked lists"""
        if not headA or not headB:
            return None
        
        pA, pB = headA, headB
        
        while pA != pB:
            pA = pA.next if pA else headB
            pB = pB.next if pB else headA
        
        return pA
    
    def add_two_numbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        """Add two numbers represented as linked lists"""
        dummy = ListNode(0)
        current = dummy
        carry = 0
        
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            total = val1 + val2 + carry
            
            carry = total // 10
            current.next = ListNode(total % 10)
            current = current.next
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        
        return dummy.next

# =============================================================================
# DAYS 9-10: TREES AND HEAPS
# =============================================================================

class TreeNode:
    """Binary Tree Node"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinaryTreeProblems:
    """Days 9-10: Binary Tree Problems"""
    
    def inorder_traversal(self, root: TreeNode) -> List[int]:
        """Inorder traversal of binary tree"""
        result = []
        
        def inorder(node):
            if node:
                inorder(node.left)
                result.append(node.val)
                inorder(node.right)
        
        inorder(root)
        return result
    
    def preorder_traversal(self, root: TreeNode) -> List[int]:
        """Preorder traversal of binary tree"""
        result = []
        
        def preorder(node):
            if node:
                result.append(node.val)
                preorder(node.left)
                preorder(node.right)
        
        preorder(root)
        return result
    
    def postorder_traversal(self, root: TreeNode) -> List[int]:
        """Postorder traversal of binary tree"""
        result = []
        
        def postorder(node):
            if node:
                postorder(node.left)
                postorder(node.right)
                result.append(node.val)
        
        postorder(root)
        return result
    
    def level_order_traversal(self, root: TreeNode) -> List[List[int]]:
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
    
    def max_depth(self, root: TreeNode) -> int:
        """Maximum depth of binary tree"""
        if not root:
            return 0
        return 1 + max(self.max_depth(root.left), self.max_depth(root.right))
    
    def is_symmetric(self, root: TreeNode) -> bool:
        """Check if tree is symmetric"""
        def is_mirror(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            return (left.val == right.val and 
                    is_mirror(left.left, right.right) and 
                    is_mirror(left.right, right.left))
        
        return is_mirror(root.left, root.right) if root else True
    
    def is_valid_bst(self, root: TreeNode) -> bool:
        """Validate Binary Search Tree"""
        def validate(node, min_val, max_val):
            if not node:
                return True
            if node.val <= min_val or node.val >= max_val:
                return False
            return (validate(node.left, min_val, node.val) and 
                    validate(node.right, node.val, max_val))
        
        return validate(root, float('-inf'), float('inf'))
    
    def lowest_common_ancestor_bst(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """Find LCA in BST"""
        if not root:
            return None
        
        if p.val < root.val and q.val < root.val:
            return self.lowest_common_ancestor_bst(root.left, p, q)
        elif p.val > root.val and q.val > root.val:
            return self.lowest_common_ancestor_bst(root.right, p, q)
        else:
            return root
    
    def sorted_array_to_bst(self, nums: List[int]) -> TreeNode:
        """Convert sorted array to BST"""
        def build_tree(left, right):
            if left > right:
                return None
            
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            root.left = build_tree(left, mid - 1)
            root.right = build_tree(mid + 1, right)
            return root
        
        return build_tree(0, len(nums) - 1)
    
    def diameter_of_tree(self, root: TreeNode) -> int:
        """Find diameter of binary tree"""
        self.diameter = 0
        
        def depth(node):
            if not node:
                return 0
            
            left_depth = depth(node.left)
            right_depth = depth(node.right)
            
            self.diameter = max(self.diameter, left_depth + right_depth)
            return 1 + max(left_depth, right_depth)
        
        depth(root)
        return self.diameter
    
    def has_path_sum(self, root: TreeNode, target_sum: int) -> bool:
        """Check if tree has path with given sum"""
        if not root:
            return False
        
        if not root.left and not root.right:
            return root.val == target_sum
        
        remaining = target_sum - root.val
        return (self.has_path_sum(root.left, remaining) or 
                self.has_path_sum(root.right, remaining))
    
    def path_sum_ii(self, root: TreeNode, target_sum: int) -> List[List[int]]:
        """Find all paths with given sum"""
        result = []
        
        def dfs(node, remaining, path):
            if not node:
                return
            
            path.append(node.val)
            
            if not node.left and not node.right and remaining == node.val:
                result.append(path[:])
            
            dfs(node.left, remaining - node.val, path)
            dfs(node.right, remaining - node.val, path)
            path.pop()
        
        dfs(root, target_sum, [])
        return result
    
    def max_path_sum(self, root: TreeNode) -> int:
        """Binary tree maximum path sum"""
        self.max_sum = float('-inf')
        
        def max_gain(node):
            if not node:
                return 0
            
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)
            
            price_newpath = node.val + left_gain + right_gain
            self.max_sum = max(self.max_sum, price_newpath)
            
            return node.val + max(left_gain, right_gain)
        
        max_gain(root)
        return self.max_sum
    
    def serialize_tree(self, root: TreeNode) -> str:
        """Serialize binary tree to string"""
        def preorder(node):
            if not node:
                return "null,"
            return str(node.val) + "," + preorder(node.left) + preorder(node.right)
        
        return preorder(root)
    
    def deserialize_tree(self, data: str) -> TreeNode:
        """Deserialize string to binary tree"""
        def build_tree():
            val = next(vals)
            if val == "null":
                return None
            node = TreeNode(int(val))
            node.left = build_tree()
            node.right = build_tree()
            return node
        
        vals = iter(data.split(","))
        return build_tree()

class BST:
    """Binary Search Tree Implementation"""
    
    def __init__(self):
        self.root = None
    
    def insert(self, val: int) -> None:
        """Insert value into BST"""
        self.root = self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node: TreeNode, val: int) -> TreeNode:
        if not node:
            return TreeNode(val)
        
        if val < node.val:
            node.left = self._insert_recursive(node.left, val)
        else:
            node.right = self._insert_recursive(node.right, val)
        
        return node
    
    def search(self, val: int) -> bool:
        """Search for value in BST"""
        return self._search_recursive(self.root, val)
    
    def _search_recursive(self, node: TreeNode, val: int) -> bool:
        if not node:
            return False
        if node.val == val:
            return True
        elif val < node.val:
            return self._search_recursive(node.left, val)
        else:
            return self._search_recursive(node.right, val)
    
    def delete(self, val: int) -> None:
        """Delete value from BST"""
        self.root = self._delete_recursive(self.root, val)
    
    def _delete_recursive(self, node: TreeNode, val: int) -> TreeNode:
        if not node:
            return node
        
        if val < node.val:
            node.left = self._delete_recursive(node.left, val)
        elif val > node.val:
            node.right = self._delete_recursive(node.right, val)
        else:
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            
            # Find inorder successor
            temp = self._find_min(node.right)
            node.val = temp.val
            node.right = self._delete_recursive(node.right, temp.val)
        
        return node
    
    def _find_min(self, node: TreeNode) -> TreeNode:
        while node.left:
            node = node.left
        return node

class HeapProblems:
    """Day 10: Heap Problems"""
    
    def kth_largest_element(self, nums: List[int], k: int) -> int:
        """Find kth largest element using heap"""
        return heapq.nlargest(k, nums)[-1]
    
    def kth_largest_in_stream(self, k: int, nums: List[int]):
        """Kth largest element in stream"""
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        
        while len(self.heap) > k:
            heapq.heappop(self.heap)
    
    def add_to_stream(self, val: int) -> int:
        """Add value to stream and return kth largest"""
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]
    
    def merge_k_sorted_lists(self, lists: List[ListNode]) -> ListNode:
        """Merge k sorted linked lists"""
        if not lists:
            return None
        
        heap = []
        
        for i, head in enumerate(lists):
            if head:
                heapq.heappush(heap, (head.val, i, head))
        
        dummy = ListNode(0)
        current = dummy
        
        while heap:
            val, i, node = heapq.heappop(heap)
            current.next = ListNode(val)
            current = current.next
            
            if node.next:
                heapq.heappush(heap, (node.next.val, i, node.next))
        
        return dummy.next
    
    def top_k_frequent(self, nums: List[int], k: int) -> List[int]:
        """Find top k frequent elements"""
        count = Counter(nums)
        return [item for item, freq in count.most_common(k)]
    
    def find_median_from_stream(self):
        """Find median from data stream"""
        self.max_heap = []  # For smaller half
        self.min_heap = []  # For larger half
    
    def add_number(self, num: int) -> None:
        """Add number to stream"""
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)
        
        # Balance heaps
        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap) + 1:
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
    
    def find_median(self) -> float:
        """Find median of current stream"""
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2.0
        elif len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        else:
            return self.min_heap[0]

# =============================================================================
# DAYS 11-12: GRAPHS
# =============================================================================

class GraphProblems:
    """Days 11-12: Graph Problems"""
    
    def __init__(self):
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def dfs_recursive(self, graph: Dict[int, List[int]], start: int, visited: Set[int]) -> None:
        """DFS traversal recursive"""
        visited.add(start)
        print(start, end=" ")
        
        for neighbor in graph[start]:
            if neighbor not in visited:
                self.dfs_recursive(graph, neighbor, visited)
    
    def dfs_iterative(self, graph: Dict[int, List[int]], start: int) -> List[int]:
        """DFS traversal iterative"""
        visited = set()
        stack = [start]
        result = []
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                result.append(node)
                
                for neighbor in reversed(graph[node]):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return result
    
    def bfs(self, graph: Dict[int, List[int]], start: int) -> List[int]:
        """BFS traversal"""
        visited = set()
        queue = deque([start])
        result = []
        
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                result.append(node)
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return result
    
    def num_islands(self, grid: List[List[str]]) -> int:
        """Count number of islands"""
        if not grid:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        islands = 0
        
        def dfs(r, c):
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
                return
            grid[r][c] = '0'  # Mark as visited
            
            for dr, dc in self.directions:
                dfs(r + dr, c + dc)
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '1':
                    dfs(i, j)
                    islands += 1
        
        return islands
    
    def clone_graph(self, node) -> 'Node':
        """Clone graph"""
        if not node:
            return None
        
        visited = {}
        
        def clone(node):
            if node in visited:
                return visited[node]
            
            copy = Node(node.val)
            visited[node] = copy
            
            for neighbor in node.neighbors:
                copy.neighbors.append(clone(neighbor))
            
            return copy
        
        return clone(node)
    
    def word_ladder(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        """Word ladder problem"""
        if endWord not in wordList:
            return 0
        
        wordSet = set(wordList)
        queue = deque([(beginWord, 1)])
        visited = set([beginWord])
        
        while queue:
            word, length = queue.popleft()
            
            if word == endWord:
                return length
            
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + c + word[i+1:]
                    
                    if next_word in wordSet and next_word not in visited:
                        visited.add(next_word)
                        queue.append((next_word, length + 1))
        
        return 0
    
    def oranges_rotting(self, grid: List[List[int]]) -> int:
        """Rotting oranges problem"""
        if not grid:
            return -1
        
        rows, cols = len(grid), len(grid[0])
        queue = deque()
        fresh_count = 0
        
        # Find all rotten oranges and count fresh ones
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 2:
                    queue.append((i, j, 0))
                elif grid[i][j] == 1:
                    fresh_count += 1
        
        if fresh_count == 0:
            return 0
        
        minutes = 0
        
        while queue:
            row, col, time = queue.popleft()
            minutes = time
            
            for dr, dc in self.directions:
                nr, nc = row + dr, col + dc
                
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh_count -= 1
                    queue.append((nr, nc, time + 1))
        
        return minutes if fresh_count == 0 else -1
    
    def has_cycle_directed(self, graph: Dict[int, List[int]]) -> bool:
        """Detect cycle in directed graph"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(int)
        
        def dfs(node):
            if color[node] == GRAY:
                return True
            if color[node] == BLACK:
                return False
            
            color[node] = GRAY
            
            for neighbor in graph[node]:
                if dfs(neighbor):
                    return True
            
            color[node] = BLACK
            return False
        
        for node in graph:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        
        return False
    
    def topological_sort(self, graph: Dict[int, List[int]], num_nodes: int) -> List[int]:
        """Topological sort using Kahn's algorithm"""
        in_degree = [0] * num_nodes
        
        # Calculate in-degrees
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1
        
        queue = deque()
        for i in range(num_nodes):
            if in_degree[i] == 0:
                queue.append(i)
        
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result if len(result) == num_nodes else []
    
    def can_finish_courses(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """Course schedule problem"""
        graph = defaultdict(list)
        in_degree = [0] * numCourses
        
        # Build graph
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1
        
        queue = deque()
        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)
        
        completed = 0
        
        while queue:
            course = queue.popleft()
            completed += 1
            
            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
        
        return completed == numCourses
    
    def dijkstra(self, graph: Dict[int, List[Tuple[int, int]]], start: int) -> Dict[int, int]:
        """Dijkstra's shortest path algorithm"""
        distances = defaultdict(lambda: float('inf'))
        distances[start] = 0
        pq = [(0, start)]
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_dist > distances[current_node]:
                continue
            
            for neighbor, weight in graph[current_node]:
                distance = current_dist + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
        
        return dict(distances)
    
    def network_delay_time(self, times: List[List[int]], n: int, k: int) -> int:
        """Network delay time problem"""
        graph = defaultdict(list)
        
        for u, v, w in times:
            graph[u].append((v, w))
        
        distances = self.dijkstra(graph, k)
        
        if len(distances) != n:
            return -1
        
        return max(distances.values())
    
    def find_cheapest_price(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        """Cheapest flights within k stops"""
        graph = defaultdict(list)
        
        for u, v, w in flights:
            graph[u].append((v, w))
        
        # (cost, node, stops)
        pq = [(0, src, 0)]
        visited = {}
        
        while pq:
            cost, node, stops = heapq.heappop(pq)
            
            if node == dst:
                return cost
            
            if stops > k:
                continue
            
            if node in visited and visited[node] <= stops:
                continue
            
            visited[node] = stops
            
            for neighbor, price in graph[node]:
                heapq.heappush(pq, (cost + price, neighbor, stops + 1))
        
        return -1

class UnionFind:
    """Union-Find (Disjoint Set Union) Data Structure"""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Find root with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union by rank"""
        px, py = self.find(x), self.find(y)
        
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """Check if two elements are connected"""
        return self.find(x) == self.find(y)

# =============================================================================
# DAYS 13-14: SPECIALIZED TOPICS
# =============================================================================

class BitManipulation:
    """Day 13: Bit Manipulation Problems"""
    
    def single_number(self, nums: List[int]) -> int:
        """Find single number using XOR"""
        result = 0
        for num in nums:
            result ^= num
        return result
    
    def single_number_ii(self, nums: List[int]) -> int:
        """Find single number when others appear 3 times"""
        ones = twos = 0
        
        for num in nums:
            ones = (ones ^ num) & ~twos
            twos = (twos ^ num) & ~ones
        
        return ones
    
    def counting_bits(self, n: int) -> List[int]:
        """Count number of 1s in binary representation"""
        result = [0] * (n + 1)
        
        for i in range(1, n + 1):
            result[i] = result[i >> 1] + (i & 1)
        
        return result
    
    def is_power_of_two(self, n: int) -> bool:
        """Check if number is power of 2"""
        return n > 0 and (n & (n - 1)) == 0
    
    def hamming_distance(self, x: int, y: int) -> int:
        """Calculate Hamming distance"""
        xor = x ^ y
        count = 0
        
        while xor:
            count += xor & 1
            xor >>= 1
        
        return count
    
    def reverse_bits(self, n: int) -> int:
        """Reverse bits of 32-bit integer"""
        result = 0
        for i in range(32):
            result = (result << 1) | (n & 1)
            n >>= 1
        return result
    
    def find_missing_number(self, nums: List[int]) -> int:
        """Find missing number using XOR"""
        n = len(nums)
        result = n
        
        for i in range(n):
            result ^= i ^ nums[i]
        
        return result

class TwoPointerProblems:
    """Day 13: Two Pointer Problems"""
    
    def three_sum(self, nums: List[int]) -> List[List[int]]:
        """Find all triplets that sum to zero"""
        nums.sort()
        result = []
        
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
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
    
    def container_with_most_water(self, height: List[int]) -> int:
        """Find container with most water"""
        left, right = 0, len(height) - 1
        max_area = 0
        
        while left < right:
            width = right - left
            area = width * min(height[left], height[right])
            max_area = max(max_area, area)
            
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return max_area
    
    def remove_duplicates(self, nums: List[int]) -> int:
        """Remove duplicates from sorted array"""
        if not nums:
            return 0
        
        write_index = 1
        
        for read_index in range(1, len(nums)):
            if nums[read_index] != nums[read_index - 1]:
                nums[write_index] = nums[read_index]
                write_index += 1
        
        return write_index
    
    def is_palindrome(self, s: str) -> bool:
        """Check if string is palindrome"""
        left, right = 0, len(s) - 1
        
        while left < right:
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1
            
            if s[left].lower() != s[right].lower():
                return False
            
            left += 1
            right -= 1
        
        return True

class SlidingWindowProblems:
    """Day 13: Sliding Window Problems"""
    
    def length_of_longest_substring(self, s: str) -> int:
        """Longest substring without repeating characters"""
        char_set = set()
        left = 0
        max_length = 0
        
        for right in range(len(s)):
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1
            
            char_set.add(s[right])
            max_length = max(max_length, right - left + 1)
        
        return max_length
    
    def min_window_substring(self, s: str, t: str) -> str:
        """Minimum window substring"""
        if not s or not t:
            return ""
        
        t_count = Counter(t)
        required = len(t_count)
        
        left = right = 0
        formed = 0
        window_counts = {}
        
        ans = float('inf'), None, None
        
        while right < len(s):
            char = s[right]
            window_counts[char] = window_counts.get(char, 0) + 1
            
            if char in t_count and window_counts[char] == t_count[char]:
                formed += 1
            
            while left <= right and formed == required:
                char = s[left]
                
                if right - left + 1 < ans[0]:
                    ans = (right - left + 1, left, right)
                
                window_counts[char] -= 1
                if char in t_count and window_counts[char] < t_count[char]:
                    formed -= 1
                
                left += 1
            
            right += 1
        
        return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]
    
    def find_all_anagrams(self, s: str, p: str) -> List[int]:
        """Find all anagrams of p in s"""
        if len(p) > len(s):
            return []
        
        p_count = Counter(p)
        window_count = Counter()
        result = []
        
        for i in range(len(s)):
            # Add current character to window
            window_count[s[i]] += 1
            
            # Remove character that's out of window
            if i >= len(p):
                if window_count[s[i - len(p)]] == 1:
                    del window_count[s[i - len(p)]]
                else:
                    window_count[s[i - len(p)]] -= 1
            
            # Check if current window is an anagram
            if window_count == p_count:
                result.append(i - len(p) + 1)
        
        return result

class TrieNode:
    """Trie Node"""
    def __init__(self):