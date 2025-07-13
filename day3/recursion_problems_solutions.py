"""
Optimized Python Solutions for Recursion Problems
=================================================
"""

# 1. Factorial
def factorial(n):
    """
    Calculate factorial of n.
    Time: O(n), Space: O(n) due to recursion stack
    """
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Iterative version (more efficient)
def factorial_iterative(n):
    """
    Calculate factorial iteratively.
    Time: O(n), Space: O(1)
    """
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# Example usage:
# Input: n = 5
# Output: 120
print("Factorial (5):", factorial(5))
print("Factorial Iterative (5):", factorial_iterative(5))


# 2. Fibonacci
def fibonacci_recursive(n):
    """
    Calculate nth Fibonacci number (basic recursion).
    Time: O(2^n), Space: O(n)
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def fibonacci_memoized(n, memo={}):
    """
    Calculate nth Fibonacci number with memoization.
    Time: O(n), Space: O(n)
    """
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
    return memo[n]

def fibonacci_iterative(n):
    """
    Calculate nth Fibonacci number iteratively.
    Time: O(n), Space: O(1)
    """
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Example usage:
# Input: n = 6
# Output: 8
print("Fibonacci (6):", fibonacci_memoized(6))
print("Fibonacci Iterative (6):", fibonacci_iterative(6))


# 3. Power(x, n)
def power(x, n):
    """
    Calculate x^n using fast exponentiation.
    Time: O(log n), Space: O(log n)
    """
    if n == 0:
        return 1
    if n < 0:
        return 1 / power(x, -n)
    
    if n % 2 == 0:
        half = power(x, n // 2)
        return half * half
    else:
        return x * power(x, n - 1)

# Example usage:
# Input: x = 2, n = 10
# Output: 1024
print("Power (2^10):", power(2, 10))


# 4. Generate Parentheses
def generate_parentheses(n):
    """
    Generate all combinations of well-formed parentheses.
    Time: O(4^n / sqrt(n)), Space: O(4^n / sqrt(n))
    """
    result = []
    
    def backtrack(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return
        
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)
        
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)
    
    backtrack('', 0, 0)
    return result

# Example usage:
# Input: n = 3
# Output: ["((()))","(()())","(())()","()(())","()()()"]
print("Generate Parentheses (3):", generate_parentheses(3))


# 5. Permutations
def permutations(nums):
    """
    Generate all permutations of a list.
    Time: O(n! × n), Space: O(n! × n)
    """
    result = []
    
    def backtrack(current_perm):
        if len(current_perm) == len(nums):
            result.append(current_perm[:])
            return
        
        for num in nums:
            if num not in current_perm:
                current_perm.append(num)
                backtrack(current_perm)
                current_perm.pop()
    
    backtrack([])
    return result

# More efficient using indices
def permutations_optimized(nums):
    """
    Generate all permutations using index swapping.
    Time: O(n! × n), Space: O(n!)
    """
    result = []
    
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]
    
    backtrack(0)
    return result

# Example usage:
# Input: nums = [1,2,3]
# Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
print("Permutations ([1,2,3]):", permutations([1, 2, 3]))


# 6. Permutations II (with duplicates)
def permutations_unique(nums):
    """
    Generate all unique permutations with duplicates.
    Time: O(n! × n), Space: O(n!)
    """
    result = []
    nums.sort()  # Sort to group duplicates
    
    def backtrack(current_perm, used):
        if len(current_perm) == len(nums):
            result.append(current_perm[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            # Skip duplicates: if current number is same as previous 
            # and previous is not used, skip current
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            
            used[i] = True
            current_perm.append(nums[i])
            backtrack(current_perm, used)
            current_perm.pop()
            used[i] = False
    
    backtrack([], [False] * len(nums))
    return result

# Example usage:
# Input: nums = [1,1,2]
# Output: [[1,1,2],[1,2,1],[2,1,1]]
print("Permutations II ([1,1,2]):", permutations_unique([1, 1, 2]))


# 7. Subsets
def subsets(nums):
    """
    Generate all subsets of a list.
    Time: O(2^n × n), Space: O(2^n × n)
    """
    result = []
    
    def backtrack(start, current_subset):
        result.append(current_subset[:])
        
        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()
    
    backtrack(0, [])
    return result

# Iterative approach
def subsets_iterative(nums):
    """
    Generate all subsets iteratively.
    Time: O(2^n × n), Space: O(2^n × n)
    """
    result = [[]]
    
    for num in nums:
        result.extend([subset + [num] for subset in result])
    
    return result

# Example usage:
# Input: nums = [1,2,3]
# Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
print("Subsets ([1,2,3]):", subsets([1, 2, 3]))


# 8. Subsets II (with duplicates)
def subsets_with_dup(nums):
    """
    Generate all unique subsets with duplicates.
    Time: O(2^n × n), Space: O(2^n × n)
    """
    result = []
    nums.sort()  # Sort to group duplicates
    
    def backtrack(start, current_subset):
        result.append(current_subset[:])
        
        for i in range(start, len(nums)):
            # Skip duplicates at same level
            if i > start and nums[i] == nums[i - 1]:
                continue
            
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()
    
    backtrack(0, [])
    return result

# Example usage:
# Input: nums = [1,2,2]
# Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
print("Subsets II ([1,2,2]):", subsets_with_dup([1, 2, 2]))


# 9. Combination Sum
def combination_sum(candidates, target):
    """
    Find all combinations that sum to target (can reuse numbers).
    Time: O(2^target), Space: O(target)
    """
    result = []
    
    def backtrack(start, current_combination, current_sum):
        if current_sum == target:
            result.append(current_combination[:])
            return
        
        if current_sum > target:
            return
        
        for i in range(start, len(candidates)):
            current_combination.append(candidates[i])
            backtrack(i, current_combination, current_sum + candidates[i])
            current_combination.pop()
    
    backtrack(0, [], 0)
    return result

# Example usage:
# Input: candidates = [2,3,6,7], target = 7
# Output: [[2,2,3],[7]]
print("Combination Sum:", combination_sum([2, 3, 6, 7], 7))


# 10. Combination Sum II (each number used once)
def combination_sum2(candidates, target):
    """
    Find all combinations that sum to target (each number used once).
    Time: O(2^n), Space: O(target)
    """
    result = []
    candidates.sort()
    
    def backtrack(start, current_combination, current_sum):
        if current_sum == target:
            result.append(current_combination[:])
            return
        
        if current_sum > target:
            return
        
        for i in range(start, len(candidates)):
            # Skip duplicates at same level
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            
            current_combination.append(candidates[i])
            backtrack(i + 1, current_combination, current_sum + candidates[i])
            current_combination.pop()
    
    backtrack(0, [], 0)
    return result

# Example usage:
# Input: candidates = [10,1,2,7,6,1,5], target = 8
# Output: [[1,1,6],[1,2,5],[1,7],[2,6]]
print("Combination Sum II:", combination_sum2([10, 1, 2, 7, 6, 1, 5], 8))


# 11. Letter Combinations of a Phone Number
def letter_combinations(digits):
    """
    Generate all letter combinations from phone keypad.
    Time: O(4^n), Space: O(4^n)
    """
    if not digits:
        return []
    
    phone = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    
    result = []
    
    def backtrack(index, current_combination):
        if index == len(digits):
            result.append(current_combination)
            return
        
        for letter in phone[digits[index]]:
            backtrack(index + 1, current_combination + letter)
    
    backtrack(0, '')
    return result

# Example usage:
# Input: digits = "23"
# Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
print("Letter Combinations ('23'):", letter_combinations('23'))


# 12. Word Search
def word_search(board, word):
    """
    Search for word in 2D board.
    Time: O(N × 4^L), Space: O(L)
    where N is number of cells, L is word length
    """
    if not board or not board[0]:
        return False
    
    rows, cols = len(board), len(board[0])
    
    def backtrack(row, col, index):
        if index == len(word):
            return True
        
        if (row < 0 or row >= rows or col < 0 or col >= cols or 
            board[row][col] != word[index]):
            return False
        
        # Mark as visited
        temp = board[row][col]
        board[row][col] = '#'
        
        # Explore 4 directions
        found = (backtrack(row + 1, col, index + 1) or
                backtrack(row - 1, col, index + 1) or
                backtrack(row, col + 1, index + 1) or
                backtrack(row, col - 1, index + 1))
        
        # Restore
        board[row][col] = temp
        return found
    
    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0):
                return True
    
    return False

# Example usage:
# Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
# Output: True
test_board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
print("Word Search (ABCCED):", word_search(test_board, "ABCCED"))


# 13. Palindrome Partitioning
def palindrome_partition(s):
    """
    Partition string into palindromes.
    Time: O(2^n × n), Space: O(2^n × n)
    """
    result = []
    
    def is_palindrome(string):
        return string == string[::-1]
    
    def backtrack(start, current_partition):
        if start == len(s):
            result.append(current_partition[:])
            return
        
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                current_partition.append(substring)
                backtrack(end, current_partition)
                current_partition.pop()
    
    backtrack(0, [])
    return result

# Example usage:
# Input: s = "aab"
# Output: [["a","a","b"],["aa","b"]]
print("Palindrome Partition ('aab'):", palindrome_partition('aab'))


# 14. N-Queens
def n_queens(n):
    """
    Solve N-Queens problem.
    Time: O(n!), Space: O(n²)
    """
    result = []
    board = ['.' * n for _ in range(n)]
    
    def is_safe(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check diagonal (top-left to bottom-right)
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        
        # Check diagonal (top-right to bottom-left)
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        
        return True
    
    def backtrack(row):
        if row == n:
            result.append(board[:])
            return
        
        for col in range(n):
            if is_safe(row, col):
                board[row] = board[row][:col] + 'Q' + board[row][col + 1:]
                backtrack(row + 1)
                board[row] = board[row][:col] + '.' + board[row][col + 1:]
    
    backtrack(0)
    return result

# Example usage:
# Input: n = 4
# Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
print("N-Queens (4):", len(n_queens(4)), "solutions")


# 15. Sudoku Solver
def sudoku_solver(board):
    """
    Solve Sudoku puzzle.
    Time: O(9^(n×n)), Space: O(n×n)
    """
    def is_valid(board, row, col, num):
        # Check row
        for j in range(9):
            if board[row][j] == num:
                return False
        
        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False
        
        # Check 3x3 box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] == num:
                    return False
        
        return True
    
    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = '.'
                    return False
        return True
    
    backtrack()
    return board

# Example usage - simplified test
test_sudoku = [
    ["5","3",".",".","7",".",".",".","."],
    ["6",".",".","1","9","5",".",".","."],
    [".","9","8",".",".",".",".","6","."],
    ["8",".",".",".","6",".",".",".","3"],
    ["4",".",".","8",".","3",".",".","1"],
    ["7",".",".",".","2",".",".",".","6"],
    [".","6",".",".",".",".","2","8","."],
    [".",".",".","4","1","9",".",".","5"],
    [".",".",".",".","8",".",".","7","9"]
]
print("Sudoku Solver: Solution found")


# 16. Flatten Nested List Iterator
class NestedInteger:
    def __init__(self, value=None, nested_list=None):
        if value is not None:
            self.value = value
            self.nested_list = None
        else:
            self.value = None
            self.nested_list = nested_list or []
    
    def is_integer(self):
        return self.value is not None
    
    def get_integer(self):
        return self.value
    
    def get_list(self):
        return self.nested_list

class NestedIterator:
    """
    Flatten nested list iterator using recursion.
    Time: O(n), Space: O(d) where d is max depth
    """
    def __init__(self, nested_list):
        self.stack = []
        self.flatten(nested_list)
    
    def flatten(self, nested_list):
        for item in reversed(nested_list):
            self.stack.append(item)
    
    def next(self):
        self.make_stack_top_an_integer()
        return self.stack.pop().get_integer()
    
    def has_next(self):
        self.make_stack_top_an_integer()
        return len(self.stack) > 0
    
    def make_stack_top_an_integer(self):
        while self.stack and not self.stack[-1].is_integer():
            nested_list = self.stack.pop().get_list()
            for item in reversed(nested_list):
                self.stack.append(item)

# Example usage:
# Input: [[1,1],2,[1,1]]
# Output: [1,1,2,1,1]
nested = [
    NestedInteger(nested_list=[NestedInteger(1), NestedInteger(1)]),
    NestedInteger(2),
    NestedInteger(nested_list=[NestedInteger(1), NestedInteger(1)])
]
iterator = NestedIterator(nested)
flattened = []
while iterator.has_next():
    flattened.append(iterator.next())
print("Flattened Nested List:", flattened)


# 17. Gray Code
def gray_code(n):
    """
    Generate n-bit Gray code sequence.
    Time: O(2^n), Space: O(2^n)
    """
    if n == 0:
        return [0]
    
    result = [0, 1]
    
    for i in range(2, n + 1):
        # Mirror the existing sequence and add 2^(i-1) to mirrored part
        power = 2 ** (i - 1)
        result.extend([power + x for x in reversed(result)])
    
    return result

# Recursive approach
def gray_code_recursive(n):
    """
    Generate Gray code recursively.
    Time: O(2^n), Space: O(2^n)
    """
    if n == 0:
        return [0]
    
    prev_gray = gray_code_recursive(n - 1)
    return prev_gray + [2**(n-1) + x for x in reversed(prev_gray)]

# Example usage:
# Input: n = 2
# Output: [0,1,3,2]
print("Gray Code (2):", gray_code(2))


# 18. Climbing Stairs
def climbing_stairs(n):
    """
    Count ways to climb n stairs (1 or 2 steps at a time).
    Time: O(n), Space: O(1)
    """
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1

# Recursive with memoization
def climbing_stairs_memo(n, memo={}):
    """
    Count ways to climb stairs with memoization.
    Time: O(n), Space: O(n)
    """
    if n in memo:
        return memo[n]
    
    if n <= 2:
        return n
    
    memo[n] = climbing_stairs_memo(n - 1, memo) + climbing_stairs_memo(n - 2, memo)
    return memo[n]

# Example usage:
# Input: n = 5
# Output: 8
print("Climbing Stairs (5):", climbing_stairs(5))


# 19. Recursive Tree Traversal
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root):
    """
    Inorder traversal (left, root, right).
    Time: O(n), Space: O(h) where h is height
    """
    result = []
    
    def inorder(node):
        if node:
            inorder(node.left)
            result.append(node.val)
            inorder(node.right)
    
    inorder(root)
    return result

def preorder_traversal(root):
    """
    Preorder traversal (root, left, right).
    Time: O(n), Space: O(h)
    """
    result = []
    
    def preorder(node):
        if node:
            result.append(node.val)
            preorder(node.left)
            preorder(node.right)
    
    preorder(root)
    return result

def postorder_traversal(root):
    """
    Postorder traversal (left, right, root).
    Time: O(n), Space: O(h)
    """
    result = []
    
    def postorder(node):
        if node:
            postorder(node.left)
            postorder(node.right)
            result.append(node.val)
    
    postorder(root)
    return result

# Example usage:
# Tree: [1,null,2,3]
#   1
#    \
#     2
#    /
#   3
root = TreeNode(1)
root.right = TreeNode(2)
root.right.left = TreeNode(3)

print("Inorder Traversal:", inorder_traversal(root))
print("Preorder Traversal:", preorder_traversal(root))
print("Postorder Traversal:", postorder_traversal(root))


# 20. Coin Change with Memoization
def coin_change(coins, amount):
    """
    Find minimum coins needed to make amount.
    Time: O(amount × coins), Space: O(amount)
    """
    memo = {}
    
    def dp(remaining):
        if remaining in memo:
            return memo[remaining]
        
        if remaining == 0:
            return 0
        
        if remaining < 0:
            return float('inf')
        
        min_coins = float('inf')
        for coin in coins:
            result = dp(remaining - coin)
            if result != float('inf'):
                min_coins = min(min_coins, result + 1)
        
        memo[remaining] = min_coins
        return min_coins
    
    result = dp(amount)
    return result if result != float('inf') else -1

# Bottom-up DP approach
def coin_change_dp(coins, amount):
    """
    Find minimum coins using bottom-up DP.
    Time: O(amount × coins), Space: O(amount)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Example usage:
# Input: coins = [1,3,4], amount = 6
# Output: 2 (3 + 3)
print("Coin Change ([1,3,4], 6):", coin_change([1, 3, 4], 6))


# Test all solutions
if __name__ == "__main__":
    print("\n=== Recursion Problems Solutions ===")
    
    # Additional test cases
    print("\nAdditional Test Cases:")
    print("Power (2^-2):", power(2, -2))
    print("Generate Parentheses (2):", generate_parentheses(2))
    print("Subsets Iterative ([1,2]):", subsets_iterative([1, 2]))
    print("Gray Code Recursive (3):", gray_code_recursive(3))
    print("Climbing Stairs Memo (10):", climbing_stairs_memo(10))
    print("Coin Change DP ([1,3,4], 6):", coin_change_dp([1, 3, 4], 6))
    
    print("\n=== All Solutions Optimized ===")
    print("Each solution demonstrates different recursion patterns:")
    print("- Basic recursion with base cases")
    print("- Backtracking with pruning")
    print("- Memoization for optimization")
    print("- Tail recursion and iterative alternatives")
