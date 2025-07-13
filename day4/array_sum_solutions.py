"""
Optimized Python Solutions for Array Sum Problems
Time and Space Complexity Analysis Included
"""

from collections import defaultdict, Counter
from typing import List, Dict, Set

# =============================================================================
# 1. TWO SUM
# =============================================================================
def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Find two numbers that add up to target.
    Time: O(n), Space: O(n)
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Sample Input/Output
print("1. Two Sum:")
print("Input: [2,7,11,15], target=9")
print("Output:", two_sum([2, 7, 11, 15], 9))
print()

# =============================================================================
# 2. THREE SUM
# =============================================================================
def three_sum(nums: List[int]) -> List[List[int]]:
    """
    Find all unique triplets that sum to 0.
    Time: O(n²), Space: O(1)
    """
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
            
        left, right = i + 1, n - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            if current_sum == 0:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < 0:
                left += 1
            else:
                right -= 1
    
    return result

print("2. Three Sum:")
print("Input: [-1,0,1,2,-1,-4]")
print("Output:", three_sum([-1, 0, 1, 2, -1, -4]))
print()

# =============================================================================
# 3. FOUR SUM
# =============================================================================
def four_sum(nums: List[int], target: int) -> List[List[int]]:
    """
    Find all unique quadruplets that sum to target.
    Time: O(n³), Space: O(1)
    """
    nums.sort()
    result = []
    n = len(nums)
    
    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
            
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
                
            left, right = j + 1, n - 1
            
            while left < right:
                current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                
                if current_sum == target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1
    
    return result

print("3. Four Sum:")
print("Input: [1,0,-1,0,-2,2], target=0")
print("Output:", four_sum([1, 0, -1, 0, -2, 2], 0))
print()

# =============================================================================
# 4. TWO SUM II - SORTED ARRAY
# =============================================================================
def two_sum_sorted(numbers: List[int], target: int) -> List[int]:
    """
    Two sum on sorted array (1-indexed result).
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []

print("4. Two Sum II - Sorted:")
print("Input: [2,7,11,15], target=9")
print("Output:", two_sum_sorted([2, 7, 11, 15], 9))
print()

# =============================================================================
# 5. TWO SUM - BST
# =============================================================================
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def find_target_bst(root: TreeNode, k: int) -> bool:
    """
    Find if there exist two elements in BST that sum to k.
    Time: O(n), Space: O(n)
    """
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    
    values = inorder(root)
    left, right = 0, len(values) - 1
    
    while left < right:
        current_sum = values[left] + values[right]
        if current_sum == k:
            return True
        elif current_sum < k:
            left += 1
        else:
            right -= 1
    
    return False

print("5. Two Sum - BST:")
print("Input: BST [5,3,6,2,4,null,7], k=9")
print("Output: True (3 + 6 = 9)")
print()

# =============================================================================
# 6. COUNT PAIRS WITH GIVEN SUM
# =============================================================================
def count_pairs_with_sum(arr: List[int], target: int) -> int:
    """
    Count pairs that sum to target.
    Time: O(n), Space: O(n)
    """
    freq = Counter(arr)
    count = 0
    
    for num in freq:
        complement = target - num
        if complement in freq:
            if num == complement:
                # Same number, choose 2 from frequency
                count += freq[num] * (freq[num] - 1) // 2
            elif num < complement:
                # Avoid double counting
                count += freq[num] * freq[complement]
    
    return count

print("6. Count Pairs with Given Sum:")
print("Input: [1,5,7,-1,5], target=6")
print("Output:", count_pairs_with_sum([1, 5, 7, -1, 5], 6))
print()

# =============================================================================
# 7. LONGEST SUBARRAY WITH 0 SUM
# =============================================================================
def longest_subarray_zero_sum(nums: List[int]) -> int:
    """
    Find length of longest subarray with sum 0.
    Time: O(n), Space: O(n)
    """
    sum_map = {0: -1}  # sum -> first index
    current_sum = 0
    max_length = 0
    
    for i, num in enumerate(nums):
        current_sum += num
        
        if current_sum in sum_map:
            max_length = max(max_length, i - sum_map[current_sum])
        else:
            sum_map[current_sum] = i
    
    return max_length

print("7. Longest Subarray with 0 Sum:")
print("Input: [15,-2,2,-8,1,7,10,23]")
print("Output:", longest_subarray_zero_sum([15, -2, 2, -8, 1, 7, 10, 23]))
print()

# =============================================================================
# 8. SUBARRAY SUM EQUALS K
# =============================================================================
def subarray_sum_k(nums: List[int], k: int) -> int:
    """
    Count subarrays with sum equal to k.
    Time: O(n), Space: O(n)
    """
    count = 0
    current_sum = 0
    sum_count = defaultdict(int)
    sum_count[0] = 1  # Empty subarray
    
    for num in nums:
        current_sum += num
        count += sum_count[current_sum - k]
        sum_count[current_sum] += 1
    
    return count

print("8. Subarray Sum Equals K:")
print("Input: [1,1,1], k=2")
print("Output:", subarray_sum_k([1, 1, 1], 2))
print()

# =============================================================================
# 9. K-DIFF PAIRS IN ARRAY
# =============================================================================
def find_pairs_k_diff(nums: List[int], k: int) -> int:
    """
    Find number of unique k-diff pairs.
    Time: O(n), Space: O(n)
    """
    if k < 0:
        return 0
    
    counter = Counter(nums)
    count = 0
    
    for num in counter:
        if k == 0:
            # Special case: need at least 2 occurrences
            if counter[num] >= 2:
                count += 1
        else:
            # Check if num + k exists
            if num + k in counter:
                count += 1
    
    return count

print("9. K-diff Pairs in Array:")
print("Input: [3,1,4,1,5], k=2")
print("Output:", find_pairs_k_diff([3, 1, 4, 1, 5], 2))
print()

# =============================================================================
# 10. FIND ALL DUPLICATES IN ARRAY
# =============================================================================
def find_duplicates(nums: List[int]) -> List[int]:
    """
    Find all duplicates in array where 1 ≤ a[i] ≤ n.
    Time: O(n), Space: O(1)
    """
    result = []
    
    for num in nums:
        index = abs(num) - 1
        if nums[index] < 0:
            result.append(abs(num))
        else:
            nums[index] = -nums[index]
    
    # Restore array
    for i in range(len(nums)):
        nums[i] = abs(nums[i])
    
    return result

print("10. Find All Duplicates:")
print("Input: [4,3,2,7,8,2,3,1]")
print("Output:", find_duplicates([4, 3, 2, 7, 8, 2, 3, 1]))
print()

# =============================================================================
# 11. CONTINUOUS SUBARRAY SUM
# =============================================================================
def check_subarray_sum(nums: List[int], k: int) -> bool:
    """
    Check if array has continuous subarray of size ≥ 2 with sum multiple of k.
    Time: O(n), Space: O(min(n, k))
    """
    if len(nums) < 2:
        return False
    
    remainder_map = {0: -1}  # remainder -> index
    current_sum = 0
    
    for i, num in enumerate(nums):
        current_sum += num
        remainder = current_sum % k
        
        if remainder in remainder_map:
            if i - remainder_map[remainder] > 1:
                return True
        else:
            remainder_map[remainder] = i
    
    return False

print("11. Continuous Subarray Sum:")
print("Input: [23,2,4,6,7], k=6")
print("Output:", check_subarray_sum([23, 2, 4, 6, 7], 6))
print()

# =============================================================================
# 12. LONGEST CONSECUTIVE SEQUENCE
# =============================================================================
def longest_consecutive(nums: List[int]) -> int:
    """
    Find length of longest consecutive sequence.
    Time: O(n), Space: O(n)
    """
    if not nums:
        return 0
    
    num_set = set(nums)
    max_length = 0
    
    for num in num_set:
        # Only start counting if it's the beginning of a sequence
        if num - 1 not in num_set:
            current_length = 1
            current_num = num
            
            while current_num + 1 in num_set:
                current_length += 1
                current_num += 1
            
            max_length = max(max_length, current_length)
    
    return max_length

print("12. Longest Consecutive Sequence:")
print("Input: [100,4,200,1,3,2]")
print("Output:", longest_consecutive([100, 4, 200, 1, 3, 2]))
print()

# =============================================================================
# 13. PAIRS OF SONGS WITH TOTAL DURATIONS DIVISIBLE BY 60
# =============================================================================
def num_pairs_divisible_by_60(time: List[int]) -> int:
    """
    Count pairs of songs with total duration divisible by 60.
    Time: O(n), Space: O(1)
    """
    remainders = [0] * 60
    count = 0
    
    for t in time:
        remainder = t % 60
        complement = (60 - remainder) % 60
        count += remainders[complement]
        remainders[remainder] += 1
    
    return count

print("13. Pairs of Songs Divisible by 60:")
print("Input: [30,20,150,100,40]")
print("Output:", num_pairs_divisible_by_60([30, 20, 150, 100, 40]))
print()

# =============================================================================
# 14. SUM OF TWO INTEGERS (BIT MANIPULATION)
# =============================================================================
def get_sum(a: int, b: int) -> int:
    """
    Add two integers without using + or - operators.
    Time: O(1), Space: O(1)
    """
    # Handle negative numbers in Python
    MAX_INT = 0x7FFFFFFF
    MASK = 0xFFFFFFFF
    
    while b != 0:
        carry = (a & b) << 1
        a = (a ^ b) & MASK
        b = carry & MASK
    
    return a if a <= MAX_INT else ~(a ^ MASK)

print("14. Sum of Two Integers (Bit Manipulation):")
print("Input: a=1, b=2")
print("Output:", get_sum(1, 2))
print()

# =============================================================================
# 15. EQUAL ZERO ONE SUBARRAY
# =============================================================================
def find_max_length_equal_01(nums: List[int]) -> int:
    """
    Find maximum length of subarray with equal 0s and 1s.
    Time: O(n), Space: O(n)
    """
    count_map = {0: -1}  # count -> first index
    count = 0
    max_length = 0
    
    for i, num in enumerate(nums):
        count += 1 if num == 1 else -1
        
        if count in count_map:
            max_length = max(max_length, i - count_map[count])
        else:
            count_map[count] = i
    
    return max_length

print("15. Equal Zero One Subarray:")
print("Input: [0,1,0,0,1,1,0]")
print("Output:", find_max_length_equal_01([0, 1, 0, 0, 1, 1, 0]))
print()

# =============================================================================
# 16. PAIR WITH GIVEN DIFFERENCE
# =============================================================================
def find_pair_with_diff(arr: List[int], diff: int) -> bool:
    """
    Find if there exists a pair with given difference.
    Time: O(n), Space: O(n)
    """
    num_set = set(arr)
    
    for num in arr:
        if num + diff in num_set:
            return True
    
    return False

print("16. Pair with Given Difference:")
print("Input: [1,8,30,40,100], diff=60")
print("Output:", find_pair_with_diff([1, 8, 30, 40, 100], 60))
print()

# =============================================================================
# 17. COUNT QUADRUPLETS THAT SUM TO TARGET
# =============================================================================
def count_quadruplets_sum(nums: List[int], target: int) -> int:
    """
    Count quadruplets that sum to target.
    Time: O(n³), Space: O(n²)
    """
    n = len(nums)
    count = 0
    
    # Use hashmap to store sum of pairs
    pair_sums = defaultdict(int)
    
    for i in range(n):
        for j in range(i + 1, n):
            current_sum = nums[i] + nums[j]
            complement = target - current_sum
            
            # Count how many pairs sum to complement
            if complement in pair_sums:
                count += pair_sums[complement]
        
        # Add current element with all previous elements
        for k in range(i):
            pair_sum = nums[i] + nums[k]
            pair_sums[pair_sum] += 1
    
    return count

print("17. Count Quadruplets Sum to Target:")
print("Input: [1,2,3,6], target=10")
print("Output:", count_quadruplets_sum([1, 2, 3, 6], 10))
print()

# =============================================================================
# 18. XOR PAIRS
# =============================================================================
def count_xor_pairs(arr: List[int], target: int) -> int:
    """
    Count pairs with XOR equal to target.
    Time: O(n), Space: O(n)
    """
    freq = Counter(arr)
    count = 0
    
    for num in freq:
        xor_pair = num ^ target
        if xor_pair in freq:
            if num == xor_pair:
                # Same number, choose 2 from frequency
                count += freq[num] * (freq[num] - 1) // 2
            elif num < xor_pair:
                # Avoid double counting
                count += freq[num] * freq[xor_pair]
    
    return count

print("18. XOR Pairs:")
print("Input: [1,2,3,4], target=6")
print("Output:", count_xor_pairs([1, 2, 3, 4], 6))
print()

# =============================================================================
# 19. MIN OPERATIONS TO MAKE ARRAY SUM ZERO
# =============================================================================
def min_operations_zero_sum(nums: List[int]) -> int:
    """
    Minimum operations to make array sum zero (increment/decrement by 1).
    Time: O(n), Space: O(1)
    """
    total_sum = sum(nums)
    return abs(total_sum)

print("19. Min Operations to Make Array Sum Zero:")
print("Input: [1,1,2,2,3,3]")
print("Output:", min_operations_zero_sum([1, 1, 2, 2, 3, 3]))
print()

# =============================================================================
# 20. MAX NUMBER OF K-SUM PAIRS
# =============================================================================
def max_k_sum_pairs(nums: List[int], k: int) -> int:
    """
    Maximum number of k-sum pairs we can form.
    Time: O(n), Space: O(n)
    """
    counter = Counter(nums)
    pairs = 0
    
    for num in counter:
        complement = k - num
        if complement in counter:
            if num == complement:
                # Same number, pairs = frequency // 2
                pairs += counter[num] // 2
            elif num < complement:
                # Avoid double counting
                pairs += min(counter[num], counter[complement])
    
    return pairs

print("20. Max Number of K-Sum Pairs:")
print("Input: [1,2,3,4], k=5")
print("Output:", max_k_sum_pairs([1, 2, 3, 4], 5))
print()

print("=" * 60)
print("COMPLEXITY SUMMARY:")
print("=" * 60)
print("1. Two Sum: O(n) time, O(n) space")
print("2. Three Sum: O(n²) time, O(1) space")
print("3. Four Sum: O(n³) time, O(1) space")
print("4. Two Sum II: O(n) time, O(1) space")
print("5. Two Sum BST: O(n) time, O(n) space")
print("6-20. Various optimized solutions with detailed complexity analysis")
