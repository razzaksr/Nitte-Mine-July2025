"""
Optimized Python Solutions for Common Array Problems
===================================================
"""

# 1. Two Sum
def two_sum(nums, target):
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

# Example usage:
# Input: nums = [2,7,11,15], target = 9
# Output: [0,1]
print("Two Sum:", two_sum([2,7,11,15], 9))


# 2. Maximum Subarray (Kadane's Algorithm)
def max_subarray(nums):
    """
    Find maximum sum of contiguous subarray.
    Time: O(n), Space: O(1)
    """
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

# Example usage:
# Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
# Output: 6 (subarray [4,-1,2,1])
print("Maximum Subarray:", max_subarray([-2,1,-3,4,-1,2,1,-5,4]))


# 3. Move Zeroes
def move_zeroes(nums):
    """
    Move all zeroes to end while maintaining relative order.
    Time: O(n), Space: O(1)
    """
    write_pos = 0
    for read_pos in range(len(nums)):
        if nums[read_pos] != 0:
            nums[write_pos], nums[read_pos] = nums[read_pos], nums[write_pos]
            write_pos += 1
    return nums

# Example usage:
# Input: nums = [0,1,0,3,12]
# Output: [1,3,12,0,0]
print("Move Zeroes:", move_zeroes([0,1,0,3,12]))


# 4. Best Time to Buy and Sell Stock
def max_profit(prices):
    """
    Find maximum profit from buying and selling stock once.
    Time: O(n), Space: O(1)
    """
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price
    
    return max_profit

# Example usage:
# Input: prices = [7,1,5,3,6,4]
# Output: 5 (buy at 1, sell at 6)
print("Max Profit:", max_profit([7,1,5,3,6,4]))


# 5. Find Pivot Index
def pivot_index(nums):
    """
    Find index where left sum equals right sum.
    Time: O(n), Space: O(1)
    """
    total = sum(nums)
    left_sum = 0
    
    for i, num in enumerate(nums):
        if left_sum == total - left_sum - num:
            return i
        left_sum += num
    
    return -1

# Example usage:
# Input: nums = [1,7,3,6,5,6]
# Output: 3 (left sum = 1+7+3 = 11, right sum = 5+6 = 11)
print("Pivot Index:", pivot_index([1,7,3,6,5,6]))


# 6. Product of Array Except Self
def product_except_self(nums):
    """
    Return array where each element is product of all others.
    Time: O(n), Space: O(1) extra space
    """
    n = len(nums)
    result = [1] * n
    
    # Left pass
    for i in range(1, n):
        result[i] = result[i-1] * nums[i-1]
    
    # Right pass
    right = 1
    for i in range(n-1, -1, -1):
        result[i] *= right
        right *= nums[i]
    
    return result

# Example usage:
# Input: nums = [1,2,3,4]
# Output: [24,12,8,6]
print("Product Except Self:", product_except_self([1,2,3,4]))


# 7. Merge Intervals
def merge_intervals(intervals):
    """
    Merge overlapping intervals.
    Time: O(n log n), Space: O(1)
    """
    if not intervals:
        return []
    
    intervals.sort()
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)
    
    return merged

# Example usage:
# Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
# Output: [[1,6],[8,10],[15,18]]
print("Merge Intervals:", merge_intervals([[1,3],[2,6],[8,10],[15,18]]))


# 8. Insert Interval
def insert_interval(intervals, new_interval):
    """
    Insert new interval and merge if necessary.
    Time: O(n), Space: O(1)
    """
    result = []
    i = 0
    n = len(intervals)
    
    # Add intervals before new_interval
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge overlapping intervals
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    
    result.append(new_interval)
    
    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result

# Example usage:
# Input: intervals = [[1,3],[6,9]], new_interval = [2,5]
# Output: [[1,5],[6,9]]
print("Insert Interval:", insert_interval([[1,3],[6,9]], [2,5]))


# 9. Spiral Matrix
def spiral_matrix(matrix):
    """
    Return matrix elements in spiral order.
    Time: O(m*n), Space: O(1)
    """
    if not matrix:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        
        # Down
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        
        # Left
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        
        # Up
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    
    return result

# Example usage:
# Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
# Output: [1,2,3,6,9,8,7,4,5]
print("Spiral Matrix:", spiral_matrix([[1,2,3],[4,5,6],[7,8,9]]))


# 10. Set Matrix Zeroes
def set_matrix_zeroes(matrix):
    """
    Set entire row and column to zero if element is zero.
    Time: O(m*n), Space: O(1)
    """
    m, n = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][j] == 0 for j in range(n))
    first_col_zero = any(matrix[i][0] == 0 for i in range(m))
    
    # Mark zeros in first row/column
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    
    # Set zeros based on markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    
    # Handle first row
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0
    
    # Handle first column
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0
    
    return matrix

# Example usage:
# Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
# Output: [[1,0,1],[0,0,0],[1,0,1]]
test_matrix = [[1,1,1],[1,0,1],[1,1,1]]
print("Set Matrix Zeroes:", set_matrix_zeroes(test_matrix))


# 11. Rotate Image
def rotate_image(matrix):
    """
    Rotate matrix 90 degrees clockwise in-place.
    Time: O(n²), Space: O(1)
    """
    n = len(matrix)
    
    # Transpose
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Reverse each row
    for i in range(n):
        matrix[i].reverse()
    
    return matrix

# Example usage:
# Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
# Output: [[7,4,1],[8,5,2],[9,6,3]]
test_matrix = [[1,2,3],[4,5,6],[7,8,9]]
print("Rotate Image:", rotate_image(test_matrix))


# 12. Subarray Sum Equals K
def subarray_sum_k(nums, k):
    """
    Count number of subarrays with sum equal to k.
    Time: O(n), Space: O(n)
    """
    count = 0
    current_sum = 0
    sum_count = {0: 1}  # Initialize with 0 sum having count 1
    
    for num in nums:
        current_sum += num
        if current_sum - k in sum_count:
            count += sum_count[current_sum - k]
        sum_count[current_sum] = sum_count.get(current_sum, 0) + 1
    
    return count

# Example usage:
# Input: nums = [1,1,1], k = 2
# Output: 2 (subarrays [1,1] at indices 0-1 and 1-2)
print("Subarray Sum K:", subarray_sum_k([1,1,1], 2))


# 13. Longest Consecutive Sequence
def longest_consecutive(nums):
    """
    Find length of longest consecutive sequence.
    Time: O(n), Space: O(n)
    """
    if not nums:
        return 0
    
    num_set = set(nums)
    max_length = 0
    
    for num in num_set:
        if num - 1 not in num_set:  # Start of sequence
            current_num = num
            current_length = 1
            
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            max_length = max(max_length, current_length)
    
    return max_length

# Example usage:
# Input: nums = [100,4,200,1,3,2]
# Output: 4 (sequence 1,2,3,4)
print("Longest Consecutive:", longest_consecutive([100,4,200,1,3,2]))


# 14. Sliding Window Maximum
from collections import deque

def sliding_window_maximum(nums, k):
    """
    Find maximum in each sliding window of size k.
    Time: O(n), Space: O(k)
    """
    if not nums:
        return []
    
    dq = deque()
    result = []
    
    for i in range(len(nums)):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements from back
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result if window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# Example usage:
# Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
# Output: [3,3,5,5,6,7]
print("Sliding Window Maximum:", sliding_window_maximum([1,3,-1,-3,5,3,6,7], 3))


# 15. Sort Colors (Dutch National Flag)
def sort_colors(nums):
    """
    Sort array of 0s, 1s, and 2s in-place.
    Time: O(n), Space: O(1)
    """
    left = current = 0
    right = len(nums) - 1
    
    while current <= right:
        if nums[current] == 0:
            nums[left], nums[current] = nums[current], nums[left]
            left += 1
            current += 1
        elif nums[current] == 1:
            current += 1
        else:  # nums[current] == 2
            nums[current], nums[right] = nums[right], nums[current]
            right -= 1
    
    return nums

# Example usage:
# Input: nums = [2,0,2,1,1,0]
# Output: [0,0,1,1,2,2]
print("Sort Colors:", sort_colors([2,0,2,1,1,0]))


# 16. Count Inversions
def count_inversions(nums):
    """
    Count number of inversions (i < j but nums[i] > nums[j]).
    Time: O(n log n), Space: O(n)
    """
    def merge_sort(arr, temp, left, right):
        inv_count = 0
        if left < right:
            mid = (left + right) // 2
            inv_count += merge_sort(arr, temp, left, mid)
            inv_count += merge_sort(arr, temp, mid + 1, right)
            inv_count += merge(arr, temp, left, mid, right)
        return inv_count
    
    def merge(arr, temp, left, mid, right):
        i, j, k = left, mid + 1, left
        inv_count = 0
        
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp[k] = arr[i]
                i += 1
            else:
                temp[k] = arr[j]
                inv_count += (mid - i + 1)
                j += 1
            k += 1
        
        while i <= mid:
            temp[k] = arr[i]
            i += 1
            k += 1
        
        while j <= right:
            temp[k] = arr[j]
            j += 1
            k += 1
        
        for i in range(left, right + 1):
            arr[i] = temp[i]
        
        return inv_count
    
    temp = [0] * len(nums)
    return merge_sort(nums[:], temp, 0, len(nums) - 1)

# Example usage:
# Input: nums = [2,3,8,6,1]
# Output: 5 (inversions: (2,1), (3,1), (8,6), (8,1), (6,1))
print("Count Inversions:", count_inversions([2,3,8,6,1]))


# 17. Next Permutation
def next_permutation(nums):
    """
    Find next lexicographically greater permutation.
    Time: O(n), Space: O(1)
    """
    # Find first decreasing element from right
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    
    if i >= 0:
        # Find element just larger than nums[i]
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    
    # Reverse the suffix
    nums[i + 1:] = nums[i + 1:][::-1]
    return nums

# Example usage:
# Input: nums = [1,2,3]
# Output: [1,3,2]
print("Next Permutation:", next_permutation([1,2,3]))


# 18. 3Sum
def three_sum(nums):
    """
    Find all unique triplets that sum to zero.
    Time: O(n²), Space: O(1)
    """
    nums.sort()
    result = []
    
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
    
    return result

# Example usage:
# Input: nums = [-1,0,1,2,-1,-4]
# Output: [[-1,-1,2],[-1,0,1]]
print("3Sum:", three_sum([-1,0,1,2,-1,-4]))


# 19. 4Sum
def four_sum(nums, target):
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
                total = nums[i] + nums[j] + nums[left] + nums[right]
                
                if total < target:
                    left += 1
                elif total > target:
                    right -= 1
                else:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
    
    return result

# Example usage:
# Input: nums = [1,0,-1,0,-2,2], target = 0
# Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
print("4Sum:", four_sum([1,0,-1,0,-2,2], 0))


# 20. Majority Element
def majority_element(nums):
    """
    Find element that appears more than n/2 times.
    Time: O(n), Space: O(1) - Boyer-Moore Voting Algorithm
    """
    candidate = None
    count = 0
    
    for num in nums:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    
    return candidate

# Example usage:
# Input: nums = [3,2,3]
# Output: 3
print("Majority Element:", majority_element([3,2,3]))


# Test all solutions
if __name__ == "__main__":
    print("\n=== All Solutions Tested ===")
    print("All functions are optimized for time and space complexity.")
    print("Each solution includes detailed comments explaining the approach.")
