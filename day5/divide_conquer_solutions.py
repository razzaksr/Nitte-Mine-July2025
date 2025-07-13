"""
Optimized Python Solutions for Divide and Conquer Problems
Time and Space Complexity Analysis Included
"""

import heapq
from typing import List, Optional, Tuple
import math

# =============================================================================
# 1. MERGE SORT
# =============================================================================
def merge_sort(arr: List[int]) -> List[int]:
    """
    Merge sort implementation.
    Time: O(n log n), Space: O(n)
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left: List[int], right: List[int]) -> List[int]:
    """Helper function to merge two sorted arrays"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

print("1. Merge Sort:")
print("Input: [64,34,25,12,22,11,90]")
print("Output:", merge_sort([64, 34, 25, 12, 22, 11, 90]))
print()

# =============================================================================
# 2. QUICK SORT
# =============================================================================
def quick_sort(arr: List[int]) -> List[int]:
    """
    Quick sort implementation.
    Time: O(n log n) average, O(n²) worst, Space: O(log n)
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

def quick_sort_inplace(arr: List[int], low: int = 0, high: int = None) -> None:
    """In-place quick sort"""
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pi = partition(arr, low, high)
        quick_sort_inplace(arr, low, pi - 1)
        quick_sort_inplace(arr, pi + 1, high)

def partition(arr: List[int], low: int, high: int) -> int:
    """Partition function for quick sort"""
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

print("2. Quick Sort:")
print("Input: [64,34,25,12,22,11,90]")
print("Output:", quick_sort([64, 34, 25, 12, 22, 11, 90]))
print()

# =============================================================================
# 3. BINARY SEARCH
# =============================================================================
def binary_search(arr: List[int], target: int) -> int:
    """
    Binary search implementation.
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def binary_search_recursive(arr: List[int], target: int, left: int = 0, right: int = None) -> int:
    """Recursive binary search"""
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

print("3. Binary Search:")
print("Input: [1,2,3,4,5,6,7,8,9], target=5")
print("Output:", binary_search([1, 2, 3, 4, 5, 6, 7, 8, 9], 5))
print()

# =============================================================================
# 4. SEARCH IN ROTATED SORTED ARRAY
# =============================================================================
def search_rotated(nums: List[int], target: int) -> int:
    """
    Search in rotated sorted array.
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        
        # Check if left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

print("4. Search in Rotated Sorted Array:")
print("Input: [4,5,6,7,0,1,2], target=0")
print("Output:", search_rotated([4, 5, 6, 7, 0, 1, 2], 0))
print()

# =============================================================================
# 5. MEDIAN OF TWO SORTED ARRAYS
# =============================================================================
def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    """
    Find median of two sorted arrays.
    Time: O(log(min(m,n))), Space: O(1)
    """
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1
        
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            right = partition1 - 1
        else:
            left = partition1 + 1
    
    return 0.0

print("5. Median of Two Sorted Arrays:")
print("Input: [1,3], [2]")
print("Output:", find_median_sorted_arrays([1, 3], [2]))
print()

# =============================================================================
# 6. KTH LARGEST ELEMENT
# =============================================================================
def find_kth_largest(nums: List[int], k: int) -> int:
    """
    Find kth largest element using quickselect.
    Time: O(n) average, O(n²) worst, Space: O(1)
    """
    def quickselect(left: int, right: int, k_smallest: int) -> int:
        if left == right:
            return nums[left]
        
        pivot_index = partition_kth(left, right)
        
        if k_smallest == pivot_index:
            return nums[k_smallest]
        elif k_smallest < pivot_index:
            return quickselect(left, pivot_index - 1, k_smallest)
        else:
            return quickselect(pivot_index + 1, right, k_smallest)
    
    def partition_kth(left: int, right: int) -> int:
        pivot = nums[right]
        i = left
        
        for j in range(left, right):
            if nums[j] <= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        
        nums[i], nums[right] = nums[right], nums[i]
        return i
    
    return quickselect(0, len(nums) - 1, len(nums) - k)

print("6. Kth Largest Element:")
print("Input: [3,2,1,5,6,4], k=2")
print("Output:", find_kth_largest([3, 2, 1, 5, 6, 4], 2))
print()

# =============================================================================
# 7. COUNT INVERSIONS
# =============================================================================
def count_inversions(arr: List[int]) -> int:
    """
    Count inversions using merge sort.
    Time: O(n log n), Space: O(n)
    """
    def merge_and_count(left: List[int], right: List[int]) -> Tuple[List[int], int]:
        result = []
        i = j = inv_count = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                inv_count += len(left) - i  # All remaining elements in left are greater
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result, inv_count
    
    def merge_sort_and_count(arr: List[int]) -> Tuple[List[int], int]:
        if len(arr) <= 1:
            return arr, 0
        
        mid = len(arr) // 2
        left, left_inv = merge_sort_and_count(arr[:mid])
        right, right_inv = merge_sort_and_count(arr[mid:])
        merged, split_inv = merge_and_count(left, right)
        
        return merged, left_inv + right_inv + split_inv
    
    _, inversions = merge_sort_and_count(arr)
    return inversions

print("7. Count Inversions:")
print("Input: [2,3,8,6,1]")
print("Output:", count_inversions([2, 3, 8, 6, 1]))
print()

# =============================================================================
# 8. MAJORITY ELEMENT (DIVIDE AND CONQUER)
# =============================================================================
def majority_element_dc(nums: List[int]) -> int:
    """
    Find majority element using divide and conquer.
    Time: O(n log n), Space: O(log n)
    """
    def majority_element_rec(left: int, right: int) -> int:
        if left == right:
            return nums[left]
        
        mid = (left + right) // 2
        left_majority = majority_element_rec(left, mid)
        right_majority = majority_element_rec(mid + 1, right)
        
        if left_majority == right_majority:
            return left_majority
        
        # Count occurrences
        left_count = sum(1 for i in range(left, right + 1) if nums[i] == left_majority)
        right_count = sum(1 for i in range(left, right + 1) if nums[i] == right_majority)
        
        return left_majority if left_count > right_count else right_majority
    
    return majority_element_rec(0, len(nums) - 1)

print("8. Majority Element (Divide and Conquer):")
print("Input: [2,2,1,1,1,2,2]")
print("Output:", majority_element_dc([2, 2, 1, 1, 1, 2, 2]))
print()

# =============================================================================
# 9. CLOSEST PAIR OF POINTS
# =============================================================================
def closest_pair_points(points: List[List[int]]) -> float:
    """
    Find closest pair of points.
    Time: O(n log n), Space: O(n)
    """
    def distance(p1: List[int], p2: List[int]) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def closest_pair_rec(px: List[List[int]], py: List[List[int]]) -> float:
        n = len(px)
        
        # Base case: brute force for small arrays
        if n <= 3:
            min_dist = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    min_dist = min(min_dist, distance(px[i], px[j]))
            return min_dist
        
        mid = n // 2
        midpoint = px[mid]
        
        pyl = [point for point in py if point[0] <= midpoint[0]]
        pyr = [point for point in py if point[0] > midpoint[0]]
        
        dl = closest_pair_rec(px[:mid], pyl)
        dr = closest_pair_rec(px[mid:], pyr)
        
        d = min(dl, dr)
        
        # Check points near the dividing line
        strip = [point for point in py if abs(point[0] - midpoint[0]) < d]
        
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and (strip[j][1] - strip[i][1]) < d:
                d = min(d, distance(strip[i], strip[j]))
                j += 1
        
        return d
    
    points.sort()
    py = sorted(points, key=lambda x: x[1])
    return closest_pair_rec(points, py)

print("9. Closest Pair of Points:")
print("Input: [[2,3],[12,30],[40,50],[5,1],[12,10],[3,4]]")
print("Output:", closest_pair_points([[2, 3], [12, 30], [40, 50], [5, 1], [12, 10], [3, 4]]))
print()

# =============================================================================
# 10. MERGE K SORTED LISTS
# =============================================================================
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Merge k sorted linked lists.
    Time: O(n log k), Space: O(log k)
    """
    if not lists:
        return None
    
    def merge_two_lists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        curr = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        
        curr.next = l1 or l2
        return dummy.next
    
    while len(lists) > 1:
        merged_lists = []
        
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i + 1] if i + 1 < len(lists) else None
            merged_lists.append(merge_two_lists(l1, l2))
        
        lists = merged_lists
    
    return lists[0]

print("10. Merge K Sorted Lists:")
print("Input: [[1,4,5],[1,3,4],[2,6]]")
print("Output: Linked list representation")
print()

# =============================================================================
# 11. FIND PEAK ELEMENT
# =============================================================================
def find_peak_element(nums: List[int]) -> int:
    """
    Find peak element using binary search.
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    
    return left

print("11. Find Peak Element:")
print("Input: [1,2,3,1]")
print("Output:", find_peak_element([1, 2, 3, 1]))
print()

# =============================================================================
# 12. POWER(x, n)
# =============================================================================
def my_pow(x: float, n: int) -> float:
    """
    Calculate x^n using divide and conquer.
    Time: O(log n), Space: O(log n)
    """
    if n == 0:
        return 1
    
    if n < 0:
        return 1 / my_pow(x, -n)
    
    if n % 2 == 0:
        half = my_pow(x, n // 2)
        return half * half
    else:
        return x * my_pow(x, n - 1)

def my_pow_iterative(x: float, n: int) -> float:
    """Iterative version"""
    if n == 0:
        return 1
    
    if n < 0:
        x = 1 / x
        n = -n
    
    result = 1
    while n > 0:
        if n % 2 == 1:
            result *= x
        x *= x
        n //= 2
    
    return result

print("12. Power(x, n):")
print("Input: x=2.0, n=10")
print("Output:", my_pow(2.0, 10))
print()

# =============================================================================
# 13. FIND MINIMUM IN ROTATED SORTED ARRAY
# =============================================================================
def find_min_rotated(nums: List[int]) -> int:
    """
    Find minimum in rotated sorted array.
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    
    return nums[left]

print("13. Find Minimum in Rotated Sorted Array:")
print("Input: [3,4,5,1,2]")
print("Output:", find_min_rotated([3, 4, 5, 1, 2]))
print()

# =============================================================================
# 14. MAXIMUM SUBARRAY (DIVIDE AND CONQUER)
# =============================================================================
def max_subarray_dc(nums: List[int]) -> int:
    """
    Maximum subarray using divide and conquer.
    Time: O(n log n), Space: O(log n)
    """
    def max_subarray_rec(left: int, right: int) -> int:
        if left == right:
            return nums[left]
        
        mid = (left + right) // 2
        
        left_max = max_subarray_rec(left, mid)
        right_max = max_subarray_rec(mid + 1, right)
        
        # Find max crossing subarray
        left_sum = float('-inf')
        curr_sum = 0
        for i in range(mid, left - 1, -1):
            curr_sum += nums[i]
            left_sum = max(left_sum, curr_sum)
        
        right_sum = float('-inf')
        curr_sum = 0
        for i in range(mid + 1, right + 1):
            curr_sum += nums[i]
            right_sum = max(right_sum, curr_sum)
        
        cross_sum = left_sum + right_sum
        
        return max(left_max, right_max, cross_sum)
    
    return max_subarray_rec(0, len(nums) - 1)

print("14. Maximum Subarray (Divide and Conquer):")
print("Input: [-2,1,-3,4,-1,2,1,-5,4]")
print("Output:", max_subarray_dc([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
print()

# =============================================================================
# 15. CONVERT SORTED ARRAY TO BST
# =============================================================================
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def sorted_array_to_bst(nums: List[int]) -> Optional[TreeNode]:
    """
    Convert sorted array to BST.
    Time: O(n), Space: O(log n)
    """
    def helper(left: int, right: int) -> Optional[TreeNode]:
        if left > right:
            return None
        
        mid = (left + right) // 2
        root = TreeNode(nums[mid])
        root.left = helper(left, mid - 1)
        root.right = helper(mid + 1, right)
        
        return root
    
    return helper(0, len(nums) - 1)

print("15. Convert Sorted Array to BST:")
print("Input: [-10,-3,0,5,9]")
print("Output: BST representation")
print()

# =============================================================================
# 16. CONSTRUCT BINARY TREE FROM INORDER AND PREORDER
# =============================================================================
def build_tree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Construct binary tree from preorder and inorder traversals.
    Time: O(n), Space: O(n)
    """
    if not preorder or not inorder:
        return None
    
    root = TreeNode(preorder[0])
    mid = inorder.index(preorder[0])
    
    root.left = build_tree(preorder[1:mid+1], inorder[:mid])
    root.right = build_tree(preorder[mid+1:], inorder[mid+1:])
    
    return root

print("16. Construct Binary Tree from Inorder and Preorder:")
print("Input: preorder=[3,9,20,15,7], inorder=[9,3,15,20,7]")
print("Output: Binary tree representation")
print()

# =============================================================================
# 17. LONGEST COMMON PREFIX (DIVIDE AND CONQUER)
# =============================================================================
def longest_common_prefix_dc(strs: List[str]) -> str:
    """
    Find longest common prefix using divide and conquer.
    Time: O(S) where S is sum of all characters, Space: O(m log n)
    """
    if not strs:
        return ""
    
    def lcp_helper(left: int, right: int) -> str:
        if left == right:
            return strs[left]
        
        mid = (left + right) // 2
        lcp_left = lcp_helper(left, mid)
        lcp_right = lcp_helper(mid + 1, right)
        
        return common_prefix(lcp_left, lcp_right)
    
    def common_prefix(str1: str, str2: str) -> str:
        min_len = min(len(str1), len(str2))
        for i in range(min_len):
            if str1[i] != str2[i]:
                return str1[:i]
        return str1[:min_len]
    
    return lcp_helper(0, len(strs) - 1)

print("17. Longest Common Prefix (Divide and Conquer):")
print("Input: ['flower','flow','flight']")
print("Output:", longest_common_prefix_dc(["flower", "flow", "flight"]))
print()

# =============================================================================
# 18. NUMBER OF REVERSE PAIRS
# =============================================================================
def reverse_pairs(nums: List[int]) -> int:
    """
    Count reverse pairs where i < j and nums[i] > 2*nums[j].
    Time: O(n log n), Space: O(n)
    """
    def merge_sort_and_count(left: int, right: int) -> int:
        if left >= right:
            return 0
        
        mid = (left + right) // 2
        count = merge_sort_and_count(left, mid) + merge_sort_and_count(mid + 1, right)
        
        # Count reverse pairs
        j = mid + 1
        for i in range(left, mid + 1):
            while j <= right and nums[i] > 2 * nums[j]:
                j += 1
            count += j - (mid + 1)
        
        # Merge
        temp = []
        i, j = left, mid + 1
        while i <= mid and j <= right:
            if nums[i] <= nums[j]:
                temp.append(nums[i])
                i += 1
            else:
                temp.append(nums[j])
                j += 1
        
        temp.extend(nums[i:mid+1])
        temp.extend(nums[j:right+1])
        
        for i, val in enumerate(temp):
            nums[left + i] = val
        
        return count
    
    return merge_sort_and_count(0, len(nums) - 1)

print("18. Number of Reverse Pairs:")
print("Input: [1,3,2,3,1]")
print("Output:", reverse_pairs([1, 3, 2, 3, 1]))
print()

# =============================================================================
# 19. SMALLEST RANGE COVERING ELEMENTS FROM K LISTS
# =============================================================================
def smallest_range(nums: List[List[int]]) -> List[int]:
    """
    Find smallest range covering elements from k lists.
    Time: O(n log k), Space: O(k)
    """
    heap = []
    max_val = float('-inf')
    
    # Initialize heap with first element from each list
    for i, lst in enumerate(nums):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
            max_val = max(max_val, lst[0])
    
    range_start, range_end = 0, float('inf')
    
    while heap:
        min_val, list_idx, elem_idx = heapq.heappop(heap)
        
        # Update range if current is smaller
        if max_val - min_val < range_end - range_start:
            range_start, range_end = min_val, max_val
        
        # Add next element from the same list
        if elem_idx + 1 < len(nums[list_idx]):
            next_val = nums[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
            max_val = max(max_val, next_val)
        else:
            break
    
    return [range_start, range_end]

print("19. Smallest Range Covering Elements from K Lists:")
print("Input: [[4,10,15,24,26],[0,9,12,20],[5,18,22,30]]")
print("Output:", smallest_range([[4, 10, 15, 24, 26], [0, 9, 12, 20], [5, 18, 22, 30]]))
print()

# =============================================================================
# 20. SKYLINE PROBLEM
# =============================================================================
def get_skyline(buildings: List[List[int]]) -> List[List[int]]:
    """
    Solve skyline problem using divide and conquer.
    Time: O(n log n), Space: O(n)
    """
    def merge_skylines(left: List[List[int]], right: List[List[int]]) -> List[List[int]]:
        result = []
        i = j = 0
        left_height = right_height = 0
        
        while i < len(left) and j < len(right):
            if left[i][0] < right[j][0]:
                left_height = left[i][1]
                max_height = max(left_height, right_height)
                if not result or result[-1][1] != max_height:
                    result.append([left[i][0], max_height])
                i += 1
            elif left[i][0] > right[j][0]:
                right_height = right[j][1]
                max_height = max(left_height, right_height)
                if not result or result[-1][1] != max_height:
                    result.append([right[j][0], max_height])
                j += 1
            else:  # Same x coordinate
                left_height = left[i][1]
                right_height = right[j][1]
                max_height = max(left_height, right_height)
                if not result or result[-1][1] != max_height:
                    result.append([left[i][0], max_height])
                i += 1
                j += 1
        
        # Add remaining points
        while i < len(left):
            result.append(left[i])
            i += 1
        while j < len(right):
            result.append(right[j])
            j += 1
        
        return result
    
    def divide_conquer(buildings: List[List[int]]) -> List[List[int]]:
        if not buildings:
            return []
        
        if len(buildings) == 1:
            left, right, height = buildings[0]
            return [[left, height], [right, 0]]
        
        mid = len(buildings) // 2
        left_skyline = divide_conquer(buildings[:mid])
        right_skyline = divide_conquer(buildings[mid:])
        
        return merge_skylines(left_skyline, right_skyline)
    
    return divide_conquer(buildings)

print("20. Skyline Problem:")
print("Input: [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]")
print("Output:", get_skyline([[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]))
print()

# =============================================================================
# COMPLEXITY ANALYSIS SUMMARY
# =============================================================================
print("=" * 80)
print("COMPLEXITY ANALYSIS SUMMARY")
print("=" * 80)

complexity_analysis = {
    "Merge Sort": "Time: O(n log n), Space: O(n)",
    "Quick Sort": "Time: O(n log n) avg, O(n²) worst, Space: O(log n)",
    "Binary Search": "Time: O(log n), Space: O(1)",
    "Search Rotated Array": "Time: O(log n), Space: O(1)",
    "Median Two Arrays": "Time: O(log(min(m,n))), Space: O(1)",
    "Kth Largest": "Time: O(n) avg, O(n²) worst, Space: O(1)",
    "Count Inversions": "Time: O(n log n), Space: O(n)",
    "Majority Element DC": "Time: O(n log n), Space: O(log n)",
    "Closest Pair Points": "Time: O(n log n), Space: O(n)",
    "Merge K Lists": "Time: O(n log k), Space: O(log k)",
    "Find Peak": "Time: O(log n), Space: O(1)",
    "Power(x,n)": "Time: O(log n), Space: O(log n)",
    "Find Min Rotated": "Time: O(log n), Space: O(1)",
    "Max Subarray DC": "Time: O(n log n), Space: O(log n)",
    "Array to BST": "Time: O(n), Space: O(log n)",
    "Build Tree": "Time: O(n), Space: O(n)",
    "LCP Divide Conquer": "Time: O(S), Space: O(m log n)",
    "Reverse Pairs": "Time: O(n log n), Space: O(n)",
    "Smallest Range K Lists": "Time: O(n log k), Space: O(k)",
    "Skyline Problem": "Time: O(n log n), Space: O(n)"
}

for problem, complexity in complexity_analysis.items():
    print(f"{problem:25} : {complexity}")

print()
print("=" * 80)
print("KEY DIVIDE AND CONQUER PATTERNS")
print("=" * 80)

patterns = {
    "Sorting Algorithms": [
        "Merge Sort: Divide array, sort halves, merge results",
        "Quick Sort: Choose pivot, partition, recursively sort partitions"
    ],
    "Search Algorithms": [
        "Binary Search: Compare middle, search appropriate half",
        "Peak Finding: Compare with neighbors, go to higher side"
    ],
    "Array Problems": [
        "Median Finding: Binary search on smaller array",
        "Inversions: Count during merge process",
        "Maximum Subarray: Find max in left, right, and crossing"
    ],
    "Tree Construction": [
        "Array to BST: Use middle as root, recursively build subtrees",
        "Build from traversals: Use preorder for root, inorder for position"
    ],
    "Geometric Problems": [
        "Closest Pair: Divide by x-coordinate, check strip near boundary",
        "Skyline: Merge skylines from left and right halves"
    ],
    "Advanced Applications": [
        "Reverse Pairs: Count during merge, similar to inversions",
        "Range Problems: Use heap/priority queue with divide strategy"
    ]
}

for category, pattern_list in patterns.items():
    print(f"\n{category}:")
    for pattern in pattern_list:
        print(f"  • {pattern}")

print()
print("=" * 80)
print("OPTIMIZATION TECHNIQUES")
print("=" * 80)

optimizations = [
    "1. Choose optimal pivot in quicksort (median-of-three)",
    "2. Use iterative versions to avoid recursion overhead",
    "3. Implement in-place algorithms when possible",
    "4. Use binary search for logarithmic time complexity",
    "5. Optimize merge operations with sentinel values",
    "6. Apply tail recursion optimization where applicable",
    "7. Use bit manipulation for power operations",
    "8. Implement early termination conditions",
    "9. Cache intermediate results to avoid recomputation",
    "10. Use appropriate data structures (heaps, sets, maps)"
]

for opt in optimizations:
    print(opt)

print()
print("=" * 80)
print("WHEN TO USE DIVIDE AND CONQUER")
print("=" * 80)

use_cases = [
    "✓ Problem can be broken into similar subproblems",
    "✓ Subproblems are independent",
    "✓ Results can be combined efficiently",
    "✓ Recursive structure is natural",
    "✓ Need better than O(n²) complexity",
    "✓ Problem has optimal substructure",
    "✓ Searching in sorted/partially sorted data",
    "✓ Tree-based problems",
    "✓ Geometric algorithms",
    "✓ Sorting and selection problems"
]

for use_case in use_cases:
    print(use_case)

print()
print("=" * 80)
print("IMPLEMENTATION TIPS")
print("=" * 80)

tips = [
    "1. Base Case: Always define clear base cases",
    "2. Divide: Ensure proper problem decomposition",
    "3. Conquer: Solve subproblems recursively",
    "4. Combine: Efficiently merge results",
    "5. Space: Consider iterative versions for space optimization",
    "6. Stability: Maintain stability in sorting algorithms",
    "7. Edge Cases: Handle empty arrays, single elements",
    "8. Overflow: Watch for integer overflow in calculations",
    "9. Indexing: Be careful with array bounds",
    "10. Testing: Test with various input sizes and patterns"
]

for tip in tips:
    print(tip)