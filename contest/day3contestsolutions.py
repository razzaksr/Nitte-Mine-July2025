# Problem 1: Merge k Sorted Lists using Divide and Conquer

def merge_two_lists(l1, l2):
    """Helper function to merge two sorted linked lists"""
    if not l1:
        return l2
    if not l2:
        return l1
    
    if l1[0] <= l2[0]:
        return [l1[0]] + merge_two_lists(l1[1:], l2)
    else:
        return [l2[0]] + merge_two_lists(l1, l2[1:])

def merge_k_sorted_lists(lists):
    """
    Merge k sorted linked lists using divide and conquer approach.
    Time Complexity: O(n log k) where n is total number of nodes
    Space Complexity: O(log k) for recursion stack
    """
    if not lists or len(lists) == 0:
        return []
    
    if len(lists) == 1:
        return lists[0]
    
    if len(lists) == 2:
        return merge_two_lists(lists[0], lists[1])
    
    mid = len(lists) // 2
    left = merge_k_sorted_lists(lists[:mid])
    right = merge_k_sorted_lists(lists[mid:])
    
    return merge_two_lists(left, right)


# Problem 2: Majority Element Finder using Divide and Conquer

def count_occurrences(nums, target):
    """Count occurrences of target in nums"""
    count = 0
    for num in nums:
        if num == target:
            count += 1
    return count

def find_majority_element(nums):
    """
    Find majority element using divide and conquer approach.
    Time Complexity: O(n log n)
    Space Complexity: O(log n) for recursion stack
    """
    if len(nums) == 1:
        return nums[0]
    
    mid = len(nums) // 2
    
    left_majority = find_majority_element(nums[:mid])
    right_majority = find_majority_element(nums[mid:])
    
    if left_majority == right_majority:
        return left_majority
    
    left_count = count_occurrences(nums, left_majority)
    right_count = count_occurrences(nums, right_majority)
    
    if left_count > right_count:
        return left_majority
    else:
        return right_majority


# Problem 3: Josephus Survivor Position

def josephus_position(n, k):
    """
    Find the position of the last survivor in Josephus problem.
    Uses recursive approach with mathematical formula.
    Time Complexity: O(n)
    Space Complexity: O(n) for recursion stack
    """
    if n == 1:
        return 0
    
    return (josephus_position(n - 1, k) + k) % n


# Test functions for all problems

def test_merge_k_lists():
    """Test merge k sorted lists"""
    print("Testing Merge k Sorted Lists:")
    
    test_cases = [
        [[1,4,5], [1,3,4], [2,6]],
        [[1,2,3], [], [4,5]],
        [[], [], []],
        [[5], [1,2,3,4]],
        [[10,20,30], [5,15,25], [1,2,3]]
    ]
    
    expected = [
        [1,1,2,3,4,4,5,6],
        [1,2,3,4,5],
        [],
        [1,2,3,4,5],
        [1,2,3,5,10,15,20,25,30]
    ]
    
    for i, test in enumerate(test_cases):
        result = merge_k_sorted_lists(test)
        print(f"Test {i+1}: {result} - {'PASS' if result == expected[i] else 'FAIL'}")

def test_majority_element():
    """Test majority element finder"""
    print("\nTesting Majority Element Finder:")
    
    test_cases = [
        [2,2,1,1,1,2,2],
        [3,3,4],
        [1,1,1,2,3,4,1],
        [5,5,5,5,2,2,2],
        [9,9,9,9,1,2,3,9]
    ]
    
    expected = [2, 3, 1, 5, 9]
    
    for i, test in enumerate(test_cases):
        result = find_majority_element(test)
        print(f"Test {i+1}: {result} - {'PASS' if result == expected[i] else 'FAIL'}")

def test_josephus():
    """Test Josephus position"""
    print("\nTesting Josephus Survivor Position:")
    
    test_cases = [
        (7, 3),
        (5, 2),
        (6, 1),
        (10, 2),
        (1, 1)
    ]
    
    expected = [4, 2, 5, 4, 0]
    
    for i, (n, k) in enumerate(test_cases):
        result = josephus_position(n, k)
        print(f"Test {i+1}: {result} - {'PASS' if result == expected[i] else 'FAIL'}")

# Run all tests
if __name__ == "__main__":
    test_merge_k_lists()
    test_majority_element()
    test_josephus()
