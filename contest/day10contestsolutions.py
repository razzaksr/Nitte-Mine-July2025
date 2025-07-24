# instagram
from collections import defaultdict
import heapq

class Instagram:
    def __init__(self):
        self.reels = []  # List of tuples: (timestamp, userId, reelId)
        self.following = defaultdict(set)  # userId -> set of followees
        self.time = 0   # Simulated timestamp

    def postReel(self, userId: int, reelId: int) -> None:
        self.reels.append((self.time, userId, reelId))
        self.time += 1

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:
            self.following[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.following[followerId].discard(followeeId)

    def getNewsFeed(self, userId: int) -> list:
        recent = []
        followees = self.following[userId] | {userId}

        # Get last 10 reels from self.reels that belong to followees
        for time, uid, rid in reversed(self.reels):
            if uid in followees:
                recent.append(rid)
                if len(recent) == 10:
                    break
        return recent
# insta = Instagram()
# insta.postReel(1, 101)           # User 1 posts reel 101
# print(insta.getNewsFeed(1))     # [101]
# insta.follow(1, 2)
# insta.postReel(2, 102)          # User 2 posts reel 102
# print(insta.getNewsFeed(1))     # [102, 101]
# insta.unfollow(1, 2)
# print(insta.getNewsFeed(1))     # [101]

# Add two nums
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(0)
    curr = dummy
    carry = 0

    while l1 or l2 or carry:
        sum_val = carry

        if l1:
            sum_val += l1.val
            l1 = l1.next

        if l2:
            sum_val += l2.val
            l2 = l2.next

        carry = sum_val // 10
        curr.next = ListNode(sum_val % 10)
        curr = curr.next

    return dummy.next

def build_linked_list(nums):
    dummy = ListNode(0)
    curr = dummy
    for num in nums:
        curr.next = ListNode(num)
        curr = curr.next
    return dummy.next

def print_linked_list(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    print(result)
    
# Test Case 1
# l1 = build_linked_list([2,4,3])
# l2 = build_linked_list([5,6,4])
# result = addTwoNumbers(l1, l2)
# print_linked_list(result)  # Output: [7,0,8]


# Convert Binary Number in a Linked List to Integer
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def getDecimalValue(head: ListNode) -> int:
    result = 0
    while head:
        result = result * 2 + head.val
        head = head.next
    return result
def build_linked_list(bits):
    dummy = ListNode(0)
    current = dummy
    for bit in bits:
        current.next = ListNode(bit)
        current = current.next
    return dummy.next

# Test Case 1
# head1 = build_linked_list([1, 0, 1])
# print(getDecimalValue(head1))  # Output: 5

# # Test Case 2
# head2 = build_linked_list([0])
# print(getDecimalValue(head2))  # Output: 0