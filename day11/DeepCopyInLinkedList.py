class Node:
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

def copyRandomList(head):
    if not head:
        return None

    # Step 1: Interleave copied nodes with original nodes
    curr = head
    while curr:
        copy = Node(curr.val)
        copy.next = curr.next
        curr.next = copy
        curr = copy.next

    # Step 2: Assign random pointers to copied nodes
    curr = head
    while curr:
        if curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next

    # Step 3: Detach copied list from original
    curr = head
    copy_head = head.next
    while curr:
        copy = curr.next
        curr.next = copy.next
        if copy.next:
            copy.next = copy.next.next
        curr = curr.next

    return copy_head
# Let's assume the Node class and copyRandomList function are defined as per above.

# Create sample list: Node 1 → Node 2
node1 = Node(1)
node2 = Node(2)
node1.next = node2
node1.random = node2
node2.random = node2

# Call function
copied = copyRandomList(node1)

# You can print values to verify
print(copied.val)             # 1
print(copied.next.val)        # 2
print(copied.random.val)      # 2
print(copied.next.random.val) # 2