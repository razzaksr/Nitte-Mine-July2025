# 1. Invert Binary Tree
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None

        # Swap left and right
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root

'''
Input:
      4
     / \
    2   7
   / \ / \
  1  3 6  9
Output:
      4
     / \
    7   2
   / \ / \
  9  6 3  1 
'''
root = TreeNode(4)
root.left = TreeNode(2, TreeNode(1), TreeNode(3))
root.right = TreeNode(7, TreeNode(6), TreeNode(9))

solver = Solution()
# inverted = solver.invertTree(root)

# 2. Diameter of Binary Tree
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.max_diameter = 0

        def depth(node):
            if not node:
                return 0

            left = depth(node.left)
            right = depth(node.right)

            # Update diameter if current path is larger
            self.max_diameter = max(self.max_diameter, left + right)

            # Return the height
            return 1 + max(left, right)

        depth(root)
        return self.max_diameter

root = TreeNode(1)
root.left = TreeNode(2, TreeNode(4), TreeNode(5))
root.right = TreeNode(3)
solver = Solution()
# print(solver.diameterOfBinaryTree(root))  # Output: 3

root = TreeNode(1)
root.left = TreeNode(2)
solver = Solution()
# print(solver.diameterOfBinaryTree(root))  # Output: 1

# 3. Construct BT from Inorder and PreOrder
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def buildTree(self, preorder, inorder):
        if not preorder or not inorder:
            return None

        # Map value to index in inorder for quick lookup
        in_index_map = {val: idx for idx, val in enumerate(inorder)}
        self.pre_idx = 0

        def build(left, right):
            if left > right:
                return None

            root_val = preorder[self.pre_idx]
            self.pre_idx += 1

            root = TreeNode(root_val)

            # Recursively build left and right subtree
            index = in_index_map[root_val]
            root.left = build(left, index - 1)
            root.right = build(index + 1, right)

            return root

        return build(0, len(inorder) - 1)
preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
solver = Solution()
tree = solver.buildTree(preorder, inorder)
# print(tree)
# print results via basics
# print(tree.val)                   # 3
# print(tree.left.val)             # 9
# print(tree.right.val)            # 20
# print(tree.right.left.val)       # 15
# print(tree.right.right.val)      # 7
# print results via serialize helper function
from collections import deque
def serialize(root):
    if not root: return []
    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else: result.append(None)
    # Clean trailing None values
    while result and result[-1] is None:
        result.pop()
    return result
print(serialize(tree))
