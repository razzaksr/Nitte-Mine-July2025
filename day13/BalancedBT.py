class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def check(node):
            if not node:
                return 0  # height 0

            left = check(node.left)
            if left == -1:
                return -1  # not balanced

            right = check(node.right)
            if right == -1:
                return -1

            if abs(left - right) > 1:
                return -1  # unbalanced tree

            return 1 + max(left, right)  # return height

        return check(root) != -1
'''
       1
      / \
     2   3
    / \ / \
   4  5 6  7
'''
root = TreeNode(1)
root.left = TreeNode(2, TreeNode(4), TreeNode(5))
root.right = TreeNode(3, TreeNode(6), TreeNode(7))
print(Solution().isBalanced(root))
'''
        1
       /
      2
     /
    3
   /
  4
'''
root = TreeNode(1)
root.left = TreeNode(2)
root.left.left = TreeNode(3)
root.left.left.left = TreeNode(4)
print(Solution().isBalanced(root))