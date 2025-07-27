class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        self.max_sum = float('-inf')

        def max_gain(node):
            if not node:
                return 0

            # Recursively get max gain from left and right
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)

            # Current path sum through the node
            current_sum = node.val + left_gain + right_gain

            # Update global max sum
            self.max_sum = max(self.max_sum, current_sum)

            # Return max gain if continue the same path
            return node.val + max(left_gain, right_gain)

        max_gain(root)
        return self.max_sum
# Step 1: Build your binary tree
# root = TreeNode(-10)
# root.left = TreeNode(9)
# root.right = TreeNode(20)
# root.right.left = TreeNode(15)
# root.right.right = TreeNode(7)

# # Step 2: Create the Solution object
# solver = Solution()

# # Step 3: Call maxPathSum and print the result
# result = solver.maxPathSum(root)
# print("Maximum Path Sum:", result)

'''
         5
        / \
      4   8
     /   / \
   11  13  4
  /  \       \
 7    2       1
'''
root = TreeNode(5)
root.left = TreeNode(4)
root.right = TreeNode(8)

root.left.left = TreeNode(11)
root.left.left.left = TreeNode(7)
root.left.left.right = TreeNode(2)

root.right.left = TreeNode(13)
root.right.right = TreeNode(4)
root.right.right.right = TreeNode(1)
solver = Solution()
result = solver.maxPathSum(root)
print("Maximum Path Sum:", result)