def largestRectangleArea(heights):
    stack = []  # Stack to store indices
    max_area = 0
    heights.append(0)  # Sentinel to flush remaining bars in stack

    for i, h in enumerate(heights):
        # While current bar is lower than the one on top of the stack
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]  # Height of the popped bar
            width = i if not stack else i - stack[-1] - 1  # Width of the rectangle
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area
print(largestRectangleArea([2,1,5,6,2,3]))  # Output: 10
print(largestRectangleArea([2,4]))         # Output: 4
