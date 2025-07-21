from collections import deque
def nextGreaterElements(arr):
    n = len(arr)
    result = [-1] * n
    stack = []  # Stack to store indices

    for i in range(n):
        while stack and arr[i] > arr[stack[-1]]:
            index = stack.pop()
            result[index] = arr[i]
        stack.append(i)

    return result

def timeRequiredToBuy(tickets, k):
    queue = deque([(i, t) for i, t in enumerate(tickets)])
    time = 0

    while queue:
        idx, t = queue.popleft()
        t -= 1
        time += 1

        if t > 0:
            queue.append((idx, t))

        if idx == k and t == 0:
            return time


from collections import deque

def can_assign(tasks, workers, pills, strength, mid):
    task_deque = deque(tasks[:mid])
    available_pills = pills
    worker_idx = len(workers) - 1

    for _ in range(mid):
        if worker_idx < 0:
            return False
        if workers[worker_idx] >= task_deque[-1]:
            task_deque.pop()
            worker_idx -= 1
        elif available_pills > 0:
            left = 0
            while left <= worker_idx and workers[left] + strength < task_deque[0]:
                left += 1
            if left > worker_idx:
                return False
            task_deque.popleft()
            available_pills -= 1
            worker_idx -= 1
        else:
            return False
    return True

def max_tasks_assignable(tasks, workers, pills, strength):
    tasks.sort()
    workers.sort()

    low, high = 0, min(len(tasks), len(workers))
    answer = 0

    while low <= high:
        mid = (low + high) // 2
        if can_assign(tasks, workers, pills, strength, mid):
            answer = mid
            low = mid + 1
        else:
            high = mid - 1

    return answer


from collections import deque

def can_assign(tasks, workers, pills, strength, mid):
    task_deque = deque(tasks[:mid])
    available_pills = pills
    worker_idx = len(workers) - 1

    for _ in range(mid):
        if worker_idx < 0:
            return False
        if workers[worker_idx] >= task_deque[-1]:
            task_deque.pop()
            worker_idx -= 1
        elif available_pills > 0:
            left = 0
            while left <= worker_idx and workers[left] + strength < task_deque[0]:
                left += 1
            if left > worker_idx:
                return False
            task_deque.popleft()
            available_pills -= 1
            worker_idx -= 1
        else:
            return False
    return True

def max_tasks_assignable(tasks, workers, pills, strength):
    tasks.sort()
    workers.sort()

    low, high = 0, min(len(tasks), len(workers))
    answer = 0

    while low <= high:
        mid = (low + high) // 2
        if can_assign(tasks, workers, pills, strength, mid):
            answer = mid
            low = mid + 1
        else:
            high = mid - 1

    return answer

print(max_tasks_assignable([3, 2, 1], [0, 3, 3], 1, 1))          # Output: 3
print(max_tasks_assignable([5, 4], [0, 0, 0], 1, 5))             # Output: 1
print(max_tasks_assignable([10, 15, 30], [0, 10, 10, 10, 10], 3, 10))  # Output: 2


# print(max_tasks_assignable([3, 2, 1], [0, 3, 3], 1, 1))          # Output: 3
# print(max_tasks_assignable([5, 4], [0, 0, 0], 1, 5))             # Output: 1
# print(max_tasks_assignable([10, 15, 30], [0, 10, 10, 10, 10], 3, 10))  # Output: 2


# print(timeRequiredToBuy([2, 3, 2], 2))  # Output: 6
# print(timeRequiredToBuy([5, 1, 1, 1], 0))  # Output: 8


# print(nextGreaterElements([1, 3, 2, 4]))       # Output: [3, 4, 4, -1]
# print(nextGreaterElements([6, 8, 0, 1, 3]))    # Output: [8, -1, 1, 3, -1]
# print(nextGreaterElements([10, 20, 30, 50]))   # Output: [20, 30, 50, -1]
# print(nextGreaterElements([50, 40, 30, 10]))   # Output: [-1, -1, -1, -1]
