from collections import deque

def countStudents(students, sandwiches):
    students = deque(students)
    sandwiches = list(sandwiches)
    count = 0  # Count of consecutive students who didn't take sandwich

    while students and count < len(students):
        if students[0] == sandwiches[0]:
            students.popleft()
            sandwiches.pop(0)
            count = 0  # reset as someone ate
        else:
            students.append(students.popleft())  # move to end
            count += 1  # one more student skipped

    return len(students)
print(countStudents([1,1,0,0], [0,1,0,1]))  # Output: 0
print(countStudents([1,1,1,0,0,1], [1,0,0,0,1,1]))  # Output: 3
