def carFleet(target, position, speed):
    # Pair cars with their position and time to reach the target
    cars = sorted(zip(position, speed), reverse=True)
    times = [(target - pos) / spd for pos, spd in cars]
    fleets = 0
    stack = []

    for time in times:
        # If the current car takes longer, it starts a new fleet
        if not stack or time > stack[-1]:
            fleets += 1
            stack.append(time)

    return fleets

print(carFleet(12, [10,8,0,5,3], [2,4,1,1,3]))  # Output: 3
print(carFleet(10, [3], [3]))                  # Output: 1
print(carFleet(100, [0,2,4], [4,2,1]))         # Output: 1
