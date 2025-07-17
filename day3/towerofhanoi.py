def hanoi_recursive(n, source, destination, auxiliary, moves):
    if n == 1:
        moves.append(f"Move container from {source} to {destination}")
        return

    # Move n-1 containers from source to auxiliary
    hanoi_recursive(n - 1, source, auxiliary, destination, moves)

    # Move the largest container to destination
    moves.append(f"Move container from {source} to {destination}")

    # Move n-1 containers from auxiliary to destination
    hanoi_recursive(n - 1, auxiliary, destination, source, moves)

def move_containers(n, source='A', destination='C', auxiliary='B'):
    moves = []
    hanoi_recursive(n, source, destination, auxiliary, moves)
    return moves

def get_minimum_moves(n):
    return 2 ** n - 1

containers = 3
moves = move_containers(containers, "Position A", "Position C", "Position B")
print(f"Moving {containers} containers:")
for i, move in enumerate(moves[:5], 1):  # Show first 5 moves
    print(f"  Step {i}: {move}")
