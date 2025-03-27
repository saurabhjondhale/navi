import heapq
import numpy as np

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost to goal
        self.f = 0  # Total cost

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))  # Euclidean distance

def a_star(grid, start, goal):
    open_list = []
    closed_set = set()
    start_node = Node(start)
    goal_node = Node(goal)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.position == goal:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Reverse the path

        closed_set.add(current_node.position)

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:  # 4 directions
            neighbor_pos = (current_node.position[0] + dx, current_node.position[1] + dy)

            if neighbor_pos in closed_set or not (0 <= neighbor_pos[0] < len(grid) and 0 <= neighbor_pos[1] < len(grid[0])) or grid[neighbor_pos[0]][neighbor_pos[1]] == 1:
                continue  # Skip obstacles or out-of-bounds

            neighbor = Node(neighbor_pos, current_node)
            neighbor.g = current_node.g + 1
            neighbor.h = heuristic(neighbor_pos, goal_node.position)
            neighbor.f = neighbor.g + neighbor.h

            if any(n.position == neighbor.position and n.f <= neighbor.f for n in open_list):
                continue  # Skip if a better path exists

            heapq.heappush(open_list, neighbor)

    return None  # No path found

# Example Grid (0: Free, 1: Obstacle)
grid = np.array([
    [0, 0, 0, 0, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
])

start = (0, 0)
goal = (4, 4)
path = a_star(grid, start, goal)

print("Path:", path)
