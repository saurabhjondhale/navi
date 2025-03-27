# Navigation
Autonomous drone navigation 
1. Understanding A* Algorithm 

A* is a graph-based pathfinding algorithm that finds the shortest path from a start node to a goal node by using:

g(n): Cost from start to node 𝑛 

h(n): Heuristic (estimated cost from 𝑛 to the goal).

f(n) = g(n) + h(n): Total estimated cost.

2. Setting Up the Environment 

Define the grid or graph for navigation. 

Mark obstacles and free spaces. 

Choose a heuristic function (Euclidean or Manhattan distance). 
