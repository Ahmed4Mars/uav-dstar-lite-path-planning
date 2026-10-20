"""
Demonstration of the DynamicPlanner class in a stand‑alone environment.

This script illustrates how to use ``dynamic_planner.DynamicPlanner`` to
compute a path on a static grid, then update the grid to simulate api
dynamic obstacle and replan a new path from the agent’s current
position.  It uses ``matplotlib`` to visualize the grid and both
paths, but only prints basic information if visualization is not
available.

Usage:
    python run_dynamic_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from dynamic_planner import DynamicPlanner


def visualize(grid: np.ndarray, path: list[tuple[int, int]], path2: list[tuple[int, int]] | None = None) -> None:
    """Plot the occupancy grid and one or two paths using matplotlib."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap="Greys", origin="lower")
    if path:
        coords = np.array(path)
        ax.plot(coords[:, 1], coords[:, 0], label="initial path", linewidth=2)
    if path2:
        coords2 = np.array(path2)
        ax.plot(coords2[:, 1], coords2[:, 0], label="replanned path", linestyle="--", linewidth=2)
    ax.scatter([path[0][1], path[-1][1]], [path[0][0], path[-1][0]], marker="*", s=100, label="start/goal")
    ax.legend()
    ax.set_title("Dynamic path planning demo")
    plt.show()


def main() -> None:
    # Create a 30x30 grid with random obstacles
    np.random.seed(42)
    grid = np.zeros((30, 30), dtype=int)
    obstacle_mask = np.random.rand(30, 30) < 0.2  # 20% obstacles
    grid[obstacle_mask] = 1
    start = (0, 0)
    goal = (29, 29)
    # Ensure start and goal are free
    grid[start] = 0
    grid[goal] = 0

    planner = DynamicPlanner(grid, diag=True)
    path = planner.plan(start, goal)
    if path is None:
        print("No path found from start to goal in initial grid.")
        return
    print(f"Initial path length: {len(path)}")

    # Simulate a dynamic obstacle appearing at the 5th step along the path
    if len(path) > 6:
        new_obstacle_cell = path[5]
        planner.update_obstacles({new_obstacle_cell: True})
        print(f"Added obstacle at {new_obstacle_cell}, replanning...")
        # Replan from the current position (just before the blocked cell)
        replanning_start = path[4]
        new_path = planner.plan(replanning_start, goal)
        if new_path is None:
            print("No path found after obstacle update.")
        else:
            print(f"Replanned path length: {len(new_path)}")
            # Visualize both paths
            visualize(grid, path, new_path)
    else:
        # Not enough path steps to add a dynamic obstacle
        visualize(grid, path)


if __name__ == "__main__":
    main()
