"""
Dynamic planning utilities for grid-based pathfinding.

This module implements a simple dynamic path planner using the A* search
algorithm.  The planner allows you to modify the underlying occupancy
grid at runtime (for example, when obstacles move) and replan a path
from the agent's current position to the goal.  While this approach
does not implement a full incremental algorithm like D*‑Lite, it
illustrates the core idea of dynamic replanning: when the world
changes, recompute the route using updated information.

The planner works on a 2‑D numpy array where a value of ``0``
represents free space and a value of ``1`` represents an obstacle.

Example usage:

    import numpy as np
    from dynamic_planner import DynamicPlanner

    # create a 10x10 empty grid and block a few cells
    grid = np.zeros((10, 10), dtype=int)
    grid[3:5, 4] = 1
    planner = DynamicPlanner(grid)

    start = (0, 0)
    goal = (9, 9)
    path = planner.plan(start, goal)
    print("initial path length", len(path))

    # simulate an obstacle appearing along the path
    planner.update_obstacles({(path[5][0], path[5][1]): True})
    new_path = planner.plan(path[4], goal)
    print("replanned path length", len(new_path))

The planner stores no internal state between calls to ``plan``
other than the occupancy grid itself.  It is therefore stateless
between planning queries; this keeps the implementation simple and
easy to integrate into an existing simulation loop.  For a more
sophisticated incremental planner (e.g., D*‑Lite), you would need to
maintain additional per‑node information (``g``/``rhs`` values and a
priority queue keyed by both path cost and heuristic) and update
only the affected nodes when the environment changes.
"""

from __future__ import annotations
import heapq
from typing import Dict, List, Tuple, Iterable, Optional, Any

import numpy as np

# Type aliases for clarity
Coord = Tuple[int, int]
Grid = np.ndarray

class DynamicPlanner:
    """A simple grid‑based dynamic path planner using A* search.

    Parameters
    ----------
    grid : np.ndarray
        A 2‑D array of shape (rows, cols) where ``0`` denotes free space
        and ``1`` denotes an obstacle.

    diag : bool, optional
        If ``True``, the planner allows diagonal moves (8‑connected grid).
        Otherwise, only four cardinal directions are used.

    weight : float, optional
        A multiplier for the heuristic.  Setting this to a value >1.0
        will produce sub‑optimal but often faster searches (weighted A*).
    """

    def __init__(self, grid: Grid, diag: bool = True, weight: float = 1.0) -> None:
        self.grid: Grid = grid
        self.rows, self.cols = grid.shape
        self.diag: bool = diag
        self.weight: float = weight
        # Precompute movement offsets and costs
        if diag:
            # 8‑connected grid: diagonals cost sqrt(2)
            self._neighbors: List[Tuple[int, int, float]] = [
                (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
            ]
        else:
            # 4‑connected grid: no diagonal moves
            self._neighbors = [
                (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            ]

    def in_bounds(self, coord: Coord) -> bool:
        """Check whether a coordinate is within the grid bounds."""
        r, c = coord
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, coord: Coord) -> bool:
        """Return ``True`` if the cell at ``coord`` is not an obstacle."""
        return self.grid[coord] == 0

    def neighbors(self, coord: Coord) -> Iterable[Tuple[Coord, float]]:
        """Yield valid neighbor coordinates and their movement cost."""
        r0, c0 = coord
        for dr, dc, cost in self._neighbors:
            r1, c1 = r0 + dr, c0 + dc
            if 0 <= r1 < self.rows and 0 <= c1 < self.cols:
                if self.grid[r1, c1] == 0:
                    yield (r1, c1), cost

    def heuristic(self, a: Coord, b: Coord) -> float:
        """Return an admissible heuristic (octile distance for 8‑conn, manhattan for 4‑conn)."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if self.diag:
            # Octile distance with diagonal moves costing sqrt(2)
            return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)
        else:
            return dx + dy

    def _reconstruct_path(self, came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
        path: List[Coord] = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def astar(self, start: Coord, goal: Coord) -> Optional[List[Coord]]:
        """Compute a path from ``start`` to ``goal`` using A* search.

        Returns a list of coordinates forming the path, including ``start`` and
        ``goal``.  If no path exists, returns ``None``.
        """
        if not (self.in_bounds(start) and self.in_bounds(goal)):
            return None
        if not (self.passable(start) and self.passable(goal)):
            return None

        open_heap: List[Tuple[float, Coord]] = []
        heapq.heappush(open_heap, (0.0, start))
        came_from: Dict[Coord, Coord] = {}
        g_cost: Dict[Coord, float] = {start: 0.0}
        f_cost: Dict[Coord, float] = {start: self.weight * self.heuristic(start, goal)}
        closed: set[Coord] = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct_path(came_from, current)
            if current in closed:
                continue
            closed.add(current)
            for neighbor, move_cost in self.neighbors(current):
                tentative_g = g_cost[current] + move_cost
                if neighbor in g_cost and tentative_g >= g_cost[neighbor]:
                    continue  # not a better path
                came_from[neighbor] = current
                g_cost[neighbor] = tentative_g
                h = self.heuristic(neighbor, goal)
                f = tentative_g + self.weight * h
                f_cost[neighbor] = f
                heapq.heappush(open_heap, (f, neighbor))
        return None

    def plan(self, start: Coord, goal: Coord) -> Optional[List[Coord]]:
        """Plan a path from ``start`` to ``goal`` and return a list of coordinates."""
        return self.astar(start, goal)

    def update_obstacles(self, changes: Dict[Coord, bool]) -> None:
        """Update the occupancy grid with a dict of changes.

        Parameters
        ----------
        changes : Dict[Coord, bool]
            A mapping from cell coordinates to a boolean indicating whether
            that cell should become an obstacle (True) or free space (False).
        """
        for coord, blocked in changes.items():
            r, c = coord
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r, c] = 1 if blocked else 0
