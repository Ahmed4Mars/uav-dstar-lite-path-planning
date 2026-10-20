"""
2D gridworld with:
- ONE start and ONE goal
- Large environment (120 x 80)
- Static and moving obstacles
- D*-style replanning (A* re-run periodically)
- Agent trajectory in GREY
- Old planned paths in BROWN, GREEN, BLUE, RED, YELLOW (cycling)
- Current main path in BLACK
"""

import math
import heapq
import time
from dataclasses import dataclass

SHOW_PLOT = True      # Set to False if you only want logic (no animation)
REPLAN_INTERVAL = 5   # Steps between replans

if SHOW_PLOT:
    import matplotlib.pyplot as plt
    import numpy as np


# ======================== Grid map ===========================
class GridMap:
    """
    2D occupancy grid.
    0 = free, 1 = static obstacle, 2 = moving obstacle
    """
    def __init__(self, width=120, height=80):
        self.w = width
        self.h = height
        self.grid = [[0 for _ in range(self.h)] for _ in range(self.w)]

    def in_bounds(self, x, y):
        return 0 <= x < self.w and 0 <= y < self.h

    def is_blocked(self, x, y):
        if not self.in_bounds(x, y):
            return True
        return self.grid[x][y] != 0

    def add_static_rect(self, x0, y0, x1, y1):
        """
        Add a static rectangular obstacle [x0,x1] × [y0,y1].
        """
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                if self.in_bounds(x, y):
                    self.grid[x][y] = 1

    def clear_moving(self):
        for x in range(self.w):
            for y in range(self.h):
                if self.grid[x][y] == 2:
                    self.grid[x][y] = 0

    def add_moving_disk(self, cx, cy, radius=3):
        """
        Add a circular moving obstacle centered at (cx, cy).
        Radius is in grid cells.
        """
        r2 = radius * radius
        for x in range(cx - radius - 1, cx + radius + 2):
            for y in range(cy - radius - 1, cy + radius + 2):
                if not self.in_bounds(x, y):
                    continue
                if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                    if self.grid[x][y] == 0:
                        self.grid[x][y] = 2


# ==================== D*-style planner (A*) ==================
class DStarLite2D:
    """
    Very small "D* style" planner.
    We re-run A* whenever the map or start changes / every N steps.
    """

    def __init__(self, grid: GridMap):
        self.grid = grid

    @staticmethod
    def heuristic(a, b):
        (x1, y1), (x2, y2) = a, b
        return abs(x1 - x2) + abs(y1 - y2)  # Manhattan

    def neighbors(self, s):
        x, y = s
        nbs = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if not self.grid.is_blocked(nx, ny):
                nbs.append((nx, ny))
        return nbs

    def plan(self, start, goal):
        """
        A*: returns list of (x,y) from start to goal.
        """
        if self.grid.is_blocked(*start) or self.grid.is_blocked(*goal):
            return None

        open_heap = []
        heapq.heappush(open_heap, (0 + self.heuristic(start, goal), 0, start))
        came_from = {}
        g = {start: 0}

        while open_heap:
            f, cost, current = heapq.heappop(open_heap)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for nb in self.neighbors(current):
                new_cost = cost + 1.0
                if nb not in g or new_cost < g[nb]:
                    g[nb] = new_cost
                    came_from[nb] = current
                    fnb = new_cost + self.heuristic(nb, goal)
                    heapq.heappush(open_heap, (fnb, new_cost, nb))

        return None


# ====================== Moving obstacle ======================
@dataclass
class MovingObstacle:
    x1: int
    y1: int
    x2: int
    y2: int
    speed: float  # "parametric" speed
    t: float = 0.0

    def update(self):
        """
        Ping-pong motion between (x1,y1) and (x2,y2).
        """
        self.t += self.speed
        phase = math.sin(self.t)   # -1..1
        alpha = 0.5 * (phase + 1)  # 0..1
        x = int(round(self.x1 * (1 - alpha) + self.x2 * alpha))
        y = int(round(self.y1 * (1 - alpha) + self.y2 * alpha))
        return x, y


# ============================ Sim ============================
def run_sim():
    # --------- Larger grid + static obstacles ----------
    grid = GridMap(width=120, height=80)

    # Long vertical walls
    grid.add_static_rect(20, 5, 20, 70)
    grid.add_static_rect(50, 10, 50, 75)
    grid.add_static_rect(85, 0, 85, 60)

    # Horizontal walls
    grid.add_static_rect(0, 25, 40, 25)
    grid.add_static_rect(30, 45, 80, 45)
    grid.add_static_rect(60, 60, 119, 60)

    # Big blocks / rooms
    grid.add_static_rect(5, 35, 15, 42)
    grid.add_static_rect(55, 20, 70, 30)
    grid.add_static_rect(90, 10, 110, 20)
    grid.add_static_rect(95, 35, 115, 50)

    # Narrow “gate”
    grid.add_static_rect(35, 55, 35, 61)
    grid.add_static_rect(35, 65, 35, 70)

    # --------- Start & goal ----------
    start_pos = (5, 5)
    goal_pos = (110, 70)
    agent_pos = start_pos
    initial_start = start_pos

    planner = DStarLite2D(grid)

    # --------- Dynamic obstacles ----------
    moving_obs = [
        MovingObstacle(10, 40, 110, 40, speed=0.04),
        MovingObstacle(45, 5, 45, 75, speed=0.06),
        MovingObstacle(75, 10, 75, 70, speed=0.05),
        MovingObstacle(95, 65, 115, 65, speed=0.055),
    ]

    # --------- Storage: paths & trajectory ----------
    paths_history = []      # old planned paths (will be colored)
    current_path = None
    last_path = None

    agent_traj = [agent_pos]  # trajectory for grey "back path"

    # Colors for old paths (cycled)
    history_colors = [ "green", "blue", "red", "yellow"]

    if SHOW_PLOT:
        fig, ax = plt.subplots(figsize=(10, 7))
        plt.ion()
        plt.show(block=False)

    def draw():
        if not SHOW_PLOT:
            return

        # Occupancy to image
        arr = np.zeros((grid.h, grid.w))
        for x in range(grid.w):
            for y in range(grid.h):
                if grid.grid[x][y] == 1:      # static
                    arr[y, x] = 0.4
                elif grid.grid[x][y] == 2:    # moving
                    arr[y, x] = 0.8
                else:
                    arr[y, x] = 0.0

        ax.clear()
        ax.imshow(arr, origin="lower", cmap="gray_r", vmin=0, vmax=1)

        # 1) Agent trajectory (back path) in grey
        if len(agent_traj) > 1:
            tx = [p[0] for p in agent_traj]
            ty = [p[1] for p in agent_traj]
            ax.plot(tx, ty, "-", color="0.4", linewidth=2.0, alpha=0.9, label="Trajectory")

        # 2) Old full planned paths in brown, green, blue, red, yellow (cycling)
        for idx, pth in enumerate(paths_history):
            if not pth:
                continue
            color = history_colors[idx % len(history_colors)]
            px = [p[0] for p in pth]
            py = [p[1] for p in pth]
            ax.plot(px, py, "-", color=color, linewidth=1.8, alpha=0.9)

        # 3) Current active (main) path in BLACK (on top)
        if current_path:
            px = [p[0] for p in current_path]
            py = [p[1] for p in current_path]
            ax.plot(px, py, "-k", linewidth=2.5, label="Current path")

        # Agent
        ax.scatter([agent_pos[0]], [agent_pos[1]],
                   c="blue", s=80, marker="o", edgecolors="k", label="Agent")

        # Start (fixed)
        ax.scatter([initial_start[0]], [initial_start[1]],
                   c="yellow", s=120, marker="*", edgecolors="k", label="Start")

        # Goal (fixed)
        ax.scatter([goal_pos[0]], [goal_pos[1]],
                   c="green", s=120, marker="*", edgecolors="k", label="Goal")

        ax.set_xlim(-0.5, grid.w - 0.5)
        ax.set_ylim(-0.5, grid.h - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"2D D* style | Start {initial_start} | Goal {goal_pos} | Agent {agent_pos}"
        )
        ax.legend(loc="upper right", fontsize=8)
        plt.pause(0.001)

    # --------- Main loop ----------
    step = 0
    max_steps = 5000  # bigger env → more steps

    while step < max_steps and agent_pos != goal_pos:
        step += 1

        # Move dynamic obstacles
        grid.clear_moving()
        for mo in moving_obs:
            ox, oy = mo.update()
            grid.add_moving_disk(ox, oy, radius=3)

        # Replan every REPLAN_INTERVAL steps (and at first step)
        if step == 1 or step % REPLAN_INTERVAL == 0 or current_path is None:
            new_path = planner.plan(agent_pos, goal_pos)
            if new_path is None:
                print(f"[WARN] No path from {agent_pos} to {goal_pos} at step {step}")
                current_path = None
            else:
                # store old full path if new one is different
                if last_path is not None and new_path != last_path:
                    paths_history.append(last_path)
                last_path = new_path
                current_path = new_path

        # Move one step along current_path
        if current_path and len(current_path) > 1:
            agent_pos = current_path[1]
            # Update trajectory if new point
            if agent_traj[-1] != agent_pos:
                agent_traj.append(agent_pos)

        draw()
        time.sleep(0.02)

        if agent_pos == goal_pos:
            print(f"[INFO] Reached goal at step {step}, pos={agent_pos}")
            if last_path is not None:
                paths_history.append(last_path)
            break

    print("[INFO] Simulation finished.")
    if SHOW_PLOT:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    run_sim()
