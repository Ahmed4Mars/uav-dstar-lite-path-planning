# run_uav_nav.py — Kinematic 3D D* path follower with SmartFollowCam (LoS guard)
import argparse, math, time, random, heapq
import numpy as np
import pybullet as p, pybullet_data

from dstar_lite_3d import DStarLite3D

# ---- Config (robust import) ----
try:
    import uav_config as cfg
    DT           = getattr(cfg, "DT", 1.0/240.0)
    TERRAIN_SIZE = getattr(cfg, "TERRAIN_SIZE", 80.0)
except Exception:
    DT, TERRAIN_SIZE = 1.0/240.0, 80.0

# ---- Optional world builder ----
try:
    from create_uav_world import create_world
except Exception:
    create_world = None

# ---- Occupancy helpers ----
from occupancy import (
    world_to_cell_3d, cell_to_world_3d,
    grid3d_from_buildings,
)

# ----------------- Utilities -----------------
def buildings_as_rects(buildings):
    """Return [(cx,cy,w,d,h,bid), ...] for provided building/obstacle bodies or rects."""
    rects = []
    for b in buildings:
        if isinstance(b, (list, tuple)) and len(b) >= 6:
            cx, cy, w, d, h, bid = b[:6]
            rects.append((cx, cy, w, d, h, bid))
        else:
            bid = int(b)
            (minx, miny, minz), (maxx, maxy, maxz) = p.getAABB(bid)
            cx = 0.5 * (minx + maxx)
            cy = 0.5 * (miny + maxy)
            w  = maxx - minx
            d  = maxy - miny
            h  = maxz - minz
            rects.append((cx, cy, w, d, h, bid))
    return rects

def draw_polyline3d(points_xyz, life=0, width=2, rgb=(0,0,1)):
    if len(points_xyz) < 2:
        return []
    ids = []
    for a, b in zip(points_xyz, points_xyz[1:]):
        ids.append(p.addUserDebugLine(a, b, rgb, width, life))
    return ids

def ground_z_under(x, y, probe_top=200.0, probe_bottom=-200.0):
    """Return Z of first hit along vertical ray; 0.0 if nothing hit."""
    if not p.isConnected():
        return 0.0
    hit = p.rayTest([x, y, probe_top], [x, y, probe_bottom])[0]
    return hit[3][2] if hit[0] >= 0 else 0.0

def half_extent():
    return float(TERRAIN_SIZE) * 0.5

# ----------------- 3D A* (26-connected) -----------------
# Kept for reference / comparison; not used once D* Lite is enabled.
def astar3d(occ, start, goal, res_xy=0.5, res_z=0.5):
    """
    occ: bool array (nx, ny, nz), True=blocked
    start/goal: (i,j,k)
    """
    nx, ny, nz = occ.shape
    si, sj, sk = start
    gi, gj, gk = goal
    if not (0 <= si < nx and 0 <= sj < ny and 0 <= sk < nz):
        return []
    if not (0 <= gi < nx and 0 <= gj < ny and 0 <= gk < nz):
        return []
    if occ[si, sj, sk] or occ[gi, gj, gk]:
        return []

    nbrs = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    step_cost = lambda dx, dy, dz: math.sqrt(
        (dx * res_xy) ** 2 + (dy * res_xy) ** 2 + (dz * res_z) ** 2
    )

    def h(i, j, k):
        return math.sqrt(
            ((i - gi) * res_xy) ** 2
            + ((j - gj) * res_xy) ** 2
            + ((k - gk) * res_z) ** 2
        )

    openq = []
    heapq.heappush(openq, (h(si, sj, sk), 0.0, (si, sj, sk)))
    came = {}
    gscore = {(si, sj, sk): 0.0}
    closed = set()

    while openq:
        _, gcur, cur = heapq.heappop(openq)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == (gi, gj, gk):
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        ci, cj, ck = cur
        for dx, dy, dz in nbrs:
            ni, nj, nk = ci + dx, cj + dy, ck + dz
            if not (0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz):
                continue
            if occ[ni, nj, nk]:
                continue
            tentative = gcur + step_cost(dx, dy, dz)
            key = (ni, nj, nk)
            if tentative < gscore.get(key, 1e18):
                gscore[key] = tentative
                came[key] = cur
                f = tentative + h(ni, nj, nk)
                heapq.heappush(openq, (f, tentative, key))
    return []


# ----------------- Smart Follow Camera -----------------
class SmartFollowCam:
    """
    Keep the UAV framed and visible. If an occlusion is detected via rayTest,
    adapt pitch/yaw/distance to regain line-of-sight. Smoothing avoids jumpiness.

    Keys:
      c  -> toggle follow
      f  -> refocus (snap to current target)
      [ / ] -> zoom in / out
      o  -> toggle occlusion guard
    """
    def __init__(
        self,
        dist=7.0,
        pitch=-38.0,
        yaw=0.0,
        follow=True,
        smooth=0.85,
        los_guard=True,
        yaw_bias=0.0,
        target_lift=0.25,
        dist_limits=(3.0, 30.0),
    ):
        self.dist = float(dist)
        self.pitch = float(pitch)
        self.yaw = float(yaw)  # current camera yaw
        self.follow = bool(follow)
        self.smooth = float(smooth)
        self.los_guard = bool(los_guard)
        self.yaw_bias = float(yaw_bias)
        self.target_lift = float(target_lift)
        self.dist_min, self.dist_max = dist_limits
        self._target = None
        self._yaw_target = self.yaw

    @staticmethod
    def _to_rad(deg):
        return math.radians(deg)

    @staticmethod
    def _lerp(a, b, t):
        return a * (1.0 - t) + b * t

    def _smooth_target(self, pos):
        pt = np.array([pos[0], pos[1], pos[2] + self.target_lift], dtype=float)
        if self._target is None:
            self._target = pt
        else:
            self._target = self._lerp(self._target, pt, 1.0 - self.smooth)
        return self._target

    def _eye_from(self, target, dist, yaw_deg, pitch_deg):
        yaw_r = self._to_rad(yaw_deg)
        pitch_r = self._to_rad(pitch_deg)
        ex = target[0] - dist * math.cos(pitch_r) * math.cos(yaw_r)
        ey = target[1] - dist * math.cos(pitch_r) * math.sin(yaw_r)
        ez = target[2] + dist * math.sin(pitch_r)
        return [ex, ey, ez]

    def _has_los(self, eye, uav_id, uav_pos):
        target = [uav_pos[0], uav_pos[1], uav_pos[2] + 0.25]
        hit = p.rayTest(eye, target)[0]
        return (hit[0] < 0) or (hit[0] == uav_id)

    def _compute_yaw_from_dir(self, dir_xy):
        if dir_xy is None:
            return self.yaw
        dx, dy = float(dir_xy[0]), float(dir_xy[1])
        if abs(dx) + abs(dy) < 1e-8:
            return self.yaw
        return math.degrees(math.atan2(dy, dx)) + self.yaw_bias

    def update(self, uav_id, dir_hint_xy=None):
        if (
            (not self.follow)
            or p.getConnectionInfo().get("connectionMethod") != p.GUI
        ):
            return
        (ux, uy, uz), _ = p.getBasePositionAndOrientation(uav_id)
        tgt = self._smooth_target([ux, uy, uz])

        desired_yaw = self._compute_yaw_from_dir(dir_hint_xy)
        self._yaw_target = self._lerp(
            self._yaw_target, desired_yaw, 1.0 - self.smooth
        )
        yaw_try_list = [
            self._yaw_target,
            self._yaw_target + 18.0,
            self._yaw_target - 18.0,
            self._yaw_target + 32.0,
            self._yaw_target - 32.0,
        ]

        pitch_try_list = [-38.0, -45.0, -52.0, -60.0]  # prefer slightly lower first
        dist_try_list = [
            self.dist,
            max(self.dist_min, self.dist * 0.9),
            min(self.dist_max, self.dist * 1.1),
        ]

        eye0 = self._eye_from(tgt, self.dist, self._yaw_target, self.pitch)
        if (not self.los_guard) or self._has_los(eye0, uav_id, [ux, uy, uz]):
            p.resetDebugVisualizerCamera(
                self.dist, self._yaw_target, self.pitch, tgt
            )
            self.yaw = self._yaw_target
            return

        found = False
        best = None
        for pit in pitch_try_list:
            for yaw in yaw_try_list:
                for dist in dist_try_list:
                    eye = self._eye_from(tgt, dist, yaw, pit)
                    if self._has_los(eye, uav_id, [ux, uy, uz]):
                        best = (dist, yaw, pit)
                        found = True
                        break
                if found:
                    break
            if found:
                break

        if found:
            self.dist, self.yaw, self.pitch = best
        else:
            self.pitch = -65.0
            self.dist = max(self.dist_min, self.dist * 0.85)

        p.resetDebugVisualizerCamera(self.dist, self.yaw, self.pitch, tgt)


# ----------------- Helpers -----------------
def path_length(points):
    if len(points) < 2:
        return 0.0
    return float(sum(math.dist(points[i], points[i+1]) for i in range(len(points)-1)))


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    # Visuals / sim
    ap.add_argument("--gui", action="store_true", help="Open PyBullet GUI")
    ap.add_argument("--rate", type=float, default=None, help="Physics Hz (movement tick)")
    ap.add_argument("--cam", action="store_true", help="Enable camera follow (GUI only)")
    ap.add_argument(
        "--linger",
        type=float,
        default=2.0,
        help="Seconds to linger after arrival (-1 = forever)",
    )

    # Grid / obstacles
    ap.add_argument("--res", type=float, default=0.6, help="XY resolution (m)")
    ap.add_argument("--vres", type=float, default=0.6, help="Z resolution (m)")
    ap.add_argument("--margin", type=float, default=0.6, help="Obstacle inflation in XY (m)")
    ap.add_argument("--zmargin", type=float, default=0.6, help="Obstacle inflation in Z (m)")
    ap.add_argument("--zmin", type=float, default=0.0, help="Min world Z to plan")
    ap.add_argument("--zmax", type=float, default=35.0, help="Max world Z to plan")

    # Start / goal
    ap.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start XYZ as 'x,y,z' (z ignored; we spawn on ground)",
    )
    ap.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Goal  XYZ as 'x,y,z' (z ignored; snapped to ground)",
    )
    ap.add_argument(
        "--min_goal_dist",
        type=float,
        default=8.0,
        help="Min straight-line distance for random goal (m)",
    )

    # Motion
    ap.add_argument("--speed", type=float, default=3.0, help="Kinematic path speed (m/s)")
    ap.add_argument("--quiet", action="store_true", help="Less console output")

    # Camera tuning
    ap.add_argument("--cam_dist", type=float, default=7.0, help="Follow camera base distance")
    ap.add_argument("--cam_pitch", type=float, default=-38.0, help="Follow camera base pitch (deg)")
    ap.add_argument("--cam_yaw_bias", type=float, default=0.0, help="Additive yaw bias (deg)")
    ap.add_argument("--cam_los_guard", action="store_true", help="Enable occlusion avoidance")

    args = ap.parse_args()

    # Connect
    if p.isConnected():
        p.disconnect()
    p.connect(p.GUI if args.gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)  # kinematic
    dt = 1.0 / args.rate if args.rate else DT

    # Build world
    buildings = []
    if create_world is not None:
        ret = create_world()
        if isinstance(ret, tuple) and len(ret) >= 2:
            buildings = ret[1]
    else:
        p.loadURDF("plane.urdf")

    half = half_extent()

    # ---- Moving buildings (dynamic buildings) ----
    moving_buildings = []
    if buildings:
        num_moving = min(3, len(buildings))  # move up to 3 buildings
        used = 0
        for bid in buildings:
            if not isinstance(bid, int):
                continue
            base_pos, base_orn = p.getBasePositionAndOrientation(bid)
            axis = 0 if (used % 2 == 0) else 1  # move either in x or y
            moving_buildings.append(
                {
                    "id": bid,
                    "base_pos": np.array(base_pos, dtype=float),
                    "orn": base_orn,
                    "axis": axis,
                    "amp": 4.0,
                    "omega": 0.25 + 0.1 * used,
                }
            )
            used += 1
            if used >= num_moving:
                break

    # ---- Initial 3D occupancy using static buildings only ----
    rects_static = buildings_as_rects(buildings)
    occ3 = grid3d_from_buildings(
        rects_static,
        res=args.res,
        vres=args.vres,
        zmin=args.zmin,
        zmax=args.zmax,
        margin=args.margin,
        zmargin=args.zmargin,
    )
    nx, ny, nz = occ3.shape  # NOTE: .shape, NOT .shape()

    # Helper: check if a world point is free in the current occ3
    def point_free(x, y, z):
        i, j, k = world_to_cell_3d(x, y, z, args.res, args.vres, args.zmin)
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            return False
        return not occ3[i, j, k]

    # ---- START ON GROUND ----
    if args.start:
        svals = [float(v) for v in args.start.split(",")]
        sx, sy = svals[0], svals[1]
    else:
        sx, sy = 0.0, 0.0

    sz = float(
        np.clip(
            ground_z_under(sx, sy) + 0.35,
            args.zmin + 0.35,
            args.zmax - 0.35,
        )
    )
    if not point_free(sx, sy, sz):
        found = False
        # Search locally on the ground around (sx,sy)
        for r in np.linspace(0.6, 8.0, 20):
            for ang in np.linspace(0, 2 * math.pi, 24, endpoint=False):
                tx = sx + r * math.cos(ang)
                ty = sy + r * math.sin(ang)
                tz = float(
                    np.clip(
                        ground_z_under(tx, ty) + 0.35,
                        args.zmin + 0.35,
                        args.zmax - 0.35,
                    )
                )
                if point_free(tx, ty, tz):
                    sx, sy, sz = tx, ty, tz
                    found = True
                    break
            if found:
                break
        # Global random fallback on ground
        if not found:
            for _ in range(1000):
                tx = float(
                    np.clip(
                        random.uniform(-half * 0.9, half * 0.9),
                        -half * 0.95,
                        half * 0.95,
                    )
                )
                ty = float(
                    np.clip(
                        random.uniform(-half * 0.9, half * 0.9),
                        -half * 0.95,
                        half * 0.95,
                    )
                )
                tz = float(
                    np.clip(
                        ground_z_under(tx, ty) + 0.35,
                        args.zmin + 0.35,
                        args.zmax - 0.35,
                    )
                )
                if point_free(tx, ty, tz):
                    sx, sy, sz = tx, ty, tz
                    found = True
                    break
        if not found:
            print(
                "[nav] Could not find a free ground start — "
                "relax --margin/--zmargin or bounds."
            )
            return

    # ---- GOAL ON GROUND ----
    if args.goal:
        gvals = [float(v) for v in args.goal.split(",")]
        gx, gy = gvals[0], gvals[1]
        gz = float(
            np.clip(
                ground_z_under(gx, gy) + 0.35,
                args.zmin + 0.35,
                args.zmax - 0.35,
            )
        )
        if not point_free(gx, gy, gz):
            if not args.quiet:
                print(
                    "[nav] Provided goal cell is occupied; "
                    "selecting random free ground goal."
                )
            args.goal = None

    if not args.goal:
        ok = False
        for _ in range(1000):
            gx = float(
                np.clip(
                    random.uniform(-half * 0.9, half * 0.9),
                    -half * 0.95,
                    half * 0.95,
                )
            )
            gy = float(
                np.clip(
                    random.uniform(-half * 0.9, half * 0.9),
                    -half * 0.95,
                    half * 0.95,
                )
            )
            gz = float(
                np.clip(
                    ground_z_under(gx, gy) + 0.35,
                    args.zmin + 0.35,
                    args.zmax - 0.35,
                )
            )
            if not point_free(gx, gy, gz):
                continue
            if math.dist([sx, sy, sz], [gx, gy, gz]) < args.min_goal_dist:
                continue
            ok = True
            break
        if not ok:
            print(
                "[nav] Could not find a free random ground goal — "
                "try relaxing --margin/--zmargin or bounds."
            )
            return

    # Marks (GUI only)
    if p.getConnectionInfo().get("connectionMethod") == p.GUI:
        for (x, y, z, rgba) in [
            (sx, sy, sz, (0.9, 0.2, 0.2, 1)),
            (gx, gy, gz, (0.2, 0.9, 0.2, 1)),
        ]:
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25)
            vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.25, rgbaColor=rgba
            )
            p.createMultiBody(0, col, vis, [x, y, z])

    # ---- Initial 3D D* Lite plan (static map) ----
    si, sj, sk = world_to_cell_3d(sx, sy, sz, args.res, args.vres, args.zmin)
    gi, gj, gk = world_to_cell_3d(gx, gy, gz, args.res, args.vres, args.zmin)

    planner = DStarLite3D(
        occ3,
        start=(si, sj, sk),
        goal=(gi, gj, gk),
        res_xy=args.res,
        res_z=args.vres,
    )
    path_ijk = planner.plan()
    if not path_ijk:
        print(
            "[nav] No 3D path found with D* Lite. "
            "Try smaller --vres, larger --zmax, or smaller --margin/--zmargin."
        )
        return

    # Voxel centers -> world
    waypoints = [
        cell_to_world_3d(i, j, k, args.res, args.vres, args.zmin)
        for (i, j, k) in path_ijk
    ]
    # Optional sparsification
    if len(waypoints) > 3:
        waypoints = waypoints[::2] + [waypoints[-1]]

    # Draw initial nominal path in BLUE
    if args.gui:
        draw_polyline3d(waypoints, width=2, rgb=(0, 0, 1))

    # Spawn UAV as fixed base (physics off)
    uav = p.loadURDF("uav.urdf", waypoints[0], useFixedBase=True)

    # Camera (GUI only) with occlusion guard
    cam_follow = bool(args.cam or args.gui)
    cam = SmartFollowCam(
        dist=args.cam_dist,
        pitch=args.cam_pitch,
        yaw=0.0,
        follow=cam_follow,
        smooth=0.86,
        los_guard=bool(args.cam_los_guard),
        yaw_bias=args.cam_yaw_bias,
        target_lift=0.25,
    )
    if args.gui:
        p.resetDebugVisualizerCamera(
            cam.dist, cam.yaw, cam.pitch, waypoints[0]
        )

    KEY = lambda c: ord(c)
    ESC = 27

    # ---- Flying moving obstacles (colored spheres along the original path) ----
    moving_obstacles = []
    num_dyn_obs = 20  # increase this number to add more flying balls
    if len(waypoints) > 1:
        for k in range(num_dyn_obs):
            frac = (k + 1.0) / (num_dyn_obs + 1.0)
            idx_wp = min(int(frac * (len(waypoints) - 1)), len(waypoints) - 1)
            wx, wy, wz = waypoints[idx_wp]
            gz = ground_z_under(wx, wy)
            base_z = max(wz, gz + 1.0) + 1.5  # a bit above ground/buildings

            radius = 0.6
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=radius,
                rgbaColor=(1.0, 0.3 + 0.03 * (k % 10), 0.1, 1.0),
            )
            oid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[wx, wy, base_z],
            )

            moving_obstacles.append(
                {
                    "id": oid,
                    "base_pos": np.array([wx, wy, base_z], dtype=float),
                    "amp_xy": 3.0 + 0.3 * (k % 5),
                    "omega": 0.6 + 0.15 * (k % 7),
                    "phase": 0.8 * k,
                }
            )

    # Helper to list all obstacle bodies for occupancy rebuilds
    def all_obstacle_bodies():
        return list(buildings) + [mo["id"] for mo in moving_obstacles]

    # ---- Kinematic runner + dynamic buildings + flying balls + repeated replans ----
    idx = 0
    pos = np.array(waypoints[0], dtype=float)
    speed = float(max(0.05, args.speed))
    eps = 1e-6

    orig_path_len = path_length(waypoints)
    replan_step_dist = 0.05 * orig_path_len if orig_path_len > 0 else float("inf")  # 5% of ORIGINAL path
    dist_since_replan = 0.0
    prev_pos_for_replan = pos.copy()
    replan_count = 0
    replan_colors = [
        (0, 1, 0),      # first replan: green
        (1, 1, 0),      # second: yellow
        (1, 0, 1),      # third: magenta
        (0, 1, 1),      # fourth+: cyan
    ]

    if not args.quiet:
        print(
            f"[nav] Start {tuple(np.round(waypoints[0], 2))} -> "
            f"Goal {tuple(np.round(waypoints[-1], 2))} | "
            f"{len(waypoints)} waypoints | speed={speed} m/s "
            f"| replan every ~{replan_step_dist:.2f} m (≈5% of original path)"
        )

    sim_time = 0.0
    done_path = False

    def step_towards(a, b, max_step):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        d = b - a
        L = float(np.linalg.norm(d))
        if L <= max_step + eps:
            return b, True
        return a + d / (L + eps) * max_step, False

    while True:
        if not p.isConnected():
            break

        sim_time += dt

        # 1) Move selected buildings (dynamic buildings)
        for mb in moving_buildings:
            bid = mb["id"]
            base_pos = mb["base_pos"]
            orn_b = mb["orn"]
            axis = mb["axis"]
            amp = mb["amp"]
            omega = mb["omega"]

            offset = amp * math.sin(omega * sim_time)
            if axis == 0:
                new_pos = [base_pos[0] + offset, base_pos[1], base_pos[2]]
            else:
                new_pos = [base_pos[0], base_pos[1] + offset, base_pos[2]]
            p.resetBasePositionAndOrientation(bid, new_pos, orn_b)

        # 2) Move flying obstacles (spheres) in circles in the XY-plane
        for mo in moving_obstacles:
            oid = mo["id"]
            bx, by, bz = mo["base_pos"]
            amp_xy = mo["amp_xy"]
            omega = mo["omega"]
            phase = mo["phase"]

            x = bx + amp_xy * math.cos(omega * sim_time + phase)
            y = by + amp_xy * math.sin(omega * sim_time + phase)
            z = bz
            p.resetBasePositionAndOrientation(oid, [x, y, z], [0, 0, 0, 1])

        # 3) Keyboard, camera control
        if args.gui:
            ev = p.getKeyboardEvents()
            if ev.get(ESC) == p.KEY_WAS_TRIGGERED:
                break
            if ev.get(KEY("c")) == p.KEY_WAS_TRIGGERED:
                cam.follow = not cam.follow
            if ev.get(KEY("f")) == p.KEY_WAS_TRIGGERED and cam.follow:
                p.resetDebugVisualizerCamera(
                    cam.dist, cam.yaw, cam.pitch, pos.tolist()
                )
            if ev.get(KEY("o")) == p.KEY_WAS_TRIGGERED:
                cam.los_guard = not cam.los_guard
            if ev.get(KEY("[")) == p.KEY_IS_DOWN:
                cam.dist = max(cam.dist_min, cam.dist - 0.25)
            if ev.get(KEY("]")) == p.KEY_IS_DOWN:
                cam.dist = min(cam.dist_max, cam.dist + 0.25)

        # 4) Follow current waypoint list
        target = np.array(waypoints[idx], dtype=float)
        step_len = speed * dt
        pos, reached = step_towards(pos, target, step_len)
        if reached:
            if idx < len(waypoints) - 1:
                idx += 1
            else:
                done_path = True

        # distance travelled since last replan
        step_dist = float(np.linalg.norm(pos - prev_pos_for_replan))
        dist_since_replan += step_dist
        prev_pos_for_replan = pos.copy()

        # 5) Trigger a replan every ~5% of ORIGINAL path length
        if (not done_path) and (dist_since_replan >= replan_step_dist) and (orig_path_len > 0):
            dist_since_replan = 0.0
            replan_count += 1

            # Rebuild occupancy from *current* poses of buildings + flying obstacles
            rects_dyn = buildings_as_rects(all_obstacle_bodies())
            occ3 = grid3d_from_buildings(
                rects_dyn,
                res=args.res,
                vres=args.vres,
                zmin=args.zmin,
                zmax=args.zmax,
                margin=args.margin,
                zmargin=args.zmargin,
            )

            # Current UAV position -> grid
            sx_cur, sy_cur, sz_cur = pos.tolist()
            si, sj, sk = world_to_cell_3d(
                sx_cur, sy_cur, sz_cur, args.res, args.vres, args.zmin
            )
            gi, gj, gk = world_to_cell_3d(
                gx, gy, gz, args.res, args.vres, args.zmin
            )

            planner = DStarLite3D(
                occ3,
                start=(si, sj, sk),
                goal=(gi, gj, gk),
                res_xy=args.res,
                res_z=args.vres,
            )
            new_path = planner.plan()
            if new_path:
                new_waypoints = [
                    cell_to_world_3d(
                        i, j, k, args.res, args.vres, args.zmin
                    )
                    for (i, j, k) in new_path
                ]
                if len(new_waypoints) > 3:
                    new_waypoints = new_waypoints[::2] + [new_waypoints[-1]]

                # Draw corrected path in color
                if args.gui:
                    color_idx = min(replan_count - 1, len(replan_colors) - 1)
                    draw_polyline3d(new_waypoints, width=2, rgb=replan_colors[color_idx])

                waypoints = new_waypoints
                idx = 0  # follow from start of new path

                dist_since_replan = 0.0
                prev_pos_for_replan = pos.copy()

                if not args.quiet:
                    print(
                        f"[nav] Replanned path #{replan_count} at t={sim_time:.1f}s, "
                        f"{len(waypoints)} waypoints."
                    )

        # 6) Update UAV pose & camera
        if idx < len(waypoints):
            dir_xy = np.array(
                [waypoints[idx][0] - pos[0], waypoints[idx][1] - pos[1]]
            )
        else:
            dir_xy = np.array([1.0, 0.0])
        if np.linalg.norm(dir_xy) < 1e-6:
            yaw = 0.0
        else:
            yaw = math.atan2(dir_xy[1], dir_xy[0])
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])

        p.resetBasePositionAndOrientation(uav, pos.tolist(), quat)

        if args.gui and cam.follow:
            cam.update(uav, dir_hint_xy=dir_xy)

        p.stepSimulation()
        time.sleep(dt)

        if done_path:
            break

    # Linger for visualization
    if args.linger != 0:
        if args.linger < 0:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(dt)
        else:
            t_end = time.time() + float(args.linger)
            while p.isConnected() and time.time() < t_end:
                p.stepSimulation()
                time.sleep(dt)

if __name__ == "__main__":
    main()
