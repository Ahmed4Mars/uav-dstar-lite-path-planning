# run_direct_nav.py  — A* follower with fixed-alt option & linger at goal
import argparse, math, time
import numpy as np
import pybullet as p, pybullet_data

from occupancy import grid_from_buildings, world_to_cell, cell_to_world
from planner_grid import astar
try:
    from create_uav_world import create_world
except Exception:
    create_world = None

DT = 1.0/240.0
GRAVITY = -9.81
TERRAIN_SIZE = 80.0

def ground_z(x, y, top=200.0, bottom=-200.0):
    hit = p.rayTest([x,y,top], [x,y,bottom])[0]
    return hit[3][2] if hit[0] >= 0 else 0.0

def add_marker(x, y, z, rgba):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.28)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.28, rgbaColor=rgba)
    p.createMultiBody(0, col, vis, [x, y, z+0.3])

def buildings_as_rects(buildings):
    rects = []
    for b in buildings:
        if isinstance(b, (list, tuple)) and len(b) >= 6:
            cx, cy, w, d, h, bid = b[:6]
            rects.append((cx, cy, w, d, h, bid))
        else:
            bid = int(b)
            (minx, miny, minz), (maxx, maxy, maxz) = p.getAABB(bid)
            cx = 0.5*(minx+maxx); cy = 0.5*(miny+maxy)
            w = maxx-minx; d = maxy-miny; h = maxz-minz
            rects.append((cx, cy, w, d, h, bid))
    return rects

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--start", type=str, default=None)  # "x,y"
    ap.add_argument("--goal",  type=str, default=None)  # "x,y"

    # altitude control
    ap.add_argument("--alt_mode", choices=["abs","agl"], default="abs",
                    help="altitude mode: abs=fixed world Z, agl=above ground")
    ap.add_argument("--alt", type=float, default=2.5,
                    help="if abs: world Z; if agl: meters above ground")

    # motion & arrival behavior
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--vclamp", type=float, default=2.5)
    ap.add_argument("--tol",   type=float, default=0.5)
    ap.add_argument("--linger", type=float, default=-1,
                    help="-1 hover forever at goal; 0 exit; >0 seconds then exit")

    # follow camera (GUI only)
    ap.add_argument("--cam", action="store_true")
    ap.add_argument("--cam_dist", type=float, default=8.0)
    ap.add_argument("--cam_pitch", type=float, default=-20.0)
    ap.add_argument("--cam_yaw_offset", type=float, default=20.0)
    ap.add_argument("--cam_ahead", type=float, default=1.8)
    ap.add_argument("--cam_smooth", type=float, default=0.18)
    args = ap.parse_args()

    if p.isConnected(): p.disconnect()
    cid = p.connect(p.GUI if args.gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, GRAVITY)
    p.setTimeStep(DT)

    # world
    buildings = []
    if create_world:
        world_ret = create_world()
        if isinstance(world_ret, tuple) and len(world_ret) >= 2:
            buildings = world_ret[1]
    else:
        p.loadURDF("plane.urdf")

    # start/goal
    if args.start: sx, sy = map(float, args.start.split(","))
    else: sx, sy = 0.0, 0.0
    half = TERRAIN_SIZE*0.5
    if args.goal:
        gx, gy = map(float, args.goal.split(","))
    else:
        gx = np.clip(half*0.75, -half*0.9, half*0.9)
        gy = np.clip(-half*0.75, -half*0.9, half*0.9)

    sz = ground_z(sx, sy)
    gz = ground_z(gx, gy)
    add_marker(sx, sy, sz, (0.9,0.2,0.2,1))
    add_marker(gx, gy, gz, (0.2,0.9,0.2,1))

    # vehicle
    start_pos = [sx, sy, (args.alt if args.alt_mode=="abs" else sz+max(0.3, args.alt))]
    try:
        uav = p.loadURDF("uav.urdf", start_pos, useFixedBase=False)
    except Exception:
        col = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.15, height=0.2)
        vis = p.createVisualShape(p.GEOM_CAPSULE, radius=0.15, length=0.2, rgbaColor=[0.4,0.6,1.0,1])
        uav = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=start_pos)

    # plan (A*)
    rects = buildings_as_rects(buildings)
    occ = grid_from_buildings(rects, res=args.res, margin=args.margin)
    s_i, s_j = world_to_cell(sx, sy, args.res)
    g_i, g_j = world_to_cell(gx, gy, args.res)
    path_ij = astar(occ, (s_i, s_j), (g_i, g_j))
    if not path_ij:
        print("[nav] No path. Try --margin 0.6 or --res 0.3"); return
    waypoints = [cell_to_world(i, j, args.res) for (i, j) in path_ij]
    print(f"[nav] Start ({sx:.1f},{sy:.1f}) → Goal ({gx:.1f},{gy:.1f}) | {len(waypoints)} waypoints")

    # visualize A* as a blue polyline (yes: this is the path it follows)
    z_line = (args.alt if args.alt_mode=="abs" else max(0.2, args.alt*0.2))
    for a, b in zip(waypoints, waypoints[1:]):
        p.addUserDebugLine([a[0],a[1],z_line],[b[0],b[1],z_line],[0,0,1],2,0)

    # camera
    cam_target = np.array([sx, sy, start_pos[2]])
    def cam_update(x, y, z, heading_yaw):
        nonlocal cam_target
        ahead = np.array([math.cos(heading_yaw), math.sin(heading_yaw), 0.0]) * args.cam_ahead
        desired = np.array([x, y, z]) + ahead
        cam_target = cam_target*(1.0-args.cam_smooth) + desired*args.cam_smooth
        yaw_deg = math.degrees(heading_yaw) + args.cam_yaw_offset
        p.resetDebugVisualizerCamera(args.cam_dist, args.cam_pitch, yaw_deg, cam_target.tolist())

    # altitude reference
    fixed_world_Z = args.alt if args.alt_mode == "abs" else None

    idx = 0
    max_hz_print = 4.0
    t_last = 0.0
    arrived_t0 = None

    while True:
        pos, orn = p.getBasePositionAndOrientation(uav)
        x,y,z = pos

        # altitude setpoint
        if args.alt_mode == "abs":
            z_ref = fixed_world_Z
        else:
            z_ref = ground_z(x, y) + args.alt  # AGL

        # arrival / linger logic
        if idx >= len(waypoints):
            # hold position at goal
            vz = np.clip((z_ref - z) * 1.2, -args.vclamp, args.vclamp)
            p.resetBaseVelocity(uav, [0,0,vz], [0,0,0])
            if arrived_t0 is None:
                arrived_t0 = time.time()
                print("[done] reached goal — hovering.")
            if args.linger == 0:
                break
            if args.linger > 0 and (time.time() - arrived_t0) >= args.linger:
                break
            if args.gui and args.cam:
                # keep camera alive pointing at us; derive yaw from last heading (zero here)
                cam_update(x, y, z, 0.0)
            p.stepSimulation(); time.sleep(DT)
            continue

        # guidance to next A* waypoint
        tx, ty = waypoints[idx]
        dx, dy = tx - x, ty - y
        dist_xy = math.hypot(dx, dy)
        if dist_xy < args.tol:
            idx += 1
            continue

        v = args.speed
        vx = (dx / (dist_xy + 1e-9)) * v
        vy = (dy / (dist_xy + 1e-9)) * v
        vz = np.clip((z_ref - z) * 1.2, -args.vclamp, args.vclamp)

        # yaw along velocity
        heading_yaw = math.atan2(vy, vx) if abs(vx)+abs(vy) > 1e-6 else 0.0
        cy, sy = math.cos(heading_yaw*0.5), math.sin(heading_yaw*0.5)
        q = [0.0, 0.0, sy, cy]
        p.resetBasePositionAndOrientation(uav, [x,y,z], q)

        p.resetBaseVelocity(uav, [vx, vy, vz], [0,0,0])

        if args.gui and args.cam:
            cam_update(x, y, z, heading_yaw)

        now = time.time()
        if now - t_last > 1.0/max_hz_print:
            t_last = now
            print(f"[trk] wp {idx+1}/{len(waypoints)} | pos=({x:.2f},{y:.2f},{z:.2f}) "
                  f"→ ({tx:.2f},{ty:.2f}) d={dist_xy:.2f} | v=({vx:.2f},{vy:.2f},{vz:.2f}) zref={z_ref:.2f}")

        p.stepSimulation()
        time.sleep(DT)

if __name__ == "__main__":
    main()
