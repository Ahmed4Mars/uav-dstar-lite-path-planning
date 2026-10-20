# follower.py — 3D waypoint follower (returns roll/pitch in radians + alt)
import math
import numpy as np
import pybullet as p

class WaypointFollower:
    def __init__(self, kp=0.20, kd=0.08, arrive=0.75, slow_radius=2.5,
                 max_deg=12.0, quiet=True):
        self.kp = float(kp)
        self.kd = float(kd)
        self.arrive = float(arrive)
        self.slow_radius = float(slow_radius)
        self.max_rad = float(max_deg) * math.pi / 180.0
        self.idx = 0
        self.quiet = quiet
        self.prev_err_xy = np.array([0.0, 0.0])

    def _level_cmd(self, to_xy, vel_xy):
        """Simple PD -> desired tilt (roll,pitch) in *radians*."""
        dist = float(np.linalg.norm(to_xy))
        if dist < 1e-9:
            return 0.0, 0.0

        # direction towards target in XY
        dir_xy = to_xy / (dist + 1e-9)

        # speed scale: tilt less when close (arrive / slow radius)
        if dist < self.slow_radius:
            scale = dist / max(self.slow_radius, 1e-6)
        else:
            scale = 1.0

        # PD vector in world XY
        err_xy = dir_xy * self.kp * dist
        dterm  = -self.kd * np.array(vel_xy[:2])
        a_xy   = (err_xy + dterm) * scale

        # Map desired horizontal acceleration to tilt (small-angle approx)
        # roll ~ +ay, pitch ~ -ax  (common convention; axis mapper will fix if needed)
        pitch = float(-a_xy[0])
        roll  = float(+a_xy[1])

        # limit to max tilt
        mag = math.hypot(roll, pitch)
        if mag > self.max_rad > 1e-9:
            roll  *= self.max_rad / mag
            pitch *= self.max_rad / mag
        return roll, pitch

    def step(self, body_id, waypoints_xyz):
        # guard
        if not waypoints_xyz:
            return {"roll": 0.0, "pitch": 0.0, "alt": 0.0, "done": True}

        # current pose/vel
        pos, _ = p.getBasePositionAndOrientation(body_id)
        vel, _ = p.getBaseVelocity(body_id)

        # progress logic
        n = len(waypoints_xyz)
        self.idx = min(max(int(self.idx), 0), n - 1)
        tgt = np.array(waypoints_xyz[self.idx], dtype=float)
        to_vec = tgt - np.array(pos, dtype=float)
        dist_xy = float(np.linalg.norm(to_vec[:2]))
        dist_3d = float(np.linalg.norm(to_vec))

        # advance when within arrive radius (in 3D, but mostly XY-focused)
        if dist_3d <= self.arrive and self.idx < n - 1:
            self.idx += 1
            tgt = np.array(waypoints_xyz[self.idx], dtype=float)
            to_vec = tgt - np.array(pos, dtype=float)
            dist_xy = float(np.linalg.norm(to_vec[:2]))

        # compute level commands
        roll, pitch = self._level_cmd(to_vec[:2], vel[:2])

        # desired altitude is target waypoint z (tracked by outer loop)
        alt = float(tgt[2])

        done = (self.idx >= n - 1) and (dist_3d <= self.arrive)
        return {"roll": roll, "pitch": pitch, "alt": alt, "done": done}

