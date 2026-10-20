
import math, numpy as np
import pybullet as p
from uav_config import (UAV_MASS, UAV_THRUST_MAX, UAV_THRUST_MIN, GRAVITY, DRAG_COEF,
                        ALT_KP, ALT_KD, ROLL_KP, ROLL_KD, PITCH_KP, PITCH_KD, YAW_KP, YAW_KD)

def _rot_matrix(quat):
    R = p.getMatrixFromQuaternion(quat)
    return np.array(R, dtype=float).reshape(3,3)

def _wrap(a): return (a + math.pi) % (2*math.pi) - math.pi

class HoverController:
    def __init__(self, target_alt):
        self.z_ref = target_alt
        self.g = abs(GRAVITY)
        self.m = UAV_MASS

    def step(self, body_id, dt, yaw_ref=0.0, level_ref=(0.0, 0.0)):
        pos, quat = p.getBasePositionAndOrientation(body_id)
        vel_lin, vel_ang = p.getBaseVelocity(body_id)
        x,y,z = pos
        vx,vy,vz = vel_lin
        wx,wy,wz = vel_ang

        R = _rot_matrix(quat)
        roll, pitch, yaw = p.getEulerFromQuaternion(quat)

        # Altitude
        e_z   = self.z_ref - z
        ed_z  = -vz
        u_thrust = self.m*(self.g + ALT_KP*e_z + ALT_KD*ed_z)

        # Force along body +Z in world frame
        z_body_world = R[:,2]
        F_world = z_body_world * max(UAV_THRUST_MIN, min(4*UAV_THRUST_MAX, u_thrust))

        # Attitude
        e_roll  = _wrap(level_ref[0] - roll)
        e_pitch = _wrap(level_ref[1] - pitch)
        e_yaw   = _wrap(yaw_ref - yaw)

        tau_roll  = ROLL_KP*e_roll  + ROLL_KD*(-wx)
        tau_pitch = PITCH_KP*e_pitch+ PITCH_KD*(-wy)
        tau_yaw   = YAW_KP*e_yaw    + YAW_KD*(-wz)
        tau_body  = np.array([tau_roll, tau_pitch, tau_yaw], dtype=float)

        # Drag in world frame
        F_drag = -DRAG_COEF * np.array([vx, vy, vz], dtype=float)

        p.applyExternalForce(body_id, -1, F_world.tolist(), [0,0,0], p.WORLD_FRAME)
        p.applyExternalForce(body_id, -1, F_drag.tolist(),  [0,0,0], p.WORLD_FRAME)
        p.applyExternalTorque(body_id, -1, tau_body.tolist(), p.LINK_FRAME)
