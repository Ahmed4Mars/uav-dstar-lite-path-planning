import time, math
import pybullet as p, pybullet_data
from uav_config import (DT, GRAVITY, LINEAR_DAMPING, ANGULAR_DAMPING,
                        START_AGL, HOVER_AGL)
from create_uav_world import create_world
from uav_controller import HoverController

# --- Camera helper (accepts extra kwargs like 'follow') ---
def focus_on(body, dist=12.0, yaw=50, pitch=-25, **_):
    pos, _ = p.getBasePositionAndOrientation(body)
    p.resetDebugVisualizerCamera(dist, yaw, pitch, pos)

def total_mass(body_id):
    m = p.getDynamicsInfo(body_id, -1)[0]
    for j in range(p.getNumJoints(body_id)):
        m += p.getDynamicsInfo(body_id, j)[0]
    return m

def ground_z_under(x, y, exclude_body_id=None, probe_top=200.0, probe_bottom=-200.0):
    """
    Robust ground query:
      1) If exclude_body_id is provided, start the ray BELOW that body so we can't hit ourselves.
      2) Otherwise, cast top->bottom as usual.
    """
    if exclude_body_id is not None:
        # Start below the UAV bottom (assumes the quad is small; adjust margin if you change the model)
        pos, _ = p.getBasePositionAndOrientation(exclude_body_id)
        z_start = pos[2] - 0.30  # 30 cm below the UAV center → below its collision box
        from_pt = [x, y, z_start]
        to_pt   = [x, y, probe_bottom]
        hit = p.rayTest(from_pt, to_pt)[0]
        if hit[0] >= 0:
            return hit[3][2]
        return 0.0
    else:
        hit = p.rayTest([x, y, probe_top], [x, y, probe_bottom])[0]
        if hit[0] >= 0:
            return hit[3][2]
        return 0.0

def main():
    p.connect(p.GUI)
    p.setTimeStep(DT)
    p.setGravity(0,0,GRAVITY)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    terrain_id, buildings, heights = create_world()

    # Spawn above ground at origin
    gx = gy = 0.0
    gz = ground_z_under(gx, gy)
    start_pos = [gx, gy, gz + START_AGL]
    uav = p.loadURDF("uav.urdf", start_pos, useFixedBase=False)
    p.changeDynamics(uav, -1, linearDamping=LINEAR_DAMPING, angularDamping=ANGULAR_DAMPING)
    try: p.changeVisualShape(uav, -1, rgbaColor=[0.2, 0.3, 1.0, 1.0])
    except: pass

    # Match controller mass to the actual sim mass (single-link URDF should already be 1.6 kg)
    ctrl = HoverController(target_alt=gz + HOVER_AGL)
    ctrl.m = total_mass(uav)

    # Camera and follow
    cam = {"dist": 14.0, "yaw": 45.0, "pitch": -25.0, "follow": True}
    focus_on(uav, **cam)

    print("[INFO] Hovering AGL. Keys: F=focus, C=toggle follow, Esc=quit")
    KEY = lambda s: ord(s); ESC = 27
    while p.isConnected():
        ev = p.getKeyboardEvents()
        if ev.get(ESC) == p.KEY_WAS_TRIGGERED: break
        if ev.get(KEY('f')) == p.KEY_WAS_TRIGGERED: focus_on(uav, **cam)
        if ev.get(KEY('c')) == p.KEY_WAS_TRIGGERED:
            cam["follow"] = not cam["follow"]
            print(f"[INFO] Follow camera: {cam['follow']}")

        # Keep camera on UAV if following
        if cam["follow"]:
            pos, _ = p.getBasePositionAndOrientation(uav)
            p.resetDebugVisualizerCamera(cam["dist"], cam["yaw"], cam["pitch"], pos)

        # Maintain AGL target, IGNORING the UAV in the ray
        pos, _ = p.getBasePositionAndOrientation(uav)
        ground_z = ground_z_under(pos[0], pos[1], exclude_body_id=uav)
        ctrl.z_ref = ground_z + HOVER_AGL

        ctrl.step(uav, DT, yaw_ref=0.0, level_ref=(0.0, 0.0))
        p.stepSimulation()
        time.sleep(DT)

if __name__ == "__main__":
    main()
