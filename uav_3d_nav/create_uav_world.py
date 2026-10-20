import math, random, numpy as np
import pybullet as p
import pybullet_data
from uav_config import (
    SEED, TERRAIN_SIZE, HF_RES, HF_MAX_H, HF_BASE,
    PAD_RADIUS, PAD_HEIGHT, N_BUILDINGS, BLDG_W_RANGE, BLDG_D_RANGE,
    BLDG_H_RANGE, BLDG_CLEAR_RADIUS, WORLD_MODE, USE_TERRAIN_TEXTURE
)

random.seed(SEED)
np.random.seed(SEED)

def _fbm_noise(res, octaves=5, persistence=0.5):
    def smooth_noise(size):
        a = np.random.rand(size, size)
        # light smoothing w/out SciPy
        b = (np.roll(a,1,0)+a+np.roll(a,-1,0))/3.0
        b = (np.roll(b,1,1)+b+np.roll(b,-1,1))/3.0
        return b
    total = np.zeros((res,res), dtype=np.float32)
    amp = 1.0
    for _ in range(octaves):
        total += amp * smooth_noise(res)
        amp *= persistence
    total -= total.min()
    total /= max(1e-6, total.max())
    return total

def build_heightfield():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    hf = _fbm_noise(HF_RES, octaves=5)
    hf = hf**1.7
    heights = HF_BASE + hf * HF_MAX_H

    xs = np.linspace(-TERRAIN_SIZE/2, TERRAIN_SIZE/2, HF_RES)
    ys = np.linspace(-TERRAIN_SIZE/2, TERRAIN_SIZE/2, HF_RES)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    mask = (X**2 + Y**2) <= (PAD_RADIUS**2)
    heights[mask] = HF_BASE + PAD_HEIGHT

    mesh_scale = [TERRAIN_SIZE/(HF_RES-1), TERRAIN_SIZE/(HF_RES-1), 1.0]
    hf_flat = heights.reshape(-1)

    col = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=mesh_scale,
        heightfieldTextureScaling=1.0,
        heightfieldData=hf_flat.tolist(),
        numHeightfieldRows=HF_RES,
        numHeightfieldColumns=HF_RES
    )
    terrain_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=-1)

    # Texture is optional; you asked to remove grass, so default False
    if USE_TERRAIN_TEXTURE:
        try:
            tex_id = p.loadTexture("grass.png")
            p.changeVisualShape(terrain_id, -1, textureUniqueId=tex_id)
        except Exception:
            pass
    p.changeVisualShape(terrain_id, -1, rgbaColor=[0.55, 0.6, 0.55, 1.0])

    return terrain_id, heights

def _height_at(x, y, heights):
    sx = (x + TERRAIN_SIZE/2) * (HF_RES-1) / TERRAIN_SIZE
    sy = (y + TERRAIN_SIZE/2) * (HF_RES-1) / TERRAIN_SIZE
    ix = int(np.clip(np.floor(sx), 0, HF_RES-2))
    iy = int(np.clip(np.floor(sy), 0, HF_RES-2))
    tx, ty = sx - ix, sy - iy
    z00 = heights[iy, ix];   z10 = heights[iy, ix+1]
    z01 = heights[iy+1, ix]; z11 = heights[iy+1, ix+1]
    z0 = z00*(1-tx) + z10*tx
    z1 = z01*(1-tx) + z11*tx
    return z0*(1-ty) + z1*ty

def place_buildings_on_heightfield(heights, n):
    placed = []
    for _ in range(n):
        w = float(np.random.uniform(*BLDG_W_RANGE))
        d = float(np.random.uniform(*BLDG_D_RANGE))
        h = float(np.random.uniform(*BLDG_H_RANGE))
        ok=False
        for _try in range(200):
            x = float(np.random.uniform(-TERRAIN_SIZE/2+w/2, TERRAIN_SIZE/2-w/2))
            y = float(np.random.uniform(-TERRAIN_SIZE/2+d/2, TERRAIN_SIZE/2-d/2))
            if (x**2 + y**2) < (BLDG_CLEAR_RADIUS**2): continue
            z0 = _height_at(x, y, heights); ok=True; break
        if not ok: continue
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[w/2, d/2, h/2])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[w/2, d/2, h/2])
        body = p.createMultiBody(0, col, vis, [x, y, z0 + h/2])
        p.changeVisualShape(body, -1, rgbaColor=[
            float(np.random.uniform(0.4,0.9)),
            float(np.random.uniform(0.4,0.9)),
            float(np.random.uniform(0.4,0.9)),
            1.0])
        placed.append(body)
    return placed

def place_buildings_on_plane(n):
    placed = []
    for _ in range(n):
        w = float(np.random.uniform(*BLDG_W_RANGE))
        d = float(np.random.uniform(*BLDG_D_RANGE))
        h = float(np.random.uniform(*BLDG_H_RANGE))
        ok=False
        for _try in range(200):
            x = float(np.random.uniform(-TERRAIN_SIZE/2+w/2, TERRAIN_SIZE/2-w/2))
            y = float(np.random.uniform(-TERRAIN_SIZE/2+d/2, TERRAIN_SIZE/2-d/2))
            if (x**2 + y**2) < (BLDG_CLEAR_RADIUS**2): continue
            ok=True; break
        if not ok: continue
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[w/2, d/2, h/2])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[w/2, d/2, h/2])
        body = p.createMultiBody(0, col, vis, [x, y, h/2])  # plane at z=0
        p.changeVisualShape(body, -1, rgbaColor=[
            float(np.random.uniform(0.4,0.9)),
            float(np.random.uniform(0.4,0.9)),
            float(np.random.uniform(0.4,0.9)),
            1.0])
        placed.append(body)
    return placed

def create_world():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(numSolverIterations=120, numSubSteps=1)

    if WORLD_MODE == "plane":
        # **Flat world**: plane + buildings, no heightfield
        plane = p.loadURDF("plane.urdf", [0,0,0])
        buildings = place_buildings_on_plane(N_BUILDINGS)
        heights = None
        terrain_id = plane
    else:
        # **Heightfield world** (no grass texture unless USE_TERRAIN_TEXTURE=True)
        terrain_id, heights = build_heightfield()
        buildings = place_buildings_on_heightfield(heights, N_BUILDINGS)

    return terrain_id, buildings, heights
