# occupancy.py — 3D occupancy grid helpers (rect buildings -> voxel grid)
import numpy as np

try:
    import uav_config as cfg
    TERRAIN_SIZE = getattr(cfg, "TERRAIN_SIZE", 80.0)
except Exception:
    TERRAIN_SIZE = 80.0

def _half():
    return float(TERRAIN_SIZE) * 0.5

def world_to_cell_3d(x, y, z, res, vres, zmin):
    """World (x,y,z) -> integer (i,j,k) using grid origin (-half,-half,zmin)."""
    half = _half()
    i = int(np.floor((x + half) / float(res)))
    j = int(np.floor((y + half) / float(res)))
    k = int(np.floor((z - float(zmin)) / float(vres)))
    return i, j, k

def cell_to_world_3d(i, j, k, res, vres, zmin):
    """Integer (i,j,k) -> world (x,y,z) at cell centers."""
    half = _half()
    x = -half + (i + 0.5) * float(res)
    y = -half + (j + 0.5) * float(res)
    z = float(zmin) + (k + 0.5) * float(vres)
    return [x, y, z]

def grid3d_from_buildings(rects, res=0.6, vres=0.6, zmin=0.0, zmax=35.0, margin=0.6, zmargin=0.6):
    """
    rects: [(cx,cy,w,d,h,bid), ...] where h is height above ground.
    Returns occ[nx,ny,nz] bool, True=occupied.
    """
    half = _half()
    nx = int(np.ceil((2 * half) / float(res)))
    ny = int(np.ceil((2 * half) / float(res)))
    nz = int(np.ceil((float(zmax) - float(zmin)) / float(vres)))
    occ = np.zeros((nx, ny, nz), dtype=bool)

    if not rects:
        return occ

    # Precompute voxel centers for z slices only; xy is done by index math
    z_centers = (np.arange(nz) + 0.5) * float(vres) + float(zmin)

    # For each building, fill voxels whose center lies inside the inflated prism
    for (cx, cy, w, d, h, _bid) in rects:
        x_min = float(cx) - 0.5 * float(w) - float(margin)
        x_max = float(cx) + 0.5 * float(w) + float(margin)
        y_min = float(cy) - 0.5 * float(d) - float(margin)
        y_max = float(cy) + 0.5 * float(d) + float(margin)
        z_top = float(h) + float(zmargin)  # base at ~0, inflate upwards

        # Clamp to grid bounds
        i0 = int(np.clip(np.floor((x_min + half) / res), 0, nx - 1))
        i1 = int(np.clip(np.ceil ((x_max + half) / res), 0, nx))
        j0 = int(np.clip(np.floor((y_min + half) / res), 0, ny - 1))
        j1 = int(np.clip(np.ceil ((y_max + half) / res), 0, ny))
        # z indices by center threshold
        kz = np.where((z_centers >= float(zmin)) & (z_centers <= min(z_top, float(zmax))))[0]
        if kz.size == 0: continue

        # Fill
        occ[i0:i1, j0:j1, kz] = True

    return occ

