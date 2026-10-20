# 3D A* / D* Lite Path Planning for a UAV in PyBullet

Term project for **CIE 483 – AI and ML for Robotics** (KFUPM, Fall 2025 / T251).

A simulated quadrotor plans and flies collision-free paths through a randomly generated 3D city (buildings on flat ground or a heightfield terrain) using grid-based search and replans as obstacles move. A PD hover controller that holds altitude above the local ground is included as a separate demo.

![D* replanning demo](media/dstar_2d_replanning.gif)

## Features

- **PyBullet city environment** – configurable terrain size, building count and dimensions, flat plane or heightfield.
- **3D occupancy grid** – buildings voxelised with configurable XY/Z resolution and obstacle inflation margins.
- **3D A\*** global planner and **3D D\* Lite** (`DStarLite3D`) planner on the voxel grid.
- **Dynamic obstacles** – moving buildings and flying spheres; the occupancy grid is rebuilt and the path replanned every ~5% of the original path length (each replan drawn in a new color).
- **Path following** – the UAV moves kinematically along the waypoints, with a smart follow camera (optional line-of-sight guard).
- **PD hover controller** that holds a target altitude above ground using ray-cast ground-height queries.
- **2D dynamic replanning demos** – grid world with static and moving obstacles; A* is re-run periodically (D*-style replanning) and every superseded path is drawn.

| PyBullet city (top view) | 3D planned path |
|---|---|
| ![](media/pybullet_city_top_view.png) | ![](media/dstar_3d_demo.jpg) |

| Heightfield terrain | 2D D* demo |
|---|---|
| ![](media/heightfield_colormap.png) | ![](media/dstar_2d_demo.png) |

## Repository structure

```
uav_3d_nav/                 # 3D planning + flight in PyBullet
  run_uav_nav.py            # main script: build city, plan in 3D, fly the path
  dstar_lite_3d.py          # 3D D* Lite on a voxel grid
  occupancy.py              # building -> occupancy grid helpers
  planner_grid.py           # 2D A* on a grid
  follower.py               # path follower
  run_direct_nav.py         # early 2D navigation experiment (older occupancy API, not maintained)
  run_uav_hover.py          # hover-only test
  create_uav_world.py       # terrain + building generation
  uav_controller.py         # PD hover controller
  uav_config.py             # physics, world and controller parameters
  uav.urdf                  # quadrotor model
dstar_2d_demo/              # standalone 2D replanning demos (matplotlib)
  dynamic_planner.py        # DynamicPlanner class
  run_dynamic_demo.py       # static grid -> add obstacle -> replan
  run_dstar_gridworld.py    # 120x80 grid with moving obstacles, animated
media/                      # figures and demo GIF
```

## Setup

Python 3.12 recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

3D city navigation (final configuration used in the project):

```bash
cd uav_3d_nav
python run_uav_nav.py --gui --cam --res 0.6 --vres 0.6 --zmax 150 --speed 3
```

Useful flags: `--start x,y,z` / `--goal x,y,z`, `--margin` / `--zmargin` (obstacle inflation), `--cam_los_guard` (keep the camera from being blocked by buildings), `--quiet`. Run `python run_uav_nav.py -h` for the full list. World size and building counts are set in `uav_config.py`.

Hover only:

```bash
python run_uav_hover.py
```

2D replanning demos:

```bash
cd dstar_2d_demo
python run_dynamic_demo.py
python run_dstar_gridworld.py
```

## Limitations

- `DStarLite3D` uses the D* Lite structure (g/rhs values, priority keys) but is re-instantiated at each replan instead of updating incrementally.
- In the navigation script the UAV pose is set kinematically; the PD hover controller is exercised in `run_uav_hover.py`, and full physics-based waypoint tracking is not tuned.
- Perception is ideal – obstacles are read directly from the simulator.

## Team

KFUPM CIE 483, Section 02, Group 01 – Jassim AlSayegh, Ahmed AlSayegh, Mohsn Alsaidi, Renas Koko.
Instructor: Dr. Ramy Rashad. The base PyBullet UAV hover environment was provided as course starter code.
