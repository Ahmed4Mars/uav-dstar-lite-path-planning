
# Global configuration for the UAV environment

# --- Randomness ---
SEED = None


# --- Terrain (heightfield) ---
TERRAIN_SIZE = 120  # meters (square width/length)
HF_RES = 256             # grid resolution per side (must be power-of-two-ish)
HF_MAX_H = 10.0          # maximum height (meters)
HF_BASE = 0.0            # base offset

# Flat pad at origin for takeoff/landing
PAD_RADIUS = 4.0         # meters
PAD_HEIGHT = 0.2         # meters above base

# --- Buildings ---
N_BUILDINGS = 250
BLDG_W_RANGE = (2.0, 10)     # width  (x) meters
BLDG_D_RANGE = (2.0, 10)     # depth  (y) meters
BLDG_H_RANGE = (2.0, 70)     # height (z) meters
BLDG_CLEAR_RADIUS = 10      # keep area near (0,0) clear of buildings

# --- Physics ---
DT = 1.0/240.0
GRAVITY = -9.81
LINEAR_DAMPING = 0.02
ANGULAR_DAMPING = 0.02
DRAG_COEF = 0.25     # N*s/m "air drag" (simple linear model)

# --- UAV physical params ---
UAV_MASS = 1.6       # kg
UAV_ARM = 0.22       # meters from center to rotor
UAV_THRUST_MAX = 20.0   # N per rotor (for saturation margin)
UAV_THRUST_MIN = 0.0

# --- Controller gains ---
ALT_KP = 2.0
ALT_KD = 1.2

ROLL_KP = 2.8
ROLL_KD = 0.45

PITCH_KP = 2.8
PITCH_KD = 0.45

YAW_KP = 2.0
YAW_KD = 0.30

# --- World selection ---
WORLD_MODE = "plane"   # "heightfield" or "plane"
USE_TERRAIN_TEXTURE = False  # False = no grass texture

# Hover references (AGL = above ground level)
START_AGL = 0.6    # start 0.6 m above ground
HOVER_AGL = 50.0   # hover 12 m above ground

