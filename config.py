import os
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "data": os.path.join(BASE_DIR, "data", "processed"),
    "figures": os.path.join(BASE_DIR, "outputs", "figures"),
    "intermediate": os.path.join(BASE_DIR, "outputs", "intermediate"),
}

# create directories if they don’t exist
for p in PATHS.values():
    os.makedirs(p, exist_ok=True)


# =========================================================
# DOMAIN + GRID
# =========================================================

Lx = 0.20  # meters
Ly = 0.20  # meters

Nx = 64
Ny = 64

Nx_fdtd = 100
Ny_fdtd = 100

# =========================================================
# PHYSICAL CONSTANTS
# =========================================================

c0 = 3.0e8
eps0 = 8.854e-12
mu0 = 4 * np.pi * 1e-7

# =========================================================
# FREQUENCY
# =========================================================

f = 1.0e9
omega = 2 * np.pi * f

# =========================================================
# BACKGROUND MEDIUM
# =========================================================

eps_r_bg = 9.0
sigma_bg = 0.4

# =========================================================
# ANTENNA CONFIGURATION
# =========================================================

N_ant = 16
R_ant = 0.09

theta = np.linspace(0, 2 * np.pi, N_ant, endpoint=False)
ant_x = R_ant * np.cos(theta)
ant_y = R_ant * np.sin(theta)

# =========================================================
# FDTD SETTINGS
# =========================================================

dx_fdtd = Lx / Nx_fdtd
dy_fdtd = Ly / Ny_fdtd

courant = 0.5
dt = courant * min(dx_fdtd, dy_fdtd) / (c0 * np.sqrt(2))

n_steps = 500

# =========================================================
# PARAMETER RANGES (DATA GENERATION)
# =========================================================

# keep tumor away from boundaries / antennas
XY_LIMIT = 0.06

X_RANGE = (-XY_LIMIT, XY_LIMIT)
Y_RANGE = (-XY_LIMIT, XY_LIMIT)

R_RANGE = (0.005, 0.02)     # meters
C_RANGE = (20.0, 60.0)      # relative permittivity (eps_r_t)

SIGMA_T_RANGE = (2.0, 6.0)  # conductivity range (optional)

# =========================================================
# DATASET SIZES
# =========================================================

N_SAMPLES_XY = 300
N_SAMPLES_XYR = 400
N_SAMPLES_XYRC = 500

# =========================================================
# NORMALIZATION HELPERS
# =========================================================

def normalize_targets(y):
    """
    Normalize targets to roughly [-1, 1] scale
    """
    y_norm = y.copy()

    # x, y
    y_norm[..., 0] /= XY_LIMIT
    y_norm[..., 1] /= XY_LIMIT

    if y.shape[-1] >= 3:
        y_norm[..., 2] /= R_RANGE[1]

    if y.shape[-1] >= 4:
        y_norm[..., 3] /= C_RANGE[1]

    return y_norm


def denormalize_targets(y_norm):
    y = y_norm.copy()

    y[..., 0] *= XY_LIMIT
    y[..., 1] *= XY_LIMIT

    if y.shape[-1] >= 3:
        y[..., 2] *= R_RANGE[1]

    if y.shape[-1] >= 4:
        y[..., 3] *= C_RANGE[1]

    return y


# =========================================================
# SAVE / LOAD HELPERS
# =========================================================

def save_npz(name, **arrays):
    path = os.path.join(PATHS["data"], name + ".npz")
    np.savez(path, **arrays)
    print(f"Saved: {path}")


def load_npz(name):
    path = os.path.join(PATHS["data"], name + ".npz")
    return np.load(path)


# =========================================================
# FIGURE HELPER
# =========================================================

def savefig(fig, name, dpi=300):
    path = os.path.join(PATHS["figures"], name + ".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {path}")
