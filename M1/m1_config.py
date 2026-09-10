"""Revised M1 baseline parameters. Clean ablation of M0.

c_t = production/repair competence
v_t = verification/error-detection competence
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"

MASTER_SEED = 42
N_RUNS = 100
T = 1000

C0 = 0.35
V0 = 0.35

ALPHA0 = -0.50
ALPHA_C = 3.90
ALPHA_Z = 2.00

BETA0 = 1.05
BETA_Z = 1.20

COSTS = {"H": 0.260, "L": 0.020, "A": 0.055}
MODES = ("H", "L", "A")
POLICIES = ("H-only", "L-only", "A-only", "Myopic", "Dynamic")

# Same adaptation rate as M0, for every action.
RHO = 0.024
RHOS = {"H": 0.024, "L": 0.024, "A": 0.024}

# Production attractors: exactly M0.
ATTRACTORS_C = {
    "H": 1.00,
    "L": 0.00,
    "A": 0.45,
}

# Verification attractors: provisional M1 assumptions, not estimates.
ATTRACTORS_V = {
    "H": 0.00,
    "L": 0.00,
    "A": 0.45,
}

# J_{T+1}(c,v) = B * ((1-w)*c + w*v). w=0.5 is provisional equal weight.
TERMINAL_B = 20.0
TERMINAL_W = 0.50

# DP discretization (reported, not silently changed).
N_C_GRID = 51
N_V_GRID = 51
N_Z_GRID = 25
C_GRID = tuple(float(x) for x in np.linspace(0.0, 1.0, N_C_GRID))
V_GRID = tuple(float(x) for x in np.linspace(0.0, 1.0, N_V_GRID))
Z_GRID = tuple(float(x) for x in np.linspace(0.0, 1.0, N_Z_GRID))
INTERPOLATION = "bilinear on a uniform (c,v) grid; z quadrature is the uniform Z_GRID mean"
