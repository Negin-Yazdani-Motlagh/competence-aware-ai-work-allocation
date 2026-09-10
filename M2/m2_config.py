"""M2 baseline parameters. Four workflows; dual competence (c, v).

All NEW M2 parameters below are provisional modeling assumptions,
not empirical estimates.
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

# Human production/repair: same alpha as M0/M1.
ALPHA0 = -0.50
ALPHA_C = 3.90
ALPHA_Z = 2.00

# Verification detection sigmoid. Provisional; same shape as the human logit.
GAMMA0 = -0.50
GAMMA_V = 3.90
GAMMA_Z = 2.00

# Co-production weight in q_A = omega*h + (1-omega)*l. Provisional.
OMEGA = 0.50

# H/L/A costs match M1. kappa_V is a new provisional assumption.
COSTS = {"H": 0.260, "L": 0.020, "A": 0.055, "V": 0.100}
MODES = ("H", "L", "A", "V")
POLICIES = ("H-only", "L-only", "A-only", "V-only", "Myopic", "Dynamic")

RHO = 0.024
RHOS = {"H": 0.024, "L": 0.024, "A": 0.024, "V": 0.024}

# Production attractors. H/L inherited in spirit from M0/M1; A and V are new.
ATTRACTORS_C = {
    "H": 1.00,
    "L": 0.00,
    "A": 0.60,
    "V": 0.25,
}

# Verification attractors. All M2 values are provisional.
ATTRACTORS_V = {
    "H": 0.20,
    "L": 0.00,
    "A": 0.40,
    "V": 0.90,
}

TERMINAL_B = 20.0
TERMINAL_W = 0.50

N_C_GRID = 51
N_V_GRID = 51
C_GRID = tuple(float(x) for x in np.linspace(0.0, 1.0, N_C_GRID))
V_GRID = tuple(float(x) for x in np.linspace(0.0, 1.0, N_V_GRID))

# Task expectation: 3-point tensor product on (z, r, l, phi), 81 nodes.
# Endpoints 0 and 1 are included so the Bellman sees the structural limits.
N_TASK_1D = 3
TASK_1D = tuple(float(x) for x in np.linspace(0.0, 1.0, N_TASK_1D))
INTERPOLATION = (
    "bilinear on a uniform 51 x 51 (c,v) grid; "
    "E_{z,r,l,phi} is the mean over the 3^4=81 tensor product "
    "linspace(0,1,3)^4"
)
