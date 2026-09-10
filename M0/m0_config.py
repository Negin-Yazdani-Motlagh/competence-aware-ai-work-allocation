"""M0 baseline parameters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

MASTER_SEED = 42
N_RUNS = 100
T = 1000
C0 = 0.35

M_HUMAN = 1.0
M_ASSISTED = 0.45
M_LLM = 0.0
RHO = 0.024

ATTRACTORS = {
    "H": M_HUMAN,
    "L": M_LLM,
    "A": M_ASSISTED,
}

ALPHA0 = -0.5
ALPHA_C = 3.9
ALPHA_Z = 2.0
BETA0 = 1.05
BETA_Z = 1.2

COSTS = {"H": 0.26, "L": 0.02, "A": 0.055}
MODES = ("H", "L", "A")

POLICIES = ("H-only", "L-only", "A-only", "Myopic", "Dynamic")

DISCOUNT_GAMMA = 1.0
TERMINAL_BEQUEST = 20.0

N_C_GRID = 51
N_Z_GRID = 25
C_GRID = tuple(float(x) for x in np.linspace(0.0, 1.0, N_C_GRID))
Z_GRID = tuple(float(x) for x in np.linspace(0.0, 1.0, N_Z_GRID))
