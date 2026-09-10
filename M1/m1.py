"""Revised M1: dual-competence ablation of M0. Actions remain {H, L, A}.

Assisted quality uses v_t directly as detection capability:
    p_A = p_L + (1 - p_L) * v * p_H

Competence follows M0 attractor updates, separately for c and v.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import m1_config as cfg

_J_CACHE: dict[tuple, np.ndarray] = {}


def sigmoid(x: float | np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def p_H(c: float | np.ndarray, z: float | np.ndarray) -> np.ndarray:
    return sigmoid(cfg.ALPHA0 + cfg.ALPHA_C * np.asarray(c) - cfg.ALPHA_Z * np.asarray(z))


def p_L(z: float | np.ndarray) -> np.ndarray:
    return sigmoid(cfg.BETA0 - cfg.BETA_Z * np.asarray(z))


def p_A(c: float | np.ndarray, v: float | np.ndarray, z: float | np.ndarray) -> np.ndarray:
    pl = p_L(z)
    return pl + (1.0 - pl) * np.asarray(v) * p_H(c, z)


def quality(
    mode: str,
    c: float | np.ndarray,
    v: float | np.ndarray,
    z: float | np.ndarray,
) -> np.ndarray:
    if mode == "H":
        return p_H(c, z)
    if mode == "L":
        return p_L(z)
    if mode == "A":
        return p_A(c, v, z)
    raise ValueError(mode)


def utility(
    mode: str,
    c: float | np.ndarray,
    v: float | np.ndarray,
    z: float | np.ndarray,
) -> np.ndarray:
    return quality(mode, c, v, z) - cfg.COSTS[mode]


def _rho(mode: str, rho: float | None) -> float:
    if rho is not None:
        return float(rho)
    return float(cfg.RHOS[mode])


def next_c(
    c: float | np.ndarray,
    mode: str,
    *,
    rho: float | None = None,
    attractors_c: dict[str, float] | None = None,
) -> np.ndarray:
    attr = cfg.ATTRACTORS_C if attractors_c is None else attractors_c
    r = _rho(mode, rho)
    c = np.asarray(c, dtype=float)
    return np.clip(c + r * (attr[mode] - c), 0.0, 1.0)


def next_v(
    v: float | np.ndarray,
    mode: str,
    *,
    rho: float | None = None,
    attractors_v: dict[str, float] | None = None,
) -> np.ndarray:
    attr = cfg.ATTRACTORS_V if attractors_v is None else attractors_v
    r = _rho(mode, rho)
    v = np.asarray(v, dtype=float)
    return np.clip(v + r * (attr[mode] - v), 0.0, 1.0)


def terminal_value(
    c: float | np.ndarray,
    v: float | np.ndarray,
    *,
    B: float | None = None,
    w: float | None = None,
) -> np.ndarray:
    B = cfg.TERMINAL_B if B is None else B
    w = cfg.TERMINAL_W if w is None else w
    return B * ((1.0 - w) * np.asarray(c) + w * np.asarray(v))


def _bilinear(grid: np.ndarray, c_vals: np.ndarray, v_vals: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a 2D value table on the uniform (c,v) grid."""
    c_grid = np.asarray(cfg.C_GRID)
    v_grid = np.asarray(cfg.V_GRID)
    c_vals = np.clip(np.asarray(c_vals, dtype=float), 0.0, 1.0)
    v_vals = np.clip(np.asarray(v_vals, dtype=float), 0.0, 1.0)
    dc = c_grid[1] - c_grid[0]
    dv = v_grid[1] - v_grid[0]
    ic = np.clip((c_vals - c_grid[0]) / dc, 0.0, len(c_grid) - 1 - 1e-12)
    iv = np.clip((v_vals - v_grid[0]) / dv, 0.0, len(v_grid) - 1 - 1e-12)
    i0 = np.floor(ic).astype(int)
    j0 = np.floor(iv).astype(int)
    i1 = np.minimum(i0 + 1, len(c_grid) - 1)
    j1 = np.minimum(j0 + 1, len(v_grid) - 1)
    wc = ic - i0
    wv = iv - j0
    g00 = grid[i0, j0]
    g10 = grid[i1, j0]
    g01 = grid[i0, j1]
    g11 = grid[i1, j1]
    return (1.0 - wc) * (1.0 - wv) * g00 + wc * (1.0 - wv) * g10 + (1.0 - wc) * wv * g01 + wc * wv * g11


def _attr_key(attr: dict[str, float]) -> tuple:
    return tuple((m, float(attr[m])) for m in cfg.MODES)


def build_value_function_finite(
    *,
    rho: float | None = None,
    attractors_c: dict[str, float] | None = None,
    attractors_v: dict[str, float] | None = None,
    B: float | None = None,
    w: float | None = None,
    horizon: int | None = None,
) -> np.ndarray:
    attr_c = cfg.ATTRACTORS_C if attractors_c is None else attractors_c
    attr_v = cfg.ATTRACTORS_V if attractors_v is None else attractors_v
    B = cfg.TERMINAL_B if B is None else B
    w = cfg.TERMINAL_W if w is None else w
    T = cfg.T if horizon is None else horizon
    r_key = cfg.RHO if rho is None else rho
    key = (
        T,
        r_key,
        _attr_key(attr_c),
        _attr_key(attr_v),
        B,
        w,
        cfg.N_C_GRID,
        cfg.N_V_GRID,
        cfg.N_Z_GRID,
    )
    if key in _J_CACHE:
        return _J_CACHE[key]

    c_grid = np.asarray(cfg.C_GRID)
    v_grid = np.asarray(cfg.V_GRID)
    z_grid = np.asarray(cfg.Z_GRID)
    C, V = np.meshgrid(c_grid, v_grid, indexing="ij")
    J = np.zeros((T + 2, len(c_grid), len(v_grid)))
    J[T + 1] = terminal_value(C, V, B=B, w=w)

    next_states = {
        mode: (
            next_c(C, mode, rho=rho, attractors_c=attr_c),
            next_v(V, mode, rho=rho, attractors_v=attr_v),
        )
        for mode in cfg.MODES
    }

    for t in range(T, 0, -1):
        j_next = J[t + 1]
        cont = {mode: _bilinear(j_next, nc, nv) for mode, (nc, nv) in next_states.items()}
        best_sum = np.zeros_like(C)
        for z in z_grid:
            best = np.full_like(C, -1e18)
            for mode in cfg.MODES:
                q = utility(mode, C, V, z) + cont[mode]
                best = np.maximum(best, q)
            best_sum += best
        J[t] = best_sum / len(z_grid)

    _J_CACHE[key] = J
    return J


def choose_myopic(c: float, v: float, z: float) -> str:
    return max(cfg.MODES, key=lambda m: float(utility(m, c, v, z)))


def choose_dynamic(
    c: float,
    v: float,
    z: float,
    t: int,
    J: np.ndarray,
    *,
    rho: float | None = None,
    attractors_c: dict[str, float] | None = None,
    attractors_v: dict[str, float] | None = None,
) -> str:
    j_next = J[t + 1]
    best_m, best_q = cfg.MODES[0], -1e18
    for mode in cfg.MODES:
        cn = float(next_c(c, mode, rho=rho, attractors_c=attractors_c))
        vn = float(next_v(v, mode, rho=rho, attractors_v=attractors_v))
        cont = float(_bilinear(j_next, cn, vn))
        q = float(utility(mode, c, v, z)) + cont
        if q > best_q:
            best_q, best_m = q, mode
    return best_m


def choose_action(
    policy: str,
    c: float,
    v: float,
    z: float,
    t: int,
    J: np.ndarray | None,
    *,
    rho: float | None = None,
    attractors_c: dict[str, float] | None = None,
    attractors_v: dict[str, float] | None = None,
) -> str:
    if policy == "H-only":
        return "H"
    if policy == "L-only":
        return "L"
    if policy == "A-only":
        return "A"
    if policy == "Myopic":
        return choose_myopic(c, v, z)
    if policy == "Dynamic":
        if J is None:
            raise ValueError("Dynamic policy requires J")
        return choose_dynamic(
            c, v, z, t, J,
            rho=rho, attractors_c=attractors_c, attractors_v=attractors_v,
        )
    raise ValueError(policy)


def task_sequence(seed: int, horizon: int | None = None) -> np.ndarray:
    t = cfg.T if horizon is None else horizon
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=t)


def run_seeds(n_runs: int | None = None) -> list[int]:
    n = cfg.N_RUNS if n_runs is None else n_runs
    rng = np.random.default_rng(cfg.MASTER_SEED)
    return [int(x) for x in rng.choice(2**31 - 1, size=n, replace=False)]


def run_trajectory(
    policy: str,
    z_seq: np.ndarray,
    seed: int,
    *,
    J: np.ndarray | None = None,
    c0: float | None = None,
    v0: float | None = None,
    rho: float | None = None,
    attractors_c: dict[str, float] | None = None,
    attractors_v: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    c = cfg.C0 if c0 is None else c0
    v = cfg.V0 if v0 is None else v0
    total_u = 0.0
    for t_idx, z in enumerate(z_seq):
        t = t_idx + 1
        z = float(z)
        action = choose_action(
            policy, c, v, z, t, J,
            rho=rho, attractors_c=attractors_c, attractors_v=attractors_v,
        )
        p = float(quality(action, c, v, z))
        u = float(utility(action, c, v, z))
        c_n = float(next_c(c, action, rho=rho, attractors_c=attractors_c))
        v_n = float(next_v(v, action, rho=rho, attractors_v=attractors_v))
        total_u += u
        rows.append(
            {
                "seed": seed,
                "t": t,
                "policy": policy,
                "z_t": z,
                "action": action,
                "c_t": c,
                "v_t": v,
                "p_correct": p,
                "utility": u,
                "cumulative_utility": total_u,
                "c_next": c_n,
                "v_next": v_n,
            }
        )
        c, v = c_n, v_n
    return rows
