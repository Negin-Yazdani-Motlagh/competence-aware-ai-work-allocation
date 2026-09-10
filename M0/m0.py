"""M0: original HCOMP one-competence, three-action model."""

from __future__ import annotations

from typing import Any

import numpy as np

import m0_config as cfg

_V_CACHE: dict[tuple, np.ndarray] = {}


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def p_H(c: float, z: float) -> float:
    return float(sigmoid(cfg.ALPHA0 + cfg.ALPHA_C * c - cfg.ALPHA_Z * z))


def p_L(z: float) -> float:
    return float(sigmoid(cfg.BETA0 - cfg.BETA_Z * z))


def p_A(c: float, z: float) -> float:
    pl = p_L(z)
    return float(pl + (1.0 - pl) * c * p_H(c, z))


def quality(mode: str, c: float, z: float) -> float:
    if mode == "H":
        return p_H(c, z)
    if mode == "L":
        return p_L(z)
    if mode == "A":
        return p_A(c, z)
    raise ValueError(mode)


def utility(mode: str, c: float, z: float) -> float:
    return quality(mode, c, z) - cfg.COSTS[mode]


def competence_update(c: float, mode: str, *, rho: float | None = None) -> float:
    r = cfg.RHO if rho is None else rho
    return float(np.clip(c + r * (cfg.ATTRACTORS[mode] - c), 0.0, 1.0))


def choose_myopic(c: float, z: float) -> str:
    return max(cfg.MODES, key=lambda m: utility(m, c, z))


def _value_next(V: np.ndarray, t: int) -> np.ndarray:
    return V[t + 1] if V.ndim == 2 else V


def choose_dynamic(c: float, z: float, t: int, V: np.ndarray) -> str:
    c_grid = np.asarray(cfg.C_GRID)
    v_next = _value_next(V, t)
    best_m, best_q = cfg.MODES[0], -1e18
    for mode in cfg.MODES:
        c_next = competence_update(c, mode)
        vn = float(np.interp(c_next, c_grid, v_next))
        q = utility(mode, c, z) + cfg.DISCOUNT_GAMMA * vn
        if q > best_q:
            best_q, best_m = q, mode
    return best_m


def choose_action(policy: str, c: float, z: float, t: int, V: np.ndarray | None) -> str:
    if policy == "H-only":
        return "H"
    if policy == "L-only":
        return "L"
    if policy == "A-only":
        return "A"
    if policy == "Myopic":
        return choose_myopic(c, z)
    if policy == "Dynamic":
        if V is None:
            raise ValueError("Dynamic policy requires a value function")
        return choose_dynamic(c, z, t, V)
    raise ValueError(policy)


def build_value_function_finite(*, rho: float | None = None) -> np.ndarray:
    r = cfg.RHO if rho is None else rho
    key = (cfg.T, r, cfg.TERMINAL_BEQUEST, cfg.N_C_GRID, cfg.N_Z_GRID)
    if key in _V_CACHE:
        return _V_CACHE[key]

    c_grid = np.asarray(cfg.C_GRID)
    z_grid = np.asarray(cfg.Z_GRID)
    n_c = len(c_grid)
    V = np.zeros((cfg.T + 2, n_c))
    V[cfg.T + 1, :] = cfg.TERMINAL_BEQUEST * c_grid

    next_c = {
        mode: np.clip(c_grid + r * (cfg.ATTRACTORS[mode] - c_grid), 0.0, 1.0)
        for mode in cfg.MODES
    }

    for t in range(cfg.T, 0, -1):
        v_next = V[t + 1]
        vn = {mode: np.interp(next_c[mode], c_grid, v_next) for mode in cfg.MODES}
        best_over_z = np.zeros(n_c)
        for z in z_grid:
            best = np.full(n_c, -1e18)
            for mode in cfg.MODES:
                q = np.array([utility(mode, float(c), float(z)) for c in c_grid])
                best = np.maximum(best, q + cfg.DISCOUNT_GAMMA * vn[mode])
            best_over_z += best
        V[t, :] = best_over_z / len(z_grid)

    _V_CACHE[key] = V
    return V


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
    V: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    c = cfg.C0
    total_u = 0.0
    for t_idx, z in enumerate(z_seq):
        t = t_idx + 1
        z = float(z)
        action = choose_action(policy, c, z, t, V)
        p = quality(action, c, z)
        u = utility(action, c, z)
        c_next = competence_update(c, action)
        total_u += u
        rows.append(
            {
                "seed": seed,
                "t": t,
                "policy": policy,
                "z_t": z,
                "action": action,
                "c_t": c,
                "p_correct": p,
                "utility": u,
                "cumulative_utility": total_u,
                "c_next": c_next,
            }
        )
        c = c_next
    return rows
