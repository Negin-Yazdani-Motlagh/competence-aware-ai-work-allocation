"""M2: four workflows with dual competence (c, v).

Workflows: H human-only, L AI delegation, A co-production, V structured verification.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import m2_config as cfg

_J_CACHE: dict[tuple, np.ndarray] = {}


def sigmoid(x: float | np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def h(c: float | np.ndarray, z: float | np.ndarray) -> np.ndarray:
    return sigmoid(cfg.ALPHA0 + cfg.ALPHA_C * np.asarray(c) - cfg.ALPHA_Z * np.asarray(z))


def d(v: float | np.ndarray, z: float | np.ndarray, phi: float | np.ndarray) -> np.ndarray:
    return np.asarray(phi, dtype=float) * sigmoid(
        cfg.GAMMA0 + cfg.GAMMA_V * np.asarray(v) - cfg.GAMMA_Z * np.asarray(z)
    )


def q_H(c: float | np.ndarray, z: float | np.ndarray) -> np.ndarray:
    return h(c, z)


def q_L(l: float | np.ndarray) -> np.ndarray:
    return np.asarray(l, dtype=float)


def q_A(
    c: float | np.ndarray,
    z: float | np.ndarray,
    l: float | np.ndarray,
    *,
    omega: float | None = None,
) -> np.ndarray:
    w = cfg.OMEGA if omega is None else omega
    return w * h(c, z) + (1.0 - w) * np.asarray(l, dtype=float)


def q_V(
    c: float | np.ndarray,
    v: float | np.ndarray,
    z: float | np.ndarray,
    l: float | np.ndarray,
    phi: float | np.ndarray,
) -> np.ndarray:
    ll = np.asarray(l, dtype=float)
    return ll + (1.0 - ll) * d(v, z, phi) * h(c, z)


def quality(
    mode: str,
    c: float | np.ndarray,
    v: float | np.ndarray,
    z: float | np.ndarray,
    l: float | np.ndarray,
    phi: float | np.ndarray,
    *,
    omega: float | None = None,
) -> np.ndarray:
    if mode == "H":
        return q_H(c, z)
    if mode == "L":
        return q_L(l)
    if mode == "A":
        return q_A(c, z, l, omega=omega)
    if mode == "V":
        return q_V(c, v, z, l, phi)
    raise ValueError(mode)


def utility(
    mode: str,
    c: float | np.ndarray,
    v: float | np.ndarray,
    z: float | np.ndarray,
    r: float | np.ndarray,
    l: float | np.ndarray,
    phi: float | np.ndarray,
    *,
    omega: float | None = None,
) -> np.ndarray:
    q = quality(mode, c, v, z, l, phi, omega=omega)
    return q - np.asarray(r, dtype=float) * (1.0 - q) - cfg.COSTS[mode]


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


def task_quadrature() -> np.ndarray:
    """81-point tensor product of linspace(0,1,3) on (z, r, l, phi)."""
    vals = np.asarray(cfg.TASK_1D, dtype=float)
    Z, R, L, PHI = np.meshgrid(vals, vals, vals, vals, indexing="ij")
    return np.stack([Z.ravel(), R.ravel(), L.ravel(), PHI.ravel()], axis=1)


def _bilinear(grid: np.ndarray, c_vals: np.ndarray, v_vals: np.ndarray) -> np.ndarray:
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
    omega: float | None = None,
    horizon: int | None = None,
) -> np.ndarray:
    attr_c = cfg.ATTRACTORS_C if attractors_c is None else attractors_c
    attr_v = cfg.ATTRACTORS_V if attractors_v is None else attractors_v
    B = cfg.TERMINAL_B if B is None else B
    w = cfg.TERMINAL_W if w is None else w
    omega = cfg.OMEGA if omega is None else omega
    T = cfg.T if horizon is None else horizon
    r_key = cfg.RHO if rho is None else rho
    key = (
        T,
        r_key,
        _attr_key(attr_c),
        _attr_key(attr_v),
        B,
        w,
        omega,
        cfg.N_C_GRID,
        cfg.N_V_GRID,
        cfg.N_TASK_1D,
    )
    if key in _J_CACHE:
        return _J_CACHE[key]

    c_grid = np.asarray(cfg.C_GRID)
    v_grid = np.asarray(cfg.V_GRID)
    C, Vst = np.meshgrid(c_grid, v_grid, indexing="ij")
    J = np.zeros((T + 2, len(c_grid), len(v_grid)))
    J[T + 1] = terminal_value(C, Vst, B=B, w=w)
    nodes = task_quadrature()

    next_states = {
        mode: (
            next_c(C, mode, rho=rho, attractors_c=attr_c),
            next_v(Vst, mode, rho=rho, attractors_v=attr_v),
        )
        for mode in cfg.MODES
    }

    for t in range(T, 0, -1):
        j_next = J[t + 1]
        cont = {mode: _bilinear(j_next, nc, nv) for mode, (nc, nv) in next_states.items()}
        best_sum = np.zeros_like(C)
        for z, r, l, phi in nodes:
            best = np.full_like(C, -1e18)
            for mode in cfg.MODES:
                q = utility(mode, C, Vst, z, r, l, phi, omega=omega) + cont[mode]
                best = np.maximum(best, q)
            best_sum += best
        J[t] = best_sum / len(nodes)

    _J_CACHE[key] = J
    return J


def choose_myopic(
    c: float,
    v: float,
    z: float,
    r: float,
    l: float,
    phi: float,
    *,
    omega: float | None = None,
) -> str:
    return max(
        cfg.MODES,
        key=lambda m: float(utility(m, c, v, z, r, l, phi, omega=omega)),
    )


def choose_dynamic(
    c: float,
    v: float,
    z: float,
    r: float,
    l: float,
    phi: float,
    t: int,
    J: np.ndarray,
    *,
    rho: float | None = None,
    attractors_c: dict[str, float] | None = None,
    attractors_v: dict[str, float] | None = None,
    omega: float | None = None,
) -> str:
    j_next = J[t + 1]
    best_m, best_q = cfg.MODES[0], -1e18
    for mode in cfg.MODES:
        cn = float(next_c(c, mode, rho=rho, attractors_c=attractors_c))
        vn = float(next_v(v, mode, rho=rho, attractors_v=attractors_v))
        cont = float(_bilinear(j_next, cn, vn))
        q = float(utility(mode, c, v, z, r, l, phi, omega=omega)) + cont
        if q > best_q:
            best_q, best_m = q, mode
    return best_m


def choose_action(
    policy: str,
    c: float,
    v: float,
    z: float,
    r: float,
    l: float,
    phi: float,
    t: int,
    J: np.ndarray | None,
    *,
    rho: float | None = None,
    attractors_c: dict[str, float] | None = None,
    attractors_v: dict[str, float] | None = None,
    omega: float | None = None,
) -> str:
    if policy == "H-only":
        return "H"
    if policy == "L-only":
        return "L"
    if policy == "A-only":
        return "A"
    if policy == "V-only":
        return "V"
    if policy == "Myopic":
        return choose_myopic(c, v, z, r, l, phi, omega=omega)
    if policy == "Dynamic":
        if J is None:
            raise ValueError("Dynamic policy requires J")
        return choose_dynamic(
            c, v, z, r, l, phi, t, J,
            rho=rho, attractors_c=attractors_c, attractors_v=attractors_v, omega=omega,
        )
    raise ValueError(policy)


def task_sequence(seed: int, horizon: int | None = None) -> dict[str, np.ndarray]:
    t = cfg.T if horizon is None else horizon
    rng = np.random.default_rng(seed)
    return {
        "z": rng.uniform(0.0, 1.0, size=t),
        "r": rng.uniform(0.0, 1.0, size=t),
        "l": rng.uniform(0.0, 1.0, size=t),
        "phi": rng.uniform(0.0, 1.0, size=t),
    }


def run_seeds(n_runs: int | None = None) -> list[int]:
    n = cfg.N_RUNS if n_runs is None else n_runs
    rng = np.random.default_rng(cfg.MASTER_SEED)
    return [int(x) for x in rng.choice(2**31 - 1, size=n, replace=False)]


def run_trajectory(
    policy: str,
    tasks: dict[str, np.ndarray],
    seed: int,
    *,
    J: np.ndarray | None = None,
    c0: float | None = None,
    v0: float | None = None,
    rho: float | None = None,
    attractors_c: dict[str, float] | None = None,
    attractors_v: dict[str, float] | None = None,
    omega: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    c = cfg.C0 if c0 is None else c0
    v = cfg.V0 if v0 is None else v0
    total_u = 0.0
    n = len(tasks["z"])
    for t_idx in range(n):
        t = t_idx + 1
        z = float(tasks["z"][t_idx])
        r = float(tasks["r"][t_idx])
        l = float(tasks["l"][t_idx])
        phi = float(tasks["phi"][t_idx])
        action = choose_action(
            policy, c, v, z, r, l, phi, t, J,
            rho=rho, attractors_c=attractors_c, attractors_v=attractors_v, omega=omega,
        )
        q = float(quality(action, c, v, z, l, phi, omega=omega))
        u = float(utility(action, c, v, z, r, l, phi, omega=omega))
        c_n = float(next_c(c, action, rho=rho, attractors_c=attractors_c))
        v_n = float(next_v(v, action, rho=rho, attractors_v=attractors_v))
        total_u += u
        rows.append(
            {
                "seed": seed,
                "t": t,
                "policy": policy,
                "z_t": z,
                "r_t": r,
                "l_t": l,
                "phi_t": phi,
                "action": action,
                "c_t": c,
                "v_t": v,
                "q": q,
                "utility": u,
                "cumulative_utility": total_u,
                "c_next": c_n,
                "v_next": v_n,
            }
        )
        c, v = c_n, v_n
    return rows
