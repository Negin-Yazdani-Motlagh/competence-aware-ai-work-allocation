#!/usr/bin/env python3
"""Run baseline M2: four workflows, dual competence."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np

import m2_config as cfg
from m2 import (
    build_value_function_finite,
    choose_dynamic,
    run_seeds,
    run_trajectory,
    task_sequence,
)
from test_m2 import run_all as run_unit_tests


def mean_sd_ci(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "n": n,
    }


def summarize(rows: list[dict]) -> tuple[dict, dict]:
    last = defaultdict(list)
    shares = defaultdict(lambda: {a: 0 for a in cfg.MODES} | {"n": 0})
    traj_c = defaultdict(lambda: defaultdict(list))
    traj_v = defaultdict(lambda: defaultdict(list))
    traj_act = defaultdict(lambda: defaultdict(lambda: {a: 0 for a in cfg.MODES}))
    horizon = max(r["t"] for r in rows)
    for r in rows:
        shares[r["policy"]][r["action"]] += 1
        shares[r["policy"]]["n"] += 1
        traj_c[r["policy"]][r["t"]].append(r["c_t"])
        traj_v[r["policy"]][r["t"]].append(r["v_t"])
        traj_act[r["policy"]][r["t"]][r["action"]] += 1
        if r["t"] == horizon:
            last[r["policy"]].append(r)

    policies = [p for p in cfg.POLICIES if p in last]
    out = {}
    for policy in policies:
        items = last[policy]
        n_share = shares[policy]["n"]
        out[policy] = {
            "c_T": mean_sd_ci([r["c_next"] for r in items]),
            "v_T": mean_sd_ci([r["v_next"] for r in items]),
            "cumulative_u": mean_sd_ci([r["cumulative_utility"] for r in items]),
            "action_share": {a: shares[policy][a] / n_share for a in cfg.MODES},
        }
    mean_traj = {}
    for policy in policies:
        n_t = {t: sum(traj_act[policy][t].values()) for t in range(1, horizon + 1)}
        mean_traj[policy] = {
            "t": list(range(1, horizon + 1)),
            "c": [float(np.mean(traj_c[policy][t])) for t in range(1, horizon + 1)],
            "v": [float(np.mean(traj_v[policy][t])) for t in range(1, horizon + 1)],
            "share": {
                a: [traj_act[policy][t][a] / n_t[t] for t in range(1, horizon + 1)]
                for a in cfg.MODES
            },
        }
    return out, mean_traj


def print_table(summary: dict, title: str) -> None:
    print(f"\n{title}")
    hdr = f"{'Policy':<10} {'Mean c_T':>10} {'Mean v_T':>10} {'Mean sum u':>12} {'H':>7} {'L':>7} {'A':>7} {'V':>7}"
    print(hdr)
    for policy in summary:
        s = summary[policy]
        print(
            f"{policy:<10} {s['c_T']['mean']:10.4f} {s['v_T']['mean']:10.4f} "
            f"{s['cumulative_u']['mean']:12.4f} "
            f"{s['action_share']['H']:7.3f} {s['action_share']['L']:7.3f} "
            f"{s['action_share']['A']:7.3f} {s['action_share']['V']:7.3f}"
        )


def make_figures(mean_traj: dict, J: np.ndarray) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {
        "H-only": "#1565C0",
        "L-only": "#C62828",
        "A-only": "#2E7D32",
        "V-only": "#EF6C00",
        "Myopic": "#7B1FA2",
        "Dynamic": "#004D40",
    }
    paths = []

    for key, ylabel, fname in (
        ("c", "Production competence $c_t$", "m2_c_trajectory.png"),
        ("v", "Verification competence $v_t$", "m2_v_trajectory.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.4, 4.3))
        for policy in cfg.POLICIES:
            ax.plot(
                mean_traj[policy]["t"],
                mean_traj[policy][key],
                label=policy,
                color=colors[policy],
                lw=2.0,
            )
        ax.set_xlabel("Task $t$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, cfg.T)
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=False, ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        path = cfg.FIGURE_DIR / fname
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    t = mean_traj["Dynamic"]["t"]
    shares = mean_traj["Dynamic"]["share"]
    mode_colors = {"H": "#1565C0", "L": "#C62828", "A": "#2E7D32", "V": "#EF6C00"}
    stack = np.vstack([shares[a] for a in cfg.MODES])
    ax.stackplot(t, stack, labels=list(cfg.MODES), colors=[mode_colors[a] for a in cfg.MODES], alpha=0.9)
    ax.set_xlabel("Task $t$")
    ax.set_ylabel("Dynamic action share")
    ax.set_xlim(1, cfg.T)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = cfg.FIGURE_DIR / "m2_dynamic_action_share.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    n = 41
    l_vals = np.linspace(0.0, 1.0, n)
    r_vals = np.linspace(0.0, 1.0, n)
    code = {m: i for i, m in enumerate(cfg.MODES)}
    grid = np.zeros((n, n), dtype=int)
    c_fix, v_fix, z_fix, phi_fix, t_fix = 0.5, 0.5, 0.5, 0.5, 1
    for i, r in enumerate(r_vals):
        for j, l in enumerate(l_vals):
            grid[i, j] = code[
                choose_dynamic(c_fix, v_fix, z_fix, float(r), float(l), phi_fix, t_fix, J)
            ]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    cmap = ListedColormap(["#1565C0", "#C62828", "#2E7D32", "#EF6C00"])
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[0.0, 1.0, 0.0, 1.0],
        vmin=0,
        vmax=3,
        cmap=cmap,
    )
    ax.set_xlabel("AI reliability $l$")
    ax.set_ylabel("Error consequence $r$")
    ax.set_title(r"Dynamic action at $c=v=z=\phi=0.5$, $t=1$")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(list(cfg.MODES))
    fig.tight_layout()
    path = cfg.FIGURE_DIR / "m2_policy_map.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def simulate_all(J: np.ndarray) -> tuple[list[dict], list[int], list[dict]]:
    seeds = run_seeds()
    task_list = [task_sequence(seed) for seed in seeds]
    rows: list[dict] = []
    for i, (seed, tasks) in enumerate(zip(seeds, task_list), start=1):
        for policy in cfg.POLICIES:
            rows.extend(
                run_trajectory(policy, tasks, seed, J=J if policy == "Dynamic" else None)
            )
        if i % 10 == 0 or i == 1:
            print(f"  run {i}/{len(seeds)} seed={seed}")
    return rows, seeds, task_list


def main() -> None:
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== M2 unit tests ===")
    test_results = run_unit_tests()

    print("\nBuilding M2 DP J_t(c,v) ...")
    print(f"  grid {cfg.N_C_GRID} x {cfg.N_V_GRID}; task quadrature 3^4=81")
    J = build_value_function_finite()
    print("DP done.")

    print("\n=== Baseline M2 ===")
    rows, seeds, _ = simulate_all(J)
    summary, mean_traj = summarize(rows)
    print_table(summary, "M2 baseline")

    fig_paths = make_figures(mean_traj, J)
    csv_path = cfg.OUTPUT_DIR / "m2_runs.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    never = [a for a in cfg.MODES if summary["Dynamic"]["action_share"][a] < 1e-12]
    report = {
        "model": "M2",
        "note": (
            "First M2 baseline. New parameters (omega, kappa_V, gamma, "
            "A/V attractors, m_H^v) are provisional modeling assumptions."
        ),
        "T": cfg.T,
        "n_runs": cfg.N_RUNS,
        "seed": cfg.MASTER_SEED,
        "c0": cfg.C0,
        "v0": cfg.V0,
        "omega": cfg.OMEGA,
        "rho": cfg.RHO,
        "costs": dict(cfg.COSTS),
        "attractors_c": dict(cfg.ATTRACTORS_C),
        "attractors_v": dict(cfg.ATTRACTORS_V),
        "alpha": {"alpha0": cfg.ALPHA0, "alpha_c": cfg.ALPHA_C, "alpha_z": cfg.ALPHA_Z},
        "gamma": {"gamma0": cfg.GAMMA0, "gamma_v": cfg.GAMMA_V, "gamma_z": cfg.GAMMA_Z},
        "terminal": {"B": cfg.TERMINAL_B, "w": cfg.TERMINAL_W},
        "dp": {
            "n_c_grid": cfg.N_C_GRID,
            "n_v_grid": cfg.N_V_GRID,
            "task_quadrature": "tensor product linspace(0,1,3)^4 = 81 nodes on (z,r,l,phi)",
            "interpolation": cfg.INTERPOLATION,
        },
        "summary": summary,
        "validation_unit_tests": {
            k: v if isinstance(v, dict) else v for k, v in test_results.items()
        },
        "dynamic_unused_workflows": never,
        "csv": str(csv_path),
        "figures": fig_paths,
        "n_seeds": len(seeds),
    }
    json_path = cfg.OUTPUT_DIR / "m2_summary.json"
    with json_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    for p in fig_paths:
        print(f"Wrote {p}")
    if never:
        print(f"NOTE: Dynamic never chose {never} in the baseline.")


if __name__ == "__main__":
    main()
