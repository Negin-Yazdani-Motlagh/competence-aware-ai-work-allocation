#!/usr/bin/env python3
"""Run revised M1: clean ablation of M0 with split (c,v) competence."""

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

import m1_config as cfg
from m1 import (
    build_value_function_finite,
    run_seeds,
    run_trajectory,
    task_sequence,
)
from test_m1 import run_all as run_unit_tests


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
    shares = defaultdict(lambda: {"H": 0, "L": 0, "A": 0, "n": 0})
    traj_c = defaultdict(lambda: defaultdict(list))
    traj_v = defaultdict(lambda: defaultdict(list))
    horizon = max(r["t"] for r in rows)
    for r in rows:
        shares[r["policy"]][r["action"]] += 1
        shares[r["policy"]]["n"] += 1
        traj_c[r["policy"]][r["t"]].append(r["c_t"])
        traj_v[r["policy"]][r["t"]].append(r["v_t"])
        if r["t"] == horizon:
            last[r["policy"]].append(r)

    policies = sorted({r["policy"] for r in rows}, key=lambda p: (cfg.POLICIES.index(p) if p in cfg.POLICIES else 99, p))
    out = {}
    for policy in policies:
        items = last[policy]
        n_share = shares[policy]["n"]
        out[policy] = {
            "c_T": mean_sd_ci([r["c_next"] for r in items]),
            "v_T": mean_sd_ci([r["v_next"] for r in items]),
            "cumulative_u": mean_sd_ci([r["cumulative_utility"] for r in items]),
            "action_share": {
                a: shares[policy][a] / n_share for a in ("H", "L", "A")
            },
        }
    mean_traj = {}
    for policy in policies:
        mean_traj[policy] = {
            "t": list(range(1, horizon + 1)),
            "c": [float(np.mean(traj_c[policy][t])) for t in range(1, horizon + 1)],
            "v": [float(np.mean(traj_v[policy][t])) for t in range(1, horizon + 1)],
        }
    return out, mean_traj


def print_table(summary: dict, title: str) -> None:
    print(f"\n{title}")
    print(f"{'Policy':<10} {'Mean c_T':>10} {'Mean v_T':>10} {'Mean sum u':>12} {'Share H':>9} {'Share L':>9} {'Share A':>9}")
    for policy in summary:
        s = summary[policy]
        print(
            f"{policy:<10} {s['c_T']['mean']:10.4f} {s['v_T']['mean']:10.4f} "
            f"{s['cumulative_u']['mean']:12.4f} "
            f"{s['action_share']['H']:9.4f} {s['action_share']['L']:9.4f} {s['action_share']['A']:9.4f}"
        )


def make_figures(mean_traj: dict) -> list[str]:
    import matplotlib.pyplot as plt

    cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {
        "H-only": "#1565C0",
        "L-only": "#C62828",
        "A-only": "#2E7D32",
        "Myopic": "#7B1FA2",
        "Dynamic": "#004D40",
    }
    paths = []
    for key, ylabel, fname in (
        ("c", "Production competence $c_t$", "m1_c_trajectory.png"),
        ("v", "Verification competence $v_t$", "m1_v_trajectory.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
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
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        path = cfg.FIGURE_DIR / fname
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def simulate_policies(
    z_seqs: list[np.ndarray],
    seeds: list[int],
    policies: tuple[str, ...],
    *,
    J: np.ndarray | None = None,
    c0: float | None = None,
    v0: float | None = None,
    rho: float | None = None,
    attractors_c: dict[str, float] | None = None,
    attractors_v: dict[str, float] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for i, (seed, z_seq) in enumerate(zip(seeds, z_seqs), start=1):
        for policy in policies:
            rows.extend(
                run_trajectory(
                    policy,
                    z_seq,
                    seed,
                    J=J if policy == "Dynamic" else None,
                    c0=c0,
                    v0=v0,
                    rho=rho,
                    attractors_c=attractors_c,
                    attractors_v=attractors_v,
                )
            )
        if i % 10 == 0 or i == 1:
            print(f"  run {i}/{len(seeds)} seed={seed}")
    return rows


def compare_recovery_to_m0(
    m1_rows: list[dict],
    z_seqs: list[np.ndarray],
    seeds: list[int],
) -> dict:
    m0_dir = str(HERE.parent / "M0")
    if m0_dir not in sys.path:
        sys.path.insert(0, m0_dir)
    import m0 as m0_mod  # noqa: WPS433

    print("Building M0 value function for recovery comparison...")
    V = m0_mod.build_value_function_finite()
    m0_rows: list[dict] = []
    for seed, z_seq in zip(seeds, z_seqs):
        for policy in cfg.POLICIES:
            m0_rows.extend(
                m0_mod.run_trajectory(policy, z_seq, seed, V=V if policy == "Dynamic" else None)
            )

    m1_by = {(r["seed"], r["policy"], r["t"]): r for r in m1_rows}
    out: dict = {"by_policy": {}, "overall": {}}
    max_c = 0.0
    max_u = 0.0
    agree_all = 0
    total_all = 0
    for policy in cfg.POLICIES:
        agree = 0
        total = 0
        max_c_p = 0.0
        max_u_p = 0.0
        for r0 in m0_rows:
            if r0["policy"] != policy:
                continue
            r1 = m1_by[(r0["seed"], r0["policy"], r0["t"])]
            max_c_p = max(max_c_p, abs(r1["c_t"] - r0["c_t"]))
            max_u_p = max(max_u_p, abs(r1["utility"] - r0["utility"]))
            agree += int(r1["action"] == r0["action"])
            total += 1
        out["by_policy"][policy] = {
            "action_agreement": agree / total,
            "max_abs_c": max_c_p,
            "max_abs_u": max_u_p,
            "n_steps": total,
        }
        max_c = max(max_c, max_c_p)
        max_u = max(max_u, max_u_p)
        agree_all += agree
        total_all += total
    out["overall"] = {
        "action_agreement": agree_all / total_all,
        "max_abs_c": max_c,
        "max_abs_u": max_u,
        "n_steps": total_all,
    }
    return out


def compare_m0_baseline(m1_summary: dict) -> dict:
    m0_path = HERE.parent / "M0" / "outputs" / "m0_summary.json"
    if not m0_path.exists():
        return {"available": False}
    m0 = json.loads(m0_path.read_text())
    out = {"available": True}
    for policy in cfg.POLICIES:
        m1s = m1_summary[policy]
        m0s = m0["summary"][policy]
        out[policy] = {
            "m0_c_T": m0s["mean_final_c"],
            "m1_c_T": m1s["c_T"]["mean"],
            "m1_v_T": m1s["v_T"]["mean"],
            "m0_U": m0s["mean_cumulative_u"],
            "m1_U": m1s["cumulative_u"]["mean"],
            "m0_share": m0s.get("action_share"),
            "m1_share": m1s["action_share"],
        }
    return out


def run_sensitivity(z_seqs: list[np.ndarray], seeds: list[int], baseline_J: np.ndarray) -> list[dict]:
    rows_out: list[dict] = []

    def record(check: str, param: str, value: float, summary: dict) -> None:
        s = summary["Dynamic"]
        rec = {
            "check": check,
            "param": param,
            "value": value,
            "share_H": s["action_share"]["H"],
            "share_L": s["action_share"]["L"],
            "share_A": s["action_share"]["A"],
            "mean_c_T": s["c_T"]["mean"],
            "mean_v_T": s["v_T"]["mean"],
            "mean_cumulative_u": s["cumulative_u"]["mean"],
        }
        rows_out.append(rec)
        print(
            f"  {check} {param}={value}: H={rec['share_H']:.4f} L={rec['share_L']:.4f} "
            f"A={rec['share_A']:.4f} c_T={rec['mean_c_T']:.4f} v_T={rec['mean_v_T']:.4f} "
            f"U={rec['mean_cumulative_u']:.4f}"
        )

    print("\nSensitivity A: m_A^v in {0.25, 0.45, 0.65}")
    for m_av in (0.25, 0.45, 0.65):
        attr_v = dict(cfg.ATTRACTORS_V)
        attr_v["A"] = m_av
        if abs(m_av - cfg.ATTRACTORS_V["A"]) < 1e-15:
            J = baseline_J
        else:
            print(f"  building J for m_A^v={m_av}...")
            J = build_value_function_finite(attractors_v=attr_v)
        rows = simulate_policies(
            z_seqs, seeds, ("Dynamic",), J=J, attractors_v=attr_v,
        )
        summary, _ = summarize(rows)
        record("A_verification_attractor", "m_A^v", m_av, summary)

    print("\nSensitivity B: w in {0.25, 0.50, 0.75}, B=20")
    for w in (0.25, 0.50, 0.75):
        if abs(w - cfg.TERMINAL_W) < 1e-15:
            J = baseline_J
        else:
            print(f"  building J for w={w}...")
            J = build_value_function_finite(w=w, B=20.0)
        rows = simulate_policies(z_seqs, seeds, ("Dynamic",), J=J)
        summary, _ = summarize(rows)
        record("B_terminal_weight", "w", w, summary)

    return rows_out


def main() -> None:
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Revised M1 unit / validation tests ===")
    test_results = run_unit_tests()

    print(f"\nBuilding revised M1 DP J_t(c,v)  grid=({cfg.N_C_GRID} x {cfg.N_V_GRID}), z={cfg.N_Z_GRID}...")
    J = build_value_function_finite()
    print("DP done.")

    seeds = run_seeds()
    z_seqs = [task_sequence(seed) for seed in seeds]

    print("\n=== Baseline revised M1 ===")
    baseline_rows = simulate_policies(z_seqs, seeds, cfg.POLICIES, J=J)
    summary, mean_traj = summarize(baseline_rows)
    print_table(summary, "Revised M1 baseline")

    fig_paths = make_figures(mean_traj)
    csv_path = cfg.OUTPUT_DIR / "m1_runs.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
        w.writeheader()
        w.writerows(baseline_rows)

    print("\n=== Test 4 full M0 recovery on the same task sequences ===")
    recovery_attr_v = dict(cfg.ATTRACTORS_C)
    print("Building recovery DP with m^v = m^c ...")
    J_rec = build_value_function_finite(attractors_v=recovery_attr_v)
    recovery_rows = simulate_policies(
        z_seqs,
        seeds,
        cfg.POLICIES,
        J=J_rec,
        c0=cfg.C0,
        v0=cfg.C0,
        attractors_v=recovery_attr_v,
    )
    recovery_summary, _ = summarize(recovery_rows)
    print_table(recovery_summary, "M1 with v=c attractors (M0 recovery world)")
    max_cv = max(abs(r["c_t"] - r["v_t"]) for r in recovery_rows)
    recovery_vs_m0 = compare_recovery_to_m0(recovery_rows, z_seqs, seeds)
    recovery_vs_m0["max_abs_c_minus_v"] = max_cv
    print(f"  max |c_t - v_t| = {max_cv:.3e}")
    for policy, d in recovery_vs_m0["by_policy"].items():
        print(
            f"  {policy:<10} agree={d['action_agreement']:.6f} "
            f"max|dc|={d['max_abs_c']:.3e} max|du|={d['max_abs_u']:.3e}"
        )

    print("\n=== Sensitivity (does not replace baseline) ===")
    sensitivity = run_sensitivity(z_seqs, seeds, J)
    sens_path = cfg.OUTPUT_DIR / "m1_sensitivity.csv"
    with sens_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sensitivity[0].keys()))
        w.writeheader()
        w.writerows(sensitivity)

    comparison = compare_m0_baseline(summary)
    report = {
        "model": "M1_revised",
        "note": "Supersedes the previous M1 run (sigmoid d, lambda/delta/exposure transitions).",
        "T": cfg.T,
        "n_runs": cfg.N_RUNS,
        "seed": cfg.MASTER_SEED,
        "c0": cfg.C0,
        "v0": cfg.V0,
        "rho": cfg.RHO,
        "attractors_c": dict(cfg.ATTRACTORS_C),
        "attractors_v": dict(cfg.ATTRACTORS_V),
        "terminal": {
            "B": cfg.TERMINAL_B,
            "w": cfg.TERMINAL_W,
            "formula": "B*((1-w)*c + w*v)",
            "note": "provisional equal weights when w=0.5",
        },
        "dp": {
            "n_c_grid": cfg.N_C_GRID,
            "n_v_grid": cfg.N_V_GRID,
            "n_z_grid": cfg.N_Z_GRID,
            "interpolation": cfg.INTERPOLATION,
        },
        "summary": summary,
        "validation_unit_tests": {
            k: v if isinstance(v, dict) else v for k, v in test_results.items()
        },
        "m0_recovery": {
            "summary": recovery_summary,
            "vs_m0": recovery_vs_m0,
        },
        "m0_baseline_comparison": comparison,
        "sensitivity": sensitivity,
        "csv": str(csv_path),
        "sensitivity_csv": str(sens_path),
        "figures": fig_paths,
    }
    json_path = cfg.OUTPUT_DIR / "m1_summary.json"
    with json_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {sens_path}")
    for p in fig_paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
