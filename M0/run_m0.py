#!/usr/bin/env python3
"""Run M0: formula check, then H/L/A/myopic/dynamic trajectories."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import m0_config as cfg
from m0 import (
    build_value_function_finite,
    competence_update,
    p_A,
    p_H,
    p_L,
    run_seeds,
    run_trajectory,
    task_sequence,
    utility,
)


def formula_check() -> dict:
    c, z = cfg.C0, 0.5
    return {
        "c0": c,
        "z": z,
        "p_H": p_H(c, z),
        "p_L": p_L(z),
        "p_A": p_A(c, z),
        "u_H": utility("H", c, z),
        "u_L": utility("L", c, z),
        "u_A": utility("A", c, z),
        "c_next_H": competence_update(c, "H"),
        "c_next_L": competence_update(c, "L"),
        "c_next_A": competence_update(c, "A"),
    }


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    by_policy = defaultdict(list)
    for r in rows:
        if r["t"] == cfg.T:
            by_policy[r["policy"]].append(r)
    out = {}
    for policy, last_rows in by_policy.items():
        n = len(last_rows)
        out[policy] = {
            "n_runs": n,
            "mean_final_c": sum(r["c_next"] for r in last_rows) / n,
            "mean_cumulative_u": sum(r["cumulative_utility"] for r in last_rows) / n,
            "mean_final_p": sum(r["p_correct"] for r in last_rows) / n,
        }
    return out


def main() -> None:
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check = formula_check()
    print("M0 formula check at (c0, z=0.5)")
    for k, v in check.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nBuilding finite-horizon DP (T={cfg.T})...")
    V = build_value_function_finite()
    seeds = run_seeds()
    all_rows: list[dict] = []
    for i, seed in enumerate(seeds, start=1):
        z_seq = task_sequence(seed)
        for policy in cfg.POLICIES:
            all_rows.extend(
                run_trajectory(policy, z_seq, seed, V=V if policy == "Dynamic" else None)
            )
        print(f"  run {i}/{len(seeds)} seed={seed}")

    summary = summarize(all_rows)
    print("\nM0 policy summary")
    print(f"{'policy':<10} {'mean final c':>14} {'mean cum. u':>14} {'mean final p':>14}")
    for policy in cfg.POLICIES:
        s = summary[policy]
        print(
            f"{policy:<10} {s['mean_final_c']:14.4f} "
            f"{s['mean_cumulative_u']:14.4f} {s['mean_final_p']:14.4f}"
        )

    csv_path = cfg.OUTPUT_DIR / "m0_runs.csv"
    fields = list(all_rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    report = {
        "model": "M0",
        "T": cfg.T,
        "n_runs": cfg.N_RUNS,
        "c0": cfg.C0,
        "attractors": dict(cfg.ATTRACTORS),
        "rho": cfg.RHO,
        "formula_check": check,
        "summary": summary,
        "csv": str(csv_path),
    }
    report_path = cfg.OUTPUT_DIR / "m0_summary.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
