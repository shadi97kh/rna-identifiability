#!/usr/bin/env python3
"""Type B, Experiment 1: is control (c)'s "depleted" finding the constant-shift mismatch in
reverse? EXPLORATORY; asked by the user 2026-10-12 after amendment 2.

Control (c) rotates delta at random inside each table's block of used columns. Amendment 2
showed (c) cannot produce a per-table constant vector (the false rank of 100 on the
intrinsic directions). The reverse may hold for the nontrivial component: published tables
differ from Turner 2004 partly by a constant shift per table; the 426 (214) null directions
are mostly differences between int22 entries and may contain almost no per-table constant;
the rotated controls carry part of each block's norm, including its constant part, into
directions that do. A delta with large per-table shifts would then rank low against (c)
without any real depletion.

Per set (Turner 1999, Andronescu 2007, Langdon 2018) and per basis (TR0+TS0, Rfam):
  overlap   for each of the four null tables, the squared overlap of its normalised
            constant vector (used columns on the shared support) with span(N_nt), beside
            its exact mean under (c)'s rotation, ||N_nt[block]||_F^2 / |block|, and a
            sampled mean as a cross-check
  shares    how much of ||P_nt delta||^2 the per-table-constant part of delta carries
  ranks     (c) recomputed with scripts/turner_typeb.decompose itself: on delta with the
            registered seed (must reproduce the artifact's rank exactly), and on delta with
            its per-table means removed (new seed)
Reading (the user's rule): if the ranks move toward 50, the stated finding becomes "not
enriched; the apparent depletion against table-matched rotations comes from constant
per-table shifts, which the null directions barely contain".

Reads the Type B artifact and bases, never writes them (the fold's analysis re-checks PRED).
Run (one core): OMP_NUM_THREADS=1 .venv/bin/python scripts/turner_typeb_c_check.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_par import template  # noqa: E402
from scripts.turner_typeb import (  # noqa: E402
    NULL_TABLES,
    REGISTERED,
    SEED,
    SETS,
    decompose,
    load_basis,
    read_out,
    set_delta,
)

RUN_ID = "2026-10-12_turner_typeb_c_check"
OUT = R / "runs" / RUN_ID
DRAWS = 1000


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def blocks_of(B, support):
    cols, used = B["cols"], B["used"]
    bl = {}
    for i, c in enumerate(used):
        if support[c]:
            bl.setdefault(cols[c][0], []).append(i)
    return {t: np.array(ix) for t, ix in bl.items()}


def main():
    reg = read_out()["experiment_1"]["per_set"]
    out = {"run_id": RUN_ID, "status": "running", "label": "EXPLORATORY (asked 2026-10-12)",
           **git_provenance(R)}
    write(out)
    theta0 = template()["theta"]
    Bs = {name: load_basis(name) for name in REGISTERED}
    cols = Bs["tr0ts0"]["cols"]
    res = {}
    for xi, X in enumerate(SETS):
        delta, _, support = set_delta(X, cols, theta0)
        res[X] = {}
        for bi, (name, B) in enumerate(Bs.items()):
            used, NT = B["used"], B["NT"]
            bl = blocks_of(B, support)
            du = delta[used]
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 7, xi, bi]))
            overlap = {}
            for t in NULL_TABLES:
                if t not in bl:
                    continue
                ix = bl[t]
                u = np.zeros(len(used))
                u[ix] = 1 / np.sqrt(len(ix))
                samp = []
                for _ in range(DRAWS):
                    g = rng.standard_normal(len(ix))
                    v = np.zeros(len(used))
                    v[ix] = g / np.linalg.norm(g)
                    samp.append(float(np.sum((NT.T @ v) ** 2)))
                overlap[t] = {"n_columns": int(len(ix)),
                              "constant_vector_squared_overlap": float(np.sum((NT.T @ u) ** 2)),
                              "rotated_mean_exact": float(np.sum(NT[ix] ** 2) / len(ix)),
                              "rotated_mean_sampled": float(np.mean(samp))}
            const = np.zeros(len(used))
            for ix in bl.values():
                const[ix] = du[ix].mean()
            p_all, p_const = NT.T @ du, NT.T @ const
            dm = delta.copy()
            dm[used] = du - const
            again = decompose(delta, B, np.random.default_rng(np.random.SeedSequence(
                [SEED, 3, xi, bi])), support)["control_c_table_matched"]
            dmr = decompose(dm, B, np.random.default_rng(np.random.SeedSequence(
                [SEED, 6, xi, bi])), support)
            r0 = reg[X][name]["control_c_table_matched"]["true_percentile_rank"]
            r1 = dmr["control_c_table_matched"]["true_percentile_rank"]
            res[X][name] = {
                "overlap_constant_vs_rotated": overlap,
                "nontrivial_projection_squared": {
                    "delta": float(p_all @ p_all), "per_table_constant_part": float(
                        p_const @ p_const), "demeaned_part": float((p_all - p_const)
                                                                   @ (p_all - p_const))},
                "constant_part_share_of_delta_norm_squared": float(
                    (const @ const) / (du @ du)) if du @ du else None,
                "rank_registered": r0,
                "rank_reproduced_with_registered_seed": again["true_percentile_rank"],
                "reproduces_registered": again["true_percentile_rank"] == r0,
                "rank_demeaned": r1,
                "demeaned_nontrivial_fraction": dmr["fraction"]["nontrivial"],
                "demeaned_q50": dmr["control_c_table_matched"]["q50_fraction"],
                "demeaned_q99": dmr["control_c_table_matched"]["q99_fraction"],
                "moves_toward_50": abs(r1 - 50) < abs(r0 - 50)}
            out["per_set"] = res
            write(out)
    moved = {f"{X}/{n}": v["moves_toward_50"] for X, d in res.items() for n, v in d.items()}
    out["reading"] = {
        "rule_user": "if the ranks move toward 50, the stated finding becomes 'not enriched; the "
                     "apparent depletion against table-matched rotations comes from constant "
                     "per-table shifts, which the null directions barely contain'",
        "moves_toward_50": moved,
        "all_reproduce_registered": all(v["reproduces_registered"] for d in res.values()
                                        for v in d.values())}
    out["status"] = "complete"
    write(out)
    print(json.dumps({X: {n: {k: v[k] for k in ("rank_registered", "rank_demeaned",
                                                "reproduces_registered",
                                                "constant_part_share_of_delta_norm_squared",
                                                "nontrivial_projection_squared")}
                          | {"overlap": {t: [round(o["constant_vector_squared_overlap"], 5),
                                             round(o["rotated_mean_exact"], 5)]
                                         for t, o in v["overlap_constant_vs_rotated"].items()}}
                          for n, v in d.items()} for X, d in res.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
