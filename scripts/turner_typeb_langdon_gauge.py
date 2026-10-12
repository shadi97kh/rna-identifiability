#!/usr/bin/env python3
"""Type B amendment 2: is Langdon 2018's intrinsic component a gauge move, or only tables
shifted by constants? And how much of each published table's apparent difference from Turner
2004 does canonicalisation along the exact gauge remove?

EXPLORATORY whatever it returns: registered after Experiment 1 was seen
(runs/2026-10-12_turner_typeb, predictions.amendments_2026_10_12.
amendment_2_langdon_intrinsic_control_user).

The 2 intrinsic directions at dangles 0, special_hp 0 are per-table constant shifts:
  N1  +1 on every mismatchH entry, -1 on every hairpin[3..30]
  N2  +1 on every mismatchI, mismatch1nI, mismatch23I entry, -2 on every internal_loop[4..30]
(turner_nullspace.analytic_directions, over all 12,607 reachable columns: MODEL level, not
the TR0+TS0-restricted copies). So delta's intrinsic component is set entirely by delta's
per-table means over those six tables, and any revision that shifts them by constants
projects onto the directions whether or not the shifts obey the gauge relations (a shift c
along N1 moves mismatchH by +c and hairpin by -c; along N2, the three mismatch tables by +c and
internal_loop by -2c).

Control (per-table constant shift): keep each of the six tables' |mean shift|, flip signs; all
2^6 = 64 patterns are enumerated exactly. The statistic is the intrinsic fraction of the
per-table-constant part of delta. A pattern and its negation give the same value, so the
smallest attainable tail share is 2/64.

Canonicaliser: the exact gauge transformation theta_X + a1 N1 + a2 N2 (applied to every
column; INF stays INF) with (a1, a2) minimising the distance to Turner 2004 on the shared
support, i.e. least squares of delta on the directions restricted to the support. Reported:
||delta|| before and after, per table rms and mean |.| per entry for the six tables. The
transformation's exactness is checked on the 23,741 generating-set rows (G (N a) = 0).

Run (one core): OMP_NUM_THREADS=1 .venv/bin/python scripts/turner_typeb_langdon_gauge.py
"""

from __future__ import annotations

import itertools
import json
import os
import pathlib
import sys

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_nullspace import (  # noqa: E402
    Extractor,
    _reachable,
    analytic_directions,
    build_matrix,
    generating_set,
)
from scripts.turner_par import template  # noqa: E402
from scripts.turner_typeb import AMEND_ID, PRED, SETS, set_delta  # noqa: E402

RUN_ID = "2026-10-12_turner_typeb_langdon_gauge"
OUT = R / "runs" / RUN_ID
SIX = ("mismatchH", "hairpin", "mismatchI", "mismatch1nI", "mismatch23I", "internal_loop")


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def frac_on(Q, v):
    n = np.linalg.norm(v)
    return float(np.linalg.norm(Q.T @ v) / n) if n else None


def analyse_set(X, cols, theta0, N, G):
    delta, rec, support = set_delta(X, cols, theta0)
    Ns = N * support[:, None]
    Q, _ = np.linalg.qr(Ns)
    tab = np.array([k[0] for k in cols])
    ent = {t: np.nonzero((tab == t) & support)[0] for t in SIX}
    means = {t: float(delta[ix].mean()) for t, ix in ent.items()}
    const = np.zeros(len(cols))
    for t, ix in ent.items():
        const[ix] = means[t]
    # the intrinsic component is the per-table-constant part's
    gap = float(np.linalg.norm(Q.T @ delta - Q.T @ const))
    assert gap <= 1e-9 * max(1.0, np.linalg.norm(delta)), gap
    true = frac_on(Q, const)
    pats = []
    for sg in itertools.product((1, -1), repeat=len(SIX)):
        v = np.zeros(len(cols))
        for s, t in zip(sg, SIX, strict=True):
            v[ent[t]] = s * means[t]
        pats.append(frac_on(Q, v))
    pats = np.array(pats)
    # canonicaliser: exact gauge move, distance minimised on the support
    a, *_ = np.linalg.lstsq(Ns, -delta, rcond=None)
    move = N @ a
    exact = float(np.abs(G @ move).max())
    dc = delta + Ns @ a
    per_table = {}
    for t, ix in ent.items():
        per_table[t] = {"n_entries": int(len(ix)), "mean_shift_kcal": means[t],
                        "rms_before": float(np.sqrt(np.mean(delta[ix] ** 2))),
                        "rms_after": float(np.sqrt(np.mean(dc[ix] ** 2))),
                        "mean_abs_before": float(np.mean(np.abs(delta[ix]))),
                        "mean_abs_after": float(np.mean(np.abs(dc[ix])))}
    six = np.concatenate(list(ent.values()))
    return {
        "shared_support": rec["shared_support"],
        "intrinsic_norm_kcal_model_level": float(np.linalg.norm(Q.T @ delta)),
        "intrinsic_fraction_model_level": frac_on(Q, delta),
        "control_constant_shift": {
            "statistic": "intrinsic fraction of the per-table-constant part of delta",
            "true": true, "patterns": 64, "median": float(np.median(pats)),
            "max": float(pats.max()),
            "n_patterns_at_or_above_true": int((pats >= true - 1e-12).sum()),
            "tail_share": float((pats >= true - 1e-12).mean()),
            "smallest_attainable_tail_share": 2 / 64},
        "gauge_coefficients_kcal": {"N1": float(a[0]), "N2": float(a[1])},
        "gauge_relations_read": {
            "N1_wants_mismatchH_equal_minus_hairpin": [means["mismatchH"], -means["hairpin"]],
            "N2_wants_three_mismatch_tables_equal_and_internal_loop_minus_twice": [
                means["mismatchI"], means["mismatch1nI"], means["mismatch23I"],
                -means["internal_loop"] / 2]},
        "canonicaliser": {
            "norm_before_kcal": float(np.linalg.norm(delta)),
            "norm_after_kcal": float(np.linalg.norm(dc)),
            "fraction_of_norm_removed": float(1 - np.linalg.norm(dc) / np.linalg.norm(delta)),
            "six_tables_norm_before": float(np.linalg.norm(delta[six])),
            "six_tables_norm_after": float(np.linalg.norm(dc[six])),
            "six_tables_fraction_removed": float(1 - np.linalg.norm(dc[six])
                                                 / np.linalg.norm(delta[six])),
            "per_table": per_table,
            "exactness_max_abs_on_generating_set": exact, "exact": exact <= 1e-9},
    }


def main():
    out = {"run_id": RUN_ID, "status": "running", "label": "EXPLORATORY (amendment 2)",
           "registered_design": PRED[AMEND_ID]["amendment_2_langdon_intrinsic_control_user"],
           **git_provenance(R)}
    write(out)
    T = template()
    theta0 = T["theta"]
    cols = [k for k in sorted(theta0, key=repr) if _reachable(k, 0, 0)]
    an = analytic_directions(cols, 0, 0)
    N = np.stack([an["N1_hairpin_mismatch_vs_init"], an["N2_interior_mismatch_vs_init"]], 1)
    ex = Extractor(T["special"], 0, 0)
    G = build_matrix([ex.counts(s, d) for _, s, d in generating_set(T["special"])], cols)
    out["directions_exact_on_generating_set"] = float(np.abs(G @ N).max())
    assert out["directions_exact_on_generating_set"] == 0.0
    out["per_set"] = {X: analyse_set(X, cols, theta0, N, G) for X in SETS}
    out["status"] = "complete"
    write(out)
    print(json.dumps({X: {k: v[k] for k in ("intrinsic_norm_kcal_model_level",
                                            "intrinsic_fraction_model_level",
                                            "control_constant_shift", "gauge_coefficients_kcal",
                                            "gauge_relations_read")}
                      | {"canonicaliser": {k: v["canonicaliser"][k] for k in (
                          "fraction_of_norm_removed", "six_tables_fraction_removed", "exact")}}
                      for X, v in out["per_set"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
