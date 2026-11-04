#!/usr/bin/env python3
"""POST HOC correction: the gap-held-fixed retention interval, resampling whole sequences.

The record runs/2026-10-21_retention_gap drew units within each stratum independently, so one
sequence's pairs in different gap deciles were resampled apart and the dependence between them
left the interval. This draws the 1,288 TS0 sequences with replacement once per replicate, and
a drawn sequence carries all its pairs into every stratum. The pool, the strata (fixed from the
full data), the drop rule and the statistic are that record's, reused from its own code, and
every point estimate must reproduce it exactly. Nothing here edits that record.

A stratum with no certified or no uncertified pair on a replicate is undefined there. It is
counted, and the pooled interval is then reported only over the replicates where every stratum
is defined, with the count beside it.

Run (CPU only, about a minute):
  CUDA_VISIBLE_DEVICES= .venv-certfold/bin/python scripts/retention_gap_sequence_bootstrap.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold_checks as rv  # noqa: E402
import scripts.certfold_default_cert as dc  # noqa: E402
import scripts.certfold_exploratory as cx  # noqa: E402
import scripts.posthoc_2026_10_21 as ph  # noqa: E402

r2, cf, cr = dc.r2, dc.cf, dc.cr
GuardFailure = dc.GuardFailure

RUN_ID = "2026-11-03_retention_gap_sequence_bootstrap"
OUT = R / "runs" / RUN_ID
OLD = R / "runs" / "2026-10-21_retention_gap" / "metrics.json"
TAG = 97
N_BOOT = 10000
N_CHECK = 25                     # replicates recomputed from explicit records


def _write(out: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, OUT / "metrics.json")


def _interval(xs: np.ndarray) -> list[float]:
    half = (100.0 - cx.BOOT_LEVEL) / 2
    lo, hi = np.percentile(xs, [half, 100.0 - half])
    return [float(lo), float(hi)]


def analyse(kept: dict, units: list[int], counts: np.ndarray, spec: dict) -> dict:
    """The statistic per stratum and pooled over the replicates in `counts` (replicate by unit:
    how many times each unit was drawn), with the pooled weights the replicate's own pair counts."""
    strata = sorted(kept)
    at = {u: i for i, u in enumerate(units)}
    n_c, n_u = (np.zeros((len(units), len(strata))) for _ in range(2))
    s_c, s_u = (np.zeros((len(units), len(strata))) for _ in range(2))
    for j, s in enumerate(strata):
        for u, rs in kept[s].items():
            for r in rs:
                n, t = (n_c, s_c) if r["certified_l1"] else (n_u, s_u)
                n[at[u], j] += 1
                t[at[u], j] += r["ret_primary"]
    nc, nu = counts @ n_c, counts @ n_u
    tc, tu = counts @ s_c, counts @ s_u
    defined = (nc > 0) & (nu > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = tc / nc - tu / nu
    w = nc + nu
    all_defined = defined.all(axis=1)
    pooled = (w[all_defined] * vals[all_defined]).sum(axis=1) / w[all_defined].sum(axis=1)

    # the matrix route against explicit records, on the first replicates
    for b in range(N_CHECK):
        got, wt = {}, {}
        for j, s in enumerate(strata):
            rs = [r for i, u in enumerate(units) for _ in range(int(counts[b, i]))
                  for r in kept[s].get(u, ())]
            if not spec["defined"](rs):
                if defined[b, j]:
                    raise GuardFailure("the two routes disagree on a stratum's definedness")
                continue
            got[s], wt[s] = spec["stat"](rs), spec["weight"](rs)
            if abs(got[s] - vals[b, j]) > 1e-12 or wt[s] != w[b, j]:
                raise GuardFailure(f"replicate {b} stratum {s}: the matrix route disagrees")
        if all_defined[b] and abs(cx._pooled(got, wt) - pooled[int(all_defined[:b].sum())]) > 1e-12:
            raise GuardFailure(f"replicate {b}: the pooled value disagrees")

    by = {}
    for j, s in enumerate(strata):
        rs = [r for u in kept[s].values() for r in u]
        ok = defined[:, j]
        by[str(s)] = {"point": spec["stat"](rs), "interval": _interval(vals[ok, j]),
                      "resamples_undefined": int((~ok).sum()), "n_pairs": len(rs),
                      **spec["extra"](rs)}
    full = {str(s): by[str(s)]["point"] for s in strata}
    weights = {str(s): by[str(s)]["n_pairs"] for s in strata}
    return {"pooled_point": cx._pooled(full, weights), "pooled_interval": _interval(pooled),
            "pooled_resamples_undefined": int((~all_defined).sum()), "by_stratum": by,
            "strata": [str(s) for s in strata]}


def main() -> int:
    cr._in_certfold_env()
    prov = r2.clean()
    t0 = time.time()
    old = json.loads(OLD.read_text())["c1"]
    rows, recs = rv.base_records()
    gap = {(r["index"], tuple(q["pair"])): float(q["nominal_gap_kcal"]) for r in rows
           for q in r["pairs"]}
    stratum = {r["index"]: r["stratum"] for r in rows}
    pool = [r for r in recs if r["kind"] == "mfe" and r["p"] >= cr.P_HI]
    g = [gap[(r["index"], tuple(r["pair"]))] for r in pool]
    dec, ter = ph.rank_bins(g, 10), ph.rank_bins(g, 3)
    pool = [{**r, "gap_kcal": gv, "gap_decile": d, "gap_tercile": t}
            for r, gv, d, t in zip(pool, g, dec, ter, strict=True)]
    units = sorted(r["index"] for r in rows)
    if len(units) != len(set(units)) or len(units) != 1288:
        raise GuardFailure(f"{len(units)} TS0 sequences, not 1,288 distinct")
    spec = {**r2._r4_spec("ret_primary"), "merge": {}}

    rng = np.random.default_rng(np.random.SeedSequence([cf.SEED, TAG]))
    draws = rng.integers(0, len(units), size=(N_BOOT, len(units)))
    counts = np.zeros((N_BOOT, len(units)))
    for b in range(N_BOOT):
        counts[b] = np.bincount(draws[b], minlength=len(units))
    if not (counts.sum(axis=1) == len(units)).all():
        raise GuardFailure("a replicate did not draw every unit's worth of sequences")

    res = {}
    for name, key in (("primary_gap_deciles", lambda r: r["gap_decile"]),
                      ("secondary_length_x_gap_tercile",
                       lambda r: 3 * (0 if stratum[r["index"]] == 1 else stratum[r["index"]])
                       + r["gap_tercile"])):
        kept, dropped = ph.c1_groups(pool, key)
        a = analyse(kept, units, counts, spec)
        was = old[name]
        if a["pooled_point"] != was["pooled_point"] and \
                abs(a["pooled_point"] - was["pooled_point"]) > 1e-12:
            raise GuardFailure(f"{name}: pooled point {a['pooled_point']} != the record's")
        if set(a["by_stratum"]) != set(was["by_stratum"]):
            raise GuardFailure(f"{name}: the strata differ from the record's")
        for s, v in a["by_stratum"].items():
            w = was["by_stratum"][s]
            if abs(v["point"] - w["point"]) > 1e-12 or any(
                    v[k] != w[k] for k in ("n_certified", "n_uncertified", "n_pairs")):
                raise GuardFailure(f"{name} stratum {s}: differs from the record")
        if dropped != was["strata_dropped"]:
            raise GuardFailure(f"{name}: the dropped strata differ from the record's")
        a.update(gap_range_kcal_by_stratum=was["gap_range_kcal_by_stratum"],
                 strata_dropped=dropped, n_pairs_dropped=was["n_pairs_dropped"],
                 n_pairs_kept=was["n_pairs_kept"],
                 interval_within_strata_bootstrap=was["pooled_interval"])
        res[name] = a
    iv = res["primary_gap_deciles"]["pooled_interval"]
    out = {
        "run_id": RUN_ID, "status": "done",
        "label": "POST HOC correction of the interval in runs/2026-10-21_retention_gap",
        "c1": {
            "pool": old["pool"], "n_pool": len(pool),
            "n_certified": sum(1 for r in pool if r["certified_l1"]),
            "statistic": old["statistic"], "drop_rule": old["drop_rule"],
            "bootstrap": {"n_boot": N_BOOT, "seed": [cf.SEED, TAG], "n_units": len(units),
                          "units": "whole TS0 sequences, drawn once per replicate; a drawn "
                                   "sequence carries all its pairs into every stratum"},
            **res,
            "reading": ("certified pairs keep a retention advantage when the gap is held fixed"
                        if iv[0] > 0 else
                        "the advantage is not shown once the gap is held fixed")},
        "must_hold": "points, counts and dropped strata equal the record's; the matrix route "
                     f"equals explicit records on the first {N_CHECK} replicates",
        "seconds": round(time.time() - t0, 1), "provenance": prov}
    _write(out)
    print(json.dumps({k: {kk: vv for kk, vv in out["c1"][k].items() if "pooled" in kk or kk ==
                          "interval_within_strata_bootstrap"}
                      for k in res}, indent=1))
    print(out["c1"]["reading"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
