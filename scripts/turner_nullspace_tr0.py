#!/usr/bin/env python3
"""The data-restricted Turner 2004 null space on TR0 + TS0 reference structures.

An independent check on the Rfam figure (nontrivial dimension 214 in all four
configurations, runs/2026-10-07_turner_nullspace_rfam_r2): bpRNA per-molecule reference
structures (the MXfold2 datasets' TR0 and TS0 canonical-pair sets, registered as
`mxfold2_datasets`, CC BY 4.0) instead of Rfam consensus projections.

The linear algebra is scripts/turner_nullspace_rfam.py's, imported and unchanged:
`analyse` (exact per-row reconstruction against ViennaRNA eval_structure; count matrix
over reachable columns; coverage; null dimension within the used columns from qr_null,
gram_null, reduced_null and exact rank mod p; the nontrivial dimension against
analytic_directions; controls C0-C4, R5, R6). Only the row loader is new.

ROW LOADER (fixed before running; the Rfam loader's rules where they apply)
  1. uppercase, T -> U; a sequence still containing a non-ACGU letter is dropped
  2. a pair closing a hairpin with fewer than 3 unpaired nucleotides is removed,
     repeated to a fixpoint (Rfam rule 4; MXfold2's own constraint)
  3. guards: a crossing (pseudoknotted) pair or a non-canonical pair fails the run
     (none were found when the data were profiled read-only)
  4. (sequence, structure) deduplicated; nominal, distinct, pair-bearing and distinct
     pair-bearing counts are reported, and the TR0/TS0 overlap
An unpaired chain has an all-zero count vector, so pair-bearing rows are the ones that
can affect a rank, and every size match below is on DISTINCT PAIR-BEARING rows.

CONTROL AND SATURATION (Rfam r2 rows; population = distinct pair-bearing rows)
  matched-size control  8 recorded seeds x 4 configurations, subsample size = TR0+TS0's
                        distinct pair-bearing count
  saturation run        1,000 / 3,000 / 10,000 / 30,000 rows x 8 seeds, d0_special_hp0,
                        plus the whole population as the top anchor (no seeds)
The artifact is rewritten atomically after every analysis.

Run: OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/turner_nullspace_tr0.py
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import RNA  # noqa: E402

from scripts.rfam_projection_robustness import recorded_seeds  # noqa: E402
from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_nullspace import PTYPE, dump_params, theta_from_par  # noqa: E402
from scripts.turner_nullspace_rfam import analyse, load_rows  # noqa: E402

RUN_ID = "2026-10-08_turner_nullspace_tr0"
OUT = R / "runs" / RUN_ID
DATA = R / ".cache" / "mxfold2-data"
RFAM_R2 = R / "runs" / "2026-10-07_turner_nullspace_rfam_r2" / "metrics.json"
ROBUST = R / "runs" / "2026-10-07_rfam_projection_robustness" / "metrics.json"
CONFIGS = [(0, 1), (0, 0), (2, 1), (2, 0)]
SAT_SIZES = [1_000, 3_000, 10_000, 30_000]
SAT_CONFIG = (0, 0)  # d0_special_hp0
ON_CURVE_REL_GAP = 0.01

PRED = {
    "T1": {
        "rule_user": "The TR0-restricted nontrivial dimension is below Rfam's 214, because "
                     "TR0 has fewer families and shorter sequences, and above the intrinsic 2.",
        "operational": "0 < nontrivial < 214 in every configuration ('nontrivial' already "
                       "excludes the 2 (d0) or 3 (d2) intrinsic directions)",
        "fact_recorded_before_running": "TR0 is NOT shorter than Rfam: median length 105 nt "
                                        "(max 498) vs Rfam's 84 (max 4,065), profiled "
                                        "read-only before registration",
        "prior_claude": "likely refuted on the 'below 214' side: a random 62% subsample of "
                        "Rfam rows had MORE nontrivial directions (287.4 +- 5.4) than the full "
                        "set, and TR0+TS0 is about 11% of Rfam's rows; uncertain, because "
                        "fewer rows also use fewer columns",
    },
    "C2": "every reported null basis vector moves no row by more than 1e-9; the worst row "
          "change is reported",
    "matched_size_control": {
        "rule": "per configuration, recorded BEFORE anything else is interpreted: with r = "
                "[min, max] of the 8 matched-size Rfam subsamples' nontrivial dimension, "
                "ABOVE (TR0 > max) or BELOW (TR0 < min) = at matched row count the dimension "
                "depends on corpus identity (Rfam consensus projections vs bpRNA per-molecule "
                "references), direction reported; INSIDE (min <= TR0 <= max) = it does not, "
                "so the measurement is a property of the model's structure space rather than "
                "of a corpus and Rfam's 214 stops being a single-source number. INSIDE is an "
                "informative outcome, not a failed control, and does not void the run.",
    },
    "joint_read": "d0_special_hp0: interpolate the saturation run's seed-mean nontrivial "
                  "dimension, and its seed SD, linearly in log row count between the two "
                  "saturation sizes bracketing TR0+TS0's distinct pair-bearing count; z = (TR0 "
                  "- interpolated mean) / interpolated SD; |z| <= 2: row count explains TR0's "
                  "dimension; |z| > 2: corpus identity does; also whether TR0 lies inside the "
                  "min-max envelope interpolated between the bracketing sizes. Reported "
                  "alongside the range test, not instead of it.",
    "anchor_check": "the whole distinct pair-bearing Rfam population in d0_special_hp0 must "
                    "reproduce the full Rfam set's nontrivial dimension exactly (unpaired rows "
                    "have zero count vectors; duplicates cannot change a rank)",
    "context_point_rule": f"the existing 67,418-row Rfam subsamples (drawn from NOMINAL rows) "
                          f"are 'effectively on the curve' only if the expected number of "
                          f"distinct pair-bearing rows in such a draw is within "
                          f"{ON_CURVE_REL_GAP:.0%} of 67,418; otherwise they stay context",
    "inherited": "exact reconstruction on every row (any failure stops the run), C0, C1, C3, "
                 "C4, R5, R6, agreement of the dimension methods",
}


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def bpseq(path):
    rows = [r.split() for r in path.read_text().split("\n") if r.strip() and not r.startswith("#")]
    return "".join(r[1] for r in rows), [0] + [int(r[2]) for r in rows]


def tr0_ts0_rows():
    """(acc, name, seq, db) rows per the registered loader rules, plus counts."""
    stats = {"raw": 0, "dropped_non_ACGU": 0, "pairs_removed_short_hairpin": 0,
             "structures_touched_by_short_hairpin_rule": 0, "by_list": {}}
    rows = []
    for lst, tag in (("TR0-canonicals.lst", "TR0"), ("TS0-canonicals.lst", "TS0")):
        n_list = 0
        for ln in (DATA / "data" / lst).read_text().split("\n"):
            if not ln.strip():
                continue
            f = DATA / ln.split()[0]
            raw, partner = bpseq(f)
            stats["raw"] += 1
            n_list += 1
            seq = raw.upper().replace("T", "U")
            if any(c not in "ACGU" for c in seq):
                stats["dropped_non_ACGU"] += 1
                continue
            n = len(seq)
            pt = [-1] * (n + 1)
            pairs = [(i, partner[i]) for i in range(1, n + 1) if partner[i] > i]
            for i, j in pairs:
                assert seq[i - 1] + seq[j - 1] in PTYPE, f"non-canonical pair in {f.name}"
                pt[i], pt[j] = j, i
            for i, j in pairs:
                for k in range(i + 1, j):
                    assert not (pt[k] > j or (0 < pt[k] < i)), f"crossing pair in {f.name}"
            touched, changed = False, True
            while changed:
                changed = False
                for a in range(1, n + 1):
                    b = pt[a]
                    if b > a and b - a - 1 < 3 and all(pt[x] == -1 for x in range(a + 1, b)):
                        pt[a] = pt[b] = -1
                        stats["pairs_removed_short_hairpin"] += 1
                        changed = touched = True
            stats["structures_touched_by_short_hairpin_rule"] += int(touched)
            db = "".join("(" if pt[k] > k else ")" if pt[k] > 0 else "." for k in range(1, n + 1))
            rows.append((tag, f.name, seq, db))
        stats["by_list"][tag] = n_list
    return rows, stats


def counts(rows):
    pb = [r for r in rows if "(" in r[3]]
    return {"nominal": len(rows), "distinct": len({(r[2], r[3]) for r in rows}),
            "pair_bearing": len(pb), "distinct_pair_bearing": len({(r[2], r[3]) for r in pb})}


def distinct_rows(rows):
    seen, out = set(), []
    for r in rows:
        k = (r[2], r[3])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def summary(res):
    """The fields kept for a control or saturation analysis."""
    return {k: res.get(k) for k in ("null_dim_total", "used_columns", "shape")} | {
        "coverage_dim": res.get("coverage", {}).get("dim"),
        "nontrivial_dim": res.get("nontrivial", {}).get("dim"),
        "null_dim_within_used": {k: v for k, v in res.get("null_dim_within_used", {}).items()
                                 if k in ("qr_svd", "eigh_gram", "eigh_reduced", "exact_mod_p",
                                          "agree")},
        "reconstruction_failures": res.get("reconstruction", {}).get("n_failed"),
        "C2_max_abs_change": res.get("C2_null_basis_unchanged", {}).get("max_abs_change"),
        "C2_passes": res.get("C2_null_basis_unchanged", {}).get("passes")}


def expected_distinct_pb(rows, n_draw):
    """Expected number of distinct pair-bearing rows in a uniform draw of n_draw NOMINAL rows
    without replacement (as the 67,418-row robustness subsamples were drawn)."""
    N = len(rows)
    mult = {}
    for r in rows:
        if "(" in r[3]:
            mult[(r[2], r[3])] = mult.get((r[2], r[3]), 0) + 1
    lg = math.lgamma
    base = lg(N - n_draw + 1) - lg(N + 1)  # log[(N-n)!/N!]
    tot = 0.0
    for m in mult.values():
        if N - m < n_draw:
            tot += 1.0
            continue
        # P(none of m copies drawn) = C(N-m, n)/C(N, n) = (N-m)!(N-n)! / (N!(N-m-n)!)
        lp = lg(N - m + 1) - lg(N - m - n_draw + 1) + base
        tot += 1.0 - math.exp(lp)
    return tot


def main():
    t0 = time.time()
    seeds = recorded_seeds()
    out = {"run_id": RUN_ID, "status": "PRE-REGISTERED, not yet run", "predictions": PRED,
           "seeds": list(seeds), "saturation_sizes": SAT_SIZES,
           "data": "TR0 + TS0 canonical-pair bpRNA reference structures (mxfold2_datasets, "
                   "CC BY 4.0); Rfam 15.1 seed rows exactly as in "
                   "runs/2026-10-07_turner_nullspace_rfam_r2 for the control and saturation",
           **git_provenance(R), "viennarna_version": RNA.__version__}
    write(out)

    rows, lstats = tr0_ts0_rows()
    tr = distinct_rows(rows)
    tags = {}
    for r in rows:
        tags.setdefault((r[2], r[3]), set()).add(r[0])
    out["tr0_ts0"] = {"loader": lstats, "counts": counts(rows),
                      "tr0_ts0_overlap_distinct": sum(1 for v in tags.values() if len(v) > 1)}
    n_match = out["tr0_ts0"]["counts"]["distinct_pair_bearing"]
    trpb = [r for r in tr if "(" in r[3]]
    assert len(trpb) == n_match
    write(out)

    rf_all, _ = load_rows()
    pop = [r for r in distinct_rows(rf_all) if "(" in r[3]]
    exp_pb = expected_distinct_pb(rf_all, 67_418)
    gap = abs(exp_pb - 67_418) / 67_418
    out["rfam"] = {"counts": counts(rf_all), "sampling_population": "distinct pair-bearing rows",
                   "population_size": len(pop),
                   "context_point_67418": {
                       "expected_distinct_pair_bearing_in_a_nominal_draw": exp_pb,
                       "relative_gap": gap, "threshold": ON_CURVE_REL_GAP,
                       "effectively_on_curve": gap <= ON_CURVE_REL_GAP}}
    out["match_size_distinct_pair_bearing"] = n_match
    write(out)

    P = theta_from_par(dump_params())
    name = lambda d, s: f"d{d}_special_hp{s}"  # noqa: E731

    # 1. TR0 + TS0, all four configurations (distinct pair-bearing rows; unpaired rows and
    #    duplicates cannot change a rank, and the counts above say how many were set aside)
    out["tr0_results"] = {}
    for d, s in CONFIGS:
        rng = np.random.default_rng(np.random.SeedSequence([seeds[0], d, s]))
        res = analyse(d, s, P, trpb, rng)
        out["tr0_results"][name(d, s)] = res
        write(out)
        if not res["reconstruction"]["passes"]:
            out["status"] = f"STOPPED: reconstruction failed in TR0 {name(d, s)}"
            write(out)
            return 1

    # 2. matched-size control: 8 seeds x 4 configurations; the branch per configuration is
    #    recorded as soon as its 8 subsamples are in
    out["control"], out["control_branch"] = {}, {}
    for d, s in CONFIGS:
        cfg = name(d, s)
        out["control"][cfg] = {}
        for sd in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([sd, 0]))
            idx = np.sort(rng.choice(len(pop), size=n_match, replace=False))
            sub = [pop[i] for i in idx]
            res = analyse(d, s, P, sub, np.random.default_rng(np.random.SeedSequence([sd, 1])))
            out["control"][cfg][str(sd)] = summary(res) | {"counts": counts(sub)}
            write(out)
        dims = [v["nontrivial_dim"] for v in out["control"][cfg].values()]
        tdim = out["tr0_results"][cfg]["nontrivial"]["dim"]
        lo, hi = min(dims), max(dims)
        mean, sd_ = float(np.mean(dims)), float(np.std(dims, ddof=1))
        out["control_branch"][cfg] = {
            "tr0_nontrivial": tdim, "subsample_dims": dims, "min": lo, "max": hi,
            "mean": mean, "sd": sd_, "z": (tdim - mean) / sd_ if sd_ > 0 else None,
            "branch": "ABOVE" if tdim > hi else "BELOW" if tdim < lo else "INSIDE",
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        write(out)

    # 3. saturation run (d0_special_hp0) and the top anchor
    d, s = SAT_CONFIG
    out["saturation"] = {}
    for size in SAT_SIZES:
        out["saturation"][str(size)] = {}
        for sd in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([sd, size]))
            idx = np.sort(rng.choice(len(pop), size=size, replace=False))
            res = analyse(d, s, P, [pop[i] for i in idx],
                          np.random.default_rng(np.random.SeedSequence([sd, size, 1])))
            out["saturation"][str(size)][str(sd)] = summary(res)
            write(out)
    res = analyse(d, s, P, pop, np.random.default_rng(np.random.SeedSequence([seeds[0], 99])))
    r2 = json.loads(RFAM_R2.read_text())["configs"][name(d, s)]["nontrivial"]["dim"]
    out["saturation_anchor"] = summary(res) | {
        "rows": len(pop), "rfam_r2_full_set_nontrivial": r2,
        "reproduces_full_set": res["nontrivial"]["dim"] == r2}
    rob = json.loads(ROBUST.read_text())["configs"][name(d, s)]["summary"]
    out["saturation_context_67418"] = {"random_mean": rob["random_mean"],
                                       "random_sd": rob["random_sd"],
                                       "basis": "nominal rows (see rfam.context_point_67418)"}
    write(out)

    # 4. verdicts
    cf = out["tr0_results"]
    nt = {k: v["nontrivial"]["dim"] for k, v in cf.items()}
    curve = {}
    for size in SAT_SIZES:
        dims = [v["nontrivial_dim"] for v in out["saturation"][str(size)].values()]
        curve[size] = (float(np.mean(dims)), float(np.std(dims, ddof=1)), min(dims), max(dims))
    below = [x for x in SAT_SIZES if x <= n_match]
    above = [x for x in SAT_SIZES if x >= n_match]
    lo_s = max(below) if below else None
    hi_s = min(above) if above else None
    joint = {"tr0_rows": n_match, "bracket": [lo_s, hi_s]}
    if lo_s is not None and hi_s is not None and lo_s != hi_s:
        w = (math.log(n_match) - math.log(lo_s)) / (math.log(hi_s) - math.log(lo_s))
        ip = lambda a, b: a + w * (b - a)  # noqa: E731
        m_i = ip(curve[lo_s][0], curve[hi_s][0])
        s_i = ip(curve[lo_s][1], curve[hi_s][1])
        env = (ip(curve[lo_s][2], curve[hi_s][2]), ip(curve[lo_s][3], curve[hi_s][3]))
        t = nt[name(*SAT_CONFIG)]
        z = (t - m_i) / s_i if s_i > 0 else None
        joint.update({"interp_mean": m_i, "interp_sd": s_i, "interp_envelope": list(env),
                      "tr0_nontrivial": t, "z": z,
                      "reading": ("row count explains it" if z is not None and abs(z) <= 2
                                  else "corpus identity does"),
                      "inside_envelope": env[0] <= t <= env[1]})
    else:
        joint["status"] = "TR0's row count is not bracketed by the saturation sizes; no read"
    all_sub = ([v for c in out["control"].values() for v in c.values()]
               + [v for c in out["saturation"].values() for v in c.values()]
               + [out["saturation_anchor"]])
    out["verdicts"] = {
        "matched_size_control_branch": {k: v["branch"] for k, v in out["control_branch"].items()},
        "T1": {"verdict": "confirmed" if all(0 < x < 214 for x in nt.values()) else "refuted",
               "nontrivial": nt},
        "C2": {"worst_row_change_tr0": max(v["C2_null_basis_unchanged"]["max_abs_change"]
                                           for v in cf.values()),
               "worst_row_change_all_analyses": max(x["C2_max_abs_change"] for x in all_sub),
               "passes": all(v["C2_null_basis_unchanged"]["passes"] for v in cf.values())
               and all(x["C2_passes"] for x in all_sub)},
        "joint_read": joint,
        "saturation_curve": {str(k): {"mean": v[0], "sd": v[1], "min": v[2], "max": v[3]}
                             for k, v in curve.items()}
        | {"anchor_full_population": out["saturation_anchor"]["nontrivial_dim"]},
        "anchor_check": out["saturation_anchor"]["reproduces_full_set"],
        "dimensions_tr0": {k: {"null_dim_total": v["null_dim_total"],
                               "null_dim_within_used": v["null_dim_within_used"]["qr_svd"],
                               "nontrivial": v["nontrivial"]["dim"],
                               "never_exercised": v["coverage"]["dim"]} for k, v in cf.items()},
        "inherited_checks_tr0": {k: {"reconstruction": v["reconstruction"]["passes"],
                                     "C0": all(x["detected"] for x in
                                               v["C0_corrupted_entry_detected"].values()),
                                     "C1": v["C1_random_nonnull_changes"]["fails_as_required"],
                                     "C3": v["C3_row_shuffle"]["changes_dim"],
                                     "C4": v["C4_planted_defect"]["detected"],
                                     "R5_not_null":
                                         not v["R5_uniform_stack_shift"]["in_null_space"],
                                     "methods_agree": v["null_dim_within_used"]["agree"]}
                                 for k, v in cf.items()},
        "reconstruction_failures_all_analyses": sum(x["reconstruction_failures"] or 0
                                                    for x in all_sub),
    }
    out["status"] = "complete"
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    write(out)
    print(json.dumps(out["verdicts"]["matched_size_control_branch"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
