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

AMENDMENT 1 (decided 2026-10-08 after the run, with both reads known; user's decision; the
reason's wording approved 2026-10-10): the joint
read is NOT APPLICABLE, recorded as a registered test that its own data invalidated, not as
a result. Its registered interpolation is linear in log row count across the bracketing
saturation sizes (10k, 30k), which presumes the curve is approximately linear in log row
count there; the registration states no such condition. The saturation curve's slope
reverses at or near 10k (seed means 378, 376, 455, 382 at 1k-30k; 214 on the whole
population), so 442.9 is not supported as an estimate of the curve at 11,955 rows, and the
matched-size control, a direct measurement at exactly that row count, supersedes it.
--amend-joint-read applies it, keeping the interpolated numbers with the reason they are not
used and the replaced block verbatim. main() refuses to overwrite an existing artifact; to
re-run, move the artifact aside, run, commit, then --amend-joint-read and commit again.

Run: OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/turner_nullspace_tr0.py
Amend: .venv/bin/python scripts/turner_nullspace_tr0.py --amend-joint-read [--dry-run]
       (--dry-run only together with --amend-joint-read; any other argument is refused)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import subprocess
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
    if (OUT / "metrics.json").exists():
        raise SystemExit(f"{(OUT / 'metrics.json').relative_to(R)} exists; a re-run rewrites the "
                         f"registered computation and drops its amendments, so move it aside "
                         f"first (see AMENDMENT 1 in the docstring)")
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


AMEND_JOINT_READ_ID = "joint_read_not_applicable"
# the reason as approved by the user on 2026-10-10, worded by Claude from the original below
AMEND_JOINT_READ_REASON_USER = (
    "The registered interpolation is linear in log row count between the bracketing "
    "saturation sizes, which presumes the curve is approximately linear in log row count "
    "between them. The measured curve (378, 376, 455, 382, 214) turns over at or near 10k (the "
    "rise from 3k to 10k and the fall from 10k to 30k are each about 5 SE), with the 10k and "
    "30k seed ranges barely overlapping, so its 442.9 is a property of the straight line rather "
    "than an estimate of the curve at 11,955 rows. The direct matched-size measurement at "
    "exactly 11,955 rows (468 +/- 25) is the valid comparator and supersedes it.")
# the user's original wording (2026-10-08), kept in the amendment record
AMEND_JOINT_READ_REASON_USER_ORIGINAL = (
    "The registered interpolation is linear in log row count between the bracketing "
    "saturation sizes. That presumes monotonicity between them. The measured curve (378, 376, "
    "455, 382, 214) has a real peak between 10k and 30k, with the 10k and 30k seed ranges "
    "barely overlapping, so the interpolation crosses a hump and its 442.9 is a property of "
    "the straight line rather than an estimate of the curve at 11,955 rows. The direct "
    "matched-size measurement at exactly 11,955 rows (468 +/- 25) is the valid comparator and "
    "supersedes it.")
AMEND_JOINT_READ_REASON_PROVENANCE = (
    "worded by Claude from the user's original reason (amendments[].reason_user_original), "
    "which named monotonicity as the premise (the operative one is linearity, and a curve "
    "non-increasing on [10k, 30k] is not rejected) and placed a peak inside [10k, 30k] (not "
    "resolved); approved by the user on 2026-10-10")
AMEND_JOINT_READ_TIMING = ("decided after the run, with the range test (BELOW in all four "
                           "configurations) and this read ('row count explains it') both known")


def _refuse(msg):
    raise SystemExit(f"refusing to amend: {msg}")


def amend_joint_read(dry_run=False):
    """Amendment 1 (user, 2026-10-08, after the run): the joint read is NOT APPLICABLE.

    A registered test that its own data invalidated, not a result. verdicts.joint_read is
    replaced by its status, the user's reason, the premise and the timing, the raw facts
    recomputed from the artifact's per-seed records, descriptive statistics and a note
    (both Claude's, labelled so), the superseding comparator, and the interpolated numbers
    with the reason they are not used. The replaced block is kept verbatim under
    `amendments`. The pre-registration (`predictions`) is not touched.

    Refuses unless the artifact is committed and unmodified, the code tree is clean, the
    stored interpolation recomputes in full, and every number the user's reason quotes is
    the artifact's. --dry-run prints the result and writes nothing."""
    p = OUT / "metrics.json"
    rel = str(p.relative_to(R))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=R,
                             capture_output=True).returncode == 0
    unmodified = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel],
                                cwd=R).returncode == 0
    prov = git_provenance(R)
    gate = {"artifact_tracked": tracked, "artifact_unmodified": unmodified,
            "code_tree_clean": not prov["git_dirty_excluding_runs"]}
    if not all(gate.values()) and not dry_run:
        _refuse(gate)
    raw = p.read_bytes()
    out = json.loads(raw)
    if out.get("status") != "complete":
        _refuse(f"status is {out.get('status')!r}")
    if any(a.get("id") == AMEND_JOINT_READ_ID for a in out.get("amendments", [])):
        _refuse(f"amendment {AMEND_JOINT_READ_ID} is already recorded")

    jr = out["verdicts"]["joint_read"]
    lo_s, hi_s = jr["bracket"]
    n, t = jr["tr0_rows"], jr["tr0_nontrivial"]
    sat = {x: [v["nontrivial_dim"] for v in out["saturation"][str(x)].values()]
           for x in SAT_SIZES}
    mean = {x: float(np.mean(v)) for x, v in sat.items()}
    sd = {x: float(np.std(v, ddof=1)) for x, v in sat.items()}
    se = {x: sd[x] / math.sqrt(len(v)) for x, v in sat.items()}
    anchor = out["saturation_anchor"]["nontrivial_dim"]

    # the registered interpolation, recomputed in full from the per-seed records
    w = (math.log(n) - math.log(lo_s)) / (math.log(hi_s) - math.log(lo_s))
    ip = lambda a, b: a + w * (b - a)  # noqa: E731
    m_i, s_i = ip(mean[lo_s], mean[hi_s]), ip(sd[lo_s], sd[hi_s])
    env = [ip(min(sat[lo_s]), min(sat[hi_s])), ip(max(sat[lo_s]), max(sat[hi_s]))]
    if not (abs(m_i - jr["interp_mean"]) < 1e-9 and abs(s_i - jr["interp_sd"]) < 1e-9
            and np.allclose(env, jr["interp_envelope"], rtol=0, atol=1e-9)
            and abs((t - m_i) / s_i - jr["z"]) < 1e-9
            and (env[0] <= t <= env[1]) == jr["inside_envelope"]):
        _refuse("the stored interpolation does not recompute from the per-seed records")

    # the superseding comparator: the matched-size control in the saturation configuration
    cfg = f"d{SAT_CONFIG[0]}_special_hp{SAT_CONFIG[1]}"
    ctl = out["control_branch"][cfg]
    k = len(ctl["subsample_dims"])
    se_d = ctl["sd"] / math.sqrt(k)
    rows = sorted({v["counts"]["distinct_pair_bearing"] for v in out["control"][cfg].values()})

    # every number the user's reason quotes must be this artifact's
    quoted = {"curve_means": [round(mean[x]) for x in SAT_SIZES] + [anchor],
              "interp_mean_1dp": round(m_i, 1), "tr0_rows": n, "bracket": [lo_s, hi_s],
              "bracket_ranges": [[min(sat[x]), max(sat[x])] for x in (lo_s, hi_s)],
              "direct_rows": rows, "direct_mean": round(ctl["mean"]),
              "direct_sd": round(ctl["sd"])}
    expected = {"curve_means": [378, 376, 455, 382, 214], "interp_mean_1dp": 442.9,
                "tr0_rows": 11_955, "bracket": [10_000, 30_000],
                "bracket_ranges": [[397, 515], [358, 399]], "direct_rows": [11_955],
                "direct_mean": 468, "direct_sd": 25}
    if quoted != expected:
        _refuse(f"the reason quotes {expected}, the artifact has {quoted}")

    def diff(a_mean, a_se, b_mean, b_se):
        d, s = b_mean - a_mean, math.hypot(a_se, b_se)
        return {"difference": d, "se": s, "in_se": d / s}

    steps = {f"{a}_to_{b}": diff(mean[a], se[a], mean[b], se[b])
             for a, b in zip(SAT_SIZES[:-1], SAT_SIZES[1:], strict=True)}
    l_lo, l_hi = math.log(10_000 / 3_000), math.log(30_000 / 10_000)
    slope_change = ((mean[30_000] - mean[10_000]) / l_hi
                    - (mean[10_000] - mean[3_000]) / l_lo)
    slope_change_se = math.sqrt((se[3_000] / l_lo) ** 2
                                + (se[10_000] * (1 / l_lo + 1 / l_hi)) ** 2
                                + (se[30_000] / l_hi) ** 2)
    se_line = math.hypot((1 - w) * se[lo_s], w * se[hi_s])
    vs_line = diff(m_i, se_line, ctl["mean"], se_d)
    vs_lo = diff(mean[lo_s], se[lo_s], ctl["mean"], se_d)
    rise, fall = steps["3000_to_10000"], steps["10000_to_30000"]
    if [round(abs(rise["in_se"])), round(abs(fall["in_se"]))] != [5, 5]:
        _refuse(f"the reason quotes a rise and a fall of about 5 SE, the artifact has "
                f"{rise['in_se']:+.2f} and {fall['in_se']:+.2f}")

    evidence = {
        "saturation_seed_dims": {str(x): sat[x] for x in SAT_SIZES},
        "saturation_seed_means": {str(x): mean[x] for x in SAT_SIZES}
        | {"anchor_full_population": anchor},
        "saturation_seed_sds": {str(x): sd[x] for x in SAT_SIZES},
        "bracket_seed_ranges": {str(x): [min(sat[x]), max(sat[x])] for x in (lo_s, hi_s)},
        "bracket_seed_range_overlap": [max(min(sat[lo_s]), min(sat[hi_s])),
                                       min(max(sat[lo_s]), max(sat[hi_s]))],
        "direct_at_tr0_rows": {"rows_per_subsample": rows, "config": cfg,
                               "seed_dims": ctl["subsample_dims"], "mean": ctl["mean"],
                               "sd": ctl["sd"], "n_seeds": k, "range": [ctl["min"], ctl["max"]]},
        "direct_minus_interpolated": ctl["mean"] - m_i,
        "direct_minus_10000_mean": ctl["mean"] - mean[lo_s],
        "independence": "each saturation size draws with SeedSequence([seed, size]) and the "
                        "matched-size control with SeedSequence([seed, 0]), each a fresh draw "
                        "without replacement from the same population, so the means are "
                        "independent estimates",
    }
    descriptive = {
        "status": f"descriptive and unregistered (Claude, {time.strftime('%Y-%m-%d')}); every "
                  f"SE combines the sampling error of each mean involved",
        "steps": steps,
        "slope_change_per_ln_rows_at_10000": {"value": slope_change, "se": slope_change_se,
                                              "in_se": slope_change / slope_change_se},
        "direct_minus_10000_mean": vs_lo,
        "direct_minus_interpolated": vs_line | {"se_direct_mean": se_d,
                                                "se_interpolated_mean": se_line},
    }
    note = (
        f"Resolved: the seed-mean curve turns over at or near the bracket's lower end. It "
        f"changes by {rise['difference']:+.1f} from 3k to 10k ({rise['in_se']:+.2f} SE) and by "
        f"{fall['difference']:+.1f} from 10k to 30k ({fall['in_se']:+.2f} SE); its slope per "
        f"ln(rows) changes by {slope_change:+.1f} at 10k ({slope_change / slope_change_se:+.2f} "
        f"SE). Not resolved: where the maximum lies relative to the bracket (the direct point "
        f"at {n:,} rows exceeds the 10k mean by {vs_lo['difference']:.1f}, "
        f"{vs_lo['in_se']:.2f} SE), and the direct point's departure from the line "
        f"({vs_line['difference']:.1f}, {vs_line['in_se']:.2f} SE once the line's own sampling "
        f"error is included). A curve non-increasing on [10k, 30k] is therefore not rejected; "
        f"the premise these data leave unsupported is the one linear interpolation needs, "
        f"approximate linearity in log row count across the bracket, because the slope "
        f"reverses at its lower end. The supersession needs no interpolation premise: the "
        f"matched-size control is a direct "
        f"measurement at exactly {n:,} rows, by the same procedure, configuration and "
        f"population as the saturation run.")
    new = {
        "status": "NOT APPLICABLE",
        "kind": "registered test invalidated by its own data; not a result",
        "amendment": AMEND_JOINT_READ_ID,
        "reason_user": AMEND_JOINT_READ_REASON_USER,
        "reason_user_provenance": AMEND_JOINT_READ_REASON_PROVENANCE,
        "premise_registered": False,
        "premise": "predictions.joint_read states no validity condition; interpolating linearly "
                   "in log row count presumes the seed-mean curve is approximately linear in log "
                   "row count across the bracket",
        "timing": AMEND_JOINT_READ_TIMING,
        "evidence_from_this_artifact": evidence,
        "descriptive_unregistered_claude": descriptive,
        "note_claude": note,
        "superseded_by": {
            "comparator": "the matched-size control: Rfam subsamples of exactly TR0+TS0's "
                          "distinct pair-bearing row count (control_branch)",
            "registered_rule": "the three-branch range test (predictions.matched_size_control)",
            "branch": {c: v["branch"] for c, v in out["control_branch"].items()},
            "no_reading_transferred_claude": "the |z| <= 2 rule was registered for the "
                                             "interpolated comparator only and is not applied "
                                             "to the matched-size control's z"},
        "interpolation_not_used": {
            **{x: jr[x] for x in ("bracket", "tr0_rows", "tr0_nontrivial", "interp_mean",
                                  "interp_sd", "interp_envelope", "z", "inside_envelope")},
            "why_not_used": "linear in log row count across [10k, 30k], while the curve's slope "
                            "reverses at or near 10k (note_claude), so the straight line is not "
                            "supported as an estimate of the curve at 11,955 rows; a direct "
                            "measurement at exactly that row count supersedes it (reason_user)"},
    }
    commit_before = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=R,
                                   capture_output=True, text=True).stdout.strip()
    record = {"id": AMEND_JOINT_READ_ID, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
              "by": "user (the decision, and reason_user as approved); Claude (the wording of "
                    "reason_user, evidence, descriptive statistics and notes)",
              "target": "verdicts.joint_read",
              "change": "status NOT APPLICABLE; the reading 'row count explains it' is "
                        "withdrawn; the interpolated numbers are kept under "
                        "interpolation_not_used",
              "reason_user": AMEND_JOINT_READ_REASON_USER,
              "reason_user_provenance": AMEND_JOINT_READ_REASON_PROVENANCE,
              "reason_user_original": AMEND_JOINT_READ_REASON_USER_ORIGINAL,
              "timing": AMEND_JOINT_READ_TIMING,
              "replaced": jr, "artifact_sha256_before": hashlib.sha256(raw).hexdigest(),
              "artifact_commit_before": commit_before, **prov}
    if dry_run:
        print(json.dumps({"gate": gate, "joint_read": new, "amendment": record}, indent=1,
                         default=str))
        return 0
    out["verdicts"]["joint_read"] = new
    out.setdefault("amendments", []).append(record)
    write(out)
    print(f"amendment {AMEND_JOINT_READ_ID} written; commit it")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        sys.exit(main())
    if sorted(argv) in (["--amend-joint-read"], ["--amend-joint-read", "--dry-run"]):
        sys.exit(amend_joint_read(dry_run="--dry-run" in argv))
    raise SystemExit(f"usage: {sys.argv[0]} [--amend-joint-read [--dry-run]]; got {argv}")
