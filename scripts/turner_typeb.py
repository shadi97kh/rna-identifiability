#!/usr/bin/env python3
"""Type B: how much of a published RNA table is invisible to structure data, and do the
invisible directions change predictions? (Advisor's criticism 3.)

Model throughout: ViennaRNA 2.7.2, Turner 2004, dangles 0, special_hp 0, 37 C
(thermo.vienna_ref.reference_md), the configuration in which the data-restricted null
spaces were measured. d2 is out of scope: with dangles 2 ViennaRNA clips four tables at
zero, so the model is not linear in theta there (runs/2026-10-12_turner_par_writer, V2).

Structure sets A (each a count matrix over the 12,607 reachable parameter-file columns):
  tr0ts0  TR0 + TS0 bpRNA reference structures, distinct pair-bearing (11,955 rows),
          registered nontrivial null dimension 426 (runs/2026-10-08_turner_nullspace_tr0)
  rfam    Rfam 15.1 seed consensus projections, distinct pair-bearing, registered 214
          (runs/2026-10-07_turner_nullspace_rfam_r2)
R^12607 splits orthogonally, per A, into coverage (columns A never uses), intrinsic (2
analytic directions null for every structure), nontrivial (null for A's structures,
not intrinsic) and identified (the rest).

Phases (each reads and extends runs/<RUN_ID>/metrics.json, written atomically):
  preregister  writes PRED; refuses if the artifact exists
  bases        recomputes both bases with the registered functions (the artifacts kept only
               dimensions), checks them against the registered dimensions, saves them
               (bases_<A>.npz, A_<A>.npz; gitignored, sha256 recorded)
  project      Experiment 1: Turner 1999, Andronescu 2007 and Langdon 2018 (shipped in
               ViennaRNA) minus Turner 2004, decomposed; controls (a)-(c); then fixes the
               Experiment 2 grid by the registered rule, before any fold
  fold         Experiment 2 (added after Experiment 1 is reported; design registered in PRED)

Run: OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/turner_typeb.py \
       --phase {preregister,bases,project}
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp
from scipy import stats

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import RNA  # noqa: E402

from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_nullspace import (  # noqa: E402
    INF,
    Extractor,
    _reachable,
    analytic_directions,
    build_matrix,
    theta_from_par,
)
from scripts.turner_nullspace_rfam import load_rows, qr_null  # noqa: E402
from scripts.turner_nullspace_tr0 import distinct_rows, tr0_ts0_rows  # noqa: E402
from scripts.turner_par import _rebuild, template  # noqa: E402
from thermo.vienna_ref import assert_turner2004_params, reference_md  # noqa: E402

RUN_ID = "2026-10-12_turner_typeb"
OUT = R / "runs" / RUN_ID
SEED = 20261012
TOL = 1e-9
NULL_TABLES = ("int22", "int21", "int11", "bulge")
SETS = {"Turner1999": "params_load_RNA_Turner1999",
        "Andronescu2007": "params_load_RNA_Andronescu2007",
        "Langdon2018": "params_load_RNA_Langdon2018"}
REGISTERED = {
    "tr0ts0": {"nontrivial": 426, "null_within_used": 428, "coverage": 8046,
               "source": "runs/2026-10-08_turner_nullspace_tr0 tr0_results.d0_special_hp0"},
    "rfam": {"nontrivial": 214, "null_within_used": 216, "coverage": 4796,
             "source": "runs/2026-10-07_turner_nullspace_rfam_r2 configs.d0_special_hp0"},
}
N_NULL, N_CONTRAST = 8, 8
GRID_BASE = 0.125
CONTROL_C_DRAWS = 1000
FLOOR_SHARE = 0.05
CARRY_ROWS = 5
N_WORKERS = 12

PRED = {
    "registered": "2026-10-12, before bases, projection or any fold were computed",
    "scope": {
        "model": "ViennaRNA 2.7.2 Turner 2004, dangles 0, special_hp 0, 37 C",
        "d2_out_of_scope": "with dangles 2 ViennaRNA stores min(x, 0) for mismatchExt, "
                           "mismatchM, dangle5, dangle3 (runs/2026-10-12_turner_par_writer V2): "
                           "the model is not linear in theta there and the null-space analysis "
                           "does not transfer",
        "wording_rule_claude": "the 426 (214) are directions that THESE reference structures "
                               "cannot distinguish; only the 2 intrinsic directions are "
                               "invisible to every possible structure. Sentences say 'none of "
                               "the 11,955 bpRNA reference structures can distinguish', never "
                               "'no structure data can distinguish'.",
        "metric_note_claude": "the null space is metric-free; an orthogonal projection and its "
                              "fraction are not. Metric: Euclidean in raw parameter-file "
                              "coordinates, kcal/mol. Reported beside it, metric-free: the rms "
                              "change of A's reference energies, A delta.",
    },
    "bases": {
        "rule": "recompute with the registered functions (Extractor with exact reconstruction "
                "against eval_structure, build_matrix over the reachable columns, qr_null on the "
                "used columns, analytic_directions removed as in turner_nullspace_rfam.analyse); "
                "the dimensions must equal the registered ones exactly, or the run stops",
        "registered_dimensions": REGISTERED,
        "C2": "max |A N| <= 1e-9 for the intrinsic and the nontrivial bases",
        "basis_invariance": "every random direction is drawn as P g = N N^T g with g Gaussian on "
                            "the used columns, so no number depends on the basis's rotation",
    },
    "experiment_1_projection": {
        "rule_user": "for each shipped full RNA set other than Turner 2004, delta = theta_X - "
                     "theta_2004 on the shared columns; project onto the nontrivial null space "
                     "of TR0+TS0 and of Rfam; report ||P_N delta|| (kcal/mol), "
                     "||P_N delta|| / ||delta||, and the per-table breakdown of the null "
                     "component. Assert first that the column sets are identical; if not, "
                     "restrict to the shared support and say so.",
        "sets": SETS,
        "also_reported_claude": "the full orthogonal split (coverage, intrinsic, nontrivial, "
                                "identified) with squared fractions summing to 1; the "
                                "full-null fraction; the per-table distribution of delta "
                                "itself; the metric-free rms of A delta",
        "control_a_user": "MUST FAIL: a uniformly random subspace of R^n of the same dimension. "
                          "Its squared fraction is Beta(k/2, (n-k)/2) exactly, so quantiles are "
                          "computed, not sampled. The true fraction must exceed the control's "
                          "99th percentile, or the number is dimension counting.",
        "control_b_claude": "reported: uniform within the used columns (where the null space "
                            "lives), the same exact Beta, scaled by ||delta_used|| / ||delta||",
        "control_c_claude": f"reported AS A STATED FINDING whatever it returns (user's ruling): "
                            f"table-matched, {CONTROL_C_DRAWS} draws. A random rotation inside "
                            f"each table's block of used columns, applied to the basis, keeps "
                            f"every vector's per-table loading; equivalently delta's per-table "
                            f"blocks are rotated at random and projected on the true basis. If "
                            f"the true fraction does not clear the 99th percentile the finding "
                            f"is: 'the overlap is explained by both objects living in the "
                            f"internal-loop tables', and Type B's claim rests on Experiment 2.",
        "prior_claude": "every set's nontrivial fraction clears control (a)'s 99th percentile; "
                        "uncertain against (c); Andronescu 2007's fraction exceeds Turner "
                        "1999's (fitted, not revised)",
        "sentences_user": "Andronescu 2007: 'a published machine-learned parameter table "
                          "differs from Turner 2004 by X% in directions that none of the 11,955 "
                          "bpRNA reference structures can distinguish'; Turner 1999: the "
                          "companion sentence on successive revisions of the same model",
    },
    "experiment_2_fold": {
        "A": "TR0+TS0 (user's choice 2026-10-12): every reference structure in the corpus keeps "
             "exactly its energy, so no train/test framing; all 11,970 TR0+TS0 sequences "
             "kept by the loader are folded",
        "claim_user": "we perturb along a direction invisible to every one of the 11,955 "
                      "observed reference structures, and the predictions change anyway, "
                      "because folding a sequence evaluates structures nobody observed",
        "directions": {
            "null": f"{N_NULL} random unit directions in the TR0+TS0 nontrivial null space, "
                    f"seeds SeedSequence([{SEED}, 1, i])",
            "published": "unit(P_nt delta_X) for the three shipped sets, TR0+TS0 basis",
            "contrast": f"{N_CONTRAST} random unit directions in the identified complement "
                        f"restricted to {list(NULL_TABLES)} (g Gaussian on those used columns, "
                        f"nontrivial component removed; the intrinsic directions have no "
                        f"support there), seeds SeedSequence([{SEED}, 2, j])",
            "amendment_user_2026_10_12": "contrast raised from 4 to 8 directions: null vs "
                                         "contrast is a registered comparison of interest, so "
                                         "the arm doing the interpretive work needs the same "
                                         "number of draws as the null arm. If the box is busy, "
                                         "drop the lowest one or two grid points (below the "
                                         "expected floor, hatched anyway) to buy back the cost.",
            "excluded_by_design": "coverage directions (columns A never uses) are also invisible "
                                  "to A but are excluded: entries never observed are a weaker "
                                  "object",
        },
        "grid_rule": f"||delta|| = {GRID_BASE} * 2^k kcal/mol, k = 0, 1, ..., up to the first "
                     f"value >= 2 * max_X ||P_nt delta_X|| (TR0+TS0 basis); fixed by the "
                     f"project phase before any fold; each published component is also run at "
                     f"its own norm (the headline points)",
        "per_point": "theta_r = round(theta0 + delta) written by scripts/turner_par.write_par; "
                     "the rounding residual and ||theta_r - theta0 - delta|| / ||delta|| recorded",
        "metrics": "per sequence against the unperturbed fold: MFE base-pair F1 (1 if both "
                   "empty), MFE changed, D_bpp = (1/L) sum_{i<j} |p_ij - p'_ij|; F1 against the "
                   "reference, baseline and perturbed (reported, no prediction); means with "
                   "bootstrap 95% CIs over sequences and by length tercile",
        "leak_user": "at every point: eval_structure of all 11,955 references under the loaded "
                     "file must equal c(s).theta_r exactly (integer dcal; must hold); the leak "
                     "c(s).(theta_r - theta0) reported as median, 95th percentile and max in "
                     "kcal/mol beside every divergence value",
        "gap_user": "once under theta0: the gap from the MFE to the best other structure "
                    "(Wuchty subopt, delta 0, 1, 2, 5, 10, 20, 50, 100 dcal until a second "
                    "structure appears); its distribution and the degenerate fraction (gap 0)",
        "leak_scaling_note_claude": "the per-entry rounding residual is <= 0.5 dcal at every "
                                    "scale, so the leak saturates rather than grows with "
                                    "||delta||; the floor is defined by attribution instead "
                                    "of a leak-gap crossing (accepted by the user)",
        "attribution_user_and_claude": "for every sequence whose MFE changed (s0 -> s1): exact "
                                       "unrounded E(s) = c(s).(theta0 + delta) + offset(s); "
                                       "genuine if E(s1) <= E(s0), rounding-caused otherwise. "
                                       "Reported at EVERY point: the number changed and the "
                                       "genuine share",
        "floor_rule": f"the smallest grid ||delta|| at which the rounding-caused share of changed "
                      f"MFEs, pooled over the null directions, is <= {FLOOR_SHARE:.0%}; points "
                      f"below it are drawn hatched and not interpreted",
        "claim_rule": "at the three published-component points, if above the floor: the genuine "
                      "changed-MFE fraction's bootstrap 95% CI excludes 0",
        "comparison_of_interest_user": "genuine changed-MFE fraction (and F1, D_bpp), null vs "
                                       "contrast, at matched ||delta||, every grid point. "
                                       "Comparable movement is the expected outcome and not a "
                                       "failure: the claim is that these directions are "
                                       "invisible to the fitting data (the leak check) while "
                                       "visible to the predictor (this comparison). If "
                                       "comparable, the reading is 'they move predictions as "
                                       "much as any int-table perturbation of that size while no "
                                       "reference sees them', not 'unidentified directions are "
                                       "special'. Expectation, user's: comparable.",
        "guards_user_and_claude": f"per worker, per loaded file: turner_par.loaded() forces a "
                                  f"table rebuild; params_match on the first compound; a CARRY "
                                  f"CHECK: the {CARRY_ROWS} generating-set rows with the largest "
                                  f"|c(s).(theta_r - theta0)| evaluate to c(s).theta_r exactly "
                                  f"and that change is nonzero (params_match's entries do not "
                                  f"include int22/int21/int11, so it alone cannot see a stale "
                                  f"table here). Any failure stops the run; nothing is repaired.",
        "execution": f"{N_WORKERS} workers sharded by sequence, each computing its baseline once; "
                     f"driver under setsid, completion read from recorded status; uptime "
                     f"checked before launch; full run only on the user's go",
        "prior_claude": "floor between 0.5 and 2 kcal/mol; genuine changed-MFE fraction at the "
                        "Andronescu point well above 0",
    },
}


# ----------------------------------------------------------------------------------
# artifact
# ----------------------------------------------------------------------------------
def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def read_out():
    return json.loads((OUT / "metrics.json").read_text())


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def preregister():
    if (OUT / "metrics.json").exists():
        raise SystemExit(f"{OUT / 'metrics.json'} exists; the registration is not rewritten")
    write({"run_id": RUN_ID, "status": "PRE-REGISTERED, nothing computed", "predictions": PRED,
           "viennarna_version": RNA.__version__, **git_provenance(R)})
    return 0


# ----------------------------------------------------------------------------------
# bases
# ----------------------------------------------------------------------------------
def structure_rows():
    rows, _ = tr0_ts0_rows()
    rf, _ = load_rows()
    return {"tr0ts0": [r for r in distinct_rows(rows) if "(" in r[3]],
            "rfam": [r for r in distinct_rows(rf) if "(" in r[3]]}


def compute_basis(rows, P):
    theta, special, lxc = P["theta"], P["special"], P["lxc"]
    md = reference_md()
    assert md.dangles == 0 and md.special_hp == 0
    ex = Extractor(special, 0, 0, lxc=lxc)
    _rebuild()
    cnts, fails = [], []
    for n, (_, name, seq, db) in enumerate(rows):
        fc = RNA.fold_compound(seq, md)
        if n == 0:
            assert_turner2004_params(fc)
        e = int(round(fc.eval_structure(db) * 100))
        c = ex.counts(seq, db)
        off = c.pop(("_offset",), 0)
        if sum(theta[k] * v for k, v in c.items()) + off != e:
            fails.append(name)
        cnts.append(c)
    if fails:
        raise SystemExit(f"reconstruction failed on {len(fails)} rows: {fails[:5]}")
    cols = [k for k in sorted(theta, key=repr) if _reachable(k, 0, 0)]
    M = build_matrix(cnts, cols)
    nnz = np.asarray((M != 0).sum(axis=0)).ravel()
    used, unused = np.nonzero(nnz)[0], np.nonzero(nnz == 0)[0]
    A = M[:, used].tocsr()
    Nf, info = qr_null(A)
    an = analytic_directions(cols, 0, 0)
    Au = np.stack([g[used] for g in an.values() if np.abs(g[used]).max() > 0], axis=1)
    assert np.linalg.matrix_rank(Au) == Au.shape[1] == 2
    Qa, _ = np.linalg.qr(Au)
    resid = Nf - Qa @ (Qa.T @ Nf)
    U_, s_, _ = np.linalg.svd(resid, full_matrices=False)
    NT = U_[:, s_ > 1e-8]
    return {"cols": cols, "used": used, "unused": unused, "A": A, "I": Qa, "NT": NT,
            "null_within_used": int(Nf.shape[1]), "qr": info, "n_rows": len(rows)}


def table_weights(V, keys):
    w = {}
    for i, k in enumerate(keys):
        w[k[0]] = w.get(k[0], 0.0) + float((V[i] ** 2).sum())
    tot = sum(w.values())
    return {t: round(x / tot, 4) for t, x in sorted(w.items(), key=lambda z: -z[1])}


def bases():
    out = read_out()
    assert out["status"].startswith("PRE-REGISTERED"), out["status"]
    t0 = time.time()
    P = template()
    sets = structure_rows()
    res = {}
    for name, rows in sets.items():
        t1 = time.time()
        b = compute_basis(rows, P)
        ucols = [b["cols"][i] for i in b["used"]]
        c2 = {"intrinsic": float(np.abs(b["A"] @ b["I"]).max()),
              "nontrivial": float(np.abs(b["A"] @ b["NT"]).max())}
        dims = {"nontrivial": int(b["NT"].shape[1]), "null_within_used": b["null_within_used"],
                "coverage": int(len(b["unused"]))}
        reg = REGISTERED[name]
        match = all(dims[k] == reg[k] for k in dims)
        bp, ap = OUT / f"bases_{name}.npz", OUT / f"A_{name}.npz"
        np.savez(bp, used=b["used"], unused=b["unused"], I=b["I"], NT=b["NT"],
                 cols=np.array([repr(k) for k in b["cols"]]))
        sp.save_npz(ap, b["A"])
        res[name] = {"n_rows": b["n_rows"], "dims": dims, "registered": reg,
                     "dims_match_registered": match, "C2_max_abs": c2,
                     "C2_passes": all(v <= TOL for v in c2.values()),
                     "intrinsic_orthogonal_to_nontrivial": float(
                         np.abs(b["I"].T @ b["NT"]).max()),
                     "nontrivial_weight_by_table": table_weights(b["NT"], ucols),
                     "used_columns": int(len(b["used"])), "qr": b["qr"],
                     "files": {bp.name: sha256(bp), ap.name: sha256(ap)},
                     "seconds": round(time.time() - t1, 1)}
        out["bases"] = res
        write(out)
        if not (match and res[name]["C2_passes"]):
            out["status"] = f"STOPPED: basis {name} does not reproduce the registered dimensions"
            write(out)
            print(json.dumps(res[name], indent=1))
            return 1
    out["status"] = "bases complete"
    out["bases_seconds"] = round(time.time() - t0, 1)
    write(out)
    print(json.dumps({k: {kk: v[kk] for kk in ("n_rows", "dims", "dims_match_registered",
                                               "C2_max_abs", "nontrivial_weight_by_table")}
                      for k, v in res.items()}, indent=1))
    return 0


def load_basis(name):
    z = np.load(OUT / f"bases_{name}.npz")
    rec = read_out()["bases"][name]["files"]
    for f, h in rec.items():
        assert sha256(OUT / f) == h, f"{f} changed since the bases phase"
    cols = [ast.literal_eval(c) for c in z["cols"]]
    return {"cols": cols, "used": z["used"], "unused": z["unused"], "I": z["I"],
            "NT": z["NT"], "A": sp.load_npz(OUT / f"A_{name}.npz")}


# ----------------------------------------------------------------------------------
# Experiment 1: projection of the shipped tables
# ----------------------------------------------------------------------------------
def dump_set(loader: str):
    """theta of a shipped set, read through params_save; Turner 2004 restored after."""
    getattr(RNA, loader)()
    _rebuild()
    with tempfile.TemporaryDirectory() as t:
        p = pathlib.Path(t) / "set.par"
        RNA.params_save(str(p))
        P, h = theta_from_par(p), sha256(p)
    RNA.params_load_RNA_Turner2004()
    _rebuild()
    assert_turner2004_params(RNA.fold_compound("GGGGAAACCCC", reference_md()))
    return P, h


def by_table(keys):
    out = {}
    for k in keys:
        out[k[0]] = out.get(k[0], 0) + 1
    return dict(sorted(out.items()))


def decompose(delta, B, rng):
    """Orthogonal split of delta (kcal/mol over B['cols']) against structure set B."""
    cols, used, unused, NI, NT = B["cols"], B["used"], B["unused"], B["I"], B["NT"]
    du, dc = delta[used], delta[unused]
    pI, pNT = NI @ (NI.T @ du), NT @ (NT.T @ du)
    pid = du - pI - pNT
    n2 = float(delta @ delta)
    parts = {"coverage": float(dc @ dc), "intrinsic": float(pI @ pI),
             "nontrivial": float(pNT @ pNT), "identified": float(pid @ pid)}
    assert abs(sum(parts.values()) - n2) <= 1e-9 * n2, (sum(parts.values()), n2)
    nd = np.sqrt(n2)
    ucols = [cols[i] for i in used]
    k, n, nu = NT.shape[1], len(cols), len(used)
    frac = float(np.sqrt(parts["nontrivial"]) / nd)
    q99a = float(np.sqrt(stats.beta.ppf(0.99, k / 2, (n - k) / 2)))
    q99b = float(np.sqrt(stats.beta.ppf(0.99, k / 2, (nu - k) / 2)) * np.linalg.norm(du) / nd)
    blocks = {}
    for i, key in enumerate(ucols):
        blocks.setdefault(key[0], []).append(i)
    blocks = {t: np.array(ix) for t, ix in blocks.items()}
    norms = {t: float(np.linalg.norm(du[ix])) for t, ix in blocks.items()}
    draws = np.empty(CONTROL_C_DRAWS)
    for d in range(CONTROL_C_DRAWS):
        dp = np.zeros(nu)
        for t, ix in blocks.items():
            if norms[t] > 0:
                g = rng.standard_normal(len(ix))
                dp[ix] = norms[t] * g / np.linalg.norm(g)
        draws[d] = np.linalg.norm(NT.T @ dp) / nd
    q99c = float(np.percentile(draws, 99))
    clears_c = frac > q99c
    Adu = B["A"] @ du
    delta_by_table = {}
    for i, key in enumerate(cols):
        delta_by_table[key[0]] = delta_by_table.get(key[0], 0.0) + float(delta[i] ** 2)
    return {
        "norm_delta_kcal": nd,
        "norm_kcal": {p: float(np.sqrt(v)) for p, v in parts.items()},
        "fraction": {p: float(np.sqrt(v) / nd) for p, v in parts.items()},
        "squared_fraction": {p: v / n2 for p, v in parts.items()},
        "full_null_fraction": float(np.sqrt(parts["coverage"] + parts["intrinsic"]
                                            + parts["nontrivial"]) / nd),
        "nontrivial_component_by_table": table_weights(pNT[:, None], ucols),
        "delta_by_table_squared_share": {t: round(v / n2, 4) for t, v in sorted(
            delta_by_table.items(), key=lambda z: -z[1]) if v > 0},
        "metric_free_rms_reference_energy_change_kcal": float(np.sqrt(np.mean(Adu ** 2))),
        "metric_free_max_abs_reference_energy_change_kcal": float(np.abs(Adu).max()),
        "control_a_uniform_subspace": {"k": k, "n": n, "q99_fraction": q99a,
                                       "mean_fraction_approx": float(np.sqrt(k / n)),
                                       "true_exceeds_q99": frac > q99a},
        "control_b_uniform_in_used_columns": {"k": k, "n_used": nu, "q99_fraction": q99b,
                                              "true_exceeds_q99": frac > q99b},
        "control_c_table_matched": {
            "draws": CONTROL_C_DRAWS, "q50_fraction": float(np.percentile(draws, 50)),
            "q99_fraction": q99c, "max_fraction": float(draws.max()),
            "true_percentile_rank": float((draws < frac).mean() * 100),
            "true_exceeds_q99": clears_c,
            "finding": ("the nontrivial component exceeds what delta's own per-table "
                        "distribution gives a random direction: the overlap is not explained by "
                        "both objects living in the same tables alone" if clears_c else
                        "the overlap is explained by both objects living in the internal-loop "
                        "tables")},
    }


def project():
    out = read_out()
    assert out["status"] == "bases complete", out["status"]
    T = template()
    theta0 = T["theta"]
    Bs = {name: load_basis(name) for name in REGISTERED}
    cols = Bs["tr0ts0"]["cols"]
    assert cols == Bs["rfam"]["cols"]
    res = {"sets": {}, "per_set": {}}
    for X, loader in SETS.items():
        P, h = dump_set(loader)
        th = P["theta"]
        missing, extra = set(theta0) - set(th), set(th) - set(theta0)
        absent = [k for k in cols if k not in th]
        sentinel = [k for k in cols if k in th and (th[k] >= INF or theta0[k] >= INF)]
        drop = set(absent) | set(sentinel)
        res["sets"][X] = {"loader": loader, "par_sha256": h,
                          "keys_identical": not missing and not extra,
                          "missing_vs_turner2004_by_table": by_table(missing),
                          "extra_vs_turner2004_by_table": by_table(extra),
                          "reachable_absent": [repr(k) for k in absent],
                          "reachable_sentinel_INF": [repr(k) for k in sentinel],
                          "shared_support": len(cols) - len(drop)}
        delta = np.array([0.0 if k in drop else (th[k] - theta0[k]) / 100.0 for k in cols])
        res["per_set"][X] = {}
        for name, B in Bs.items():
            rng = np.random.default_rng(np.random.SeedSequence(
                [SEED, 3, list(SETS).index(X), list(REGISTERED).index(name)]))
            res["per_set"][X][name] = decompose(delta, B, rng)
        out["experiment_1"] = res
        write(out)
    # the Experiment 2 grid, by the registered rule, before any fold
    m = max(res["per_set"][X]["tr0ts0"]["norm_kcal"]["nontrivial"] for X in SETS)
    grid, g = [], GRID_BASE
    while True:
        grid.append(g)
        if g >= 2 * m:
            break
        g *= 2
    out["experiment_2_grid"] = {
        "rule": PRED["experiment_2_fold"]["grid_rule"], "max_published_nontrivial_norm": m,
        "grid_kcal": grid,
        "published_points_kcal": {X: res["per_set"][X]["tr0ts0"]["norm_kcal"]["nontrivial"]
                                  for X in SETS},
        "fixed": "by the project phase, before any fold"}
    out["status"] = "projection complete"
    write(out)
    print(json.dumps({X: {"keys_identical": res["sets"][X]["keys_identical"],
                          "shared_support": res["sets"][X]["shared_support"]} | {
        name: {kk: d[kk] for kk in ("norm_delta_kcal", "fraction", "full_null_fraction",
                                    "nontrivial_component_by_table",
                                    "metric_free_rms_reference_energy_change_kcal")}
        | {c: {kk: v for kk, v in d[c].items() if kk in ("q99_fraction", "true_exceeds_q99",
                                                         "true_percentile_rank")}
           for c in ("control_a_uniform_subspace", "control_b_uniform_in_used_columns",
                     "control_c_table_matched")}
        for name, d in res["per_set"][X].items()} for X in SETS}
        | {"grid": out["experiment_2_grid"]}, indent=1))
    return 0


PHASES = {"preregister": preregister, "bases": bases, "project": project}

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--phase" or sys.argv[2] not in PHASES:
        raise SystemExit(f"usage: turner_typeb.py --phase {{{','.join(PHASES)}}}")
    sys.exit(PHASES[sys.argv[2]]())
