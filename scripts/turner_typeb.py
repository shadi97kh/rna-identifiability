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
import contextlib
import hashlib
import json
import os
import pathlib
import subprocess
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
    generating_set,
    theta_from_par,
)
from scripts.turner_nullspace_rfam import load_rows, qr_null  # noqa: E402
from scripts.turner_nullspace_tr0 import distinct_rows, tr0_ts0_rows  # noqa: E402
from scripts.turner_par import _rebuild, loaded, params_match, template, write_par  # noqa: E402
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

AMEND_ID = "amendments_2026_10_12"
BOOT_DRAWS = 2000
PRED[AMEND_ID] = {
    "timing": "written 2026-10-12 after Experiment 1 and the fold smoke test were seen, before "
              "the full fold was launched; written into the artifact by --phase amend",
    "experiment_1_outcome": {
        "priors_failed": [
            "every table's nontrivial fraction clears control (a)'s 99th percentile: FAILED, "
            "no table clears it in either basis",
            "Andronescu 2007's nontrivial fraction exceeds Turner 1999's: FAILED (0.180 vs "
            "0.184, TR0+TS0 basis)"],
        "attribution": "registered under prior_claude; the user records them as their guesses "
                       "too",
        "why_none_was_expected_user": "TR0+TS0's null space comes from TR0+TS0 and none of "
                                      "these tables was fitted to it; enrichment would be "
                                      "expected only against each table's own training null "
                                      "space, which cannot be rebuilt; not pursued before the "
                                      "scope freeze",
        "what_survives_user": "calibration: Andronescu's P_nt delta is 21.0 kcal/mol of real "
                              "difference between published tables, all of it in directions "
                              "no TR0+TS0 reference can see. Experiment 2 asks whether moving "
                              "that far changes predictions; if it does: 'two tables that "
                              "agree on all 11,955 reference energies disagree on X% of "
                              "predicted MFEs, at a distance the size of what separates "
                              "published tables'. No enrichment is needed for it.",
    },
    "amendment_1_less_reading_user": {
        "reason": "the smoke test hinted that null directions may move predictions much less "
                  "than contrast at equal norm (0/50 vs 2/50 changed MFEs; mean D_bpp 0.00036 "
                  "vs 0.0025; one direction each, 50 sequences: a reason to register the "
                  "branch, not evidence)",
        "rule_user": "'less': the null-minus-contrast difference in genuine changed-MFE "
                     "fraction has a 95% CI below 0, bootstrapped over sequences and directions",
        "reading_if_it_fires_user": "the references' blindness to these directions largely "
                                    "carries over to competing structures on the same "
                                    "sequences",
        "effect_on_expectation_user": "the user's 'comparable' expectation is then simply "
                                      "wrong, and the artifact says so",
        "effect_on_main_claim_user": "none: the claim at the published points stands or falls "
                                     "on its own CI either way",
        "operational_note_claude": f"{BOOT_DRAWS} bootstrap draws; each resamples sequences "
                                   f"with replacement (one resample shared by both arms: the "
                                   f"arms fold the same sequences) and, independently, the 8 "
                                   f"directions of each arm with replacement; the statistic is "
                                   f"mean genuine changed-MFE fraction (null) minus (contrast); "
                                   f"95% percentile CI. Evaluated at every grid scale at or "
                                   f"above the floor with at least one changed MFE in either "
                                   f"arm. Overall: 'less' if the CI is below 0 at every such "
                                   f"scale, 'more' if above 0 at every such scale, "
                                   f"'comparable' if it contains 0 at every such scale, "
                                   f"otherwise 'mixed', with the per-scale results reported.",
    },
    "amendment_2_langdon_intrinsic_control_user": {
        "status": "registered after seeing Experiment 1; EXPLORATORY whatever it returns",
        "why_user": "if it holds, the paper's most concrete Type A example: a published table "
                    "sitting 3.8 kcal/mol along an exact gauge direction; and the natural test "
                    "of the canonicaliser",
        "structure_note_claude": "the 2 intrinsic directions are per-table constant shifts (N1: "
                                 "+1 on every mismatchH entry, -1 on every hairpin[3..30]; N2: "
                                 "+1 on every mismatchI, mismatch1nI, mismatch23I entry, -2 on "
                                 "every internal_loop[4..30]), so delta's intrinsic component "
                                 "is set entirely by delta's per-table means over those six "
                                 "tables. Any revision that shifts those tables by constants "
                                 "projects onto them, whether or not the shifts satisfy the "
                                 "gauge relations.",
        "control": "per-table constant-shift control: keep each of the six tables' |mean "
                   "shift|, flip the signs; all 2^6 = 64 sign patterns enumerated exactly; the "
                   "statistic is the intrinsic fraction of the per-table-constant part of "
                   "delta; report the true pattern's rank among the 64 and the per-table mean "
                   "shifts themselves (kcal/mol)",
        "basis_note_claude": "the MODEL-level intrinsic directions (analytic_directions over all "
                             "reachable columns, not the TR0+TS0-restricted ones), restricted "
                             "to each table's shared support and orthonormalised; their "
                             "exactness is checked on the 23,741 generating-set rows (max "
                             "|G g| <= 1e-9 before restriction)",
        "canonicaliser_user": "canonicalise toward Turner 2004 along the 2 intrinsic "
                              "directions: g* = argmin ||delta + g|| over their span on the "
                              "shared support; report ||delta|| vs ||delta + g*||, and per "
                              "table the rms per-entry difference before and after for the "
                              "six tables: how much of the apparent per-entry difference goes "
                              "away. Langdon is the case of interest; Andronescu and Turner "
                              "1999 alongside.",
        "execution": "scripts/turner_typeb_langdon_gauge.py, its own artifact "
                     "(runs/2026-10-12_turner_typeb_langdon_gauge), one core, while the fold "
                     "runs; it never writes the Type B artifact",
    },
    "experiment_1_control_c_correction": "control (c) for Langdon 2018 rotated delta into 359 "
                                         "used columns where delta is undefined (INF in "
                                         "Langdon); restricted to the shared support before "
                                         "the fold, the superseded values kept in "
                                         "experiment_1.corrections",
    "null_arm_leak_assertion_user": "for every null-kind point, A.(theta_r - theta0) must equal "
                                    "A.r: |A.delta| <= 1e-9 kcal/mol per kcal/mol of ||delta|| "
                                    "on all 11,955 references, asserted when the points are "
                                    "built (the reference-energy identity in the worker holds "
                                    "for any theta_r, so it cannot stop a non-null 'null' "
                                    "direction)",
    "pred_recheck": "fold-analyse refuses unless the artifact's predictions equal this PRED",
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


def amend():
    """Write PRED's amendment block into the artifact. Refused once a fold has started, and
    refused unless everything registered before is unchanged."""
    out = read_out()
    if "fold" in out:
        raise SystemExit("a fold has started; PRED is not amended while it runs")
    reg = out["predictions"]
    if AMEND_ID in reg:
        raise SystemExit(f"{AMEND_ID} is already in the artifact")
    new = json.loads(json.dumps(PRED, default=str))
    if {k: v for k, v in new.items() if k != AMEND_ID} != reg:
        raise SystemExit("PRED differs from the registration outside the amendment block")
    out["predictions"] = new
    out.setdefault("amendment_log", []).append(
        {"id": AMEND_ID, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "status_when_written": out["status"], **git_provenance(R)})
    write(out)
    print(f"{AMEND_ID} written; status {out['status']}")
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
    if tot == 0:
        return {}
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


def decompose(delta, B, rng, support=None):
    """Orthogonal split of delta (kcal/mol over B['cols']) against structure set B.

    `support` (bool over cols; default all) is where delta is defined. delta must be zero
    off it. Control (c) rotates delta only within each table's used columns ON the support:
    rotating into columns where delta is undefined would let the draws reach directions the
    true delta cannot (corrected 2026-10-12 for Langdon 2018, whose 1,266 INF columns include
    359 used ones carrying 9.3% of the nontrivial basis's mass)."""
    if support is None:
        support = np.ones(len(B["cols"]), dtype=bool)
    assert not np.any(delta[~support]), "delta is nonzero off its support"
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
        if support[used[i]]:
            blocks.setdefault(key[0], []).append(i)
    blocks = {t: np.array(ix) for t, ix in blocks.items()}
    norms = {t: float(np.linalg.norm(du[ix])) for t, ix in blocks.items()}
    draws, draws_i = np.empty(CONTROL_C_DRAWS), np.empty(CONTROL_C_DRAWS)
    for d in range(CONTROL_C_DRAWS):
        dp = np.zeros(nu)
        for t, ix in blocks.items():
            if norms[t] > 0:
                g = rng.standard_normal(len(ix))
                dp[ix] = norms[t] * g / np.linalg.norm(g)
        draws[d] = np.linalg.norm(NT.T @ dp) / nd
        draws_i[d] = np.linalg.norm(NI.T @ dp) / nd
    q99c = float(np.percentile(draws, 99))
    clears_c = frac > q99c
    Adu = B["A"] @ du
    delta_by_table = {}
    for i, key in enumerate(cols):
        delta_by_table[key[0]] = delta_by_table.get(key[0], 0.0) + float(delta[i] ** 2)
    fi, ki = float(np.sqrt(parts["intrinsic"]) / nd), NI.shape[1]
    q99ai = float(np.sqrt(stats.beta.ppf(0.99, ki / 2, (n - ki) / 2)))
    post_hoc_intrinsic = {
        "label": "POST HOC: added 2026-10-12 after Experiment 1's registered results were seen; "
                 "not registered, same two controls as the nontrivial fraction",
        "fraction": fi, "control_a_q99_fraction": q99ai,
        "control_a_mean_fraction_approx": float(np.sqrt(ki / n)),
        "exceeds_control_a_q99": fi > q99ai,
        "control_c_q99_fraction": float(np.percentile(draws_i, 99)),
        "control_c_percentile_rank": float((draws_i < fi).mean() * 100),
        "exceeds_control_c_q99": fi > float(np.percentile(draws_i, 99))}
    return {
        "post_hoc_intrinsic": post_hoc_intrinsic,
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


def set_delta(X, cols, theta0):
    """delta = theta_X - theta_2004 over `cols` in kcal/mol, zero off the shared support, and
    the record of how the key sets compare."""
    P, h = dump_set(SETS[X])
    th = P["theta"]
    missing, extra = set(theta0) - set(th), set(th) - set(theta0)
    absent = [k for k in cols if k not in th]
    sentinel = [k for k in cols if k in th and (th[k] >= INF or theta0[k] >= INF)]
    drop = set(absent) | set(sentinel)
    rec = {"loader": SETS[X], "par_sha256": h, "keys_identical": not missing and not extra,
           "missing_vs_turner2004_by_table": by_table(missing),
           "extra_vs_turner2004_by_table": by_table(extra),
           "reachable_absent": [repr(k) for k in absent],
           "reachable_sentinel_INF": [repr(k) for k in sentinel],
           "shared_support": len(cols) - len(drop)}
    delta = np.array([0.0 if k in drop else (th[k] - theta0[k]) / 100.0 for k in cols])
    support = np.array([k not in drop for k in cols])
    return delta, rec, support


def project():
    out = read_out()
    assert out["status"] in ("bases complete", "projection complete"), out["status"]
    assert "fold" not in out, "a fold has started; the projection is not re-run"
    out["provenance_project_phase"] = git_provenance(R)
    T = template()
    theta0 = T["theta"]
    Bs = {name: load_basis(name) for name in REGISTERED}
    cols = Bs["tr0ts0"]["cols"]
    assert cols == Bs["rfam"]["cols"]
    prev = out.get("experiment_1")
    res = {"sets": {}, "per_set": {}, "corrections": (prev or {}).get("corrections", [])}
    for X in SETS:
        delta, res["sets"][X], support = set_delta(X, cols, theta0)
        res["per_set"][X] = {}
        for name, B in Bs.items():
            rng = np.random.default_rng(np.random.SeedSequence(
                [SEED, 3, list(SETS).index(X), list(REGISTERED).index(name)]))
            res["per_set"][X][name] = decompose(delta, B, rng, support)
            old = (prev or {}).get("per_set", {}).get(X, {}).get(name)
            if old is not None and not support.all() and not any(
                    c["set"] == X and c["basis"] == name for c in res["corrections"]):
                res["corrections"].append({
                    "date": "2026-10-12", "set": X, "basis": name,
                    "what": "control (c) and the post-hoc intrinsic control (c) rotated delta "
                            "into used columns off the shared support, where delta is "
                            "undefined; now restricted to the support. The projection, its "
                            "fraction, controls (a) and (b) and P_nt delta were valid and "
                            "are unchanged.",
                    "superseded": {"control_c_table_matched": old["control_c_table_matched"],
                                   "post_hoc_intrinsic": old.get("post_hoc_intrinsic")}})
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
    v = {f"{X}/{name}": {
        "nontrivial_fraction": d["fraction"]["nontrivial"],
        "control_a_mean": d["control_a_uniform_subspace"]["mean_fraction_approx"],
        "control_a_q99": d["control_a_uniform_subspace"]["q99_fraction"],
        "control_a_failed_as_required": d["control_a_uniform_subspace"]["true_exceeds_q99"],
        "control_c_percentile_rank": d["control_c_table_matched"]["true_percentile_rank"]}
        for X in SETS for name, d in res["per_set"][X].items()}
    ok = [k for k, x in v.items() if x["control_a_failed_as_required"]]
    res["verdict"] = {
        "by_set_and_basis": v, "control_a_failed_as_required_in": ok,
        "reading": (
            "REGISTERED CONTROL (a) DID NOT FAIL in any set or basis: every nontrivial "
            "fraction lies at or below a uniformly random subspace of the same dimension "
            "(squared fractions track dim/n for every component). By the registered rule the "
            "fraction is dimension counting, and 'differs by X% in directions none of the "
            "references can distinguish' is not supported as a property of these tables. "
            "Control (c), stated as a finding: see control_c_percentile_rank; a rank below 50 "
            "means the published difference is not enriched in the unidentified directions "
            "even within the tables it occupies. Per the user's registered ruling, Type B's "
            "claim rests on Experiment 2.") if not ok else
        f"control (a) failed as required in {ok}; see by_set_and_basis for the rest"}
    out["experiment_1"] = res
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


# ----------------------------------------------------------------------------------
# Experiment 2: fold perturbation
# ----------------------------------------------------------------------------------
FOLD = OUT / "fold"
GAP_STEPS = (0, 1, 2, 5, 10, 20, 50, 100)  # dcal
EXACT_TIE = 1e-9  # dcal; float round-off in c.(theta0 + delta), not a tolerance on energies


class GuardFailure(RuntimeError):
    """A loaded table is not the intended one, or an energy identity broke. Never repaired."""


def fold_rows():
    rows, _ = tr0_ts0_rows()
    assert len({r[2] for r in rows}) == len(rows), "TR0+TS0 sequences are not distinct"
    return rows


def build_points(out, B):
    """Every Experiment 2 point: direction, scale, theta_r (dcal ints) and delta (dcal)."""
    cols, used, NT, NI, A = B["cols"], B["used"], B["NT"], B["I"], B["A"]
    n = len(cols)
    T = template()
    theta0 = T["theta"]
    t0 = np.array([theta0[k] for k in cols], dtype=np.int64)
    dirs = []
    for i in range(N_NULL):
        g = np.random.default_rng(np.random.SeedSequence([SEED, 1, i])).standard_normal(len(used))
        v = NT @ (NT.T @ g)
        dirs.append(("null", str(i), v / np.linalg.norm(v)))
    own = {}
    for X in SETS:
        d, _, _ = set_delta(X, cols, theta0)
        v = NT @ (NT.T @ d[used])
        own[X] = float(np.linalg.norm(v))
        dirs.append(("published", X, v / own[X]))
    # the nontrivial basis lies in the four tables up to SVD round-off (max 1.2e-14 outside,
    # 2026-10-12); the contrast is built exactly in-table against that restriction,
    # re-orthonormalised, and its residual overlap with the full basis is recorded per point
    mask = np.array([cols[i][0] in NULL_TABLES for i in used])
    Qm, _ = np.linalg.qr(NT[mask])
    contrast_overlap = []
    for j in range(N_CONTRAST):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 2, j]))
        g = rng.standard_normal(int(mask.sum()))
        v = np.zeros(len(used))
        v[mask] = g - Qm @ (Qm.T @ g)
        if np.abs(NI.T @ v).max() != 0.0:
            raise GuardFailure("intrinsic directions have support in the four tables")
        v /= np.linalg.norm(v)
        contrast_overlap.append(float(np.abs(NT.T @ v).max()))
        dirs.append(("contrast", str(j), v))
    grid = out["experiment_2_grid"]["grid_kcal"]
    pts = [(kind, name, s, u) for kind, name, u in dirs for s in grid]
    pts += [("published_own", X, own[X], u) for kind, X, u in dirs if kind == "published"]
    gen = generating_set(T["special"])
    ex = Extractor(T["special"], 0, 0)
    G = build_matrix([ex.counts(s, d) for _, s, d in gen], cols)
    meta, TH, DL = [], np.zeros((len(pts), n), np.int64), np.zeros((len(pts), n))
    for p, (kind, name, s, u) in enumerate(pts):
        full = np.zeros(n)
        full[used] = u
        dl = 100.0 * s * full
        th = np.rint(t0 + dl).astype(np.int64)
        TH[p], DL[p] = th, dl
        ch = G @ (th - t0)
        top = np.argsort(-np.abs(ch), kind="stable")[:CARRY_ROWS]
        if ch[top[0]] == 0:
            raise GuardFailure(f"point {p}: no generating-set row sees the written change")
        ref_change = A @ (th - t0)[used]
        # the null arm's leak must be the rounding residual's alone: A.(theta_r - theta0) =
        # A.r + A.delta with |A.delta| <= TOL kcal/mol per kcal/mol of ||delta|| (the
        # registered C2 bar, scaled), on all 11,955 references; asserted, not only reported
        null_check = float(np.abs(A @ (dl[used] / 100.0)).max())
        if kind != "contrast" and null_check > TOL * s:
            raise GuardFailure(f"point {p}: {kind} {name} is not null on the references "
                               f"(max |A delta| = {null_check:.3g} kcal/mol at ||delta|| = {s})")
        meta.append({"p": p, "kind": kind, "name": name, "norm_kcal": s,
                     "contrast_max_overlap_with_null_basis": (
                         contrast_overlap[int(name)] if kind == "contrast" else None),
                     "max_abs_null_check_kcal": null_check,
                     "null_check_bar_kcal": TOL * s if kind != "contrast" else None,
                     "written_rel_error": float(np.linalg.norm(th - t0 - dl) / np.linalg.norm(dl)),
                     "n_entries_changed": int((th != t0).sum()),
                     "reference_change_dcal_max_abs": int(np.abs(ref_change).max()),
                     "carry": [[int(r), int((G[r] @ th)[0])] for r in top]})
    FOLD.mkdir(parents=True, exist_ok=True)
    np.savez(FOLD / "points.npz", theta_r=TH, delta_dcal=DL, t0=t0)
    (FOLD / "points.json").write_text(json.dumps(meta))
    return meta


def f1(a: str, b: str) -> float:
    pa, pb = set(_pairs(a)), set(_pairs(b))
    if not pa and not pb:
        return 1.0
    return 2 * len(pa & pb) / (len(pa) + len(pb))


def _pairs(db):
    st, out = [], []
    for i, c in enumerate(db):
        if c == "(":
            st.append(i)
        elif c == ")":
            out.append((st.pop(), i))
    return out


def _bpp(fc, n):
    return np.triu(np.array(fc.bpp())[1:n + 1, 1:n + 1], 1).astype(np.float32)


def _vec(cnt, idx):
    ks = [idx[k] for k in cnt]
    return np.array(ks, dtype=np.int64), np.array([cnt[k] for k in cnt], dtype=np.int64)


def worker(w, W, dst, limit=None, only=None, skip_load=False, stale_boltzmann=False):
    """Shard w of W: baseline once, then every point. Writes dst (JSON lines)."""
    T = template()
    theta0, special, lxc = T["theta"], T["special"], T["lxc"]
    B = load_basis("tr0ts0")
    cols = B["cols"]
    idx = {k: i for i, k in enumerate(cols)}
    Z = np.load(FOLD / "points.npz")
    meta = json.loads((FOLD / "points.json").read_text())
    t0 = Z["t0"]
    rows = fold_rows()[:limit]
    mine = [r for i, r in enumerate(rows) if i % W == w]
    ex = Extractor(special, 0, 0, lxc=lxc)
    gen = generating_set(special)
    md = reference_md()
    md_sub = reference_md()
    md_sub.uniq_ML = 1
    out = open(dst, "w")  # noqa: SIM115

    def emit(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    _rebuild()
    base = []
    for n, (tag, name, seq, db) in enumerate(mine):
        fc = RNA.fold_compound(seq, md)
        if n == 0:
            assert_turner2004_params(fc)
        s0, e0 = fc.mfe()
        fc.exp_params_rescale(e0)
        fc.pf()
        c0 = ex.counts(seq, s0)
        off0 = c0.pop(("_offset",), 0)
        cr = ex.counts(seq, db)
        cr.pop(("_offset",), 0)
        fs = RNA.fold_compound(seq, md_sub)
        gap = None
        for d in GAP_STEPS:
            es = sorted(int(round(s.energy * 100)) for s in fs.subopt(d))
            if len(es) >= 2:
                gap = es[1] - es[0]
                break
        base.append({"tag": tag, "name": name, "seq": seq, "db": db, "s0": s0,
                     "e0": int(round(e0 * 100)), "P0": _bpp(fc, len(seq)), "c0": _vec(c0, idx),
                     "off0": off0, "cr": _vec(cr, idx),
                     "ref_e0": int(round(fc.eval_structure(db) * 100))})
        base[-1]["gap"] = gap
    emit({"baseline": [{k: b[k] for k in ("tag", "name", "e0", "gap")}
                       | {"len": len(b["seq"]), "f1_ref": f1(b["s0"], b["db"]),
                          "ref_pairs": len(_pairs(b["db"]))} for b in base]})
    _rebuild()

    with tempfile.TemporaryDirectory() as t:
        for pm in meta:
            p = pm["p"]
            if only is not None and p not in only:
                continue
            th, dl = Z["theta_r"][p], Z["delta_dcal"][p]
            dth = th - t0
            full = dict(theta0) | {k: int(th[i]) for i, k in enumerate(cols)}
            path = write_par(full, pathlib.Path(t) / f"p{p}.par")
            ctx = (contextlib.nullcontext() if skip_load else
                   _stale_boltzmann(path) if stale_boltzmann else loaded(path))
            with ctx:
                first = RNA.fold_compound(mine[0][2], md)
                bad = params_match(first, full, 0, 0)
                if bad:
                    raise GuardFailure(f"point {p}: params_match {bad}")
                for r, want in pm["carry"]:
                    _, s, d = gen[r]
                    got = int(round(RNA.fold_compound(s, md).eval_structure(d) * 100))
                    if got != want:
                        raise GuardFailure(f"point {p}: carry row {r} evaluates to {got}, the "
                                           f"written table gives {want}")
                # the Boltzmann factors are a separate cache (turner_par V3): a partition
                # function constrained to the top carry row's one structure must give its
                # counts . theta_r free energy
                r, want = pm["carry"][0]
                _, s, d = gen[r]
                fcc = RNA.fold_compound(s, md)
                fcc.hc_add_from_db(d.replace(".", "x"), RNA.CONSTRAINT_DB_DEFAULT
                                   | RNA.CONSTRAINT_DB_ENFORCE_BP)
                _, ec = fcc.mfe()
                fcc.exp_params_rescale(ec)
                gc = int(round(fcc.pf()[1] * 100))
                if gc != want:
                    raise GuardFailure(f"point {p}: Boltzmann carry row {r} gives {gc}, the "
                                       f"written table gives {want}")
                rec = {"p": p, "changed": [], "genuine": [], "f1_base": [], "dbpp": [],
                       "f1_ref": [], "leak": []}
                for b in base:
                    seq, db = b["seq"], b["db"]
                    ki, kv = b["cr"]
                    leak = int(kv @ dth[ki])
                    e_ref = int(round(RNA.fold_compound(seq, md).eval_structure(db) * 100))
                    if e_ref != b["ref_e0"] + leak:
                        raise GuardFailure(f"point {p}: {b['name']} reference energy {e_ref} != "
                                           f"{b['ref_e0']} + {leak}")
                    fc = RNA.fold_compound(seq, md)
                    s1, e1 = fc.mfe()
                    fc.exp_params_rescale(e1)
                    fc.pf()
                    P1 = _bpp(fc, len(seq))
                    changed = s1 != b["s0"]
                    genuine = None
                    if not changed:
                        i0, v0 = b["c0"]
                        if int(v0 @ th[i0]) + b["off0"] != int(round(e1 * 100)):
                            raise GuardFailure(f"point {p}: {b['name']} MFE energy is not "
                                               f"counts . theta_r (unchanged structure)")
                    if changed:
                        c1 = ex.counts(seq, s1)
                        off1 = c1.pop(("_offset",), 0)
                        i1, v1 = _vec(c1, idx)
                        i0, v0 = b["c0"]
                        if int(v1 @ th[i1]) + off1 != int(round(e1 * 100)):
                            raise GuardFailure(f"point {p}: {b['name']} MFE energy is not "
                                               f"counts . theta_r")
                        ex1 = v1 @ (t0[i1] + dl[i1]) + off1
                        ex0 = v0 @ (t0[i0] + dl[i0]) + b["off0"]
                        genuine = bool(ex1 <= ex0 + EXACT_TIE)
                    rec["changed"].append(changed)
                    rec["genuine"].append(genuine)
                    rec["f1_base"].append(round(f1(s1, b["s0"]), 6))
                    rec["dbpp"].append(round(float(np.abs(P1 - b["P0"]).sum()) / len(seq), 8))
                    rec["f1_ref"].append(round(f1(s1, db), 6))
                    rec["leak"].append(leak)
            emit(rec)
    out.write(json.dumps({"done": True}) + "\n")
    out.close()
    return 0


@contextlib.contextmanager
def _stale_boltzmann(path):
    """Planted control only: load the file with the OLD, MFE-only sentinel, so the energy
    table is the file's and the Boltzmann factors are the previous model's."""
    fc = RNA.fold_compound("GGGGAAACCCC", reference_md())  # model A's Boltzmann table,
    fc.mfe()                                               # built from Turner 2004
    fc.pf()
    RNA.params_load(str(path))
    sent = reference_md()
    sent.temperature = 25.0
    RNA.fold_compound("GGGGAAACCCC", sent)
    try:
        yield
    finally:
        RNA.params_load_RNA_Turner2004()
        _rebuild()


def _launch_workers(W, dsts, extra=()):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="")
    return [subprocess.Popen([sys.executable, __file__, "--phase", "fold-worker", str(w), str(W),
                              str(d), *extra], env=env) for w, d in enumerate(dsts)]


def fold_smoke():
    """2 workers, 50 sequences, one null and one contrast point at 1.0 kcal/mol, then the
    planted stale-table run (load skipped) that the carry check must catch."""
    out = read_out()
    assert out["status"] == "projection complete", out["status"]
    B = load_basis("tr0ts0")
    meta = build_points(out, B)
    pick = [next(m["p"] for m in meta if m["kind"] == k and m["name"] == "0"
                 and m["norm_kcal"] == 1.0) for k in ("null", "contrast")]
    sm = FOLD / "smoke"
    sm.mkdir(parents=True, exist_ok=True)
    dsts = [sm / f"shard{w}.jsonl" for w in range(2)]
    procs = _launch_workers(2, dsts, ["--limit", "50", "--only", ",".join(map(str, pick))])
    rc = [p.wait() for p in procs]
    recs = [json.loads(ln) for d in dsts for ln in d.read_text().splitlines()]
    res = {"workers_exit": rc, "points": {}}
    for p in pick:
        rs = [r for r in recs if r.get("p") == p]
        ch = sum(sum(r["changed"]) for r in rs)
        gen = sum(sum(1 for g in r["genuine"] if g) for r in rs)
        leaks = [abs(x) for r in rs for x in r["leak"]]
        res["points"][str(p)] = {k: meta[p][k] for k in ("kind", "name", "norm_kcal",
                                                          "written_rel_error",
                                                          "max_abs_null_check_kcal",
                                                          "reference_change_dcal_max_abs")} | {
            "n_seq": sum(len(r["changed"]) for r in rs), "changed": ch, "genuine": gen,
            "leak_max_abs_dcal_on_these_refs": max(leaks) if leaks else None,
            "mean_dbpp": float(np.mean([x for r in rs for x in r["dbpp"]]))}
    planted = sm / "planted.jsonl"
    pr = subprocess.run([sys.executable, __file__, "--phase", "fold-worker", "0", "1",
                         str(planted), "--limit", "10", "--only", str(pick[0]), "--skip-load"],
                        capture_output=True, text=True,
                        env=dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=""))
    res["planted_stale_table"] = {"exit": pr.returncode,
                                  "caught_by_carry_check": pr.returncode != 0
                                  and "carry row" in pr.stderr,
                                  "stderr_tail": pr.stderr.strip().splitlines()[-1:]}
    pb = subprocess.run([sys.executable, __file__, "--phase", "fold-worker", "0", "1",
                         str(sm / "planted_boltzmann.jsonl"), "--limit", "10", "--only",
                         str(pick[0]), "--stale-boltzmann"], capture_output=True, text=True,
                        env=dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=""))
    res["planted_stale_boltzmann"] = {"exit": pb.returncode,
                                      "caught_by_boltzmann_carry": pb.returncode != 0
                                      and "Boltzmann carry row" in pb.stderr,
                                      "stderr_tail": pb.stderr.strip().splitlines()[-1:]}
    res["passes"] = (rc == [0, 0] and res["planted_stale_table"]["caught_by_carry_check"]
                     and res["planted_stale_boltzmann"]["caught_by_boltzmann_carry"]
                     and all(v["mean_dbpp"] > 0 for v in res["points"].values()))
    out["fold_smoke"] = res | {"provenance": git_provenance(R)}
    write(out)
    print(json.dumps(res, indent=1))
    return 0 if res["passes"] else 1


def fold_run():
    """The full Experiment 2 run. Launch only on the user's go, under setsid; completion is
    the recorded status, never process presence."""
    out = read_out()
    assert out["status"] == "projection complete", out["status"]
    assert out.get("fold_smoke", {}).get("passes"), "the smoke test has not passed"
    if os.getsid(0) != os.getpid():
        raise SystemExit("launch under setsid: the driver must survive the session")
    B = load_basis("tr0ts0")
    meta = build_points(out, B)
    full = FOLD / "full"
    full.mkdir(parents=True, exist_ok=True)
    dsts = [full / f"shard{w}.jsonl" for w in range(N_WORKERS)]
    procs = _launch_workers(N_WORKERS, dsts)
    out["fold"] = {"provenance": git_provenance(R), "n_points": len(meta),
                   "n_workers": N_WORKERS, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "load_at_start": os.getloadavg(), "driver_pid": os.getpid(),
                   "worker_pids": [p.pid for p in procs],
                   "points_sha256": {f: sha256(FOLD / f) for f in ("points.npz", "points.json")}}
    out["status"] = "fold RUNNING"
    write(out)
    rc = [p.wait() for p in procs]
    done = [d.read_text().rstrip().endswith('{"done": true}') for d in dsts]
    out = read_out()
    out["fold"].update({"workers_exit": rc, "shards_done": done,
                        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "shards_sha256": {d.name: sha256(d) for d in dsts}})
    out["status"] = ("fold complete (workers)" if rc == [0] * N_WORKERS and all(done)
                     else f"fold FAILED: exits {rc}")
    write(out)
    return 0 if out["status"].startswith("fold complete") else 1


def _boot_mean(x, rng, n=1000):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [None, None]
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def fold_analyse(src=None, n_workers=N_WORKERS, record=True):
    out = read_out()
    if record:
        assert out["status"] == "fold complete (workers)", out["status"]
    if json.loads(json.dumps(PRED, default=str)) != out["predictions"]:
        raise SystemExit("the artifact's predictions differ from PRED; nothing is analysed")
    src = FOLD / "full" if src is None else src
    meta = json.loads((FOLD / "points.json").read_text())
    shards = [[json.loads(ln) for ln in (src / f"shard{w}.jsonl").read_text()
               .splitlines()] for w in range(n_workers)]
    base = [b for s in shards for b in s[0]["baseline"]]
    L = np.array([b["len"] for b in base])
    terc = np.digitize(L, np.percentile(L, [100 / 3, 200 / 3]))
    pb = np.array([b["ref_pairs"] > 0 for b in base])
    gaps = [b["gap"] for b in base]
    gk = np.array([g for g in gaps if g is not None]) / 100.0
    res = {"n_sequences": len(base),
           "baseline_f1_ref_mean": float(np.mean([b["f1_ref"] for b in base])),
           "gap_kcal": {"median": float(np.median(gk)), "p25": float(np.percentile(gk, 25)),
                        "p75": float(np.percentile(gk, 75)),
                        "fraction_degenerate_gap0": float(np.mean([g == 0 for g in gaps])),
                        "fraction_above_1kcal": float(np.mean([g is None for g in gaps]))},
           "points": []}
    per = {}
    for pm in meta:
        p = pm["p"]
        rs = [next(r for r in s if r.get("p") == p) for s in shards]
        cat = {k: [x for r in rs for x in r[k]] for k in ("changed", "genuine", "f1_base",
                                                          "dbpp", "f1_ref", "leak")}
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 4, p]))
        ch = np.array(cat["changed"])
        gen = np.array([g is True for g in cat["genuine"]])
        leak = np.abs(np.array(cat["leak"])[pb]) / 100.0
        row = {k: pm[k] for k in ("p", "kind", "name", "norm_kcal", "written_rel_error",
                                  "n_entries_changed", "reference_change_dcal_max_abs")} | {
            "n_changed": int(ch.sum()), "n_genuine": int(gen.sum()),
            "genuine_share_of_changed": float(gen.sum() / ch.sum()) if ch.sum() else None,
            "changed_fraction": float(ch.mean()),
            "genuine_fraction": float(gen.mean()), "genuine_fraction_ci": _boot_mean(gen, rng),
            "f1_base_mean": float(np.mean(cat["f1_base"])),
            "f1_base_ci": _boot_mean(cat["f1_base"], rng),
            "dbpp_mean": float(np.mean(cat["dbpp"])), "dbpp_ci": _boot_mean(cat["dbpp"], rng),
            "f1_ref_mean": float(np.mean(cat["f1_ref"])),
            "leak_kcal": {"median": float(np.median(leak)), "p95": float(np.percentile(leak, 95)),
                          "max": float(leak.max())},
            "by_length_tercile": {str(t): {"changed_fraction": float(ch[terc == t].mean()),
                                           "dbpp_mean": float(np.array(cat["dbpp"])[terc == t]
                                                              .mean())} for t in range(3)}}
        res["points"].append(row)
        per[p] = (row, ch, gen)
    # floor: pooled over the null directions, per grid scale
    grid = out["experiment_2_grid"]["grid_kcal"]
    fl = []
    for s in grid:
        ps = [m["p"] for m in meta if m["kind"] == "null" and m["norm_kcal"] == s]
        c = sum(int(per[p][1].sum()) for p in ps)
        g = sum(int(per[p][2].sum()) for p in ps)
        fl.append({"norm_kcal": s, "changed": c, "genuine": g,
                   "rounding_share": (c - g) / c if c else None})
    floor = next((f["norm_kcal"] for f in fl if f["rounding_share"] is not None
                  and f["rounding_share"] <= FLOOR_SHARE), None)
    res["floor"] = {"rule": PRED["experiment_2_fold"]["floor_rule"], "by_scale": fl,
                    "floor_kcal": floor}
    claim = {}
    for row in res["points"]:
        if row["kind"] == "published_own":
            above = floor is not None and row["norm_kcal"] >= floor
            lo = row["genuine_fraction_ci"][0]
            claim[row["name"]] = {"norm_kcal": row["norm_kcal"], "above_floor": above,
                                  "genuine_fraction": row["genuine_fraction"],
                                  "ci": row["genuine_fraction_ci"],
                                  "claim_holds": bool(above and lo is not None and lo > 0)}
    res["claim"] = claim
    comp = []
    for s in grid:
        arm = {}
        for kind in ("null", "contrast"):
            rows = [r for r in res["points"] if r["kind"] == kind and r["norm_kcal"] == s]
            arm[kind] = {m: {"mean_over_directions": float(np.mean([r[m] for r in rows])),
                             "sd_over_directions": float(np.std([r[m] for r in rows], ddof=1)),
                             "n_directions": len(rows)}
                         for m in ("genuine_fraction", "f1_base_mean", "dbpp_mean")}
        comp.append({"norm_kcal": s} | arm | {
            "difference_null_minus_contrast": {
                m: arm["null"][m]["mean_over_directions"]
                - arm["contrast"][m]["mean_over_directions"]
                for m in ("genuine_fraction", "f1_base_mean", "dbpp_mean")}})
    res["comparison_null_vs_contrast"] = comp
    # amendment 1: null minus contrast, genuine changed-MFE fraction, bootstrapped over
    # sequences (shared by both arms) and directions (within each arm)
    rng_b = np.random.default_rng(np.random.SeedSequence([SEED, 5]))
    a1 = []
    for s in grid:
        Gn = np.stack([per[m["p"]][2] for m in meta if m["kind"] == "null"
                       and m["norm_kcal"] == s]).astype(float)
        Gc = np.stack([per[m["p"]][2] for m in meta if m["kind"] == "contrast"
                       and m["norm_kcal"] == s]).astype(float)
        nseq = Gn.shape[1]
        diffs = np.empty(BOOT_DRAWS)
        for b in range(BOOT_DRAWS):
            si = rng_b.integers(0, nseq, nseq)
            dn = rng_b.integers(0, len(Gn), len(Gn))
            dc = rng_b.integers(0, len(Gc), len(Gc))
            diffs[b] = Gn[dn][:, si].mean() - Gc[dc][:, si].mean()
        ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        chg = int(sum(per[m["p"]][1].sum() for m in meta if m["kind"] in ("null", "contrast")
                      and m["norm_kcal"] == s))
        a1.append({"norm_kcal": s, "difference": float(Gn.mean() - Gc.mean()), "ci95": ci,
                   "changed_in_either_arm": chg,
                   "eligible": floor is not None and s >= floor and chg > 0,
                   "side": "below" if ci[1] < 0 else "above" if ci[0] > 0 else "contains_0"})
    el = [x["side"] for x in a1 if x["eligible"]]
    overall = ("no eligible scale" if not el else "less" if all(x == "below" for x in el)
               else "more" if all(x == "above" for x in el)
               else "comparable" if all(x == "contains_0" for x in el) else "mixed")
    res["amendment_1_null_vs_contrast"] = {
        "rule": PRED[AMEND_ID]["amendment_1_less_reading_user"], "by_scale": a1,
        "overall": overall,
        "reading": (PRED[AMEND_ID]["amendment_1_less_reading_user"]["reading_if_it_fires_user"]
                    + "; the user's 'comparable' expectation is wrong") if overall == "less"
        else f"overall: {overall}; see by_scale"}
    if not record:
        return res
    out["experiment_2"] = res | {"provenance": git_provenance(R)}
    out["status"] = "fold analysed"
    write(out)
    print(json.dumps({"floor": res["floor"]["floor_kcal"], "claim": claim,
                      "gap": res["gap_kcal"]}, indent=1))
    return 0


def fold_mini():
    """Every point on 12 sequences with 2 workers, then the analysis as a dry run (nothing
    recorded): exercises build_points' assertions, every point's guards, and fold_analyse."""
    out = read_out()
    assert out["status"] == "projection complete", out["status"]
    meta = build_points(out, load_basis("tr0ts0"))
    mini = FOLD / "mini"
    mini.mkdir(parents=True, exist_ok=True)
    dsts = [mini / f"shard{w}.jsonl" for w in range(2)]
    rc = [p.wait() for p in _launch_workers(2, dsts, ["--limit", "12"])]
    if rc != [0, 0]:
        raise SystemExit(f"mini run workers exited {rc}")
    res = fold_analyse(src=mini, n_workers=2, record=False)
    print(json.dumps({"n_points": len(meta), "workers_exit": rc,
                      "floor": res["floor"]["floor_kcal"],
                      "amendment_1": res["amendment_1_null_vs_contrast"]["overall"],
                      "claim": res["claim"], "n_points_analysed": len(res["points"]),
                      "null_check_max_over_null_points": max(
                          m["max_abs_null_check_kcal"] / m["norm_kcal"] for m in meta
                          if m["kind"] != "contrast")}, indent=1))
    return 0


def fold_worker_cli(argv):
    w, W, dst = int(argv[0]), int(argv[1]), argv[2]
    rest = argv[3:]
    limit = int(rest[rest.index("--limit") + 1]) if "--limit" in rest else None
    only = ({int(x) for x in rest[rest.index("--only") + 1].split(",")}
            if "--only" in rest else None)
    return worker(w, W, dst, limit=limit, only=only, skip_load="--skip-load" in rest,
                  stale_boltzmann="--stale-boltzmann" in rest)


PHASES = {"preregister": preregister, "amend": amend, "bases": bases, "project": project,
          "fold-smoke": fold_smoke, "fold-mini": fold_mini, "fold-run": fold_run,
          "fold-analyse": fold_analyse}

if __name__ == "__main__":
    if sys.argv[1:3] == ["--phase", "fold-worker"]:
        sys.exit(fold_worker_cli(sys.argv[3:]))
    if len(sys.argv) != 3 or sys.argv[1] != "--phase" or sys.argv[2] not in PHASES:
        raise SystemExit(f"usage: turner_typeb.py --phase {{{','.join(PHASES)}}}")
    sys.exit(PHASES[sys.argv[2]]())
