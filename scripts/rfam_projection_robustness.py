#!/usr/bin/env python3
"""Is the nontrivial Rfam null space a property of RNA structure, or of the projection rule?

THE QUESTION

`scripts/turner_nullspace_rfam.py` (r2) projects each family's consensus structure
onto its seed sequences and finds a NONTRIVIAL data null space: directions inside
the used parameter entries that are neither intrinsic to the model nor coverage.
The projection drops pairs (noncanonical in that sequence, gapped, or closing a
too-short hairpin), so a reviewer can ask whether those directions are made by the
dropping rather than by RNA structure. Settle it.

THE DESIGN

1. For every row, the fraction of its family's consensus pairs that survive
   projection: kept pairs / consensus pairs. Rows from a family with no consensus
   pairs have no fraction and are counted separately (they contribute a zero count
   vector, so they cannot affect any rank).
2. LOW-DROP subset: rows keeping at least 90% of their consensus pairs.
3. Fewer rows can only remove constraints, so low-drop is NOT compared with the full
   set. It is compared with RANDOM subsamples of the full set at exactly its row
   count, one per recorded seed: the 8 SEEDS of scripts/pooled_alignment.py, read
   from that file's source so they are not retyped.
4. Nontrivial dimension, per subset and per configuration (dangles 0/2 x special_hp
   1/0), is   dim null(A_used) - rank(intrinsic directions restricted to the used
   entries),   where A_used is the subset's count matrix on the entries it uses. The
   intrinsic directions are exactly null on every valid structure, so they lie
   inside null(A_used) and the subtraction is exact. It is cross-checked against the
   residual-SVD count of r2's definition.

Everything else is r2's machinery unchanged: projection rules, ViennaRNA 2.7.2
pinned through thermo.vienna_ref, an exact reconstruction check on every row, the
null basis from a streamed QR of A (qr_null) with a Gram eigh as the second
dimension count.

PRE-REGISTERED (written into metrics.json before computing; Q1 and Q2 are the
user's, verbatim in substance)
Q1. The low-drop nontrivial dimension is within 2 SD of the matched random-subsample
    mean. If it is, the projection is not manufacturing degeneracies.
Q2. Both subset dimensions (low-drop, and every random subsample) exceed the
    full-set dimension, purely from having fewer rows.

CONTROLS (each must detect something)
K1. A random direction with its null component removed changes some row by more
    than 1e-6, in every subset.
K2. Every reported null basis vector (qr_null) changes no row by more than 1e-9, the
    same bar as r2, in every subset.
K3. Row assignment matters: across the low-drop subset and the 8 random
    reassignments of the same row count, the nontrivial dimension is not constant.
    If it were, the instrument could not tell subsets apart and Q1 would be vacuous.

Writes runs/2026-10-07_rfam_projection_robustness/metrics.json.
Run: .venv/bin/python scripts/rfam_projection_robustness.py
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import RNA  # noqa: E402

from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_nullspace import (  # noqa: E402
    Extractor,
    _reachable,
    analytic_directions,
    build_matrix,
    dump_params,
    theta_from_par,
)
from scripts.turner_nullspace_rfam import (  # noqa: E402
    SEED_FILE,
    consensus_pairs,
    families,
    gram_null,
    load_rows,
    project,
    qr_null,
)
from thermo.vienna_ref import assert_turner2004_params, reference_md  # noqa: E402

RUN_ID = "2026-10-07_rfam_projection_robustness"
OUT = R / "runs" / RUN_ID
LOW_DROP = 0.90
TOL_ROW = 1e-9
NONNULL_MIN = 1e-6


def recorded_seeds() -> tuple[int, ...]:
    """The SEEDS tuple of scripts/pooled_alignment.py, read from its source."""
    src = (R / "scripts" / "pooled_alignment.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SEEDS"
                                                for t in node.targets):
            seeds = tuple(ast.literal_eval(node.value))
            assert len(seeds) == 8, seeds
            return seeds
    raise LookupError("SEEDS not found in scripts/pooled_alignment.py")


PRED = {
    "Q1": "low-drop nontrivial dimension within 2 SD of the matched random-subsample mean "
          "(then the projection is not manufacturing degeneracies)",
    "Q2": "both subset dimensions (low-drop and every random subsample) exceed the full-set "
          "dimension, purely from having fewer rows",
}


def rows_with_fraction():
    """The r2 rows, each with kept / consensus pairs, in r2's row order."""
    stats = {"pairs_dropped_noncanonical": 0, "pairs_dropped_gap": 0,
             "pairs_dropped_short_hairpin": 0, "dropped_non_ACGU": 0}
    out = []
    for acc, ss, seqs in families(SEED_FILE):
        cp, _ = consensus_pairs(ss)
        for name, aligned in seqs:
            r = project(aligned, cp, stats)
            if r is None:
                continue
            seq, db = r
            frac = db.count("(") / len(cp) if cp else None
            out.append((acc, name, seq, db, frac, len(cp)))
    return out


def fraction_report(fr: np.ndarray, n_undefined: int) -> dict:
    edges = np.linspace(0.0, 1.0, 11)
    hist, _ = np.histogram(fr, bins=edges)
    return {
        "n_rows_with_fraction": int(fr.size),
        "n_rows_family_has_no_consensus_pairs": int(n_undefined),
        "mean": float(fr.mean()),
        "quantiles": {q: float(np.quantile(fr, float(q))) for q in
                      ("0.01", "0.05", "0.1", "0.25", "0.5", "0.75", "0.9")},
        "histogram_bins_0.1": {f"{edges[i]:.1f}-{edges[i + 1]:.1f}": int(h)
                               for i, h in enumerate(hist)},
        "n_exactly_1": int((fr == 1.0).sum()),
        "n_at_least_0.9": int((fr >= LOW_DROP).sum()),
    }


def subset_dims(M, idx, cols, an, rng):
    A_all = M[idx]
    nnz = np.asarray((A_all != 0).sum(axis=0)).ravel()
    used = np.nonzero(nnz)[0]
    A = A_all[:, used].tocsr()
    Nq, inq = qr_null(A)
    Ng, _ = gram_null(A)
    an_u = [g[used] for g in an.values()]
    an_u = [g for g in an_u if np.abs(g).max() > 0]
    an_change = max(float(np.abs(A @ g).max()) for g in an_u)
    Au = np.stack(an_u, axis=1)
    rank_an = int(np.linalg.matrix_rank(Au))
    # r2's definition: residual of the null basis after removing the intrinsic span
    Qa, _ = np.linalg.qr(Au)
    sv = np.linalg.svd(Nq - Qa @ (Qa.T @ Nq), compute_uv=False) if Nq.size else np.array([])
    nontrivial_resid = int((sv > 1e-8).sum())
    nontrivial = int(Nq.shape[1]) - rank_an
    g = rng.standard_normal(len(used))
    g -= Nq @ (Nq.T @ g)
    k1 = float(np.abs(A @ g).max())
    k2 = float(np.abs(A @ Nq).max()) if Nq.size else 0.0
    return {
        "rows": int(len(idx)), "distinct_rows": None, "used_columns": int(len(used)),
        "coverage_dim": int(len(cols) - len(used)),
        "null_dim_within_used": {"qr_svd": int(Nq.shape[1]), "eigh_gram": int(Ng.shape[1]),
                                 "agree": int(Nq.shape[1]) == int(Ng.shape[1]), "qr": inq},
        "intrinsic_restricted": {"rank": rank_an, "max_abs_row_change": an_change,
                                 "all_null": an_change <= TOL_ROW},
        "nontrivial_dim": nontrivial,
        "nontrivial_dim_residual_svd": nontrivial_resid,
        "nontrivial_definitions_agree": nontrivial == nontrivial_resid,
        "K1_random_nonnull": {"max_abs_change": k1, "detected": k1 > NONNULL_MIN},
        "K2_basis_row_bar": {"max_abs_change": k2, "passes": k2 <= TOL_ROW},
    }


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=str))


def main():
    t0 = time.time()
    seeds = recorded_seeds()
    out = {"run_id": RUN_ID, "status": "PRE-REGISTERED, not yet run", "predictions": PRED,
           "low_drop_threshold": LOW_DROP, "seeds": list(seeds),
           "seeds_source": "SEEDS in scripts/pooled_alignment.py, read from source",
           **git_provenance(R), "viennarna_version": RNA.__version__}
    write(out)

    rows = rows_with_fraction()
    r2_rows, _ = load_rows()
    same = [(a, n, s, d) for a, n, s, d, _, _ in rows] == r2_rows
    assert same, "row set differs from turner_nullspace_rfam.load_rows()"
    fr_all = [f for *_, f, _ in rows]
    defined = np.array([f is not None for f in fr_all])
    fr = np.array([f for f in fr_all if f is not None])
    low = np.nonzero(np.array([f is not None and f >= LOW_DROP for f in fr_all]))[0]
    out["rows"] = {"full": len(rows), "matches_r2_rows": same,
                   "low_drop": int(len(low)),
                   "low_drop_fraction_of_full": round(len(low) / len(rows), 4)}
    out["retained_fraction"] = fraction_report(fr, int((~defined).sum()))
    subsets = {"full": np.arange(len(rows)), "low_drop": low}
    for s in seeds:
        rng = np.random.default_rng(s)
        subsets[f"random_{s}"] = np.sort(rng.choice(len(rows), size=len(low), replace=False))
    write(out)

    P = theta_from_par(dump_params())
    out["configs"] = {}
    for dangles in (0, 2):
        for special_hp in (1, 0):
            name = f"d{dangles}_special_hp{special_hp}"
            md = reference_md()
            md.dangles, md.special_hp = dangles, special_hp
            ex = Extractor(P["special"], dangles, special_hp, lxc=P["lxc"])
            counts, fails, checked = [], 0, False
            for _, _, seq, db, _, _ in rows:
                fc = RNA.fold_compound(seq, md)
                if not checked:
                    assert_turner2004_params(fc)
                    checked = True
                e = int(round(fc.eval_structure(db) * 100))
                c = ex.counts(seq, db)
                off = c.pop(("_offset",), 0)
                if sum(P["theta"][k] * v for k, v in c.items()) + off != e:
                    fails += 1
                counts.append(c)
            res = {"reconstruction": {"n_checked": len(rows), "n_failed": fails,
                                      "passes": fails == 0}}
            out["configs"][name] = res
            if fails:
                out["status"] = f"STOPPED: reconstruction failed in {name}"
                write(out)
                return 1
            cols = [k for k in sorted(P["theta"], key=repr) if _reachable(k, dangles, special_hp)]
            M = build_matrix(counts, cols).tocsr()
            an = analytic_directions(cols, dangles, special_hp)
            res["subsets"] = {}
            krng = np.random.default_rng(seeds[0])
            for sname, idx in subsets.items():
                t1 = time.time()
                d = subset_dims(M, idx, cols, an, krng)
                d["distinct_rows"] = len({(rows[i][2], rows[i][3]) for i in idx})
                d["seconds"] = round(time.time() - t1, 1)
                res["subsets"][sname] = d
                write(out)
            rnd = np.array([res["subsets"][f"random_{s}"]["nontrivial_dim"] for s in seeds])
            lowd = res["subsets"]["low_drop"]["nontrivial_dim"]
            full = res["subsets"]["full"]["nontrivial_dim"]
            mean, sd = float(rnd.mean()), float(rnd.std(ddof=1))
            res["summary"] = {
                "nontrivial_full": full, "nontrivial_low_drop": lowd,
                "nontrivial_random": rnd.tolist(), "random_mean": mean, "random_sd": sd,
                "low_minus_mean_in_sd": (lowd - mean) / sd if sd > 0 else None,
                "Q1_within_2sd": abs(lowd - mean) <= 2 * sd,
                "Q2_low_exceeds_full": lowd > full,
                "Q2_every_random_exceeds_full": bool((rnd > full).all()),
                "K3_assignment_changes_result": len(set(rnd.tolist()) | {lowd}) > 1,
            }
            write(out)

    cf = out["configs"]
    s_ = {k: c["summary"] for k, c in cf.items()}
    allsub = [d for c in cf.values() for d in c["subsets"].values()]
    out["verdicts"] = {
        "Q1": {"verdict": "confirmed" if all(v["Q1_within_2sd"] for v in s_.values())
               else "refuted",
               "per_config": {k: {"low_drop": v["nontrivial_low_drop"],
                                  "random_mean": v["random_mean"], "random_sd": v["random_sd"],
                                  "z": v["low_minus_mean_in_sd"]} for k, v in s_.items()}},
        "Q2": {"verdict": "confirmed" if all(v["Q2_low_exceeds_full"]
                                             and v["Q2_every_random_exceeds_full"]
                                             for v in s_.values()) else "refuted",
               "per_config": {k: {"full": v["nontrivial_full"],
                                  "low_drop": v["nontrivial_low_drop"],
                                  "random": v["nontrivial_random"]} for k, v in s_.items()}},
        "controls": {
            "K1_all_detected": all(d["K1_random_nonnull"]["detected"] for d in allsub),
            "K2_all_pass": all(d["K2_basis_row_bar"]["passes"] for d in allsub),
            "K3_all_detected": all(v["K3_assignment_changes_result"] for v in s_.values()),
            "dimension_methods_agree": all(d["null_dim_within_used"]["agree"] for d in allsub),
            "nontrivial_definitions_agree": all(d["nontrivial_definitions_agree"]
                                                for d in allsub),
            "intrinsic_null_on_every_subset": all(d["intrinsic_restricted"]["all_null"]
                                                  for d in allsub),
        },
    }
    out["status"] = "complete"
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    write(out)
    print(json.dumps(out["verdicts"], indent=1))
    return 0


# ==================================================================================
# Family-matched control: python scripts/rfam_projection_robustness.py --family-matched
# ==================================================================================
FM_RUN_ID = "2026-10-08_rfam_projection_family_matched"
FM_OUT = R / "runs" / FM_RUN_ID
PRIOR = R / "runs" / RUN_ID / "metrics.json"

FM_DOC = {
    "question": "The low-drop subset (rows keeping >= 90% of their consensus pairs) has 199 "
                "nontrivial null directions against 287 +- 5.4 for random subsamples of the "
                "same row count (runs/2026-10-07_rfam_projection_robustness). The low-drop "
                "subset differs from a random one in two ways at once: its rows are projected "
                "with higher fidelity, AND its family composition is different (families whose "
                "consensus fits their members poorly are under-represented). This control "
                "separates the two.",
    "design": "For each recorded seed, draw a random subsample of the FULL set matched to the "
              "low-drop subset on total row count AND on the number of rows from each Rfam "
              "family: for every family f, draw n_f(low-drop) rows uniformly without "
              "replacement from all of f's rows. Family composition is then identical to "
              "low-drop's; only within-family selection is random, so it mixes high- and "
              "low-fidelity rows of the same families.",
    "split": "plain random - family-matched = the part of the gap due to FAMILY COMPOSITION; "
             "family-matched - low-drop = the part due to PROJECTION FIDELITY within families. "
             "Shares are reported as fractions of plain random - low-drop.",
    "plain_random_source": "the 8 plain-random subsamples of runs/2026-10-07_rfam_projection_"
                           "robustness/metrics.json (same seeds, same row count), read from "
                           "that artifact; its git sha is recorded and checked. The low-drop "
                           "subset is recomputed here and must reproduce that artifact exactly.",
    "forced_rows": "in a family where every row is low-drop, the matched draw must take all of "
                   "them, so the matched subsample coincides with low-drop there; the number "
                   "of such families and rows is reported, since they cannot contribute to the "
                   "fidelity contrast.",
}
FM_PRED = {
    "F1": "Projection fidelity contributes: the family-matched mean nontrivial dimension is more "
          "than 2 SD (the family-matched SD) above the low-drop value.",
    "F2": "Family composition contributes: the plain-random mean exceeds the family-matched "
          "mean by more than 2 x sqrt(SD_plain^2 + SD_matched^2).",
}


def matched_subsample(fams, low, seed):
    rng = np.random.default_rng(seed)
    by_fam, need = {}, {}
    for i, f in enumerate(fams):
        by_fam.setdefault(f, []).append(i)
    for i in low:
        need[fams[i]] = need.get(fams[i], 0) + 1
    out = []
    for f in sorted(need):
        out.extend(rng.choice(by_fam[f], size=need[f], replace=False).tolist())
    return np.sort(np.array(out)), by_fam, need


def main_family_matched():
    from scripts.turner_nullspace_rfam import count_matrix

    t0 = time.time()
    seeds = recorded_seeds()
    prior = json.loads(PRIOR.read_text())
    out = {"run_id": FM_RUN_ID, "status": "PRE-REGISTERED, not yet run", **FM_DOC,
           "predictions": FM_PRED, "seeds": list(seeds), "low_drop_threshold": LOW_DROP,
           "prior_artifact": {"path": str(PRIOR.relative_to(R)), "git_sha": prior["git_sha"],
                              "git_dirty_excluding_runs": prior["git_dirty_excluding_runs"]},
           **git_provenance(R), "viennarna_version": RNA.__version__}
    FM_OUT.mkdir(parents=True, exist_ok=True)

    def save():
        (FM_OUT / "metrics.json").write_text(json.dumps(out, indent=2, sort_keys=True,
                                                        default=str))
    save()
    assert prior["status"] == "complete" and prior["seeds"] == list(seeds)
    rows = rows_with_fraction()
    fams = [r[0] for r in rows]
    low = np.nonzero(np.array([r[4] is not None and r[4] >= LOW_DROP for r in rows]))[0]
    assert len(low) == prior["rows"]["low_drop"]
    subsets = {"low_drop": low}
    forced = None
    for sd in seeds:
        idx, by_fam, need = matched_subsample(fams, low, sd)
        assert len(idx) == len(low)
        subsets[f"family_matched_{sd}"] = idx
        if forced is None:
            ff = [f for f in need if need[f] == len(by_fam[f])]
            forced = {"families_in_low_drop": len(need),
                      "families_fully_low_drop": len(ff),
                      "rows_in_fully_low_drop_families": int(sum(need[f] for f in ff)),
                      "rows_free_to_vary": int(len(low) - sum(need[f] for f in ff))}
    out["forced"] = forced
    lowset = set(low.tolist())
    out["overlap_with_low_drop"] = {k: int(len(lowset & set(v.tolist())))
                                    for k, v in subsets.items() if k != "low_drop"}
    save()

    P = theta_from_par(dump_params())
    rows4 = [(a, n, s_, d) for a, n, s_, d, _, _ in rows]
    out["configs"] = {}
    for dangles in (0, 2):
        for special_hp in (1, 0):
            name = f"d{dangles}_special_hp{special_hp}"
            M, cols, fails = count_matrix(dangles, special_hp, P, rows4)
            res = {"reconstruction_failures": int(fails), "subsets": {}}
            out["configs"][name] = res
            if fails:
                out["status"] = f"STOPPED: reconstruction failed in {name}"
                save()
                return 1
            an = analytic_directions(cols, dangles, special_hp)
            krng = np.random.default_rng(seeds[0])
            for sname, idx in subsets.items():
                t1 = time.time()
                dd = subset_dims(M.tocsr(), idx, cols, an, krng)
                dd["distinct_rows"] = len({(rows[i][2], rows[i][3]) for i in idx})
                dd["seconds"] = round(time.time() - t1, 1)
                res["subsets"][sname] = dd
                save()
            pr = prior["configs"][name]["summary"]
            fm = np.array([res["subsets"][f"family_matched_{x}"]["nontrivial_dim"]
                           for x in seeds])
            fm_used = np.array([res["subsets"][f"family_matched_{x}"]["used_columns"]
                                for x in seeds])
            pr_used = np.array([prior["configs"][name]["subsets"][f"random_{x}"]["used_columns"]
                                for x in seeds])
            lowd = res["subsets"]["low_drop"]["nontrivial_dim"]
            low_used = res["subsets"]["low_drop"]["used_columns"]
            rmean, rsd = pr["random_mean"], pr["random_sd"]
            fmean, fsd = float(fm.mean()), float(fm.std(ddof=1))
            gap = rmean - lowd
            res["summary"] = {
                "low_drop_reproduces_prior": lowd == pr["nontrivial_low_drop"],
                "nontrivial_low_drop": lowd, "nontrivial_family_matched": fm.tolist(),
                "family_matched_mean": fmean, "family_matched_sd": fsd,
                "plain_random_mean": rmean, "plain_random_sd": rsd,
                "share_projection_fidelity": (fmean - lowd) / gap if gap else None,
                "share_family_composition": (rmean - fmean) / gap if gap else None,
                "F1_fm_above_low_by_sd": (fmean - lowd) / fsd if fsd > 0 else None,
                "F1": fsd > 0 and fmean - lowd > 2 * fsd,
                "F2_z": (rmean - fmean) / float(np.hypot(rsd, fsd)),
                "F2": rmean - fmean > 2 * float(np.hypot(rsd, fsd)),
                "used_columns": {"low_drop": low_used, "family_matched_mean":
                                 float(fm_used.mean()), "plain_random_mean":
                                 float(pr_used.mean())},
                "K3_assignment_changes_result": len(set(fm.tolist()) | {lowd}) > 1,
            }
            save()
    cf = out["configs"]
    allsub = [d for c in cf.values() for d in c["subsets"].values()]
    out["verdicts"] = {
        "F1": "confirmed" if all(c["summary"]["F1"] for c in cf.values()) else "refuted",
        "F2": "confirmed" if all(c["summary"]["F2"] for c in cf.values()) else "refuted",
        "split": {k: {kk: c["summary"][kk] for kk in (
            "nontrivial_low_drop", "family_matched_mean", "family_matched_sd",
            "plain_random_mean", "plain_random_sd", "share_projection_fidelity",
            "share_family_composition")} for k, c in cf.items()},
        "controls": {
            "low_drop_reproduces_prior": all(c["summary"]["low_drop_reproduces_prior"]
                                             for c in cf.values()),
            "K1_all_detected": all(d["K1_random_nonnull"]["detected"] for d in allsub),
            "K2_all_pass": all(d["K2_basis_row_bar"]["passes"] for d in allsub),
            "K3_all_detected": all(c["summary"]["K3_assignment_changes_result"]
                                   for c in cf.values()),
            "dimension_methods_agree": all(d["null_dim_within_used"]["agree"] for d in allsub),
            "nontrivial_definitions_agree": all(d["nontrivial_definitions_agree"]
                                                for d in allsub),
        },
    }
    out["status"] = "complete"
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    save()
    print(json.dumps(out["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    if "--family-matched" in sys.argv:
        sys.exit(main_family_matched())
    sys.exit(main())
