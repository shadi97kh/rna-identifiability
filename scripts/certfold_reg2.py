#!/usr/bin/env python3
"""CERTIFIED FOLDING, REGISTRATION 2: stability under whole-table changes (C) and which
measurements would certify the most pairs (D). The authors' design of 2026-10-15.

C  For each TS0 Turner 2004 MFE pair, kept(p, X) = p is in the MFE under table X (same pinned
   model). X primary: the three whole published tables as shipped; X secondary: theta0 plus only
   the data-VISIBLE part of each difference (delta_X - Pi delta_X on B_four, INF entries set to 0),
   where certification gives no guarantee; tertiary, consistency only: the released MXfold2's TS0
   predictions. Retention = the fraction of the three tables that keep the pair.
   R4 (directional): among MFE pairs with p >= 0.99, mean retention of pairs certified at
   lambda = 1 minus uncertified ones, within the length strata, pooled by pair count, stratified
   bootstrap (seed tag 60). Passes iff the interval lies entirely above 0. Same for the secondary.
   R5 (no direction): matched coverage with kept in place of correct, rho_env against p, the
   nominal gap and the length score, per primary table, paired bootstrap (seed tag 61).
D  Ranking on TR0 only (the pilot's 300 sequences): for each MFE pair uncertain at lambda = 1, the
   witness is the structure the constrained fold returns; count per reached entry the uncertain
   pairs whose witness's differing loops read it; rank, ties by entry key; frozen in the
   registration. Evaluation on TS0: labels at lambda = 1 with u_e = 0 for the top K entries
   (K = 10, 50), against K random reached entries (20 draws, seed tag 62). R6: the top-K gain in
   the certified-present fraction exceeds the 95th percentile of the random-K gains, each K.
   Before TS0, G2 on the u_e = 0 sets.

PHASES, each refusing unless its predecessor's record is committed:
  rank-entries      D's ranking on TR0 (before the registration; no TS0 sequence folded)
  preregister2      PRED2 with the frozen ranking, the reading rules and the seed tags
  gate-g2-measured  G2 (n <= 20 enumeration) on the u_e = 0 sets, after PRED2, before TS0
  fold-tables       C: TS0 under the six tables (workers)
  analyse-c         R4 (primary, secondary), R5, the tertiary consistency
  measured-labels   D: TS0 labels with the measured sets (workers)
  analyse-d         R6, the gain by stratum, the top entries as motifs
Run in .venv-certfold. Records go to runs/2026-10-15_certfold_registration2/metrics.json.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold_run as cr  # noqa: E402
from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402

cf = cr.cf
ME = "scripts/certfold_reg2.py"
RUN_ID2 = "2026-10-15_certfold_registration2"
OUT2 = R / "runs" / RUN_ID2
SHARDS2 = OUT2 / "shards"
TAG_R4, TAG_R5, TAG_RANDOM_K = 60, 61, 62
K_VALUES = (10, 50)
N_RANDOM = 20
N_BOOT = 10000
WORKERS = 12
TABLES = ("Turner1999", "Andronescu2007", "Langdon2018")
MX_PRED = "runs/2026-10-07_mxfold2_ts0_inference/pred_A_cpu.txt"
REGISTERED2 = "PRE-REGISTERED (registration 2): nothing folded under a whole table on TS0 yet"


# ----------------------------------------------------------------------------------
# records
# ----------------------------------------------------------------------------------
def read_out2() -> dict:
    p = OUT2 / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {"run_id": RUN_ID2}


def write2(out: dict) -> None:
    OUT2.mkdir(parents=True, exist_ok=True)
    tmp = OUT2 / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT2 / "metrics.json")


def committed2(unmodified: bool = True) -> dict:
    rel = str((OUT2 / "metrics.json").relative_to(R))
    if cf._git("ls-files", "--error-unmatch", rel).returncode != 0:
        raise SystemExit(f"{rel} is not tracked: commit it first")
    if unmodified and cf._git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"{rel} has uncommitted changes: commit it first")
    return json.loads(cf._git("show", f"HEAD:{rel}").stdout)


def clean() -> dict:
    prov = git_provenance(R)
    if prov["git_dirty_excluding_runs"]:
        raise SystemExit(f"commit first: {prov.get('git_dirty_paths_outside_runs')}")
    return prov


def canon_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), sort_keys=True,
                                     default=str).encode()).hexdigest()


def drive(phase: str, n_w: int, extra: list[str] | None = None) -> list[int]:
    """Run `phase` in n_w worker processes (nice 10, single-threaded); stop all on a failure."""
    SHARDS2.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="")
    procs = [subprocess.Popen(
        ["nice", "-n", "10", sys.executable, str(R / ME), "--phase", phase, "--shard", str(w),
         "--of", str(n_w)] + (extra or []), cwd=R, env=env,
        stdout=(SHARDS2 / f"{phase}_{w}.log").open("w"), stderr=subprocess.STDOUT)
        for w in range(n_w)]
    while True:
        codes = [p.poll() for p in procs]
        if any(c not in (None, 0) for c in codes):
            for q in procs:
                if q.poll() is None:
                    q.terminate()
            codes = [q.wait() for q in procs]
            break
        if all(c == 0 for c in codes):
            break
        time.sleep(15)
    if any(codes):
        raise GuardFailure(f"{phase} workers exited {codes}; shards kept at {SHARDS2}")
    return [p.pid for p in procs]


def shard_args() -> tuple[int, int]:
    return (int(sys.argv[sys.argv.index("--shard") + 1]),
            int(sys.argv[sys.argv.index("--of") + 1]))


# ----------------------------------------------------------------------------------
# D: the witnesses' entries
# ----------------------------------------------------------------------------------
def _codes(seq: str) -> list[int]:
    return [0] + [cf.BCODE[c] for c in seq] + [0]


def witness_entries(lc, B: dict, seq: str, star: str, u_env: np.ndarray) -> list[dict]:
    """For each MFE pair of seq: certified or not at lambda = 1 under the envelope with these
    u_e, and, if uncertain, the reached entries read by the differing interior loops of the
    witness (the structure lacking the pair that the constrained fold returns)."""
    S = _codes(seq)
    F = cf.Folder(lc, seq, star, cf.Weights(u_env, B["cols"], 1.0))
    F.set_radius(1.0)
    pt = cf.pair_table(star)
    loops_star = cf.star_interior_loops(lc, seq, star)
    idx = {k: n for n, k in enumerate(B["cols"])}
    out = []
    for i in range(1, len(seq) + 1):
        j = int(pt[i])
        if j <= i:
            continue
        s, x, _e0 = F.without(i, j)
        rec = {"pair": [i, j], "certified": bool(x - F.eprime_star > 0)}
        if not rec["certified"]:
            diff = loops_star ^ cf.star_interior_loops(lc, seq, s)
            ents = set()
            for (a, b, c, d) in diff:
                for k in cf.il_keys(S, a, b, c, d):
                    if k in idx and u_env[idx[k]] > 0:
                        ents.add(k)
            rec["entries"] = sorted(map(str, ents))
        out.append(rec)
    return out


def rank_entries_worker() -> int:
    w, n_w = shard_args()
    B = cf.load_tr0_basis()
    P = cf.template()
    lc = cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    rows, _ = cf.tr0_rows()
    seqs, _bounds = cf.pilot_sequences(rows)
    import RNA

    from thermo.vienna_ref import reference_md
    with open(SHARDS2 / f"rank_{w}.jsonl", "w") as fh:
        for n, (name, seq, stratum) in enumerate(seqs):
            if n % n_w != w:
                continue
            star = RNA.fold_compound(seq, reference_md()).mfe()[0]
            fh.write(json.dumps({"name": name, "n_nt": len(seq), "stratum": stratum,
                                 "pairs": witness_entries(lc, B, seq, star, B["u_env"])}) + "\n")
            fh.flush()
    return 0


def rank_entries() -> int:
    """D's ranking, on the TR0 pilot's 300 sequences; before the registration, no TS0 fold."""
    prov = clean()
    out = read_out2()
    if "ranking" in out:
        raise SystemExit("the ranking exists")
    t0 = time.time()
    pids = drive("rank-entries-worker", 8)
    recs = []
    for w in range(8):
        recs += [json.loads(ln) for ln in (SHARDS2 / f"rank_{w}.jsonl").read_text().splitlines()
                 if ln.strip()]
    if len(recs) != 300:
        raise GuardFailure(f"{len(recs)} of the pilot's 300 sequences ranked")
    count = collections.Counter()
    n_unc = n_pairs = 0
    for r in recs:
        for q in r["pairs"]:
            n_pairs += 1
            if not q["certified"]:
                n_unc += 1
                count.update(q["entries"])
    ranked = sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))
    out["ranking"] = {
        "label": "D's ranking, TR0 only (the pilot's 300 sequences), frozen in registration 2",
        "rule": "for each MFE pair uncertain at lambda = 1 under the envelope, the witness is the "
                "structure lacking the pair that the constrained fold returns at lambda = 1; per "
                "reached entry, the number of uncertain pairs whose witness's differing interior "
                "loops (symmetric difference with s*) read it; ranked by that count, ties by the "
                "entry key as a string",
        "n_sequences": len(recs), "n_pairs": n_pairs, "n_uncertain": n_unc,
        "n_entries_with_a_count": len(ranked),
        "ranked": [[k, c] for k, c in ranked],
        "ranked_sha256": canon_sha([[k, c] for k, c in ranked]),
        "files_sha256": {f"rank_{w}.jsonl": cf._sha(SHARDS2 / f"rank_{w}.jsonl")
                         for w in range(8)},
        "pids": pids, "seconds": round(time.time() - t0, 1), "provenance": prov,
    }
    out["status"] = "ranking done"
    write2(out)
    print(json.dumps({k: out["ranking"][k] for k in ("n_pairs", "n_uncertain",
                                                    "n_entries_with_a_count")}, indent=1))
    print("top 10:", out["ranking"]["ranked"][:10])
    return 0


# ----------------------------------------------------------------------------------
# the registration
# ----------------------------------------------------------------------------------
PRED2 = {
    "version": "certified folding, registration 2",
    "C_outcome_user": "For each TS0 Turner 2004 MFE pair, kept(p, X) = p is in the MFE under X, "
                      "same pinned model. X, primary: the three whole published tables as shipped. "
                      "X, secondary: theta0 plus only the data-VISIBLE part of each difference "
                      "(delta_X - Pi delta_X, INF entries set to 0 as before), where certification "
                      "gives no guarantee. Tertiary (consistency only): the released MXfold2's TS0 "
                      "predictions from runs/2026-10-07_mxfold2_ts0_inference. Retention = the "
                      "fraction of the three tables that keep the pair.",
    "R4_user": "R4 (directional): among MFE pairs with p >= 0.99, mean retention of pairs "
               "certified at lambda = 1 minus uncertified ones, within the ten length strata, "
               "pooled by pair count, stratified bootstrap 10,000 (new seed tag). Passes iff the "
               "interval lies entirely above 0. Run the same statistic for the secondary outcome.",
    "R5_user": "R5 (no direction): matched-coverage comparison with \"kept\" in place of "
               "\"correct\": rho_env against p, the nominal gap and the length score, k = the "
               "number of p >= 0.99 pairs, paired bootstrap 10,000 (new seed tag).",
    "reading_rules_user": [
        "R4 primary passes: \"pairs certified against directions the data cannot see are also "
        "more often kept when the whole table is replaced, beyond p and length.\"",
        "R4 primary fails: \"certification does not predict stability under whole-table changes "
        "beyond p and length.\"",
        "If the primary passes but the secondary does not: say the effect comes from the part "
        "the set covers by construction.",
        "R5 is read by rule 3. Only if rho_env beats p with the interval excluding 0 may the paper "
        "say \"the certified radius predicts which pairs survive a change of parameter table "
        "better than base-pair probability does\".",
        "The tertiary outcome is \"consistent\" or \"not consistent\", nothing more."],
    "D_user": "Which measurements would certify the most pairs. Ranking, TR0 only (the pilot's "
              "300 sequences): for each MFE pair uncertain at lambda = 1, the witness is the "
              "structure lacking the pair that the constrained fold returns at lambda = 1. Count, "
              "per reached entry, the uncertain pairs whose witness's differing loops read it. "
              "Rank; ties by entry key. Freeze the ranked list in the commit. Evaluation, TS0: "
              "labels at lambda = 1 with u_e = 0 for the top K entries (the set with those "
              "entries measured exactly; a subset of E, so still sound), K = 10 and 50. Control: "
              "K random reached entries, 20 seeded draws. R6: the top-K gain in the "
              "certified-present fraction exceeds the 95th percentile of the random-K gains, for "
              "each K. Report the gain by stratum, and in the supplement the top entries as "
              "motifs (closing pairs and loop bases). Before TS0: run G2 on the u_e = 0 sets "
              "(n <= 20 enumeration). Wording: \"if these motifs were measured exactly\"; say "
              "exact measurement is an idealization.",
    "seeds_claude": {"R4_stratified_bootstrap": [cf.SEED, TAG_R4],
                     "R5_paired_bootstrap": [cf.SEED, TAG_R5],
                     "D_random_K_draws": [cf.SEED, TAG_RANDOM_K],
                     "tags_in_use_before": [1, 2, 3, 10, 11, 12, 20, 21, 30, 31, 40, 41, 50, 70]},
    "C_tables_claude": "Primary: each table loaded through ViennaRNA's own shipped loader "
                       "(params_load_RNA_Turner1999, _Andronescu2007, _Langdon2018), with the "
                       "fold guards (params_match on the first compound, generating-set carry "
                       "rows, and counts.theta + the asymmetry correction equal to ViennaRNA's MFE "
                       "energy on every fold), Turner 2004 restored after. Secondary: delta_X over "
                       "the basis columns in kcal/mol with the table's INF entries set to 0, Pi "
                       "the orthogonal projection onto B_four (B4 B4^T on the used columns), "
                       "theta = theta0 + delta_X - Pi delta_X rounded to the parameter file's 0.01 "
                       "kcal/mol (certfold_baselines.shifted_theta), loaded with the same guards. "
                       "delta_X - Pi delta_X is everything outside the set's span: the identified "
                       "part, the two gauge directions (which change no MFE) and the shift on "
                       "columns no TR0 reference reads; the record reports each part's norm per "
                       "table. "
                       "kept is read off the Turner 2004 MFE pairs of registration 1's run.",
    "R4_claude": "Pool: TS0 MFE pairs with p >= 0.99 (labels are complete, so every one). "
                 "Certified = certified present under the envelope at lambda = 1 (registration "
                 "1's labels). Per reporting stratum: mean retention(certified) - mean "
                 "retention(uncertified); pooled weighted by the stratum's p >= 0.99 pair count, "
                 "over the strata where both groups are non-empty on the full data. Strata 0 and "
                 "1 are reported merged, as in E2 and for the same reason (the uncertified group "
                 "is D1, which 8 of stratum 0's 124 sequences hold); the draws stay within the ten "
                 "TR0 strata (certfold_exploratory.stratified_bootstrap, seed tag 60). A resample "
                 "on which a stratum's statistic is undefined leaves the interval undefined, "
                 "counted, never dropped. The secondary uses the same statistic and draws with "
                 "retention under the three data-visible tables.",
    "R5_claude": "Per primary table X, kept(p, X) takes the place of correct in the frozen "
                 "matched_coverage_comparison on the MFE pairs carrying a rho_env, coverage the "
                 "p >= 0.99 set's (k = 13,878), against p, the nominal gap and the length score "
                 "(-n_nt), with the score, baseline and tie-break keys R3 and E3 used; held to "
                 "rule 3. Paired bootstrap over TS0 "
                 "sequences, seed tag 61. The sentence on predicting survival better than p is "
                 "allowed only if rho_env beats p with the interval excluding 0 for EVERY primary "
                 "table; otherwise each table is reported by rule 3 without it.",
    "tertiary_claude": "kept(p, MXfold2) = p is in the released MXfold2's predicted structure "
                       "(pred_A_cpu.txt, sha256 recorded here, the shipped TrainSetAB model). R4's "
                       "statistic with kept(p, MXfold2) in place of retention, same strata and "
                       "draws. \"Consistent\" iff its point estimate has the primary's sign and "
                       "its interval does not lie entirely on the other side of 0; otherwise \"not "
                       "consistent\".",
    "D_claude": "Measured set at K: u_e = 0 on the top K entries of the frozen ranking (or on K "
                "entries drawn uniformly without replacement from the 923 reached entries, 20 "
                "draws per K, seed tag 62), u_e = delta elsewhere; lambda = 1. Only TS0 pairs "
                "uncertain at lambda = 1 in registration 1 are refolded: lowering any u_e lowers "
                "every charge, so E'(s) - E'(s*) can only grow and a certified pair stays "
                "certified. A sequence none of whose admissible interior loops reads a zeroed "
                "entry keeps its labels. Gain = (certified present under the measured set - "
                "under E) / TS0 MFE pairs. Before any TS0 fold, G2's n <= 20 enumeration on the "
                "measured sets at K = 10 and 50 (top and the first random draw of each): 0 false "
                "certificates required.",
    "order_user": "Registration 2 committed before any fold under a whole table on TS0 and before "
                  "any pair-level comparison. Do not reuse E3's folds; refold. C, then D; report "
                  "against the reading rules; then the text, figures and S8 rows.",
}


def preregister2() -> int:
    out = read_out2()
    if "predictions" in out:
        raise SystemExit("registration 2 exists")
    head2 = committed2(unmodified=True)
    if "ranking" not in head2:
        raise SystemExit("commit D's ranking first")
    prov = clean()
    mx = R / MX_PRED
    out.update({
        "status": REGISTERED2, "predictions": PRED2,
        "frozen": {"ranking_sha256": head2["ranking"]["ranked_sha256"],
                   "top_50": head2["ranking"]["ranked"][:50],
                   "mxfold2_predictions": {"path": MX_PRED, "sha256": cf._sha(mx),
                                           "n_lines": sum(1 for _ in mx.open())}},
        "registration_provenance": prov,
        "registration1": {"run_files_sha256": cf.read_out()["run"]["files_sha256"]},
    })
    write2(out)
    print(REGISTERED2)
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"rank-entries": rank_entries,
              "rank-entries-worker": rank_entries_worker,
              "preregister2": preregister2}[ph]())
