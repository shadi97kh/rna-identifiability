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


# ----------------------------------------------------------------------------------
# after the registration
# ----------------------------------------------------------------------------------
G2_MEASURED = "G2 on the measured sets passed"
C_FOLDED = "C folded"
C_ANALYSED = "C analysed"
D_LABELLED = "D labelled"
D_ANALYSED = "D analysed"


def registered2(status) -> tuple[dict, dict, dict]:
    """(record, committed record, provenance): PRED2 unchanged and committed with the record
    unmodified, the status one of `status`, a clean tree, and registration 1's shards exactly as
    its run recorded them."""
    out = read_out2()
    st = (status,) if isinstance(status, str) else tuple(status)
    if out.get("status") not in st:
        raise SystemExit(f"status is {out.get('status')!r}; this phase needs one of {st}")
    head2 = committed2(unmodified=True)
    if head2.get("predictions") != json.loads(json.dumps(PRED2)):
        raise SystemExit("PRED2 changed since the registration")
    prov = clean()
    for f, h in out["registration1"]["run_files_sha256"].items():
        if cf._sha(cf.SHARDS / f) != h:
            raise GuardFailure(f"registration 1's {f} changed since its run")
    return out, head2, prov


def reached_entries(B: dict) -> list[tuple]:
    """The reached entries (u_e > 0), in B['cols'] order: the population D's draws come from."""
    return [k for k, u in zip(B["cols"], B["u_env"], strict=True) if u > 0]


def measured_sets(B: dict, ranked: list) -> dict[str, list[tuple]]:
    """{set id: the entries measured exactly (u_e = 0)}. top:K, the frozen ranking's first K;
    random:K:d, d < 20, K entries drawn without replacement from the reached entries in
    B['cols'] order, by one generator SeedSequence([SEED, 62]) drawing K = 10's twenty sets and
    then K = 50's."""
    by_str = {str(k): k for k in B["cols"]}
    reached = reached_entries(B)
    rs = set(reached)
    out = {}
    for K in K_VALUES:
        top = [by_str[k] for k, _c in ranked[:K]]
        if len(top) != K or not set(top) <= rs:
            raise GuardFailure(f"top {K}: an entry is not a reached column")
        out[f"top:{K}"] = top
    rng = np.random.default_rng(np.random.SeedSequence([cf.SEED, TAG_RANDOM_K]))
    for K in K_VALUES:
        for d in range(N_RANDOM):
            pick = rng.choice(len(reached), size=K, replace=False)
            out[f"random:{K}:{d}"] = [reached[i] for i in sorted(pick.tolist())]
    return out


def measured_u(B: dict, entries) -> np.ndarray:
    """u with the measured entries set to 0 and every other entry as the envelope has it."""
    idx = {k: n for n, k in enumerate(B["cols"])}
    u = np.array(B["u_env"], dtype=np.float64, copy=True)
    for e in entries:
        u[idx[e]] = 0.0
    return u


def _touching(lc, seq: str, zero: set) -> int:
    """How many of seq's enumerated structures hold an interior loop reading a zeroed entry."""
    S = _codes(seq)
    n = 0
    for s in cf.enumerate_structures(seq):
        for d in lc.loops(seq, s):
            if d["kind"] == "interior" and set(cf.il_keys(S, d["i"], d["j"], d["k"],
                                                          d["l"])) & zero:
                n += 1
                break
    return n


def gate_g2_measured() -> int:
    """G2's n <= 20 enumeration on the measured sets at K = 10 and 50 (top, and the first
    random draw of each), after PRED2 and before any TS0 fold.

    The same 200 sequences as registration 1's G2 (g2_sequences, same seed), each labelled at
    lambda = 1 under the measured set and held to the box bound with the measured u: zero false
    certificates required. Recorded beside it: how many sequences and structures read a zeroed
    entry at all (a set no G2 structure reads would test nothing new), the certified-present
    count under the unmeasured envelope on the same sequences, and the halved-weights power
    check on the first 50, which must still produce false certificates.
    """
    out, head2, prov = registered2(REGISTERED2)
    t0 = time.time()
    B = cf.load_tr0_basis()
    P = cf.template()
    lc = cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    rows, _ = cf.tr0_rows()
    W1 = cf.Weights(B["a_four"], B["cols"], 1.0)
    pos = {B["cols"][e]: k for k, e in enumerate(B["used"])}
    seqs, screen = cf.g2_sequences(rows, cf.G2_SEQUENCES, 16, cf.G2_MAX_N, cf.SEED, lc, W1)
    sets = measured_sets(B, head2["ranking"]["ranked"])
    base = {"n_certified_present": 0, "n_exactly_robust_present": 0}
    for _src, seq, _n in seqs:
        env = cf.g2_one(lc, B, pos, W1, seq, (1.0,), with_rho=False)["envelope"]
        if env["n_false_certificates"]:
            raise GuardFailure(f"the unmeasured envelope fails G2 on {seq}")
        for k in base:
            base[k] += env[k]
    rec = {"sequences": screen, "n_sequences": len(seqs),
           "unmeasured_envelope_lambda1": base, "sets": {}}
    for sid in [f"top:{K}" for K in K_VALUES] + [f"random:{K}:0" for K in K_VALUES]:
        ents = sets[sid]
        Bm = dict(B, u_env=measured_u(B, ents))
        tot = {"n_certified_present": 0, "n_certified_absent": 0,
               "n_exactly_robust_present": 0, "n_exactly_robust_absent": 0,
               "n_false_certificates": 0}
        n_seq_touch = n_struct_touch = n_struct = 0
        for _src, seq, _n in seqs:
            r = cf.g2_one(lc, Bm, pos, W1, seq, (1.0,), with_rho=False)
            bad = r["envelope"]["n_false_certificates"] + r["by_radius"]["1"][
                "n_false_certificates"]
            if bad:
                raise GuardFailure(f"G2 on {sid}: {bad} false certificates on {seq}: "
                                   f"{r['envelope']['false_certificates'][:2]}")
            for k in tot:
                tot[k] += r["envelope"][k]
            t = _touching(lc, seq, set(ents))
            n_struct_touch += t
            n_seq_touch += t > 0
            n_struct += r["n_structures"]
        planted = 0
        for _src, seq, _n in seqs[:cf.G2_POWER_SEQUENCES]:
            planted += cf.g2_one(lc, Bm, pos, W1, seq, (1.0,), scale=0.5,
                                 with_rho=False)["envelope"]["n_false_certificates"]
        if planted == 0:
            raise GuardFailure(f"G2 on {sid}: halving the weights produced no false "
                               f"certificate on {cf.G2_POWER_SEQUENCES} sequences")
        rec["sets"][sid] = {"entries": [str(e) for e in ents], **tot,
                            "n_sequences_reading_a_zeroed_entry": n_seq_touch,
                            "n_structures_reading_a_zeroed_entry": n_struct_touch,
                            "n_structures": n_struct,
                            "power_check_false_certificates_at_half_weights": planted,
                            "power_check_sequences": cf.G2_POWER_SEQUENCES}
        print(sid, json.dumps(rec["sets"][sid] | {"entries": len(ents)}), flush=True)
    rec.update({"pass": True, "test": "certified present or absent at lambda = 1 under the "
                "measured set must be robust under the box bound N(s) - sum_e |dc_e| u_e with "
                "the measured u, on every enumerated structure (certfold.g2_one)",
                "random_draws_recorded": {k: [str(e) for e in v] for k, v in sets.items()},
                "seconds": round(time.time() - t0, 1), "provenance": prov})
    out["gate_g2_measured"] = rec
    out["status"] = G2_MEASURED
    write2(out)
    return 0


# ----------------------------------------------------------------------------------
# C: TS0 under the six tables
# ----------------------------------------------------------------------------------
def ts0_rows_with_stars() -> list[dict]:
    """TS0's 1,288 rows in registration 1's order, each with its Turner 2004 MFE structure and
    MFE-pair records as registration 1's run wrote them."""
    rows, _ = cr.test_rows()
    seq_recs, _ = cr.read_shards(cf.read_out()["run"]["workers"])
    by = {s["index"]: s for s in seq_recs}
    out = []
    for r in rows:
        if r["set"] != "TS0":
            continue
        s = by[r["index"]]
        if s["name"] != r["name"] or len(s["star"]) != len(r["seq"]):
            raise GuardFailure(f"registration 1's record {r['index']} is not row {r['name']}")
        out.append({**r, "star": s["star"], "pairs": s["pairs"], "stratum": s["stratum"]})
    if len(out) != 1288:
        raise GuardFailure(f"{len(out)} TS0 rows, not 1,288")
    return out


def visible_shifts(B: dict) -> list[tuple[str, np.ndarray, dict]]:
    """[(table, delta_X - Pi delta_X in kcal/mol over B['cols'], its parts' norms)].

    delta_X = theta_X - theta0 with the table's INF entries set to a zero shift first, and Pi
    the orthogonal projection onto B_four on the used columns: exactly
    certfold.published_component_shifts's construction, of which this is the complement. The
    parts: Pi delta (the set's, whose norm must equal build-basis's radius), the shift on columns
    no TR0 reference reads, and on the used columns the two gauge directions and the rest."""
    from scripts.turner_nullspace import analytic_directions
    from scripts.typeb_common import full_tables

    cols, used, B4 = B["cols"], B["used"], B["B4"]
    t0 = cf.template()["theta"]
    unused = np.setdiff1d(np.arange(len(cols)), used)
    G = np.array([g[used] for g in analytic_directions(cols, 0, 0).values()]).T
    Qg, _ = np.linalg.qr(G)
    out = []
    for tb in full_tables():
        X = tb["id"].split(":")[-1]
        th = tb["theta"]
        d = np.array([th[k] - t0[k] for k in cols], dtype=np.float64) / 100.0
        inf = np.array([th[k] >= 100000 for k in cols])
        d = np.where(inf, 0.0, d)
        comp = np.zeros(len(cols))
        comp[used] = B4 @ (B4.T @ d[used])
        vis = d - comp
        want = float(B["radii"][X]["radius_on_B_four_kcal"])
        if abs(float(np.linalg.norm(comp)) - want) > 1e-9:
            raise GuardFailure(f"{X}: Pi delta has norm {np.linalg.norm(comp)}, build-basis "
                               f"recorded {want}")
        vu = vis[used]
        gauge = Qg @ (Qg.T @ vu)
        out.append((X, vis, {
            "norm_delta_kcal": float(np.linalg.norm(d)),
            "norm_set_part_kcal": float(np.linalg.norm(comp)),
            "norm_outside_the_set_kcal": float(np.linalg.norm(vis)),
            "of_which_on_columns_no_TR0_reference_reads_kcal":
                float(np.linalg.norm(vis[unused])),
            "of_which_gauge_kcal": float(np.linalg.norm(gauge)),
            "of_which_the_rest_on_used_columns_kcal": float(np.linalg.norm(vu - gauge)),
            "n_INF_entries_set_to_0": int(inf.sum())}))
    return out


def generating_counts(cols: list[tuple]):
    """The generating set and its count matrix over `cols`, for the whole-table carry guard
    (typeb_common.fold_rows_under_tables builds the same; E3's folds used it)."""
    from scripts.turner_nullspace import Extractor, build_matrix, generating_set

    special = cf.template()["special"]
    gen = generating_set(special)
    return gen, build_matrix([Extractor(special, 0, 0).counts(s, d) for _, s, d in gen], cols)


def fold_whole(seqs: list[str], tb: dict, cols: list[tuple], gen, G) -> tuple[list[str], dict]:
    """Every sequence's MFE under one whole published table as shipped, with E3's fold guards:
    the shipped loader, params_match on the first compound, the five generating rows the table
    moves most (typeb_common._carry, over the count columns: the pinned model has no special
    hairpins, so the tables' own special-loop lists never enter), and on every fold ViennaRNA's
    MFE energy equal to counts . theta under the table's own lxc plus the exact Ninio
    correction. Turner 2004 is restored and asserted afterwards, energies and Boltzmann factors
    both."""
    import RNA

    from scripts.certfold_baselines import _energy, _ninio_correction
    from scripts.turner_nullspace import Extractor
    from scripts.turner_par import _rebuild, params_match
    from scripts.typeb_common import _carry
    from thermo.vienna_ref import (
        assert_turner2004_boltzmann,
        assert_turner2004_params,
        reference_md,
    )

    th = tb["theta"]
    ex = Extractor(cf.template()["special"], 0, 0, lxc=tb["lxc"])
    m, cap = th[("ninio_m",)], th[("ninio_max",)]
    out = []
    getattr(RNA, tb["loader"])()
    _rebuild()
    try:
        md = reference_md()
        first = RNA.fold_compound(seqs[0], md)
        bad = params_match(first, th, 0, 0)
        if bad:
            raise GuardFailure(f"{tb['id']}: the loaded table is not the shipped one: {bad}")
        carry = _carry(tb, G, cols, cf.template()["theta"])
        for r, want in carry:
            _t, s, d = gen[r]
            got = int(round(RNA.fold_compound(s, md).eval_structure(d) * 100))
            if got != want:
                raise GuardFailure(f"{tb['id']}: carry row {r}: {got} != {want}")
        for k, seq in enumerate(seqs):
            fc = first if k == 0 else RNA.fold_compound(seq, md)
            s, e = fc.mfe()
            want = _energy(ex.counts(seq, s), th) + _ninio_correction(s, m, cap)
            if want != int(round(e * 100)):
                raise GuardFailure(f"{tb['id']}, sequence {k}: ViennaRNA's MFE energy "
                                   f"{int(round(e * 100))} dcal, counts . theta {want}")
            out.append(s)
    finally:
        RNA.params_load_RNA_Turner2004()
        _rebuild()
    fc = RNA.fold_compound("GGGGAAACCCC", reference_md())
    assert_turner2004_params(fc)
    _s, e = fc.mfe()
    fc.exp_params_rescale(e)
    fc.pf()
    assert_turner2004_boltzmann(fc)
    return out, {"loader": tb["loader"], "par_sha256": tb["par_sha256"],
                 "params_match": bad, "carry_rows": carry,
                 "mfe_energy_identity": {"n_folds": len(out), "n_mismatch": 0},
                 "turner2004_restored": True}


def fold_tables_worker() -> int:
    import tempfile

    from scripts.certfold_baselines import pair_frequencies
    from scripts.typeb_common import full_tables

    w, n_w = shard_args()
    rows = ts0_rows_with_stars()
    assign = cr.shard_of(rows, n_w)
    mine = [r for r in rows if assign[r["index"]] == w]
    seqs = [r["seq"] for r in mine]
    B = cf.load_tr0_basis()
    gen, G = generating_counts(B["cols"])
    mfe = {r["index"]: {} for r in mine}
    guards = {}
    for tb in full_tables():
        X = tb["id"].split(":")[-1]
        ss, guards[f"whole:{X}"] = fold_whole(seqs, tb, B["cols"], gen, G)
        for r, s in zip(mine, ss, strict=True):
            mfe[r["index"]][f"whole:{X}"] = s
    vis = visible_shifts(B)
    got: dict = {}

    def keep(d, k, s):
        got[(d, k)] = s

    with tempfile.TemporaryDirectory(dir=SHARDS2) as t:
        _freq, guards["visible"] = pair_frequencies(
            seqs, [v for _X, v, _p in vis], cf.template()["theta"], t, cols=B["cols"],
            on_fold=keep)
    for d, (X, _v, _p) in enumerate(vis):
        for k, r in enumerate(mine):
            mfe[r["index"]][f"visible:{X}"] = got[(d, k)]
    with open(SHARDS2 / f"tables_{w}.jsonl", "w") as fh:
        for r in mine:
            fh.write(json.dumps({"index": r["index"], "name": r["name"],
                                 "mfe": mfe[r["index"]]}) + "\n")
    (SHARDS2 / f"tables_guard_{w}.json").write_text(json.dumps(guards, default=str))
    return 0


TABLE_IDS = tuple(f"{k}:{X}" for k in ("whole", "visible") for X in TABLES)


def fold_tables() -> int:
    """C's folds: TS0 under the three whole tables and the three data-visible ones."""
    out, _head2, prov = registered2(G2_MEASURED)
    if "fold_tables" in out:
        raise SystemExit("C's folds exist")
    t0 = time.time()
    rows = ts0_rows_with_stars()
    vis = visible_shifts(cf.load_tr0_basis())
    load = os.getloadavg()
    pids = drive("fold-tables-worker", WORKERS)
    recs, guards = [], {}
    for w in range(WORKERS):
        recs += [json.loads(ln) for ln in (SHARDS2 / f"tables_{w}.jsonl").read_text()
                 .splitlines() if ln.strip()]
        guards[w] = json.loads((SHARDS2 / f"tables_guard_{w}.json").read_text())
    by = {r["index"]: r for r in recs}
    if sorted(by) != sorted(r["index"] for r in rows):
        raise GuardFailure("the folds do not cover TS0 exactly once")
    differs = collections.Counter()
    for r in rows:
        got = by[r["index"]]["mfe"]
        if set(got) != set(TABLE_IDS) or any(len(s) != len(r["seq"]) for s in got.values()):
            raise GuardFailure(f"sequence {r['index']}: tables {sorted(got)}")
        for t, s in got.items():
            differs[t] += s != r["star"]
    out["fold_tables"] = {
        "tables": list(TABLE_IDS), "n_sequences": len(rows),
        "visible_parts": {X: p for X, _v, p in vis},
        "n_sequences_mfe_differs_from_turner2004": dict(differs),
        "guards": {"whole_params_match_pass": all(not g[t]["params_match"] for g in
                                                  guards.values() for t in g if t != "visible"),
                   "visible_params_match_pass": all(g["visible"]["all_params_match_pass"]
                                                    for g in guards.values()),
                   "visible_max_rounding_kcal": max(g["visible"]["max_rounding_kcal"]
                                                    for g in guards.values()),
                   "mfe_energy_identity_folds": sum(
                       g[t]["mfe_energy_identity"]["n_folds"] for g in guards.values()
                       for t in g),
                   "per_worker": guards},
        "files_sha256": {f.name: cf._sha(f) for w in range(WORKERS)
                         for f in (SHARDS2 / f"tables_{w}.jsonl",
                                   SHARDS2 / f"tables_guard_{w}.json")},
        "workers": WORKERS, "pids": pids, "load_at_start": load,
        "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = C_FOLDED
    write2(out)
    print(json.dumps({k: out["fold_tables"][k] for k in
                      ("n_sequences_mfe_differs_from_turner2004", "seconds")}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# C: the analysis
# ----------------------------------------------------------------------------------
_JOB: dict = {}                  # set in the parent before the pool forks


def mxfold2_structures(rows: list[dict], frozen: dict) -> dict[int, str]:
    """{TS0 index: the released MXfold2's predicted structure}, from the file PRED2 pinned."""
    p = R / frozen["path"]
    if cf._sha(p) != frozen["sha256"]:
        raise GuardFailure(f"{frozen['path']} changed since the registration")
    pred = {}
    for ln in p.read_text().splitlines():
        if ln.strip():
            d = json.loads(ln)
            pred[d["name"]] = d["structure"]
    out = {}
    for r in rows:
        s = pred[r["name"].removesuffix(".bpseq")]
        if len(s) != len(r["seq"]) or set(s) - set("()."):
            raise GuardFailure(f"MXfold2's structure for {r['name']} does not fit the sequence")
        out[r["index"]] = s
    return out


def kept_records(rows: list[dict], tables: dict[int, dict], mx: dict[int, str]) -> list[dict]:
    """Registration 1's TS0 records (certfold_run.pair_records), each MFE pair given kept(p, X)
    for the six tables and MXfold2, its retention over the three whole tables and over the three
    data-visible ones, and neg_len for the length score. Unit records pass through."""
    seq_recs, freqs = cr.read_shards(cf.read_out()["run"]["workers"])
    fbi = {int(k): v for w in freqs.values() for k, v in w["by_index"].items()}
    recs = cr.pair_records([s for s in seq_recs if s["set"] == "TS0"], fbi)["TS0"]
    n_nt = {r["index"]: len(r["seq"]) for r in rows}
    pairs = {i: {t: set(cr._pairs_of(s)) for t, s in tables[i].items()} for i in tables}
    for i, s in mx.items():
        pairs[i]["mxfold2"] = set(cr._pairs_of(s))
    out = []
    for r in recs:
        if r["kind"] != "mfe":
            if r["kind"] == "unit":
                out.append(r)
            continue
        k = {t: r["pair"] in pairs[r["index"]][t] for t in (*TABLE_IDS, "mxfold2")}
        out.append({**r, **{f"kept:{t}": v for t, v in k.items()},
                    "ret_primary": sum(k[f"whole:{X}"] for X in TABLES) / len(TABLES),
                    "ret_visible": sum(k[f"visible:{X}"] for X in TABLES) / len(TABLES),
                    "ret_mxfold2": float(k["mxfold2"]),
                    "neg_len": -float(n_nt[r["index"]])})
    return out


def _r4_spec(key: str) -> dict:
    """R4's statistic on `key`: mean over p >= 0.99 MFE pairs certified at lambda = 1 minus mean
    over the uncertified ones, pooled weights the stratum's p >= 0.99 pair count."""
    def hi(rs):
        return [r for r in rs if r["kind"] == "mfe" and r["p"] >= cr.P_HI]

    def split(rs):
        h = hi(rs)
        return [r[key] for r in h if r["certified_l1"]], [r[key] for r in h
                                                          if not r["certified_l1"]]

    def stat(rs):
        c, u = split(rs)
        return sum(c) / len(c) - sum(u) / len(u)

    def defined(rs):
        c, u = split(rs)
        return bool(c) and bool(u)

    def extra(rs):
        c, u = split(rs)
        return {"n_certified": len(c), "n_uncertified": len(u),
                "mean_certified": sum(c) / len(c) if c else None,
                "mean_uncertified": sum(u) / len(u) if u else None}

    return {"stat": stat, "weight": lambda rs: len(hi(rs)), "defined": defined,
            "extra": extra, "merge": {1: 0}}


R4_SPECS = (("R4_primary", "ret_primary"), ("R4_secondary", "ret_visible"),
            ("R4_tertiary", "ret_mxfold2"))
R5_BASELINES = (("p", "p", "tiebreak"), ("nominal_gap", "gap", "tiebreak"),
                ("length", "neg_len", "tiebreak"))


def _c_task(tid: str):
    import scripts.certfold_exploratory as cx
    from scripts.certfold_metrics import paired_bootstrap

    recs = _JOB["recs"]
    if tid == "R4":
        hi = [r for r in recs if r["kind"] == "unit"
              or (r["kind"] == "mfe" and r["p"] >= cr.P_HI)]
        groups = cx.strata_groups(hi, _JOB["stratum_of"])
        return tid, cx.stratified_bootstrap(groups, n_boot=N_BOOT, seed_tag=TAG_R4,
                                            specs={n: _r4_spec(k) for n, k in R4_SPECS})
    _r5, X, name = tid.split("|")
    key, tb = next((k, t) for n, k, t in R5_BASELINES if n == name)
    rx = [{**r, "correct": r[f"kept:whole:{X}"]} if r["kind"] == "mfe" else r for r in recs]
    got = cr._matched_point(cr._mfe(rx), key, tb, False)
    sub = cr.task_records(rx, "matched", (key, tb, False))
    try:
        p, lo, hi = paired_bootstrap(sub, lambda rs: cr.stat_matched(rs, key, tb, False),
                                     n_boot=N_BOOT, seed_tag=TAG_R5)
    except GuardFailure as e:
        return tid, {**got, "interval": None, "interval_undefined_because": str(e)}
    if p != got["difference"]:
        raise GuardFailure(f"{tid}: the bootstrap's point {p} is not {got['difference']}")
    return tid, {**got, "interval": [lo, hi], "baseline_key": key}


def rule3(iv) -> str:
    """Reading rule 3: higher or lower only if the interval excludes 0."""
    if iv is None:
        return "interval undefined"
    return "higher" if iv[0] > 0 else "lower" if iv[1] < 0 else "no difference shown"


def analyse_c() -> int:
    """R4 (primary, secondary), R5 per primary table, the tertiary consistency; then the
    reading by registration 2's rules."""
    import multiprocessing

    out, head2, prov = registered2(C_FOLDED)
    t0 = time.time()
    for f, h in out["fold_tables"]["files_sha256"].items():
        if cf._sha(SHARDS2 / f) != h:
            raise GuardFailure(f"{f} changed since the folds")
    rows = ts0_rows_with_stars()
    tables = {}
    for w in range(out["fold_tables"]["workers"]):
        for ln in (SHARDS2 / f"tables_{w}.jsonl").read_text().splitlines():
            if ln.strip():
                d = json.loads(ln)
                tables[d["index"]] = d["mfe"]
    mx = mxfold2_structures(rows, head2["frozen"]["mxfold2_predictions"])
    recs = kept_records(rows, tables, mx)
    mfe = [r for r in recs if r["kind"] == "mfe"]
    _JOB.update(recs=recs, stratum_of={r["index"]: r["stratum"] for r in rows})
    tids = ["R4"] + [f"R5|{X}|{n}" for X in TABLES for n, _k, _t in R5_BASELINES]
    with multiprocessing.get_context("fork").Pool(len(tids)) as pool:
        got = dict(pool.map(_c_task, tids, chunksize=1))
    r4 = got["R4"]
    res = {"R4": {n: r4[n] for n, _k in R4_SPECS}, "R4_bootstrap": r4["bootstrap"],
           "R5": {X: {n: got[f"R5|{X}|{n}"] for n, _k, _t in R5_BASELINES} for X in TABLES}}

    # descriptive: what each table keeps
    hi = [r for r in mfe if r["p"] >= cr.P_HI]
    desc = {}
    for t in (*TABLE_IDS, "mxfold2"):
        k = f"kept:{t}"
        desc[t] = {"kept_fraction_all_mfe_pairs": sum(r[k] for r in mfe) / len(mfe),
                   "kept_fraction_p_ge_0.99": sum(r[k] for r in hi) / len(hi),
                   "kept_fraction_certified_p_ge_0.99":
                       _mean([r[k] for r in hi if r["certified_l1"]]),
                   "kept_fraction_uncertified_p_ge_0.99":
                       _mean([r[k] for r in hi if not r["certified_l1"]])}
    res["describe"] = {"n_mfe_pairs": len(mfe), "n_p_ge_0.99": len(hi),
                       "n_certified_p_ge_0.99": sum(r["certified_l1"] for r in hi),
                       "by_table": desc}

    # the reading, registration 2's rules
    def iv(n):
        return res["R4"][n]["pooled_interval"]

    p_iv, s_iv, t_iv = iv("R4_primary"), iv("R4_secondary"), iv("R4_tertiary")
    p_pass = p_iv is not None and p_iv[0] > 0
    s_pass = s_iv is not None and s_iv[0] > 0
    reading = {"R4_primary_pass": p_pass, "R4_secondary_pass": s_pass}
    rr = PRED2["reading_rules_user"]
    reading["R4_sentence"] = rr[0] if p_pass else rr[1]
    if p_pass and not s_pass:
        reading["R4_secondary_clause"] = rr[2]
    sign = np.sign(res["R4"]["R4_primary"]["pooled_point"])
    tp = res["R4"]["R4_tertiary"]["pooled_point"]
    other_side = t_iv is not None and ((sign > 0 and t_iv[1] < 0) or (sign < 0 and t_iv[0] > 0))
    reading["tertiary"] = ("consistent" if np.sign(tp) == sign and sign != 0 and not other_side
                           else "not consistent")
    reading["R5_rule3"] = {X: {n: rule3(res["R5"][X][n].get("interval"))
                               for n, _k, _t in R5_BASELINES} for X in TABLES}
    beats = all(reading["R5_rule3"][X]["p"] == "higher" for X in TABLES)
    reading["R5_claim_allowed"] = beats
    reading["R5_claim"] = (rr[3].split("may the paper say ")[1].strip('".')
                           if beats else None)
    res["reading"] = reading
    out["analysis_c"] = {**res, "seed_tags": {"R4": TAG_R4, "R5": TAG_R5}, "n_boot": N_BOOT,
                         "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = C_ANALYSED
    write2(out)
    print(json.dumps(reading, indent=1))
    return 0


def _mean(xs: list) -> float | None:
    return sum(xs) / len(xs) if xs else None


# ----------------------------------------------------------------------------------
# D: TS0 labels under the measured sets
# ----------------------------------------------------------------------------------
def readable_entries(seq: str, wanted: set) -> set:
    """The entries of `wanted` that some admissible interior loop of seq reads.

    Every canonical (i, j) with every loop shape the wanted entries have, (k, l) =
    (i + u1 + 1, j - u2 - 1) canonical with at least three unpaired bases inside. That is a
    superset of the loops any structure can hold, so a sequence it clears reads no zeroed
    entry, its weights are all unchanged, and its labels are registration 1's."""
    shapes = cf.reached_shapes(list(wanted), np.ones(len(wanted)))
    S = _codes(seq)
    adm = set(cf.admissible_pairs(seq))
    found = set()
    for (i, j) in adm:
        for (u1, u2) in shapes:
            k, l_ = i + u1 + 1, j - u2 - 1
            if (k, l_) in adm:
                found.update(set(cf.il_keys(S, i, j, k, l_)) & wanted)
    return found


def measured_labels_worker() -> int:
    """Shard w: for every TS0 sequence and every measured set whose zeroed entries it can read,
    the pairs uncertain at lambda = 1 in registration 1 refolded under the measured set. Each
    refolded margin must be at least registration 1's (lowering u lowers every charge)."""
    w, n_w = shard_args()
    head2 = committed2(unmodified=True)
    B = cf.load_tr0_basis()
    P = cf.template()
    lc = cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    sets = measured_sets(B, head2["ranking"]["ranked"])
    union = set().union(*map(set, sets.values()))
    rows = ts0_rows_with_stars()
    assign = cr.shard_of(rows, n_w)
    mine = sorted((r for r in rows if assign[r["index"]] == w),
                  key=lambda r: (-len(r["seq"]), r["index"]))
    done = set()
    path = SHARDS2 / f"measured_{w}.jsonl"
    if path.exists():
        done = {json.loads(ln)["index"] for ln in path.read_text().splitlines() if ln.strip()}
    with open(path, "a") as fh:
        for r in mine:
            if r["index"] in done:
                continue
            ts = time.time()
            reads = readable_entries(r["seq"], union)
            unc = [q for q in r["pairs"] if q["margins_dcal"][cr.PRIMARY] <= 0]
            rec = {"index": r["index"], "n_nt": len(r["seq"]), "stratum": r["stratum"],
                   "n_pairs": len(r["pairs"]), "n_uncertain": len(unc), "sets": {}}
            folds = 0
            for sid, ents in sets.items():
                if not (reads & set(ents)) or not unc:
                    rec["sets"][sid] = {"refolded": False, "newly_certified": []}
                    continue
                F = cf.Folder(lc, r["seq"], r["star"],
                              cf.Weights(measured_u(B, ents), B["cols"], 1.0))
                new = []
                for q in unc:
                    i, j = q["pair"]
                    _s, x, _e0 = F.without(i, j)
                    m = x - F.eprime_star
                    if m < q["margins_dcal"][cr.PRIMARY]:
                        raise GuardFailure(f"{r['index']} {sid} {(i, j)}: margin {m} below "
                                           f"registration 1's {q['margins_dcal'][cr.PRIMARY]}")
                    if m > 0:
                        new.append([i, j])
                folds += F.folds
                rec["sets"][sid] = {"refolded": True, "newly_certified": new,
                                    "recursion": F.recursion}
            rec.update({"folds": folds, "seconds": round(time.time() - ts, 2)})
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    return 0


def measured_labels() -> int:
    """D's labels: TS0 under every measured set (2 top, 40 random), uncertain pairs refolded."""
    out, head2, prov = registered2((C_ANALYSED, D_LABELLED))
    if "measured_labels" in out and "--resume" not in sys.argv:
        raise SystemExit("D's labels exist")
    t0 = time.time()
    load = os.getloadavg()
    pids = drive("measured-labels-worker", WORKERS)
    recs = []
    for w in range(WORKERS):
        recs += [json.loads(ln) for ln in (SHARDS2 / f"measured_{w}.jsonl").read_text()
                 .splitlines() if ln.strip()]
    rows = ts0_rows_with_stars()
    if sorted(r["index"] for r in recs) != sorted(r["index"] for r in rows):
        raise GuardFailure("D's labels do not cover TS0 exactly once")
    viol = sum(s.get("recursion", {}).get("violations", 0) for r in recs
               for s in r["sets"].values())
    out["measured_labels"] = {
        "n_sequences": len(recs), "n_sets": len(measured_sets(cf.load_tr0_basis(),
                                                               head2["ranking"]["ranked"])),
        "n_refolded_sequence_sets": sum(s["refolded"] for r in recs for s in r["sets"].values()),
        "folds": sum(r["folds"] for r in recs), "recursion_violations": viol,
        "files_sha256": {f"measured_{w}.jsonl": cf._sha(SHARDS2 / f"measured_{w}.jsonl")
                         for w in range(WORKERS)},
        "workers": WORKERS, "pids": pids, "load_at_start": load,
        "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = D_LABELLED
    write2(out)
    print(json.dumps({k: out["measured_labels"][k] for k in
                      ("n_refolded_sequence_sets", "folds", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"rank-entries": rank_entries,
              "rank-entries-worker": rank_entries_worker,
              "preregister2": preregister2,
              "gate-g2-measured": gate_g2_measured,
              "fold-tables": fold_tables,
              "fold-tables-worker": fold_tables_worker,
              "analyse-c": analyse_c,
              "measured-labels": measured_labels,
              "measured-labels-worker": measured_labels_worker}[ph]())
