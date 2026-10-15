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


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"rank-entries": rank_entries,
              "rank-entries-worker": rank_entries_worker}[ph]())
