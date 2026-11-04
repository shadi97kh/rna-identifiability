#!/usr/bin/env python3
"""POST HOC: the certificate's ranking of what to measure against two rankings that need no
certificate, from the same 300-sequence TR0 pilot.
Specification: docs/posthoc_additions_2026-11-03.md.

Phases, each refusing unless the specification and the previous phase's record are committed:
  rank       (a) reads by the pilot's MFE structures; (b) reads by its admissible interior loops
  g2         G2 on each top 50, before any TS0 fold
  labels     TS0 under each top 50 and, as a control, the certificate's own (workers)
  analyse    gains at K = 50, the paired sequence bootstrap, the reading

Run (CPU only), one phase at a time:
  CUDA_VISIBLE_DEVICES= .venv-certfold/bin/python scripts/certfold_frequency_rankings.py \
      --phase rank
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold_default_cert as dc  # noqa: E402

r2, cf, cr = dc.r2, dc.cf, dc.cr
GuardFailure = dc.GuardFailure

ME = "scripts/certfold_frequency_rankings.py"
SPEC = "docs/posthoc_additions_2026-11-03.md"
OUT = R / "runs" / "2026-11-03_certfold_frequency_rankings"
SHARDS = OUT / "shards"          # gitignored (runs/**/shards/); sha256 recorded
K = 50
TAG = 98
N_BOOT = 10000
WORKERS = 12
BASELINES = ("mfe_reads", "admissible_reads")
RANKED, GATED, LABELLED, DONE = "ranked", "G2 passed", "TS0 labelled", "analysed"


def read_out() -> dict:
    p = OUT / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {"run_id": OUT.name}


def write(out: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, OUT / "metrics.json")


def tracked(rel: str) -> None:
    if cf._git("ls-files", "--error-unmatch", rel).returncode != 0:
        raise SystemExit(f"{rel} is not tracked: commit it first")
    if cf._git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"{rel} has uncommitted changes: commit it first")


def gate(status) -> tuple[dict, dict]:
    tracked(SPEC)
    cr._in_certfold_env()
    prov = r2.clean()
    out = read_out()
    if out.get("status") != status:
        raise SystemExit(f"status is {out.get('status')!r}; this phase needs {status!r}")
    if status is not None:
        tracked(str((OUT / "metrics.json").relative_to(R)))
    out.setdefault("specification", {
        "path": SPEC, "commit": cf._git("log", "-1", "--format=%H", "--", SPEC).stdout.strip(),
        "sha256": cf._sha(R / SPEC)})
    out["label"] = "POST HOC (docs/posthoc_additions_2026-11-03.md)"
    return out, prov


def model():
    B = cf.load_tr0_basis()
    P = cf.template()
    return B, cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])


def measured(B: dict, ranked: list) -> list[tuple]:
    """The top K of a ranking, as column keys."""
    by_str = {str(k): k for k in B["cols"]}
    top = [by_str[k] for k, _c in ranked[:K]]
    if len(top) != K or not set(top) <= set(r2.reached_entries(B)):
        raise GuardFailure(f"top {K}: an entry is not a reached column")
    return top


def by_table(keys: list[str]) -> dict:
    return dict(collections.Counter(k.split("'")[1] for k in keys))


# ----------------------------------------------------------------------------------
# phase: the rankings, TR0 only
# ----------------------------------------------------------------------------------
def rank() -> int:
    out, prov = gate(None)
    t0 = time.time()
    import RNA

    from thermo.vienna_ref import reference_md
    B, lc = model()
    rows, _ = cf.tr0_rows()
    seqs, _bounds = cf.pilot_sequences(rows)
    if len(seqs) != cf.PILOT_SEQUENCES:
        raise GuardFailure(f"{len(seqs)} pilot sequences")
    reached = set(r2.reached_entries(B))
    if len(reached) != 923:
        raise GuardFailure(f"{len(reached)} reached entries, not 923")
    W1b = cf.Weights(B["a_four"], B["cols"], 1.0)
    mfe, adm_reads = collections.Counter(), collections.Counter()
    n_mfe_loops = n_adm_loops = 0
    for _name, seq, _stratum in seqs:
        S = r2._codes(seq)
        star = RNA.fold_compound(seq, reference_md()).mfe()[0]
        for (a, b, c, d) in cf.star_interior_loops(lc, seq, star):
            ents = [str(x) for x in cf.il_keys(S, a, b, c, d) if x in reached]
            n_mfe_loops += bool(ents)
            mfe.update(ents)
        adm = set(cf.admissible_pairs(seq))
        n_here = 0
        for (i, j) in adm:
            for (u1, u2) in W1b.shapes:
                k, l_ = i + u1 + 1, j - u2 - 1
                if (k, l_) in adm:
                    ents = [str(x) for x in cf.il_keys(S, i, j, k, l_) if x in reached]
                    n_here += bool(ents)
                    adm_reads.update(ents)
        if n_here != cf.exposure(seq, W1b, W1b.shapes):
            raise GuardFailure(f"{seq[:30]}: {n_here} loops against certfold.exposure")
        n_adm_loops += n_here
    cert = r2.committed2()["ranking"]["ranked"]
    cert_top = {k for k, _c in cert[:K]}
    res = {}
    for name, count, n_loops in (("mfe_reads", mfe, n_mfe_loops),
                                 ("admissible_reads", adm_reads, n_adm_loops)):
        ranked = sorted(((str(k), count[str(k)]) for k in reached),
                        key=lambda kv: (-kv[1], kv[0]))
        top = [k for k, _c in ranked[:K]]
        res[name] = {"ranked": [list(x) for x in ranked],
                     "ranked_sha256": r2.canon_sha([list(x) for x in ranked]),
                     "n_entries_with_a_count": sum(1 for _k, c in ranked if c > 0),
                     "n_loops_reading_a_reached_entry": n_loops,
                     "top50_by_table": by_table(top),
                     "smallest_count_in_top50": ranked[K - 1][1],
                     "overlap_top50_with_certificate_top50": len(set(top) & cert_top)}
    res["mfe_reads"]["overlap_top50_with_admissible_reads_top50"] = len(
        {k for k, _c in res["mfe_reads"]["ranked"][:K]}
        & {k for k, _c in res["admissible_reads"]["ranked"][:K]})
    out["rankings"] = {**res, "certificate_top50_by_table": by_table(sorted(cert_top)),
                       "n_sequences": len(seqs), "n_reached_entries": len(reached),
                       "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = RANKED
    write(out)
    print(json.dumps({n: {k: v for k, v in res[n].items() if k != "ranked"} for n in res},
                     indent=1))
    return 0


# ----------------------------------------------------------------------------------
# phase: G2 on each top 50, as registration 2's gate_g2_measured
# ----------------------------------------------------------------------------------
def g2() -> int:
    out, prov = gate(RANKED)
    t0 = time.time()
    B, lc = model()
    rows, _ = cf.tr0_rows()
    W1 = cf.Weights(B["a_four"], B["cols"], 1.0)
    pos = {B["cols"][e]: k for k, e in enumerate(B["used"])}
    seqs, screen = cf.g2_sequences(rows, cf.G2_SEQUENCES, 16, cf.G2_MAX_N, cf.SEED, lc, W1)
    rec = {"sequences": screen, "n_sequences": len(seqs), "sets": {}}
    for name in BASELINES:
        ents = measured(B, out["rankings"][name]["ranked"])
        Bm = dict(B, u_env=r2.measured_u(B, ents))
        tot = {"n_certified_present": 0, "n_certified_absent": 0,
               "n_exactly_robust_present": 0, "n_exactly_robust_absent": 0,
               "n_false_certificates": 0}
        n_seq_touch = n_struct_touch = n_struct = 0
        for _src, seq, _n in seqs:
            r = cf.g2_one(lc, Bm, pos, W1, seq, (1.0,), with_rho=False)
            bad = r["envelope"]["n_false_certificates"] + r["by_radius"]["1"][
                "n_false_certificates"]
            if bad:
                raise GuardFailure(f"G2 on {name}: {bad} false certificates on {seq}")
            for k in tot:
                tot[k] += r["envelope"][k]
            t = r2._touching(lc, seq, set(ents))
            n_struct_touch += t
            n_seq_touch += t > 0
            n_struct += r["n_structures"]
        planted = sum(cf.g2_one(lc, Bm, pos, W1, seq, (1.0,), scale=0.5, with_rho=False)
                      ["envelope"]["n_false_certificates"]
                      for _s, seq, _n in seqs[:cf.G2_POWER_SEQUENCES])
        if planted == 0:
            raise GuardFailure(f"G2 on {name}: halving the weights produced no false certificate")
        rec["sets"][name] = {**tot, "n_sequences_reading_a_zeroed_entry": n_seq_touch,
                             "n_structures_reading_a_zeroed_entry": n_struct_touch,
                             "n_structures": n_struct,
                             "power_check_false_certificates_at_half_weights": planted,
                             "power_check_sequences": cf.G2_POWER_SEQUENCES}
        print(name, json.dumps(rec["sets"][name]), flush=True)
    rec.update({"pass": True, "seconds": round(time.time() - t0, 1), "provenance": prov})
    out["g2"] = rec
    out["status"] = GATED
    write(out)
    return 0


# ----------------------------------------------------------------------------------
# phase: TS0 under each top 50, as registration 2's measured_labels_worker
# ----------------------------------------------------------------------------------
def the_sets(B: dict, out: dict) -> dict[str, list[tuple]]:
    sets = {"certificate": measured(B, r2.committed2()["ranking"]["ranked"])}
    for name in BASELINES:
        sets[name] = measured(B, out["rankings"][name]["ranked"])
    return sets


def labels_worker() -> int:
    w, n_w = r2.shard_args()
    B, lc = model()
    sets = the_sets(B, read_out())
    union = set().union(*map(set, sets.values()))
    rows = r2.ts0_rows_with_stars()
    assign = cr.shard_of(rows, n_w)
    mine = sorted((r for r in rows if assign[r["index"]] == w),
                  key=lambda r: (-len(r["seq"]), r["index"]))
    path = SHARDS / f"labels_{w}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(ln)["index"] for ln in path.read_text().splitlines() if ln.strip()}
    with open(path, "a") as fh:
        for r in mine:
            if r["index"] in done:
                continue
            ts = time.time()
            reads = r2.readable_entries(r["seq"], union)
            unc = [q for q in r["pairs"] if q["margins_dcal"][cr.PRIMARY] <= 0]
            rec = {"index": r["index"], "n_nt": len(r["seq"]), "stratum": r["stratum"],
                   "n_pairs": len(r["pairs"]), "n_uncertain": len(unc), "sets": {}}
            folds = 0
            for sid, ents in sets.items():
                if not (reads & set(ents)) or not unc:
                    rec["sets"][sid] = {"refolded": False, "newly_certified": []}
                    continue
                F = cf.Folder(lc, r["seq"], r["star"],
                              cf.Weights(r2.measured_u(B, ents), B["cols"], 1.0))
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


def drive(phase: str, n_w: int) -> list[int]:
    """n_w worker processes (nice 10, single-threaded); stop all on a failure."""
    SHARDS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="")
    procs = [subprocess.Popen(
        ["nice", "-n", "10", sys.executable, str(R / ME), "--phase", phase, "--shard", str(w),
         "--of", str(n_w)], cwd=R, env=env,
        stdout=(SHARDS / f"{phase}_{w}.log").open("w"), stderr=subprocess.STDOUT)
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
        raise GuardFailure(f"{phase} workers exited {codes}; shards kept at {SHARDS}")
    return [p.pid for p in procs]


def read_labels() -> list[dict]:
    recs = []
    for w in range(WORKERS):
        recs += [json.loads(ln) for ln in (SHARDS / f"labels_{w}.jsonl").read_text()
                 .splitlines() if ln.strip()]
    return recs


def labels() -> int:
    out, prov = gate(GATED)
    t0 = time.time()
    load = os.getloadavg()
    pids = drive("labels-worker", WORKERS)
    recs = read_labels()
    rows = r2.ts0_rows_with_stars()
    if sorted(r["index"] for r in recs) != sorted(r["index"] for r in rows):
        raise GuardFailure("the labels do not cover TS0 exactly once")
    out["labels"] = {
        "n_sequences": len(recs),
        "n_refolded_sequence_sets": sum(s["refolded"] for r in recs for s in r["sets"].values()),
        "folds": sum(r["folds"] for r in recs),
        "recursion_violations": sum(s.get("recursion", {}).get("violations", 0)
                                    for r in recs for s in r["sets"].values()),
        "files_sha256": {f"labels_{w}.jsonl": cf._sha(SHARDS / f"labels_{w}.jsonl")
                         for w in range(WORKERS)},
        "workers": WORKERS, "pids": pids, "load_at_start": load,
        "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = LABELLED
    write(out)
    print(json.dumps({k: out["labels"][k] for k in ("n_refolded_sequence_sets", "folds",
                                                    "recursion_violations", "seconds")}))
    return 0


# ----------------------------------------------------------------------------------
# phase: the comparison
# ----------------------------------------------------------------------------------
def reading(iv: list[float]) -> str:
    return "higher" if iv[0] > 0 else "lower" if iv[1] < 0 else "comparable"


def analyse() -> int:
    out, prov = gate(LABELLED)
    t0 = time.time()
    for f, h in out["labels"]["files_sha256"].items():
        if cf._sha(SHARDS / f) != h:
            raise GuardFailure(f"{f} changed since the labels")
    if out["labels"]["recursion_violations"]:
        raise GuardFailure("recursion violations in the labels")
    recs = sorted(read_labels(), key=lambda r: r["index"])
    head2 = r2.committed2()
    reg = {}
    for w in range(head2["measured_labels"]["workers"]):
        for ln in (r2.SHARDS2 / f"measured_{w}.jsonl").read_text().splitlines():
            if ln.strip():
                d = json.loads(ln)
                reg[d["index"]] = d["sets"]["top:50"]["newly_certified"]
    n_diff = sum(sorted(r["sets"]["certificate"]["newly_certified"]) != sorted(reg[r["index"]])
                 for r in recs)
    if n_diff:
        raise GuardFailure(f"the control differs from registration 2 on {n_diff} sequences")
    n_pairs = np.array([r["n_pairs"] for r in recs], dtype=np.float64)
    new = {s: np.array([len(r["sets"][s]["newly_certified"]) for r in recs], dtype=np.float64)
           for s in ("certificate", *BASELINES)}
    total = float(n_pairs.sum())
    gain = {s: float(v.sum() / total) for s, v in new.items()}
    if gain["certificate"] != head2["analysis_d"]["gain"]["top:50"]:
        raise GuardFailure("the control's gain differs from registration 2's record")
    if sum(sum(r["sets"][s]["refolded"] for r in recs) for s in new) == 0:
        raise GuardFailure("nothing was refolded")
    rng = np.random.default_rng(np.random.SeedSequence([cf.SEED, TAG]))
    draws = rng.integers(0, len(recs), size=(N_BOOT, len(recs)))
    counts = np.zeros((N_BOOT, len(recs)))
    for b in range(N_BOOT):
        counts[b] = np.bincount(draws[b], minlength=len(recs))
    denom = counts @ n_pairs
    res = {}
    for s in BASELINES:
        diff = (counts @ (new["certificate"] - new[s])) / denom
        lo, hi = np.percentile(diff, [2.5, 97.5])
        iv = [float(lo), float(hi)]
        res[s] = {"gain": gain[s], "newly_certified": int(new[s].sum()),
                  "certificate_minus_baseline": gain["certificate"] - gain[s],
                  "interval": iv, "reading": reading(iv),
                  "n_sequences_refolded": sum(r["sets"][s]["refolded"] for r in recs)}
    fires = any(res[s]["reading"] != "higher" for s in BASELINES)
    out["analysis"] = {
        "certificate": {"gain": gain["certificate"],
                        "newly_certified": int(new["certificate"].sum())},
        "n_mfe_pairs": int(total), "baselines": res,
        "bootstrap": {"n_boot": N_BOOT, "seed": [cf.SEED, TAG], "n_units": len(recs),
                      "units": "TS0 sequences, drawn once per resample, one set of draws for "
                               "both comparisons; the rankings are held fixed"},
        "control": {"certificate_top50_equals_registration_2_on_every_sequence": True,
                    "n_sequences_compared": len(recs)},
        "stop_rule_fires": fires,
        "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = DONE
    write(out)
    print(json.dumps({k: out["analysis"][k] for k in ("certificate", "baselines",
                                                      "stop_rule_fires")}, indent=1))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"rank": rank, "g2": g2, "labels": labels, "labels-worker": labels_worker,
              "analyse": analyse}[ph]())
