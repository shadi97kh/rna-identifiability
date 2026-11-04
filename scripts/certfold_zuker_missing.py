#!/usr/bin/env python3
"""POST HOC correctness check: is any admissible pair missing from the registered run's Zuker set?

The registered run's code would have labeled a pair missing from the Zuker set
certified absent without recording it, and its logs cannot rule that out: a backtracking failure
is logged at level 30 while ViennaRNA's default threshold is 40, and the failure path writes no
"Added N bps" line either. So the certified-absent labels had no direct check.

This reruns the Zuker pass of the primary set, the envelope at lambda = 1, on all 5,429 test
sequences, in the registered run's rows, environment and compiled callback, through the released
absent_margins, which raises on a missing pair. One pass per sequence; no X' folds, no radii.

Per sequence it records:
  n_missing          admissible pairs in no returned structure, counted directly;
  stopped            whether the released absent_margins raised;
  failure lines      ViennaRNA's "Backtracking failed" lines, visible because the level is set to
                     30 here (the level changes logging only; the counts below are the check);
  the labels         certified-absent counts over admissible and contested pairs.

Must hold for the record to be read as a confirmation: the labels reproduce the registered run's
counts on every sequence. Nothing here edits a registered record.

Run (CPU only, 8 workers at nice 10):
  CUDA_VISIBLE_DEVICES= .venv-certfold/bin/python scripts/certfold_zuker_missing.py --phase run
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold as cf  # noqa: E402
import scripts.certfold_run as cr  # noqa: E402
from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402

RUN_ID = "2026-11-03_certfold_zuker_missing"
OUT = R / "runs" / RUN_ID
SHARDS = OUT / "shards"          # gitignored (runs/**/shards/); sha256 recorded
REG_SHARDS = R / "runs" / "2026-10-13_certified_folding" / "shards"
PRIMARY = cr.PRIMARY
WORKERS = cr.RUN_WORKERS
REG_WORKERS = 8
ADDED = re.compile(r"\[ERROR\]\s+Added \d+ bps")
FAILED = "Backtracking failed"


def _write(out: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, OUT / "metrics.json")


def _read_err(fd: int) -> list[str]:
    """Everything fd 2 received since the last call; the file is emptied."""
    size = os.lseek(fd, 0, os.SEEK_END)
    os.lseek(fd, 0, os.SEEK_SET)
    data = os.read(fd, size) if size else b""
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    return data.decode("utf-8", "replace").splitlines()


def one_sequence(lc, B: dict, scale: float, r: dict, err_fd: int) -> dict:
    """One Zuker pass under the primary set; the same steps as certify_sequence, in its order."""
    import RNA

    from thermo.vienna_ref import (
        assert_turner2004_boltzmann,
        assert_turner2004_params,
        reference_md,
    )

    t0 = time.time()
    seq = r["seq"]
    plain = RNA.fold_compound(seq, reference_md())
    assert_turner2004_params(plain)
    star, e_mfe = plain.mfe()
    plain.exp_params_rescale(e_mfe)
    plain.pf()
    assert_turner2004_boltzmann(plain)
    bpp = np.array(plain.bpp(), dtype=np.float64)
    pt = cf.pair_table(star)
    star_set = {(i, pt[i]) for i in range(1, len(seq) + 1) if pt[i] > i}
    adm = cf.admissible_pairs(seq)
    outside = [q for q in adm if q not in star_set]
    contested = {q for q in outside if bpp[q[0], q[1]] >= cf.CONTESTED_P}
    F = cf.Folder(lc, seq, star, cf.Weights(B["u_env"], B["cols"], 1.0))
    F.set_radius(scale)
    _read_err(err_fd)
    structures = F.zuker()
    lines = _read_err(err_fd)
    zmap = cf.zprime_map(F, structures)
    missing = [q for q in adm if q not in zmap]
    rec = {"index": r["index"], "set": r["set"], "name": r["name"], "n_nt": len(seq),
           "stratum": r["stratum"], "n_admissible": len(adm), "n_outside": len(outside),
           "n_contested": len(contested), "n_structures": len(structures),
           "n_missing": len(missing), "missing": [list(q) for q in missing[:50]],
           "n_lines_added": sum(1 for x in lines if ADDED.fullmatch(x.strip())),
           "n_lines_backtrack_failed": sum(1 for x in lines if FAILED in x),
           "other_lines": [x for x in lines if not ADDED.fullmatch(x.strip())][:5]}
    try:
        margins = cf.absent_margins(zmap, outside, F.eprime_star)
    except GuardFailure as exc:
        rec.update(stopped=True, stopped_message=str(exc), n_absent_admissible=None,
                   n_absent_contested=None)
    else:
        cert = {q for q, m in margins.items() if m > 0}
        rec.update(stopped=False, n_absent_admissible=len(cert),
                   n_absent_contested=len(cert & contested))
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def worker(w: int, n_w: int) -> int:
    import RNA

    saved = os.dup(2)
    SHARDS.mkdir(parents=True, exist_ok=True)
    err = os.open(SHARDS / f"stderr_{w}.tmp", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(err, 2)
    try:
        cr._in_certfold_env()
        if cf.callback_backend() != "compiled":
            raise GuardFailure("the rerun folds with the compiled callback only")
        RNA.log_level_set(30)
        B = cf.load_tr0_basis()
        P = cf.template()
        lc = cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
        kind, scale = cf.run_sets(B)[PRIMARY]
        if kind != "envelope" or scale != 1.0:
            raise GuardFailure(f"{PRIMARY} is {kind} at {scale}")
        rows, _ = cr.test_rows()
        assign = cr.shard_of(rows, n_w)
        mine = sorted((r for r in rows if assign[r["index"]] == w),
                      key=lambda r: (-len(r["seq"]), r["index"]))
        with open(SHARDS / f"zuker_{w}.jsonl", "w") as fh:
            for r in mine:
                fh.write(json.dumps(one_sequence(lc, B, scale, r, 2)) + "\n")
                fh.flush()
    except BaseException:
        os.write(saved, traceback.format_exc().encode())
        return 1
    finally:
        os.close(err)
    (SHARDS / f"stderr_{w}.tmp").unlink(missing_ok=True)
    return 0


def registered_counts() -> dict[int, dict]:
    out = {}
    for w in range(REG_WORKERS):
        with open(REG_SHARDS / f"run_{w}.jsonl") as fh:
            for ln in fh:
                d = json.loads(ln)
                s = d["sets"][PRIMARY]
                out[d["index"]] = {"set": d["set"], "n_outside": d["n_admissible_outside_star"],
                                   "n_contested": d["n_contested"],
                                   "n_absent_admissible": s["n_certified_absent_admissible"],
                                   "n_absent_contested": s["n_certified_absent_contested"]}
    return out


def reduce(meta: dict) -> dict:
    recs, digests = [], {}
    for w in range(WORKERS):
        p = SHARDS / f"zuker_{w}.jsonl"
        digests[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        recs += [json.loads(x) for x in p.read_text().splitlines()]
    idx = sorted(r["index"] for r in recs)
    if idx != list(range(len(idx))):
        raise GuardFailure(f"{len(idx)} records, not one per row")
    reg = registered_counts()
    keys = ("set", "n_outside", "n_contested", "n_absent_admissible", "n_absent_contested")
    mismatch = [{"index": r["index"], "name": r["name"],
                 "rerun": {k: r[k] for k in keys}, "registered": reg[r["index"]]}
                for r in recs if {k: r[k] for k in keys} != reg[r["index"]]]
    by_set = {}
    for tset in cr.SETS:
        sub = [r for r in recs if r["set"] == tset]
        strata = {}
        for st in sorted({r["stratum"] for r in sub}):
            ss = [r for r in sub if r["stratum"] == st]
            strata[str(st)] = {
                "n_sequences": len(ss), "contested": sum(r["n_contested"] for r in ss),
                "absent_contested": sum(r["n_absent_contested"] or 0 for r in ss),
                "outside": sum(r["n_outside"] for r in ss),
                "absent_admissible": sum(r["n_absent_admissible"] or 0 for r in ss)}
        by_set[tset] = {
            "n_sequences": len(sub), "n_admissible": sum(r["n_admissible"] for r in sub),
            "n_structures": sum(r["n_structures"] for r in sub),
            "n_missing": sum(r["n_missing"] for r in sub),
            "n_sequences_with_missing": sum(1 for r in sub if r["n_missing"]),
            "n_stopped": sum(1 for r in sub if r["stopped"]),
            "n_lines_backtrack_failed": sum(r["n_lines_backtrack_failed"] for r in sub),
            "contested": sum(r["n_contested"] for r in sub),
            "absent_contested": sum(r["n_absent_contested"] or 0 for r in sub),
            "outside": sum(r["n_outside"] for r in sub),
            "absent_admissible": sum(r["n_absent_admissible"] or 0 for r in sub),
            "by_stratum": strata}
    other = [x for r in recs for x in r["other_lines"]]
    total = {k: sum(v[k] for v in by_set.values())
             for k in ("n_sequences", "n_admissible", "n_structures", "n_missing",
                       "n_sequences_with_missing", "n_stopped", "n_lines_backtrack_failed")}
    ok = total["n_missing"] == 0 and total["n_stopped"] == 0 and not mismatch
    return {**meta, "status": "complete: no pair missing, labels reproduced" if ok
            else "COMPLETE WITH FINDINGS: read n_missing, n_stopped and label_mismatches",
            "total": total, "by_set": by_set, "n_label_mismatches": len(mismatch),
            "label_mismatches": mismatch[:50], "reproduces_registered_counts": not mismatch,
            "n_other_log_lines_kept": len(other), "other_log_lines_sample": other[:20],
            "missing_examples": [{"name": r["name"], "set": r["set"], "missing": r["missing"]}
                                 for r in recs if r["n_missing"]][:50],
            "cpu_seconds": round(sum(r["seconds"] for r in recs), 1),
            "shards_sha256": digests}


def run() -> int:
    cr._in_certfold_env()
    OUT.mkdir(parents=True, exist_ok=True)
    SHARDS.mkdir(parents=True, exist_ok=True)
    prov = git_provenance(R)
    if prov["git_dirty_excluding_runs"]:
        raise SystemExit(f"commit first: {prov['git_dirty_paths_outside_runs']}")
    meta = {"run_id": RUN_ID, "post_hoc": True, "script": "scripts/certfold_zuker_missing.py",
            "question": "is any admissible pair missing from the Zuker set under the primary set?",
            "set": PRIMARY, "workers": WORKERS, "nice": 10, **prov,
            "load_at_start": os.getloadavg(), "env": cf.env_record()}
    _write({**meta, "status": "RUNNING"})
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    t0 = time.time()
    procs = [subprocess.Popen(["nice", "-n", "10", sys.executable, __file__, "--phase", "worker",
                               "--shard", str(w), "--of", str(WORKERS)], env=env, cwd=R)
             for w in range(WORKERS)]
    codes = [p.wait() for p in procs]
    if any(codes):
        _write({**meta, "status": f"FAILED: worker exit codes {codes}"})
        return 1
    out = reduce(meta)
    out["wall_seconds"] = round(time.time() - t0, 1)
    _write(out)
    print(out["status"], out["total"], "label mismatches", out["n_label_mismatches"],
          "wall", out["wall_seconds"], flush=True)
    return 0


def main() -> int:
    phase = sys.argv[sys.argv.index("--phase") + 1]
    if phase == "worker":
        return worker(int(sys.argv[sys.argv.index("--shard") + 1]),
                      int(sys.argv[sys.argv.index("--of") + 1]))
    if phase == "reduce":
        out = reduce({"run_id": RUN_ID, "post_hoc": True})
        print(out["status"], out["total"])
        return 0
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
