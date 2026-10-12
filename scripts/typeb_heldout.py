#!/usr/bin/env python3
"""E2: do the TR0+TS0 blind spots matter on RNA families bpRNA-1m never contained? (Preregistered.)

THE QUESTION. A Type B direction leaves every TR0+TS0 reference energy unchanged by
construction. On references from families outside the training distribution it need not. If the
blind spots matter for generalisation, moving along different null directions should change F1
on held-out families by different amounts. On TR0+TS0 it barely changes F1 at all.

HELD-OUT SET (phase build, fixed before preregistration):
  1. Rfam 15.1 seed rows under the registered projection rules
     (scripts/turner_nullspace_rfam.load_rows; consensus projections, not determined structures).
  2. Keep families whose accession is ABSENT from Rfam 12.2. bpRNA-1m drew "all families from
     RFAM 12.2" (Danaee 2018, PMC6009582; data/SOURCES.yaml rfam_12_2_families).
  3. Pair-bearing rows only; length <= 499 (TR0+TS0's maximum).
  4. Distinct (sequence, structure), then distinct sequences (first kept; conflicts counted).
  5. Remove every sequence at >= 80% identity to TR0+TS0:
     cd-hit-est-2d -i TR0TS0 -i2 HELDOUT -c 0.8 -n 5 (CD-HIT 4.8.1, built in user space;
     the release tarball's and the binary's sha256 recorded).
  Counts of families and sequences are recorded at every step.

DIRECTIONS (the fold's, runs/2026-10-12_turner_typeb, re-checked on read; each point's own
unrounded perturbation also re-checked): the 3 published null components at their own norms;
the 8 random null directions at 4, 16 and 64 kcal/mol. 27 written tables, plus the 3 whole
published tables (E3 on the held-out set). MFE only.

GUARDS: those of scripts/typeb_common.fold_rows_under_tables (rebuild, params_match, carry rows,
reference-energy and MFE-energy identities on every sequence, and for points the exact
reference-change identity).

Phases: build | preregister | run (driver, under setsid) | analyse.
Run: .venv/bin/python scripts/typeb_heldout.py --phase build
     .venv/bin/python scripts/typeb_heldout.py --phase preregister
     setsid nohup .venv/bin/python scripts/typeb_heldout.py --phase run > LOG 2>&1 &
     .venv/bin/python scripts/typeb_heldout.py --phase analyse
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.predecessor import RUNNING_PREFIX  # noqa: E402
from scripts.runmeta import git_provenance  # noqa: E402

RUN_ID = "2026-10-12_typeb_heldout"
OUT = R / "runs" / RUN_ID
SHARDS = OUT / "shards"  # gitignored (runs/**/shards/); sha256 recorded
HELDOUT = SHARDS / "heldout.tsv"  # 10.7 MB, gitignored like the shards; sha256 in the artifact
SEED = 20261012
N_WORKERS = 8
BOOT = 2000
NORMS = (4.0, 16.0, 64.0)
TEST_NORM = 16.0
MAX_LEN = 499
RFAM122 = "data/raw/structure/Rfam_12.2.family.txt.gz"
RFAM151 = "data/raw/structure/Rfam_15.1.seed.gz"
CDHIT = pathlib.Path.home() / ".local" / "bin" / "cd-hit-est-2d"
CDHIT_ARGS = ["-c", "0.8", "-n", "5", "-M", "0", "-d", "0"]
CDHIT_SOURCE = {
    "release": "https://github.com/weizhongli/cdhit/releases/download/V4.8.1/"
               "cd-hit-v4.8.1-2019-0228.tar.gz",
    "tarball_sha256_ours": "26172dba3040d1ae5c73ff0ac6c3be8c8e60cc49fc7379e434cdf9cb1e7415de",
    "build": "make zlib=no openmp=no (no zlib headers on the box; single-threaded)",
}
RUNNING = f"{RUNNING_PREFIX}: E2 held-out folds"
DONE = "folded (workers)"

PRED = {
    "registered": "2026-10-12, after the held-out set was built (no folding), before any fold",
    "question_user": "do the TR0+TS0 blind spots matter on families bpRNA-1m never contained?",
    "main_statistic_user": "per set and norm, the NOISE-CORRECTED between-direction spread of "
                           "mean paired dF1 over the 8 random null directions: corrected = "
                           "Var_directions(mean dF1) (ddof 1) - mean_directions(SE^2), floored "
                           "at 0, reported beside the raw spread. SE = bootstrap SD of each "
                           "direction's mean dF1: over FAMILIES for held-out (cluster "
                           f"bootstrap, sequence-weighted means), over sequences for in-sample "
                           f"(the fold's shards), {BOOT} draws each",
    "registered_test_user": f"at {TEST_NORM:g} kcal/mol: held-out corrected spread MINUS "
                            "in-sample corrected spread > 0, joint-bootstrap 95% CI lower bound "
                            "> 0 (families and sequences resampled independently; in each "
                            "replicate the variance of the replicate means minus the fixed mean "
                            "SE^2, floored at 0)",
    "descriptive_user": "the ratio held-out corrected / in-sample corrected, with replicates "
                        "whose in-sample corrected spread is 0 counted as 0 (against the "
                        "prediction). Descriptive only; this rule replaces any degenerate-case "
                        "rule",
    "secondary_registered_user": "at each norm (4, 16, 64), the number of the 8 null directions "
                                 "whose dF1 95% CI excludes 0, held-out vs in-sample",
    "published_user": "the 3 published null components at their own norms: held-out dF1 with "
                      "family-bootstrap CIs; no direction predicted",
    "null_result_user": "a null result is reportable: 'real but small'",
    "metrics": "per point: paired dF1 against the held-out references, changed-MFE fraction, "
               "rms change of held-out reference energies (exact, c(ref).(theta_r - theta_0))",
    "e3_on_heldout_user": "the held-out set folded under the whole Turner 1999, Andronescu 2007 "
                          "and Langdon 2018 tables: changed fraction and paired dF1 (descriptive)",
    "seed": SEED,
}


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def read_out():
    p = OUT / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {"run_id": RUN_ID}


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _registered(rel):
    import yaml
    reg = yaml.safe_load((R / "data" / "SOURCES.yaml").read_text())["sources"]
    want = [f["sha256"] for s in reg.values() for f in s.get("files", []) if f["path"] == rel]
    assert len(want) == 1, f"{rel}: not registered exactly once"
    assert _sha(R / rel) == want[0], f"{rel}: sha256 differs from the registry"
    return want[0]


# ----------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------
def build():
    from scripts.turner_nullspace_rfam import load_rows
    from scripts.turner_typeb import fold_rows

    out = read_out()
    if "predictions" in out:
        raise SystemExit("preregistered: the held-out set is fixed")
    OUT.mkdir(parents=True, exist_ok=True)
    shas = {RFAM122: _registered(RFAM122), RFAM151: _registered(RFAM151)}
    # a MySQL dump: latin-1 text, one row per family, accession first (2,588 rows; bpRNA's
    # paper counts 2,495 families, so the comparison may hold out fewer families, never more)
    with gzip.open(R / RFAM122, "rt", encoding="latin-1") as fh:
        fam122 = {ln.split("\t", 1)[0] for ln in fh if re.match(r"RF\d{5}\t", ln)}
    assert len(fam122) == 2588, len(fam122)
    rows, proj = load_rows()
    steps = []

    def step(name, rs):
        steps.append({"step": name, "families": len({r[0] for r in rs}), "sequences": len(rs)})
        return rs

    step("Rfam 15.1 seed rows (projection rules)", rows)
    rows = step("family accession absent from Rfam 12.2", [r for r in rows if r[0] not in fam122])
    rows = step("pair-bearing", [r for r in rows if "(" in r[3]])
    rows = step(f"length <= {MAX_LEN}", [r for r in rows if len(r[2]) <= MAX_LEN])
    seen, dist = set(), []
    for r in rows:
        if (r[2], r[3]) not in seen:
            seen.add((r[2], r[3]))
            dist.append(r)
    rows = step("distinct (sequence, structure)", dist)
    first, conflicts = {}, 0
    for r in rows:
        if r[2] in first:
            conflicts += first[r[2]][3] != r[3]
        else:
            first[r[2]] = r
    rows = step("distinct sequence (first kept)", list(first.values()))
    assert min(len(r[2]) for r in rows) > 10, "CD-HIT would drop sequences <= 10 nt"
    train = fold_rows()
    # CD-HIT's working files live in the gitignored shards directory (about 20 CPU-minutes,
    # single-threaded): a run whose inputs are byte-identical reuses the previous output
    t = SHARDS / "cdhit"
    t.mkdir(parents=True, exist_ok=True)
    fa_t, fa_h, fa_o, log = t / "train.fa", t / "heldout.fa", t / "kept.fa", t / "cdhit.log"
    ft = "".join(f">t{i}\n{r[2].replace('U', 'T')}\n" for i, r in enumerate(train))
    fh_ = "".join(f">h{i}\n{r[2].replace('U', 'T')}\n" for i, r in enumerate(rows))
    reuse = (fa_o.exists() and log.exists() and fa_t.exists() and fa_h.exists()
             and fa_t.read_text() == ft and fa_h.read_text() == fh_)
    if not reuse:
        fa_o.unlink(missing_ok=True)
        fa_t.write_text(ft)
        fa_h.write_text(fh_)
        cmd = [str(CDHIT), "-i", str(fa_t), "-i2", str(fa_h), "-o", str(fa_o), *CDHIT_ARGS]
        log.write_text(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    kept = {int(ln[2:].split()[0]) for ln in fa_o.read_text().splitlines()
            if ln.startswith(">h")}
    version = subprocess.run([str(CDHIT), "-h"], capture_output=True, text=True).stdout
    cdhit = {"command": ["cd-hit-est-2d", "-i", "TR0+TS0.fa", "-i2", "heldout.fa", "-o",
                         "kept.fa", *CDHIT_ARGS],
             "version": next(ln.strip() for ln in version.splitlines() if "version" in ln),
             "binary_sha256": _sha(CDHIT), **CDHIT_SOURCE,
             "inputs_sha256": {"TR0+TS0.fa": _sha(fa_t), "heldout.fa": _sha(fa_h)},
             "kept_fasta_sha256": _sha(fa_o), "n_train": len(train), "reused_output": reuse,
             "stdout_tail": log.read_text().strip().splitlines()[-3:]}
    rows = step("< 80% identity to TR0+TS0 (cd-hit-est-2d -c 0.8 -n 5)",
                [r for i, r in enumerate(rows) if i in kept])
    HELDOUT.write_text("".join("\t".join(r) + "\n" for r in rows))
    fam = collections.Counter(r[0] for r in rows)
    sizes = np.array(sorted(fam.values()))
    lens = np.array([len(r[2]) for r in rows])
    out.update({"status": "built", "build": {
        "sources_sha256": shas, "rfam_12_2_families": len(fam122), "projection": proj,
        "steps": steps, "conflicting_structures_same_sequence": conflicts, "cd_hit": cdhit,
        "heldout_tsv_sha256": _sha(HELDOUT), "n_families": len(fam), "n_sequences": len(rows),
        "family_size_min_median_max": [int(sizes.min()), float(np.median(sizes)),
                                       int(sizes.max())],
        "largest_family_share": float(sizes.max() / sizes.sum()),
        "length_min_median_max": [int(lens.min()), float(np.median(lens)), int(lens.max())],
        "references": "Rfam consensus projections, not structures determined per molecule",
        "provenance": git_provenance(R)}})
    write(out)
    print(json.dumps({k: v for k, v in out["build"].items()
                      if k in ("steps", "n_families", "n_sequences", "family_size_min_median_max",
                               "largest_family_share", "length_min_median_max",
                               "conflicting_structures_same_sequence")}, indent=1))
    return 0


def heldout_rows():
    out = read_out()
    assert _sha(HELDOUT) == out["build"]["heldout_tsv_sha256"], "heldout.tsv changed"
    return [tuple(ln.split("\t")) for ln in HELDOUT.read_text().splitlines()]


# ----------------------------------------------------------------------------------
# preregister, run
# ----------------------------------------------------------------------------------
def preregister():
    out = read_out()
    assert out.get("status") == "built", out.get("status")
    if "predictions" in out:
        raise SystemExit("the registration exists")
    out.update({"status": "PRE-REGISTERED, nothing folded", "predictions": PRED,
                "registration_provenance": git_provenance(R)})
    write(out)
    return 0


def points():
    from scripts.turner_typeb import FOLD
    meta = json.loads((FOLD / "points.json").read_text())
    own = [("published_own", m["name"], m["norm_kcal"]) for m in meta
           if m["kind"] == "published_own"]
    assert len(own) == 3, own
    return own + [("null", str(k), n) for n in NORMS for k in range(8)]


def worker(w, W, dst):
    from scripts.typeb_common import fold_rows_under_tables, full_tables, point_tables
    rows = heldout_rows()
    return fold_rows_under_tables(rows, point_tables(points()) + full_tables(), dst, w, W)


def run():
    out = read_out()
    if str(out.get("status", "")).startswith("FAILED"):
        out.setdefault("failed_attempts", []).append({"status": out["status"], **out["run"]})
        out["status"] = "PRE-REGISTERED, nothing folded"
    assert out["status"].startswith("PRE-REGISTERED"), out["status"]
    assert out["predictions"] == json.loads(json.dumps(PRED)), "PRED changed since registration"
    if os.getsid(0) != os.getpid():
        raise SystemExit("launch under setsid")
    heldout_rows()
    SHARDS.mkdir(parents=True, exist_ok=True)
    dsts = [SHARDS / f"shard{w}.jsonl" for w in range(N_WORKERS)]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    procs = [subprocess.Popen([sys.executable, __file__, "--phase", "worker", str(w),
                               str(N_WORKERS), str(d)], env=env) for w, d in enumerate(dsts)]
    out.update({"status": RUNNING, "run": {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                           "driver_pid": os.getpid(),
                                           "load_at_start": os.getloadavg(),
                                           "provenance": git_provenance(R)}})
    write(out)
    rc = [p.wait() for p in procs]
    done = [d.exists() and d.read_text().rstrip().endswith('{"done": true}') for d in dsts]
    out = read_out()
    out["run"].update({"workers_exit": rc, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "shards_sha256": {d.name: _sha(d) for d in dsts if d.exists()}})
    ok = rc == [0] * N_WORKERS and all(done)
    out["status"] = DONE if ok else f"FAILED: workers exited {rc}"
    write(out)
    return 0 if ok else 1


# ----------------------------------------------------------------------------------
# analyse
# ----------------------------------------------------------------------------------
def _ci(x):
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def _corrected(means, se2):
    """Noise-corrected between-direction variance: Var_d(means) (ddof 1) - mean(SE^2), >= 0.
    `means` is (..., 8); se2 the fixed per-direction SE^2."""
    return np.maximum(np.var(means, axis=-1, ddof=1) - se2.mean(), 0.0)


def in_sample(ps):
    """Per-sequence dF1 on TR0+TS0 at the fold's points ps, from the fold's sha-checked shards."""
    from scripts.turner_typeb import FOLD
    from scripts.turner_typeb import read_out as fold_out
    rec = fold_out()["fold"]["shards_sha256"]
    for f, h in rec.items():
        assert _sha(FOLD / "full" / f) == h, f"fold shard {f} changed"
    d = {p: [] for p in ps}
    for f in sorted(rec, key=lambda s: int(s[5:-6])):
        sh = [json.loads(ln) for ln in (FOLD / "full" / f).read_text().splitlines()]
        f0 = np.array([b["f1_ref"] for b in sh[0]["baseline"]])
        for p in ps:
            f1 = np.array(next(r for r in sh if r.get("p") == p)["f1_ref"])
            assert f1.shape == f0.shape
            d[p].append(f1 - f0)
    return {p: np.concatenate(v) for p, v in d.items()}


def analyse():
    from scripts.turner_typeb import FOLD

    out = read_out()
    assert out["status"] in (DONE, "analysed"), out["status"]
    assert out["predictions"] == json.loads(json.dumps(PRED)), "PRED changed since registration"
    for f, h in out["run"]["shards_sha256"].items():
        assert _sha(SHARDS / f) == h, f"{f} changed"
    sh = [[json.loads(ln) for ln in (SHARDS / f"shard{w}.jsonl").read_text().splitlines()]
          for w in range(N_WORKERS)]
    base = [b for s in sh for b in s[0]["baseline"]]
    f0 = np.array([b["f1_ref0"] for b in base])
    fams = sorted({b["group"] for b in base})
    fi = np.array([fams.index(b["group"]) for b in base])
    nF, n = len(fams), len(base)
    nper = np.bincount(fi, minlength=nF).astype(float)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 20]))
    cnt = np.stack([np.bincount(rng.integers(0, nF, nF), minlength=nF) for _ in range(BOOT)])
    den = cnt @ nper

    def fboot(x):  # family-cluster bootstrap of a per-sequence mean
        return cnt @ np.bincount(fi, weights=x, minlength=nF) / den

    tabs = {}
    for t in [r["table"] for r in sh[0][1:] if "table" in r]:
        rs = [next(r for r in s if r.get("table") == t) for s in sh]
        ch = np.array([x for r in rs for x in r["changed"]], dtype=float)
        d = np.array([x for r in rs for x in r["f1_ref"]]) - f0
        dref = np.array([np.nan if x is None else x for r in rs for x in r["dref_dcal"]]) / 100
        bd = fboot(d)
        tabs[t] = {"changed_fraction": float(ch.mean()), "changed_ci95": _ci(fboot(ch)),
                   "paired_dF1": float(d.mean()), "paired_dF1_ci95": _ci(bd),
                   "paired_dF1_se": float(bd.std(ddof=1)), "_boot": bd,
                   "rms_reference_energy_change_kcal": float(np.sqrt(np.nanmean(dref ** 2))),
                   "n_reference_forbidden": int(sum(r["n_reference_forbidden"] for r in rs))}
    # in-sample, the same directions and norms, from the fold
    meta = json.loads((FOLD / "points.json").read_text())
    pid = {(m["kind"], m["name"], m["norm_kcal"]): m["p"] for m in meta}
    ps = [pid[("null", str(k), s)] for s in NORMS for k in range(8)]
    ins = in_sample(ps)
    nI = len(next(iter(ins.values())))
    rng_i = np.random.default_rng(np.random.SeedSequence([SEED, 21]))
    bi = rng_i.integers(0, nI, (BOOT, nI))
    spread, secondary = {}, {}
    for s in NORMS:
        ids = [f"null:{k}@{s:g}" for k in range(8)]
        mh = np.array([tabs[i]["paired_dF1"] for i in ids])
        Bh = np.stack([tabs[i]["_boot"] for i in ids], axis=1)  # BOOT x 8
        se2h = Bh.var(axis=0, ddof=1)
        di = [ins[pid[("null", str(k), s)]] for k in range(8)]
        mi = np.array([x.mean() for x in di])
        Bi = np.stack([x[bi].mean(axis=1) for x in di], axis=1)
        se2i = Bi.var(axis=0, ddof=1)
        ch_, ci_ = _corrected(mh, se2h), _corrected(mi, se2i)
        rh, ri = _corrected(Bh, se2h), _corrected(Bi, se2i)
        diff = rh - ri
        ratio = np.where(ri > 0, rh / np.where(ri > 0, ri, 1.0), 0.0)

        def cov_aware(B, m):  # exploratory: noise term from the full bootstrap covariance
            S, k = np.cov(B, rowvar=False), B.shape[1]
            return float(max(np.var(m, ddof=1) - (np.trace(S) - S.sum() / k) / (k - 1), 0.0))

        spread[f"{s:g}"] = {
            "heldout": {"raw_var": float(np.var(mh, ddof=1)), "raw_sd": float(np.std(mh, ddof=1)),
                        "mean_se2": float(se2h.mean()), "corrected": float(ch_),
                        "corrected_sd": float(np.sqrt(ch_)),
                        "exploratory_cov_aware_corrected": cov_aware(Bh, mh)},
            "in_sample": {"raw_var": float(np.var(mi, ddof=1)),
                          "raw_sd": float(np.std(mi, ddof=1)),
                          "mean_se2": float(se2i.mean()), "corrected": float(ci_),
                          "corrected_sd": float(np.sqrt(ci_)),
                          "exploratory_cov_aware_corrected": cov_aware(Bi, mi)},
            "difference": float(ch_ - ci_), "difference_ci95": _ci(diff),
            "ratio_descriptive": (float(ch_ / ci_) if ci_ > 0 else None),
            "ratio_descriptive_ci95": _ci(ratio),
            "replicates_in_sample_zero": int((ri == 0).sum()),
            "per_direction": {i: {"heldout_dF1": float(mh[k]), "heldout_se": float(se2h[k] ** .5),
                                  "in_sample_dF1": float(mi[k]),
                                  "in_sample_se": float(se2i[k] ** .5)}
                              for k, i in enumerate(ids)}}
        lo_h, hi_h = np.percentile(Bh, [2.5, 97.5], axis=0)
        lo_i, hi_i = np.percentile(Bi, [2.5, 97.5], axis=0)
        secondary[f"{s:g}"] = {"heldout_ci_excludes_0": int(((lo_h > 0) | (hi_h < 0)).sum()),
                               "in_sample_ci_excludes_0": int(((lo_i > 0) | (hi_i < 0)).sum())}
    t = spread[f"{TEST_NORM:g}"]
    for v in tabs.values():
        v.pop("_boot")
    out["results"] = {
        "n_sequences": n, "n_families": nF, "n_in_sample": nI,
        "registered_test": {"norm_kcal": TEST_NORM, "difference": t["difference"],
                            "ci95": t["difference_ci95"], "passes": t["difference_ci95"][0] > 0},
        "spread_by_norm": spread, "secondary_registered": secondary,
        "published_components": {k: v for k, v in tabs.items() if k.startswith("published_own")},
        "full_tables_e3": {k: v for k, v in tabs.items() if k.startswith("full:")},
        "per_point": tabs, "baseline_f1_ref_mean": float(f0.mean()),
        "provenance": git_provenance(R)}
    out["status"] = "analysed"
    write(out)
    r = out["results"]
    print(json.dumps({"registered_test": r["registered_test"], "secondary": secondary,
                      "spread": {s: {k: v for k, v in x.items() if k != "per_direction"}
                                 for s, x in spread.items()}}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# exploratory: how many of TR0+TS0's 426 nontrivial directions the held-out set also misses
# ----------------------------------------------------------------------------------
OVERLAP = {
    "label": "EXPLORATORY, not registered. Asked by the user on 2026-10-12 after E2 was "
             "analysed (its registered test failed because held-out reference energies barely "
             "move along these directions); recorded with that timing. PRED is unchanged.",
    "question_user": "how many of the 426 TR0+TS0 nontrivial directions are also invisible to "
                     "the 46,130 held-out structures: the null space of the two count matrices "
                     "stacked, with the phase-0 code",
    "method": "held-out counts M_h by the registered Extractor, exact reconstruction on every "
              "row (any failure stops); on TR0+TS0's used columns, S = [A_TR0+TS0; M_h] and its "
              "null dimension by streamed QR + SVD, by the Gram eigenproblem after exact "
              "singleton elimination, and exactly (rank mod two primes after singleton "
              "elimination), minus the 2 intrinsic directions (asserted null for M_h); "
              "independently, 426 - rank(M_h N_nt). All must agree.",
}


def overlap():
    import RNA
    import scipy.sparse as sp

    from scripts.turner_nullspace import Extractor, build_matrix
    from scripts.turner_nullspace_rfam import (
        PRIMES,
        gram_null,
        projected_rank_mod_p,
        qr_null,
        reduced_null,
    )
    from scripts.turner_par import _rebuild, template
    from scripts.turner_typeb import TOL, load_basis, table_weights
    from thermo.vienna_ref import assert_turner2004_params, reference_md

    out = read_out()
    assert out["status"] == "analysed", out["status"]
    assert out["predictions"] == json.loads(json.dumps(PRED)), "PRED changed since registration"
    rows = heldout_rows()
    T = template()
    B = load_basis("tr0ts0")
    cols, used, A, NT, Qa = B["cols"], B["used"], B["A"], B["NT"], B["I"]
    md = reference_md()
    ex = Extractor(T["special"], 0, 0, lxc=T["lxc"])
    _rebuild()
    t0 = time.time()
    cnts, fails = [], []
    for n, (_, name, seq, db) in enumerate(rows):
        fc = RNA.fold_compound(seq, md)
        if n == 0:
            assert_turner2004_params(fc)
        e = int(round(fc.eval_structure(db) * 100))
        c = ex.counts(seq, db)
        off = c.pop(("_offset",), 0)
        if sum(T["theta"][k] * v for k, v in c.items()) + off != e:
            fails.append(name)
        cnts.append(c)
    if fails:
        raise SystemExit(f"reconstruction failed on {len(fails)} held-out rows: {fails[:5]}")
    Mh = build_matrix(cnts, cols).tocsr()
    Mhu = Mh[:, used].tocsr()
    nnz = np.asarray((Mh != 0).sum(axis=0)).ravel()
    intrinsic_null = float(np.abs(Mhu @ Qa).max())
    assert intrinsic_null <= TOL, f"intrinsic directions not null for held-out ({intrinsic_null})"
    # route 1: the stacked matrix on TR0+TS0's used columns, the phase-0 functions
    S = sp.vstack([A, Mhu]).tocsr()
    Ng, inf_g = gram_null(S)
    Nf, inf_q = qr_null(S)
    Nr, inf_r, red = reduced_null(S)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 30]))
    ranks = {str(p): projected_rank_mod_p(red, p, rng) for p in PRIMES}
    within = {"qr_svd": int(Nf.shape[1]), "eigh_gram": int(Ng.shape[1]),
              "eigh_reduced": int(Nr.shape[1]),
              "exact_mod_p": int(red.shape[1] - max(ranks.values()))}
    # route 2: project the held-out counts onto the 426 directions
    K = np.asarray(Mhu @ NT)
    sv = np.linalg.svd(K, compute_uv=False)
    tol = max(K.shape) * np.finfo(float).eps * sv.max()
    rank_K = int((sv > tol).sum())
    shared_proj = NT.shape[1] - rank_K
    shared_stack = {k: v - Qa.shape[1] for k, v in within.items()}
    agree = len(set(shared_stack.values()) | {shared_proj}) == 1
    if not agree:
        raise SystemExit(f"routes disagree: stacked {shared_stack}, projected {shared_proj}")
    _, _, Vt = np.linalg.svd(K, full_matrices=False)
    Vnull = Vt[rank_K:].T if rank_K < NT.shape[1] else np.zeros((NT.shape[1], 0))
    shared_basis = NT @ Vnull
    ucols = [cols[i] for i in used]
    seen = sv[sv > tol]
    # post hoc (Claude, to explain why E2's registered test failed): along each folded point,
    # the fraction of held-out reference energies that change at all, from the sha-checked shards
    for f, h in out["run"]["shards_sha256"].items():
        assert _sha(SHARDS / f) == h, f"{f} changed"
    sh = [[json.loads(ln) for ln in (SHARDS / f"shard{w}.jsonl").read_text().splitlines()]
          for w in range(N_WORKERS)]
    ref_change = {}
    for t in [r["table"] for r in sh[0][1:] if "table" in r]:
        d = np.array([np.nan if x is None else x for s in sh for r in s if r.get("table") == t
                      for x in r["dref_dcal"]], dtype=float) / 100
        d = d[~np.isnan(d)]
        nz = d != 0
        ref_change[t] = {"fraction_changed": float(nz.mean()), "n_compared": int(d.size),
                         "rms_all_kcal": float(np.sqrt(np.mean(d ** 2))),
                         "rms_among_changed_kcal": (float(np.sqrt(np.mean(d[nz] ** 2)))
                                                    if nz.any() else 0.0),
                         "max_abs_kcal": float(np.abs(d).max())}
    out["exploratory_overlap"] = {
        **OVERLAP,
        "results": {
            "n_heldout_rows": len(rows), "reconstruction_failures": 0,
            "heldout_used_columns": int((nnz > 0).sum()),
            "heldout_uses_tr0ts0_used_columns": int((nnz[used] > 0).sum()),
            "intrinsic_null_for_heldout_max_abs": intrinsic_null,
            "stacked_null_within_tr0ts0_used": within,
            "stacked_qr": inf_q, "stacked_gram": inf_g, "stacked_reduced": inf_r,
            "exact_ranks_mod_p": ranks, "stacked_shape": list(S.shape),
            "shared_nontrivial_by_route": {"stacked": shared_stack,
                                           "projected_426_minus_rank": shared_proj},
            "shared_nontrivial": shared_proj, "of": int(NT.shape[1]),
            "projected_sv_tol": float(tol),
            "projected_smallest_nonnull_sv": float(seen.min()) if seen.size else None,
            "projected_largest_null_sv": (float(sv[sv <= tol].max())
                                          if (sv <= tol).any() else None),
            "shared_weight_by_table": table_weights(shared_basis, ucols),
            "heldout_reference_change_by_point": ref_change,
            "heldout_reference_change_note": "post hoc (Claude), to explain E2's failure: "
                                             "fraction of held-out reference energies that "
                                             "change at all (exact integer differences under "
                                             "the written table; references forbidden under a "
                                             "whole table excluded)",
            "seconds": round(time.time() - t0, 1),
            "provenance": git_provenance(R)}}
    write(out)
    print(json.dumps({k: v for k, v in out["exploratory_overlap"]["results"].items()
                      if k not in ("stacked_qr", "stacked_gram", "stacked_reduced",
                                   "provenance")}, indent=1))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    if ph == "worker":
        a = sys.argv[sys.argv.index("worker") + 1:]
        sys.exit(worker(int(a[0]), int(a[1]), a[2]))
    sys.exit({"build": build, "preregister": preregister, "run": run,
              "analyse": analyse, "overlap": overlap}[ph]())
