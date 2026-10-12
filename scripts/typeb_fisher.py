#!/usr/bin/env python3
"""E1: how much does likelihood training on TR0+TS0 see each direction class? (Preregistered.)

THEORY. For P(s|x) proportional to exp(-c(s).theta/kT), the Hessian of -log P(ref|x) in theta is
Cov_{s~P(.|x)}[c(s)]/kT^2, whatever the reference is. The directional Fisher information of
likelihood training on a corpus is therefore
    I(v) = sum_x Var_{s~P(.|x)}[c(s).v] / kT^2
at Turner 2004, dangles 0, special_hp 0, 37 C (kT from the compound's own Boltzmann factors).
v is a unit vector in kcal/mol coordinates, so I is in (kcal/mol)^-2 and the Cramer-Rao standard
error along v is 1/sqrt(I(v)) kcal/mol. Type A directions give exactly 0, because c(s).v = 0
for every structure. Type B directions give small but nonzero values, through competing
structures. Identified directions give large ones.
Scope: E1 measures likelihood training only. Margin training (MXfold2's loss) sees competing
structures only through the highest-scoring one, and E1 does not measure it.

ESTIMATE. Every TR0+TS0 sequence (11,970) gets K = 2,000 stochastic-backtracking samples
(pbacktrack; md.uniq_ML = 1) under the guarded Turner 2004 model. Counts come from Extractor
(offsets dropped: they do not depend on theta). Per sequence, the sample variance of c(s).v is
taken per direction, and the covariance of z = N_nt^T c is taken over all samples and over
each half of the samples.

Directions: the intrinsic N1 and N2 (integer vectors, normalised); the fold's 8 null, 3
published and 8 contrast directions (scripts/typeb_common.fold_directions, re-checked on read);
exploratory, 8 random unit directions supported on the coverage columns.

MUST HOLD (any failure stops the run): c(s).N1 = c(s).N2 = 0 exactly, in integers, for every
sampled structure; and exact reconstruction, c(s).theta_0 + o(s) equal to eval_structure, on the
first sample of every fourth sequence in each shard.

Phases: preregister | run (driver, under setsid) | analyse.
Run: .venv/bin/python scripts/typeb_fisher.py --phase preregister
     setsid nohup .venv/bin/python scripts/typeb_fisher.py --phase run > LOG 2>&1 &
     .venv/bin/python scripts/typeb_fisher.py --phase analyse
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.predecessor import RUNNING_PREFIX  # noqa: E402
from scripts.runmeta import git_provenance  # noqa: E402

RUN_ID = "2026-10-12_typeb_fisher"
OUT = R / "runs" / RUN_ID
SHARDS = OUT / "shards"  # gitignored (runs/**/shards/); sha256 recorded
SEED = 20261012
K = 2000
N_WORKERS = 8
BOOT = 2000
RECON_EVERY = 4
RUNNING = f"{RUNNING_PREFIX}: E1 Fisher sampling"
DONE = "sampled (workers)"

PRED = {
    "registered": "2026-10-12, before any sampling",
    "question_user": "how much information does likelihood training on TR0+TS0 carry about each "
                     "direction class?",
    "theory_user": "for P(s|x) proportional to exp(-c(s).theta/kT), the Hessian of "
                   "-log P(ref|x) is Cov_P[c(s)]/(kT)^2, independent of the reference; "
                   "I(v) = sum over sequences of Var_{s~P(.|x)}[c(s).v]/(kT)^2 at Turner 2004, "
                   "d0, special_hp 0, 37 C",
    "registered_test_user": "the contrast/null ratio of mean I (8 contrast vs 8 random null "
                            "directions) exceeds 1: bootstrap 95% CI lower bound > 1. No "
                            "magnitude predicted. THE ONLY REGISTERED TEST.",
    "reported_check_not_a_prediction_user": "I(v) > 0 for every random null direction "
                                            "(bootstrap lower bound). It cannot fail: a sum of "
                                            "variances is positive for any direction some "
                                            "competing structure uses. Reported beside the "
                                            "exact zero for N1 and N2, because 'Type A exactly "
                                            "0, Type B above 0' is the taxonomy's central "
                                            "point.",
    "must_hold": "c(s).N1 = c(s).N2 = 0 exactly (integers) for every sampled structure; exact "
                 "reconstruction c(s).theta_0 + o(s) = eval_structure on a subsample (the first "
                 f"sample of every {RECON_EVERY}th sequence per shard)",
    "estimate": f"all 11,970 TR0+TS0 sequences, K = {K} pbacktrack samples each (md.uniq_ML = "
                f"1), {N_WORKERS} workers; per-sequence sample variances (ddof 1); intervals "
                f"by bootstrap over sequences, {BOOT} draws",
    "directions": "intrinsic N1, N2 (analytic, normalised); the fold's 8 null, 3 published, 8 "
                  "contrast unit directions (delta/||delta|| at 1.0 kcal/mol, re-checked on "
                  "read: |M v| <= 1e-9 for null/published, contrast orthogonal to the null "
                  "bases and inside int22/int21/int11/bulge); exploratory: 8 random unit "
                  f"directions on the coverage columns, SeedSequence([{SEED}, 10, j])",
    "spectrum_user": "eigenvalues of N_nt^T F N_nt (426 x 426); split-half noise floor = largest "
                     "|eigenvalue| of (F_A - F_B)/2, halves = first and second K/2 samples; "
                     "report how many eigenvalues fall below 1% of the median contrast "
                     "information, and how many below the noise floor",
    "report_user": "every I(v) also as the Cramer-Rao standard error 1/sqrt(I(v)) in kcal/mol",
    "exploratory_user": "Spearman correlation between I(v) and the fold's genuine changed-MFE "
                        "fraction at matched norm (1 and 16 kcal/mol), across all 19 fold "
                        "directions AND within the null group (8 random + 3 published = 11); "
                        "the across-19 value mostly measures the gap between the groups",
    "scope_user": "E1 measures likelihood training only. Margin training, which is MXfold2's "
                  "loss, sees competing structures only through the highest-scoring one, and E1 "
                  "does not measure it.",
}


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def read_out():
    return json.loads((OUT / "metrics.json").read_text())


def preregister():
    if (OUT / "metrics.json").exists():
        raise SystemExit("the registration exists")
    write({"run_id": RUN_ID, "status": "PRE-REGISTERED, nothing sampled", "predictions": PRED,
           **git_provenance(R)})
    return 0


def directions():
    """Names, the dense direction matrix over the basis columns, the integer intrinsic
    vectors, and the lifted nontrivial basis."""
    from scripts.turner_nullspace import analytic_directions
    from scripts.typeb_common import fold_directions

    fold, B = fold_directions()
    cols, used, unused, NT = B["cols"], B["used"], B["unused"], B["NT"]
    an = analytic_directions(cols, 0, 0)
    n1 = an["N1_hairpin_mismatch_vs_init"]
    n2 = an["N2_interior_mismatch_vs_init"]
    assert np.all(n1 == np.round(n1)) and np.all(n2 == np.round(n2))
    names, vecs = ["intrinsic:N1", "intrinsic:N2"], [n1 / np.linalg.norm(n1),
                                                      n2 / np.linalg.norm(n2)]
    for (kind, name), v in sorted(fold.items()):
        names.append(f"{kind}:{name}")
        vecs.append(v)
    for j in range(8):
        g = np.zeros(len(cols))
        g[unused] = np.random.default_rng(np.random.SeedSequence([SEED, 10, j])).standard_normal(
            len(unused))
        names.append(f"coverage:{j}")
        vecs.append(g / np.linalg.norm(g))
    NTf = np.zeros((len(cols), NT.shape[1]))
    NTf[used] = NT
    return names, np.stack(vecs, axis=1), np.round(np.stack([n1, n2], 1)).astype(np.int64), NTf, B


def worker(w, W, dst):
    import RNA
    import scipy.sparse as sp

    from scripts.turner_nullspace import Extractor
    from scripts.turner_par import _rebuild, template
    from scripts.turner_typeb import fold_rows
    from thermo.vienna_ref import (
        assert_turner2004_boltzmann,
        assert_turner2004_params,
        reference_md,
    )

    T = template()
    theta0 = T["theta"]
    names, D, Nint, NTf, B = directions()
    cols = B["cols"]
    idx = {k: i for i, k in enumerate(cols)}
    t0 = np.array([theta0[k] for k in cols], dtype=np.int64)
    ex = Extractor(T["special"], 0, 0, lxc=T["lxc"])
    md = reference_md()
    md.uniq_ML = 1
    rows = fold_rows()
    mine = [(i, r) for i, r in enumerate(rows) if i % W == w]
    _rebuild()
    k_nt = NTf.shape[1]
    Ix = np.zeros((len(mine), D.shape[1]))
    covF, covA, covB = (np.zeros((k_nt, k_nt)) for _ in range(3))
    recon, kT = 0, None
    seq_index = []
    for n, (i, (_, name, seq, _db)) in enumerate(mine):
        fc = RNA.fold_compound(seq, md)
        if n == 0:
            assert_turner2004_params(fc)
        _, e = fc.mfe()
        fc.exp_params_rescale(e)
        fc.pf()
        if n == 0:
            assert_turner2004_boltzmann(fc)
            kT = fc.exp_params.kT / 1000.0  # kcal/mol
        ss = fc.pbacktrack(K)
        assert len(ss) == K, (name, len(ss))
        ri, ci, va = [], [], []
        for r, s in enumerate(ss):
            c = ex.counts(seq, s)
            off = c.pop(("_offset",), 0)
            for key, v in c.items():
                ri.append(r)
                ci.append(idx[key])
                va.append(v)
            if r == 0 and n % RECON_EVERY == 0:
                cv = np.array([c[k] for k in c], dtype=np.int64)
                ix = np.array([idx[k] for k in c], dtype=np.int64)
                want = int(round(fc.eval_structure(s) * 100))
                if int(cv @ t0[ix]) + off != want:
                    raise SystemExit(f"reconstruction failed: {name} {s}")
                recon += 1
        C = sp.csr_matrix((np.array(va, dtype=np.int64), (ri, ci)), shape=(K, len(cols)))
        if np.any(C @ Nint != 0):
            raise SystemExit(f"must-hold failed: c(s).N != 0 on {name}")
        P = np.asarray(C @ D)
        Ix[n] = P.var(axis=0, ddof=1) / kT**2
        Z = np.asarray(C @ NTf)
        for M_, rows_ in ((covF, slice(None)), (covA, slice(0, K // 2)),
                          (covB, slice(K // 2, K))):
            z = Z[rows_] - Z[rows_].mean(axis=0)
            M_ += (z.T @ z) / (z.shape[0] - 1)
        seq_index.append(i)
    np.savez(dst, Ix=Ix, seq_index=np.array(seq_index), covF=covF / kT**2, covA=covA / kT**2,
             covB=covB / kT**2, names=np.array(names), kT=kT, recon_checked=recon)
    return 0


def run():
    out = read_out()
    assert out["status"].startswith("PRE-REGISTERED"), out["status"]
    assert out["predictions"] == json.loads(json.dumps(PRED)), "PRED changed since registration"
    if os.getsid(0) != os.getpid():
        raise SystemExit("launch under setsid")
    SHARDS.mkdir(parents=True, exist_ok=True)
    dsts = [SHARDS / f"shard{w}.npz" for w in range(N_WORKERS)]
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
    out = read_out()
    ok = rc == [0] * N_WORKERS and all(d.exists() for d in dsts)
    out["run"].update({"workers_exit": rc, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "shards_sha256": {d.name: _sha(d) for d in dsts if d.exists()}})
    out["status"] = DONE if ok else f"FAILED: workers exited {rc}"
    write(out)
    return 0 if ok else 1


def _sha(p):
    import hashlib
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _boot(rng, n):
    return rng.integers(0, n, (BOOT, n))


def analyse():
    from scipy import stats

    out = read_out()
    assert out["status"] == DONE, out["status"]
    assert out["predictions"] == json.loads(json.dumps(PRED)), "PRED changed since registration"
    for f, h in out["run"]["shards_sha256"].items():
        assert _sha(SHARDS / f) == h, f"{f} changed"
    sh = [np.load(SHARDS / f"shard{w}.npz") for w in range(N_WORKERS)]
    names = [str(x) for x in sh[0]["names"]]
    Ix = np.concatenate([s["Ix"] for s in sh])
    F = sum(s["covF"] for s in sh)
    FA, FB = sum(s["covA"] for s in sh), sum(s["covB"] for s in sh)
    kTs = {float(s["kT"]) for s in sh}
    assert len(kTs) == 1
    n = Ix.shape[0]
    assert n == 11970, n
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 11]))
    bi = _boot(rng, n)
    Itot = Ix.sum(axis=0)
    Ib = np.stack([Ix[b].sum(axis=0) for b in bi])
    per = {}
    for j, nm in enumerate(names):
        lo, hi = np.percentile(Ib[:, j], [2.5, 97.5])
        per[nm] = {"I": float(Itot[j]), "ci95": [float(lo), float(hi)],
                   "cr_se_kcal": float(1 / np.sqrt(Itot[j])) if Itot[j] > 0 else None,
                   "cr_se_ci95_kcal": ([float(1 / np.sqrt(hi)), float(1 / np.sqrt(lo))]
                                       if lo > 0 else None)}
    grp = {g: [j for j, nm in enumerate(names) if nm.startswith(g + ":")]
           for g in ("intrinsic", "null", "published", "contrast", "coverage")}
    ratio = Ib[:, grp["contrast"]].mean(axis=1) / Ib[:, grp["null"]].mean(axis=1)
    r_hat = float(Itot[grp["contrast"]].mean() / Itot[grp["null"]].mean())
    r_ci = [float(x) for x in np.percentile(ratio, [2.5, 97.5])]
    ev = np.linalg.eigvalsh((F + F.T) / 2)
    noise = float(np.abs(np.linalg.eigvalsh(((FA - FB) + (FA - FB).T) / 4)).max())
    med_con = float(np.median(Itot[grp["contrast"]]))
    # exploratory: Spearman against the fold's genuine changed-MFE fraction at matched norm
    typeb = json.loads((R / "runs" / "2026-10-12_turner_typeb" / "metrics.json").read_text())
    gf = {(p["kind"], p["name"], p["norm_kcal"]): p["genuine_fraction"]
          for p in typeb["experiment_2"]["points"]}
    sp_ = {}
    for s_ in (1.0, 16.0):
        rows = [(Itot[j], gf[(nm.split(":")[0], nm.split(":")[1], s_)])
                for j, nm in enumerate(names)
                if nm.split(":")[0] in ("null", "published", "contrast")]
        rows_null = [(Itot[j], gf[(nm.split(":")[0], nm.split(":")[1], s_)])
                     for j, nm in enumerate(names) if nm.split(":")[0] in ("null", "published")]
        a, b = zip(*rows, strict=True)
        c_, d_ = zip(*rows_null, strict=True)
        sp_[str(s_)] = {"all_19": float(stats.spearmanr(a, b).statistic), "n_all": len(rows),
                        "null_group_11": float(stats.spearmanr(c_, d_).statistic),
                        "n_null": len(rows_null)}
    res = {
        "kT_kcal": kTs.pop(), "n_sequences": n, "K": K,
        "per_direction": per,
        "registered_test": {"contrast_over_null_mean_I": r_hat, "ci95": r_ci,
                            "passes": r_ci[0] > 1},
        "reported_check": {"intrinsic_exactly_zero": bool(np.all(Itot[grp["intrinsic"]] == 0)),
                           "null_lower_bounds": {names[j]: per[names[j]]["ci95"][0]
                                                 for j in grp["null"]},
                           "all_null_above_zero": all(per[names[j]]["ci95"][0] > 0
                                                      for j in grp["null"])},
        "spectrum": {"eigenvalues": ev.tolist(), "noise_floor": noise,
                     "median_contrast_I": med_con,
                     "n_below_1pct_median_contrast": int((ev < 0.01 * med_con).sum()),
                     "n_below_noise_floor": int((ev < noise).sum()),
                     "max": float(ev.max()), "median": float(np.median(ev))},
        "exploratory_spearman": sp_,
        "must_hold": {"intrinsic_counts_zero_every_sample": True,
                      "reconstruction_checked": int(sum(int(s["recon_checked"]) for s in sh))},
        "provenance": git_provenance(R),
    }
    out["results"] = res
    out["status"] = "analysed"
    write(out)
    print(json.dumps({k: res[k] for k in ("registered_test", "reported_check")}
                     | {"spectrum": {k: v for k, v in res["spectrum"].items()
                                     if k != "eigenvalues"}, "spearman": sp_}, indent=1))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    if ph == "worker":
        a = sys.argv[sys.argv.index("worker") + 1:]
        sys.exit(worker(int(a[0]), int(a[1]), a[2]))
    sys.exit({"preregister": preregister, "run": run, "analyse": analyse}[ph]())
