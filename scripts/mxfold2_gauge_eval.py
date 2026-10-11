#!/usr/bin/env python3
"""Phase 2 of the MXfold2 gauge study: evaluate every model on TS0 and test the predictions.

CPU only (CUDA_VISIBLE_DEVICES=""), so that stage 2b reproduces stage 2a bitwise for the
models they share. Refuses to run unless the pre-registration (with its amendments) is
committed and unmodified.

Stages (registered: PREREG["analysis"]):
  2a  the live arms, when the live seed study has ended: AdamW w=1.0 and w=0.125 at epoch
      10 (8 seeds each, plus the _dup runs), and the epoch-1 floor pairs (original
      phase-epoch1 run, or its OOM re-run, vs the live run's epoch-1 checkpoint).
  2b  everything: 2a plus the SGD, regulariser and regulariser-duplicate runs, and the
      A/B checkpoints at the loss-matched epochs (G1).

Per TS0 sequence x and model k (one pass over all models of the stage, sharded over x):
  theta_k(x) exactly as scripts/mxfold2_decompose.py (Geometry, corrections c1-c3);
  V1-V3 exactly as scripts/mxfold2_decompose.py (same tolerances); the predicted
  structure (for F); C1: the canonicalised model's predicted structure and score (near-tie
  rule and tolerance of P0.4, for the predicted and the reference structure); the Gram
  matrices over models of theta and of P_G theta (centred on the stage's first model,
  the same model in 2a and 2b; every statistic is translation-invariant);
  kappa_k(x) = <theta_k, c_x'> / ||c_x'||^2 (W2); the null share of 20 random
  directions with theta's support, pooled and per x, and dim G / dim theta (G1
  baselines). Shards hold only these per-x summaries (no theta on disk).

Reduction (registered statistics, seed-level): per-arm V_G, V_tot, V_I = V_tot - V_G;
W1, W2, R1, R2 by a paired bootstrap over seed indices (10,000 resamples,
default_rng(20261006); resamples drawing a single seed are dropped and counted) with a
leave-one-seed-out jackknife alongside; G1 and G3 from the Gram matrices; F per model and
per arm; C1 per model; S1 from the SGD runs' logged v.g; S2 from the per-epoch
checkpoints. No statistic is written unless V1-V3 and C1 pass on every model.

Usage (in .venv-mxfold2):
  python scripts/mxfold2_gauge_eval.py --stage 2a --workers 4 --threads 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import zlib

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

OUT = R / "runs" / "2026-10-08_mxfold2_gauge_eval"
LIVE = R / "runs" / "2026-10-07_mxfold2_seed_study"
GRUNS = R / "runs" / "2026-10-08_mxfold2_gauge_runs"
DATA = R / ".cache" / "mxfold2-data"
REL_TOL = 1e-3          # V1-V3 tolerance, as scripts/mxfold2_decompose.py
BOOT = 10_000
BOOT_SEED = 20261006
N_RANDOM_DIRS = 20


# ----------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------
def train_losses(log_glob):
    """{epoch: loss} parsed from MXfold2's 'Train Epoch: e\\tLoss: x' lines (last wins)."""
    pat = re.compile(r"Train Epoch: (\d+)\tLoss: ([-+0-9.eE]+|nan|inf)")
    out = {}
    for f in sorted(log_glob):
        for e, v in pat.findall(f.read_text(errors="ignore")):
            out[int(e)] = float(v)
    return out


def gauge_runs_meta():
    """Every gauge-study run with its record: status, trajectory_exact, arm, seed."""
    meta = json.loads((GRUNS / "metrics.json").read_text())
    runs = {}
    for block in ("runs", "fresh_rerun"):
        for n, r in meta.get(block, {}).get("runs", {}).items():
            runs[n] = r
    return meta, runs


def matched_epochs(seeds):
    """The registered G1 loss matching: e_A = 10; e_B = earliest SGD epoch with L_B <= L_A(10);
    if none, e_B = 10 and e_A = the AdamW epoch whose loss is closest to L_B(10)."""
    s0 = seeds[0]
    LA = train_losses((LIVE / "seeds" / f"full_seed{s0}_w1.0").glob("stdout.attempt*.log"))
    b_name = f"sgd_seed{s0}_w1.0"
    meta, runs = gauge_runs_meta()
    fresh = meta.get("replacements", {}).get(b_name)
    if fresh and runs.get(fresh, {}).get("status") == "complete":
        b_name = fresh
    LB = train_losses((GRUNS / "seeds" / b_name).glob("stdout.attempt*.log"))
    eA, eB = 10, None
    for e in sorted(LB):
        if LB[e] <= LA[10]:
            eB = e
            break
    if eB is None:
        eB = 10
        eA = min(LA, key=lambda e: abs(LA[e] - LB[10]))
    gap = abs(LA[eA] - LB[eB]) / max(abs(LA[eA]), 1e-300)
    return {"A": f"full_seed{s0}_w1.0", "B": b_name, "e_A": eA, "e_B": eB,
            "L_A": LA[eA], "L_B": LB[eB], "relative_gap": gap,
            "descriptive_only": gap > 0.10, "losses_A": LA, "losses_B": LB}


def model_list(stage, seeds):
    L = LIVE / "seeds"
    m = []
    for w in ("1.0", "0.125"):
        for s in seeds:
            m.append({"name": f"adamw_w{w}_seed{s}", "path": L / f"full_seed{s}_w{w}" /
                      "checkpoints" / "epoch-10", "arm": f"adamw_w{w}", "seed": s})
        m.append({"name": f"adamw_w{w}_seed{seeds[0]}_dup", "path": L /
                  f"full_seed{seeds[0]}_w{w}_dup" / "checkpoints" / "epoch-10",
                  "arm": f"adamw_w{w}_dup", "seed": seeds[0]})
    for s in seeds:
        orig = L / f"seed{s}_w1.0" / "checkpoints" / "epoch-1"
        rerun = L / f"seed{s}_w1.0_rerun" / "checkpoints" / "epoch-1"
        m.append({"name": f"e1orig_seed{s}", "path": rerun if rerun.exists() else orig,
                  "arm": "e1orig", "seed": s})
        m.append({"name": f"e1live_seed{s}", "path": L / f"full_seed{s}_w1.0" /
                  "checkpoints" / "epoch-1", "arm": "e1live", "seed": s})
    if stage == "2b":
        G = GRUNS / "seeds"
        _, runs = gauge_runs_meta()
        for n, r in sorted(runs.items()):
            if r.get("status") != "complete":   # diverged / failed runs are not evaluated
                continue
            m.append({"name": n, "path": G / n / "checkpoints" / "epoch-10", "arm": r["arm"],
                      "seed": r["seed"], "trajectory_exact": r.get("trajectory_exact", True)})
        mt = matched_epochs(seeds)
        if mt["e_A"] != 10:
            m.append({"name": f"A_epoch{mt['e_A']}", "path": L / mt["A"] / "checkpoints" /
                      f"epoch-{mt['e_A']}", "arm": "A_matched", "seed": seeds[0]})
        if mt["e_B"] != 10:
            m.append({"name": f"B_epoch{mt['e_B']}", "path": G / mt["B"] / "checkpoints" /
                      f"epoch-{mt['e_B']}", "arm": "B_matched", "seed": seeds[0]})
    missing = [x["name"] for x in m if not x["path"].exists()]
    if missing:
        raise SystemExit(f"stage {stage}: missing checkpoints: {missing}")
    return m


# ----------------------------------------------------------------------------------
# worker: per-sequence summaries over all models
# ----------------------------------------------------------------------------------
def worker(stage, w, n_workers, threads):
    import torch

    from scripts.mxfold2_counting import MX
    from scripts.mxfold2_decompose import (
        build_model,
        constrained,
        dot_bracket,
        nested,
        positional_value,
        predict,
        read_bpseq,
    )
    from scripts.mxfold2_gauge import Geometry, Masks, canonicalise
    from scripts.mxfold2_seed_study import recorded_seeds

    torch.set_num_threads(threads)
    seeds = recorded_seeds()
    models = model_list(stage, seeds)
    nets = []
    for mm in models:
        net = build_model(LIVE / "arch.conf")
        net.load_state_dict(torch.load(mm["path"], map_location="cpu",
                                       weights_only=True)["model_state_dict"])
        net.eval()
        nets.append(net)
    study = json.loads((LIVE / "metrics.json").read_text())
    entries = study["evaluation_set"]["entries"]
    mx = MX("C")
    K = len(models)
    keys = ("Sin", "Sout", "Min", "Mout", "U", "Ulow", "HL")
    rec = {k: [] for k in ("x", "Q_tot", "Q_G", "kappa", "tp", "fp", "fn", "rand_share",
                           "dimG", "dimT", "h_in_G")}
    val = [{"V1_n": 0, "V1_fail": 0, "V1_max_rel": 0.0, "V1_excluded": 0, "V2_n": 0,
            "V2_fail": 0, "V2_max_rel": 0.0, "V3_n": 0, "V3_detected": 0, "C1_identical": 0,
            "C1_explained": 0, "C1_unexplained": [], "C1_dE_excess_max": -1.0,
            "C1_dE_ref_excess_max": -1.0}
           for _ in models]
    t0 = time.time()
    for xi, e in enumerate(entries):
        if xi % n_workers != w:
            continue
        seq, partner = read_bpseq(DATA / e["file"])
        assert hashlib.sha256(seq.encode()).hexdigest() == e["sha256"], e["file"]
        g, mk = Geometry(seq), Masks(seq)
        ref_db = dot_bracket(partner)
        ref_pairs = {(i, partner[i]) for i in range(1, len(partner)) if partner[i] > i}
        realisable = nested(partner)
        flat, proj = [], []
        for k, (mm, net) in enumerate(zip(models, nets, strict=True)):
            # one RNG stream per (sequence, model): a model's checks do not depend on which
            # other models the stage contains, so 2a reproduces inside 2b
            rng = np.random.default_rng(np.random.SeedSequence(
                [BOOT_SEED, xi, zlib.crc32(mm["name"].encode())]))
            with torch.no_grad():
                param = net.make_param([seq])
            pp = {kk: v.detach().cpu() for kk, v in param[0]["positional"].items()}
            th = g.theta(pp)
            flat.append(np.concatenate([th[kk] for kk in keys]))
            pg = g.project(th)
            proj.append(np.concatenate([pg[kk] for kk in keys]))
            s_pred, db_pred, bp_pred = predict(net, seq, param)
            pp_pairs = {(i, bp_pred[i]) for i in range(1, len(bp_pred)) if bp_pred[i] > i}
            rec_tp = len(pp_pairs & ref_pairs)
            rec.setdefault("_tp", []).append(rec_tp)
            rec.setdefault("_fp", []).append(len(pp_pairs - ref_pairs))
            rec.setdefault("_fn", []).append(len(ref_pairs - pp_pairs))
            v = val[k]
            q = canonicalise(pp, mk)
            pc = [{"turner": param[0]["turner"], "positional": q}]
            # V1 (decompose.py logic)
            if realisable:
                s_mix, db_mix = constrained(net, seq, param, partner)
                if db_mix == ref_db:
                    with torch.no_grad():
                        s_tur, _, _ = net.turner([seq], constraint=[torch.tensor(partner)],
                                                 max_internal_length=None)
                    pos = s_mix - float(s_tur[0])
                    rc = sum(c * positional_value(kk, pp) for kk, c in
                             mx.counts(seq, ref_db).items())
                    v["V1_n"] += 1
                    v["V1_max_rel"] = max(v["V1_max_rel"], abs(rc - pos) / max(1.0, abs(pos)))
                    v["V1_fail"] += int(abs(rc - pos) > REL_TOL * max(1.0, abs(pos)))
                    # C1 on the reference structure (P0.4 tolerance)
                    e_ref_c = constrained(net, seq, pc, partner)[0]
                    v["C1_dE_ref_excess_max"] = max(
                        v["C1_dE_ref_excess_max"],
                        abs(e_ref_c - s_mix) - (1e-6 * max(1.0, abs(s_mix)) + 2e-5))
                else:
                    v["V1_excluded"] += 1
            else:
                v["V1_excluded"] += 1
            targets = [bp_pred] + ([partner] if realisable else [])
            base = [constrained(net, seq, param, t)[0] for t in targets]
            gg = g.random_in_G(rng)
            pg_ = [{"turner": param[0]["turner"], "positional": g.apply(pp, gg)}]
            new = [constrained(net, seq, pg_, t)[0] for t in targets]
            d2 = max(abs(a - b) / max(1.0, abs(a)) for a, b in zip(base, new, strict=True))
            v["V2_n"] += 1
            v["V2_max_rel"] = max(v["V2_max_rel"], d2)
            v["V2_fail"] += int(d2 > REL_TOL)
            z = {kk: rng.standard_normal(len(xx)) for kk, xx in th.items()}
            z["Sin"] *= g.e_in
            z["Sout"] *= g.e_out
            pz = g.project(z)
            z = {kk: z[kk] - pz[kk] for kk in z}
            pz_ = [{"turner": param[0]["turner"], "positional": g.apply(pp, z)}]
            new = [constrained(net, seq, pz_, t)[0] for t in targets]
            d3 = max(abs(a - b) / max(1.0, abs(a)) for a, b in zip(base, new, strict=True))
            v["V3_n"] += 1
            v["V3_detected"] += int(d3 > REL_TOL)
            # C1: canonicalised predictions (P0.4 near-tie rule and tolerance)
            s_c, db_c, _ = predict(net, seq, pc)
            tolE = 1e-6 * max(1.0, abs(s_pred)) + 2e-5
            v["C1_dE_excess_max"] = max(v["C1_dE_excess_max"], abs(s_c - s_pred) - tolE)
            if db_c == db_pred:
                v["C1_identical"] += 1
            else:
                alt = [0] * (len(seq) + 1)
                st = []
                for i_, ch in enumerate(db_c, start=1):
                    if ch == "(":
                        st.append(i_)
                    elif ch == ")":
                        j_ = st.pop()
                        alt[i_], alt[j_] = j_, i_
                e_alt, _ = constrained(net, seq, param, alt)
                if abs(e_alt - s_pred) <= 1e-5 * max(1.0, abs(s_pred)) + 2e-5:
                    v["C1_explained"] += 1
                else:
                    v["C1_unexplained"].append({"x": xi, "gap": abs(e_alt - s_pred)})
        F = np.stack(flat)
        P = np.stack(proj)
        # Gram entries as individual dot products, centred on the stage's FIRST model (the
        # same model in both stages): no GEMM blocking, so shared entries are bitwise equal
        Fc, Pc = F - F[0], P - P[0]
        Qt_x = np.array([[float(np.dot(Fc[i], Fc[j])) for j in range(K)] for i in range(K)])
        QG_x = np.array([[float(np.dot(Pc[i], Pc[j])) for j in range(K)] for i in range(K)])
        rng = np.random.default_rng(np.random.SeedSequence([BOOT_SEED, xi, 0]))
        # c_x: +1 on every S and M entry, +(b - a + 1) on U(a, b), 0 on U(a, a-1) and HL
        c = {"Sin": g.e_in.astype(float), "Sout": g.e_out.astype(float),
             "Min": np.ones(len(g.I)), "Mout": np.ones(len(g.I)),
             "U": (g.ub - g.ua + 1).astype(float), "Ulow": np.zeros(len(g.la)),
             "HL": np.zeros(len(g.hw))}
        pc_ = g.project(c)
        cp = np.concatenate([c[kk] - pc_[kk] for kk in keys])
        kappa = np.array([float(np.dot(F[i], cp)) for i in range(K)]) / float(np.dot(cp, cp))
        shares, rnull, rtot = [], 0.0, 0.0
        for _ in range(N_RANDOM_DIRS):
            r = {kk: rng.standard_normal(len(xx)) for kk, xx in th.items()}
            r["Sin"] *= g.e_in
            r["Sout"] *= g.e_out
            a_, b_ = g.proj_norm2(r), g.norm2(r)
            shares.append(a_ / b_)
            rnull, rtot = rnull + a_, rtot + b_
        dim_t = int(g.e_in.sum() + g.e_out.sum() + 2 * len(g.I) + len(g.ua) + len(g.la)
                    + len(g.hw))
        rec["x"].append(xi)
        rec["Q_tot"].append(Qt_x)
        rec["Q_G"].append(QG_x)
        rec["kappa"].append(kappa)
        for nm in ("tp", "fp", "fn"):
            rec[nm].append(np.array(rec.pop(f"_{nm}")))
        rec["rand_share"].append(float(np.mean(shares)))
        rec.setdefault("rand_null", []).append(rnull)
        rec.setdefault("rand_tot", []).append(rtot)
        rec["dimG"].append(len(g.I) + int(g.h_in_G))
        rec["dimT"].append(dim_t)
        rec["h_in_G"].append(bool(g.h_in_G))
    sd = OUT / "shards" / stage
    sd.mkdir(parents=True, exist_ok=True)
    np.savez(sd / f"w{w}.tmp.npz", **{k: np.array(v) for k, v in rec.items()})
    os.replace(sd / f"w{w}.tmp.npz", sd / f"w{w}.npz")
    (sd / f"w{w}.json").write_text(json.dumps({
        "models": [mm["name"] for mm in models], "validity": val, "K": K, "worker": w,
        "n_workers": n_workers, "threads": threads,
        "evaluator_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "trajectory_exact": {mm["name"]: mm.get("trajectory_exact", True) for mm in models},
        "seconds": time.time() - t0}, default=str))
    return 0


# ----------------------------------------------------------------------------------
# reduction
# ----------------------------------------------------------------------------------
def seed_variance(Q, idx, counts=None):
    """Sample variance over a (multi)set of models, per x, from the Gram matrices Q (X,K,K)."""
    sub = Q[:, idx][:, :, idx]
    n = len(idx)
    c = np.ones(n) if counts is None else counts
    m = c.sum()
    diag = np.einsum("xii,i->x", sub, c)
    quad = np.einsum("i,xij,j->x", c, sub, c)
    return (diag - quad / m) / (m - 1)


def paired_boot(QA, QB, ia, ib, stat, n_boot=BOOT):
    """Paired seed bootstrap of median over x of stat(VA, VB); drops single-seed draws."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(ia)
    vals, dropped = [], 0
    for _ in range(n_boot):
        draw = rng.integers(0, n, n)
        cnt = np.bincount(draw, minlength=n).astype(float)
        if (cnt > 0).sum() < 2:
            dropped += 1
            continue
        keep = cnt > 0
        VA = seed_variance(QA, np.array(ia)[keep], cnt[keep])
        VB = seed_variance(QB, np.array(ib)[keep], cnt[keep])
        vals.append(stat(VA, VB))
    vals = np.array(vals)
    return {"ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
            "n_resamples": int(len(vals)), "n_dropped_single_seed": dropped}


def jackknife(QA, QB, ia, ib, stat):
    n = len(ia)
    full = stat(seed_variance(QA, ia), seed_variance(QB, ib))
    loo = np.array([stat(seed_variance(QA, [ia[j] for j in range(n) if j != i]),
                         seed_variance(QB, [ib[j] for j in range(n) if j != i]))
                    for i in range(n)])
    se = math.sqrt((n - 1) / n * float(((loo - loo.mean()) ** 2).sum()))
    return {"estimate": float(full), "se": se, "bias": float((n - 1) * (loo.mean() - full))}


def med_log_ratio(VA, VB):
    ok = (VA > 0) & (VB > 0)
    return float(np.median(np.log(VA[ok] / VB[ok])))


def kappa_var(K_, idx, counts=None):
    sub = K_[:, idx]
    c = np.ones(len(idx)) if counts is None else counts
    m = c.sum()
    mean = (sub * c).sum(axis=1) / m
    return ((sub - mean[:, None]) ** 2 * c).sum(axis=1) / (m - 1)


def accuracy(tp, fp, fn):
    per = []
    for a, b, c in zip(tp, fp, fn, strict=True):
        if a + c == 0:
            continue
        ppv = a / (a + b) if a + b else 0.0
        sen = a / (a + c)
        per.append(2 * ppv * sen / (ppv + sen) if a else 0.0)
    T, Fp, Fn = tp.sum(), fp.sum(), fn.sum()
    ppv, sen = T / (T + Fp), T / (T + Fn)
    return {"n_scored": len(per), "n_without_reference_pairs": int(len(tp) - len(per)),
            "F_mean_per_sequence": float(np.mean(per)), "PPV_pooled": float(ppv),
            "SEN_pooled": float(sen), "F_pooled": float(2 * ppv * sen / (ppv + sen))}


def pair_split(Qt, QG, a, b):
    """Pooled ||theta_a - theta_b||^2 over x and its null part."""
    tot = float((Qt[:, a, a] + Qt[:, b, b] - 2 * Qt[:, a, b]).sum())
    nul = float((QG[:, a, a] + QG[:, b, b] - 2 * QG[:, a, b]).sum())
    return {"total": tot, "null": nul, "null_share": nul / tot if tot > 0 else None}


def load_shards(stage):
    sd = OUT / "shards" / stage
    pat = re.compile(r"^w(\d+)\.(npz|json)$")
    files = {(int(m_.group(1)), m_.group(2)): f for f in sd.iterdir()
             if (m_ := pat.match(f.name))}
    js = sorted(k for k, e in files if e == "json")
    nz = sorted(k for k, e in files if e == "npz")
    metas = [json.loads(files[(k, "json")].read_text()) for k in js]
    n_w = metas[0]["n_workers"]
    assert js == nz == list(range(n_w)), f"shards {js} / {nz} do not match {n_w} workers"
    assert len({m_["evaluator_sha256"] for m_ in metas}) == 1, "shards from different code"
    arrs = [np.load(files[(k, "npz")]) for k in nz]
    names = metas[0]["models"]
    assert all(m_["models"] == names for m_ in metas)
    xs = np.concatenate([a["x"] for a in arrs])
    n_x = len(json.loads((LIVE / "metrics.json").read_text())["evaluation_set"]["entries"])
    assert np.array_equal(np.sort(xs), np.arange(n_x)), "sequence coverage is not exact"
    order = np.argsort(xs)
    cat = {k: np.concatenate([a[k] for a in arrs])[order] for k in arrs[0].files}
    return names, metas, cat


def restrict(names, metas, cat, keep):
    """The shard data restricted to the models in `keep`, in `keep` order."""
    ii = [names.index(n) for n in keep]
    cat2 = dict(cat)
    for k in ("Q_tot", "Q_G"):
        cat2[k] = cat[k][:, ii][:, :, ii]
    for k in ("kappa", "tp", "fp", "fn"):
        cat2[k] = cat[k][:, ii]
    metas = [{**m_, "trajectory_exact": {n: m_.get("trajectory_exact", {}).get(n, True)
                                         for n in keep}} for m_ in metas]
    metas2 = [{**m_, "models": keep, "validity": [m_["validity"][i] for i in ii]} for m_ in metas]
    return keep, metas2, cat2


def boot_kappa(KAP, ia, ib):
    """Paired seed bootstrap and jackknife of the median log kappa-variance ratio (W2)."""
    def kstat(xa, xb, ca=None):
        va, vb = kappa_var(KAP, xa, ca), kappa_var(KAP, xb, ca)
        okk = (va > 0) & (vb > 0)
        return float(np.median(np.log(va[okk] / vb[okk])))
    n = len(ia)
    rng = np.random.default_rng(BOOT_SEED)
    kb, dropped = [], 0
    for _ in range(BOOT):
        cnt = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        if (cnt > 0).sum() < 2:
            dropped += 1
            continue
        keep = cnt > 0
        kb.append(kstat(list(np.array(ia)[keep]), list(np.array(ib)[keep]), cnt[keep]))
    full = kstat(ia, ib)
    loo = np.array([kstat([ia[j] for j in range(n) if j != i], [ib[j] for j in range(n) if j != i])
                    for i in range(n)])
    se = math.sqrt((n - 1) / n * float(((loo - loo.mean()) ** 2).sum()))
    return {"estimate_median_log_ratio": full,
            "ci95": [float(np.quantile(kb, 0.025)), float(np.quantile(kb, 0.975))],
            "n_resamples": len(kb), "n_dropped_single_seed": dropped,
            "jackknife": {"estimate": full, "se": se, "bias": float((n - 1) * (loo.mean() - full))},
            "larger_at_first_arm": bool(np.quantile(kb, 0.025) > 0)}


def all_pairs(Qt, QG, ids):
    return [pair_split(Qt, QG, ids[i], ids[j]) for i in range(len(ids))
            for j in range(i + 1, len(ids))]


def floor_vs_arm(Qt, QG, floor, ids):
    """A same-seed floor against its arm's seed variance (a pair distance is 2 x variance)."""
    VG, VT = seed_variance(QG, ids).sum(), seed_variance(Qt, ids).sum()
    VI = VT - VG
    fi = floor["total"] - floor["null"]
    return {**floor, "arm_V_tot_sum": float(VT), "arm_V_G_sum": float(VG),
            "arm_V_I_sum": float(VI),
            "floor_over_2V_total": floor["total"] / (2 * VT) if VT > 0 else None,
            "floor_over_2V_null": floor["null"] / (2 * VG) if VG > 0 else None,
            "floor_over_2V_identifiable": fi / (2 * VI) if VI > 0 else None}


def core_stats(stage, seeds, ix, names, cat, acc, exclude=()):
    """Every registered statistic, on the runs not in `exclude` (seed pairing preserved:
    a seed is dropped from a paired comparison if either of its runs is excluded)."""
    Qt, QG, KAP = cat["Q_tot"], cat["Q_G"], cat["kappa"]
    QI = Qt - QG   # Gram of the identifiable part (P_G is an orthogonal projection)
    s0 = seeds[0]

    def arm_ids(prefix_fmt):
        return {s: ix[prefix_fmt.format(s=s)] for s in seeds
                if prefix_fmt.format(s=s) in ix and prefix_fmt.format(s=s) not in exclude}
    arms = {"adamw_w1.0": arm_ids("adamw_w1.0_seed{s}"),
            "adamw_w0.125": arm_ids("adamw_w0.125_seed{s}")}
    if stage == "2b":
        arms["reg"] = arm_ids("reg_seed{s}_w1.0")
        arms["sgd"] = arm_ids("sgd_seed{s}_w1.0")
    res = {"excluded": sorted(exclude), "arms": {}}
    for a, d in arms.items():
        ids = list(d.values())
        if len(ids) < 2:
            res["arms"][a] = {"n_seeds": len(ids)}
            continue
        VG, VT = seed_variance(QG, ids), seed_variance(Qt, ids)
        fs = [acc[names[i]]["F_mean_per_sequence"] for i in ids]
        res["arms"][a] = {"n_seeds": len(ids), "seeds": sorted(d),
                          "V_G_sum": float(VG.sum()), "V_tot_sum": float(VT.sum()),
                          "V_I_sum": float((VT - VG).sum()),
                          "f_G_pooled": float(VG.sum() / VT.sum()),
                          "f_G_median": float(np.median(VG / VT)),
                          "F_mean": float(np.mean(fs)), "F_sd": float(np.std(fs, ddof=1))}

    def paired(da, db):
        common = sorted(set(da) & set(db))
        return [da[s] for s in common], [db[s] for s in common], common
    a2, a1, common = paired(arms["adamw_w0.125"], arms["adamw_w1.0"])
    w1b = paired_boot(QG, QG, a2, a1, med_log_ratio)
    res["W1"] = {"seeds": common,
                 "estimate": med_log_ratio(seed_variance(QG, a2), seed_variance(QG, a1)),
                 **w1b, "jackknife": jackknife(QG, QG, a2, a1, med_log_ratio),
                 "same": bool(math.log(0.8) <= w1b["ci95"][0]
                              and w1b["ci95"][1] <= math.log(1.25))}
    VG2, VG1 = seed_variance(QG, a2), seed_variance(QG, a1)
    ok = (VG2 > 0) & (VG1 > 0)
    lr = np.log(VG2[ok] / VG1[ok])
    rng = np.random.default_rng(BOOT_SEED)
    sb = [np.median(lr[rng.integers(0, len(lr), len(lr))]) for _ in range(BOOT)]
    res["W1"]["superseded_sequence_bootstrap_ci95"] = [float(np.quantile(sb, 0.025)),
                                                       float(np.quantile(sb, 0.975))]
    res["W2"] = {"seeds": common, "arms": "first = w0.125, second = w1.0",
                 **boot_kappa(KAP, a2, a1)}
    res["W2"]["larger_at_0.125"] = res["W2"].pop("larger_at_first_arm")
    # G3: every floor against its own arm's seed variance; epoch-1 floors against
    # epoch-1 seed pairs
    res["G3"] = {}
    for w in ("1.0", "0.125"):
        o, d = f"adamw_w{w}_seed{s0}", f"adamw_w{w}_seed{s0}_dup"
        if o in ix and d in ix and o not in exclude and d not in exclude:
            res["G3"][f"floor_epoch10_w{w}"] = floor_vs_arm(
                Qt, QG, pair_split(Qt, QG, ix[o], ix[d]), list(arms[f"adamw_w{w}"].values()))
    e1o = {s: ix[f"e1orig_seed{s}"] for s in seeds}
    e1l = {s: ix[f"e1live_seed{s}"] for s in seeds}
    res["G3"]["floor_epoch1"] = {str(s): floor_vs_arm(Qt, QG, pair_split(Qt, QG, e1o[s], e1l[s]),
                                                      list(e1l.values())) for s in seeds}
    res["G3"]["seed_pairs_epoch10_adamw_w1.0"] = all_pairs(Qt, QG, list(arms["adamw_w1.0"]
                                                                        .values()))
    if stage == "2b":
        o, d = f"reg_seed{s0}_w1.0", f"reg_seed{s0}_w1.0_dup"
        if o in ix and d in ix and o not in exclude and d not in exclude:
            res["G3"]["floor_epoch10_reg"] = floor_vs_arm(
                Qt, QG, pair_split(Qt, QG, ix[o], ix[d]), list(arms["reg"].values()))
        rg, a1r, common_r = paired(arms["reg"], arms["adamw_w1.0"])
        if len(common_r) >= 2:
            for key, Qx in (("R1_null", QG), ("R2_identifiable", QI)):
                b_ = paired_boot(Qx, Qx, rg, a1r, med_log_ratio)
                res[key] = {"seeds": common_r,
                            "estimate": med_log_ratio(seed_variance(Qx, rg),
                                                      seed_variance(Qx, a1r)), **b_,
                            "jackknife": jackknife(Qx, Qx, rg, a1r, med_log_ratio),
                            "smaller": bool(b_["ci95"][1] < 0)}
        mt = matched_epochs(seeds)
        a_name = f"A_epoch{mt['e_A']}" if f"A_epoch{mt['e_A']}" in ix else f"adamw_w1.0_seed{s0}"
        b_name = f"B_epoch{mt['e_B']}" if f"B_epoch{mt['e_B']}" in ix else mt["B"]
        if a_name in ix and b_name in ix and b_name not in exclude:
            ab = pair_split(Qt, QG, ix[a_name], ix[b_name])
            res["G1"] = {"A": a_name, "B": b_name,
                         "matching": {k: v for k, v in mt.items() if not k.startswith("losses")},
                         "A_vs_B": ab,
                         "baseline_isotropic_pooled": float(cat["dimG"].sum() /
                                                            cat["dimT"].sum()),
                         "baseline_random_pooled": float(cat["rand_null"].sum() /
                                                         cat["rand_tot"].sum()),
                         "baseline_random_mean_per_x": float(np.mean(cat["rand_share"])),
                         "baseline_adamw_all_pairs": [x["null_share"] for x in
                                                      all_pairs(Qt, QG, list(arms["adamw_w1.0"]
                                                                             .values()))],
                         "baseline_sgd_all_pairs": [x["null_share"] for x in
                                                    all_pairs(Qt, QG, list(arms["sgd"]
                                                                           .values()))],
                         "mostly_null": bool(ab["null_share"] is not None
                                             and ab["null_share"] > 0.5)}
        else:
            res["G1"] = {"status": "not computed: A or B missing or excluded",
                         "A": a_name, "B": b_name}
    return res


def v3_amended(stage, name, missed):
    """V3 under amendment 5: a pair misses only if directions 0, 1 and 2 all fail. Directions
    1 and 2 come from the stage's re-draw artifact (scripts/mxfold2_v3_redraw.py), drawn only
    where direction 0 missed, which the rule's symmetry makes equivalent to drawing all three
    everywhere. Fails closed: a model with direction-0 misses passes only if exactly one
    complete re-draw artifact for the stage reproduces its recorded misses and none of them
    misses all three directions."""
    if missed == 0:
        return {"pass": True, "direction0_misses": 0}
    arts = sorted(R.glob(f"runs/*_mxfold2_v3_redraw_{stage}/metrics.json"))
    if len(arts) != 1:
        return {"pass": False, "direction0_misses": missed,
                "reason": f"{len(arts)} re-draw artifacts for stage {stage}, need exactly one"}
    a = json.loads(arts[0].read_text())
    m = a.get("per_model", {}).get(name)
    ok = (a.get("status") == "complete" and m is not None
          and m["recorded_direction0_misses"] == missed and m["reproduces_recorded"]
          and a.get("provenance_of_shards", {}).get("worker_unchanged") is True)
    return {"pass": bool(ok and m["amended_misses"] == 0), "direction0_misses": missed,
            "amended_misses": None if m is None else m["amended_misses"],
            "redraw_artifact": str(arts[0].relative_to(R)),
            "redraw_artifact_sha256": hashlib.sha256(arts[0].read_bytes()).hexdigest(),
            "redraw_consistent": bool(ok)}


def reduce(stage, names=None, metas=None, cat=None):
    from scripts.mxfold2_seed_study import recorded_seeds
    seeds = recorded_seeds()
    if names is None:
        names, metas, cat = load_shards(stage)
    X = len(cat["x"])
    val = {}
    for i, n in enumerate(names):
        agg = {}
        for m_ in metas:
            for k, v in m_["validity"][i].items():
                if isinstance(v, list):
                    agg[k] = agg.get(k, []) + v
                elif k.endswith("_max_rel") or k.endswith("_excess_max"):
                    agg[k] = max(agg.get(k, -1.0), v)
                else:
                    agg[k] = agg.get(k, 0) + v
        # amendment 5 (PREREG analysis.V3_amendment_2026_10_11): the registered
        # single-direction V3 result is kept; V3 passes under the three-direction rule
        agg["V3_single_direction_pass"] = bool(agg["V3_detected"] == agg["V3_n"])
        agg["V3_amended"] = v3_amended(stage, n, agg["V3_n"] - agg["V3_detected"])
        agg["V_pass"] = bool(agg["V1_fail"] == 0 and agg["V2_fail"] == 0 and agg["V1_n"] > 0
                             and agg["V3_amended"]["pass"])
        agg["C1_pass"] = bool(not agg["C1_unexplained"] and agg["C1_dE_excess_max"] <= 0.0
                              and agg["C1_dE_ref_excess_max"] <= 0.0)
        val[n] = agg
    out = {"stage": stage, "n_sequences": X, "models": names, "validity": val,
           "V3_single_direction_all_pass": all(v["V3_single_direction_pass"]
                                               for v in val.values()),
           "evaluator_sha256_shards": sorted({m_["evaluator_sha256"] for m_ in metas}),
           "evaluator_sha256_reducer": hashlib.sha256(pathlib.Path(__file__).read_bytes())
           .hexdigest()}
    from scripts.mxfold2_gauge_study import check_prereg
    out["provenance"] = check_prereg(require_clean_tree=False)
    if not all(v["V_pass"] and v["C1_pass"] for v in val.values()):
        out["status"] = "STOPPED: V1-V3 or C1 failed on some model; no statistic reported"
        return out
    ix = {n: i for i, n in enumerate(names)}
    exact = {}
    for m_ in metas:
        exact.update(m_.get("trajectory_exact", {}))
    res = {"F_per_model": {n: accuracy(cat["tp"][:, i], cat["fp"][:, i], cat["fn"][:, i])
                           for n, i in ix.items()},
           "flagged_runs": sorted(n for n, e in exact.items() if e is False)}
    res["main"] = core_stats(stage, seeds, ix, names, cat, res["F_per_model"], exclude=())
    if res["flagged_runs"]:
        res["flagged_excluded"] = core_stats(stage, seeds, ix, names, cat, res["F_per_model"],
                                             exclude=set(res["flagged_runs"]))
    res["S2"] = s2_adamw_traces(seeds)
    if stage == "2b":
        res["S1"] = s1_sgd(seeds)
    out["results"] = res
    out["status"] = "complete"
    return out


# ----------------------------------------------------------------------------------
# S1 / S2: the weight-space gauge coordinate v.w
# ----------------------------------------------------------------------------------
SGD_LR, SGD_MU = 1e-3, 0.9


def init_vdot(seed):
    """v.w at initialisation, rebuilt as mxfold2/train.py does: seed, then build_model."""
    import random

    import torch

    from scripts.mxfold2_decompose import build_model
    from scripts.mxfold2_gauge import v_dot
    torch.manual_seed(seed)
    random.seed(seed)
    return v_dot(build_model(LIVE / "arch.conf").state_dict())


def ckpt_vdot(path):
    import torch

    from scripts.mxfold2_gauge import v_dot
    return v_dot(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"])


def s2_adamw_traces(seeds, max_epoch=10):
    """Seed SD of v.w per epoch (0 = initialisation) for the live AdamW arms."""
    out = {}
    for w in ("1.0", "0.125"):
        tr = {}
        for s in seeds:
            t = {0: init_vdot(s)}
            for e in range(1, max_epoch + 1):
                ck = LIVE / "seeds" / f"full_seed{s}_w{w}" / "checkpoints" / f"epoch-{e}"
                if ck.exists():
                    t[e] = ckpt_vdot(ck)
            tr[str(s)] = t
        eps = sorted(set.intersection(*[set(t) for t in tr.values()]))
        sd = {str(e): float(np.std([tr[str(s)][e] for s in seeds], ddof=1)) for e in eps}
        step = [abs(tr[str(s)][e] - tr[str(s)][e - 1]) for s in seeds for e in eps if e >= 1]
        out[f"adamw_w{w}"] = {"traces": {k: {str(e): v for e, v in t.items()} for k, t in
                                         tr.items()},
                              "seed_sd_by_epoch": sd,
                              "mean_abs_epoch_step": float(np.mean(step)) if step else None,
                              "grows_0_to_last": bool(sd[str(eps[-1])] > sd["0"])}
    return out


def s1_sgd(seeds):
    """Registered S1 (amendment 2): reconstruction and conservation for every SGD run."""
    _, runs = gauge_runs_meta()
    names = [n for n, r in runs.items()
             if r.get("optimizer") == "SGD" and r.get("status") == "complete"]
    res = {}
    for n in sorted(names):
        d = GRUNS / "seeds" / n
        g0 = json.loads((d / "gauge" / "gauge-epoch-1.json").read_text())
        v0 = g0["v_dot_w_start"]["v_dot_w"]
        seed = int(n.split("seed")[1].split("_")[0])
        actual = {0: v0}
        recon, longonly = {0: v0}, {0: v0}
        r_full, r_long, b_full, b_long, first = v0, v0, 0.0, 0.0, True
        for e in range(1, 11):
            npz = d / "gauge" / f"steps-epoch-{e}.npz"
            js = d / "gauge" / f"gauge-epoch-{e}.json"
            if not (npz.exists() and js.exists()):
                break
            z = np.load(npz)
            for vg, bw, lh in zip(z["vg"], z["backward"], z["long_helix"], strict=True):
                if not bw:
                    continue
                gl = vg if lh else 0.0
                if first:
                    b_full, b_long, first = vg, gl, False
                else:
                    b_full, b_long = SGD_MU * b_full + vg, SGD_MU * b_long + gl
                r_full -= SGD_LR * (vg + SGD_MU * b_full)
                r_long -= SGD_LR * (gl + SGD_MU * b_long)
            actual[e] = json.loads(js.read_text())["v_dot_w"]
            recon[e], longonly[e] = r_full, r_long
        rng_ = max(actual.values()) - min(actual.values())
        dev = max(abs(actual[e] - recon[e]) for e in actual)
        last = max(actual)
        v_init = init_vdot(seed)
        res[n] = {"seed": seed, "init_matches_rebuild":
                  abs(v0 - v_init) <= 1e-12 * max(1.0, abs(v0)),
                  "actual": {str(k): v for k, v in actual.items()},
                  "reconstruction": {str(k): v for k, v in recon.items()},
                  "long_only": {str(k): v for k, v in longonly.items()},
                  "max_reconstruction_dev": dev, "trace_range": rng_,
                  "i_reconstruction_ok": bool(dev <= 1e-3 + 1e-6 * rng_),
                  "conservation_residual": abs(actual[last] - longonly[last]),
                  "ii_conservation_ok": bool(last == 10 and
                                             abs(actual[last] - longonly[last]) <= 1e-2),
                  "trajectory_exact": runs[n].get("trajectory_exact", True)}
    exact = {n: r for n, r in res.items() if r["trajectory_exact"]}
    ok = lambda rr: bool(rr and all(r["i_reconstruction_ok"] and r["ii_conservation_ok"]  # noqa: E731
                                    for r in rr.values()))
    return {"runs": res, "S1": ok(res), "S1_flagged_excluded": ok(exact)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("2a", "2b"), required=True)
    ap.add_argument("--worker", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--reduce-only", action="store_true")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.worker is not None:
        return worker(a.stage, a.worker, a.workers, a.threads)
    from scripts.mxfold2_gauge_study import check_prereg
    prov = check_prereg()
    from scripts.runmeta import git_provenance  # noqa: F401  (recorded via check_prereg)
    t0 = time.time()
    path = OUT / "metrics.json"
    prior = json.loads(path.read_text()) if path.exists() else {}
    if a.stage == "2b" and "2a" in prior and prior["2a"].get("threads") != a.threads:
        raise SystemExit("stage 2b must use the same --threads as 2a for bitwise reproduction "
                         f"(2a used {prior['2a'].get('threads')})")
    if not a.reduce_only:
        import shutil
        shutil.rmtree(OUT / "shards" / a.stage, ignore_errors=True)  # no stale shards
        tenv = {k: str(a.threads) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                            "MKL_NUM_THREADS")}
        procs = [subprocess.Popen([sys.executable, __file__, "--stage", a.stage, "--worker",
                                   str(w), "--workers", str(a.workers), "--threads",
                                   str(a.threads)],
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES="", **tenv))
                 for w in range(a.workers)]
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit(f"worker exit codes {codes}")
    out = reduce(a.stage)
    allm = prior or {"run_id": OUT.name}
    if a.stage == "2b" and "2a" in allm and allm["2a"].get("status") == "complete":
        names, metas, cat = load_shards("2b")
        sub = restrict(names, metas, cat, allm["2a"]["models"])
        again = reduce("2a", *sub)
        out["reproduces_2a_bitwise"] = (json.dumps(again.get("results"), sort_keys=True)
                                        == json.dumps(allm["2a"].get("results"),
                                                      sort_keys=True))
    out.update({"launch_provenance": prov, "workers": a.workers, "threads": a.threads,
                "elapsed_seconds": round(time.time() - t0, 1)})
    allm[a.stage] = out
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json.tmp").write_text(json.dumps(allm, indent=1, sort_keys=True,
                                                     default=str))
    os.replace(OUT / "metrics.json.tmp", path)
    print(json.dumps({k: out.get(k) for k in ("stage", "status", "n_sequences")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
