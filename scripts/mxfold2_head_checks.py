"""Phase 0 of the MXfold2 gauge study: validate the head, the v symmetry and the runner.

Called by `scripts/mxfold2_gauge_study.py --phase phase0` (in .venv-mxfold2, CPU only),
after the pre-registration is committed. Every threshold comes from the registered
PREREG["phase0"], passed in as `cfg`. Output: runs/2026-10-08_mxfold2_gauge_phase0/
metrics.json, written through the study's guarded `write`. Any failure sets
all_pass = false, and every later phase refuses to run.

P0.1 masks: torch Masks == numpy Geometry on every TR0 and TS0 sequence; torch
     null_norm2 == Geometry.proj_norm2 on TS0 theta of the shipped weights.
P0.2 v symmetry: moving the weights along v moves the outputs along h exactly; the
     loss is flat along v (off long-helix steps).
P0.3 head is a training no-op: weight gradients with and without the head agree;
     a non-G projection (positive control) changes them.
P0.4 defining check + negative controls NC1-NC3, all 1,305 TS0 sequences.
P0.5 runner equivalence (subprocesses, CPU): lambda = 0 hooks are bitwise inert; SGD;
     RNG-sidecar resume; stdout parse; lambda > 0 is active; float64 crosscheck of R.
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import types
from argparse import ArgumentParser

import numpy as np
import torch

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import mxfold2.interface as interface  # noqa: E402
from mxfold2.loss import StructuredLossWithTurner  # noqa: E402
from mxfold2.predict import Predict  # noqa: E402

from scripts.mxfold2_decompose import dot_bracket, nested  # noqa: E402
from scripts.mxfold2_gauge import (  # noqa: E402
    BIAS_KEY,
    HELIX_W_KEY,
    MAX_HELIX,
    Geometry,
    Masks,
    canonicalise,
    null_norm2,
    v_dot_grad,
    v_norm2,
)
from scripts.mxfold2_seed_study import compare_checkpoints  # noqa: E402

DATA = R / ".cache" / "mxfold2-data"
MODELS = R / "vendor" / "mxfold2" / "mxfold2" / "models"
LIVE = R / "runs" / "2026-10-07_mxfold2_seed_study"
PY_MX = R / ".venv-mxfold2" / "bin" / "python"
SEED = 20261008
MISMATCH_KEYS = ("score_mismatch_external", "score_mismatch_hairpin", "score_mismatch_internal",
                 "score_mismatch_multi")
BASE_KEYS = ("score_base_hairpin", "score_base_internal", "score_base_multi",
             "score_base_external")


# ----------------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------------
def shipped_model():
    p = ArgumentParser(fromfile_prefix_chars="@")
    sub = p.add_subparsers()
    Predict.add_args(sub)
    model, _ = Predict().build_model(p.parse_args(["predict", f"@{MODELS}/TrainSetAB.conf",
                                                   "x.fa"]))
    model.load_state_dict(torch.load(MODELS / "TrainSetAB.pth", map_location="cpu",
                                     weights_only=True))
    model.eval()
    return model


def bpseq_raw(path):
    rows = [r.split() for r in path.read_text().split("\n") if r.strip() and not r.startswith("#")]
    return "".join(r[1] for r in rows), [0] + [int(r[2]) for r in rows]


def listed(lst):
    return [DATA / ln.split()[0] for ln in (DATA / "data" / lst).read_text().split("\n")
            if ln.strip()]


def loss_fn(model):
    # the train.py defaults: loss_pos_paired 0.5, loss_neg_paired 0.005, sl weight 1.0
    return StructuredLossWithTurner(model, loss_pos_paired=0.5, loss_neg_paired=0.005,
                                    l1_weight=0.0, l2_weight=0.0, sl_weight=1.0)


class Capture:
    """Record the partner lists of the DP calls (pred / ref) inside a block."""

    def __enter__(self):
        self._mx = interface.predict_mxfold
        self.calls = []

        def wrap(*a, **k):
            r = self._mx(*a, **k)
            self.calls.append(("ref" if k.get("constraint") is not None else "pred",
                               float(r[0]), list(r[2])))
            return r

        interface.predict_mxfold = wrap
        return self

    def __exit__(self, *exc):
        interface.predict_mxfold = self._mx


def longest_helix(partner):
    best = run = 0
    prev = None
    for i in range(1, len(partner)):
        j = partner[i]
        if j > i:
            run = run + 1 if prev == (i - 1, j + 1) else 1
            best = max(best, run)
            prev = (i, j)
    return best


def with_positional(param, pos):
    return [{"turner": param[0]["turner"], "positional": pos}]


def install_head(model, head):
    """Instance-level make_param override: positional outputs pass through `head`."""
    orig = type(model).make_param

    def make_param(self, seq):
        r = orig(self, seq)
        return [{"turner": d["turner"], "positional": head(d["positional"], s)}
                for d, s in zip(r, seq, strict=True)]

    model.make_param = types.MethodType(make_param, model)


def remove_head(model):
    model.__dict__.pop("make_param", None)


def flat_grad(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()]).double()


def random_theta_like(p, rng):
    n1 = p["score_helix_stacking"].shape[0]
    t = lambda *s: torch.as_tensor(rng.standard_normal(s), dtype=torch.float32)  # noqa: E731
    return {"S": t(n1, n1), "M": t(n1, n1), "U": t(n1, n1), "hl": t(31)}


def as_param(base, S, M, U, hl):
    q = dict(base)
    q["score_helix_stacking"] = S.float().contiguous()
    for k in MISMATCH_KEYS:
        q[k] = M.float().contiguous()
    for k in BASE_KEYS:
        q[k] = U.float().contiguous()
    q["score_helix_length"] = hl.float().contiguous()
    return q


def theta_mask(mk):
    """Which entries of the dense S, M, U, hl tensors are theta coordinates (decompose's theta):
    S at Ein (upper) and Eout (lower), M at A and A^T, U on the upper triangle a <= b from 1
    and at (a, a-1), helix_length[1..30]."""
    n1 = mk.n + 1
    A = mk.A
    S = mk.Ein | mk.Eout.transpose(0, 1)
    M = A | A.transpose(0, 1)
    idx = torch.arange(n1)
    U = (idx[:, None] <= idx[None, :]) & (idx[:, None] >= 1)
    U = U | ((idx[:, None] - idx[None, :] == 1) & (idx[:, None] >= 2))
    hl = torch.zeros(31, dtype=torch.bool)
    hl[1:MAX_HELIX + 1] = True
    return {"S": S, "M": M, "U": U, "hl": hl}


def perp_direction(base, r, mk):
    """A unit direction (theta-space norm) on theta's coordinates with its G component removed."""
    tm = theta_mask(mk)
    r = {k: torch.where(tm[k], r[k], torch.zeros_like(r[k])) for k in r}
    rp = as_param(base, r["S"], r["M"], r["U"], r["hl"])
    q = canonicalise(rp, mk)
    d = {"S": q["score_helix_stacking"].double(), "M": q["score_mismatch_hairpin"].double(),
         "U": r["U"].double(), "hl": q["score_helix_length"].double()}
    d = {k: torch.where(tm[k], v, torch.zeros_like(v)) for k, v in d.items()}
    nrm = torch.sqrt(sum((v * v).sum() for v in d.values()))
    return {k: v / nrm for k, v in d.items()}


def g_direction(base, r, mk):
    """A direction in G: r minus canonicalise(r), on S, M and hl (U is not in G)."""
    rp = as_param(base, r["S"], r["M"], torch.zeros_like(r["U"]), r["hl"])
    q = canonicalise(rp, mk)
    return {"S": r["S"].double() - q["score_helix_stacking"].double(),
            "M": r["M"].double() - q["score_mismatch_hairpin"].double(),
            "hl": r["hl"].double() - q["score_helix_length"].double()}


def theta_tensors(p):
    return {"S": p["score_helix_stacking"].double(), "M": p["score_mismatch_hairpin"].double(),
            "U": p["score_base_hairpin"].double(), "hl": p["score_helix_length"].double()}


# ----------------------------------------------------------------------------------
# P0.1 masks
# ----------------------------------------------------------------------------------
def p01_masks():
    mism = []
    n = 0
    n_h_dropped = 0
    for lst in ("TR0-canonicals.lst", "TS0-canonicals.lst"):
        for f in listed(lst):
            seq, _ = bpseq_raw(f)
            g, mk = Geometry(seq.upper()), Masks(seq)
            ok = (set(zip(g.I.tolist(), g.J.tolist(), strict=True))
                  == set(zip(*np.nonzero(mk.A.numpy()), strict=True))
                  and np.array_equal(g.e_in, mk.Ein.numpy()[g.I, g.J])
                  and np.array_equal(g.e_out, mk.Eout.numpy()[g.I, g.J])
                  and g.h_in_G == mk.h_in_G)
            n += 1
            n_h_dropped += int(not mk.h_in_G)
            if not ok:
                mism.append(f.name)
    return {"n_sequences": n, "n_mismatch": len(mism), "first_mismatches": mism[:20],
            "n_h_dropped_long_helix": n_h_dropped, "pass": bool(not mism)}


# ----------------------------------------------------------------------------------
# P0.2 v symmetry
# ----------------------------------------------------------------------------------
def p02_v_symmetry(cfg, model):
    c = cfg["P0_2"]
    rng = np.random.default_rng(SEED)
    files = listed("TR0-canonicals.lst")
    pick = [files[i] for i in rng.choice(len(files), c["n_sequences"], replace=False)]
    d = c["delta"]
    m2 = copy.deepcopy(model)
    sd = m2.state_dict()
    sd[BIAS_KEY][0] += d
    sd[HELIX_W_KEY][0, 2:MAX_HELIX + 1] -= 2 * d
    m2.load_state_dict(sd)
    m2.eval()
    hw = 2 * d * torch.clamp(torch.arange(31) - 1, min=0)
    worst = {"table": 0.0, "other": 0.0, "loss_rel": 0.0, "cos": 0.0}
    n_long, rows = 0, []
    lf0, lf2 = loss_fn(model), loss_fn(m2)
    for f in pick:
        seq, partner = bpseq_raw(f)
        with torch.no_grad():
            a = model.make_param([seq])[0]["positional"]
            b = m2.make_param([seq])[0]["positional"]
        tab = max(float((b["score_helix_stacking"] - a["score_helix_stacking"] - d).abs().max()),
                  float((b["score_helix_length"] - a["score_helix_length"] + hw).abs().max()))
        oth = max(float((b["score_mismatch_hairpin"] - a["score_mismatch_hairpin"]).abs().max()),
                  float((b["score_base_hairpin"] - a["score_base_hairpin"]).abs().max()))
        pairs = [torch.tensor(partner)]
        model.zero_grad()
        with Capture() as cap:
            L0 = lf0([seq], pairs).sum()
        L0.backward()
        vg = v_dot_grad(model)
        gn = float(flat_grad(model).norm())
        with torch.no_grad():
            L2 = lf2([seq], pairs).sum()
        long_ = any(longest_helix(cl[2]) > MAX_HELIX for cl in cap.calls)
        n_long += int(long_)
        rel = abs(float(L2) - float(L0)) / max(1.0, abs(float(L0)))
        cos = abs(vg) / (np.sqrt(v_norm2()) * gn) if gn > 0 else 0.0
        worst["table"] = max(worst["table"], tab)
        worst["other"] = max(worst["other"], oth)
        if not long_:
            worst["loss_rel"] = max(worst["loss_rel"], rel)
            worst["cos"] = max(worst["cos"], cos)
        rows.append({"file": f.name, "len": len(seq), "long_helix": bool(long_),
                     "loss_rel": float(rel), "cos": float(cos)})
    worst = {k: float(v) for k, v in worst.items()}
    ok = bool(worst["table"] <= c["table_abs_tol"] and worst["other"] <= c["other_abs_tol"]
              and worst["loss_rel"] <= c["loss_rel_tol"] and worst["cos"] <= c["cos_tol"])
    return {"worst": worst, "n_long_helix_sequences": n_long, "per_sequence": rows,
            "pass": ok}


# ----------------------------------------------------------------------------------
# P0.3 head is a training no-op
# ----------------------------------------------------------------------------------
def grad_with(model, lf, seq, pairs, seed, head):
    """Weight gradient, DP partner lists and loss of one hinge_mix step, optionally headed."""
    remove_head(model)
    if head is not None:
        install_head(model, head)
    model.zero_grad()
    torch.manual_seed(seed)
    with Capture() as cap:
        L = lf([seq], pairs).sum()
    L.backward()
    remove_head(model)
    return flat_grad(model), [cl[2] for cl in cap.calls], float(L)


def canon_head(mk):
    return lambda p, s: canonicalise(p, mk)


def theta_grad(model, lf, seq, pairs, seed):
    """d loss / d theta for one step (S, M, U, helix_length), via an identity head that keeps
    the gradients of the (shared) positional tensors."""
    keep = {}

    def ident(p, s):
        for k, name in (("S", "score_helix_stacking"), ("M", "score_mismatch_hairpin"),
                        ("U", "score_base_hairpin"), ("hl", "score_helix_length")):
            p[name].retain_grad()
            keep[k] = p[name]
        return p

    grad_with(model, lf, seq, pairs, seed, ident)
    return {k: (v.grad if v.grad is not None else torch.zeros_like(v)).detach().float()
            for k, v in keep.items()}


def perp_head(direction):
    """Positive control: project out a fixed unit direction orthogonal to G."""
    def head(p, s):
        th = theta_tensors(p)
        dot = sum((th[n] * direction[n]).sum() for n in direction)
        return as_param(p, th["S"] - dot * direction["S"], th["M"] - dot * direction["M"],
                        th["U"] - dot * direction["U"], th["hl"] - dot * direction["hl"])

    return head


def p03_head_noop(cfg, model):
    c = cfg["P0_3"]
    rng = np.random.default_rng(SEED + 1)
    files = [f for f in listed("TR0-canonicals.lst")]
    rng.shuffle(files)
    pick = []
    for f in files:
        if len(bpseq_raw(f)[0]) <= c["max_len"]:
            pick.append(f)
        if len(pick) == c["n_sequences"]:
            break
    lf = loss_fn(model)
    rows, worst, ctrl_min = [], 0.0, float("inf")
    for k, f in enumerate(pick):
        seq, partner = bpseq_raw(f)
        pairs = [torch.tensor(partner)]
        mk = Masks(seq)
        for mode in ("eval", "train"):
            model.train(mode == "train")
            g0, s0, L0 = grad_with(model, lf, seq, pairs, SEED + k, None)
            g1, s1, L1 = grad_with(model, lf, seq, pairs, SEED + k, canon_head(mk))
            gt = theta_grad(model, lf, seq, pairs, SEED + k)
            with torch.no_grad():
                pp0 = model.make_param([seq])[0]["positional"]
            model.__dict__.pop("_gauge_param", None)
            g2, _, _ = grad_with(model, lf, seq, pairs, SEED + k,
                                 perp_head(perp_direction(pp0, gt, mk)))
            rel = float((g1 - g0).norm() / g0.norm()) if float(g0.norm()) > 0 else 0.0
            crel = float((g2 - g0).norm() / g0.norm()) if float(g0.norm()) > 0 else 0.0
            worst = max(worst, rel)
            ctrl_min = min(ctrl_min, crel)
            rows.append({"file": f.name, "len": len(seq), "mode": mode, "grad_rel": rel,
                         "loss_diff": abs(L1 - L0), "argmax_same": s0 == s1,
                         "control_grad_rel": crel})
    model.eval()
    ok = bool(worst <= c["grad_rel_tol"] and all(r["argmax_same"] for r in rows)
              and ctrl_min >= c["control_min_change"])
    return {"worst_grad_rel": worst, "min_control_grad_rel": ctrl_min,
            "n_argmax_differ": sum(not r["argmax_same"] for r in rows), "per_sequence": rows,
            "pass": ok}


# ----------------------------------------------------------------------------------
# P0.1 (null norm) + P0.4 defining check and negative controls, one TS0 pass
# ----------------------------------------------------------------------------------
def dp(model, seq, param, constraint=None):
    with torch.no_grad():
        if constraint is None:
            sc, pred, bp = model([seq], param=param)
        else:
            sc, pred, bp = model([seq], param=param, constraint=[torch.tensor(constraint)],
                                 max_internal_length=None)
    return float(sc[0]), pred[0], list(bp[0])


def p04_defining(cfg, model, save_progress):
    c, nc = cfg["P0_4"], cfg["P0_4_negative_controls"]["thresholds"]
    rel1 = cfg["P0_1_rel_tol"]
    rng = np.random.default_rng(SEED + 2)
    res = {"n": 0, "null_norm_rel_max": 0.0, "struct_identical": 0, "flips_explained": 0,
           "flips_unexplained": [], "dE_pred_rel_max": 0.0, "dE_ref_rel_max": 0.0,
           "n_ref_realisable": 0, "residual_rel_max": 0.0, "NC1_changed": 0,
           "NC1_struct_changed": 0, "NC2_n_h_in_G": 0, "NC2_ok": 0, "NC3_fail": 0,
           "NC3_max_abs": 0.0}
    def tol_E(e):
        return c["dE_rel_tol"] * max(1.0, abs(e)) + c["abs_floor"]

    def partner_of(db):
        pt = [0] * (len(db) + 1)
        st = []
        for i_, ch in enumerate(db, start=1):
            if ch == "(":
                st.append(i_)
            elif ch == ")":
                j_ = st.pop()
                pt[i_], pt[j_] = j_, i_
        return pt

    files = listed("TS0-canonicals.lst")
    t0 = time.time()
    for n_x, f in enumerate(files):
        seq, partner = bpseq_raw(f)
        mk, g = Masks(seq), Geometry(seq.upper())
        with torch.no_grad():
            param = model.make_param([seq])
        pp = {k: v.detach() for k, v in param[0]["positional"].items()}
        th = g.theta(pp)
        a, b = g.proj_norm2(th), float(null_norm2(pp, mk))
        res["null_norm_rel_max"] = max(res["null_norm_rel_max"], abs(a - b) / max(a, 1e-300))
        q = canonicalise(pp, mk)
        pc = with_positional(param, q)
        e_o, s_o, bp_o = dp(model, seq, param)
        e_c, s_c, _ = dp(model, seq, pc)
        res["dE_pred_rel_max"] = max(res["dE_pred_rel_max"], abs(e_c - e_o) / max(1.0, abs(e_o)))
        res["dE_pred_excess_max"] = max(res.get("dE_pred_excess_max", -1.0),
                                        abs(e_c - e_o) - tol_E(e_o))
        if s_c == s_o:
            res["struct_identical"] += 1
        else:
            pc_partner = [0] * (len(seq) + 1)
            st = []
            for i_, ch in enumerate(s_c, start=1):
                if ch == "(":
                    st.append(i_)
                elif ch == ")":
                    j_ = st.pop()
                    pc_partner[i_], pc_partner[j_] = j_, i_
            e_alt, _, _ = dp(model, seq, param, pc_partner)
            gap = abs(e_alt - e_o)
            if gap <= c["tie_bound"] * max(1.0, abs(e_o)) + c["abs_floor"]:
                res["flips_explained"] += 1
            else:
                res["flips_unexplained"].append({"file": f.name, "gap": gap})
        if nested(partner):
            er_o, sr_o, _ = dp(model, seq, param, partner)
            if sr_o == dot_bracket(partner):
                er_c, _, _ = dp(model, seq, pc, partner)
                res["n_ref_realisable"] += 1
                res["dE_ref_rel_max"] = max(res["dE_ref_rel_max"],
                                            abs(er_c - er_o) / max(1.0, abs(er_o)))
                res["dE_ref_excess_max"] = max(res.get("dE_ref_excess_max", -1.0),
                                               abs(er_c - er_o) - tol_E(er_o))
        thn = g.norm2(th)
        res["residual_rel_max"] = max(res["residual_rel_max"],
                                      float(null_norm2(q, mk)) / max(thn, 1e-300))
        # NC1: shift by ||P_G theta|| along a random direction orthogonal to G
        rdir = perp_direction(pp, random_theta_like(pp, rng), mk)
        amp = float(np.sqrt(b))
        tt = theta_tensors(pp)
        p1 = as_param(pp, tt["S"] - amp * rdir["S"], tt["M"] - amp * rdir["M"],
                      tt["U"] - amp * rdir["U"], tt["hl"] - amp * rdir["hl"])
        e1, _, _ = dp(model, seq, with_positional(param, p1), bp_o)
        res["NC1_changed"] += int(abs(e1 - e_o) > nc["NC1_rel"] * max(1.0, abs(e_o)))
        _, s1, _ = dp(model, seq, with_positional(param, p1))
        res["NC1_struct_changed"] += int(s1 != s_o)
        # NC2: the head without h
        if mk.h_in_G:
            res["NC2_n_h_in_G"] += 1
            full = float(null_norm2(q, mk))
            no_h = float(null_norm2(canonicalise(pp, mk, include_h=False), mk))
            res["NC2_ok"] += int(no_h >= nc["NC2_ratio"] * max(full, 1e-300))
        # NC3: a random G perturbation before the head does not survive it
        gd = g_direction(pp, random_theta_like(pp, rng), mk)
        gn = float(torch.sqrt(sum((v * v).sum() for v in gd.values())))
        scale = float(np.sqrt(thn)) / max(gn, 1e-300)
        p3 = as_param(pp, tt["S"] + scale * gd["S"], tt["M"] + scale * gd["M"], tt["U"],
                      tt["hl"] + scale * gd["hl"])
        q3 = canonicalise(p3, mk)
        diff = max(float((q3[k] - q[k]).abs().max()) for k in
                   ("score_helix_stacking", "score_mismatch_hairpin", "score_helix_length"))
        res["NC3_max_abs"] = max(res["NC3_max_abs"], diff)
        e3, s3, _ = dp(model, seq, with_positional(param, q3))
        tie_ok = True
        if s3 != s_c:
            e_alt3, _, _ = dp(model, seq, pc, partner_of(s3))
            tie_ok = abs(e_alt3 - e_c) <= c["tie_bound"] * max(1.0, abs(e_c)) + c["abs_floor"]
            res["NC3_near_tie_flips"] = res.get("NC3_near_tie_flips", 0) + int(tie_ok)
        res["NC3_fail"] += int(diff > nc["NC3_abs"] or not tie_ok)
        res["n"] += 1
        if n_x % 50 == 0:
            save_progress({"sequences_done": n_x + 1, "of": len(files),
                           "seconds": round(time.time() - t0, 1)})
    n = res["n"]
    res = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in res.items()}
    res["P0_1_null_norm_pass"] = bool(res["null_norm_rel_max"] <= rel1)
    res["defining_pass"] = bool(not res["flips_unexplained"]
                                and res.get("dE_pred_excess_max", -1.0) <= 0.0
                                and res.get("dE_ref_excess_max", -1.0) <= 0.0
                                and res["residual_rel_max"] <= c["residual_rel_tol"])
    res["NC1_pass"] = bool(res["NC1_changed"] >= nc["NC1_frac"] * n
                           and res["NC1_struct_changed"] > 0)
    res["NC2_pass"] = bool(res["NC2_ok"] >= nc["NC2_frac"] * max(res["NC2_n_h_in_G"], 1))
    res["NC3_pass"] = bool(res["NC3_fail"] == 0)
    res["pass"] = bool(all(res[k] for k in ("P0_1_null_norm_pass", "defining_pass", "NC1_pass",
                                            "NC2_pass", "NC3_pass")))
    return res


# ----------------------------------------------------------------------------------
# P0.5 runner equivalence (subprocesses, CPU)
# ----------------------------------------------------------------------------------
def p05_runner(cfg, out_dir):
    c = cfg["P0_5"]
    work = out_dir / "p05"
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 3)
    lines = [ln for ln in (DATA / "data" / "TR0-canonicals.lst").read_text().split("\n")
             if ln.strip()]
    short = [ln for ln in lines if len(bpseq_raw(DATA / ln.split()[0])[0]) <= c["max_len"]]
    pick = [short[i] for i in sorted(rng.choice(len(short), c["n_sequences"], replace=False))]
    lst = work / "train.lst"
    lst.write_text("\n".join(pick) + "\n")
    arch = work / "arch.conf"
    arch.write_bytes((LIVE / "arch.conf").read_bytes())

    def run(name, runner, epochs, opt=None, lam=None, resume=None, crosscheck=0):
        d = work / name
        (d / "ck").mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
                   MXF_TIMING_OUT=str(d / "timing.json"))
        if lam is not None:
            env.update(MXF_GAUGE_OUT=str(d / "gauge"), MXF_GAUGE_LAMBDA=str(lam),
                       MXF_GAUGE_CROSSCHECK=str(crosscheck))
        cmd = [str(PY_MX), str(R / "scripts" / runner), "train", f"@{arch}",
               "--param", str(d / "param.pth"), "--log-dir", str(d / "ck"),
               "--epochs", str(epochs), "--gpu", "-1", "--seed", "20261006",
               "--loss-func", "hinge_mix", "--score-loss-weight", "1.0",
               "--disable-progress-bar", str(lst)]
        if opt:
            cmd += ["--optimizer", opt]
        if resume:
            cmd += ["--resume", str(resume)]
        with (d / "stdout.log").open("a") as log:
            code = subprocess.run(cmd, cwd=DATA, env=env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        return d, code

    T, G = "mxfold2_timed_train.py", "mxfold2_gauge_train.py"
    a1, ca1 = run("a_plain_adamw", T, 1)
    a2, ca2 = run("a_gauge_adamw_lam0", G, 1, lam=0.0, crosscheck=c["crosscheck_steps"])
    b1, cb1 = run("b_plain_sgd", T, 1, opt="SGD")
    b2, cb2 = run("b_gauge_sgd_lam0", G, 1, opt="SGD", lam=0.0)
    c1, cc1 = run("c_gauge_2epochs", G, 2, lam=0.0)
    c2, cc2 = run("c_gauge_1then_resume", G, 1, lam=0.0)
    _, cc3 = run("c_gauge_1then_resume", G, 2, lam=0.0, resume=c2 / "ck" / "epoch-1")
    e1, ce1 = run("e_gauge_adamw_lam1", G, 1, lam=1.0)
    codes = {"a1": ca1, "a2": ca2, "b1": cb1, "b2": cb2, "c1": cc1, "c2": cc2, "c3": cc3,
             "e1": ce1}
    res = {"exit_codes": codes}
    if any(codes.values()):
        res["pass"] = False
        return res
    res["a_lam0_vs_plain"] = compare_checkpoints(a1 / "ck" / "epoch-1", a2 / "ck" / "epoch-1")
    res["b_sgd_lam0_vs_plain"] = compare_checkpoints(b1 / "ck" / "epoch-1", b2 / "ck" / "epoch-1")
    res["c_resume_vs_continuous"] = compare_checkpoints(c1 / "ck" / "epoch-2",
                                                        c2 / "ck" / "epoch-2")
    res["e_lam1_vs_lam0"] = compare_checkpoints(a2 / "ck" / "epoch-1", e1 / "ck" / "epoch-1")
    pat = re.compile(r"Train Epoch: (\d+)\tLoss: ([-+0-9.eE]+|nan|inf)")
    parsed = {d.name: pat.findall((d / "stdout.log").read_text()) for d in (a1, a2, b1, c1, c2)}
    res["d_parsed_epochs"] = {k: [(int(e), float(v)) for e, v in x] for k, x in parsed.items()}
    ga = json.loads((a2 / "gauge" / "gauge-epoch-1.json").read_text())
    ge = json.loads((e1 / "gauge" / "gauge-epoch-1.json").read_text())
    gc = json.loads((c2 / "gauge" / "gauge-epoch-2.json").read_text())
    res["f_crosscheck_rel_max"] = ga["crosscheck_rel_max"]
    res["e_mean_lamR"] = ge["mean_lamR"]
    res["c_resume_rng_restored"] = gc["rng_restored"]
    res["pass"] = bool(res["a_lam0_vs_plain"]["bitwise_identical"]
                   and res["b_sgd_lam0_vs_plain"]["bitwise_identical"]
                   and res["c_resume_vs_continuous"]["bitwise_identical"]
                   and gc["rng_restored"] is True
                   and not res["e_lam1_vs_lam0"]["bitwise_identical"]
                   and (ge["mean_lamR"] or 0) > 0
                   and all(len(v) >= 1 for v in res["d_parsed_epochs"].values())
                   and ga["crosscheck_rel_max"] is not None
                   and ga["crosscheck_rel_max"] <= c["crosscheck_tol"])
    return res


# ----------------------------------------------------------------------------------
def run(cfg, out_dir, write, prov) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(int(os.environ.get("MXF_THREADS", "4")))
    out = {"run_id": out_dir.name, "status": "started", "provenance": prov,
           "thresholds": cfg, "torch": torch.__version__}
    path = out_dir / "metrics.json"

    def save():
        write(path, out)
    save()
    model = shipped_model()
    t = time.time()
    out["P0_5_runner"] = p05_runner(cfg, out_dir)
    save()
    out["P0_1_masks"] = p01_masks()
    save()
    out["P0_2_v_symmetry"] = p02_v_symmetry(cfg, model)
    save()
    out["P0_3_head_noop"] = p03_head_noop(cfg, model)
    save()

    def progress(pr):
        out["P0_4_progress"] = pr
        save()
    out["P0_4_defining"] = p04_defining(cfg, model, progress)
    out["all_pass"] = bool(all(out[k]["pass"] is True for k in ("P0_1_masks", "P0_2_v_symmetry",
                                                    "P0_3_head_noop", "P0_4_defining",
                                                    "P0_5_runner")))
    out["status"] = "complete"
    out["elapsed_seconds"] = round(time.time() - t, 1)
    save()
    print(json.dumps({k: out[k]["pass"] for k in ("P0_1_masks", "P0_2_v_symmetry",
                                                   "P0_3_head_noop", "P0_4_defining",
                                                   "P0_5_runner")}, indent=1))
    return 0 if out["all_pass"] else 1
