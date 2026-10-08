#!/usr/bin/env python3
"""Decompose MXfold2 seed-to-seed (or run-to-run) score variance on TS0 against the exact gauge.

Implements the evaluation pre-registered in runs/2026-10-07_mxfold2_seed_study/metrics.json
(protocol, DECOMPOSITION), with the two corrections recorded there and here before any
decomposition was computed:

  (c1) projection_correction (2026-10-08): h is not orthogonal to g_p for a pair that can
       stack on one side only; P_G uses h' = h - sum_p (h.g_p / g_p.g_p) g_p.
  (c2) LONG HELICES (2026-10-08, found while implementing): MXfold2 scores a helix of
       m <= 30 pairs with helix_length[m], and a longer one as helix_length[30] plus
       extra stacks with no length term (zuker.cpp, C_HELIX_E). h is therefore null on
       a sequence only if NO structure of it contains a helix longer than 30 pairs. For
       a sequence whose allowed pairs admit a run of 31 stacked pairs, h is dropped from
       G(x) and the projection uses the g_p alone. The number of such sequences is reported.

  (c3) EMPTY INTERVALS (2026-10-08, found while implementing): positional.cpp reads the
       unpaired channel at (a, a-1) for a bulge's empty side and for a multiloop whose
       first branch is adjacent to its closing pair. Those strictly-lower entries are
       used positional outputs, which the pre-registered list (U(a,b), a <= b) omitted;
       theta(x) now includes U(a, a-1) for a = 2..N. The counting rule is corrected to
       match (scripts/mxfold2_counting.py).

FINDING RECORDED ALONGSIDE (not a correction): in MXfold2 type C every strictly-lower
entry of the positional score matrices is a sequence- and position-independent constant
per channel (fold/layers.py, PairedLayer.forward takes x_l from the already-triu'd x_u,
discarding the lower convolution branch). So S_out, M_out and U(a, a-1) are one learned
constant each, and the network can move the outside half of a pair gauge g_p only for
all pairs at once. The decomposition is computed on theta as defined; this constancy
shapes which parts of G seed variation can reach.

Runs in .venv-mxfold2 on the CPU (the GPUs are carrying training; an extra process there
could run a training process out of memory). Models are loaded from MXfold2 checkpoints.

PER SEQUENCE x (length N, canonical-pair TS0 entry), PER MODEL k:
  theta_k(x) = [S_in(i,j) where (i,j) and (i+1,j-1) are allowed pairs,
                S_out(j,i) where (i,j) and (i-1,j+1) are allowed pairs,
                M_in(i,j), M_out(j,i) for every allowed pair,
                U(a,b) for 1 <= a <= b <= N,
                helix_length[1..30]]
  allowed pair: canonical (AU UA GC CG GU UG) with j - i > 3, as MXfold2's make_constraint.
  G(x) = span{g_p} (+ h unless c2 applies); V_G, V_tot, f_G as pre-registered.

VALIDITY CHECKS (pre-registered; no statistic is reported unless all pass)
  V1  the counting rule (scripts/mxfold2_counting.MX, type C) applied to the extracted
      tables reproduces MXfold2's positional score of the reference structure: MixC's
      constrained score minus the Turner-only constrained score of the same model, to
      1e-3 * max(1, |score|). Only references MXfold2 can realise are used (the
      constrained fold must return the reference itself); the rest are counted.
  V2  a random g in G(x) added to the tables changes MXfold2's score of the reference and
      of the predicted structure by less than that tolerance.
  V3  a random direction orthogonal to G(x) changes at least one of them by more.

Usage:
  .venv-mxfold2/bin/python scripts/mxfold2_decompose.py --run-id ID --epoch E \\
      --group LABEL=run1,run2[,...] [--group ...]
Each run name is a directory under runs/2026-10-07_mxfold2_seed_study/seeds/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from mxfold2.train import Train  # noqa: E402

from scripts.mxfold2_counting import MX  # noqa: E402
from scripts.runmeta import git_provenance  # noqa: E402

STUDY = R / "runs" / "2026-10-07_mxfold2_seed_study"
DATA = R / ".cache" / "mxfold2-data"
REL_TOL = 1e-3
MAX_HELIX = 30
CANON = {"AU", "UA", "GC", "CG", "GU", "UG"}
SEED = 20261008

CORRECTION_LONG_HELIX = (
    "Correction recorded 2026-10-08, BEFORE any decomposition was computed. MXfold2 scores "
    "a helix of m <= 30 pairs with helix_length[m] and a longer helix as helix_length[30] "
    "plus extra stacks with no length term (zuker.cpp, C_HELIX_E). The direction h (+1 on "
    "every S, -2(m-1) on helix_length[m]) moves a structure containing a helix of L > 30 "
    "pairs by 2(L - 30), so h is in the exact null space of x only if no structure of x can "
    "contain such a helix. Where the allowed pairs of x admit a run of 31 stacked pairs, h "
    "is dropped from G(x). The step-2 sequences (<= 31 nt) could not show this.")


def build_model(arch):
    parser = ArgumentParser(fromfile_prefix_chars="@")
    sub = parser.add_subparsers()
    Train.add_args(sub)
    args = parser.parse_args(["train", f"@{arch}", "unused.lst"])
    model, _ = Train().build_model(args)
    return model


def load(run, epoch, arch):
    ck = STUDY / "seeds" / run / "checkpoints" / f"epoch-{epoch}"
    m = build_model(arch)
    state = torch.load(ck, map_location="cpu", weights_only=True)
    m.load_state_dict(state["model_state_dict"])
    m.eval()
    return m


def read_bpseq(path):
    rows = [r.split() for r in path.read_text().split("\n") if r.strip() and not r.startswith("#")]
    seq = "".join(r[1] for r in rows).upper().replace("T", "U")
    partner = [0] + [int(r[2]) for r in rows]
    return seq, partner


def dot_bracket(partner):
    n = len(partner) - 1
    db = ["."] * n
    for i in range(1, n + 1):
        j = partner[i]
        if j > i:
            db[i - 1], db[j - 1] = "(", ")"
    return "".join(db)


def nested(partner):
    n = len(partner) - 1
    for i in range(1, n + 1):
        j = partner[i]
        if j > i:
            for k in range(i + 1, j):
                if partner[k] > j or (0 < partner[k] < i):
                    return False
    return True


class Geometry:
    """Index structures of theta(x) and G(x) for one sequence."""

    def __init__(self, seq):
        n = len(seq)
        self.n = n
        al = np.zeros((n + 2, n + 2), bool)
        for i in range(1, n + 1):
            for j in range(i + 4, n + 1):
                al[i, j] = seq[i - 1] + seq[j - 1] in CANON
        self.I, self.J = np.nonzero(al)
        self.e_in = al[self.I + 1, self.J - 1]
        self.e_out = al[self.I - 1, self.J + 1]
        self.n_p = 2 + self.e_in.astype(int) + self.e_out.astype(int)
        self.hg = self.e_in.astype(int) - self.e_out.astype(int)
        self.ua, self.ub = np.triu_indices(n + 1, 0)
        keep = self.ua >= 1
        self.ua, self.ub = self.ua[keep], self.ub[keep]
        self.la = np.arange(2, n + 1)  # empty-interval entries U(a, a-1), correction c3
        self.m = np.arange(1, MAX_HELIX + 1)
        self.hw = -2.0 * (self.m - 1)
        # longest run of stacked allowed pairs, along anti-diagonals
        longest = 0
        for s in range(2, 2 * n + 1):
            i = np.arange(max(1, s - n), min(n, s - 1) + 1)
            v = al[i, s - i]
            if v.any():
                d = np.diff(np.concatenate([[0], v.astype(int), [0]]))
                longest = max(longest, int((np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]).max()))
        self.longest_helix = longest
        self.h_in_G = longest <= MAX_HELIX
        self.hh = int(self.e_in.sum() + self.e_out.sum()) + float((self.hw ** 2).sum())

    def theta(self, p):
        S = p["score_helix_stacking"].numpy().astype(np.float64)
        M = p["score_mismatch_hairpin"].numpy().astype(np.float64)
        U = p["score_base_hairpin"].numpy().astype(np.float64)
        hl = p["score_helix_length"].numpy().astype(np.float64)
        assert S.shape == (self.n + 1, self.n + 1), S.shape
        return {"Sin": np.where(self.e_in, S[self.I, self.J], 0.0),
                "Sout": np.where(self.e_out, S[self.J, self.I], 0.0),
                "Min": M[self.I, self.J], "Mout": M[self.J, self.I],
                "U": U[self.ua, self.ub], "Ulow": U[self.la, self.la - 1],
                "HL": hl[1:MAX_HELIX + 1]}

    def norm2(self, d):
        return float(sum((v ** 2).sum() for v in d.values()))

    def proj_norm2(self, d):
        dg = d["Sin"] + d["Min"] - d["Sout"] - d["Mout"]
        val = float((dg ** 2 / self.n_p).sum())
        if self.h_in_G:
            dh = d["Sin"].sum() + d["Sout"].sum() + float((self.hw * d["HL"]).sum())
            dhp = dh - float((self.hg * dg / self.n_p).sum())
            hhp = self.hh - float((self.hg ** 2 / self.n_p).sum())
            val += dhp ** 2 / hhp
        return val

    def random_in_G(self, rng):
        c = rng.standard_normal(len(self.I))
        g = {"Sin": c * self.e_in, "Min": c.copy(), "Sout": -c * self.e_out, "Mout": -c.copy(),
             "U": np.zeros(len(self.ua)), "Ulow": np.zeros(len(self.la)),
             "HL": np.zeros(MAX_HELIX)}
        if self.h_in_G:
            ch = rng.standard_normal()
            g["Sin"] = g["Sin"] + ch * self.e_in
            g["Sout"] = g["Sout"] + ch * self.e_out
            g["HL"] = ch * self.hw
        return g

    def project(self, d):
        """P_G d, as a theta-shaped dict."""
        dg = d["Sin"] + d["Min"] - d["Sout"] - d["Mout"]
        c = dg / self.n_p
        out = {"Sin": c * self.e_in, "Min": c.copy(), "Sout": -c * self.e_out, "Mout": -c.copy(),
               "U": np.zeros(len(self.ua)), "Ulow": np.zeros(len(self.la)),
               "HL": np.zeros(MAX_HELIX)}
        if self.h_in_G:
            dh = d["Sin"].sum() + d["Sout"].sum() + float((self.hw * d["HL"]).sum())
            dhp = dh - float((self.hg * c).sum())
            hhp = self.hh - float((self.hg ** 2 / self.n_p).sum())
            a = dhp / hhp
            # h' = h - sum_p (h.g_p / n_p) g_p
            ap = -a * self.hg / self.n_p
            out["Sin"] += (a + ap) * self.e_in
            out["Sout"] += (a - ap) * self.e_out
            out["Min"] += ap
            out["Mout"] -= ap
            out["HL"] = a * self.hw
        return out

    def apply(self, p, d):
        """Positional parameter dict with theta-direction d added (S, M, U, helix_length)."""
        q = dict(p)
        S = p["score_helix_stacking"].clone()
        M = p["score_mismatch_hairpin"].clone()
        U = p["score_base_hairpin"].clone()
        hl = p["score_helix_length"].clone()
        S[self.I, self.J] += torch.tensor(d["Sin"] * self.e_in, dtype=S.dtype)
        S[self.J, self.I] += torch.tensor(d["Sout"] * self.e_out, dtype=S.dtype)
        M[self.I, self.J] += torch.tensor(d["Min"], dtype=M.dtype)
        M[self.J, self.I] += torch.tensor(d["Mout"], dtype=M.dtype)
        U[self.ua, self.ub] += torch.tensor(d["U"], dtype=U.dtype)
        U[self.la, self.la - 1] += torch.tensor(d["Ulow"], dtype=U.dtype)
        hl[1:MAX_HELIX + 1] += torch.tensor(d["HL"], dtype=hl.dtype)
        q["score_helix_stacking"] = S
        for k in ("score_mismatch_external", "score_mismatch_hairpin",
                  "score_mismatch_internal", "score_mismatch_multi"):
            q[k] = M
        for k in ("score_base_hairpin", "score_base_internal", "score_base_multi",
                  "score_base_external"):
            q[k] = U
        q["score_helix_length"] = hl
        return q


def positional_value(key, p):
    t = key[0]
    if t == "S":
        return float(p["score_helix_stacking"][key[1], key[2]])
    if t == "M":
        return float(p["score_mismatch_hairpin"][key[1], key[2]])
    if t == "U":
        return float(p["score_base_hairpin"][key[1], key[2]])
    if t in ("helix_length", "hairpin_length", "bulge_length", "internal_length",
             "internal_symmetry", "internal_asymmetry"):
        return float(p[f"score_{t}"][key[1]])
    if t == "internal_explicit":
        return float(p["score_internal_explicit"][key[1], key[2]])
    raise KeyError(key)


def constrained(model, seq, param, partner):
    with torch.no_grad():
        sc, pred, _ = model([seq], param=param, constraint=[torch.tensor(partner)],
                            max_internal_length=None)
    return float(sc[0]), pred[0]


def predict(model, seq, param):
    with torch.no_grad():
        sc, pred, bp = model([seq], param=param)
    return float(sc[0]), pred[0], list(bp[0])


def tol(x):
    return REL_TOL * max(1.0, abs(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--group", action="append", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N sequences (smoke test)")
    ap.add_argument("--out-dir", default="", help="override output directory (smoke test)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out_dir = pathlib.Path(a.out_dir) if a.out_dir else R / "runs" / a.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = {g.split("=", 1)[0]: g.split("=", 1)[1].split(",") for g in a.group}
    study = json.loads((STUDY / "metrics.json").read_text())
    out = {"run_id": a.run_id, "status": "started", "epoch": a.epoch, "groups": groups,
           "protocol_source": "runs/2026-10-07_mxfold2_seed_study/metrics.json: protocol",
           "corrections": {"c1_projection": study.get("projection_correction"),
                           "c2_long_helix": CORRECTION_LONG_HELIX,
                           "c3_empty_intervals": "theta includes U(a, a-1), a = 2..N"},
           "limit": a.limit,
           "tolerance": f"{REL_TOL} * max(1, |score|)", **git_provenance(R),
           "torch": torch.__version__, "device": "cpu", "threads": a.threads}

    def save():
        (out_dir / "metrics.json").write_text(json.dumps(out, indent=1, sort_keys=True,
                                                         default=str))
    save()

    # fixed evaluation set, checked against the pre-registered list
    pre = study["evaluation_set"]["entries"]
    ev = []
    for e in pre:
        seq, partner = read_bpseq(DATA / e["file"])
        assert hashlib.sha256(seq.encode()).hexdigest() == e["sha256"], e["file"]
        ev.append((e["file"], seq, partner))
    out["evaluation_set_verified"] = len(ev)
    if a.limit:
        ev = ev[:a.limit]

    arch = STUDY / "arch.conf"
    runs = sorted({r for v in groups.values() for r in v})
    models = {r: load(r, a.epoch, arch) for r in runs}
    rng = np.random.default_rng(SEED)
    mx = MX("C")

    val = {r: {"V1_n": 0, "V1_fail": 0, "V1_max_rel": 0.0, "V1_excluded_unrealisable": 0,
               "V2_n": 0, "V2_fail": 0, "V2_max_rel": 0.0, "V3_n": 0, "V3_detected": 0,
               "V3_min_change": None} for r in runs}
    per_seq = {g: {"V_G": [], "V_tot": []} for g in groups}
    preds = {r: [] for r in runs}
    n_long = 0
    t0 = time.time()
    for n_x, (_f, seq, partner) in enumerate(ev):
        geo = Geometry(seq)
        n_long += int(not geo.h_in_G)
        ref_db = dot_bracket(partner)
        realisable_ref = nested(partner)
        thetas = {}
        for r, m in models.items():
            with torch.no_grad():
                param = m.make_param([seq])
            pp = {k: v.detach().cpu() for k, v in param[0]["positional"].items()}
            thetas[r] = geo.theta(pp)
            s_pred, db_pred, bp_pred = predict(m, seq, param)
            preds[r].append((db_pred, s_pred))
            v = val[r]
            # V1
            if realisable_ref:
                s_mix, db_mix = constrained(m, seq, param, partner)
                if db_mix == ref_db:
                    with torch.no_grad():
                        s_tur, _, _ = m.turner([seq], constraint=[torch.tensor(partner)],
                                               max_internal_length=None)
                    pos = s_mix - float(s_tur[0])
                    rec = sum(c * positional_value(k, pp) for k, c in
                              mx.counts(seq, ref_db).items())
                    v["V1_n"] += 1
                    rel = abs(rec - pos) / max(1.0, abs(pos))
                    v["V1_max_rel"] = max(v["V1_max_rel"], rel)
                    v["V1_fail"] += int(abs(rec - pos) > tol(pos))
                else:
                    v["V1_excluded_unrealisable"] += 1
            else:
                v["V1_excluded_unrealisable"] += 1
            # V2 / V3 on the reference (if realisable) and the predicted structure
            targets = [bp_pred] + ([partner] if realisable_ref else [])
            base = [constrained(m, seq, param, t)[0] for t in targets]
            g = geo.random_in_G(rng)
            param_g = [{"turner": param[0]["turner"], "positional": geo.apply(pp, g)}]
            new = [constrained(m, seq, param_g, t)[0] for t in targets]
            d2 = max(abs(x - y) / max(1.0, abs(x)) for x, y in zip(base, new, strict=True))
            v["V2_n"] += 1
            v["V2_max_rel"] = max(v["V2_max_rel"], d2)
            v["V2_fail"] += int(d2 > REL_TOL)
            z = {k: rng.standard_normal(len(x)) for k, x in thetas[r].items()}
            z["Sin"] *= geo.e_in
            z["Sout"] *= geo.e_out
            pz = geo.project(z)
            z = {k: z[k] - pz[k] for k in z}
            param_z = [{"turner": param[0]["turner"], "positional": geo.apply(pp, z)}]
            new = [constrained(m, seq, param_z, t)[0] for t in targets]
            d3 = max(abs(x - y) / max(1.0, abs(x)) for x, y in zip(base, new, strict=True))
            v["V3_n"] += 1
            v["V3_detected"] += int(d3 > REL_TOL)
            v["V3_min_change"] = d3 if v["V3_min_change"] is None else min(v["V3_min_change"], d3)
        for gname, members in groups.items():
            th = [thetas[r] for r in members]
            mean = {k: np.mean([t[k] for t in th], axis=0) for k in th[0]}
            dk = [{k: t[k] - mean[k] for k in t} for t in th]
            K = len(th)
            per_seq[gname]["V_G"].append(sum(geo.proj_norm2(d) for d in dk) / (K - 1))
            per_seq[gname]["V_tot"].append(sum(geo.norm2(d) for d in dk) / (K - 1))
        if n_x % 50 == 0:
            out["progress"] = {"sequences_done": n_x + 1, "of": len(ev),
                               "seconds": round(time.time() - t0, 1)}
            save()

    out["n_sequences_h_dropped_long_helix"] = n_long
    out["validity"] = val
    ok = all(v["V1_fail"] == 0 and v["V2_fail"] == 0 and v["V3_detected"] == v["V3_n"]
             and v["V1_n"] > 0 for v in val.values())
    out["validity_all_pass"] = ok
    if not ok:
        out["status"] = "STOPPED: a validity check failed; no statistic reported"
        save()
        return 1
    res = {}
    for gname, members in groups.items():
        vg = np.array(per_seq[gname]["V_G"])
        vt = np.array(per_seq[gname]["V_tot"])
        fg = np.where(vt > 0, vg / np.where(vt > 0, vt, 1.0), np.nan)
        pr = [preds[r] for r in members]
        same_struct = sum(all(p[i][0] == pr[0][i][0] for p in pr) for i in range(len(ev)))
        max_sc = max(abs(p[i][1] - pr[0][i][1]) for p in pr for i in range(len(ev)))
        res[gname] = {"members": members, "n_sequences": len(ev),
                      "V_G_sum": float(vg.sum()), "V_tot_sum": float(vt.sum()),
                      "f_G_pooled": float(vg.sum() / vt.sum()) if vt.sum() > 0 else None,
                      "V_G_median": float(np.median(vg)), "V_tot_median": float(np.median(vt)),
                      "f_G_median": float(np.nanmedian(fg)) if np.isfinite(fg).any() else None,
                      "n_sequences_with_V_tot_zero": int((vt == 0).sum()),
                      "predicted_structures_identical": int(same_struct),
                      "max_abs_predicted_score_diff": float(max_sc),
                      "per_sequence_V_G": [float(x) for x in vg],
                      "per_sequence_V_tot": [float(x) for x in vt]}
    out["groups_result"] = res
    out["status"] = "complete"
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    save()
    print(json.dumps({g: {k: v for k, v in r.items() if not k.startswith("per_sequence")}
                      for g, r in res.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
