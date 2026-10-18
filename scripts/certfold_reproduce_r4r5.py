#!/usr/bin/env python3
"""Independent reproduction of registration 2's R4 and R5 (reviewer additions, item 1).

Reads only the raw shard files and the MXfold2 predictions; imports nothing from the
repository. Written from the docstrings of certfold_reg2._r4_spec/_c_task,
certfold_exploratory.stratified_bootstrap/strata_groups, certfold_run.pair_records/stat_matched/
task_records and certfold_metrics.matched_coverage_comparison/_curve/paired_bootstrap.

Implementation differs from the committed one on purpose (no record dicts are copied):
  R4  per-unit value arrays, gathered in draw order and summed sequentially (np.cumsum), which is
      Python 3.11's sum() order, so the points should agree bit for bit.
  R5  a resample is a multiset of the original records (multiplicity = draws of their unit), so the
      tie-blocked ranking is computed once on the originals and each resample only reweights the
      blocks; no sort per resample. The block-expectation arithmetic is the same integer/float
      sequence as certfold_metrics._curve.

usage: python scripts/certfold_reproduce_r4r5.py OUTDIR [R4_TAG R5_TAG]
Written by a fresh-context agent (2026-10-18); its run on tags 60 and 61 is in
runs/2026-10-18_certfold_reviewer/reproduce_r4r5/ (117 of 117 quantities bit-identical).
"""
from __future__ import annotations

import glob
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path("/home/shadi/sirna-thermo")
RUN1 = REPO / "runs/2026-10-13_certified_folding/shards"
RUN2 = REPO / "runs/2026-10-15_certfold_registration2"
MX = REPO / "runs/2026-10-07_mxfold2_ts0_inference/pred_A_cpu.txt"
COMMITTED = RUN2 / "metrics.json"

SEED = 20261013
N_BOOT = 10000
P_HI = 0.99
LEVEL = 95.0
TABLES = ("Turner1999", "Andronescu2007", "Langdon2018")
TABLE_IDS = tuple(f"{w}:{X}" for w in ("whole", "visible") for X in TABLES)
R4_SPECS = (("R4_primary", "ret_primary"), ("R4_secondary", "ret_visible"),
            ("R4_tertiary", "ret_mxfold2"))
R5_BASELINES = (("p", "p"), ("nominal_gap", "gap"), ("length", "neg_len"))
MERGE = {1: 0}


def pairs_of(db: str) -> set[tuple[int, int]]:
    st, out = [], set()
    for k, c in enumerate(db, start=1):
        if c == "(":
            st.append(k)
        elif c == ")":
            out.add((st.pop(), k))
    if st:
        raise SystemExit(f"unbalanced structure {db}")
    return out


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def load() -> tuple[list[dict], dict]:
    files = sorted(glob.glob(str(RUN1 / "run_*.jsonl")))
    assert len(files) == 8, files
    seqs = []
    for f in files:
        with open(f) as fh:
            seqs += [json.loads(ln) for ln in fh if ln.strip()]
    ts0 = sorted((s for s in seqs if s["set"] == "TS0"), key=lambda s: s["index"])
    idx = [s["index"] for s in ts0]
    assert len(ts0) == 1288 and len(set(idx)) == 1288, (len(ts0), len(set(idx)))

    tables = {}
    tfiles = sorted(glob.glob(str(RUN2 / "shards/tables_[0-9]*.jsonl")))
    assert len(tfiles) == 12, tfiles
    for f in tfiles:
        for ln in open(f):
            if ln.strip():
                d = json.loads(ln)
                assert d["index"] not in tables, d["index"]
                tables[d["index"]] = d
    assert set(tables) == set(idx), "fold shards do not cover TS0 exactly"

    mx, dup = {}, []
    for ln in open(MX):
        if ln.strip():
            d = json.loads(ln)
            if d["name"] in mx and mx[d["name"]] != d["structure"]:
                dup.append(d["name"])
            mx[d["name"]] = d["structure"]          # last wins, as a dict built line by line

    info = {"n_ts0": len(ts0), "mx_lines": None, "mx_conflicting_duplicates": dup,
            "chars": {}}
    chars = {t: set() for t in (*TABLE_IDS, "mxfold2")}
    for s in ts0:
        t = tables[s["index"]]
        assert t["name"] == s["name"], (t["name"], s["name"])
        n = len(s["star"])
        assert n == s["n_nt"], s["index"]
        st = {tid: t["mfe"][tid] for tid in TABLE_IDS}
        st["mxfold2"] = mx[s["name"].removesuffix(".bpseq")]
        for tid, db in st.items():
            assert len(db) == n, (s["index"], tid)
            chars[tid] |= set(db)
        ps = {tid: pairs_of(db) for tid, db in st.items()}
        recs = []
        for q in s["pairs"]:
            pair = tuple(q["pair"])
            e = q["rho_env"]
            rho = None if e is None or e["rho"] is None else float(e["rho"])
            k = {tid: pair in ps[tid] for tid in ps}
            recs.append({
                "pair": pair, "p": float(q["p"]), "gap": float(q["nominal_gap_kcal"]),
                "cert": q["margins_dcal"]["envelope:1"] > 0, "rho": rho,
                "neg_len": -float(n),
                **{f"kept:{t_}": v for t_, v in k.items()},
                "ret_primary": sum(k[f"whole:{X}"] for X in TABLES) / len(TABLES),
                "ret_visible": sum(k[f"visible:{X}"] for X in TABLES) / len(TABLES),
                "ret_mxfold2": float(k["mxfold2"])})
        s["recs"] = recs
    info["chars"] = {t: "".join(sorted(c)) for t, c in chars.items()}
    return ts0, info


# ---------------------------------------------------------------------------------------------
# R4: stratified bootstrap, seed [SEED, tag]
# ---------------------------------------------------------------------------------------------
def r4(ts0: list[dict], tag: int, n_boot: int = N_BOOT) -> dict:
    strata = sorted({s["stratum"] for s in ts0})
    units = {st: sorted(s["index"] for s in ts0 if s["stratum"] == st) for st in strata}
    by = {s["index"]: s for s in ts0}
    labels: dict[int, list[int]] = {}
    for st in strata:
        labels.setdefault(MERGE.get(st, st), []).append(st)
    reps = {"+".join(map(str, sorted(m))): sorted(m)
            for m in sorted(labels.values(), key=lambda m: min(m))}

    # per stratum, per spec: flat value arrays (certified, uncertified) in sorted-unit order,
    # each unit's pairs in shard order, with offsets
    flat = {}
    for st in strata:
        for name, key in R4_SPECS:
            for grp in (True, False):
                vals, off, ln = [], [], []
                for u in units[st]:
                    v = [r[key] for r in by[u]["recs"] if r["p"] >= P_HI and r["cert"] == grp]
                    off.append(len(vals))
                    ln.append(len(v))
                    vals += v
                flat[(st, name, grp)] = (np.asarray(vals, dtype=np.float64),
                                         np.asarray(off, dtype=np.int64),
                                         np.asarray(ln, dtype=np.int64))

    def gather(st, name, grp, idx):
        v, off, ln = flat[(st, name, grp)]
        L = ln[idx]
        tot = int(L.sum())
        if tot == 0:
            return v[:0]
        starts = off[idx]
        cs = np.cumsum(L) - L
        g = np.repeat(starts - cs, L) + np.arange(tot)
        return v[g]

    def seqsum(a: np.ndarray) -> float:
        # Python 3.11 sum(): left to right from int 0
        return float(np.cumsum(a)[-1]) if len(a) else 0.0

    def label_stat(name, members, draw):
        c = np.concatenate([gather(m, name, True, draw[m]) for m in members])
        u = np.concatenate([gather(m, name, False, draw[m]) for m in members])
        if len(c) == 0 or len(u) == 0:
            return None, len(c) + len(u), len(c), len(u), None, None
        mc, mu = seqsum(c) / len(c), seqsum(u) / len(u)
        return mc - mu, len(c) + len(u), len(c), len(u), mc, mu

    def pooled(vals: dict, w: dict) -> float:
        tw = sum(w[s] for s in vals)
        return sum(w[s] * vals[s] for s in vals) / tw

    full_draw = {st: np.arange(len(units[st])) for st in strata}
    out = {}
    for name, _key in R4_SPECS:
        per = {}
        for lab, members in reps.items():
            v, w, nc, nu, mc, mu = label_stat(name, members, full_draw)
            per[lab] = {"members": members, "n_pairs": w,
                        "n_sequences": sum(len(units[m]) for m in members), "point": v,
                        "n_certified": nc, "n_uncertified": nu,
                        "mean_certified": mc, "mean_uncertified": mu}
        defined = [lab for lab in per if per[lab]["point"] is not None]
        out[name] = {"by_stratum": per, "strata_pooled": defined,
                     "pooled_point": pooled({s: per[s]["point"] for s in defined},
                                            {s: per[s]["n_pairs"] for s in defined})}

    rng = np.random.default_rng(np.random.SeedSequence([SEED, tag]))
    bvals = {n: {lab: [] for lab in reps} for n, _ in R4_SPECS}
    bund = {n: {lab: 0 for lab in reps} for n, _ in R4_SPECS}
    bpool = {n: [] for n, _ in R4_SPECS}
    bpund = {n: 0 for n, _ in R4_SPECS}
    for _b in range(n_boot):
        draw = {}
        for st in strata:
            draw[st] = rng.integers(0, len(units[st]), size=len(units[st]))
        for name, _key in R4_SPECS:
            got, w = {}, {}
            for lab, members in reps.items():
                v, wt, *_ = label_stat(name, members, draw)
                if v is None:
                    bund[name][lab] += 1
                    continue
                bvals[name][lab].append(v)
                got[lab], w[lab] = v, wt
            want = out[name]["strata_pooled"]
            if want and all(lab in got for lab in want):
                bpool[name].append(pooled({lab: got[lab] for lab in want}, w))
            else:
                bpund[name] += 1
    half = (100.0 - LEVEL) / 2

    def interval(xs, nu):
        if nu or not xs:
            return None
        lo, hi = np.percentile(xs, [half, 100.0 - half])
        return [float(lo), float(hi)]

    for name, _key in R4_SPECS:
        for lab in reps:
            out[name]["by_stratum"][lab]["interval"] = interval(bvals[name][lab], bund[name][lab])
            out[name]["by_stratum"][lab]["resamples_undefined"] = bund[name][lab]
        out[name]["pooled_interval"] = interval(bpool[name], bpund[name])
        out[name]["pooled_resamples_undefined"] = bpund[name]
    out["seed"] = [SEED, tag]
    return out


# ---------------------------------------------------------------------------------------------
# R5: matched coverage with kept(p, X) as "correct", paired bootstrap over TS0 sequences
# ---------------------------------------------------------------------------------------------
class Ranking:
    """Score descending, then tiebreak descending; blocks tied on both keys, over the originals."""

    def __init__(self, sc: np.ndarray, tb: np.ndarray):
        o = np.lexsort((-tb, -sc))
        s, t = sc[o], tb[o]
        new = np.ones(len(s), dtype=bool)
        new[1:] = (s[1:] != s[:-1]) | (t[1:] != t[:-1])
        self.order = o
        self.starts = np.flatnonzero(new)

    def ppv_at(self, w: np.ndarray, ok: np.ndarray, k: int) -> float:
        """PPV of the top k of the multiset with multiplicities w (int64), ties at expectation."""
        ws = w[self.order]
        hs = ws * ok[self.order]
        size = np.add.reduceat(ws, self.starts)
        hits = np.add.reduceat(hs, self.starts)
        csize = np.cumsum(size)
        chits = np.cumsum(hits)
        b = int(np.searchsorted(csize, k, side="left"))
        start = int(csize[b] - size[b])
        before = int(chits[b] - hits[b])
        # certfold_metrics._curve: before[blk] + (k - starts[blk]) * hits[blk] / sizes[blk], / k
        got = (np.int64(before)
               + (np.int64(k) - np.int64(start)) * np.int64(hits[b]) / np.int64(size[b]))
        return float(got / np.int64(k))


def matched_k(w: np.ndarray, hi: np.ndarray) -> int:
    n = int(w.sum())
    m = int(w[hi].sum())
    if m == 0:
        raise ValueError("the coverage set is empty")
    c = m / n
    cov = np.arange(1, n + 1) / n
    return int(np.searchsorted(cov, c, side="left")) + 1


def r5_table(args) -> dict:
    ts0, X, tag, n_boot = args
    units = sorted(s["index"] for s in ts0)
    upos = {u: i for i, u in enumerate(units)}
    pool = [(upos[s["index"]], r) for s in ts0 for r in s["recs"] if r["rho"] is not None]
    unit = np.array([u for u, _ in pool], dtype=np.int64)
    rho = np.array([r["rho"] for _, r in pool])
    gap = np.array([r["gap"] for _, r in pool])                # the "tiebreak" key
    keys = {"p": np.array([r["p"] for _, r in pool]), "gap": gap,
            "neg_len": np.array([r["neg_len"] for _, r in pool])}
    hi = np.array([r["p"] >= P_HI for _, r in pool])
    ok = np.array([r[f"kept:whole:{X}"] for _, r in pool], dtype=np.int64)
    rk_s = Ranking(rho, gap)
    rk_b = {n: Ranking(keys[k], gap) for n, k in R5_BASELINES}

    def stat(w):
        k = matched_k(w, hi)
        ps = rk_s.ppv_at(w, ok, k)
        res = {}
        for n, _k in R5_BASELINES:
            pb = rk_b[n].ppv_at(w, ok, k)
            res[n] = (ps - pb, ps, pb, k)
        return res

    w1 = np.ones(len(pool), dtype=np.int64)
    full = stat(w1)
    out = {"n": len(pool), "n_in_coverage_set": int(hi.sum())}
    for n, k in R5_BASELINES:
        d, ps, pb, kk = full[n]
        out[n] = {"difference": d, "ppv_score": ps, "ppv_baseline": pb, "k": kk,
                  "baseline_key": k}
    rng = np.random.default_rng(np.random.SeedSequence([SEED, tag]))
    draws = rng.integers(0, len(units), size=(n_boot, len(units)))
    vals = {n: np.empty(n_boot) for n, _ in R5_BASELINES}
    for b in range(n_boot):
        mult = np.bincount(draws[b], minlength=len(units)).astype(np.int64)
        got = stat(mult[unit])
        for n, _ in R5_BASELINES:
            vals[n][b] = got[n][0]
    half = (100.0 - LEVEL) / 2
    for n, _ in R5_BASELINES:
        lo, hi_ = np.percentile(vals[n], [half, 100.0 - half])
        out[n]["interval"] = [float(lo), float(hi_)]
    out["seed"] = [SEED, tag]
    return X, out


def _task(a):
    kind = a[0]
    if kind == "R4":
        return "R4", r4(a[1], a[2])
    return r5_table(a[1:])


# ---------------------------------------------------------------------------------------------
# comparison against the committed record
# ---------------------------------------------------------------------------------------------
def compare(mine: dict, committed: dict) -> tuple[list[tuple], float]:
    rows = []

    def add(label, a, b):
        if a is None or b is None:
            rows.append((label, b, a, None if a is b else float("inf")))
        else:
            rows.append((label, b, a, abs(a - b)))

    c4 = committed["R4"]
    for name, _ in R4_SPECS:
        m, c = mine["R4"][name], c4[name]
        add(f"{name} pooled point", m["pooled_point"], c["pooled_point"])
        for j, side in enumerate(("lo", "hi")):
            add(f"{name} pooled {side}", m["pooled_interval"][j], c["pooled_interval"][j])
        assert set(m["by_stratum"]) == set(c["by_stratum"]), (m["by_stratum"].keys(),
                                                                c["by_stratum"].keys())
        for lab in c["by_stratum"]:
            mb, cb = m["by_stratum"][lab], c["by_stratum"][lab]
            add(f"{name} [{lab}] point", mb["point"], cb["point"])
            for j, side in enumerate(("lo", "hi")):
                add(f"{name} [{lab}] {side}", None if mb["interval"] is None else
                    mb["interval"][j], None if cb["interval"] is None else cb["interval"][j])
            for key in ("n_pairs", "n_sequences", "n_certified", "n_uncertified"):
                if mb[key] != cb[key]:
                    rows.append((f"{name} [{lab}] {key}", cb[key], mb[key], float("inf")))
    c5 = committed["R5"]
    for X in TABLES:
        for n, _ in R5_BASELINES:
            m, c = mine["R5"][X][n], c5[X][n]
            add(f"R5 {X} vs {n} difference", m["difference"], c["difference"])
            for j, side in enumerate(("lo", "hi")):
                add(f"R5 {X} vs {n} {side}", m["interval"][j], c["interval"][j])
            if (m["k"], mine["R5"][X]["n"]) != (c["k"], c["n"]):
                rows.append((f"R5 {X} vs {n} (k, n)", (c["k"], c["n"]),
                             (m["k"], mine["R5"][X]["n"]), float("inf")))
    mx = max(r[3] for r in rows if r[3] is not None)
    return rows, mx


def main() -> int:
    outdir = pathlib.Path(sys.argv[1])
    tag4, tag5 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (60, 61)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ts0, info = load()
    tasks = [("R4", ts0, tag4)] + [("R5", ts0, X, tag5, N_BOOT) for X in TABLES]
    with mp.get_context("fork").Pool(len(tasks)) as pool:
        got = dict(pool.map(_task, tasks, chunksize=1))
    mine = {"R4": got["R4"], "R5": {X: got[X] for X in TABLES}, "load_info": info,
            "seconds": round(time.time() - t0, 1), "tags": [tag4, tag5]}
    (outdir / f"repro_tags_{tag4}_{tag5}.json").write_text(json.dumps(mine, indent=1))
    if (tag4, tag5) != (60, 61):
        print(json.dumps({k: mine[k] for k in ("seconds", "tags")}))
        return 0
    committed = json.loads(COMMITTED.read_text())["analysis_c"]
    rows, mx = compare(mine, committed)
    lines = [f"{'quantity':48s} {'committed':>24s} {'reproduced':>24s} {'|diff|':>10s}"]
    for lab, c, m, d in rows:
        lines.append(f"{lab:48s} {c!r:>24} {m!r:>24} {d!r:>10}")
    lines.append(f"max |diff| = {mx!r}  ({len(rows)} quantities)")
    lines.append("verdict: " + ("EXACT (<= 1e-12)" if mx <= 1e-12 else "NOT EXACT"))
    n_bitexact = sum(1 for r in rows if r[3] == 0.0)
    lines.append(f"bit-identical: {n_bitexact} of {len(rows)}")
    (outdir / "comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
