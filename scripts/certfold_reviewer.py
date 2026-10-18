#!/usr/bin/env python3
"""CERTIFIED FOLDING: the reviewer's additions, all POST HOC, as specified before any of them ran
in docs/certfold_reviewer_additions.md (the section numbers below are that file's).

  2   stability made interpretable: absolute retention, pairs lost, relative reduction in loss,
      by stratum; R4's statistic leaving one table out (seed tag 80)
  3a  scale: R4's statistic with certification at lambda = 0.5 and 2 (tag 81)
  3b  how much of each published table's difference the set covers, by parameter family
  3c  the held-out basis: B_TR0 without the directions TS0's references would pin; envelopes on it
      (i) at the registered r_max and delta, (ii) recomputed; G2 first; TS0 labels; R4 (tag 85)
  4a  the measurement-efficiency curve by bisection along nested orderings (tag 84)
  4b  the ranking's stability: the pilot bootstrapped (tag 82); an independent second pilot (83)
  5   conservativeness against exact enumeration; what an uncertain label means
  6   the certified radius against the nominal gap

PHASES, each refusing unless the specification is committed unmodified and its predecessor's
record is committed:
  descriptive    2, 3a, 3b, 4b (bootstrap), 5, 6
  pilot2-rank    4b: the second pilot's witnesses on TR0 (workers)
  gates          3c's envelopes built; G2 on them, on top-1, top-5, top-25 and pilot 2's top-50
  ts0-folds      3c labels, 4a bisection, 4b pilot 2's top-50 on TS0 (workers)
  analyse        3c, 4a, 4b
Run in .venv-certfold. Records go to runs/2026-10-18_certfold_reviewer/metrics.json.
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

import scripts.certfold_reg2 as r2  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402

cf, cr = r2.cf, r2.cr
ME = "scripts/certfold_reviewer.py"
SPEC = "docs/certfold_reviewer_additions.md"
RUN_ID3 = "2026-10-18_certfold_reviewer"
OUT3 = R / "runs" / RUN_ID3
SHARDS3 = OUT3 / "shards"
TAG_LOTO, TAG_SCALE, TAG_PILOT_BOOT, TAG_PILOT2, TAG_ORDER, TAG_ALT = 80, 81, 82, 83, 84, 85
N_BOOT = 10000
N_PILOT_BOOT = 1000
WORKERS = 12
K_MAX = 50
WHOLE = tuple(f"whole:{X}" for X in r2.TABLES)
ABBR = {"Turner1999": "T99", "Andronescu2007": "A07", "Langdon2018": "L18"}

DESCRIBED = "descriptive done"
PILOT2 = "pilot 2 ranked"
GATED = "gates passed"
FOLDED = "TS0 folds done"
ANALYSED = "analysed"


# ----------------------------------------------------------------------------------
# records and gates
# ----------------------------------------------------------------------------------
def read_out3() -> dict:
    p = OUT3 / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {"run_id": RUN_ID3}


def write3(out: dict) -> None:
    OUT3.mkdir(parents=True, exist_ok=True)
    tmp = OUT3 / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT3 / "metrics.json")


def committed3() -> dict:
    rel = str((OUT3 / "metrics.json").relative_to(R))
    if cf._git("ls-files", "--error-unmatch", rel).returncode != 0:
        raise SystemExit(f"{rel} is not tracked: commit it first")
    if cf._git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"{rel} has uncommitted changes: commit it first")
    return json.loads(cf._git("show", f"HEAD:{rel}").stdout)


def gate(status) -> tuple[dict, dict]:
    """(record, provenance): the specification committed unmodified, a clean tree, the status one
    of `status` (None: no record yet), and the predecessor's record committed unmodified."""
    if cf._git("ls-files", "--error-unmatch", SPEC).returncode != 0:
        raise SystemExit(f"{SPEC} is not committed")
    if cf._git("diff", "--quiet", "HEAD", "--", SPEC).returncode != 0:
        raise SystemExit(f"{SPEC} has uncommitted changes")
    prov = r2.clean()
    out = read_out3()
    st = (status,) if isinstance(status, str) or status is None else tuple(status)
    if out.get("status") not in st:
        raise SystemExit(f"status is {out.get('status')!r}; this phase needs one of {st}")
    if status is not None:
        committed3()
    out.setdefault("specification", {
        "path": SPEC, "commit": cf._git("log", "-1", "--format=%H", "--", SPEC).stdout.strip(),
        "sha256": cf._sha(R / SPEC)})
    return out, prov


def drive(phase: str, n_w: int) -> list[int]:
    """Run `phase` in n_w worker processes (nice 10, single-threaded); stop all on a failure."""
    SHARDS3.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="")
    procs = [subprocess.Popen(
        ["nice", "-n", "10", sys.executable, str(R / ME), "--phase", phase, "--shard", str(w),
         "--of", str(n_w)], cwd=R, env=env,
        stdout=(SHARDS3 / f"{phase}_{w}.log").open("w"), stderr=subprocess.STDOUT)
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
        raise GuardFailure(f"{phase} workers exited {codes}; shards kept at {SHARDS3}")
    return [p.pid for p in procs]


def jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ----------------------------------------------------------------------------------
# the records every section reads
# ----------------------------------------------------------------------------------
def base_records() -> tuple[list[dict], list[dict]]:
    """(TS0 rows with stars, registration 2's kept records) with, on every MFE record, the
    certified flags at lambda = 0.5 and 2, retention over each pair of whole tables, and kept
    under each single whole table as a float."""
    rows = r2.ts0_rows_with_stars()
    tables = {}
    for f in sorted(r2.SHARDS2.glob("tables_*.jsonl")):
        for d in jsonl(f):
            tables[d["index"]] = d["mfe"]
    mx = r2.mxfold2_structures(rows, r2.committed2()["frozen"]["mxfold2_predictions"])
    recs = r2.kept_records(rows, tables, mx)
    margins = {(r["index"], tuple(q["pair"])): q["margins_dcal"] for r in rows
               for q in r["pairs"]}
    out = []
    for r in recs:
        if r["kind"] == "mfe":
            m = margins[(r["index"], r["pair"])]
            if (m[cr.PRIMARY] > 0) != r["certified_l1"]:
                raise GuardFailure("the lambda = 1 label disagrees with the run record")
            r = {**r, "cert_l05": m["envelope:0.5"] > 0, "cert_l2": m["envelope:2"] > 0}
            for k in WHOLE:
                r[f"one:{k}"] = float(r[f"kept:{k}"])
            for a in range(3):
                for b in range(a + 1, 3):
                    ka, kb = WHOLE[a], WHOLE[b]
                    r[f"two:{ABBR[r2.TABLES[a]]}+{ABBR[r2.TABLES[b]]}"] = (
                        float(r[f"kept:{ka}"]) + float(r[f"kept:{kb}"])) / 2
        out.append(r)
    return rows, out


def spec(key: str, cert: str) -> dict:
    """R4's statistic on `key` with certification read from `cert`: mean over certified p >= 0.99
    MFE pairs minus mean over uncertified ones; weights the p >= 0.99 pair counts; strata 0 and 1
    merged for reporting, as R4 and E2."""
    def hi(rs):
        return [r for r in rs if r["kind"] == "mfe" and r["p"] >= cr.P_HI]

    def split(rs):
        h = hi(rs)
        return [r[key] for r in h if r[cert]], [r[key] for r in h if not r[cert]]

    def stat(rs):
        c, u = split(rs)
        return sum(c) / len(c) - sum(u) / len(u)

    def defined(rs):
        c, u = split(rs)
        return bool(c) and bool(u)

    def extra(rs):
        c, u = split(rs)
        return {"n_certified": len(c), "n_uncertified": len(u)}

    return {"stat": stat, "weight": lambda rs: len(hi(rs)), "defined": defined,
            "extra": extra, "merge": {1: 0}}


def stratified(recs: list[dict], rows: list[dict], specs: dict, tag: int) -> dict:
    import scripts.certfold_exploratory as cx

    pool = [r for r in recs if r["kind"] == "unit"
            or (r["kind"] == "mfe" and r["p"] >= cr.P_HI)]
    groups = cx.strata_groups(pool, {r["index"]: r["stratum"] for r in rows})
    return cx.stratified_bootstrap(groups, n_boot=N_BOOT, seed_tag=tag, specs=specs)


def summarise(res: dict, names) -> dict:
    return {n: {"pooled_point": res[n]["pooled_point"],
                "pooled_interval": res[n]["pooled_interval"],
                "pooled_resamples_undefined": res[n]["pooled_resamples_undefined"],
                "strata_left_out": res[n]["strata_left_out"],
                "by_stratum": {k: {kk: v[kk] for kk in ("point", "interval", "n_certified",
                                                         "n_uncertified", "n_pairs")
                                   if kk in v}
                               for k, v in res[n]["by_stratum"].items()}}
            for n in names}


# ----------------------------------------------------------------------------------
# section 2: stability, made interpretable
# ----------------------------------------------------------------------------------
def stability_tables(recs: list[dict]) -> dict:
    hi = [r for r in recs if r["kind"] == "mfe" and r["p"] >= cr.P_HI]

    def block(rs):
        c = [r for r in rs if r["certified_l1"]]
        u = [r for r in rs if not r["certified_l1"]]
        out = {"n_certified": len(c), "n_uncertified": len(u), "tables": {}}
        for k in (*WHOLE, "all three"):
            def kept(r, k=k):
                return all(r[f"kept:{t}"] for t in WHOLE) if k == "all three" else r[f"kept:{k}"]
            kc, ku = sum(map(kept, c)), sum(map(kept, u))
            fc = kc / len(c) if c else None
            fu = ku / len(u) if u else None
            loss_c = None if fc is None else 1 - fc
            loss_u = None if fu is None else 1 - fu
            out["tables"][k] = {
                "kept_certified": fc, "kept_uncertified": fu,
                "difference_points": None if fc is None or fu is None else 100 * (fc - fu),
                "lost_certified": len(c) - kc, "lost_uncertified": len(u) - ku,
                "relative_reduction_in_loss": (None if not loss_u or loss_c is None
                                               else 1 - loss_c / loss_u)}
        ret_c = [r["ret_primary"] for r in c]
        ret_u = [r["ret_primary"] for r in u]
        out["mean_retention"] = {"certified": sum(ret_c) / len(ret_c) if c else None,
                                 "uncertified": sum(ret_u) / len(ret_u) if u else None}
        return out

    by = collections.defaultdict(list)
    for r in hi:
        by[r["stratum"]].append(r)
    return {"pool": "TS0 MFE pairs with p >= 0.99; certified = registration 1's label at "
                    "lambda = 1; 'all three' = kept under every whole table",
            "raw": block(hi), "by_stratum": {str(s): block(by[s]) for s in sorted(by)}}


# ----------------------------------------------------------------------------------
# section 3b: what the set covers of each published table
# ----------------------------------------------------------------------------------
FAMILY = {"stack": "stack", "int11": "int11", "int21": "int21", "int22": "int22",
          "bulge": "bulge", "hairpin": "hairpin initiation",
          "internal_loop": "interior initiation", "mismatchI": "interior mismatches",
          "mismatch1nI": "interior mismatches", "mismatch23I": "interior mismatches",
          "mismatchH": "hairpin mismatches", "ML_cc": "multiloop", "ML_ci": "multiloop",
          "ML_cu": "multiloop", "TerminalAU": "terminal AU", "ninio_m": "Ninio",
          "ninio_max": "Ninio"}


def table_coverage(B: dict) -> dict:
    from scripts.turner_nullspace import analytic_directions
    from scripts.typeb_common import full_tables

    cols, used, B4 = B["cols"], B["used"], B["B4"]
    t0 = cf.template()["theta"]
    unused = np.setdiff1d(np.arange(len(cols)), used)
    fam = np.array([FAMILY.get(k[0], k[0]) for k in cols])
    G = np.array([g[used] for g in analytic_directions(cols, 0, 0).values()]).T
    Qg, _ = np.linalg.qr(G)
    env = cf.read_out()["build_basis"]["published_envelope"]
    reach = B["u_env"] > 0
    out = {}
    for tb in full_tables():
        X = tb["id"].split(":")[-1]
        th = tb["theta"]
        d = np.array([th[k] - t0[k] for k in cols], dtype=np.float64) / 100.0
        d = np.where(np.array([th[k] >= 100000 for k in cols]), 0.0, d)
        inside = np.zeros(len(cols))
        inside[used] = B4 @ (B4.T @ d[used])
        outside = d - inside
        unread = np.zeros(len(cols))
        unread[unused] = outside[unused]
        gauge = np.zeros(len(cols))
        gauge[used] = Qg @ (Qg.T @ outside[used])
        rest = outside - unread - gauge
        parts = {"inside the set": inside, "outside: columns no TR0 reference reads": unread,
                 "outside: the gauge": gauge, "outside: the rest (the data can see it)": rest}
        n2 = float(d @ d)
        out[X] = {
            "norm_delta_kcal": float(np.sqrt(n2)),
            "parts": {p: {"norm_kcal": float(np.linalg.norm(v)),
                          "share_of_squared_norm": float(v @ v) / n2,
                          "by_family_norm_kcal": {f: float(np.linalg.norm(v[fam == f]))
                                                  for f in sorted(set(fam))
                                                  if np.linalg.norm(v[fam == f]) > 1e-9}}
                      for p, v in parts.items()},
            "inside_part_in_E1": {
                "max_entry_shift_kcal": float(np.abs(inside[reach]).max()),
                "delta_kcal": env["delta_kcal"],
                "norm_kcal": float(np.linalg.norm(inside)), "r_max_kcal": env["r_max_kcal"],
                "contained": bool(np.abs(inside[reach]).max() <= env["delta_kcal"] + 1e-12
                                  and np.linalg.norm(inside) <= env["r_max_kcal"] + 1e-12)},
            "parts_sum_to_delta": float(np.abs(sum(parts.values()) - d).max())}
    return out


# ----------------------------------------------------------------------------------
# section 4b: the pilot, bootstrapped
# ----------------------------------------------------------------------------------
def ranking_from(seq_recs: list[dict]) -> list[tuple[str, int]]:
    count = collections.Counter()
    for r in seq_recs:
        for q in r["pairs"]:
            if not q["certified"]:
                count.update(q["entries"])
    return sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))


def pilot_bootstrap(frozen: list) -> dict:
    recs = []
    for f in sorted(r2.SHARDS2.glob("rank_*.jsonl")):
        recs += jsonl(f)
    if [list(x) for x in ranking_from(recs)] != [list(x) for x in frozen]:
        raise GuardFailure("the recorded witnesses do not reproduce the frozen ranking")
    top50 = [k for k, _ in frozen[:50]]
    top10 = set(top50[:10])
    by = collections.defaultdict(list)
    for r in recs:
        by[r["stratum"]].append(r)
    rng = np.random.default_rng(np.random.SeedSequence([cf.SEED, TAG_PILOT_BOOT]))
    stay = collections.Counter()
    ov50, ov10_in10, ov10_in50 = [], [], []
    for _b in range(N_PILOT_BOOT):
        sub = []
        for s in sorted(by):
            idx = rng.integers(0, len(by[s]), size=len(by[s]))
            sub += [by[s][i] for i in idx.tolist()]
        rk = [k for k, _ in ranking_from(sub)]
        b50, b10 = set(rk[:50]), set(rk[:10])
        stay.update(k for k in top50 if k in b50)
        ov50.append(len(b50 & set(top50)))
        ov10_in10.append(len(b10 & top10))
        ov10_in50.append(len(b50 & top10))
    frac = {k: stay[k] / N_PILOT_BOOT for k in top50}
    n80 = sum(1 for v in frac.values() if v >= 0.8)

    def q(xs):
        return {"min": int(min(xs)), "p5": float(np.percentile(xs, 5)),
                "median": float(np.median(xs)), "p95": float(np.percentile(xs, 95)),
                "max": int(max(xs))}
    return {"n_resamples": N_PILOT_BOOT, "seed": [cf.SEED, TAG_PILOT_BOOT],
            "design": "the 300 pilot sequences resampled with replacement within their length "
                      "strata; the ranking rebuilt from the recorded witnesses",
            "frozen_top50_fraction_kept_in_top50": frac,
            "n_frozen_top50_kept_in_at_least_80pct": n80,
            "stable_under_resampling": n80 >= 40,
            "overlap_top50": q(ov50), "frozen_top10_in_top10": q(ov10_in10),
            "frozen_top10_in_top50": q(ov10_in50)}


# ----------------------------------------------------------------------------------
# section 5: conservativeness; what an uncertain label means
# ----------------------------------------------------------------------------------
def conservativeness(recs: list[dict]) -> dict:
    g = cf.read_out()["gate"]
    g2, g4 = g["G2"], g["G4"]["by_radius"]
    a = {}
    for r in ("1", "5", "20"):
        rob, cert = g2["exactly_robust_present"][r], g2["certified_present"][r]
        a[f"G2 ball r = {r}"] = {"exactly_robust": rob, "certified": cert, "missed": rob - cert}
    e = g2["envelope"]
    a["G2 envelope"] = {"exactly_robust": e["n_exactly_robust_present"],
                        "certified": e["n_certified_present"],
                        "missed": e["n_exactly_robust_present"] - e["n_certified_present"],
                        "note": "robust by the box bound, an upper bound on the envelope's worst "
                                "case, so missed is a lower bound"}
    for r, lab in (("1", "G4 ball r = 1"), ("5", "G4 ball r = 5"), ("envelope", "G4 envelope")):
        v = g4[r]
        a[lab] = {"exactly_robust": v["n_robust_present"], "certified": v["n_certified_present"],
                  "missed": v["relaxation_loss_present"], "n_sequences": v["n_run"],
                  "n_skipped": v["n_skipped"]}
    for v in a.values():
        v["fraction_missed"] = v["missed"] / v["exactly_robust"] if v["exactly_robust"] else None
    a["G4 envelope"]["note"] = "witnesses by the box bound, so missed is a lower bound"
    a["G4 ball r = 20"] = {"note": "all 100 sequences skipped: the band exceeded the cap"}

    mfe = [r for r in recs if r["kind"] == "mfe"]
    by = collections.defaultdict(lambda: collections.Counter())
    for r in mfe:
        k = ("certified" if r["certified_l1"] else
             "uncertain, observed to move" if r["freq_env"] < 1 else
             "uncertain, not observed to move (unresolved)")
        by[r["stratum"]][k] += 1
        by["all"][k] += 1
    a_rec = cf.read_out()["additions"]["A_uncertain_change"]
    moved = by["all"]["uncertain, observed to move"]
    return {"panel_A_present_pairs": a,
            "panel_B": {str(k): dict(v) for k, v in by.items()},
            "panel_B_note": "observed to move = left the MFE under at least one of the 200 "
                            "envelope draws; unresolved is not stable",
            "cross_check_addition_A_changed": [moved, a_rec["pooled"]]}


# ----------------------------------------------------------------------------------
# section 6: the radius against the nominal gap
# ----------------------------------------------------------------------------------
def radius_vs_gap(rows: list[dict], recs: list[dict], B: dict) -> dict:
    from scipy.stats import rankdata, spearmanr

    pool = [r for r in recs if r["kind"] == "mfe" and r["rho_env"] is not None]
    rho = np.array([r["rho_env"] for r in pool])
    gap = np.array([r["gap"] for r in pool])
    pr, pg = rankdata(rho) / len(pool), rankdata(gap) / len(pool)
    q = lambda x: np.minimum(3, (x * 4 - 1e-12).astype(int))  # noqa: E731  quartile 0..3
    qr_, qg = q(pr), q(pg)
    a_dis = (qr_ == 0) & (pg > 0.5)          # small radius, large gap
    b_dis = (qg == 0) & (pr > 0.5)          # small gap, large radius
    seq_of = {r["index"]: r for r in rows}
    expo = {r["index"]: None for r in rows}
    for s in cr.read_shards(cf.read_out()["run"]["workers"])[0]:
        if s["index"] in expo:
            expo[s["index"]] = s["exposure"]
    eq = np.quantile(np.array([expo[i] for i in expo]), [0.25, 0.5, 0.75])
    shapes = cf.reached_shapes(B["cols"], B["u_env"])
    P = cf.template()
    lc = cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    adj_cache: dict = {}

    def adjacent(idx, pair):
        if idx not in adj_cache:
            r = seq_of[idx]
            s = set()
            for d in lc.loops(r["seq"], r["star"]):
                if d["kind"] == "interior" and (d["k"] - d["i"] - 1, d["j"] - d["l"] - 1) in shapes:
                    s.update({(d["i"], d["j"]), (d["k"], d["l"])})
            adj_cache[idx] = s
        return tuple(pair) in adj_cache[idx]

    ctx = collections.defaultdict(lambda: collections.Counter())
    for n, r in enumerate(pool):
        keys = [f"stratum {r['stratum'] + 1}",
                f"exposure quartile {1 + int(np.searchsorted(eq, expo[r['index']], 'right'))}",
                "next to a reached interior loop" if adjacent(r["index"], r["pair"])
                else "not next to one"]
        for k in keys:
            ctx[k]["n"] += 1
            ctx[k]["small radius, large gap"] += int(a_dis[n])
            ctx[k]["small gap, large radius"] += int(b_dis[n])
    by_stratum = {}
    for s in sorted({r["stratum"] for r in pool}):
        m = np.array([r["stratum"] == s for r in pool])
        by_stratum[str(s)] = float(spearmanr(rho[m], gap[m]).statistic)
    conf = [[int(((qr_ == i) & (qg == j)).sum()) for j in range(4)] for i in range(4)]
    cases = {}
    for name, score in (("small radius, large gap", pg - pr), ("small gap, large radius", pr - pg)):
        order = sorted(range(len(pool)), key=lambda n: (-score[n], pool[n]["index"]))[:2]
        cases[name] = [worked_case(lc, B, seq_of[pool[n]["index"]], pool[n]) for n in order]
    return {"n_pairs": len(pool), "spearman": float(spearmanr(rho, gap).statistic),
            "spearman_by_stratum": by_stratum,
            "quartile_agreement": {"rows": "rho_env quartile 1..4", "cols": "gap quartile 1..4",
                                   "counts": conf,
                                   "same_quartile_fraction": sum(conf[i][i] for i in range(4))
                                   / len(pool)},
            "disagreement_definition": "lowest quartile of one, upper half of the other",
            "n_small_radius_large_gap": int(a_dis.sum()),
            "n_small_gap_large_radius": int(b_dis.sum()),
            "where": {k: dict(v) for k, v in sorted(ctx.items())},
            "worked_cases": cases}


def worked_case(lc, B: dict, row: dict, rec: dict) -> dict:
    """The pair's two witnesses: the nominal one (lambda = 0, which sets the gap) and the one at
    lambda = rho_env (which sets the radius), each with N(s) = E0(s) - E0(s*) and D(s), the summed
    lambda = 1 weight of the loops where it differs from s*."""
    seq, star = row["seq"], row["star"]
    i, j = rec["pair"]
    W1e = cf.Weights(B["u_env"], B["cols"], 1.0)
    smap = cf.loop_coeff_map(lc, seq, star, W1e)
    F = cf.Folder(lc, seq, star, cf.Weights(B["u_env"], B["cols"], 1.0))
    out = {"name": row["name"], "n_nt": len(seq), "pair": [i, j], "p": rec["p"],
           "gap_kcal": rec["gap"], "rho_env": rec["rho_env"]}
    for lab, lam in (("nominal_witness", 0.0), ("witness_at_rho", rec["rho_env"])):
        F.set_radius(lam)
        s, _x, e0 = F.without(i, j)
        D = cf.d_coeff(smap, cf.loop_coeff_map(lc, seq, s, W1e))
        out[lab] = {"lambda": lam, "N_kcal": (e0 - F.e0_star) / 100.0, "D_kcal": D,
                    "N_over_D": None if D <= 0 else (e0 - F.e0_star) / 100.0 / D,
                    "n_differing_loops": len(set(smap) ^ set(cf.loop_coeff_map(lc, seq, s,
                                                                               W1e)))}
    return out


# ----------------------------------------------------------------------------------
# phase: descriptive
# ----------------------------------------------------------------------------------
def descriptive() -> int:
    out, prov = gate(None)
    if "descriptive" in out:
        raise SystemExit("exists")
    t0 = time.time()
    rows, recs = base_records()
    B = cf.load_tr0_basis()
    head2 = r2.committed2()
    res = {"stability": stability_tables(recs)}
    loto = {f"two tables {k[4:]}": spec(k, "certified_l1")
            for k in recs[next(n for n, r in enumerate(recs) if r["kind"] == "mfe")]
            if k.startswith("two:")}
    loto.update({f"one table {k[4:]}": spec(f"one:{k}", "certified_l1") for k in WHOLE})
    loto["all three (R4, reproduced)"] = spec("ret_primary", "certified_l1")
    bs = stratified(recs, rows, loto, TAG_LOTO)
    res["leave_one_table_out"] = {**summarise(bs, loto), "bootstrap": bs["bootstrap"]}
    if abs(bs["all three (R4, reproduced)"]["pooled_point"]
           - head2["analysis_c"]["R4"]["R4_primary"]["pooled_point"]) > 1e-12:
        raise GuardFailure("the R4 point does not reproduce")
    sc = {"lambda 0.5": spec("ret_primary", "cert_l05"), "lambda 2": spec("ret_primary", "cert_l2")}
    bs2 = stratified(recs, rows, sc, TAG_SCALE)
    lab = cf.read_out()["analysis"]["sets"]["TS0"]["describe"]["labels"]
    res["scale"] = {"R4_statistic": {**summarise(bs2, sc), "bootstrap": bs2["bootstrap"]},
                    "certified_present_by_stratum": {
                        f"lambda {lm}": {s: v["fraction"] for s, v in
                                         lab[f"envelope:{lm}"]["certified_present"]
                                         ["by_stratum"].items()} for lm in ("0.5", "1", "2")}}
    res["table_coverage"] = table_coverage(B)
    res["pilot_bootstrap"] = pilot_bootstrap(head2["ranking"]["ranked"])
    res["conservativeness"] = conservativeness(recs)
    res["radius_vs_gap"] = radius_vs_gap(rows, recs, B)
    out["descriptive"] = {**res, "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = DESCRIBED
    write3(out)
    print(json.dumps({k: res["leave_one_table_out"][k]["pooled_interval"]
                      for k in loto}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# phase: the second pilot's ranking (TR0)
# ----------------------------------------------------------------------------------
def pilot2_sequences() -> tuple[list, list]:
    """300 TR0 sequences disjoint from the first pilot: the same ten equal-count length strata,
    30 from each, drawn by SeedSequence([SEED, 83]) from what the first pilot left."""
    rows, _ = cf.tr0_rows()
    first = {nm for nm, _s, _st in cf.pilot_sequences(rows)[0]}
    pool = sorted(rows, key=lambda r: (len(r[2]), r[1]))
    edges = np.linspace(0, len(pool), cf.PILOT_STRATA + 1).astype(int)
    rng = np.random.default_rng(np.random.SeedSequence([cf.SEED, TAG_PILOT2]))
    out, bounds = [], []
    per = cf.PILOT_SEQUENCES // cf.PILOT_STRATA
    for s, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        left = [r for r in pool[a:b] if r[1] not in first]
        picks = sorted(rng.choice(len(left), size=per, replace=False).tolist())
        out += [(left[i][1], left[i][2], s) for i in picks]
        bounds.append({"stratum": s, "n_available": len(left)})
    assert not first & {nm for nm, _s, _st in out}
    return out, bounds


def pilot2_rank_worker() -> int:
    import RNA

    from thermo.vienna_ref import reference_md

    w, n_w = r2.shard_args()
    B = cf.load_tr0_basis()
    P = cf.template()
    lc = cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    seqs, _ = pilot2_sequences()
    with open(SHARDS3 / f"pilot2_{w}.jsonl", "w") as fh:
        for n, (name, seq, stratum) in enumerate(seqs):
            if n % n_w != w:
                continue
            star = RNA.fold_compound(seq, reference_md()).mfe()[0]
            fh.write(json.dumps({"name": name, "n_nt": len(seq), "stratum": stratum,
                                 "pairs": r2.witness_entries(lc, B, seq, star, B["u_env"])})
                     + "\n")
            fh.flush()
    return 0


def pilot2_rank() -> int:
    out, prov = gate(DESCRIBED)
    t0 = time.time()
    seqs, bounds = pilot2_sequences()
    pids = drive("pilot2-rank-worker", WORKERS)
    recs = []
    for w in range(WORKERS):
        recs += jsonl(SHARDS3 / f"pilot2_{w}.jsonl")
    if len(recs) != len(seqs):
        raise GuardFailure(f"{len(recs)} of {len(seqs)} pilot-2 sequences")
    ranked = ranking_from(recs)
    frozen = [k for k, _ in r2.committed2()["ranking"]["ranked"]]
    mine = [k for k, _ in ranked]
    out["pilot2"] = {
        "design": "300 TR0 sequences disjoint from the first pilot, 30 per equal-count length "
                  "stratum, seed tag 83; witnesses as registration 2's rank-entries",
        "strata": bounds, "n_pairs": sum(len(r["pairs"]) for r in recs),
        "n_uncertain": sum(1 for r in recs for q in r["pairs"] if not q["certified"]),
        "ranked": [[k, c] for k, c in ranked], "ranked_sha256": r2.canon_sha(ranked),
        "overlap_top10_with_frozen_top10": len(set(mine[:10]) & set(frozen[:10])),
        "overlap_top50_with_frozen_top50": len(set(mine[:50]) & set(frozen[:50])),
        "frozen_top10_in_pilot2_top50": len(set(frozen[:10]) & set(mine[:50])),
        "files_sha256": {f"pilot2_{w}.jsonl": cf._sha(SHARDS3 / f"pilot2_{w}.jsonl")
                         for w in range(WORKERS)},
        "pids": pids, "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = PILOT2
    write3(out)
    print({k: out["pilot2"][k] for k in ("n_uncertain", "overlap_top10_with_frozen_top10",
                                         "overlap_top50_with_frozen_top50")})
    return 0


# ----------------------------------------------------------------------------------
# section 3c: the alternative basis
# ----------------------------------------------------------------------------------
def alt_basis(B: dict) -> tuple[dict, dict]:
    """B_excl, the part of span(B_TR0) TS0's references cannot see either, projected to the four
    tables as B_four is; and the per-entry bounds u of envelopes (i) and (ii) on it."""
    from scripts.turner_nullspace import Extractor, build_matrix

    P = cf.template()
    rows, _ = cf.tr0_ts0_rows()
    ts0 = [r for r in cf.distinct_rows(rows) if r[0] == "TS0" and "(" in r[3]]
    ex = Extractor(P["special"], 0, 0, lxc=P["lxc"])
    cnts = []
    for _t, _n, seq, db in ts0:
        c = ex.counts(seq, db)
        c.pop(("_offset",), 0)
        cnts.append(c)
    M = build_matrix(cnts, B["cols"]).tocsc()[:, B["used"]]
    V = np.asarray(M @ B["NT"], dtype=np.float64)
    _U, s, Vt = np.linalg.svd(V, full_matrices=True)
    tol = 1e-9 * s.max()
    rank = int((s > tol).sum())
    N = Vt[rank:].T
    NT_excl = B["NT"] @ N
    B4x, diag = cf.four_table_projection(dict(B, NT=NT_excl))
    a4 = np.zeros(len(B["cols"]))
    a4[B["used"]] = np.linalg.norm(B4x, axis=1)
    reg = cf.read_out()["build_basis"]["published_envelope"]
    reach = a4 > cf.ROW_NOISE
    u_i = np.where(reach, np.minimum(reg["delta_kcal"], reg["r_max_kcal"] * a4), 0.0)
    Bx = dict(B, NT=NT_excl, B4=B4x, a_four=a4)
    radii = cf.published_radii(Bx, B4x)
    env_ii, u_ii = cf.published_envelope(Bx, B4x, a4, radii)
    rec = {"n_TS0_references": len(ts0), "rank_of_TS0_on_B_TR0": rank,
           "dim_B_excl": int(NT_excl.shape[1]), "dim_B_four_excl": int(B4x.shape[1]),
           "singular_values_at_the_gap": [float(x) for x in s[max(0, rank - 3):rank + 3]],
           "max_TS0_residual_on_B_excl": float(np.abs(V @ N).max()),
           "four_table_projection": diag,
           "reached_entries": {"registered": int((B["u_env"] > 0).sum()),
                               "B_excl": int(reach.sum())},
           "variant_i": {"delta_kcal": reg["delta_kcal"], "r_max_kcal": reg["r_max_kcal"],
                         "n_entries_u_below_registered": int((u_i < B["u_env"] - 1e-12).sum()),
                         "n_entries_u_above_registered": int((u_i > B["u_env"] + 1e-12).sum())},
           "variant_ii": {"delta_kcal": env_ii["delta_kcal"], "r_max_kcal": env_ii["r_max_kcal"],
                          "delta_from": env_ii["delta_from"], "radii": radii,
                          "n_entries_u_below_registered": int((u_ii < B["u_env"] - 1e-12).sum()),
                          "n_entries_u_above_registered": int((u_ii > B["u_env"] + 1e-12).sum())}}
    return rec, {"i": dict(Bx, u_env=u_i), "ii": dict(Bx, u_env=u_ii)}


def measured_sets_extra(B: dict, frozen: list, pilot2_ranked: list) -> dict[str, list]:
    """The nested orderings of 4a and pilot 2's top-50: top (the frozen ranking's first 50 in
    order), random d (a tag-84 permutation of registration 2's random-50 draw d), pilot2."""
    by_str = {str(k): k for k in B["cols"]}
    sets = r2.measured_sets(B, frozen)
    rng = np.random.default_rng(np.random.SeedSequence([cf.SEED, TAG_ORDER]))
    orders = {"top": [by_str[k] for k, _ in frozen[:K_MAX]]}
    for d in range(r2.N_RANDOM):
        ents = sets[f"random:{K_MAX}:{d}"]
        orders[f"random:{d}"] = [ents[i] for i in rng.permutation(len(ents)).tolist()]
    orders["pilot2"] = [by_str[k] for k, _ in pilot2_ranked[:K_MAX]]
    return orders


def g2_on(lc, B: dict, Bm: dict, seqs, pos, label: str) -> dict:
    W1 = cf.Weights(Bm["a_four"], Bm["cols"], 1.0)
    tot = collections.Counter()
    for _src, seq, _n in seqs:
        r = cf.g2_one(lc, Bm, pos, W1, seq, (1.0,), with_rho=False)
        bad = r["envelope"]["n_false_certificates"] + r["by_radius"]["1"]["n_false_certificates"]
        if bad:
            raise GuardFailure(f"G2 on {label}: {bad} false certificates on {seq}")
        for k in ("n_certified_present", "n_exactly_robust_present", "n_false_certificates"):
            tot[k] += r["envelope"][k]
    planted = sum(cf.g2_one(lc, Bm, pos, W1, seq, (1.0,), scale=0.5, with_rho=False)
                  ["envelope"]["n_false_certificates"]
                  for _s, seq, _n in seqs[:cf.G2_POWER_SEQUENCES])
    if planted == 0:
        raise GuardFailure(f"G2 on {label}: halving the weights produced no false certificate")
    return {**dict(tot), "power_check_false_certificates_at_half_weights": planted}


def gates() -> int:
    out, prov = gate(PILOT2)
    t0 = time.time()
    B = cf.load_tr0_basis()
    P = cf.template()
    lc = cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    rows, _ = cf.tr0_rows()
    W1 = cf.Weights(B["a_four"], B["cols"], 1.0)
    pos = {B["cols"][e]: k for k, e in enumerate(B["used"])}
    seqs, _ = cf.g2_sequences(rows, cf.G2_SEQUENCES, 16, cf.G2_MAX_N, cf.SEED, lc, W1)
    rec, alts = alt_basis(B)
    head2 = r2.committed2()
    orders = measured_sets_extra(B, head2["ranking"]["ranked"], committed3()["pilot2"]["ranked"])
    g2 = {}
    for v, Bm in alts.items():
        posx = {Bm["cols"][e]: k for k, e in enumerate(Bm["used"])}
        g2[f"alternative basis, envelope ({v})"] = g2_on(lc, B, Bm, seqs, posx, v)
        print(v, g2[f"alternative basis, envelope ({v})"], flush=True)
    for lab, ents in (("top-1", orders["top"][:1]), ("top-5", orders["top"][:5]),
                      ("top-25", orders["top"][:25]), ("pilot 2 top-50", orders["pilot2"])):
        g2[lab] = g2_on(lc, B, dict(B, u_env=r2.measured_u(B, ents)), seqs, pos, lab)
        print(lab, g2[lab], flush=True)
    np.savez(OUT3 / "alt_u.npz", u_i=alts["i"]["u_env"], u_ii=alts["ii"]["u_env"])
    out["gates"] = {"alternative_basis": rec, "G2": g2,
                    "orders": {k: [str(e) for e in v] for k, v in orders.items()},
                    "alt_u_sha256": cf._sha(OUT3 / "alt_u.npz"),
                    "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = GATED
    write3(out)
    return 0


# ----------------------------------------------------------------------------------
# phase: TS0 folds (3c labels, 4a bisection, 4b pilot 2's top-50)
# ----------------------------------------------------------------------------------
def ts0_folds_worker() -> int:
    w, n_w = r2.shard_args()
    rec3 = committed3()
    B = cf.load_tr0_basis()
    P = cf.template()
    lc = cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    z = np.load(OUT3 / "alt_u.npz")
    alt_u = {"i": z["u_i"], "ii": z["u_ii"]}
    by_str = {str(k): k for k in B["cols"]}
    orders = {k: [by_str[e] for e in v] for k, v in rec3["gates"]["orders"].items()}
    meas = {}
    for f in sorted(r2.SHARDS2.glob("measured_*.jsonl")):
        for d in jsonl(f):
            meas[d["index"]] = d["sets"]
    rows = r2.ts0_rows_with_stars()
    assign = cr.shard_of(rows, n_w)
    mine = sorted((r for r in rows if assign[r["index"]] == w),
                  key=lambda r: (-len(r["seq"]), r["index"]))
    p2set = set(orders["pilot2"])
    path = SHARDS3 / f"ts0_{w}.jsonl"
    done = {d["index"] for d in jsonl(path)} if path.exists() else set()
    with open(path, "a") as fh:
        for r in mine:
            if r["index"] in done:
                continue
            ts = time.time()
            seq, star = r["seq"], r["star"]
            reg1 = {tuple(q["pair"]): q["margins_dcal"][cr.PRIMARY] for q in r["pairs"]}
            out = {"index": r["index"], "alt": {}, "curve": [], "pilot2": None}
            folds = 0
            # 3c: every MFE pair under each alternative envelope
            for v, u in alt_u.items():
                F = cf.Folder(lc, seq, star, cf.Weights(u, B["cols"], 1.0))
                out["alt"][v] = {f"{i},{j}": int(F.without(i, j)[1] - F.eprime_star)
                                 for (i, j) in reg1}
                folds += F.folds
            # 4a: bisection along each nested ordering, for the pairs certified at K = 50
            m = meas[r["index"]]
            cache: dict = {}

            def margin(order, K, i, j, cache=cache, seq=seq, star=star):
                key = (order, K)
                if key not in cache:
                    cache[key] = cf.Folder(lc, seq, star, cf.Weights(
                        r2.measured_u(B, orders[order][:K]), B["cols"], 1.0))
                F = cache[key]
                return int(F.without(i, j)[1] - F.eprime_star)

            for order in [k for k in orders if k != "pilot2"]:
                sid = "top:50" if order == "top" else f"random:50:{order.split(':')[1]}"
                new50 = {tuple(p) for p in m[sid]["newly_certified"]}
                new10 = ({tuple(p) for p in m["top:10"]["newly_certified"]}
                         if order == "top" else None)
                for (i, j) in sorted(new50):
                    if margin(order, K_MAX, i, j) <= 0:
                        raise GuardFailure(f"{r['index']} {order} {(i, j)}: not certified at "
                                           f"K = 50 though registration 2 certified it")
                    lo, hi = 0, K_MAX
                    if new10 is not None:
                        lo, hi = (lo, 10) if (i, j) in new10 else (10, hi)
                    while hi - lo > 1:
                        mid = (lo + hi) // 2
                        if margin(order, mid, i, j) > 0:
                            hi = mid
                        else:
                            lo = mid
                    out["curve"].append([order, i, j, hi])
            folds += sum(F.folds for F in cache.values())
            # 4b: pilot 2's top-50 as a measured set, as registration 2's measured labels
            unc = [p for p, mg in reg1.items() if mg <= 0]
            if unc and r2.readable_entries(seq, p2set):
                F = cf.Folder(lc, seq, star, cf.Weights(r2.measured_u(B, orders["pilot2"]),
                                                        B["cols"], 1.0))
                new = []
                for (i, j) in unc:
                    mg = int(F.without(i, j)[1] - F.eprime_star)
                    if mg < reg1[(i, j)]:
                        raise GuardFailure(f"{r['index']} pilot2 {(i, j)}: margin below "
                                           f"registration 1's")
                    if mg > 0:
                        new.append([i, j])
                out["pilot2"] = new
                folds += F.folds
            else:
                out["pilot2"] = []
            out.update({"folds": folds, "seconds": round(time.time() - ts, 2)})
            fh.write(json.dumps(out) + "\n")
            fh.flush()
    return 0


def ts0_folds() -> int:
    out, prov = gate((GATED, FOLDED))
    if "ts0_folds" in out and "--resume" not in sys.argv:
        raise SystemExit("exists")
    t0 = time.time()
    if cf._sha(OUT3 / "alt_u.npz") != out["gates"]["alt_u_sha256"]:
        raise GuardFailure("alt_u.npz changed since the gates")
    load = os.getloadavg()
    pids = drive("ts0-folds-worker", WORKERS)
    recs = []
    for w in range(WORKERS):
        recs += jsonl(SHARDS3 / f"ts0_{w}.jsonl")
    if sorted(d["index"] for d in recs) != sorted(r["index"] for r in r2.ts0_rows_with_stars()):
        raise GuardFailure("the TS0 folds do not cover TS0 exactly once")
    out["ts0_folds"] = {"n_sequences": len(recs), "folds": sum(d["folds"] for d in recs),
                        "files_sha256": {f"ts0_{w}.jsonl": cf._sha(SHARDS3 / f"ts0_{w}.jsonl")
                                         for w in range(WORKERS)},
                        "workers": WORKERS, "pids": pids, "load_at_start": load,
                        "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = FOLDED
    write3(out)
    print(out["ts0_folds"]["folds"], out["ts0_folds"]["seconds"])
    return 0


# ----------------------------------------------------------------------------------
# phase: analyse
# ----------------------------------------------------------------------------------
def analyse() -> int:
    out, prov = gate(FOLDED)
    t0 = time.time()
    for f, h in out["ts0_folds"]["files_sha256"].items():
        if cf._sha(SHARDS3 / f) != h:
            raise GuardFailure(f"{f} changed since the folds")
    folds = {}
    for w in range(out["ts0_folds"]["workers"]):
        for d in jsonl(SHARDS3 / f"ts0_{w}.jsonl"):
            folds[d["index"]] = d
    rows, recs = base_records()
    head2 = r2.committed2()
    n_all = sum(1 for r in recs if r["kind"] == "mfe")
    strata = sorted({r["stratum"] for r in rows})
    n_by = collections.Counter(r["stratum"] for r in recs if r["kind"] == "mfe")
    res = {}

    # 3c: labels, transitions, R4
    for r in recs:
        if r["kind"] == "mfe":
            d = folds[r["index"]]["alt"]
            key = f"{r['pair'][0]},{r['pair'][1]}"
            r["cert_alt_i"] = d["i"][key] > 0
            r["cert_alt_ii"] = d["ii"][key] > 0
    mfe = [r for r in recs if r["kind"] == "mfe"]
    alt = {}
    for v in ("i", "ii"):
        k = f"cert_alt_{v}"
        alt[v] = {"certified_present": sum(r[k] for r in mfe) / n_all,
                  "by_stratum": {str(s): sum(r[k] for r in mfe if r["stratum"] == s) / n_by[s]
                                 for s in strata},
                  "certified_under_E_not_here": sum(r["certified_l1"] and not r[k] for r in mfe),
                  "certified_here_not_under_E": sum(r[k] and not r["certified_l1"] for r in mfe)}
    if alt["i"]["certified_under_E_not_here"]:
        raise GuardFailure("variant (i) lost a certificate: its u is not below the envelope's")
    sp = {f"alternative ({v})": spec("ret_primary", f"cert_alt_{v}") for v in ("i", "ii")}
    sp["registered labels (R4, reproduced)"] = spec("ret_primary", "certified_l1")
    bs = stratified(recs, rows, sp, TAG_ALT)
    alt["R4_statistic"] = {**summarise(bs, sp), "bootstrap": bs["bootstrap"]}
    alt["reading"] = {v: ("does not depend on the 49 TS0-pinnable directions"
                          if (bs[f"alternative ({v})"]["pooled_interval"] or [0])[0] > 0
                          else "depends on them") for v in ("i", "ii")}
    alt["registered_certified_present"] = sum(r["certified_l1"] for r in mfe) / n_all
    res["alternative_basis"] = alt

    # 4a: the curves
    kstar = collections.defaultdict(list)
    kstar_s = collections.defaultdict(lambda: collections.defaultdict(list))
    st_of = {r["index"]: r["stratum"] for r in rows}
    for idx, d in folds.items():
        for order, _i, _j, k in d["curve"]:
            kstar[order].append(k)
            kstar_s[order][st_of[idx]].append(k)
    curves = {o: [sum(1 for k in kstar[o] if k <= K) / n_all for K in range(1, K_MAX + 1)]
              for o in out["gates"]["orders"] if o != "pilot2"}
    reg_gain = head2["analysis_d"]["gain"]
    if abs(curves["top"][9] - reg_gain["top:10"]) > 1e-12 or abs(
            curves["top"][-1] - reg_gain["top:50"]) > 1e-12:
        raise GuardFailure("the top curve does not reproduce registration 2's K = 10 and 50")
    for d in range(r2.N_RANDOM):
        if abs(curves[f"random:{d}"][-1] - reg_gain[f"random:50:{d}"]) > 1e-12:
            raise GuardFailure(f"random ordering {d} does not end at its registered gain")
    rnd = np.array([curves[f"random:{d}"] for d in range(r2.N_RANDOM)])
    top = np.array(curves["top"])
    half = top[-1] / 2
    res["efficiency_curve"] = {
        "K": list(range(1, K_MAX + 1)), "top_gain": curves["top"],
        "random_median": np.median(rnd, axis=0).tolist(),
        "random_p5": np.percentile(rnd, 5, axis=0).tolist(),
        "random_p95": np.percentile(rnd, 95, axis=0).tolist(),
        "random_curves": {k: v for k, v in curves.items() if k.startswith("random")},
        "top_marginal_gain_per_entry": np.diff(np.concatenate([[0.0], top])).tolist(),
        "K_half": int(np.argmax(top >= half) + 1),
        "top_gain_by_stratum_K": {str(s): [sum(1 for k in kstar_s["top"][s] if k <= K) / n_by[s]
                                           for K in range(1, K_MAX + 1)] for s in strata},
        "orders_seed": [cf.SEED, TAG_ORDER]}

    # 4b: pilot 2's top-50 on TS0
    new2 = sum(len(d["pilot2"]) for d in folds.values())
    p2 = committed3()["pilot2"]
    res["pilot2_on_TS0"] = {"gain_K50": new2 / n_all, "newly_certified": new2,
                            "frozen_gain_K50": reg_gain["top:50"],
                            "random50_p95": head2["analysis_d"]["R6"]["50"]["random_p95"],
                            "exceeds_random_p95": new2 / n_all
                            > head2["analysis_d"]["R6"]["50"]["random_p95"],
                            "overlap_top50_with_frozen_top50":
                                p2["overlap_top50_with_frozen_top50"]}
    out["analysis"] = {**res, "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = ANALYSED
    write3(out)
    print(json.dumps({"alt": {v: alt[v]["certified_present"] for v in ("i", "ii")},
                      "alt_R4": {k: v["pooled_interval"] for k, v in alt["R4_statistic"].items()
                                 if isinstance(v, dict) and "pooled_interval" in v},
                      "K_half": res["efficiency_curve"]["K_half"],
                      "pilot2": res["pilot2_on_TS0"]}, indent=1))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"descriptive": descriptive,
              "pilot2-rank": pilot2_rank, "pilot2-rank-worker": pilot2_rank_worker,
              "gates": gates,
              "ts0-folds": ts0_folds, "ts0-folds-worker": ts0_folds_worker,
              "analyse": analyse}[ph]())
