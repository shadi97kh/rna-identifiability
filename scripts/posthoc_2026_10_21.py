#!/usr/bin/env python3
"""Four POST HOC additions, as specified before any code in
docs/posthoc_additions_2026-10-21.md (the analysis freeze lifted for them on
2026-10-21). C4 was decided in the specification itself (skipped by its own condition), so it
has no phase here.

PHASES (each refuses unless the specification is committed unmodified; C3's steps refuse unless
the previous step's record is committed):
  c1            retention with the nominal gap held fixed      runs/2026-10-21_retention_gap
  c2            identical columns inside Type B
                                              runs/2026-10-21_typeb_identical_columns
  c3-typea      C3 steps 1 and 2 (Type A, invisibility)        runs/2026-10-21_certfold_special_hp
  c3-gate       C3 step 3 (G1, G2 with the power check)
  c3-run        C3 step 4 (TS0 labels, 8 workers), then c3-analyse
Run in .venv-certfold.
"""

from __future__ import annotations

import bisect
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

import scripts.certfold_checks as rv  # noqa: E402
import scripts.certfold_default_cert as dc  # noqa: E402
import scripts.certfold_exploratory as cx  # noqa: E402

r2, cf, cr = dc.r2, dc.cf, dc.cr
GuardFailure = dc.GuardFailure

ME = "scripts/posthoc_2026_10_21.py"
SPEC = "docs/posthoc_additions_2026-10-21.md"
TAG_C1 = 95
N_BOOT = 10000
MIN_PER_GROUP = 20
PRIMES = (2_147_483_647, 1_000_000_007)          # Algorithm S1's (turner_nullspace.analyse_config)
SPECIAL = (0, 1)                                 # (dangles, special_hp) for C3
N_WORKERS = 8
OUT = {"c1": R / "runs" / "2026-10-21_retention_gap",
       "c2": R / "runs" / "2026-10-21_typeb_identical_columns",
       "c3": R / "runs" / "2026-10-21_certfold_special_hp"}
C3_TYPEA, C3_GATE, C3_RUN, C3_DONE = ("steps 1-2 passed", "gate passed", "run done", "analysed")


# ----------------------------------------------------------------------------------
# records and gates
# ----------------------------------------------------------------------------------
def read_out(item: str) -> dict:
    p = OUT[item] / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {"run_id": OUT[item].name}


def write(item: str, out: dict) -> None:
    OUT[item].mkdir(parents=True, exist_ok=True)
    tmp = OUT[item] / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT[item] / "metrics.json")


def committed(item: str) -> None:
    rel = str((OUT[item] / "metrics.json").relative_to(R))
    if cf._git("ls-files", "--error-unmatch", rel).returncode != 0:
        raise SystemExit(f"{rel} is not tracked: commit it first")
    if cf._git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"{rel} has uncommitted changes: commit it first")


def gate(item: str, status) -> tuple[dict, dict]:
    if cf._git("ls-files", "--error-unmatch", SPEC).returncode != 0:
        raise SystemExit(f"{SPEC} is not committed")
    if cf._git("diff", "--quiet", "HEAD", "--", SPEC).returncode != 0:
        raise SystemExit(f"{SPEC} has uncommitted changes")
    cr._in_certfold_env()
    prov = r2.clean()
    out = read_out(item)
    if out.get("status") != status:
        raise SystemExit(f"{item}: status is {out.get('status')!r}; this phase needs {status!r}")
    if status is not None:
        committed(item)
    out.setdefault("specification", {
        "path": SPEC, "commit": cf._git("log", "-1", "--format=%H", "--", SPEC).stdout.strip(),
        "sha256": cf._sha(R / SPEC)})
    out["label"] = "POST HOC (docs/posthoc_additions_2026-10-21.md)"
    return out, prov


# ----------------------------------------------------------------------------------
# C1: retention with the nominal gap held fixed
# ----------------------------------------------------------------------------------
def rank_bins(values: list[float], n_bins: int) -> list[int]:
    """floor(n_bins k / N), k the number of values strictly below: ties stay in one bin."""
    s = sorted(values)
    return [n_bins * bisect.bisect_left(s, v) // len(values) for v in values]


def c1_groups(pool: list[dict], key) -> tuple[dict, dict]:
    """{stratum: {sequence index: records}} after the drop rule, and what was dropped."""
    by: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in pool:
        by[key(r)][r["index"]].append(r)
    kept, dropped = {}, {}
    for s, units in sorted(by.items()):
        rs = [r for u in units.values() for r in u]
        nc = sum(1 for r in rs if r["certified_l1"])
        nu = len(rs) - nc
        if nc < MIN_PER_GROUP or nu < MIN_PER_GROUP:
            dropped[str(s)] = {"n_pairs": len(rs), "n_certified": nc, "n_uncertified": nu}
        else:
            kept[s] = dict(units)
    return kept, dropped


def c1() -> int:
    out, prov = gate("c1", None)
    t0 = time.time()
    rows, recs = rv.base_records()
    gap = {(r["index"], tuple(q["pair"])): float(q["nominal_gap_kcal"]) for r in rows
           for q in r["pairs"]}
    stratum = {r["index"]: r["stratum"] for r in rows}
    pool = [r for r in recs if r["kind"] == "mfe" and r["p"] >= cr.P_HI]
    missing = [r for r in pool if (r["index"], tuple(r["pair"])) not in gap]
    if missing:
        raise GuardFailure(f"{len(missing)} pool pairs have no nominal gap in registration 1")
    g = [gap[(r["index"], tuple(r["pair"]))] for r in pool]
    dec, ter = rank_bins(g, 10), rank_bins(g, 3)
    pool = [{**r, "gap_kcal": gv, "gap_decile": d, "gap_tercile": t}
            for r, gv, d, t in zip(pool, g, dec, ter, strict=True)]
    spec = {**r2._r4_spec("ret_primary"), "merge": {}}
    res = {}
    for name, key in (("primary_gap_deciles", lambda r: r["gap_decile"]),
                      ("secondary_length_x_gap_tercile",
                       lambda r: 3 * (0 if stratum[r["index"]] == 1 else stratum[r["index"]])
                       + r["gap_tercile"])):
        kept, dropped = c1_groups(pool, key)
        bs = cx.stratified_bootstrap(kept, n_boot=N_BOOT, seed_tag=TAG_C1, specs={name: spec})
        b = bs[name]
        edges = {}
        for s in kept:
            gs = [r["gap_kcal"] for u in kept[s].values() for r in u]
            edges[str(s)] = [min(gs), max(gs)]
        res[name] = {
            "pooled_point": b["pooled_point"], "pooled_interval": b["pooled_interval"],
            "pooled_resamples_undefined": b["pooled_resamples_undefined"],
            "by_stratum": {k: {kk: v.get(kk) for kk in ("point", "interval", "n_certified",
                                                        "n_uncertified", "n_pairs",
                                                        "mean_certified", "mean_uncertified")}
                           for k, v in b["by_stratum"].items()},
            "gap_range_kcal_by_stratum": edges,
            "strata_dropped": dropped,
            "n_pairs_dropped": sum(v["n_pairs"] for v in dropped.values()),
            "n_pairs_kept": sum(len(u) for s in kept.values() for u in s.values())}
    iv = res["primary_gap_deciles"]["pooled_interval"]
    out["c1"] = {
        "pool": "TS0 MFE pairs with p >= 0.99; certified = registration 1's lambda = 1 label; "
                "retention = ret_primary (three whole tables); gap = nominal_gap_kcal",
        "n_pool": len(pool),
        "n_certified": sum(1 for r in pool if r["certified_l1"]),
        "statistic": "certfold_reg2._r4_spec('ret_primary'), no strata merged",
        "bootstrap": {"n_boot": N_BOOT, "seed": [cf.SEED, TAG_C1],
                      "units": "each sequence's pairs within a stratum"},
        "drop_rule": f"a stratum with fewer than {MIN_PER_GROUP} certified or uncertified pairs",
        **res,
        "reading": ("certified pairs keep a retention advantage when the gap is held fixed"
                    if iv is not None and iv[0] > 0 else
                    "the advantage is not shown once the gap is held fixed"),
        "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = "done"
    write("c1", out)
    print(json.dumps({k: v for k, v in out["c1"].items()
                      if k not in ("provenance",)}, indent=1, default=str)[:4000])
    return 0


# ----------------------------------------------------------------------------------
# C2: identical columns inside Type B
# ----------------------------------------------------------------------------------
def exact_rank(A: np.ndarray) -> dict:
    """Rank over GF(p) for both primes, and over the rationals by fraction-free elimination."""
    from fractions import Fraction

    from scripts.turner_nullspace import rank_mod_p

    ranks = {str(p): rank_mod_p(A, p) for p in PRIMES}
    M = [[Fraction(int(x)) for x in row] for row in A.tolist()]
    r, rows, cols = 0, len(M), len(M[0]) if M else 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c] / M[r][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r], strict=True)]
        r += 1
        if r == rows:
            break
    ranks["rational"] = r
    if len(set(ranks.values())) != 1:
        raise GuardFailure(f"the exact ranks disagree: {ranks}")
    return ranks


def c2_one(rows, label: str) -> dict:
    from scripts.turner_nullspace import analytic_directions
    from scripts.turner_typeb import compute_basis

    P = cf.template()
    B = compute_basis(rows, P)
    A = B["A"].tocsc()
    n_used = A.shape[1]
    an = analytic_directions(B["cols"], 0, 0)
    Au = np.stack([np.rint(g[B["used"]]).astype(np.int64) for g in an.values()
                   if np.abs(g[B["used"]]).max() > 0], axis=1)
    if Au.shape[1] != 2:
        raise GuardFailure(f"{label}: {Au.shape[1]} analytic directions on the used columns")
    # identical integer columns
    groups: dict = collections.defaultdict(list)
    for j in range(n_used):
        s, e = A.indptr[j], A.indptr[j + 1]
        groups[(tuple(A.indices[s:e].tolist()), tuple(A.data[s:e].tolist()))].append(j)
    multi = [g for g in groups.values() if len(g) >= 2]
    basis = [(g[0], j) for g in multi for j in g[1:]]          # e_a - e_b
    dim_d = len(basis)
    for a, b in basis:                                           # D lies in null(M), exactly
        if (A[:, a] != A[:, b]).nnz:
            raise GuardFailure(f"{label}: grouped columns {a}, {b} differ")
    # D n Type B: d orthogonal to N1 and N2, so dim D - rank of the 2 x dim D inner products
    G = np.array([[int(Au[a, k] - Au[b, k]) for a, b in basis] for k in range(2)],
                 dtype=np.int64)
    rank_g = exact_rank(G)
    dim_db = dim_d - rank_g["rational"]
    # D n A_r: x in D iff zero off the groups and summing to zero in each; on A_r's basis
    in_group = np.zeros(n_used, bool)
    for g in multi:
        in_group[g] = True
    cons = [Au[~in_group].T]                                      # zero on ungrouped columns
    cons.append(np.stack([Au[g].sum(axis=0) for g in multi], axis=1) if multi
                else np.zeros((2, 0), np.int64))
    K = np.concatenate(cons, axis=1).T                            # constraints x 2
    rank_k = exact_rank(K)
    dim_da = 2 - rank_k["rational"]
    nt = B["NT"].shape[1]
    if not (0 <= dim_da <= 2 and 0 <= dim_db <= nt):
        raise GuardFailure(f"{label}: dimensions out of range: {dim_da}, {dim_db} of {nt}")
    # the table split of D n Type B: the trace of its projector by table (floating point)
    split = {}
    if dim_db:
        Dm = np.zeros((n_used, dim_d))
        for k, (a, b) in enumerate(basis):
            Dm[a, k], Dm[b, k] = 1.0, -1.0
        _u, s_, vt = np.linalg.svd(G.astype(float), full_matrices=True)
        null_g = vt[int((s_ > 1e-9).sum()):].T                   # dim D x dim_db
        Q, _ = np.linalg.qr(Dm @ null_g)
        if Q.shape[1] != dim_db:
            raise GuardFailure(f"{label}: the float basis of D n Type B has the wrong size")
        w = (Q ** 2).sum(axis=1)
        tables = collections.Counter()
        for j, c in enumerate(B["used"]):
            tables[B["cols"][c][0]] += float(w[j])
        split = {t: v / dim_db for t, v in sorted(tables.items(), key=lambda kv: -kv[1])
                 if v / dim_db > 1e-12}
        # the float basis lies in Type B: orthogonal to the analytic directions
        if np.abs(Au.T.astype(float) @ Q).max() > 1e-9:
            raise GuardFailure(f"{label}: D n Type B is not orthogonal to Type A")
    sizes = collections.Counter(len(g) for g in multi)
    return {"n_rows": B["n_rows"], "n_used_columns": n_used, "type_b_dim": nt,
            "n_groups_two_or_more": len(multi),
            "group_sizes": {str(k): v for k, v in sorted(sizes.items())},
            "dim_D": dim_d, "dim_D_cap_type_b": dim_db, "dim_D_cap_type_a_restricted": dim_da,
            "rank_inner_products": rank_g, "rank_membership_on_type_a": rank_k,
            "type_b_split_by_table": split}


def c2() -> int:
    out, prov = gate("c2", None)
    t0 = time.time()
    from scripts.turner_nullspace_tr0 import distinct_rows, tr0_ts0_rows

    rows, _ = tr0_ts0_rows()
    both = [r for r in distinct_rows(rows) if "(" in r[3]]
    tr0, _ = cf.tr0_rows()
    out["c2"] = {"tr0_ts0": c2_one(both, "TR0+TS0"), "tr0": c2_one(tr0, "TR0"),
                 "method": "exact: identical integer columns, dimensions by exact rank over the "
                           "rationals and modulo both primes; the table split in floating point",
                 "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = "done"
    write("c2", out)
    print(json.dumps({k: v for k, v in out["c2"].items() if k != "provenance"}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# C3: special hairpins on
# ----------------------------------------------------------------------------------
def c3_typea() -> int:
    out, prov = gate("c3", None)
    t0 = time.time()
    from scripts.turner_nullspace import (
        analyse_config,
        dump_params,
        generating_set,
        theta_from_par,
    )

    P = theta_from_par(dump_params())
    rows = generating_set(P["special"])
    res, _ = analyse_config(*SPECIAL, P["theta"], P["special"], rows,
                            np.random.default_rng(np.random.SeedSequence([cf.SEED, TAG_C1])),
                            full_check=False)
    er, an = res["exact_rank"], res["analytic"]
    if not er["agrees_with_eigh"]:
        raise GuardFailure(f"step 1: the exact rank disagrees with the eigensolver: {er}")
    if not (an["dim_equals_null_dim"] and all(v["null_to_1e-9"]
                                              for v in an["directions"].values())):
        raise GuardFailure(f"step 1: the analytic directions do not span the null space: {an}")
    # step 2: TR0's references at (0, 1) against B_four
    EN = dc.use_model(*SPECIAL)
    B = cf.load_tr0_basis()
    pos = {B["cols"][e]: k for k, e in enumerate(B["used"])}
    tr0, _ = cf.tr0_rows()
    mx1 = mx0 = 0.0
    for _t, nm, seq, db in tr0:
        a1, a0 = EN.four(EN.lc.counts(seq, db)), EN.four(EN.lc0.counts(seq, db))
        if a1 != a0:
            raise GuardFailure(f"step 2: TR0 reference {nm}'s four-table counts depend on the "
                               f"special hairpins")
        mx1 = max(mx1, float(np.abs(cf.basis_image(B["B4"], pos, a1)).max()))
        mx0 = max(mx0, float(np.abs(cf.basis_image(B["B4"], pos, a0)).max()))
    out["model"] = {"dangles": SPECIAL[0], "special_hp": SPECIAL[1],
                    "every_other_field": "as thermo.vienna_ref.REFERENCE_MODEL"}
    out["step1_type_a"] = {
        "generating_rows": len(rows), "exact_rank": er,
        "null_dim": er["null_dim"],
        "analytic_directions": an,
        "basis_in_words": "N1: +1 on every hairpin mismatch entry, -1 on every hairpin "
                          "initiation entry of size 4 to 30 (size 3 hairpins read no mismatch "
                          "when special hairpins are on); N2: +1 on every entry of the three "
                          "interior-loop mismatch tables, -2 on every interior-loop initiation "
                          "entry",
        "config": {k: res[k] for k in ("dangles", "special_hp", "n_rows") if k in res}}
    out["step2_invisibility"] = {"n_tr0_references": len(tr0),
                                 "four_table_counts_equal_to_registered": True,
                                 "max_abs_B_four_T_c_ref": mx1,
                                 "max_abs_B_four_T_c_ref_registered_configuration": mx0}
    out["seconds_steps_1_2"] = round(time.time() - t0, 1)
    out["provenance_steps_1_2"] = prov
    out["status"] = C3_TYPEA
    write("c3", out)
    print(json.dumps({k: out[k] for k in ("step1_type_a", "step2_invisibility")}, indent=1,
                     default=str)[:3000])
    return 0


def c3_gate() -> int:
    out, prov = gate("c3", C3_TYPEA)
    t0 = time.time()
    EN = dc.use_model(*SPECIAL)
    B = cf.load_tr0_basis()
    rows, _ = cf.tr0_rows()
    g1 = dc.gate_g1(EN, B, rows)
    print("G1 passed", round(time.time() - t0, 1), flush=True)
    g2 = dc.gate_g2(EN, B, rows)
    print("G2 passed", round(time.time() - t0, 1), flush=True)
    out["step3_gate"] = {"G1": g1, "G2": g2, "count_model_checks": EN.n_checked,
                         "seconds": round(time.time() - t0, 1), "provenance": prov,
                         "environment": cf.env_record()}
    out["status"] = C3_GATE
    write("c3", out)
    print(json.dumps({"G2": {k: g2[k] for k in ("n_sequences", "n_structures_total",
                                                "false_certificates", "envelope",
                                                "power_check")}}, indent=1, default=str))
    return 0


def drive(phase: str) -> list[int]:
    shards = OUT["c3"] / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="")
    procs = [subprocess.Popen(
        ["nice", "-n", "10", sys.executable, str(R / ME), "--phase", phase, "--shard", str(w),
         "--of", str(N_WORKERS)], cwd=R, env=env,
        stdout=(shards / f"{phase}_{w}.log").open("w"), stderr=subprocess.STDOUT)
        for w in range(N_WORKERS)]
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
        raise GuardFailure(f"{phase} workers exited {codes}; logs in {shards}")
    return [p.pid for p in procs]


def c3_run_worker() -> int:
    if read_out("c3").get("status") != C3_GATE:
        raise SystemExit("run workers run only under c3-run")
    cr._in_certfold_env()
    w = int(sys.argv[sys.argv.index("--shard") + 1])
    n_w = int(sys.argv[sys.argv.index("--of") + 1])
    EN = dc.use_model(*SPECIAL)
    B = cf.load_tr0_basis()
    with open(OUT["c3"] / "shards" / f"run_{w}.jsonl", "w") as fh:
        for r in r2.ts0_rows_with_stars():
            if r["index"] % n_w == w:
                fh.write(json.dumps(dc.label_one(EN, B, r, zuker=True)) + "\n")
                fh.flush()
    return 0


def c3_run() -> int:
    out, prov = gate("c3", C3_GATE)
    load, t0 = os.getloadavg(), time.time()
    pids = drive("c3-run-worker")
    shards = OUT["c3"] / "shards"
    recs = [json.loads(ln) for w in range(N_WORKERS)
            for ln in (shards / f"run_{w}.jsonl").read_text().splitlines() if ln.strip()]
    if len(recs) != 1288 or len({r["index"] for r in recs}) != 1288:
        raise GuardFailure(f"{len(recs)} run records, not 1,288 distinct")
    out["step4_run"] = {
        "n_sequences": len(recs), "set": "the published envelope at lambda = 1",
        "folds": sum(r["folds"] for r in recs),
        "recursion_side": {k: sum(r["recursion"][k] for r in recs) for k in recs[0]["recursion"]},
        "cpu_seconds": round(sum(r["seconds"] for r in recs), 1), "workers": N_WORKERS,
        "pids": pids, "load_at_start": [round(x, 2) for x in load],
        "wall_seconds": round(time.time() - t0, 1),
        "files_sha256": {f"run_{w}.jsonl": cf._sha(shards / f"run_{w}.jsonl")
                         for w in range(N_WORKERS)},
        "provenance": prov, "environment": cf.env_record()}
    out["status"] = C3_RUN
    write("c3", out)
    return 0


def c3_analyse() -> int:
    out, prov = gate("c3", C3_RUN)
    shards = OUT["c3"] / "shards"
    recs = []
    for w in range(N_WORKERS):
        p = shards / f"run_{w}.jsonl"
        if cf._sha(p) != out["step4_run"]["files_sha256"][p.name]:
            raise GuardFailure(f"{p.name} changed since the run")
        recs += [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    reg1 = cf.read_out()["analysis"]["sets"]["TS0"]["describe"]["labels"]["envelope:1"]
    stars0 = {r["index"]: r["star"] for r in r2.ts0_rows_with_stars()}

    def share(n, of):
        return {"n": n, "of": of, "fraction": n / of if of else None}

    by = collections.defaultdict(list)
    for r in recs:
        by[r["stratum"]].append(r)
    out["step4_analysis"] = {
        "certified_present": {
            "pooled": share(sum(r["n_certified_present"] for r in recs),
                            sum(r["n_pairs_star"] for r in recs)),
            "by_stratum": {str(s): share(sum(r["n_certified_present"] for r in rs),
                                         sum(r["n_pairs_star"] for r in rs))
                           for s, rs in sorted(by.items())}},
        "registered_configuration": {"pooled": reg1["certified_present"]["pooled"],
                                     "by_stratum": reg1["certified_present"]["by_stratum"]},
        "n_mfe_structures_differ_from_registered": sum(r["star"] != stars0[r["index"]]
                                                       for r in recs),
        "n_sequences": len(recs), "reading": "descriptive, post hoc", "provenance": prov}
    out["status"] = C3_DONE
    write("c3", out)
    print(json.dumps({k: v for k, v in out["step4_analysis"].items() if k != "provenance"},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"c1": c1, "c2": c2, "c3-typea": c3_typea, "c3-gate": c3_gate,
              "c3-run": c3_run, "c3-run-worker": c3_run_worker,
              "c3-analyse": c3_analyse}[ph]())
