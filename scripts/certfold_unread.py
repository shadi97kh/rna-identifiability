#!/usr/bin/env python3
"""CERTIFIED FOLDING: THE FOUR TABLES' UNREAD ENTRIES. POST HOC, as specified before any code in
docs/certfold_unread_addition.md (the authors' decision of 2026-10-19; the section numbers below
are that file's).

    E+_1 = {theta0 + B t + v : t in E_1, v_e = 0 off U, |v_e| <= m_e on U}

U is the int11, int21, int22 and bulge columns no TR0 reference reads; m_e is the largest
|published - Turner 2004| on e over Turner 1999, Andronescu 2007 and Langdon 2018, a forbidden
entry counting as a zero difference. The certificate is registration 1's, frozen, with its
per-entry bound u extended to u+ = u + m. That is the ONLY change: every label and every gate
check below is certfold's own function, called with Weights(u+, ...) or dict(B, u_env=u+). G4's
check is inline in certfold.gate_g4, so its envelope branch is mirrored here line for line.

PHASES, each refusing unless the specification is committed unmodified and its predecessor's
record is committed:
  describe   4a: U, m_e, the shares of the published differences E+'s span holds (no folding)
  gates      4b: G1 (both backends), G2 with its power check, G4, on registration 1's sequences
  attacks    4c: G5 on TS0 with vertex moves on U (workers); violation counts and digests only
  run        4d: TS0 labels under E+_1 (workers); the monotonicity must-holds
  analyse    4e: the certified fractions by stratum beside registration 1's
Run in .venv-certfold: .venv-certfold/bin/python scripts/certfold_unread.py --phase <phase>
Records go to runs/2026-10-19_certfold_unread/metrics.json. Guards raise GuardFailure and are
never repaired; a false certificate or a violation stops the addition (the drop rule).
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold_reg2 as r2  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402

cf, cr = r2.cf, r2.cr
ME = "scripts/certfold_unread.py"
SPEC = "docs/certfold_unread_addition.md"
RUN_ID = "2026-10-19_certfold_unread"
OUT = R / "runs" / RUN_ID
SHARDS = OUT / "shards"
TAG_VERTEX = 90                  # the spec's seed tag: the signs of G5's vertex moves on U
WORKERS = 8
FOUR = ("int11", "int21", "int22", "bulge")
INF_DCAL = 100000                # certfold's INF test: th[k] >= 100000
KEY = "envelope+:1"              # the label key of E+_1 in every record
REG_KEY = cr.PRIMARY             # registration 1's envelope at lambda = 1

DESCRIBED = "described"
GATED = "gates passed"
ATTACKED = "G5 passed"
LABELLED = "TS0 labelled"
ANALYSED = "analysed"


# ----------------------------------------------------------------------------------
# records and gates
# ----------------------------------------------------------------------------------
def read_out() -> dict:
    p = OUT / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {"run_id": RUN_ID}


def write(out: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def committed() -> dict:
    rel = str((OUT / "metrics.json").relative_to(R))
    if cf._git("ls-files", "--error-unmatch", rel).returncode != 0:
        raise SystemExit(f"{rel} is not tracked: commit it first")
    if cf._git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"{rel} has uncommitted changes: commit it first")
    return json.loads(cf._git("show", f"HEAD:{rel}").stdout)


def gate(status) -> tuple[dict, dict]:
    """(record, provenance): the specification committed unmodified, the registered environment,
    a clean tree, the status one of `status` (None: no record yet), and the predecessor's record
    committed unmodified."""
    if cf._git("ls-files", "--error-unmatch", SPEC).returncode != 0:
        raise SystemExit(f"{SPEC} is not committed")
    if cf._git("diff", "--quiet", "HEAD", "--", SPEC).returncode != 0:
        raise SystemExit(f"{SPEC} has uncommitted changes")
    cr._in_certfold_env()
    prov = r2.clean()
    out = read_out()
    st = (status,) if isinstance(status, str) or status is None else tuple(status)
    if out.get("status") not in st:
        raise SystemExit(f"status is {out.get('status')!r}; this phase needs one of {st}")
    if status is not None:
        committed()
    out.setdefault("specification", {
        "path": SPEC, "commit": cf._git("log", "-1", "--format=%H", "--", SPEC).stdout.strip(),
        "sha256": cf._sha(R / SPEC)})
    return out, prov


def drive(phase: str, n_w: int) -> list[int]:
    """Run `phase` in n_w worker processes (nice 10, single-threaded); stop all on a failure."""
    import subprocess

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


def jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()
                          ).hexdigest()


def _vec_sha(v: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(v, dtype=np.float64).tobytes()).hexdigest()


# ----------------------------------------------------------------------------------
# 1: U, m_e and u+
# ----------------------------------------------------------------------------------
def unread_bound(B: dict) -> dict:
    """U, m and u+ = u_env + m over B['cols'], with each published difference (INF-zeroed).

    The guards are the spec's: no column of U is INF in Turner 2004, and B_four, a_four and
    u_env are exactly zero on U, so the supports of B t and v are disjoint (the soundness
    argument needs that, and it is checked, not assumed)."""
    from scripts.typeb_common import full_tables

    cols, used = B["cols"], np.asarray(B["used"])
    t0 = cf.template()["theta"]
    fam = np.array([k[0] for k in cols])
    unread = np.isin(fam, FOUR)
    unread[used] = False
    if any(t0[cols[e]] >= INF_DCAL for e in np.nonzero(unread)[0]):
        raise GuardFailure("a column of U is INF in Turner 2004 itself")
    if np.abs(B["u_env"][unread]).max() != 0.0 or np.abs(B["a_four"][unread]).max() != 0.0:
        raise GuardFailure("the envelope's bound or B_four is nonzero on an unread column")
    m = np.zeros(len(cols))
    diffs = {}
    for tb in full_tables():
        X = tb["id"].split(":")[-1]
        th = tb["theta"]
        d = np.array([th[k] - t0[k] for k in cols], dtype=np.float64) / 100.0
        inf = np.array([th[k] >= INF_DCAL for k in cols])
        d = np.where(inf, 0.0, d)
        diffs[X] = {"d": d, "inf": inf}
        m = np.maximum(m, np.where(unread, np.abs(d), 0.0))
    u_plus = np.asarray(B["u_env"], dtype=np.float64) + m
    return {"unread": unread, "m": m, "u_plus": u_plus, "diffs": diffs, "fam": fam}


def checked_u_plus(B: dict, out: dict) -> np.ndarray:
    """u+, recomputed and required to be the vector the describe phase recorded."""
    u = unread_bound(B)["u_plus"]
    if _vec_sha(u) != out["describe"]["u_plus_sha256"]:
        raise GuardFailure("u+ differs from the one the describe phase recorded")
    return u


# ----------------------------------------------------------------------------------
# phase: describe (4a), before any folding
# ----------------------------------------------------------------------------------
def describe() -> int:
    out, prov = gate(None)
    t0 = time.time()
    B = cf.load_tr0_basis()
    ub = unread_bound(B)
    unread, m, fam = ub["unread"], ub["m"], ub["fam"]
    cols, used, B4 = B["cols"], np.asarray(B["used"]), B["B4"]
    reach = B["u_env"] > 0
    env = cf.read_out()["build_basis"]["published_envelope"]
    rev = json.loads((R / "runs" / "2026-10-18_certfold_reviewer" / "metrics.json").read_text())
    cov = rev["descriptive"]["table_coverage"]
    mU = m[unread]
    k_max = int(np.nonzero(unread)[0][int(np.argmax(mU))])
    attains = [X for X, v in ub["diffs"].items() if abs(abs(v["d"][k_max]) - m[k_max]) < 1e-12]
    shares, contained = {}, {}
    for X, v in ub["diffs"].items():
        d = v["d"]
        inside = np.zeros(len(cols))
        inside[used] = B4 @ (B4.T @ d[used])
        dU = np.where(unread, d, 0.0)
        n2 = float(d @ d)
        s_in, s_u = float(inside @ inside) / n2, float(dU @ dU) / n2
        # the cross-check against the reviewer's committed coverage table: the set's part plus
        # the four tables' unread parts, from its per-family norms
        c = cov[X]
        nr = c["parts"]["outside: columns no TR0 reference reads"]["by_family_norm_kcal"]
        want_u = sum(nr.get(f, 0.0) ** 2 for f in FOUR) / c["norm_delta_kcal"] ** 2
        want_in = c["parts"]["inside the set"]["share_of_squared_norm"]
        if abs(s_in - want_in) > 1e-12 or abs(s_u - want_u) > 1e-12:
            raise GuardFailure(f"{X}: the shares disagree with the reviewer's coverage table "
                               f"({s_in}, {s_u}) vs ({want_in}, {want_u})")
        shares[X] = {"norm_delta_kcal": float(np.sqrt(n2)),
                     "share_inside_registered_span": s_in,
                     "share_on_unread_four_table_entries": s_u,
                     "share_inside_E_plus_span": s_in + s_u,
                     "n_unread_entries_forbidden": int((v["inf"] & unread).sum()),
                     "n_unread_entries_differing": int(((d != 0) & unread).sum())}
        contained[X] = bool(np.abs(inside[reach]).max() <= env["delta_kcal"] + 1e-12
                            and np.linalg.norm(inside) <= env["r_max_kcal"] + 1e-12
                            and np.all(np.abs(dU[unread]) <= m[unread] + 1e-12))
        if not contained[X]:
            raise GuardFailure(f"E+_1 does not contain {X}'s projection onto its span")
    out["describe"] = {
        "definition": "E+_1 = {theta0 + B t + v : t in E_1, v_e = 0 off U, |v_e| <= m_e on U}; "
                      "u+ = u_env + m (disjoint supports)",
        "n_unread_by_table": {f: int((unread & (fam == f)).sum()) for f in FOUR},
        "n_unread": int(unread.sum()),
        "m_e_kcal": {"n_nonzero": int((mU > 0).sum()), "median_over_U": float(np.median(mU)),
                     "median_over_nonzero": float(np.median(mU[mU > 0])),
                     "max": float(mU.max()), "argmax_entry": repr(cols[k_max]),
                     "attained_by": attains},
        "n_reached_entries_registered": int(reach.sum()),
        "shares_of_squared_difference": shares,
        "projection_contained_in_E_plus": contained,
        "cross_check": "share_inside_registered_span and share_on_unread_four_table_entries equal "
                       "the reviewer's /descriptive/table_coverage to 1e-12",
        "u_plus_sha256": _vec_sha(ub["u_plus"]),
        "seconds": round(time.time() - t0, 1), "provenance": prov,
        "environment": cf.env_record()}
    out["status"] = DESCRIBED
    write(out)
    print(json.dumps({k: v for k, v in out["describe"].items()
                      if k not in ("provenance", "environment")}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------------
# phase: gates (4b) -- G1, G2, G4 with u+ on registration 1's own gate sequences
# ----------------------------------------------------------------------------------
def g1_plus(lc, B: dict, u_plus: np.ndarray, m: np.ndarray) -> dict:
    import RNA

    from scripts.turner_nullspace import generating_set
    from thermo.vienna_ref import reference_md

    P = cf.template()
    rows, _ = cf.tr0_rows()
    W = cf.Weights(u_plus, B["cols"], 1.0)
    Wm = cf.Weights(m, B["cols"], 1.0)        # U alone: how many loops the extension charges
    seqs = cf.gate_sequences(rows, cf.G1_SEQUENCES, 50, 500, cf.SEED)
    sampled = {b: {"n_structures": 0, "n_with_a_nonzero_offset": 0} for b in ("compiled",
                                                                               "python")}
    n_u_sampled = 0
    for n, (_t, nm, seq, _db) in enumerate(seqs):
        fc = RNA.fold_compound(seq, reference_md())
        s_star, _e = fc.mfe()
        structs = cf.sample_structures(seq, cf.G1_SAMPLES, cf.SEED + n) + [s_star]
        star = cf.star_interior_loops(lc, seq, s_star)
        n_u_sampled += sum(cf.eprime_offset(lc, seq, s, star, Wm) != 0 for s in structs)
        for b in sampled:
            res = cf.g1_identity(lc, W, seq, structs, s_star, backend=b)
            if not res["pass"]:
                raise GuardFailure(f"G1+ ({b}): the identity failed on {nm} "
                                   f"({res['n_mismatches']} of {res['n']}): "
                                   f"{res['mismatches'][:2]}")
            sampled[b]["n_structures"] += res["n"]
            sampled[b]["n_with_a_nonzero_offset"] += res["n_with_a_nonzero_offset"]
    gen = generating_set(P["special"])
    constructed = {b: {"n_rows": 0, "n_with_a_nonzero_offset": {"plus": 0, "minus": 0}}
                   for b in ("compiled", "python")}
    n_u_rows = 0
    for _tag, seq, db in gen:
        n_u_rows += cf.eprime_offset(lc, seq, db, set(), Wm) != 0
        for b in constructed:
            for sign, star_db in (("plus", db), ("minus", "." * len(seq))):
                res = cf.g1_identity(lc, W, seq, [db], star_db, backend=b)
                if not res["pass"]:
                    raise GuardFailure(f"G1+ constructed ({b}, {sign}) failed on {seq} {db}: "
                                       f"{res['mismatches'][:1]}")
                constructed[b]["n_with_a_nonzero_offset"][sign] += res["n_with_a_nonzero_offset"]
            constructed[b]["n_rows"] += 1
    return {"sampled": {"n_sequences": len(seqs), "by_backend": sampled,
                        "n_structures_charged_on_U": int(n_u_sampled)},
            "constructed": {"by_backend": constructed,
                            "n_rows_reading_an_unread_entry": int(n_u_rows)},
            "samples_per_sequence": cf.G1_SAMPLES, "pass": True}


def g2_plus(lc, B: dict, u_plus: np.ndarray) -> dict:
    rows, _ = cf.tr0_rows()
    W1 = cf.Weights(B["a_four"], B["cols"], 1.0)
    pos = {B["cols"][e]: k for k, e in enumerate(B["used"])}
    seqs, screen = cf.g2_sequences(rows, cf.G2_SEQUENCES, 16, cf.G2_MAX_N, cf.SEED, lc, W1)
    Bp = dict(B, u_env=u_plus)
    tot = collections.Counter()
    rec = collections.Counter()
    for _src, seq, _n in seqs:
        r = cf.g2_one(lc, Bp, pos, W1, seq, (1.0,), with_rho=False)
        bad = r["envelope"]["n_false_certificates"] + r["by_radius"]["1"]["n_false_certificates"]
        if bad:
            raise GuardFailure(f"G2+: {bad} false certificates on {seq}: "
                               f"{r['envelope']['false_certificates'][:2]}")
        for k in ("n_certified_present", "n_certified_absent", "n_exactly_robust_present",
                  "n_exactly_robust_absent", "n_false_certificates", "whole_structure_certified"):
            tot[k] += int(r["envelope"][k])
        tot["n_structures"] += r["n_structures"]
        rec.update(r["recursion"])
    planted = 0
    for _s, seq, _n in seqs[:cf.G2_POWER_SEQUENCES]:
        planted += cf.g2_one(lc, Bp, pos, W1, seq, (1.0,), scale=0.5, with_rho=False
                             )["envelope"]["n_false_certificates"]
    if planted == 0:
        raise GuardFailure("G2+ power check: halving the weights produced no false certificate")
    return {"n_sequences": len(seqs), "sequences": screen, **dict(tot),
            "test": "the box bound N(s) - sum_e |dc_e| u+_e: certified must imply robust over "
                    "the box {|x_e| <= u+_e}, which contains E+_1",
            "power_check": {"weights_scaled_by": 0.5, "n_sequences": cf.G2_POWER_SEQUENCES,
                            "false_certificates_produced": planted, "pass": True},
            "recursion_side": dict(rec), "pass": True}


def g4_plus(lc, B: dict, u_plus: np.ndarray, unread: np.ndarray) -> dict:
    """certfold.gate_g4's envelope branch, line for line, with u+ in place of u_env."""
    import RNA

    from thermo.vienna_ref import assert_turner2004_params, reference_md

    rows, _ = cf.tr0_rows()
    seqs = cf.gate_sequences(rows, cf.G4_SEQUENCES, 30, cf.G4_MAX_N, cf.SEED + 4)
    U = {B["cols"][e] for e in np.nonzero(unread)[0]}
    per, viol = [], []
    for _t, nm, seq, _db in seqs:
        plain = RNA.fold_compound(seq, reference_md())
        assert_turner2004_params(plain)
        star, _e = plain.mfe()
        pt = cf.pair_table(star)
        sp = [(i, pt[i]) for i in range(1, len(seq) + 1) if pt[i] > i]
        adm = cf.admissible_pairs(seq)
        Fe = cf.Folder(lc, seq, star, cf.Weights(u_plus, B["cols"], 1.0))
        c_star = lc.counts(seq, star)
        rec = {"name": nm, "n_nt": len(seq), "n_pairs_star": len(sp)}
        r = 1.0
        Fe.set_radius(r)
        _s_min, e_min, _e0 = Fe.free()
        delta = Fe.eprime_star - e_min
        if delta < 0:
            raise GuardFailure(f"E'(s*) is below the constrained minimum on {nm}")
        if delta > cf.G4_DELTA_CAP:
            rec.update({"skipped": True, "band_delta_dcal": int(delta),
                        "cap_dcal": cf.G4_DELTA_CAP})
            per.append(rec)
            continue
        band = []
        for z in Fe.fc.subopt(int(delta)):
            Fe._check(z.structure, int(round(z.energy * 100)), "wuchty")
            Fe.recursion["subopt_structures_checked"] += 1
            band.append(z.structure)
        if not band:
            raise GuardFailure(f"an empty band on {nm} with delta = {delta}")
        best = min(Fe.eprime(s)[0] for s in band)
        if best != e_min:
            raise GuardFailure(f"subopt under E' does not reach the constrained minimum on "
                               f"{nm}: {best} != {e_min}")
        c = cf.certify(Fe, sp, adm)
        cert_p = {p for p, mg in c["margins_present_dcal"].items() if mg > 0}
        cert_a = {q for q, mg in c["margins_absent_dcal"].items() if mg is None or mg > 0}
        n_witness, n_touch_u = 0, 0
        lack, hold = set(), set()
        for s in band:
            if s == star:
                continue
            ep, e0 = Fe.eprime(s)
            if ep > Fe.eprime_star:
                continue
            n_s = (e0 - Fe.e0_star) / 100.0
            cs = lc.counts(seq, s)
            n_touch_u += any(cs.get(k, 0) != c_star.get(k, 0) for k in (set(cs) | set(c_star))
                             if k in U)
            box = sum(abs(cs.get(k, 0) - c_star.get(k, 0)) * float(u_plus[Fe.W.idx[k]])
                      for k in set(cs) | set(c_star) if k in Fe.W.idx)
            margin = n_s - r * box
            if margin > 0:
                continue
            n_witness += 1
            here = set()
            p_t = cf.pair_table(s)
            for i in range(1, len(seq) + 1):
                if p_t[i] > i:
                    here.add((i, p_t[i]))
            lack |= set(sp) - here
            hold |= here - set(sp)
            for p in cert_p - here:
                viol.append({"name": nm, "kind": "present", "pair": p, "margin_kcal": margin,
                             "structure": s[:60]})
            for q in cert_a & here:
                viol.append({"name": nm, "kind": "absent", "pair": q, "margin_kcal": margin,
                             "structure": s[:60]})
        robust_p = set(sp) - lack
        robust_a = set(c["margins_absent_dcal"]) - hold
        rec.update({
            "skipped": False, "band_delta_dcal": int(delta), "band_size": len(band),
            "n_band_structures_differing_on_U": int(n_touch_u),
            "n_exact_witnesses": n_witness,
            "n_certified_present": len(cert_p), "n_certified_absent": len(cert_a),
            "n_robust_present": len(robust_p), "n_robust_absent": len(robust_a),
            "relaxation_loss_present": len(robust_p - cert_p),
            "relaxation_loss_absent": len(robust_a - cert_a),
            "folds": Fe.folds, "recursion": Fe.recursion})
        per.append(rec)
        if viol:
            raise GuardFailure(f"G4+: {len(viol)} false certificates: {viol[:2]}")
    ran = [p for p in per if not p["skipped"]]
    keys = ("band_size", "n_band_structures_differing_on_U", "n_exact_witnesses",
            "n_certified_present", "n_certified_absent", "n_robust_present", "n_robust_absent",
            "relaxation_loss_present", "relaxation_loss_absent", "n_pairs_star")
    return {"n_sequences": len(per), "n_skipped": len(per) - len(ran),
            "skipped_band_delta_dcal": [p["band_delta_dcal"] for p in per if p["skipped"]],
            "cap_dcal": cf.G4_DELTA_CAP, "lengths": [min(p["n_nt"] for p in per),
                                                     max(p["n_nt"] for p in per)],
            **{k: sum(p[k] for p in ran) for k in keys},
            "n_false_certificates": 0, "pass": True,
            "note": "robustness is judged by the box bound, an upper bound on E+_1's worst "
                    "case, so the robust counts and the relaxation loss are lower bounds"}


def gates() -> int:
    out, prov = gate(DESCRIBED)
    t0 = time.time()
    B = cf.load_tr0_basis()
    u_plus = checked_u_plus(B, out)
    ub = unread_bound(B)
    P = cf.template()
    lc = cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    g = {}
    ts = time.time()
    g["G1"] = g1_plus(lc, B, u_plus, ub["m"])
    g["G1"]["seconds"] = round(time.time() - ts, 1)
    print("G1+", json.dumps(g["G1"]), flush=True)
    ts = time.time()
    g["G2"] = g2_plus(lc, B, u_plus)
    g["G2"]["seconds"] = round(time.time() - ts, 1)
    print("G2+", json.dumps({k: v for k, v in g["G2"].items() if k != "sequences"}), flush=True)
    ts = time.time()
    g["G4"] = g4_plus(lc, B, u_plus, ub["unread"])
    g["G4"]["seconds"] = round(time.time() - ts, 1)
    print("G4+", json.dumps(g["G4"]), flush=True)
    out["gates"] = {**g, "pass": True, "seconds": round(time.time() - t0, 1),
                    "load_at_start": os.getloadavg(), "provenance": prov,
                    "environment": cf.env_record()}
    out["status"] = GATED
    write(out)
    return 0


# ----------------------------------------------------------------------------------
# phase: attacks (4c) -- G5 on TS0, with vertex moves on U
# ----------------------------------------------------------------------------------
def ts0_rows() -> list[dict]:
    rows = r2.ts0_rows_with_stars()
    if [r["index"] for r in rows] != sorted(r["index"] for r in rows):
        raise GuardFailure("TS0 rows out of order")
    return rows


def mine_of(rows: list[dict], w: int, n_w: int) -> list[dict]:
    assign = cr.shard_of(rows, n_w)
    return sorted((r for r in rows if assign[r["index"]] == w),
                  key=lambda r: (-len(r["seq"]), r["index"]))


def labels_digest(cert_p, cert_a) -> str:
    return _digest({"present": sorted(list(p) for p in cert_p),
                    "absent": sorted(list(q) for q in cert_a)})


def attacks_worker() -> int:
    w, n_w = r2.shard_args()
    out = committed()
    if out.get("status") != GATED:
        raise SystemExit("attack workers run only under the attacks phase, after the gates")
    if cf.callback_backend() != "compiled":
        raise SystemExit("G5+ workers run with the compiled callback")
    from scripts.certfold_baselines import envelope_draws, pair_frequencies

    B = cf.load_tr0_basis()
    u_plus = checked_u_plus(B, out)
    ub = unread_bound(B)
    m, unread = ub["m"], ub["unread"]
    P = cf.template()
    lc = cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    cols, used, B4 = B["cols"], B["used"], B["B4"]
    idx = {k: n for n, k in enumerate(cols)}
    pos = {cols[e]: n for n, e in enumerate(used)}
    u = B["u_env"]
    delta = float(B["envelope"]["delta_kcal"])
    r_max = float(B["envelope"]["r_max_kcal"])
    rows = mine_of(ts0_rows(), w, n_w)
    # the attacks: the three published projections onto E+'s span, and registration 1's G5
    # envelope draws (tag 50) each plus a vertex of V (signs from tag 90, one vector per draw)
    comps = [(X, c + np.where(unread, ub["diffs"][X]["d"], 0.0))
             for X, c in cf.published_component_shifts(B)]
    draws, drec = envelope_draws(B, cf.G5_DRAWS, seed_tag=cf.G5_SEED_TAG, lam=1.0)
    rng = np.random.default_rng(np.random.SeedSequence([cf.SEED, TAG_VERTEX]))
    signs = rng.integers(0, 2, size=(cf.G5_DRAWS, int(unread.sum()))) * 2 - 1
    ui = np.nonzero(unread)[0]
    vert = []
    for k, d in enumerate(draws):
        v = np.array(d, dtype=np.float64)
        v[ui] += signs[k] * m[ui]
        vert.append(v)
    shifts = [c for _n, c in comps] + vert
    kinds = [f"component:{n}" for n, _c in comps] + ["draw"] * len(vert)

    def pairs_of(s):
        pt = cf.pair_table(s)
        return {(i, pt[i]) for i in range(1, len(s) + 1) if pt[i] > i}

    def margin(k, s, shift):
        """E_theta(s) - E_theta(s*) in dcal, exact float, theta = theta0 + shift (unrounded)."""
        r = per[k]
        cs = lc.counts(r["seq"], s)
        dc = {kk: cs.get(kk, 0) - r["c_star"].get(kk, 0)
              for kk in set(cs) | set(r["c_star"]) if kk in idx}
        return (cf._turner_energy(lc, r["seq"], s) - r["e0_star"]
                + 100.0 * sum(v * shift[idx[kk]] for kk, v in dc.items() if v))

    import RNA

    from thermo.vienna_ref import assert_turner2004_params, reference_md

    per = []
    for row in rows:
        seq = row["seq"]
        plain = RNA.fold_compound(seq, reference_md())
        assert_turner2004_params(plain)
        star = plain.mfe()[0]
        if star != row["star"]:
            raise GuardFailure(f"{row['name']}: s* differs from registration 1's")
        pt = cf.pair_table(star)
        sp = [(i, pt[i]) for i in range(1, len(seq) + 1) if pt[i] > i]
        F = cf.Folder(lc, seq, star, cf.Weights(u_plus, cols, 1.0))
        c = cf.certify(F, sp, cf.admissible_pairs(seq))
        cert_p = {p for p, mg in c["margins_present_dcal"].items() if mg > 0}
        cert_a = {q for q, mg in c["margins_absent_dcal"].items() if mg is None or mg > 0}
        e0_star = cf._turner_energy(lc, seq, star)
        if e0_star != F.e0_star:
            raise GuardFailure(f"{row['name']}: the count model and eval_structure disagree")
        rec = {"index": row["index"], "name": row["name"], "seq": seq, "star": star,
               "cert_p": cert_p, "cert_a": cert_a, "c_star": lc.counts(seq, star),
               "e0_star": e0_star, "viol": collections.Counter(),
               "n_tested": collections.Counter(), "digest": labels_digest(cert_p, cert_a)}
        per.append(rec)
        # registration 1's worst case for each of the 20 lowest-E' alternatives, plus the worst
        # vertex of V on the alternative's differing unread entries: the exact minimiser over V
        alts = sorted((e, s) for s, e in F.zuker() if s != star)[:cf.G5_TOP_ALTERNATIVES]
        for _e, s in alts:
            cs = lc.counts(seq, s)
            dc_all = [(kk, cs.get(kk, 0) - rec["c_star"].get(kk, 0))
                      for kk in set(cs) | set(rec["c_star"]) if kk in idx]
            dc = [(kk, v) for kk, v in dc_all if kk in pos]
            supp = [(kk, v) for kk, v in dc if v and u[idx[kk]] > 0]
            shift = np.zeros(len(cols))
            if supp:
                M = np.array([B4[pos[kk]] for kk, _v in supp])
                target = np.array([-u[idx[kk]] * np.sign(v) for kk, v in supp])
                t, *_ = np.linalg.lstsq(M, target, rcond=None)
                shift[used] = B4 @ t
                nt, mx = float(np.linalg.norm(t)), float(np.abs(shift).max())
                scale = min(1.0, r_max / nt if nt > 0 else 1.0, delta / mx if mx > 0 else 1.0)
                shift *= scale
            for kk, v in dc_all:
                if v and unread[idx[kk]]:
                    shift[idx[kk]] = -np.sign(v) * m[idx[kk]]
            rec["n_tested"]["worst_case"] += 1
            here = pairs_of(s)
            touching = bool((cert_p - here) or (cert_a & here))
            rec["n_tested"]["worst_case_touching"] += touching
            if touching and margin(len(per) - 1, s, shift) <= 0:
                rec["viol"]["worst_case"] += 1

    def on_fold(d, k, s):
        r = per[k]
        r["n_tested"][kinds[d]] += 1
        here = pairs_of(s)
        if (r["cert_p"] - here) or (r["cert_a"] & here):
            if margin(k, s, shifts[d]) <= 0:
                r["viol"][kinds[d]] += 1

    scratch = SHARDS / f"g5_tables_{w}"
    scratch.mkdir(parents=True, exist_ok=True)
    _freq, frec = pair_frequencies([r["seq"] for r in per], shifts, P["theta"], scratch,
                                   cols=cols, on_fold=on_fold)
    with open(SHARDS / f"g5_{w}.jsonl", "w") as fh:
        for r in per:
            fh.write(json.dumps({"index": r["index"], "name": r["name"],
                                 "violations": dict(r["viol"]), "n_tested": dict(r["n_tested"]),
                                 "labels_sha256": r["digest"]}) + "\n")
    (SHARDS / f"g5_{w}.meta.json").write_text(json.dumps({
        "envelope_draws": {k: v for k, v in drec.items() if k != "draws"},
        "tables": {k: v for k, v in frec.items() if k != "draws"},
        "n_tables": len(shifts)}, default=str))
    return 0


def attacks() -> int:
    out, prov = gate(GATED)
    t0, load = time.time(), os.getloadavg()
    pids = drive("attacks-worker", WORKERS)
    per = []
    for w in range(WORKERS):
        per += jsonl(SHARDS / f"g5_{w}.jsonl")
    if sorted(d["index"] for d in per) != [r["index"] for r in ts0_rows()]:
        raise GuardFailure("G5+ did not attack TS0 exactly once")
    viol, tested = collections.Counter(), collections.Counter()
    for d in per:
        viol.update(d["violations"])
        tested.update(d["n_tested"])
    total = sum(viol.values())
    meta = json.loads((SHARDS / "g5_0.meta.json").read_text())
    out["attacks"] = {
        "rule": "violation counts only; no accuracy, coverage or PPV",
        "n_sequences": len(per), "violations": dict(viol), "n_tested": dict(tested),
        "n_violations": total,
        "attacks": {"components": "Pi delta_X + delta_X|U, three", "draws": cf.G5_DRAWS,
                    "draw_seed": [cf.SEED, cf.G5_SEED_TAG], "vertex_seed": [cf.SEED, TAG_VERTEX],
                    "worst_case_alternatives": cf.G5_TOP_ALTERNATIVES},
        "envelope_draws": meta["envelope_draws"],
        "files_sha256": {f"g5_{w}.jsonl": cf._sha(SHARDS / f"g5_{w}.jsonl")
                         for w in range(WORKERS)},
        "workers": WORKERS, "pids": pids, "load_at_start": load,
        "pass": total == 0, "seconds": round(time.time() - t0, 1), "provenance": prov,
        "environment": cf.env_record()}
    out["status"] = ATTACKED if total == 0 else "G5 FAILED: violations found"
    write(out)
    print(json.dumps({k: v for k, v in out["attacks"].items()
                      if k not in ("provenance", "files_sha256", "environment")}, indent=1,
                     default=str))
    return 0 if total == 0 else 1


# ----------------------------------------------------------------------------------
# phase: run (4d) -- TS0 labels under E+_1
# ----------------------------------------------------------------------------------
def run_worker() -> int:
    w, n_w = r2.shard_args()
    out = committed()
    if out.get("status") != ATTACKED:
        raise SystemExit("run workers run only after G5+ passed and is committed")
    if cf.callback_backend() != "compiled":
        raise SystemExit("the run folds with the compiled callback only")
    import RNA

    from thermo.vienna_ref import (
        assert_turner2004_boltzmann,
        assert_turner2004_params,
        reference_md,
    )

    B = cf.load_tr0_basis()
    u_plus = checked_u_plus(B, out)
    P = cf.template()
    lc = cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    rows = mine_of(ts0_rows(), w, n_w)
    seq_recs, _ = cr.read_shards(cf.read_out()["run"]["workers"])
    reg = {s["index"]: s for s in seq_recs}
    digests = {}
    for k in range(WORKERS):
        for d in jsonl(SHARDS / f"g5_{k}.jsonl"):
            digests[d["index"]] = d["labels_sha256"]
    path = SHARDS / f"run_{w}.jsonl"
    with open(path, "w") as fh:
        for row in rows:
            ts = time.time()
            seq, idx = row["seq"], row["index"]
            plain = RNA.fold_compound(seq, reference_md())
            assert_turner2004_params(plain)
            star, e_mfe = plain.mfe()
            plain.exp_params_rescale(e_mfe)
            plain.pf()
            assert_turner2004_boltzmann(plain)
            bpp = np.array(plain.bpp(), dtype=np.float64)
            if star != row["star"]:
                raise GuardFailure(f"{row['name']}: s* differs from registration 1's")
            pt = cf.pair_table(star)
            sp = [(i, pt[i]) for i in range(1, len(seq) + 1) if pt[i] > i]
            adm = cf.admissible_pairs(seq)
            sps = set(sp)
            outside = [q for q in adm if q not in sps]
            contested = {q for q in outside if bpp[q[0], q[1]] >= cf.CONTESTED_P}
            F = cf.Folder(lc, seq, star, cf.Weights(u_plus, B["cols"], 1.0))
            c = cf.certify(F, sp, adm)
            cert_p = {p for p, mg in c["margins_present_dcal"].items() if mg > 0}
            cert_a = {q for q, mg in c["margins_absent_dcal"].items() if mg is None or mg > 0}
            if labels_digest(cert_p, cert_a) != digests[idx]:
                raise GuardFailure(f"{row['name']}: the labels are not the ones G5+ attacked")
            # the must-holds: u+ >= u, so no margin rises above registration 1's, and no
            # certified set grows
            r1 = reg[idx]
            reg_m = {tuple(q["pair"]): q["margins_dcal"][REG_KEY] for q in r1["pairs"]}
            if set(reg_m) != set(sp):
                raise GuardFailure(f"{row['name']}: the MFE pairs differ from registration 1's")
            above = [(p, mg, reg_m[p]) for p, mg in c["margins_present_dcal"].items()
                     if mg > reg_m[p]]
            if above:
                raise GuardFailure(f"{row['name']}: a margin under E+ exceeds registration 1's "
                                   f"under E: {above[:2]}")
            n_ac = len(cert_a & contested)
            s1 = r1["sets"][REG_KEY]
            if (len(cert_a) > s1["n_certified_absent_admissible"]
                    or n_ac > s1["n_certified_absent_contested"]
                    or r1["n_contested"] != len(contested)
                    or r1["n_admissible_outside_star"] != len(outside)):
                raise GuardFailure(f"{row['name']}: the absent counts or their denominators "
                                   f"break the must-holds")
            fh.write(json.dumps({
                "index": idx, "name": row["name"], "n_nt": len(seq), "stratum": row["stratum"],
                "n_pairs_star": len(sp), "n_admissible_outside_star": len(outside),
                "n_contested": len(contested),
                "sets": {KEY: {"n_certified_present": len(cert_p),
                               "n_certified_absent_admissible": len(cert_a),
                               "n_certified_absent_contested": n_ac,
                               "whole_structure_certified":
                                   len(cert_p) == len(sp) and len(cert_a) == len(outside)}},
                "margins_present_dcal": sorted([i, j, int(mg)] for (i, j), mg
                                               in c["margins_present_dcal"].items()),
                "n_lost_vs_registered": sum(1 for p, mg in c["margins_present_dcal"].items()
                                            if reg_m[p] > 0 >= mg),
                "folds": F.folds, "recursion": F.recursion,
                "seconds": round(time.time() - ts, 2)}) + "\n")
            fh.flush()
    return 0


def run() -> int:
    out, prov = gate(ATTACKED)
    t0, load = time.time(), os.getloadavg()
    for f, h in out["attacks"]["files_sha256"].items():
        if cf._sha(SHARDS / f) != h:
            raise GuardFailure(f"{f} changed since G5+")
    pids = drive("run-worker", WORKERS)
    recs = []
    for w in range(WORKERS):
        recs += jsonl(SHARDS / f"run_{w}.jsonl")
    if sorted(d["index"] for d in recs) != [r["index"] for r in ts0_rows()]:
        raise GuardFailure("the run did not label TS0 exactly once")
    out["run"] = {"n_sequences": len(recs), "folds": sum(d["folds"] for d in recs),
                  "recursion_side": {k: sum(d["recursion"][k] for d in recs)
                                     for k in ("folds_checked", "zuker_structures_checked",
                                               "violations")},
                  "must_holds": "every margin <= registration 1's; certified-absent counts <= "
                                "registration 1's; denominators equal; labels == G5+'s digests",
                  "files_sha256": {f"run_{w}.jsonl": cf._sha(SHARDS / f"run_{w}.jsonl")
                                   for w in range(WORKERS)},
                  "workers": WORKERS, "pids": pids, "load_at_start": load,
                  "cpu_seconds": round(sum(d["seconds"] for d in recs), 1),
                  "seconds": round(time.time() - t0, 1), "provenance": prov,
                  "environment": cf.env_record()}
    out["status"] = LABELLED
    write(out)
    print(json.dumps({k: v for k, v in out["run"].items()
                      if k not in ("provenance", "files_sha256", "environment")}, default=str))
    return 0


# ----------------------------------------------------------------------------------
# phase: analyse (4e)
# ----------------------------------------------------------------------------------
def label_fractions(seq_recs: list[dict], k: str) -> dict:
    """certfold_run.describe's label part, definition for definition, for one set key."""
    strata = sorted({r["stratum"] for r in seq_recs})

    def frac(sel, num, den):
        a = sum(num(r) for r in sel)
        b = sum(den(r) for r in sel)
        return {"n": a, "of": b, "fraction": a / b if b else None}

    parts = {
        "certified_present": (lambda r: r["sets"][k]["n_certified_present"],
                              lambda r: r["n_pairs_star"]),
        "certified_absent_contested": (lambda r: r["sets"][k]["n_certified_absent_contested"],
                                       lambda r: r["n_contested"]),
        "certified_absent_admissible": (
            lambda r: r["sets"][k]["n_certified_absent_admissible"],
            lambda r: r["n_admissible_outside_star"])}
    res = {name: {"pooled": frac(seq_recs, *fn),
                  "by_stratum": {str(s): frac([r for r in seq_recs if r["stratum"] == s], *fn)
                                 for s in strata}}
           for name, fn in parts.items()}
    res["whole_structure_certified"] = {
        "pooled": {"n": sum(bool(r["sets"][k]["whole_structure_certified"]) for r in seq_recs),
                   "of": len(seq_recs)}}
    return res


def analyse() -> int:
    out, prov = gate(LABELLED)
    t0 = time.time()
    for f, h in out["run"]["files_sha256"].items():
        if cf._sha(SHARDS / f) != h:
            raise GuardFailure(f"{f} changed since the run")
    recs = []
    for w in range(WORKERS):
        recs += jsonl(SHARDS / f"run_{w}.jsonl")
    seq_recs, _ = cr.read_shards(cf.read_out()["run"]["workers"])
    reg_ts0 = [s for s in seq_recs if s["set"] == "TS0"]
    # the function must first reproduce registration 1's own TS0 numbers from its records
    reg = label_fractions(reg_ts0, REG_KEY)
    want = cf.read_out()["analysis"]["sets"]["TS0"]["describe"]["labels"][REG_KEY]
    for name in ("certified_present", "certified_absent_contested",
                 "certified_absent_admissible"):
        if reg[name] != want[name]:
            raise GuardFailure(f"label_fractions does not reproduce registration 1's {name}")
    if reg["whole_structure_certified"]["pooled"] != want["whole_structure_certified"]["pooled"]:
        raise GuardFailure("label_fractions does not reproduce registration 1's whole structures")
    got = label_fractions(recs, KEY)
    prof = {str(s): {"n_sequences": sum(r["stratum"] == s for r in recs),
                     "lengths": [min(r["n_nt"] for r in recs if r["stratum"] == s),
                                 max(r["n_nt"] for r in recs if r["stratum"] == s)]}
            for s in sorted({r["stratum"] for r in recs})}
    side = {}
    for name in ("certified_present", "certified_absent_contested",
                 "certified_absent_admissible"):
        side[name] = {
            "pooled": {"E_plus": got[name]["pooled"], "registered_E": reg[name]["pooled"],
                       "difference_points": 100.0 * (got[name]["pooled"]["fraction"]
                                                     - reg[name]["pooled"]["fraction"])},
            "by_stratum": {s: {"E_plus": got[name]["by_stratum"][s],
                               "registered_E": reg[name]["by_stratum"][s]}
                           for s in got[name]["by_stratum"]}}
    out["analysis"] = {
        "reading": "descriptive (the spec, section 5): X beside 40.9%; no adjective, no claim",
        "labels": side,
        "whole_structure_certified": {"E_plus": got["whole_structure_certified"]["pooled"],
                                      "registered_E": reg["whole_structure_certified"]["pooled"]},
        "length_profile": prof,
        "n_mfe_pairs_certified_under_E_not_under_E_plus": sum(d["n_lost_vs_registered"]
                                                              for d in recs),
        "reproduces_registration_1": True,
        "seconds": round(time.time() - t0, 1), "provenance": prov}
    out["status"] = ANALYSED
    write(out)
    print(json.dumps({k: v for k, v in out["analysis"].items() if k != "provenance"}, indent=1,
                     default=str))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"describe": describe, "gates": gates,
              "attacks": attacks, "attacks-worker": attacks_worker,
              "run": run, "run-worker": run_worker,
              "analyse": analyse}[ph]())
