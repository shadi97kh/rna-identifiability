#!/usr/bin/env python3
"""CERTIFIED FOLDING AT VIENNARNA'S DEFAULT MODEL. One POST HOC item, as specified before any
code in docs/certfold_default_model_cert.md (the request of 2026-10-20; the freeze
lifted for this one item by decision).

Registration 1's set (B_four, the published envelope at lambda = 1), weights, callback and
labels, at dangles 2 with special hairpins on. Registration 1's functions run unchanged; they
build their compounds through thermo.vienna_ref.reference_md(), which use_model() rebinds in this
process only, and certfold.Folder is wrapped by CheckedFolder, which reads back the model on both
compounds and checks every fold's E0 against the dangles 2, special-hairpin count model (and C1)
on top of registration 1's recursion-side identity. Two drivers are rewritten because their own
code fixes the dangles 0 count model: G4's loop (certfold.gate_g4's body, kept) and G5's table
attacks (certfold_baselines.pair_frequencies guards with the dangles 0 count model).

PHASES, each refusing unless the specification is committed unmodified and its predecessor's
record is committed:
  zuker       (a) registration 1's G3 at dangles 2, and the shipped documentation
  gate        C1 to C3; (b) G1, both backends; (c) G2 with the power check, and G4
  reproduce   (e) the label driver at dangles 0 reproduces registration 1 on 50 TS0 sequences
  attacks     (d) G5 on TS0 at dangles 2 (8 workers)
  g6          the addendum: G6 with the three dangles 2 gauge directions, and G3's coverage
  run         TS0 labels, the envelope at lambda = 1 (8 workers)
  analyse     the shares, beside registration 1's
Run in .venv-certfold. Record: runs/2026-10-20_certfold_default_cert/metrics.json.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold_reg2 as r2  # noqa: E402
import thermo.vienna_ref as vr  # noqa: E402
from scripts.certfold_baselines import (  # noqa: E402
    _carry,
    _energy,
    _ninio_correction,
    envelope_draws,
    shifted_theta,
)
from scripts.turner_par import D2_CLIPPED, clip_d2, loaded, params_match, write_par  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402

cf, cr = r2.cf, r2.cr

ME = "scripts/certfold_default_cert.py"
SPEC = "docs/certfold_default_model_cert.md"
RUN_ID = "2026-10-20_certfold_default_cert"
OUT = R / "runs" / RUN_ID
SHARDS = OUT / "shards"
TAG_G5 = 93                      # the spec's seed tag: G5's 200 envelope draws
N_WORKERS = 8
N_REPRODUCE = 50                 # (e): the first 50 TS0 rows in registration 1's order
DEFAULT = (2, 1)                 # (dangles, special_hp): ViennaRNA's default model
PINNED = (0, 0)                  # registration 1's model

ZUKER_CHECKED = "zuker checked"
GATE_PASSED = "gate passed"
REPRODUCED = "reproduced"
G5_PASSED = "G5 passed"
G6_PASSED = "G6 passed"
RUN_DONE = "run done"
ANALYSED = "analysed"

# captured at import, before any rebinding: the pinned model of record
_ORIG_MD = vr.reference_md
_FOLDER = cf.Folder
_ACTIVE: list = [None]


# ----------------------------------------------------------------------------------
# records and gates (the house pattern; see certfold_default_model.py)
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
    if cf._git("ls-files", "--error-unmatch", SPEC).returncode != 0:
        raise SystemExit(f"{SPEC} is not committed")
    if cf._git("diff", "--quiet", "HEAD", "--", SPEC).returncode != 0:
        raise SystemExit(f"{SPEC} has uncommitted changes")
    cr._in_certfold_env()
    prov = r2.clean()
    out = read_out()
    if out.get("status") != status:
        raise SystemExit(f"status is {out.get('status')!r}; this phase needs {status!r}")
    if status is not None:
        committed()
    out.setdefault("specification", {
        "path": SPEC, "commit": cf._git("log", "-1", "--format=%H", "--", SPEC).stdout.strip(),
        "sha256": cf._sha(R / SPEC)})
    return out, prov


def drive(phase: str, n_w: int) -> list[int]:
    """Run `phase` in n_w worker processes (nice 10, single-threaded); stop all on a failure."""
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
        raise GuardFailure(f"{phase} workers exited {codes}; logs in {SHARDS}")
    return [p.pid for p in procs]


def shard_args() -> tuple[int, int]:
    return (int(sys.argv[sys.argv.index("--shard") + 1]),
            int(sys.argv[sys.argv.index("--of") + 1]))


# ----------------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------------
def model_md(dangles: int, special_hp: int):
    """The pinned reference model with `dangles` and `special_hp` set, each read back."""
    md = _ORIG_MD()
    md.dangles = dangles
    md.special_hp = special_hp
    if (md.dangles, md.special_hp) != (dangles, special_hp):
        raise GuardFailure(f"the model did not take: dangles {md.dangles}, "
                           f"special_hp {md.special_hp}")
    return md


def model_record() -> dict:
    import RNA

    md, bare = model_md(*DEFAULT), RNA.md()
    pinned = _ORIG_MD()
    if (pinned.dangles, pinned.special_hp) != PINNED:
        raise GuardFailure("thermo.vienna_ref.reference_md is not registration 1's model")
    return {"dangles": DEFAULT[0], "special_hp": DEFAULT[1],
            "fields_differing_from_a_bare_RNA_md": {
                f: {"here": getattr(md, f), "bare_RNA_md": getattr(bare, f)}
                for f in vr.REFERENCE_MODEL if getattr(md, f) != getattr(bare, f)},
            "every_other_field": "as thermo.vienna_ref.REFERENCE_MODEL",
            "viennarna": RNA.__version__}


class Energy:
    """E0(s) from the count model at theta0: Extractor(special, dangles, special_hp, lxc) counts
    . clip_d2(theta0) (dangles 2) or theta0 (dangles 0), plus the long-loop offset and the Ninio
    correction, in integer dcal. check() also asserts C1 on the structure."""

    def __init__(self, dangles: int, special_hp: int):
        T = cf.template()
        self.lc = cf.LoopCounts(T["special"], dangles, special_hp, lxc=T["lxc"])
        self.lc0 = cf.LoopCounts(T["special"], *PINNED, lxc=T["lxc"])
        th = T["theta"]
        self.theta0 = th
        self.theta = clip_d2(th) if dangles == 2 else dict(th)
        self.m, self.cap = th[("ninio_m",)], th[("ninio_max",)]
        self.n_checked = 0

    @staticmethod
    def four(counts: dict) -> dict:
        return {k: v for k, v in sorted(counts.items(), key=lambda kv: repr(kv[0]))
                if k[0] in cf.FOUR and v}

    def e0(self, seq: str, s: str, theta: dict | None = None) -> int:
        th = self.theta if theta is None else theta
        m, cap = (self.m, self.cap) if theta is None else (th[("ninio_m",)], th[("ninio_max",)])
        return _energy(self.lc.counts(seq, s), th) + _ninio_correction(s, m, cap)

    def check(self, seq: str, s: str, e_vienna: int, what: str) -> None:
        c = self.lc.counts(seq, s)
        want = _energy(c, self.theta) + _ninio_correction(s, self.m, self.cap)
        if want != e_vienna:
            raise GuardFailure(f"{what}: ViennaRNA gives {e_vienna} dcal, the count model "
                               f"{want} on {s[:60]}")
        if self.four(c) != self.four(self.lc0.counts(seq, s)):
            raise GuardFailure(f"{what}: C1 fails, the four-table counts depend on the model "
                               f"on {s[:60]}")
        self.n_checked += 1


class CheckedFolder(_FOLDER):
    """certfold.Folder, plus the model read back on both compounds, and on every structure a
    fold returns, E0 against the count model and C1 (the spec's two per-fold checks; the
    recursion-side identity is the base class's)."""

    EN: Energy | None = None

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        for fc in (self.plain, self.fc):
            md = fc.params.model_details
            if (md.dangles, md.special_hp) != _ACTIVE[0]:
                raise GuardFailure(f"a Folder compound is at dangles {md.dangles}, special_hp "
                                   f"{md.special_hp}, not {_ACTIVE[0]}")
        self._seen: dict[str, int] = {}
        self.eprime(self.star_db)

    def eprime(self, s: str) -> tuple[int, int]:
        ep, e0 = super().eprime(s)
        if s not in self._seen:
            CheckedFolder.EN.check(self.seq, s, e0, "fold")
            self._seen[s] = e0
        return ep, e0


def use_model(dangles: int, special_hp: int) -> Energy:
    """Rebind thermo.vienna_ref.reference_md, in THIS process, to the model; wrap certfold.Folder.

    At (0, 0) the name is the pinned function itself, unchanged. Returns the process's count
    model, which every CheckedFolder checks against."""
    vr.reference_md = (_ORIG_MD if (dangles, special_hp) == PINNED
                       else (lambda: model_md(dangles, special_hp)))
    _ACTIVE[0] = (dangles, special_hp)
    cf.Folder = CheckedFolder
    CheckedFolder.EN = Energy(dangles, special_hp)
    md = vr.reference_md()
    if (md.dangles, md.special_hp) != (dangles, special_hp):
        raise GuardFailure("the rebinding did not take")
    return CheckedFolder.EN


def plain_compound(seq: str):
    import RNA

    fc = RNA.fold_compound(seq, vr.reference_md())
    vr.assert_turner2004_params(fc)
    md = fc.params.model_details
    if (md.dangles, md.special_hp) != _ACTIVE[0]:
        raise GuardFailure(f"a plain compound is at {(md.dangles, md.special_hp)}")
    return fc


def assert_boltzmann(fc) -> None:
    """thermo.vienna_ref.assert_turner2004_boltzmann with its probe at the pinned model, whose
    constant it holds; its first check reads `fc`'s own Boltzmann factors."""
    cur = vr.reference_md
    vr.reference_md = _ORIG_MD
    try:
        vr.assert_turner2004_boltzmann(fc)
    finally:
        vr.reference_md = cur


def mfe_pairs(s: str) -> list[tuple[int, int]]:
    pt = cf.pair_table(s)
    return [(i, pt[i]) for i in range(1, len(s) + 1) if pt[i] > i]


def env_weights(B: dict, scale: float = 1.0):
    return cf.Weights(B["u_env"] * scale, B["cols"], 1.0)


# ----------------------------------------------------------------------------------
# phase: zuker -- (a)
# ----------------------------------------------------------------------------------
def zuker() -> int:
    out, prov = gate(None)
    t0 = time.time()
    import RNA

    EN = use_model(*DEFAULT)
    B = cf.load_tr0_basis()
    rows, _ = cf.tr0_rows()
    seqs = cf.gate_sequences(rows, cf.G3_SEQUENCES, 120, 150, cf.SEED + 3)
    radii = {"r1": 1.0,
             "Andronescu2007": float(B["radii"]["Andronescu2007"]["radius_on_B_four_kcal"]),
             "envelope": 1.0}
    kinds = {"r1": "ball", "Andronescu2007": "ball", "envelope": "envelope"}
    rng = np.random.default_rng(np.random.SeedSequence([cf.SEED, 3]))
    per, mismatch, infeasible_bad = [], [], []
    for _t, nm, seq, _db in seqs:
        plain = plain_compound(seq)
        star, e = plain.mfe()
        EN.check(seq, star, int(round(e * 100)), f"{nm} MFE")
        adm = cf.admissible_pairs(seq)
        Fb = cf.Folder(EN.lc, seq, star, cf.Weights(B["a_four"], B["cols"], 1.0))
        Fe = cf.Folder(EN.lc, seq, star, env_weights(B))
        rec = {"name": nm, "n_nt": len(seq), "n_admissible": len(adm), "by_set": {}}
        for label, r in radii.items():
            F = Fe if label == "envelope" else Fb
            F.set_radius(r)
            zmap = cf.zprime_map(F, F.zuker())
            covered = [q for q in adm if q in zmap]
            omitted = [q for q in adm if q not in zmap]
            take_c = [covered[k] for k in rng.permutation(len(covered))[:cf.G3_PAIRS]]
            take_o = [omitted[k] for k in rng.permutation(len(omitted))[:cf.G3_PAIRS]]
            agree = 0
            for q in take_c:
                got = F.with_pair(*q)
                if got is None or got[1] != zmap[q]:
                    mismatch.append({"name": nm, "set": label, "pair": q,
                                     "zuker": int(zmap[q]),
                                     "enforced": None if got is None else int(got[1])})
                else:
                    agree += 1
            for q in take_o:
                got = F.with_pair(*q)
                if got is not None:
                    infeasible_bad.append({"name": nm, "set": label, "pair": q,
                                           "enforced": int(got[1])})
            rec["by_set"][label] = {"kind": kinds[label], "scale": r,
                                    "n_covered_by_the_zuker_set": len(covered),
                                    "n_omitted": len(omitted), "n_sampled_covered": len(take_c),
                                    "n_sampled_omitted": len(take_o), "n_agree": agree}
        rec["folds"] = Fb.folds + Fe.folds
        rec["recursion"] = {k: Fb.recursion[k] + Fe.recursion[k] for k in Fb.recursion}
        per.append(rec)
    doc = RNA.fold_compound.subopt_zuker.__doc__ or ""
    supported = not mismatch and not infeasible_bad
    out["model"] = model_record()
    out["zuker"] = {
        "decided_by": "registration 1's G3 at dangles 2 (and G2's enumeration check, in the gate)",
        "documentation": {
            "source": "RNA.fold_compound.subopt_zuker.__doc__, ViennaRNA " + RNA.__version__,
            "sha256": hashlib.sha256(doc.encode()).hexdigest(),
            "mentions_dangles": "dangle" in doc.lower(),
            "first_line": doc.strip().splitlines()[0] if doc.strip() else None},
        "n_sequences": len(per), "lengths": [p["n_nt"] for p in per],
        "sets": {k: {"kind": kinds[k], "scale": v} for k, v in radii.items()},
        "pairs_sampled_per_set": cf.G3_PAIRS,
        "n_sampled_covered": sum(v["n_sampled_covered"] for p in per for v in p["by_set"].values()),
        "n_sampled_omitted": sum(v["n_sampled_omitted"] for p in per for v in p["by_set"].values()),
        "n_zprime_disagreements": len(mismatch), "n_omitted_but_feasible": len(infeasible_bad),
        "disagreements": (mismatch + infeasible_bad)[:5],
        "recursion_side": {k: sum(p["recursion"][k] for p in per) for k in per[0]["recursion"]},
        "count_model_checks": EN.n_checked,
        "folds": sum(p["folds"] for p in per),
        "supported": supported,
        "consequence": ("absent labels computed as at dangles 0" if supported else
                        "absent labels dropped everywhere: present labels only"),
        "seconds": round(time.time() - t0, 1), "provenance": prov,
        "environment": cf.env_record()}
    out["status"] = ZUKER_CHECKED
    write(out)
    print(json.dumps({k: v for k, v in out["zuker"].items()
                      if k not in ("provenance", "environment")}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------------
# phase: gate -- C1 to C3, (b) G1, (c) G2 and G4
# ----------------------------------------------------------------------------------
def carry_over(EN: Energy, B: dict, rows) -> dict:
    """C1 on TR0's references, C2, and C3."""
    T = cf.template()
    pos = {B["cols"][e]: k for k, e in enumerate(B["used"])}
    idx = {k: n for n, k in enumerate(B["cols"])}
    c2 = {t: int(max(v for k, v in T["theta"].items() if k[0] == t)) for t in D2_CLIPPED}
    # the clipped tables are unreachable at dangles 0, where the basis was built, so they may
    # not be among the set's coordinates at all; then no t moves them by construction
    clipped = [k for k in B["cols"] if k[0] in D2_CLIPPED]
    w_b4 = max((float(np.abs(B["B4"][pos[k]]).max()) for k in clipped if k in pos), default=0.0)
    w_a = max((float(abs(B["a_four"][idx[k]])) for k in clipped), default=0.0)
    w_u = max((float(abs(B["u_env"][idx[k]])) for k in clipped), default=0.0)
    if any(v > 0 for v in c2.values()) or w_b4 != 0.0 or w_a != 0.0 or w_u != 0.0:
        raise GuardFailure(f"C2 fails: clipped-table maxima {c2}, weights {w_b4}, {w_a}, {w_u}")
    mx2 = mx0 = 0.0
    for _t, nm, seq, db in rows:
        a2, a0 = EN.four(EN.lc.counts(seq, db)), EN.four(EN.lc0.counts(seq, db))
        if a2 != a0:
            raise GuardFailure(f"C1 fails on TR0 reference {nm}")
        mx2 = max(mx2, float(np.abs(cf.basis_image(B["B4"], pos, a2)).max()))
        mx0 = max(mx0, float(np.abs(cf.basis_image(B["B4"], pos, a0)).max()))
    if mx2 != mx0:
        raise GuardFailure(f"C3 fails: max |B_four^T c(ref)| is {mx2} at dangles 2, {mx0} at 0")
    return {"C1_tr0_references": {"n": len(rows), "four_table_counts_equal": True},
            "C2": {"max_turner2004_entry_dcal_by_clipped_table": c2,
                   "n_dangle5_entries_exactly_zero": sum(
                       1 for k, v in T["theta"].items() if k[0] == "dangle5" and v == 0),
                   "n_clipped_entries_among_the_set_coordinates": len(clipped),
                   "max_weight_on_clipped_tables": {"B_four_rows": w_b4, "a_four": w_a,
                                                    "u_env": w_u},
                   "linear_on_the_set": True},
            "C3": {"max_abs_B_four_T_c_ref_dangles2": mx2, "max_abs_B_four_T_c_ref_dangles0": mx0,
                   "equal": True},
            "C1_every_fold": "checked on every structure any fold returns, with E0 "
                             "(CheckedFolder, Energy.check)"}


def gate_g1(EN: Energy, B: dict, rows) -> dict:
    from scripts.turner_nullspace import generating_set

    sets = {"envelope": ("envelope", 1.0), "r1": ("ball", 1.0),
            "Andronescu2007": ("ball", float(B["radii"]["Andronescu2007"]
                                             ["radius_on_B_four_kcal"])),
            "r_cap": ("ball", float(B["radii"]["r_cap_kcal"]))}

    def weights(kind, x):
        return (env_weights(B) if kind == "envelope" else
                cf.Weights(B["a_four"], B["cols"], x))

    seqs = cf.gate_sequences(rows, cf.G1_SEQUENCES, 50, 500, cf.SEED)
    samples = []
    for n, (_t, nm, seq, _db) in enumerate(seqs):
        plain = plain_compound(seq)
        star, e = plain.mfe()
        structs = cf.sample_structures(seq, cf.G1_SAMPLES, cf.SEED + n) + [star]
        for s in structs:
            EN.check(seq, s, int(round(plain.eval_structure(s) * 100)), f"G1 {nm}")
        samples.append((nm, seq, star, structs))
    gen = generating_set(cf.template()["special"])
    import RNA

    for _tag, seq, db in gen:
        fc = RNA.fold_compound(seq, vr.reference_md())
        EN.check(seq, db, int(round(fc.eval_structure(db) * 100)), "G1 generating row")
    res: dict = {}
    for backend in ("python", "compiled"):
        for name, (kind, x) in sets.items():
            W = weights(kind, x)
            n_s = nz = 0
            for nm, seq, star, structs in samples:
                g = cf.g1_identity(EN.lc, W, seq, structs, star, backend=backend)
                if not g["pass"]:
                    raise GuardFailure(f"G1 ({backend}, {name}) fails on {nm}: "
                                       f"{g['mismatches'][:2]}")
                n_s += g["n"]
                nz += g["n_with_a_nonzero_offset"]
            n_rows, nzc = 0, {"plus": 0, "minus": 0}
            for _tag, seq, db in gen:
                for sign, star_db in (("plus", db), ("minus", "." * len(seq))):
                    g = cf.g1_identity(EN.lc, W, seq, [db], star_db, backend=backend)
                    if not g["pass"]:
                        raise GuardFailure(f"G1 constructed ({backend}, {name}, {sign}) fails "
                                           f"on {seq} {db}: {g['mismatches'][:1]}")
                    nzc[sign] += g["n_with_a_nonzero_offset"]
                n_rows += 1
            res[f"{backend}:{name}"] = {"kind": kind, "scale": x,
                                        "sampled": {"n_sequences": len(samples),
                                                    "n_structures": n_s,
                                                    "n_with_a_nonzero_offset": nz},
                                        "constructed": {"n_rows": n_rows,
                                                        "n_with_a_nonzero_offset": nzc},
                                        "pass": True}
    return {"by_backend_and_set": res, "sequences": cf.G1_SEQUENCES,
            "samples_per_sequence": cf.G1_SAMPLES, "seed": cf.SEED,
            "lengths": [len(s[1]) for s in samples], "pass": True}


def gate_g2(EN: Energy, B: dict, rows) -> dict:
    W1 = cf.Weights(B["a_four"], B["cols"], 1.0)
    pos = {B["cols"][e]: k for k, e in enumerate(B["used"])}
    # registration 1's draw, regenerated with its own screen (the dangles 0 loop maps; the
    # interior-loop maps are the same at both settings by C1)
    seqs, screen = cf.g2_sequences(rows, cf.G2_SEQUENCES, 16, cf.G2_MAX_N, cf.SEED, EN.lc0, W1)
    reg1 = cf.read_out()["gate"]["G2"]
    if screen != reg1["sequences"]:
        raise GuardFailure(f"G2's draw is not registration 1's: {screen} vs {reg1['sequences']}")
    per, n_enum = [], 0
    for _src, seq, _n in seqs:
        plain = plain_compound(seq)
        for s in cf.enumerate_structures(seq):
            EN.check(seq, s, int(round(plain.eval_structure(s) * 100)), "G2 enumeration")
            n_enum += 1
        rec = cf.g2_one(EN.lc, B, pos, W1, seq, cf.G2_RADII, with_rho=False)
        bad = [v for v in (*rec["by_radius"].values(), rec["envelope"])
               if v["n_false_certificates"]]
        if bad:
            raise GuardFailure(f"G2: {sum(v['n_false_certificates'] for v in bad)} false "
                               f"certificates on {seq}: {bad[0]['false_certificates'][:2]}")
        per.append(rec)
    planted = planted_env = 0
    n_planted = min(cf.G2_POWER_SEQUENCES, len(seqs))
    for _src, seq, _n in seqs[:n_planted]:
        rec = cf.g2_one(EN.lc, B, pos, W1, seq, cf.G2_RADII, scale=0.5, with_rho=False)
        planted += sum(v["n_false_certificates"] for v in rec["by_radius"].values())
        planted_env += rec["envelope"]["n_false_certificates"]
    if planted == 0 or planted_env == 0:
        raise GuardFailure(f"G2 power check: halved weights gave {planted} false certificates "
                           f"for the ball and {planted_env} for the envelope")

    def agg(key):
        return {f"{r:g}": sum(p["by_radius"][f"{r:g}"][key] for p in per) for r in cf.G2_RADII}

    return {"sequences": screen, "n_sequences": len(per),
            "n_structures_total": sum(p["n_structures"] for p in per),
            "n_enumerated_energies_checked": n_enum,
            "registration_1_n_structures_total": reg1["n_structures_total"],
            "n_pairs_star_total": sum(p["n_pairs_star"] for p in per),
            "radii_kcal": list(cf.G2_RADII),
            "certified_present": agg("n_certified_present"),
            "certified_absent": agg("n_certified_absent"),
            "exactly_robust_present": agg("n_exactly_robust_present"),
            "false_certificates": agg("n_false_certificates"),
            "envelope": {k: sum(p["envelope"][k] for p in per)
                         for k in ("n_certified_present", "n_certified_absent",
                                   "n_exactly_robust_present", "n_exactly_robust_absent",
                                   "n_false_certificates", "whole_structure_certified")},
            "pair_tables_vs_enumeration": {"x_prime_mismatches": 0, "z_prime_mismatches": 0},
            "power_check": {"weights_scaled_by": 0.5, "n_sequences": n_planted,
                            "false_certificates_produced": planted,
                            "false_certificates_produced_envelope": planted_env, "pass": True},
            "recursion_side": {k: sum(p["recursion"][k] for p in per)
                               for k in per[0]["recursion"]},
            "folds": sum(p["folds"] for p in per), "rho": "not computed (no radius here)",
            "pass": True}


def gate_g4(EN: Energy, B: dict, rows) -> dict:
    """certfold.gate_g4's body, at the active model, on its own sequences."""
    lc = EN.lc
    pos = {B["cols"][e]: k for k, e in enumerate(B["used"])}
    seqs = cf.gate_sequences(rows, cf.G4_SEQUENCES, 30, cf.G4_MAX_N, cf.SEED + 4)
    reg1 = cf.read_out()["gate"]["G4"]
    lens = {"min": min(len(r[2]) for r in seqs), "max": max(len(r[2]) for r in seqs)}
    if lens != reg1["lengths"] or len(seqs) != reg1["n_sequences"]:
        raise GuardFailure(f"G4's sequences are not registration 1's: {lens}, {len(seqs)}")
    per, viol = [], []
    for _t, nm, seq, _db in seqs:
        plain = plain_compound(seq)
        star, e = plain.mfe()
        EN.check(seq, star, int(round(e * 100)), f"G4 {nm} MFE")
        sp = mfe_pairs(star)
        adm = cf.admissible_pairs(seq)
        u_star = cf.basis_image(B["B4"], pos, lc.counts(seq, star))
        F = cf.Folder(lc, seq, star, cf.Weights(B["a_four"], B["cols"], cf.G2_RADII[0]))
        Fe = cf.Folder(lc, seq, star, env_weights(B))
        c_star = lc.counts(seq, star)
        rec = {"name": nm, "n_nt": len(seq), "n_pairs_star": len(sp), "by_radius": {}}
        sets = ([(f"{r:g}", "ball", F, r) for r in cf.G2_RADII]
                + [("envelope", "envelope", Fe, 1.0)])
        for label, kind, Fx, r in sets:
            Fx.set_radius(r)
            _s_min, e_min, _e0 = Fx.free()
            delta = Fx.eprime_star - e_min
            if delta < 0:
                raise GuardFailure(f"E'(s*) is below the constrained minimum on {nm}")
            if delta > cf.G4_DELTA_CAP:
                rec["by_radius"][label] = {"kind": kind, "scale": r, "skipped": True,
                                           "band_delta_dcal": int(delta)}
                continue
            band = []
            for z in Fx.fc.subopt(int(delta)):
                Fx._check(z.structure, int(round(z.energy * 100)), "wuchty")
                Fx.recursion["subopt_structures_checked"] += 1
                band.append(z.structure)
            if not band:
                raise GuardFailure(f"an empty band on {nm} at {label} with delta = {delta}")
            best = min(Fx.eprime(s)[0] for s in band)
            if best != e_min:
                raise GuardFailure(f"subopt under E' does not reach the constrained minimum on "
                                   f"{nm}: {best} != {e_min}")
            c = cf.certify(Fx, sp, adm)
            cert_p = {p for p, m in c["margins_present_dcal"].items() if m > 0}
            cert_a = {q for q, m in c["margins_absent_dcal"].items() if m is None or m > 0}
            n_witness = 0
            for s in band:
                if s == star:
                    continue
                ep, e0 = Fx.eprime(s)
                if ep > Fx.eprime_star:
                    continue
                n_s = (e0 - Fx.e0_star) / 100.0
                cs = lc.counts(seq, s)
                if kind == "ball":
                    u = cf.basis_image(B["B4"], pos, cs)
                    margin = n_s - r * float(np.linalg.norm(u - u_star))
                else:
                    box = sum(abs(cs.get(k, 0) - c_star.get(k, 0))
                              * float(B["u_env"][Fe.W.idx[k]])
                              for k in set(cs) | set(c_star) if k in Fe.W.idx)
                    margin = n_s - r * box
                if margin > 0:
                    continue
                n_witness += 1
                here = set(mfe_pairs(s))
                for p in cert_p - here:
                    viol.append({"name": nm, "set": label, "kind": "present", "pair": p,
                                 "margin_kcal": margin, "structure": s[:60]})
                for q in cert_a & here:
                    viol.append({"name": nm, "set": label, "kind": "absent", "pair": q,
                                 "margin_kcal": margin, "structure": s[:60]})
            rec["by_radius"][label] = {"kind": kind, "scale": r, "skipped": False,
                                       "band_delta_dcal": int(delta), "band_size": len(band),
                                       "n_exact_witnesses": n_witness,
                                       "n_certified_present": len(cert_p),
                                       "n_certified_absent": len(cert_a)}
        rec["folds"] = F.folds + Fe.folds
        rec["recursion"] = {k: F.recursion[k] + Fe.recursion[k] for k in F.recursion}
        per.append(rec)
        if viol:
            raise GuardFailure(f"G4: {len(viol)} false certificates: {viol[:2]}")
    labels = [*(f"{r:g}" for r in cf.G2_RADII), "envelope"]
    return {"n_sequences": len(per), "lengths": lens, "band_cap_dcal": cf.G4_DELTA_CAP,
            "by_set": {lab: {
                "n_run": sum(1 for p in per if not p["by_radius"][lab]["skipped"]),
                "n_skipped": sum(1 for p in per if p["by_radius"][lab]["skipped"]),
                "band_size_total": sum(p["by_radius"][lab].get("band_size", 0) for p in per),
                "n_exact_witnesses": sum(p["by_radius"][lab].get("n_exact_witnesses", 0)
                                         for p in per),
                "n_certified_present": sum(p["by_radius"][lab].get("n_certified_present", 0)
                                           for p in per)} for lab in labels},
            "registration_1_by_set": {k: {kk: v[kk] for kk in ("n_run", "n_skipped")}
                                      for k, v in reg1["by_radius"].items()},
            "n_false_certificates": 0, "folds": sum(p["folds"] for p in per),
            "recursion_side": {k: sum(p["recursion"][k] for p in per)
                               for k in per[0]["recursion"]},
            "pass": True}


def gate_phase() -> int:
    out, prov = gate(ZUKER_CHECKED)
    t0 = time.time()
    EN = use_model(*DEFAULT)
    B = cf.load_tr0_basis()
    rows, _ = cf.tr0_rows()
    res = {"carry_over": carry_over(EN, B, rows)}
    ts = time.time()
    res["G1"] = gate_g1(EN, B, rows)
    res["G1"]["seconds"] = round(time.time() - ts, 1)
    print("G1 passed", res["G1"]["seconds"], flush=True)
    ts = time.time()
    res["G2"] = gate_g2(EN, B, rows)
    res["G2"]["seconds"] = round(time.time() - ts, 1)
    print("G2 passed", res["G2"]["seconds"], flush=True)
    ts = time.time()
    res["G4"] = gate_g4(EN, B, rows)
    res["G4"]["seconds"] = round(time.time() - ts, 1)
    print("G4 passed", res["G4"]["seconds"], flush=True)
    res.update({"count_model_checks": EN.n_checked, "seconds": round(time.time() - t0, 1),
                "provenance": prov, "environment": cf.env_record()})
    out["gate"] = res
    out["status"] = GATE_PASSED
    write(out)
    print(json.dumps({k: v for k, v in res.items() if k not in ("provenance", "environment")},
                     indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------------
# the label driver (the run's, and (e)'s)
# ----------------------------------------------------------------------------------
def label_one(EN: Energy, B: dict, r: dict, zuker: bool = True) -> dict:
    """The envelope at lambda = 1 on one TS0 row: certfold.certify on a checked Folder, with
    contested pairs from the plain compound's base-pair probabilities (pf ignores callbacks)."""
    t0 = time.time()
    seq = r["seq"]
    plain = plain_compound(seq)
    star, e = plain.mfe()
    EN.check(seq, star, int(round(e * 100)), f"{r['name']} MFE")
    plain.exp_params_rescale(e)
    plain.pf()
    assert_boltzmann(plain)
    bpp = np.array(plain.bpp(), dtype=np.float64)
    sp = mfe_pairs(star)
    adm = cf.admissible_pairs(seq)
    star_set = set(sp)
    outside = [q for q in adm if q not in star_set]
    contested = {q for q in outside if bpp[q[0], q[1]] >= cf.CONTESTED_P}
    F = cf.Folder(EN.lc, seq, star, env_weights(B))
    c = cf.certify(F, sp, adm, zuker=zuker)
    if zuker and c["n_absent_because_unreachable"]:
        # G3's registered property on the test sequence itself: every admissible pair is in
        # the Zuker set, so no absent label rests on a pair the set merely failed to return
        raise GuardFailure(f"{r['name']}: {c['n_absent_because_unreachable']} admissible pairs "
                           f"are missing from the Zuker set")
    mp = c["margins_present_dcal"]
    cert_a = ({q for q, m in c["margins_absent_dcal"].items() if m is None or m > 0}
              if zuker else set())
    n_p = sum(1 for m in mp.values() if m > 0)
    return {"index": r["index"], "name": r["name"], "n_nt": len(seq), "stratum": r["stratum"],
            "star": star, "e0_star_dcal": int(round(e * 100)), "n_pairs_star": len(sp),
            "n_certified_present": n_p,
            "n_admissible_outside_star": len(outside), "n_contested": len(contested),
            "n_certified_absent_admissible": len(cert_a) if zuker else None,
            "n_certified_absent_contested": len(cert_a & contested) if zuker else None,
            "whole_structure_certified": (n_p == len(sp) and len(cert_a) == len(outside)
                                          if zuker else None),
            "pairs": [{"pair": list(p), "p": float(bpp[p[0], p[1]]), "margin_dcal": int(mp[p])}
                      for p in sp],
            "folds": F.folds, "recursion": dict(F.recursion),
            "seconds": round(time.time() - t0, 2)}


def reproduce() -> int:
    out, prov = gate(GATE_PASSED)
    t0 = time.time()
    EN = use_model(*PINNED)
    B = cf.load_tr0_basis()
    rows = r2.ts0_rows_with_stars()[:N_REPRODUCE]
    seq_recs, _ = cr.read_shards(cf.read_out()["run"]["workers"])
    by = {s["index"]: s for s in seq_recs}
    n_pairs = 0
    for r in rows:
        got, want = label_one(EN, B, r), by[r["index"]]
        if got["star"] != want["star"]:
            raise GuardFailure(f"(e) {r['name']}: s* is not registration 1's")
        wm = {tuple(q["pair"]): q["margins_dcal"]["envelope:1"] for q in want["pairs"]}
        gm = {tuple(q["pair"]): q["margin_dcal"] for q in got["pairs"]}
        if gm != wm:
            raise GuardFailure(f"(e) {r['name']}: present margins differ from registration 1's")
        ws = want["sets"]["envelope:1"]
        for k in ("n_certified_present", "n_certified_absent_admissible",
                  "n_certified_absent_contested"):
            if got[k] != ws[k]:
                raise GuardFailure(f"(e) {r['name']}: {k} {got[k]} != registration 1's {ws[k]}")
        if got["n_contested"] != want["n_contested"]:
            raise GuardFailure(f"(e) {r['name']}: contested pairs differ")
        n_pairs += len(gm)
    out["reproduce"] = {"n_sequences": len(rows), "n_mfe_pairs": n_pairs,
                        "model": {"dangles": PINNED[0], "special_hp": PINNED[1]},
                        "compared": "s*, every envelope:1 present margin, the certified-present "
                                    "and certified-absent counts (admissible and contested) and "
                                    "the contested count, against registration 1's run records",
                        "pass": True, "count_model_checks": EN.n_checked,
                        "seconds": round(time.time() - t0, 1), "provenance": prov,
                        "environment": cf.env_record()}
    out["status"] = REPRODUCED
    write(out)
    print(json.dumps({k: v for k, v in out["reproduce"].items()
                      if k not in ("provenance", "environment")}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------------
# phase: attacks -- (d) G5 on TS0
# ----------------------------------------------------------------------------------
def attacks_worker() -> int:
    if read_out().get("status") != REPRODUCED:
        raise SystemExit("G5 workers run only under the attacks phase")
    cr._in_certfold_env()
    w, n_w = shard_args()
    import RNA

    EN = use_model(*DEFAULT)
    lc = EN.lc
    B = cf.load_tr0_basis()
    T = cf.template()
    cols, used, B4 = B["cols"], B["used"], B["B4"]
    idx = {k: n for n, k in enumerate(cols)}
    pos = {cols[e]: n for n, e in enumerate(used)}
    u = B["u_env"]
    delta = float(B["envelope"]["delta_kcal"])
    r_max = float(B["envelope"]["r_max_kcal"])
    rows = [r for r in r2.ts0_rows_with_stars() if r["index"] % n_w == w]
    comps = cf.published_component_shifts(B)
    draws, drec = envelope_draws(B, cf.G5_DRAWS, seed_tag=TAG_G5, lam=1.0)
    shifts = [c for _n, c in comps] + list(draws)
    kinds = [f"component:{n}" for n, _c in comps] + ["draw"] * len(draws)

    def margin(rec, s, shift):
        """E_theta(s) - E_theta(s*) in dcal, exact float, theta = theta0 + shift (unrounded),
        from the active count model (clip_d2 leaves the set's moves alone: C2)."""
        cs = lc.counts(rec["seq"], s)
        dc = {kk: cs.get(kk, 0) - rec["c_star"].get(kk, 0)
              for kk in set(cs) | set(rec["c_star"]) if kk in idx}
        return (EN.e0(rec["seq"], s) - rec["e0_star"]
                + 100.0 * sum(v * shift[idx[kk]] for kk, v in dc.items() if v))

    per = []
    for r in rows:
        seq = r["seq"]
        plain = plain_compound(seq)
        star, e = plain.mfe()
        EN.check(seq, star, int(round(e * 100)), f"{r['name']} MFE")
        sp = mfe_pairs(star)
        F = cf.Folder(lc, seq, star, env_weights(B))
        c = cf.certify(F, sp, cf.admissible_pairs(seq))
        rec = {"name": r["name"], "seq": seq, "star": star,
               "cert_p": {p for p, m in c["margins_present_dcal"].items() if m > 0},
               "cert_a": {q for q, m in c["margins_absent_dcal"].items() if m is None or m > 0},
               "c_star": lc.counts(seq, star), "e0_star": EN.e0(seq, star),
               "viol": collections.Counter(), "n_tested": collections.Counter()}
        if rec["e0_star"] != F.e0_star:
            raise GuardFailure(f"{r['name']}: the count model and eval_structure disagree on E0")
        per.append(rec)
        alts = sorted((e_, s) for s, e_ in F.zuker() if s != star)[:cf.G5_TOP_ALTERNATIVES]
        for _e, s in alts:
            cs = lc.counts(seq, s)
            dc = [(kk, cs.get(kk, 0) - rec["c_star"].get(kk, 0))
                  for kk in set(cs) | set(rec["c_star"]) if kk in pos]
            supp = [(kk, v) for kk, v in dc if v and u[idx[kk]] > 0]
            shift = np.zeros(len(cols))
            if supp:
                M = np.array([B4[pos[kk]] for kk, _v in supp])
                target = np.array([-u[idx[kk]] * np.sign(v) for kk, v in supp])
                t, *_ = np.linalg.lstsq(M, target, rcond=None)
                shift[used] = B4 @ t
                nt, mx = float(np.linalg.norm(t)), float(np.abs(shift).max())
                shift *= min(1.0, r_max / nt if nt > 0 else 1.0, delta / mx if mx > 0 else 1.0)
            rec["n_tested"]["worst_case"] += 1
            here = set(mfe_pairs(s))
            touching = bool((rec["cert_p"] - here) or (rec["cert_a"] & here))
            rec["n_tested"]["worst_case_touching"] += touching
            if touching and margin(rec, s, shift) <= 0:
                rec["viol"]["worst_case"] += 1

    # the table attacks: certfold_baselines.pair_frequencies's loop, guarded by THIS model
    clip0 = {k: v for k, v in clip_d2(T["theta"]).items() if k[0] in D2_CLIPPED}
    fd, tmp = tempfile.mkstemp(prefix="g5_draw_", suffix=".par", dir=SHARDS)
    os.close(fd)
    path = pathlib.Path(tmp)
    table_rec, n_folds = [], 0
    try:
        for d, shift in enumerate(shifts):
            theta, resid, n_changed = shifted_theta(T["theta"], cols, shift)
            th2 = clip_d2(theta)
            if {k: v for k, v in th2.items() if k[0] in D2_CLIPPED} != clip0:
                raise GuardFailure(f"table {d} moves a clipped dangles 2 table")
            write_par(theta, path)
            sha = cf._sha(path)
            with loaded(path):
                md = model_md(*DEFAULT)
                first = RNA.fold_compound(per[0]["seq"], md)
                bad = params_match(first, theta, *DEFAULT)
                if bad:
                    raise GuardFailure(f"table {d}: the loaded table is not the written one: {bad}")
                carry = _carry(theta, _ORIG_MD())
                m, cap = theta[("ninio_m",)], theta[("ninio_max",)]
                for k, rec in enumerate(per):
                    fc = first if k == 0 else RNA.fold_compound(rec["seq"], md)
                    s, e = fc.mfe()
                    want = _energy(lc.counts(rec["seq"], s), th2) + _ninio_correction(s, m, cap)
                    if want != int(round(e * 100)):
                        raise GuardFailure(f"table {d}, {rec['name']}: ViennaRNA's MFE energy "
                                           f"{int(round(e * 100))} dcal, the count model {want}")
                    n_folds += 1
                    rec["n_tested"][kinds[d]] += 1
                    here = set(mfe_pairs(s))
                    if (rec["cert_p"] - here) or (rec["cert_a"] & here):
                        if margin(rec, s, shift) <= 0:
                            rec["viol"][kinds[d]] += 1
            table_rec.append({"table": d, "kind": kinds[d], "max_rounding_kcal": resid,
                              "n_entries_changed_dcal": n_changed, "par_sha256": sha,
                              "n_carry_rows": len(carry)})
    finally:
        path.unlink(missing_ok=True)
    restored = plain_compound("GGGGAAACCCC")
    _s, e = restored.mfe()
    restored.exp_params_rescale(e)
    restored.pf()
    assert_boltzmann(restored)
    with open(SHARDS / f"g5_{w}.jsonl", "w") as fh:
        for rec in per:
            fh.write(json.dumps({"name": rec["name"], "violations": dict(rec["viol"]),
                                 "n_tested": dict(rec["n_tested"])}) + "\n")
    (SHARDS / f"g5_{w}.meta.json").write_text(json.dumps({
        "envelope_draws": {k: v for k, v in drec.items() if k != "draws"},
        "tables": table_rec, "n_tables": len(shifts), "n_table_folds": n_folds,
        "count_model_checks": EN.n_checked}, default=str))
    return 0


def attacks() -> int:
    out, prov = gate(REPRODUCED)
    load, t0 = os.getloadavg(), time.time()
    pids = drive("attacks-worker", N_WORKERS)
    per, metas = [], []
    for w in range(N_WORKERS):
        per += [json.loads(ln) for ln in (SHARDS / f"g5_{w}.jsonl").read_text().splitlines()
                if ln.strip()]
        metas.append(json.loads((SHARDS / f"g5_{w}.meta.json").read_text()))
    if len(per) != 1288:
        raise GuardFailure(f"G5 attacked {len(per)} of 1,288 TS0 sequences")
    viol, tested = collections.Counter(), collections.Counter()
    for r in per:
        viol.update(r["violations"])
        tested.update(r["n_tested"])
    total = sum(viol.values())
    out["G5"] = {
        "rule": "violation counts only; no share, accuracy or coverage",
        "n_sequences": len(per), "violations": dict(viol), "n_tested": dict(tested),
        "n_violations": total,
        "attacks": {"components": 3, "envelope_draws": cf.G5_DRAWS,
                    "worst_case_alternatives": cf.G5_TOP_ALTERNATIVES,
                    "draw_seed": [cf.SEED, TAG_G5]},
        "envelope_draws": metas[0]["envelope_draws"],
        "n_table_folds": sum(m["n_table_folds"] for m in metas),
        "max_rounding_kcal": max(t["max_rounding_kcal"] for m in metas for t in m["tables"]),
        "count_model_checks": sum(m["count_model_checks"] for m in metas),
        "pass": total == 0, "workers": N_WORKERS, "pids": pids,
        "load_at_start": [round(x, 2) for x in load], "seconds": round(time.time() - t0, 1),
        "files_sha256": {f"g5_{w}.jsonl": cf._sha(SHARDS / f"g5_{w}.jsonl")
                         for w in range(N_WORKERS)},
        "provenance": prov, "environment": cf.env_record()}
    out["status"] = G5_PASSED if total == 0 else "G5 FAILED: violations found"
    write(out)
    print(json.dumps({k: v for k, v in out["G5"].items()
                      if k not in ("provenance", "environment", "files_sha256")},
                     indent=1, default=str))
    return 0 if total == 0 else 1


# ----------------------------------------------------------------------------------
# phase: g6 -- the addendum: G6 at dangles 2, and G3's coverage stated
# ----------------------------------------------------------------------------------
def gauge_tables_default(theta0: dict, norm_kcal: float) -> dict:
    """certfold.gauge_tables, with the gauge directions of the DEFAULT model (three at dangles
    2: N1 with special hairpins on, N2, and N3 = mismatchM against ML_ci) over every entry of
    the parameter table, since N3's entries are not among the basis's dangles 0 coordinates."""
    from scripts.turner_nullspace import analytic_directions

    cols = sorted(theta0, key=repr)
    out = {}
    for name, g in analytic_directions(cols, *DEFAULT).items():
        scale = float(np.linalg.norm(g))
        step = max(1, int(round(norm_kcal * 100.0 / scale)))
        shifted = dict(theta0)
        touched = 0
        for k, c in zip(cols, g, strict=True):
            if c:
                shifted[k] = int(shifted[k] + step * int(round(c)))
                touched += 1
        out[name] = {"theta": shifted, "step_dcal": step, "n_entries_touched": touched,
                     "norm_kcal": step * scale / 100.0,
                     "max_entry_shift_dcal": int(step * max(abs(int(round(c))) for c in g if c))}
    return out


def g6() -> int:
    out, prov = gate(G5_PASSED)
    t0 = time.time()
    import RNA

    EN = use_model(*DEFAULT)
    lc = EN.lc
    B = cf.load_tr0_basis()
    T = cf.template()
    rows, _ = cf.tr0_rows()
    rad = B["radii"]
    ladder = sorted([1.0, float(rad["Langdon2018"]["radius_on_B_four_kcal"]),
                     float(rad["Andronescu2007"]["radius_on_B_four_kcal"]),
                     float(rad["Turner1999"]["radius_on_B_four_kcal"])])
    lams = list(cf.LAMBDAS)
    folders = []

    def labels(F, sp, adm):
        c = cf.certify(F, sp, adm)
        if c["n_absent_because_unreachable"]:
            raise GuardFailure("G3: an admissible pair is missing from the Zuker set")
        return ({p for p, m in c["margins_present_dcal"].items() if m > 0},
                {q for q, m in c["margins_absent_dcal"].items() if m is None or m > 0})

    # --- monotonicity: the ball over registration 1's ladder, and the envelope over lambda
    viol, mono = [], []
    for _t, nm, seq, _db in cf.gate_sequences(rows, cf.G6_SEQUENCES, 50, 150, cf.SEED + 6):
        plain = plain_compound(seq)
        star, e = plain.mfe()
        EN.check(seq, star, int(round(e * 100)), f"G6 {nm} MFE")
        sp, adm = mfe_pairs(star), cf.admissible_pairs(seq)
        F = cf.Folder(lc, seq, star, cf.Weights(B["a_four"], B["cols"], ladder[0]))
        Fe = cf.Folder(lc, seq, star, env_weights(B))
        folders += [F, Fe]
        rec = {"name": nm, "n_nt": len(seq)}
        for kind, Fx, scales in (("ball", F, ladder), ("envelope", Fe, lams)):
            got = []
            for x in scales:
                Fx.set_radius(x)
                got.append(labels(Fx, sp, adm))
            for a, b, xa, xb in zip(got[:-1], got[1:], scales[:-1], scales[1:], strict=True):
                for half, what in ((0, "present"), (1, "absent")):
                    gained = b[half] - a[half]
                    if gained:
                        viol.append({"name": nm, "set": kind, "kind": what, "from": xa,
                                     "to": xb, "n_gained": len(gained)})
            rec[kind] = {"certified_present": [len(g[0]) for g in got],
                         "certified_absent": [len(g[1]) for g in got]}
        mono.append(rec)

    # --- the gauge: the default model's three directions, written and loaded
    gauge = gauge_tables_default(T["theta"], cf.G6_GAUGE_NORM)
    sets = {"ball:1": ("ball", 1.0),
            "ball:Andronescu2007": ("ball", float(rad["Andronescu2007"]
                                                  ["radius_on_B_four_kcal"])),
            "envelope:1": ("envelope", 1.0)}

    def weights(kind, x):
        return (env_weights(B) if kind == "envelope" else
                cf.Weights(B["a_four"], B["cols"], x))

    base = {}
    for _t, nm, seq, _db in cf.gate_sequences(rows, cf.G6_GAUGE_SEQUENCES, 50, 150,
                                              cf.SEED + 7):
        plain = plain_compound(seq)
        star, e = plain.mfe()
        EN.check(seq, star, int(round(e * 100)), f"G6 gauge {nm} MFE")
        sp, adm = mfe_pairs(star), cf.admissible_pairs(seq)
        lab = {}
        for name, (kind, x) in sets.items():
            F = cf.Folder(lc, seq, star, weights(kind, x))
            folders.append(F)
            lab[name] = labels(F, sp, adm)
        base[nm] = {"seq": seq, "star": star, "e0": int(round(e * 100)), "pairs": sp,
                    "adm": adm, "labels": lab}
    g_out, g_viol = {}, []
    SHARDS.mkdir(parents=True, exist_ok=True)
    for gname, grec in gauge.items():
        path = SHARDS / f"gauge_{gname}.par"
        write_par(grec["theta"], path)
        n_e0 = n_star = n_lab = 0
        with loaded(path):
            for nm, b in base.items():
                fc = RNA.fold_compound(b["seq"], model_md(*DEFAULT))
                bad = params_match(fc, grec["theta"], *DEFAULT)
                if bad:
                    raise GuardFailure(f"G6: the loaded table is not the written one: {bad}")
                e0 = int(round(fc.eval_structure(b["star"]) * 100))
                if e0 != b["e0"]:
                    n_e0 += 1
                    # the diagnosis, recorded beside the violation: ViennaRNA's energy under the
                    # gauge table against the count model at that table, linear and clipped
                    th = grec["theta"]
                    cs = lc.counts(b["seq"], b["star"])
                    nc = _ninio_correction(b["star"], th[("ninio_m",)], th[("ninio_max",)])
                    g_viol.append({"direction": gname, "name": nm, "kind": "E0 of s*",
                                   "turner2004_dcal": b["e0"], "gauge_table_dcal": e0,
                                   "count_model_linear_dcal": _energy(cs, th) + nc,
                                   "count_model_clipped_dcal": _energy(cs, clip_d2(th)) + nc,
                                   "mismatchM_uses_in_s_star": sum(
                                       v for k, v in cs.items() if k[0] == "mismatchM")})
                star2, _e = fc.mfe()
                if star2 != b["star"]:
                    n_star += 1
                    g_viol.append({"direction": gname, "name": nm, "kind": "MFE structure"})
                    continue
                for name, (kind, x) in sets.items():
                    # pinned=False: the global table is the written gauge table, so the Turner
                    # 2004 fingerprint is the wrong pin (params_match above is the right one);
                    # CheckedFolder still holds every fold's E0 to the count model at theta0,
                    # which under a gauge table is the gauge claim itself, structure by structure
                    F = cf.Folder(lc, b["seq"], b["star"], weights(kind, x), pinned=False)
                    folders.append(F)
                    if labels(F, b["pairs"], b["adm"]) != b["labels"][name]:
                        n_lab += 1
                        g_viol.append({"direction": gname, "name": nm, "kind": "labels",
                                       "set": name})
        g_out[gname] = {k: v for k, v in grec.items() if k != "theta"}
        clipped_after = {t: [int(min(v for k, v in grec["theta"].items() if k[0] == t)),
                             int(max(v for k, v in grec["theta"].items() if k[0] == t))]
                         for t in D2_CLIPPED}
        g_out[gname].update({"par_sha256": cf._sha(path), "n_sequences": len(base),
                             "n_with_a_changed_E0": n_e0,
                             "n_with_a_changed_MFE_structure": n_star,
                             "n_with_changed_labels": n_lab,
                             "clipped_tables_range_after_shift_dcal": clipped_after})
    if viol or g_viol:
        # the failure is recorded, not only raised: the run refuses on this status, and the
        # violations stay in the artifact with their diagnosis (the house rule: keep attempts)
        out.setdefault("G6_failed_attempts", []).append({
            "monotonicity_violations": viol, "gauge_violations": g_viol,
            "gauge_by_direction": g_out,
            "n_monotonicity_violations": len(viol), "n_gauge_violations": len(g_viol),
            "count_model_checks": EN.n_checked, "seconds": round(time.time() - t0, 1),
            "provenance": prov, "environment": cf.env_record()})
        out["status"] = "G6 FAILED: violations found; the item stops (spec)"
        write(out)
        print(json.dumps({"n_monotonicity_violations": len(viol),
                          "n_gauge_violations": len(g_viol),
                          "gauge_by_direction": {k: {kk: v[kk] for kk in (
                              "n_with_a_changed_E0", "n_with_a_changed_MFE_structure",
                              "n_with_changed_labels", "step_dcal",
                              "clipped_tables_range_after_shift_dcal")}
                              for k, v in g_out.items()}}, indent=1, default=str))
        raise GuardFailure(f"G6: {len(viol)} monotonicity violations and {len(g_viol)} gauge "
                           f"violations: {(viol + g_viol)[:3]}")
    z = out["zuker"]
    out["G6"] = {
        "addendum": "the addendum of 2026-10-20 (docs/certfold_default_model_cert.md), "
                    "received after the zuker, gate and reproduce phases had run and during G5",
        "monotonicity": {"ball_radii_kcal": ladder, "envelope_lambdas": lams,
                         "n_sequences": len(mono), "n_violations": 0,
                         "ball_certified_present_totals": [
                             sum(m["ball"]["certified_present"][k] for m in mono)
                             for k in range(len(ladder))],
                         "envelope_certified_present_totals": [
                             sum(m["envelope"]["certified_present"][k] for m in mono)
                             for k in range(len(lams))]},
        "gauge": {"directions_from": "turner_nullspace.analytic_directions(all entries, "
                                     "dangles 2, special_hp 1): three, as the four-configuration "
                                     "Type A analysis found",
                  "target_norm_kcal": cf.G6_GAUGE_NORM, "sets": list(sets),
                  "directions": g_out, "n_violations": 0},
        "G3_coverage": {
            "statement": "no admissible pair is missing from the Zuker set",
            "zuker_phase": f"n_sampled_omitted = {z['n_sampled_omitted']} over "
                           f"{z['n_sequences']} sequences and {len(z['sets'])} sets, which "
                           f"samples min(60, n_omitted) per sequence and set, so n_omitted = 0 "
                           f"everywhere",
            "here": "every certify call above has n_absent_because_unreachable = 0 (enforced)",
            "run": "label_one enforces the same on every TS0 sequence"},
        "recursion_side": {k: sum(F.recursion[k] for F in folders)
                           for k in ("folds_checked", "zuker_structures_checked", "violations")},
        "count_model_checks": EN.n_checked, "pass": True,
        "seconds": round(time.time() - t0, 1), "provenance": prov,
        "environment": cf.env_record()}
    out["status"] = G6_PASSED
    write(out)
    print(json.dumps({k: v for k, v in out["G6"].items()
                      if k not in ("provenance", "environment")}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------------
# phase: run -- TS0 labels
# ----------------------------------------------------------------------------------
def run_worker() -> int:
    out = read_out()
    if out.get("status") != G6_PASSED:
        raise SystemExit("run workers run only under the run phase")
    cr._in_certfold_env()
    w, n_w = shard_args()
    EN = use_model(*DEFAULT)
    B = cf.load_tr0_basis()
    zk = bool(out["zuker"]["supported"])
    with open(SHARDS / f"run_{w}.jsonl", "w") as fh:
        for r in r2.ts0_rows_with_stars():
            if r["index"] % n_w == w:
                fh.write(json.dumps(label_one(EN, B, r, zuker=zk)) + "\n")
                fh.flush()
    (SHARDS / f"run_{w}.meta.json").write_text(json.dumps({"count_model_checks": EN.n_checked}))
    return 0


def run() -> int:
    out, prov = gate(G6_PASSED)
    load, t0 = os.getloadavg(), time.time()
    pids = drive("run-worker", N_WORKERS)
    recs = []
    for w in range(N_WORKERS):
        recs += [json.loads(ln) for ln in (SHARDS / f"run_{w}.jsonl").read_text().splitlines()
                 if ln.strip()]
    if len(recs) != 1288 or len({r["index"] for r in recs}) != 1288:
        raise GuardFailure(f"{len(recs)} run records, not 1,288 distinct")
    out["run"] = {
        "n_sequences": len(recs), "set": "the published envelope at lambda = 1",
        "zuker": bool(out["zuker"]["supported"]),
        "folds": sum(r["folds"] for r in recs),
        "recursion_side": {k: sum(r["recursion"][k] for r in recs)
                           for k in recs[0]["recursion"]},
        "count_model_checks": sum(json.loads((SHARDS / f"run_{w}.meta.json").read_text())
                                  ["count_model_checks"] for w in range(N_WORKERS)),
        "cpu_seconds": round(sum(r["seconds"] for r in recs), 1),
        "workers": N_WORKERS, "pids": pids, "load_at_start": [round(x, 2) for x in load],
        "wall_seconds": round(time.time() - t0, 1),
        "files_sha256": {f"run_{w}.jsonl": cf._sha(SHARDS / f"run_{w}.jsonl")
                         for w in range(N_WORKERS)},
        "provenance": prov, "environment": cf.env_record()}
    out["status"] = RUN_DONE
    write(out)
    print(json.dumps({k: v for k, v in out["run"].items()
                      if k not in ("provenance", "environment", "files_sha256")},
                     indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------------
# phase: analyse
# ----------------------------------------------------------------------------------
def analyse() -> int:
    out, prov = gate(RUN_DONE)
    recs = []
    for w in range(N_WORKERS):
        p = SHARDS / f"run_{w}.jsonl"
        if cf._sha(p) != out["run"]["files_sha256"][p.name]:
            raise GuardFailure(f"{p.name} changed since the run")
        recs += [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    reg1 = cf.read_out()["analysis"]["sets"]["TS0"]["describe"]["labels"]["envelope:1"]
    zk = bool(out["zuker"]["supported"])

    def share(n, of):
        return {"n": n, "of": of, "fraction": n / of if of else None}

    by = collections.defaultdict(list)
    for r in recs:
        by[r["stratum"]].append(r)
    present = {"pooled": share(sum(r["n_certified_present"] for r in recs),
                               sum(r["n_pairs_star"] for r in recs)),
               "by_stratum": {str(s): share(sum(r["n_certified_present"] for r in rs),
                                            sum(r["n_pairs_star"] for r in rs))
                              for s, rs in sorted(by.items())}}
    if set(present["by_stratum"]) != set(reg1["certified_present"]["by_stratum"]):
        raise GuardFailure("the strata are not registration 1's")
    table = {s: {"dangles2": v, "dangles0": reg1["certified_present"]["by_stratum"][s]}
             for s, v in present["by_stratum"].items()}
    res = {
        "label": "post hoc, descriptive",
        "certified_present": present,
        "dangles0_registration_1": {"pooled": reg1["certified_present"]["pooled"],
                                    "by_stratum": reg1["certified_present"]["by_stratum"]},
        "by_stratum_table": table,
        "n_mfe_pairs": {"dangles2": present["pooled"]["of"],
                        "dangles0": reg1["certified_present"]["pooled"]["of"]},
        "n_sequences_star_differs_from_dangles0": None,
        "certified_absent_contested": (share(sum(r["n_certified_absent_contested"] for r in recs),
                                             sum(r["n_contested"] for r in recs))
                                       if zk else None),
        "certified_absent_contested_dangles0": reg1["certified_absent_contested"]["pooled"],
        "whole_structure_certified": (share(sum(bool(r["whole_structure_certified"])
                                                for r in recs), len(recs)) if zk else None),
        "whole_structure_certified_dangles0": reg1["whole_structure_certified"]["pooled"],
        "reading": "if every check passed: S7 gets one paragraph and one table; Limitations' "
                   "clause becomes '; the certificate's set never moves them, and at ViennaRNA's "
                   "default model it certifies X% of TS0's MFE pairs (post hoc; Supplementary "
                   "Section S7)', X the pooled certified-present share to one decimal; the "
                   "abstract does not change",
        "provenance": prov}
    stars0 = {r["index"]: r["star"] for r in r2.ts0_rows_with_stars()}
    res["n_sequences_star_differs_from_dangles0"] = sum(r["star"] != stars0[r["index"]]
                                                        for r in recs)
    out["analysis"] = res
    out["status"] = ANALYSED
    write(out)
    print(json.dumps({k: v for k, v in res.items() if k not in ("provenance", "by_stratum_table")},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"zuker": zuker, "gate": gate_phase, "reproduce": reproduce,
              "attacks": attacks, "attacks-worker": attacks_worker, "g6": g6,
              "run": run, "run-worker": run_worker, "analyse": analyse}[ph]())
