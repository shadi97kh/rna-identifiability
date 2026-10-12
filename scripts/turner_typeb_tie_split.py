#!/usr/bin/env python3
"""Type B Experiment 2, EXPLORATORY: of the changed MFEs at the three published points, how
many are tie flips and how many are real reorderings?

Asked by the user 2026-10-12. 53% of the changed MFEs at the Andronescu point sit in sequences
whose unperturbed MFE is exactly tied (runs/2026-10-12_turner_typeb_exp2_explore), so part of
the 4.3% may be ViennaRNA picking a different structure among structures that were already
equal. The shards store whether an MFE changed, not the structures, so the changed sequences
(567 + 513 + 126 = 1,206) are refolded: under Turner 2004 for s0, and under the written
published-point table for s1.

Classification, by the exact Turner 2004 energies (counts . theta0 + offset, integer dcal/mol):
  dE0 = E0(s1) - E0(s0) >= 0 always, s0 being Turner 2004's MFE
  tie flip           dE0 == 0: s1 was already co-optimal under Turner 2004
  real reordering    dE0 >  0: the perturbation moved s1 below s0
The exact unrounded perturbed difference E_delta(s1) - E_delta(s0) is recorded beside it.

REPORTING RULE (the user's, written into the artifact by --phase preregister before any refold):
the headline reports real reorderings, as a fraction of the 11,970 sequences with a bootstrap
95% CI (2,000 draws over sequences, percentile), with the total changed and the tie-flip
count beside it.

Consistency checks (any failure stops the run): the shards' sha256 as recorded by the fold
driver; the published points' table sha256; recomputed s0 energies equal the shards'
baseline e0; every refolded s1 differs from s0 (the shards said it changed); each s1's MFE
energy equals counts . theta_r + offset exactly; the loaded table carries the written theta
on the carry rows and the constrained-pf probe; the genuine classification reproduces the
shards'.

Run: .venv/bin/python scripts/turner_typeb_tie_split.py --phase {preregister,run}
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

import numpy as np
import RNA

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_nullspace import Extractor, generating_set  # noqa: E402
from scripts.turner_par import _rebuild, loaded, params_match, template, write_par  # noqa: E402
from scripts.turner_typeb import (  # noqa: E402
    EXACT_TIE,
    FOLD,
    N_WORKERS,
    SEED,
    GuardFailure,
    _vec,
    fold_rows,
    load_basis,
    read_out,
    sha256,
)
from thermo.vienna_ref import assert_turner2004_params, reference_md  # noqa: E402

RUN_ID = "2026-10-12_turner_typeb_tie_split"
OUT = R / "runs" / RUN_ID
BOOT = 2000
RULE = {
    "rule_user": "the headline reports real reorderings, with the total changed and the "
                 "tie-flip count beside it. A reviewer can dismiss tie flips as arbitrary, but "
                 "can't dismiss reorderings.",
    "classification": "dE0 = E0(s1) - E0(s0) under Turner 2004 counts (integer dcal/mol): 0 "
                      "is a tie flip, > 0 a real reordering",
    "interval": f"real reorderings / 11,970 sequences; bootstrap 95% percentile CI, {BOOT} "
                f"draws over sequences, seed SeedSequence([{SEED}, 9, point])",
    "label": "EXPLORATORY; written before any refold (the user's instruction)",
}


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def preregister():
    if (OUT / "metrics.json").exists():
        raise SystemExit("the registration exists")
    write({"run_id": RUN_ID, "status": "PRE-REGISTERED, nothing refolded", "rule": RULE,
           **git_provenance(R)})
    return 0


def run():
    out = json.loads((OUT / "metrics.json").read_text())
    assert out["status"].startswith("PRE-REGISTERED"), out["status"]
    assert out["rule"] == json.loads(json.dumps(RULE)), "the rule changed since registration"
    art = read_out()
    assert art["status"] == "fold analysed", art["status"]
    for f, h in art["fold"]["shards_sha256"].items():
        assert sha256(FOLD / "full" / f) == h, f"{f} changed since the fold"
    for f, h in art["fold"]["points_sha256"].items():
        assert sha256(FOLD / f) == h, f"{f} changed since the fold"
    out.update({"status": "running", "provenance_run": git_provenance(R)})
    write(out)

    T = template()
    theta0, special, lxc = T["theta"], T["special"], T["lxc"]
    cols = load_basis("tr0ts0")["cols"]
    idx = {k: i for i, k in enumerate(cols)}
    Z = np.load(FOLD / "points.npz")
    t0 = Z["t0"]
    meta = json.loads((FOLD / "points.json").read_text())
    shards = [[json.loads(ln) for ln in (FOLD / "full" / f"shard{w}.jsonl").read_text()
               .splitlines()] for w in range(N_WORKERS)]
    base = [b for s in shards for b in s[0]["baseline"]]
    rows = {r[1]: r for r in fold_rows()}
    ex = Extractor(special, 0, 0, lxc=lxc)
    gen = generating_set(special)
    md = reference_md()
    own = [m for m in meta if m["kind"] == "published_own"]

    changed = {}
    for m in own:
        rs = [next(r for r in s if r.get("p") == m["p"]) for s in shards]
        ch = [x for r in rs for x in r["changed"]]
        gn = [x for r in rs for x in r["genuine"]]
        changed[m["p"]] = [(i, gn[i]) for i, c in enumerate(ch) if c]

    # s0 under Turner 2004, for every sequence any point changed
    _rebuild()
    s0 = {}
    for i in sorted({i for v in changed.values() for i, _ in v}):
        b = base[i]
        _, seq, db = rows[b["name"]][1:]
        fc = RNA.fold_compound(seq, md)
        if not s0:
            assert_turner2004_params(fc)
        s, e = fc.mfe()
        if int(round(e * 100)) != b["e0"]:
            raise GuardFailure(f"{b['name']}: recomputed e0 {e} != shard {b['e0']}")
        c = ex.counts(seq, s)
        off = c.pop(("_offset",), 0)
        s0[i] = (seq, s, int(round(e * 100)), _vec(c, idx), off)
    _rebuild()

    res = {}
    with tempfile.TemporaryDirectory() as t:
        for m in own:
            p = m["p"]
            th, dl = Z["theta_r"][p], Z["delta_dcal"][p]
            full = dict(theta0) | {k: int(th[i]) for i, k in enumerate(cols)}
            with loaded(write_par(full, pathlib.Path(t) / f"p{p}.par")):
                first = RNA.fold_compound(s0[changed[p][0][0]][0], md)
                bad = params_match(first, full, 0, 0)
                if bad:
                    raise GuardFailure(f"point {p}: params_match {bad}")
                for r, want in m["carry"]:
                    _, s, d = gen[r]
                    got = int(round(RNA.fold_compound(s, md).eval_structure(d) * 100))
                    if got != want:
                        raise GuardFailure(f"point {p}: carry row {r}: {got} != {want}")
                ties, reorders, mism = [], [], 0
                for i, genuine_shard in changed[p]:
                    seq, a0, e0, (i0, v0), off0 = s0[i]
                    s1, e1 = RNA.fold_compound(seq, md).mfe()
                    if s1 == a0:
                        raise GuardFailure(f"point {p}: sequence {i} did not change on refold")
                    c1 = ex.counts(seq, s1)
                    off1 = c1.pop(("_offset",), 0)
                    i1, v1 = _vec(c1, idx)
                    if int(v1 @ th[i1]) + off1 != int(round(e1 * 100)):
                        raise GuardFailure(f"point {p}: sequence {i}: MFE energy identity")
                    d0 = int(v1 @ t0[i1]) + off1 - e0
                    if d0 < 0:
                        raise GuardFailure(f"point {p}: s1 below Turner 2004's MFE ({d0})")
                    dx = float(v1 @ (t0[i1] + dl[i1]) + off1 - (v0 @ (t0[i0] + dl[i0]) + off0))
                    genuine = dx <= EXACT_TIE
                    mism += genuine != genuine_shard
                    (ties if d0 == 0 else reorders).append(
                        {"i": i, "dE0_dcal": d0, "dE_exact_dcal": dx})
            if mism:
                raise GuardFailure(f"point {p}: genuine classification differs on {mism}")
            n = len(base)
            v = np.zeros(n)
            v[[x["i"] for x in reorders]] = 1
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 9, p]))
            b = v[rng.integers(0, n, (BOOT, n))].mean(axis=1)
            d0s = np.array([x["dE0_dcal"] for x in reorders])
            res[m["name"]] = {
                "norm_kcal": m["norm_kcal"], "n_sequences": n, "n_changed": len(changed[p]),
                "n_tie_flips": len(ties), "n_real_reorderings": len(reorders),
                "real_reordering_fraction": float(v.mean()),
                "real_reordering_ci95": [float(np.percentile(b, 2.5)),
                                         float(np.percentile(b, 97.5))],
                "changed_fraction": len(changed[p]) / n,
                "tie_flips_with_exact_perturbed_tie": sum(abs(x["dE_exact_dcal"]) <= EXACT_TIE
                                                          for x in ties),
                "reordering_dE0_kcal": {"median": float(np.median(d0s)) / 100,
                                        "max": float(d0s.max()) / 100} if len(d0s) else None,
                "genuine_classification_reproduces_shards": True}
            out["published_points"] = res
            write(out)
    out["status"] = "complete"
    write(out)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"preregister": preregister, "run": run}[ph]())
