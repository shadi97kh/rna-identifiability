#!/usr/bin/env python3
"""E3: is 4.3% a lot? TR0+TS0 folded under the WHOLE published tables; and the DNA check.

Descriptive, no prediction (the user's E3). Folding TR0+TS0 under Turner 1999, Andronescu 2007
and Langdon 2018 in full (ViennaRNA's shipped sets, loaded by params_load_RNA_*) gives the
changed-MFE fraction against Turner 2004 and the paired change in F1 against the references.
Set beside the Type B points (each table's nontrivial component alone), it shows how much of
the disagreement between two published tables comes from directions the references cannot see.
The held-out set is folded the same way inside E2 (scripts/typeb_heldout.py).

Guards (scripts/typeb_common.fold_rows_under_tables): for each table, a rebuild of ViennaRNA's
caches, params_match against the dumped table, carry rows evaluated with that table, the
reference-energy identity (counts with the table's lxc) and the MFE-energy identity on every
sequence. A reference using an entry the table forbids (INF) is counted, not compared.

DNA CHECK. ViennaRNA's DNA set (params_load_DNA_Mathews2004), dumped and parsed like the RNA
tables: are all 12,607 reachable columns present and finite? With it loaded, does
eval_structure equal counts . theta_DNA exactly on the 23,741 constructed structures at d0,
special_hp 0? If so, the loops use the same lookups, and the intrinsic directions N1 and N2
(defined by counts alone) are null for DNA too.

Run: .venv/bin/python scripts/typeb_full_tables.py --phase {run,analyse,dna}
"""

from __future__ import annotations

import hashlib
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

RUN_ID = "2026-10-12_typeb_full_tables"
OUT = R / "runs" / RUN_ID
SHARDS = OUT / "shards"
N_WORKERS = 8
SEED = 20261012
BOOT = 2000


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def read_out():
    p = OUT / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {"run_id": RUN_ID}


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def worker(w, W, dst):
    from scripts.turner_typeb import fold_rows
    from scripts.typeb_common import fold_rows_under_tables, full_tables
    return fold_rows_under_tables(fold_rows(), full_tables(), dst, w, W)


def run():
    out = read_out()
    if str(out.get("status", "")).startswith("FAILED"):
        out.setdefault("failed_attempts", []).append({"status": out["status"], **out["run"]})
    out.update({"status": f"{RUNNING_PREFIX}: E3 full-table folds",
                "label": "DESCRIPTIVE (E3, the user's; no prediction)",
                "run": {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "provenance": git_provenance(R)}})
    write(out)
    SHARDS.mkdir(parents=True, exist_ok=True)
    dsts = [SHARDS / f"shard{w}.jsonl" for w in range(N_WORKERS)]
    env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    rc = [p.wait() for p in [subprocess.Popen([sys.executable, __file__, "--phase", "worker",
                                               str(w), str(N_WORKERS), str(d)], env=env)
                             for w, d in enumerate(dsts)]]
    done = [d.exists() and d.read_text().rstrip().endswith('{"done": true}') for d in dsts]
    out = read_out()
    out["run"].update({"workers_exit": rc, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "shards_sha256": {d.name: _sha(d) for d in dsts if d.exists()}})
    out["status"] = "folded" if rc == [0] * N_WORKERS and all(done) else f"FAILED: {rc}"
    write(out)
    return 0 if out["status"] == "folded" else 1


def analyse():
    out = read_out()
    assert out["status"] in ("folded", "analysed"), out["status"]
    for f, h in out["run"]["shards_sha256"].items():
        assert _sha(SHARDS / f) == h, f"{f} changed"
    sh = [[json.loads(ln) for ln in (SHARDS / f"shard{w}.jsonl").read_text().splitlines()]
          for w in range(N_WORKERS)]
    base = [b for s in sh for b in s[0]["baseline"]]
    f0 = np.array([b["f1_ref0"] for b in base])
    n = len(base)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 30]))
    bi = rng.integers(0, n, (BOOT, n))
    res = {}
    tables = [r["table"] for r in sh[0][1:] if "table" in r]
    for t in tables:
        rs = [next(r for r in s if r.get("table") == t) for s in sh]
        ch = np.array([x for r in rs for x in r["changed"]], dtype=float)
        f1 = np.array([x for r in rs for x in r["f1_ref"]])
        dref = np.array([np.nan if x is None else x for r in rs for x in r["dref_dcal"]]) / 100
        d = f1 - f0
        res[t] = {"changed_fraction": float(ch.mean()),
                  "changed_ci95": [float(x) for x in np.percentile(ch[bi].mean(1), [2.5, 97.5])],
                  "paired_dF1": float(d.mean()),
                  "paired_dF1_ci95": [float(x) for x in np.percentile(d[bi].mean(1), [2.5, 97.5])],
                  "f1_ref_mean": float(f1.mean()), "f1_ref0_mean": float(f0.mean()),
                  "rms_reference_energy_change_kcal": float(np.sqrt(np.nanmean(dref ** 2))),
                  "n_reference_forbidden": int(sum(r["n_reference_forbidden"] for r in rs))}
    out["results"] = {"n_sequences": n, "per_table": res, "provenance": git_provenance(R)}
    out["status"] = "analysed"
    write(out)
    print(json.dumps(res, indent=1))
    return 0


def dna():
    """DNA check, two parts. (1) Lookup structure: with the DNA set loaded, is eval_structure
    counts . theta_DNA on the constructed structures? Mismatches are classified. The count model
    linearises the Ninio term MIN2(max, asym*m) at Turner 2004's values (cap from asymmetry 5),
    and DNA's slope differs, so internal loops of large asymmetry are expected to differ there.
    (2) The gauge itself, value-independently: write DNA's table and DNA's table + 7*N1 (and
    + 5*N2) through the verified writer (DNA's values on the reachable columns, at d0 and
    special_hp 0 where nothing else is read), load each, and require identical energies on every
    constructed structure."""
    import collections
    import tempfile

    import RNA

    from scripts.turner_nullspace import (
        Extractor,
        _reachable,
        analytic_directions,
        build_matrix,
        generating_set,
        theta_from_par,
    )
    from scripts.turner_par import _rebuild, loaded, template, write_par
    from thermo.vienna_ref import assert_turner2004_params, reference_md

    T = template()
    cols = [k for k in sorted(T["theta"], key=repr) if _reachable(k, 0, 0)]
    RNA.params_load_DNA_Mathews2004()
    _rebuild()
    with tempfile.TemporaryDirectory() as t:
        p = pathlib.Path(t) / "dna.par"
        RNA.params_save(str(p))
        P, h = theta_from_par(p), _sha(p)
    th = P["theta"]
    missing = [k for k in cols if k not in th]
    inf = [k for k in cols if k in th and th[k] >= 10_000_000]
    gen = generating_set(T["special"])
    ex = Extractor(T["special"], 0, 0)
    md = reference_md()
    cnts = [ex.counts(s, d) for _, s, d in gen]
    loaded_ok = RNA.fold_compound(gen[0][1], md).params.stack[2][2] == th[("stack", 2, 2)]
    mism = collections.Counter()
    by_tag = collections.Counter()
    for (tag, s, d), c in zip(gen, cnts, strict=True):
        got = int(round(RNA.fold_compound(s, md).eval_structure(d) * 100))
        diff = got - sum(th[k] * v for k, v in c.items())
        if diff:
            mism[diff] += 1
            by_tag[(tag, c.get(("ninio_max",), 0) > 0)] += 1
    RNA.params_load_RNA_Turner2004()
    _rebuild()
    # (2) the gauge on DNA's values, through the writer and ViennaRNA's own evaluation
    an = analytic_directions(cols, 0, 0)
    base = dict(T["theta"]) | {k: th[k] for k in cols}
    gauge = {}
    with tempfile.TemporaryDirectory() as t:
        energies = {}
        for name, eps, vec in (("base", 0, None),
                               ("N1", 7, an["N1_hairpin_mismatch_vs_init"]),
                               ("N2", 5, an["N2_interior_mismatch_vs_init"])):
            full = dict(base)
            if vec is not None:
                for i, k in enumerate(cols):
                    if vec[i]:
                        full[k] = int(base[k] + eps * vec[i])
            with loaded(write_par(full, pathlib.Path(t) / f"{name}.par")):
                assert RNA.fold_compound(gen[0][1], md).params.stack[2][2] == th[("stack", 2, 2)]
                energies[name] = [int(round(RNA.fold_compound(s, md).eval_structure(d) * 100))
                                  for _, s, d in gen]
        for name in ("N1", "N2"):
            nd = sum(a != b for a, b in zip(energies[name], energies["base"], strict=True))
            gauge[name] = {"rows": len(gen), "rows_changed": nd, "null_for_dna": nd == 0}
    G = build_matrix(cnts, cols)
    vec_ok = {n: bool(np.abs(G @ an[k]).max() == 0) for n, k in
              (("N1", "N1_hairpin_mismatch_vs_init"), ("N2", "N2_interior_mismatch_vs_init"))}
    assert_turner2004_params(RNA.fold_compound("GGGGAAACCCC", reference_md()))
    out = read_out()
    out["dna_check"] = {
        "loader": "params_load_DNA_Mathews2004", "par_sha256": h,
        "reachable_columns": len(cols), "reachable_missing": [repr(k) for k in missing],
        "reachable_inf": len(inf), "table_in_effect": bool(loaded_ok),
        "ninio_dna": [th[("ninio_m",)], th[("ninio_max",)]],
        "ninio_rna": [T["theta"][("ninio_m",)], T["theta"][("ninio_max",)]],
        "lookup_check": {"rows": len(gen), "rows_mismatched": sum(mism.values()),
                         "mismatch_values_dcal": {str(k): v for k, v in mism.items()},
                         "mismatched_by_construction_and_ninio_max_counted": {
                             f"{a}|ninio_max={b}": v for (a, b), v in by_tag.items()},
                         "reading": "every mismatch is an internal loop whose counts take the "
                                    "Ninio cap: the count model linearises MIN2(max, asym*m) at "
                                    "Turner 2004's m; DNA's smaller m moves the cap to a larger "
                                    "asymmetry. Not a different lookup structure."},
        "gauge_check": gauge, "counts_null_on_generating_set": vec_ok,
        "intrinsic_directions_null_for_dna": all(g["null_for_dna"] for g in gauge.values()),
        "provenance": git_provenance(R)}
    write(out)
    print(json.dumps({k: v for k, v in out["dna_check"].items() if k != "provenance"},
                     indent=1))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    if ph == "worker":
        a = sys.argv[sys.argv.index("worker") + 1:]
        sys.exit(worker(int(a[0]), int(a[1]), a[2]))
    sys.exit({"run": run, "analyse": analyse, "dna": dna}[ph]())
