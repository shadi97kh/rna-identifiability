#!/usr/bin/env python3
"""The Type B dimension against the number of TR0+TS0 structures, and the union with Rfam 15.1.
POST HOC, as specified before any code in docs/advisor_additions_2026-10-20.md (item 2). Linear
algebra only; nothing is folded.

The function and tolerances are the registered runs' own, imported unchanged:
scripts/turner_nullspace_rfam.analyse at dangles 0 with special hairpins off, on
theta_from_par(dump_params()), exactly as runs/2026-10-08_turner_nullspace_tr0 (426) and
runs/2026-10-07_turner_nullspace_rfam_r2 (214) called it. Rows as those runs built them.

  subsamples  TR0+TS0's 11,955 distinct pair-bearing rows at 1,000 / 2,000 / 4,000 / 8,000,
              8 draws each: SeedSequence([20261007, 92, size, k]), k = 0..7 (seed tag 92, new);
              analyse's rng SeedSequence([20261007, 92, size, k, 1])
  anchors     the full TR0+TS0 set (must give 426) and the Rfam rows alone (must give 214)
  union       TR0+TS0's rows and the Rfam rows, distinct by (sequence, structure)
Every analysis must have its dimension methods agree, no reconstruction failure and C2 passing.
Records runs/2026-10-20_typeb_dimension_curve/metrics.json, rewritten after every analysis.

Run: OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/typeb_dimension_curve.py
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

from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_nullspace import dump_params, theta_from_par  # noqa: E402
from scripts.turner_nullspace_rfam import analyse, load_rows  # noqa: E402
from scripts.turner_nullspace_tr0 import distinct_rows, summary, tr0_ts0_rows  # noqa: E402

SPEC = "docs/advisor_additions_2026-10-20.md"
OUT = R / "runs" / "2026-10-20_typeb_dimension_curve"
BASE, TAG = 20261007, 92
SIZES = (1_000, 2_000, 4_000, 8_000)
N_DRAWS = 8
CONFIG = (0, 0)                      # dangles 0, special hairpins off
FULL_TR0TS0, FULL_RFAM = 426, 214    # the registered dimensions the anchors must reproduce


class CheckFailed(RuntimeError):
    pass


def _git(*a):
    return subprocess.run(["git", *a], cwd=R, capture_output=True, text=True)


def write(out: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def checked(res: dict, what: str) -> dict:
    s = summary(res)
    if not s["null_dim_within_used"].get("agree"):
        raise CheckFailed(f"{what}: the dimension methods disagree: {s['null_dim_within_used']}")
    if s["reconstruction_failures"]:
        raise CheckFailed(f"{what}: {s['reconstruction_failures']} reconstruction failures")
    if not s["C2_passes"]:
        raise CheckFailed(f"{what}: control C2 fails")
    return s


def main() -> int:
    if _git("ls-files", "--error-unmatch", SPEC).returncode != 0:
        raise SystemExit(f"{SPEC} is not committed")
    if _git("diff", "--quiet", "HEAD", "--", SPEC).returncode != 0:
        raise SystemExit(f"{SPEC} has uncommitted changes")
    prov = git_provenance(R)
    if prov["git_dirty_excluding_runs"]:
        raise SystemExit("commit first: the tree is dirty outside runs/")
    if (OUT / "metrics.json").exists():
        raise SystemExit(f"{OUT.relative_to(R)}/metrics.json exists; move it aside to re-run")
    t0 = time.time()
    out = {"run_id": OUT.name, "label": "POST HOC (docs/advisor_additions_2026-10-20.md, item 2)",
           "status": "RUNNING",
           "specification": {"path": SPEC,
                             "sha256": hashlib.sha256((R / SPEC).read_bytes()).hexdigest(),
                             "commit": _git("log", "-1", "--format=%H", "--", SPEC).stdout.strip()},
           "config": {"dangles": CONFIG[0], "special_hp": CONFIG[1]},
           "seed": {"base": BASE, "tag": TAG, "draws": N_DRAWS}, "provenance": prov}
    write(out)
    P = theta_from_par(dump_params())
    rows, _ = tr0_ts0_rows()
    pop = [r for r in distinct_rows(rows) if "(" in r[3]]
    rf_all, _ = load_rows()
    rf = [r for r in distinct_rows(rf_all) if "(" in r[3]]
    union = [r for r in distinct_rows(pop + rf) if "(" in r[3]]
    out["rows"] = {"tr0ts0_distinct_pair_bearing": len(pop),
                   "rfam_distinct_pair_bearing": len(rf),
                   "union_distinct_pair_bearing": len(union),
                   "shared_between_the_two": len(pop) + len(rf) - len(union)}
    write(out)
    d, s = CONFIG

    out["curve"] = {}
    for size in SIZES:
        out["curve"][str(size)] = {}
        for k in range(N_DRAWS):
            rng = np.random.default_rng(np.random.SeedSequence([BASE, TAG, size, k]))
            idx = np.sort(rng.choice(len(pop), size=size, replace=False))
            res = analyse(d, s, P, [pop[i] for i in idx],
                          np.random.default_rng(np.random.SeedSequence([BASE, TAG, size, k, 1])))
            out["curve"][str(size)][str(k)] = checked(res, f"size {size} draw {k}")
            write(out)
            print(size, k, out["curve"][str(size)][str(k)]["nontrivial_dim"], flush=True)
        dims = [v["nontrivial_dim"] for v in out["curve"][str(size)].values()]
        out["curve"][str(size)]["summary"] = {"dims": dims, "mean": float(np.mean(dims)),
                                              "min": min(dims), "max": max(dims)}
        write(out)

    def anchor(label, rows_, want, tag):
        res = analyse(d, s, P, rows_, np.random.default_rng(np.random.SeedSequence([BASE, TAG,
                                                                                   tag])))
        sm = checked(res, label) | {"rows": len(rows_)}
        if want is not None and sm["nontrivial_dim"] != want:
            raise CheckFailed(f"{label}: {sm['nontrivial_dim']} directions, not {want}")
        out[label] = sm
        write(out)
        print(label, sm["nontrivial_dim"], flush=True)

    anchor("full_tr0ts0", pop, FULL_TR0TS0, 1)
    anchor("rfam_alone", rf, FULL_RFAM, 2)
    anchor("union", union, None, 3)
    out["reading"] = {
        "union_nontrivial_dim": out["union"]["nontrivial_dim"],
        "union_structures": out["union"]["rows"],
        "sentence": (f"Within the data the Type B dimension does not reach 0: "
                     f"{out['union']['nontrivial_dim']} directions remain for the union of "
                     f"{out['union']['rows']:,} structures (post hoc)."),
        "no_asymptote_no_extrapolation": True}
    out["status"] = "done"
    out["seconds"] = round(time.time() - t0, 1)
    write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
