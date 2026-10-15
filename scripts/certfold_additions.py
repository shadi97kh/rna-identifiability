#!/usr/bin/env python3
"""CERTIFIED FOLDING, DESCRIPTIVE ADDITIONS (the authors, 2026-10-15). Not registered tests.

A  uncertain-change   Among TS0 MFE pairs UNCERTAIN under the envelope at lambda = 1, the fraction
                      whose pair frequency under the run's 200 random envelope tables (seed tag
                      30) is below 1: the pair left the MFE under at least one table of the set.
                      A lower bound on how many uncertain pairs actually change somewhere in the
                      set, by length stratum, beside G4's relaxation loss. From existing records:
                      no new fold. Also checks that no CERTIFIED pair has frequency below 1.
B  runtime            From the run record, per-sequence label and rho seconds against length; and
                      one new timing: a plain MFE fold against one fold with the compiled callback
                      registered at lambda = 1, on 30 seeded TR0 sequences spread over the length
                      range (3 per pilot stratum, seed tag 70), median ratio. Run on an idle box.

Both write into runs/2026-10-13_certified_folding/metrics.json under "additions".
Run (.venv-certfold): python scripts/certfold_additions.py --phase uncertain-change | runtime
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold_run as cr  # noqa: E402
from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402

cf = cr.cf
ME = "scripts/certfold_additions.py"
RUNTIME_SEED_TAG = 70            # the 30 timing sequences: SeedSequence([SEED, 70])
RUNTIME_PER_STRATUM = 3
RUNTIME_REPS = 5


def _gate() -> tuple[dict, dict]:
    """The committed run, unmodified; a clean tree; the shards as the run recorded them."""
    head = cf._committed(unmodified=True)
    if "run" not in head:
        raise SystemExit("the additions read the committed run record")
    prov = git_provenance(R)
    if prov["git_dirty_excluding_runs"]:
        raise SystemExit(f"commit first: {prov.get('git_dirty_paths_outside_runs')}")
    for f, h in head["run"]["files_sha256"].items():
        if cf._sha(cf.SHARDS / f) != h:
            raise GuardFailure(f"{f} changed since the run")
    return head, prov


def _write(key: str, rec: dict) -> None:
    out = cf.read_out()
    out.setdefault("additions", {})[key] = rec
    cf.write(out)


def uncertain_change() -> int:
    head, prov = _gate()
    seq_recs, freqs = cr.read_shards(head["run"]["workers"])
    fbi = {int(k): v for w in freqs.values() for k, v in w["by_index"].items()}
    tally = collections.defaultdict(lambda: [0, 0])       # stratum -> [uncertain, changed]
    small = [0, 0]                                         # n <= 150, as G4
    certified_changed, certified = 0, 0
    for s in seq_recs:
        if s["set"] != "TS0":
            continue
        fe = {(i, j): f for i, j, f in fbi[s["index"]]["env"]}
        for q in s["pairs"]:
            f = fe[tuple(q["pair"])]
            if q["margins_dcal"]["envelope:1"] > 0:
                certified += 1
                certified_changed += f < 1.0
                continue
            t = tally[s["stratum"]]
            t[0] += 1
            t[1] += f < 1.0
            if s["n_nt"] <= 150:
                small[0] += 1
                small[1] += f < 1.0
    by = {str(k): {"uncertain": v[0], "changed_in_a_draw": v[1],
                   "fraction": v[1] / v[0] if v[0] else None} for k, v in sorted(tally.items())}
    n_u = sum(v[0] for v in tally.values())
    n_c = sum(v[1] for v in tally.values())
    g4 = head["gate"]["G4"]["by_radius"]
    rec = {
        "label": "DESCRIPTIVE: uncertain is not unstable",
        "what": "TS0 MFE pairs uncertain under the envelope at lambda = 1; 'changed' = pair "
                "frequency below 1 under the run's 200 envelope draws (seed tag 30, inside E_1, "
                "tables written at 0.01 kcal/mol): the pair left the MFE under at least one "
                "random table of the set. A LOWER bound on the uncertain pairs that change "
                "somewhere in the set: 200 random tables need not find a change that exists.",
        "pooled": {"uncertain": n_u, "changed_in_a_draw": n_c, "fraction": n_c / n_u},
        "by_stratum": by,
        "n_le_150": {"uncertain": small[0], "changed_in_a_draw": small[1],
                     "fraction": small[1] / small[0] if small[0] else None},
        "certified_pairs_with_frequency_below_1": certified_changed,
        "certified_pairs": certified,
        "beside_G4_relaxation_loss": {k: {"robust_not_certified": v["relaxation_loss_present"],
                                          "robust": v["n_robust_present"]}
                                      for k, v in g4.items()},
        "provenance": prov,
    }
    if certified_changed:
        rec["note"] = ("a certified pair left the MFE under a drawn table: the drawn tables are "
                       "rounded to 0.01 kcal/mol and may sit just outside E_1; investigate")
    _write("A_uncertain_change", rec)
    print(json.dumps({k: rec[k] for k in ("pooled", "n_le_150",
                                         "certified_pairs_with_frequency_below_1")}, indent=1))
    return 0


def _timing_rows() -> list[tuple[str, str, int]]:
    """3 TR0 sequences per pilot stratum, seeded: (name, seq, stratum)."""
    import scripts.certfold_data as cd

    rows, _ = cf.tr0_rows()
    bounds = cd.TR0_STRATA
    rng = np.random.default_rng(np.random.SeedSequence([cf.SEED, RUNTIME_SEED_TAG]))
    out = []
    for k in range(len(bounds)):
        pool = sorted((r for r in rows if cd.length_stratum(len(r[2]), bounds) == k),
                      key=lambda r: (len(r[2]), r[1]))
        for i in sorted(rng.choice(len(pool), size=RUNTIME_PER_STRATUM, replace=False).tolist()):
            out.append((pool[i][1], pool[i][2], k))
    return out


def runtime() -> int:
    head, prov = _gate()
    if cf.callback_backend() != "compiled":
        raise SystemExit("time the compiled callback, in .venv-certfold")
    load = os.getloadavg()
    seq_recs, _ = cr.read_shards(head["run"]["workers"])
    by = collections.defaultdict(list)
    for s in seq_recs:
        by[s["stratum"]].append(s)
    from_run = {}
    for k, ss in sorted(by.items()):
        from_run[str(k)] = {
            "n_sequences": len(ss),
            "length_median": float(np.median([s["n_nt"] for s in ss])),
            "labels_seconds_median": float(np.median([s["seconds_labels"] for s in ss])),
            "rho_seconds_median": float(np.median([s["seconds_rho"] for s in ss])),
            "labels_seconds_per_pair_median": float(np.median(
                [s["seconds_labels"] / max(1, s["n_pairs_star"]) for s in ss])),
        }
    B = cf.load_tr0_basis()
    P = cf.template()
    lc = cf.LoopCounts(P["special"], 0, 0, lxc=P["lxc"])
    timings = []
    import RNA

    from thermo.vienna_ref import reference_md
    for name, seq, k in _timing_rows():
        plain = RNA.fold_compound(seq, reference_md())
        star = plain.mfe()[0]
        F = cf.Folder(lc, seq, star, cf.Weights(B["u_env"], B["cols"], 1.0))
        F.set_radius(1.0)
        F.fc.hc_init()
        tp, tc = [], []
        for _ in range(RUNTIME_REPS):
            t = time.perf_counter()
            plain.mfe()
            tp.append(time.perf_counter() - t)
            t = time.perf_counter()
            F.fc.mfe()
            tc.append(time.perf_counter() - t)
        mp, mc = float(np.median(tp)), float(np.median(tc))
        timings.append({"name": name, "n_nt": len(seq), "stratum": k, "plain_s": mp,
                        "callback_s": mc, "ratio": mc / mp})
    ratios = [t["ratio"] for t in timings]
    rec = {
        "label": "DESCRIPTIVE: runtime",
        "from_the_run": {"by_stratum": from_run,
                         "what": "per sequence, all seven label sets, and both radii (budget "
                                 "1800 s), 8 workers on 8 physical cores (load 8.0-8.2)"},
        "callback_vs_plain": {
            "n_sequences": len(timings), "seed": [cf.SEED, RUNTIME_SEED_TAG],
            "reps": RUNTIME_REPS, "median_ratio": float(np.median(ratios)),
            "ratio_range": [float(min(ratios)), float(max(ratios))],
            "what": "median over 5 repetitions each of one plain MFE fold and one MFE fold with "
                    "the compiled callback registered at the envelope's lambda = 1, the same "
                    "sequence, compounds built outside the timed region",
            "sequences": timings},
        "load_at_start": [round(x, 2) for x in load],
        "provenance": prov,
    }
    _write("B_runtime", rec)
    print(json.dumps({"median_ratio": rec["callback_vs_plain"]["median_ratio"],
                      "range": rec["callback_vs_plain"]["ratio_range"],
                      "load": rec["load_at_start"]}, indent=1))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"uncertain-change": uncertain_change, "runtime": runtime}[ph]())
