#!/usr/bin/env python3
"""CERTIFIED FOLDING, THE EXPLORATORY ANALYSES BESIDE REGISTRATION 1. Not registered tests.

The specification is docs/certfold_exploratory.md, committed (5b6f109) before any test-set
accuracy existed. This module implements it and nothing else, and its record says it is
exploratory. Why: rho falls steeply with length and MFE accuracy tends to fall with length too,
so R2 can pass on length alone.

  E1  R2 within the ten TR0 length strata (TS0): each stratum's own lowest rho_env quartile
      against its rest, pooled weighted by the stratum's pair count.
  E2  D1 within strata (TS0): the p >= 0.99 pairs the envelope at lambda = 1 does not certify,
      against those it does.
  E3  a length score, -n_nt, in the matched-coverage comparison on all three sets, beside p and
      the nominal gap.

E1 and E2 take a bootstrap that resamples sequences WITHIN strata (seed tag 41), which the frozen
certfold_metrics.paired_bootstrap cannot do, so it is written here; every PPV inside it comes
from the frozen metrics through certfold_run's statistics. E3 uses the frozen
matched_coverage_comparison and paired_bootstrap (seed tag 40, R3's draws) unchanged.

Run, after `certfold_run.py --phase analyse` (this file must be committed before that runs):
  .venv-certfold/bin/python scripts/certfold_exploratory.py --phase exploratory   # then commit
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold_run as cr  # noqa: E402
from scripts.certfold_metrics import (  # noqa: E402
    BOOT_LEVEL,
    P_HI,
    SEED,
    disagreement_sets,
    paired_bootstrap,
    pooled_ppv,
)
from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402

cf = cr.cf
ME = "scripts/certfold_exploratory.py"
SPEC = "docs/certfold_exploratory.md"
STRATIFIED_TAG = 41              # the specification's seed tag for E1 and E2
N_BOOT = 10000


# ----------------------------------------------------------------------------------
# E1 and E2: one stratum's statistic, and its weight
# ----------------------------------------------------------------------------------
def e1_stat(recs: list[dict]) -> float:
    """R2's statistic on one stratum's records: certfold_run.stat_r2_quartile unchanged."""
    return cr.stat_r2_quartile(recs)


def e1_weight(recs: list[dict]) -> int:
    return len(cr._r2_pool(recs))


def _hi(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r["kind"] == "mfe" and r["p"] >= P_HI]


def e2_stat(recs: list[dict]) -> float:
    """PPV(D1) - PPV(certified), among the stratum's MFE pairs with p >= 0.99. D1 is the frozen
    disagreement_sets's; the certified side is its complement within p >= 0.99."""
    hi = _hi(recs)
    d1, _d2 = disagreement_sets([r for r in recs if r["kind"] == "mfe"])
    return pooled_ppv(d1) - pooled_ppv(hi, [r["certified_l1"] for r in hi])


def e2_weight(recs: list[dict]) -> int:
    return len(_hi(recs))


STRATIFIED = {"E1_r2_within_strata": (e1_stat, e1_weight),
              "E2_d1_within_strata": (e2_stat, e2_weight)}


def _pooled(vals: dict, weights: dict) -> float:
    w = sum(weights[s] for s in vals)
    return sum(weights[s] * vals[s] for s in vals) / w


def _try(fn, recs):
    try:
        return fn(recs), None
    except GuardFailure as e:
        return None, str(e)


def stratified_bootstrap(groups: dict[int, dict], n_boot: int = N_BOOT,
                         seed_tag: int = STRATIFIED_TAG) -> dict:
    """E1 and E2 per stratum and pooled, with intervals from resampling sequences within strata.

    `groups` is {stratum: {unit: [records]}}, every sequence of the set a unit of its own
    stratum (the 'unit' records make a sequence with no pair in a pool a unit all the same).
    Each resample draws, stratum by stratum in order and units sorted, len(units) units with
    replacement: one rng.integers call per stratum, from default_rng(SeedSequence([SEED, tag])).
    Every stratum's statistic and the pooled value, weighted by the RESAMPLE's own pair counts,
    are recomputed. The pooled value runs over the strata where the statistic is defined on the
    full data; a resample on which any of them is undefined leaves the pooled interval undefined
    (counted), never dropped.
    """
    strata = sorted(groups)
    units = {s: sorted(groups[s]) for s in strata}
    full = {s: [r for u in units[s] for r in groups[s][u]] for s in strata}
    out: dict = {}
    for name, (stat, weight) in STRATIFIED.items():
        per = {}
        for s in strata:
            v, why = _try(stat, full[s])
            per[s] = {"n_pairs": weight(full[s]), "n_sequences": len(units[s]), "point": v,
                      **({"undefined_because": why} if why else {})}
        defined = [s for s in strata if per[s]["point"] is not None]
        out[name] = {"by_stratum": per, "strata_pooled": defined,
                     "strata_left_out": [s for s in strata if s not in defined],
                     "pooled_point": _pooled({s: per[s]["point"] for s in defined},
                                             {s: per[s]["n_pairs"] for s in defined})
                     if defined else None}
    rng = np.random.default_rng(np.random.SeedSequence([SEED, seed_tag]))
    vals = {name: {s: [] for s in strata} for name in STRATIFIED}
    pooled = {name: [] for name in STRATIFIED}
    undefined = {name: {s: 0 for s in strata} for name in STRATIFIED}
    pooled_undefined = {name: 0 for name in STRATIFIED}
    for _b in range(n_boot):
        sub = {}
        for s in strata:
            idx = rng.integers(0, len(units[s]), size=len(units[s]))
            sub[s] = [r for x in idx.tolist() for r in groups[s][units[s][x]]]
        for name, (stat, weight) in STRATIFIED.items():
            got, w = {}, {}
            for s in strata:
                v, _why = _try(stat, sub[s])
                if v is None:
                    undefined[name][s] += 1
                    continue
                vals[name][s].append(v)
                got[s], w[s] = v, weight(sub[s])
            want = out[name]["strata_pooled"]
            if want and all(s in got for s in want):
                pooled[name].append(_pooled({s: got[s] for s in want}, w))
            else:
                pooled_undefined[name] += 1
    half = (100.0 - BOOT_LEVEL) / 2

    def interval(xs, n_undef):
        if n_undef or not xs:
            return None
        lo, hi = np.percentile(xs, [half, 100.0 - half])
        return [float(lo), float(hi)]

    for name in STRATIFIED:
        for s in strata:
            out[name]["by_stratum"][s]["interval"] = interval(vals[name][s], undefined[name][s])
            out[name]["by_stratum"][s]["resamples_undefined"] = undefined[name][s]
        out[name]["pooled_interval"] = interval(pooled[name], pooled_undefined[name])
        out[name]["pooled_resamples_undefined"] = pooled_undefined[name]
        iv = out[name]["pooled_interval"]
        out[name]["pooled_interval_entirely_below_0"] = iv is not None and iv[1] < 0
    out["bootstrap"] = {"n_boot": n_boot, "seed": [SEED, seed_tag], "level": BOOT_LEVEL,
                        "rule": "within each stratum, its sequences drawn with replacement, as "
                                "many as it holds; strata in order, units sorted; pooled "
                                "weights from the resample's own pair counts"}
    return out


def strata_groups(recs: list[dict], stratum_of: dict[int, int]) -> dict[int, dict]:
    """{stratum: {unit: records}} for one set's records, the stratum looked up by index."""
    groups: dict[int, dict] = {}
    for r in recs:
        groups.setdefault(stratum_of[r["index"]], {}).setdefault(r["unit"], []).append(r)
    return groups


# ----------------------------------------------------------------------------------
# E3: the length score in the matched-coverage comparison
# ----------------------------------------------------------------------------------
def with_length(recs: list[dict], n_nt: dict[int, int]) -> list[dict]:
    """The records, each MFE pair given neg_len = -n_nt of its sequence (copies; the inputs are
    left alone)."""
    return [{**r, "neg_len": -float(n_nt[r["index"]])} if r["kind"] == "mfe" else r
            for r in recs]


def e3(recs: list[dict], fam: bool, n_boot: int = N_BOOT) -> dict:
    """rho_env against -n_nt at the p >= 0.99 coverage, as R3 compares rho_env against p:
    the same frozen comparison, pool, weighting and paired bootstrap (seed tag 40)."""
    def point():
        return cr._matched_point(cr._mfe(recs), "neg_len", "tiebreak", fam)

    def stat(rs):
        return cr.stat_matched(rs, "neg_len", "tiebreak", fam)

    got = cr._defined(point, [], None)
    if "undefined_because" in got:
        return got
    sub = cr.task_records(recs, "matched", ("neg_len", "tiebreak", fam))
    try:
        p, lo, hi = paired_bootstrap(sub, stat, n_boot=n_boot)
    except GuardFailure as e:
        return {**got, "interval": None, "interval_undefined_because": str(e)}
    if p != got["difference"]:
        raise GuardFailure(f"E3: the bootstrap's point {p} is not the reported "
                           f"{got['difference']}")
    return {**got, "interval": [lo, hi], "score": "-n_nt", "baseline_tiebreak": "tiebreak"}


# ----------------------------------------------------------------------------------
# the gated phase
# ----------------------------------------------------------------------------------
def _commit_of(path: str) -> str:
    return cf._git("log", "-1", "--format=%H", "--", path).stdout.strip()


def exploratory() -> int:
    """E1 to E3 from the committed run, after the committed analysis, on code committed before
    the analysis ran."""
    out = cf.read_out()
    head = cf._committed(unmodified=True)
    if "analysis" not in head or "run" not in head:
        raise SystemExit("the exploratory analyses follow the committed run and analysis")
    if "exploratory" in head:
        raise SystemExit("the exploratory record exists")
    if not cf._pred_ok(out):
        raise SystemExit("PRED changed since the registration")
    prov = git_provenance(R)
    if prov["git_dirty_excluding_runs"]:
        raise SystemExit(f"commit first: {prov.get('git_dirty_paths_outside_runs')}")
    analysis_sha = head["analysis"]["provenance"]["git_sha"]
    for path in (ME, SPEC):
        c = _commit_of(path)
        if not c or cf._git("merge-base", "--is-ancestor", c, analysis_sha).returncode != 0:
            raise SystemExit(f"{path} was not committed before the analysis ran ({analysis_sha})")
    run_rec = head["run"]
    for f, h in run_rec["files_sha256"].items():
        if cf._sha(cf.SHARDS / f) != h:
            raise GuardFailure(f"{f} changed since the run")
    t0 = time.time()
    seq_recs, freqs = cr.read_shards(run_rec["workers"])
    freq_by_index = {int(k): v for w in freqs.values() for k, v in w["by_index"].items()}
    recs = cr.pair_records(seq_recs, freq_by_index)
    stratum_of = {s["index"]: s["stratum"] for s in seq_recs}
    n_nt = {s["index"]: s["n_nt"] for s in seq_recs}
    res = {
        "label": "EXPLORATORY, not registered tests (docs/certfold_exploratory.md)",
        "spec": {"path": SPEC, "commit": _commit_of(SPEC),
                 "blob": cf._git("rev-parse", f"HEAD:{SPEC}").stdout.strip(),
                 "timing": "committed after the run's folds (launched 01:39 2026-10-14, before "
                           "the authors' instruction to commit it first arrived) and before "
                           "the analysis phase: no test-set accuracy existed"},
        "module_commit": _commit_of(ME), "analysis_commit": analysis_sha,
        "TS0": strat_res(recs["TS0"], stratum_of),
        "E3_length_baseline": {t: e3(with_length(recs[t], n_nt), t == "held-out")
                               for t in cr.SETS if recs[t]},
        "wording": "the held-out set's references are Rfam consensus structures projected onto "
                   "each sequence, not structures determined per molecule; TS0 is the primary "
                   "accuracy set and R2 runs on TS0",
        "provenance": prov, "seconds": None,
    }
    res["seconds"] = round(time.time() - t0, 1)
    out = cf.read_out()
    out["exploratory"] = res
    cf.write(out)
    print(json.dumps({k: res[k] for k in ("label", "spec")}, indent=1))
    return 0


def strat_res(recs: list[dict], stratum_of: dict[int, int]) -> dict:
    return stratified_bootstrap(strata_groups(recs, stratum_of))


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"exploratory": exploratory}[ph]())
