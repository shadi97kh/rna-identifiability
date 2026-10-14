#!/usr/bin/env python3
"""CERTIFIED FOLDING, THE EXPLORATORY ANALYSES BESIDE REGISTRATION 1. Not registered tests.

The specification is docs/certfold_exploratory.md (5b6f109) and the reading rules are
docs/certfold_reading_rules.md (292afc9), both committed before any test-set accuracy existed.
This module implements them and nothing else, and its record says it is exploratory. Why:
rho falls steeply with length and MFE accuracy tends to fall with length too, so R2 can pass on
length alone.

  E1      R2 within the ten TR0 length strata (TS0): each stratum's own lowest rho_env quartile
          against its rest, pooled weighted by the stratum's pair count.
  E2      D1 within strata (TS0): the p >= 0.99 pairs the envelope at lambda = 1 does not
          certify, against those it does.
  E2_raw  the same comparison without strata, on each set (reading rule 4: "raw and within
          strata"), with a paired interval (seed tag 40, the registered D1's draws).
  E3      a length score, -n_nt, in the matched-coverage comparison on all three sets, beside p
          and the nominal gap.

E1 and E2 take a bootstrap that resamples sequences WITHIN strata (seed tag 41), which the frozen
certfold_metrics.paired_bootstrap cannot do, so it is written here; every PPV inside it comes
from the frozen metrics through certfold_run's statistics. E2_raw and E3 use the frozen
paired_bootstrap (seed tag 40) unchanged.

THE REVIEW of 2026-10-14 (the authors asked for a fresh-context review of a6409f9 before this
runs) is folded in: definedness is tested explicitly, so a data-integrity guard stops the run
instead of reading as "undefined"; E2's strata follow E2_MERGE (below); the raw comparison rule 4
asks for exists; the gate ties this code to the code that produced R2 and R3; the record carries
the strata bounds, the capped counts and the module blob of every job.

Run, after `certfold_run.py --phase analyse` is committed (this file, the specification and
the reading rules must be committed strictly before the commit that carries the analysis record):
  .venv-certfold/bin/python scripts/certfold_exploratory.py --phase exploratory   # then commit
"""

from __future__ import annotations

import datetime
import json
import multiprocessing
import os
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
READING_RULES = "docs/certfold_reading_rules.md"
STRATIFIED_TAG = 41              # the specification's seed tag for E1 and E2
N_BOOT = 10000
# E2's reporting strata: {stratum: the stratum it is reported with}. The review counted, from
# labels alone, that only 8 of TS0's 124 stratum-0 sequences hold a D1 pair, so a resample
# draws none of them with probability 2.6e-4 and E2's interval would be undefined in about 92%
# of bootstraps; merged with stratum 1, the chance is about 1e-7. The authors chose the merge on
# 2026-10-14, from those counts alone, before any test-set accuracy was seen. E1 keeps all ten.
E2_MERGE: dict[int, int] = {1: 0}
AMENDMENTS = (
    "E2 reports strata 0 and 1 merged (the authors, 2026-10-14, blind, from D1-holder counts: 8 "
    "of 124 stratum-0 sequences); the resampling stays within the ten TR0 strata",
    "E2_raw added: reading rule 4 asks for D1 against the certified p >= 0.99 pairs 'raw and "
    "within strata', and the specification had only the within-strata form",
    "the fresh-context review of a6409f9 (requested by the authors): explicit definedness, a "
    "gate tying this code to the analysis's, strict ancestry, the capped count per stratum, "
    "the strata bounds and the module blob in the record, MFE-only resampling (no number "
    "changes), 'baseline_score' for E3's length score",
)


# ----------------------------------------------------------------------------------
# E1 and E2: one stratum's statistic, its weight, and when it is defined
# ----------------------------------------------------------------------------------
def e1_stat(recs: list[dict]) -> float:
    """R2's statistic on one stratum's records: certfold_run.stat_r2_quartile unchanged."""
    return cr.stat_r2_quartile(recs)


def e1_weight(recs: list[dict]) -> int:
    return len(cr._r2_pool(recs))


def e1_defined(recs: list[dict]) -> bool:
    """R2's cut needs a 'rest' beside the lowest quartile: at least two pairs."""
    return e1_weight(recs) >= 2


def e1_extra(recs: list[dict]) -> dict:
    pool = cr._r2_pool(recs)
    return {"n_capped": sum(r["capped"] for r in pool)}


def _hi(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r["kind"] == "mfe" and r["p"] >= P_HI]


def _d1_c(recs: list[dict]) -> tuple[list[dict], list[dict]]:
    """D1 (the frozen disagreement_sets's) and C, the certified pairs with p >= 0.99."""
    d1, _d2 = disagreement_sets([r for r in recs if r["kind"] == "mfe"])
    return d1, [r for r in _hi(recs) if r["certified_l1"]]


def e2_stat(recs: list[dict]) -> float:
    """PPV(D1) - PPV(C), pooled over the stratum's pairs."""
    d1, c = _d1_c(recs)
    return pooled_ppv(d1) - pooled_ppv(c)


def e2_weight(recs: list[dict]) -> int:
    return len(_hi(recs))


def e2_defined(recs: list[dict]) -> bool:
    d1, c = _d1_c(recs)
    return bool(d1) and bool(c)


def e2_extra(recs: list[dict]) -> dict:
    d1, c = _d1_c(recs)
    return {"n_d1": len(d1), "n_certified": len(c)}


def _stratified_specs() -> dict:
    return {"E1_r2_within_strata": {"stat": e1_stat, "weight": e1_weight, "defined": e1_defined,
                                    "extra": e1_extra, "merge": {}},
            "E2_d1_within_strata": {"stat": e2_stat, "weight": e2_weight, "defined": e2_defined,
                                    "extra": e2_extra, "merge": dict(E2_MERGE)}}


def _reporting(strata: list[int], merge: dict[int, int]) -> dict[str, list[int]]:
    """{label: member strata}: each stratum alone, or with the stratum `merge` sends it to."""
    out: dict[str, list[int]] = {}
    groups: dict[int, list[int]] = {}
    for s in strata:
        groups.setdefault(merge.get(s, s), []).append(s)
    for members in groups.values():
        out["+".join(str(m) for m in sorted(members))] = sorted(members)
    return dict(sorted(out.items(), key=lambda kv: kv[1][0]))


def _pooled(vals: dict, weights: dict) -> float:
    w = sum(weights[s] for s in vals)
    return sum(weights[s] * vals[s] for s in vals) / w


def stratified_bootstrap(groups: dict[int, dict], n_boot: int = N_BOOT,
                         seed_tag: int = STRATIFIED_TAG, specs: dict | None = None) -> dict:
    """E1 and E2 per reporting stratum and pooled, with intervals from resampling sequences
    within strata.

    `groups` is {stratum: {unit: [records]}}, every sequence of the set a unit of its own
    stratum (the 'unit' records make a sequence with no pair in a pool a unit all the same).
    Each resample draws, stratum by stratum in order and units sorted, len(units) units with
    replacement: one rng.integers call per stratum per resample (resample-major), from
    default_rng(SeedSequence([SEED, tag])). The draws are always made per TR0 stratum, so E1 and
    E2 share them (paired); a statistic whose strata are merged evaluates on the union of its
    members' resamples, which still holds every member's size fixed. Every reporting stratum's
    statistic and the pooled value, weighted by the RESAMPLE's own pair counts, are recomputed.

    DEFINEDNESS is tested explicitly (E1: at least two pairs; E2: D1 and C both non-empty), so
    any GuardFailure a statistic raises is a data-integrity failure and stops the run. The
    pooled value runs over the reporting strata defined on the full data; a resample on which
    any of them is undefined leaves the pooled interval undefined (counted), never dropped.
    """
    specs = _stratified_specs() if specs is None else specs
    strata = sorted(groups)
    units = {s: sorted(groups[s]) for s in strata}
    full = {s: [r for u in units[s] for r in groups[s][u]] for s in strata}
    out: dict = {}
    reps = {}
    for name, sp in specs.items():
        reps[name] = _reporting(strata, sp["merge"])
        per = {}
        for lab, members in reps[name].items():
            rs = [r for m in members for r in full[m]]
            ok = sp["defined"](rs)
            per[lab] = {"members": members, "n_pairs": sp["weight"](rs),
                        "n_sequences": sum(len(units[m]) for m in members),
                        "point": sp["stat"](rs) if ok else None, **sp["extra"](rs)}
            if not ok:
                per[lab]["undefined_because"] = "not defined on the full data"
        defined = [lab for lab in per if per[lab]["point"] is not None]
        out[name] = {"by_stratum": per, "strata_pooled": defined,
                     "strata_left_out": [lab for lab in per if lab not in defined],
                     "pooled_point": _pooled({s: per[s]["point"] for s in defined},
                                             {s: per[s]["n_pairs"] for s in defined})
                     if defined else None}
    rng = np.random.default_rng(np.random.SeedSequence([SEED, seed_tag]))
    vals = {n: {lab: [] for lab in reps[n]} for n in specs}
    pooled = {n: [] for n in specs}
    undefined = {n: {lab: 0 for lab in reps[n]} for n in specs}
    pooled_undefined = {n: 0 for n in specs}
    for _b in range(n_boot):
        sub = {}
        for s in strata:
            idx = rng.integers(0, len(units[s]), size=len(units[s]))
            sub[s] = [r for x in idx.tolist() for r in groups[s][units[s][x]]]
        for name, sp in specs.items():
            got, w = {}, {}
            for lab, members in reps[name].items():
                rs = [r for m in members for r in sub[m]]
                if not sp["defined"](rs):
                    undefined[name][lab] += 1
                    continue
                v = sp["stat"](rs)
                vals[name][lab].append(v)
                got[lab], w[lab] = v, sp["weight"](rs)
            want = out[name]["strata_pooled"]
            if want and all(lab in got for lab in want):
                pooled[name].append(_pooled({lab: got[lab] for lab in want}, w))
            else:
                pooled_undefined[name] += 1
    half = (100.0 - BOOT_LEVEL) / 2

    def interval(xs, n_undef):
        if n_undef or not xs:
            return None
        lo, hi = np.percentile(xs, [half, 100.0 - half])
        return [float(lo), float(hi)]

    for name in specs:
        for lab in reps[name]:
            out[name]["by_stratum"][lab]["interval"] = interval(vals[name][lab],
                                                                undefined[name][lab])
            out[name]["by_stratum"][lab]["resamples_undefined"] = undefined[name][lab]
        out[name]["pooled_interval"] = interval(pooled[name], pooled_undefined[name])
        out[name]["pooled_resamples_undefined"] = pooled_undefined[name]
        iv = out[name]["pooled_interval"]
        out[name]["pooled_interval_entirely_below_0"] = None if iv is None else iv[1] < 0
        out[name]["merge"] = {str(k): v for k, v in specs[name]["merge"].items()}
    out["bootstrap"] = {"n_boot": n_boot, "seed": [SEED, seed_tag], "level": BOOT_LEVEL,
                        "rule": "within each TR0 stratum, its sequences drawn with replacement, "
                                "as many as it holds; strata in order, units sorted, one draw "
                                "per stratum per resample; a merged reporting stratum uses the "
                                "union of its members' draws; pooled weights from the "
                                "resample's own pair counts; the pooled interval is undefined "
                                "if any full-data-defined reporting stratum is undefined on "
                                "some resample (counted, never dropped)"}
    return out


def strata_groups(recs: list[dict], stratum_of: dict[int, int]) -> dict[int, dict]:
    """{stratum: {unit: records}} for one set's unit and MFE records (E1 and E2 read nothing
    else, so the MEA and centroid records are left behind), the stratum looked up by index."""
    groups: dict[int, dict] = {}
    for r in recs:
        if r["kind"] in ("unit", "mfe"):
            groups.setdefault(stratum_of[r["index"]], {}).setdefault(r["unit"], []).append(r)
    return groups


# ----------------------------------------------------------------------------------
# E2_raw and E3: the frozen paired bootstrap, seed tag 40
# ----------------------------------------------------------------------------------
def e2_raw_stat(recs: list[dict], fam: bool) -> float:
    """PPV(D1) - PPV(C) over the whole set; families weighted equally on the held-out set,
    as the registered D1 PPV is there."""
    d1, c = _d1_c(recs)
    return cr._ppv(d1, fam) - cr._ppv(c, fam)


def e2_raw(recs: list[dict], fam: bool, n_boot: int = N_BOOT) -> dict:
    """Reading rule 4's raw comparison, with the registered D1's resampling: the frozen
    paired_bootstrap, seed tag 40, over the unit and MFE records (certfold_run.task_records'
    'disagreement' selection, so the draws are the registered D1 interval's)."""
    d1, c = _d1_c(recs)
    got = {"n_d1": len(d1), "n_certified": len(c),
           "weighting": "families weighted equally" if fam else "pairs pooled"}
    if not (d1 and c):
        return {**got, "undefined_because": "D1 or C is empty on the full data"}
    got.update(ppv_d1=cr._ppv(d1, fam), ppv_certified=cr._ppv(c, fam))
    got["difference"] = got["ppv_d1"] - got["ppv_certified"]
    sub = cr.task_records(recs, "disagreement", ("D1", fam))
    try:
        p, lo, hi = paired_bootstrap(sub, lambda rs: e2_raw_stat(rs, fam), n_boot=n_boot)
    except GuardFailure as e:
        return {**got, "interval": None, "interval_undefined_because": str(e)}
    if p != got["difference"]:
        raise GuardFailure(f"E2_raw: the bootstrap's point {p} is not {got['difference']}")
    return {**got, "interval": [lo, hi], "interval_entirely_below_0": hi < 0}


def with_length(recs: list[dict], n_nt: dict[int, int]) -> list[dict]:
    """The records, each MFE pair given neg_len = -n_nt of its sequence (copies; the inputs are
    left alone)."""
    return [{**r, "neg_len": -float(n_nt[r["index"]])} if r["kind"] == "mfe" else r
            for r in recs]


def e3(recs: list[dict], fam: bool, n_boot: int = N_BOOT) -> dict:
    """rho_env against -n_nt at the p >= 0.99 coverage, as R3 compares rho_env against p:
    the same frozen comparison, pool, weighting and paired bootstrap (seed tag 40). In the
    record, 'score' fields are rho_env's and 'baseline' fields the length score's."""
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
    return {**got, "interval": [lo, hi], "baseline_score": "-n_nt",
            "baseline_tiebreak": "tiebreak"}


# ----------------------------------------------------------------------------------
# the gated phase
# ----------------------------------------------------------------------------------
WORKERS = 7                      # one per job: E1+E2 on TS0, E3 and E2_raw on each set
_JOB: dict = {}                  # set in the parent before the pool forks


def _commit_of(path: str) -> str:
    return cf._git("log", "-1", "--format=%H", "--", path).stdout.strip()


def _commit_time(sha: str) -> str:
    return cf._git("show", "-s", "--format=%cI", sha).stdout.strip()


def _analysis_commit() -> str:
    """The first commit whose metrics.json carries the analysis record (parsed, not grepped):
    the moment test-set accuracy entered the repository."""
    rel = str((cf.OUT / "metrics.json").relative_to(R))
    for c in cf._git("log", "--reverse", "--full-history", "--format=%H", "--", rel).stdout.split():
        if "analysis" in json.loads(cf._git("show", f"{c}:{rel}").stdout):
            return c
    raise SystemExit("no commit carries the analysis record")


def _strictly_before(c: str, a: str) -> bool:
    """c is an ancestor of a's first parent: committed strictly before a, never folded into a."""
    return bool(c) and c != a and cf._git("merge-base", "--is-ancestor", c,
                                          f"{a}^").returncode == 0


def _job(name: str):
    """One job, on the records the parent built before forking."""
    recs, tset = _JOB["recs"], name.split(":", 1)[1]
    if name.startswith("E1_E2:"):
        return name, stratified_bootstrap(strata_groups(recs[tset], _JOB["stratum_of"]))
    if name.startswith("E2_raw:"):
        return name, e2_raw(recs[tset], tset == "held-out")
    return name, e3(with_length(recs[tset], _JOB["n_nt"]), tset == "held-out")


def _cross_check(results: dict, analysis: dict) -> dict:
    """E1 and E2 recount what the analysis recorded: the same R2 pool, the same D1."""
    st = results.get("E1_E2:TS0")
    if st is None:
        return {}
    e1_n = sum(v["n_pairs"] for v in st["E1_r2_within_strata"]["by_stratum"].values())
    e2_d1 = sum(v["n_d1"] for v in st["E2_d1_within_strata"]["by_stratum"].values())
    got = {"E1_pairs": e1_n, "R2_pairs": analysis["R2"]["n_pairs"],
           "E2_d1": e2_d1, "registered_D1": analysis["sets"]["TS0"]["disagreement"]["D1"]["n"]}
    if (got["E1_pairs"], got["E2_d1"]) != (got["R2_pairs"], got["registered_D1"]):
        raise GuardFailure(f"the exploratory records do not recount the analysis's: {got}")
    return got


def exploratory() -> int:
    """E1, E2, E2_raw and E3 from the committed run, after the committed analysis.

    Refuses unless: the run, the analysis and nothing exploratory are committed and unmodified;
    the analysis ran on the code HEAD carries (certfold's closure) and certfold_run.py is
    unchanged since it started; the specification is the one the analysis recorded; this file,
    the specification and the reading rules were committed STRICTLY before the commit that first
    carries the analysis record; every shard digest matches. Each job is written into
    metrics.json["exploratory"] as it finishes, by this process alone, with this file's blob;
    a job that raises is recorded under "failed" and stops the phase.
    """
    out = cf.read_out()
    mtime = os.path.getmtime(cf.OUT / "metrics.json")       # before anything writes it
    head = cf._committed(unmodified=True)
    if "analysis" not in head or "run" not in head:
        raise SystemExit("the exploratory analyses follow the committed run and analysis")
    if "exploratory" in head or "exploratory" in out:
        raise SystemExit("an exploratory record exists")
    if not cf._pred_ok(out):
        raise SystemExit("PRED changed since the registration")
    prov = git_provenance(R)
    if prov["git_dirty_excluding_runs"]:
        raise SystemExit(f"commit first: {prov.get('git_dirty_paths_outside_runs')}")
    analysis = head["analysis"]
    started = analysis["provenance"]["git_sha"]
    cf._same_code(analysis["provenance"], "analysis")
    if cf._git("diff", "--quiet", started, "HEAD", "--", cr.ME).returncode != 0:
        raise SystemExit(f"{cr.ME} changed since the analysis started ({started[:12]})")
    if _commit_of(SPEC) != analysis["code"]["exploratory_spec_commit"]:
        raise SystemExit(f"{SPEC} changed since the analysis recorded it")
    a = _analysis_commit()
    commits = {}
    for path in (ME, SPEC, READING_RULES):
        c = _commit_of(path)
        if not _strictly_before(c, a):
            raise SystemExit(f"{path} was not committed strictly before the analysis record "
                             f"({a[:12]})")
        commits[path] = {"commit": c, "time": _commit_time(c)}
    run_rec = head["run"]
    for f, h in run_rec["files_sha256"].items():
        if cf._sha(cf.SHARDS / f) != h:
            raise GuardFailure(f"{f} changed since the run")
    import scripts.certfold_data as cd

    t0 = time.time()
    blob = cf._git("rev-parse", f"HEAD:{ME}").stdout.strip()
    seq_recs, freqs = cr.read_shards(run_rec["workers"])
    freq_by_index = {int(k): v for w in freqs.values() for k, v in w["by_index"].items()}
    recs = cr.pair_records(seq_recs, freq_by_index)
    _JOB.update(recs=recs, stratum_of={s["index"]: s["stratum"] for s in seq_recs},
                n_nt={s["index"]: s["n_nt"] for s in seq_recs})
    bounds = cd.TR0_STRATA
    lo, hi = bounds[0][0], bounds[-1][1]
    outside = [{"index": s["index"], "n_nt": s["n_nt"], "stratum": s["stratum"]}
               for s in seq_recs if s["set"] == "TS0" and not lo <= s["n_nt"] <= hi]
    jobs = (["E1_E2:TS0"] + [f"E3:{t}" for t in cr.SETS if recs[t]]
            + [f"E2_raw:{t}" for t in cr.SETS if recs[t]])
    res = {
        "label": "EXPLORATORY, not registered tests (docs/certfold_exploratory.md, read by "
                 "docs/certfold_reading_rules.md)",
        "commits": commits, "module_blob": blob,
        "analysis_record_commit": {"commit": a, "time": _commit_time(a)},
        "analysis_started_on": started,
        "analysis_written_at": datetime.datetime.fromtimestamp(mtime).astimezone().isoformat(),
        "analysis_written_at_source": "metrics.json's mtime when this phase began, before its "
                                      "first write; the working copy equalled the committed "
                                      "analysis record, so the analysis was its last writer",
        "timing": "the specification (5b6f109) and the reading rules (292afc9) were committed "
                  "before the analysis wrote anything; this module was amended while the "
                  "analysis ran, after the authors' review of it, and committed strictly before "
                  "the analysis record; the test-set run itself was launched at 01:39 on "
                  "2026-10-14, before the authors' launch conditions arrived",
        "strata": {"bounds_TR0": [list(b) for b in bounds], "E2_merge": dict(E2_MERGE),
                   "TS0_outside_TR0_range": outside,
                   "rule": "certfold_data.length_stratum: a boundary length goes to the lower "
                           "stratum; a length beyond TR0's range to the nearest end stratum"},
        "numpy": np.__version__, "amendments_to_the_specification": list(AMENDMENTS),
        "wording": "the held-out set's references are Rfam consensus structures projected onto "
                   "each sequence, not structures determined per molecule; TS0 is the primary "
                   "accuracy set and R2 runs on TS0",
        "pending": list(jobs), "results": {}, "complete": False, "provenance": prov,
    }
    out = cf.read_out()
    out["exploratory"] = res
    cf.write(out)
    ctx = multiprocessing.get_context("fork")
    try:
        with ctx.Pool(min(WORKERS, len(jobs))) as pool:
            for name, got in pool.imap_unordered(_job, jobs):
                out = cf.read_out()             # one writer: this process
                ex = out["exploratory"]
                ex["results"][name] = {**got, "module_blob": blob,
                                       "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                ex["pending"] = [j for j in ex["pending"] if j != name]
                ex["seconds_at_last_write"] = round(time.time() - t0, 1)
                cf.write(out)
                print(f"{name} written at {time.strftime('%H:%M:%S')}", flush=True)
        out = cf.read_out()
        ex = out["exploratory"]
        ex["cross_check"] = _cross_check(ex["results"], analysis)
        ex["complete"] = not ex["pending"]
        cf.write(out)
    except BaseException as e:
        out = cf.read_out()
        out["exploratory"]["failed"] = repr(e)
        cf.write(out)
        raise
    print(json.dumps({k: res[k] for k in ("label", "commits", "analysis_record_commit",
                                         "analysis_written_at")}, indent=1))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"exploratory": exploratory}[ph]())
