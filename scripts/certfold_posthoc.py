#!/usr/bin/env python3
"""CERTIFIED FOLDING, THE ONE POST HOC CHECK (the authors, 2026-10-15). Not a registered test.

The rule is the dated addendum in docs/certfold_reading_rules.md (60af83d), committed before this
runs. It asks whether R2's signal is specific to the certificate: R2's statistic with the nominal
gap X0(p) - E0(s*) in place of rho_env, on R2's pool (TS0 MFE pairs with p >= 0.99 carrying a
rho_env), with R2's tie rules and R2's 10,000 bootstrap draws, plus the paired difference
Delta = d_rho - d_gap. It may only qualify the 2a sentence, never strengthen it.

Every number comes from the frozen metrics through certfold_run's statistics; the draws are
paired_bootstrap's (seed tag 40) over the unit universe certfold_run.task_records gives R2, so
they are R2's own -- which this checks, by requiring d_rho to reproduce R2's committed point and
interval exactly.

Run, after the addendum and this file are committed:
  .venv-certfold/bin/python scripts/certfold_posthoc.py --phase posthoc-gap   # then commit
"""

from __future__ import annotations

import json
import multiprocessing
import pathlib
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.certfold_run as cr  # noqa: E402
from scripts.certfold_metrics import (  # noqa: E402
    paired_bootstrap,
    ppv_at_coverage,
    risk_coverage,
)
from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_typeb import GuardFailure  # noqa: E402

cf = cr.cf
ME = "scripts/certfold_posthoc.py"
RULES = "docs/certfold_reading_rules.md"
N_BOOT = 10000


def with_neg_gap(recs: list[dict]) -> list[dict]:
    """MFE records given neg_gap = -gap, so the gap's reversed ranking can be read (copies)."""
    return [{**r, "neg_gap": -r["gap"]} if r["kind"] == "mfe" else r for r in recs]


def stat_gap(recs: list[dict]) -> float:
    """R2's statistic with the nominal gap as the score: PPV(lowest gap quartile) - PPV(rest)
    on R2's pool, read exactly as certfold_run.stat_r2_quartile reads rho_env's (the lowest
    quarter off the reversed ranking at coverage 1/4, the rest off the forward ranking at
    (n - k_low) / n). The gap is both score and tiebreak, so ties on it are fractional."""
    pool = cr._r2_pool(recs)
    n = len(pool)
    covs = np.arange(1, n + 1) / n
    k_low = int(np.searchsorted(covs, 0.25, side="left")) + 1
    if k_low >= n:
        raise GuardFailure(f"{n} pairs leave no 'rest' beside a lowest quartile of {k_low}")
    low = ppv_at_coverage(risk_coverage(pool, "neg_gap", "neg_tiebreak")[0], 0.25)
    rest = ppv_at_coverage(risk_coverage(pool, "gap", "tiebreak")[0], (n - k_low) / n)
    return low - rest


def stat_rho(recs: list[dict]) -> float:
    return cr.stat_r2_quartile(recs)


def stat_delta(recs: list[dict]) -> float:
    return stat_rho(recs) - stat_gap(recs)


STATS = {"d_rho": stat_rho, "d_gap": stat_gap, "delta": stat_delta}
_JOB: dict = {}


def _boot(name: str):
    p, lo, hi = paired_bootstrap(_JOB["sub"], STATS[name], n_boot=N_BOOT)
    return name, {"point": p, "interval": [lo, hi]}


def reading(res: dict) -> dict:
    """The addendum's three branches, decided by intervals only (rule 5)."""
    d_iv, g_iv = res["delta"]["interval"], res["d_gap"]["interval"]
    if d_iv[1] < 0:
        return {"branch": "rho's difference is larger (Delta's interval entirely below 0)",
                "main_text": "unchanged", "supplement": "reported as a post hoc observation"}
    if g_iv[1] < 0:
        return {"branch": "the gap's difference is as large or larger (Delta's interval not below "
                          "0; the gap's own interval entirely below 0)",
                "main_text": "add: the energy gap alone shows the same pattern, so this signal is "
                             "not specific to the certificate",
                "abstract": "add after 'within length strata': a pattern the energy gap alone "
                            "also shows"}
    return {"branch": "inconclusive (the gap's interval includes 0; Delta's not below 0); the "
                      "agent's qualifying-only branch, for the authors to confirm",
            "main_text": "add: whether the signal is specific to the certificate is unresolved: "
                         "the energy gap's pattern cannot be told apart from it"}


def posthoc_gap() -> int:
    out = cf.read_out()
    head = cf._committed(unmodified=True)
    for key in ("run", "analysis", "exploratory"):
        if key not in head:
            raise SystemExit(f"the post hoc check follows the committed {key} record")
    if "posthoc" in head or "posthoc" in out:
        raise SystemExit("a post hoc record exists")
    if not cf._pred_ok(out):
        raise SystemExit("PRED changed since the registration")
    prov = git_provenance(R)
    if prov["git_dirty_excluding_runs"]:
        raise SystemExit(f"commit first: {prov.get('git_dirty_paths_outside_runs')}")
    analysis = head["analysis"]
    cf._same_code(analysis["provenance"], "analysis")
    started = analysis["provenance"]["git_sha"]
    if cf._git("diff", "--quiet", started, "HEAD", "--", cr.ME).returncode != 0:
        raise SystemExit(f"{cr.ME} changed since the analysis started")
    for f, h in head["run"]["files_sha256"].items():
        if cf._sha(cf.SHARDS / f) != h:
            raise GuardFailure(f"{f} changed since the run")
    rules_commit = cf._git("log", "-1", "--format=%H", "--", RULES).stdout.strip()
    t0 = time.time()
    seq_recs, freqs = cr.read_shards(head["run"]["workers"])
    fbi = {int(k): v for w in freqs.values() for k, v in w["by_index"].items()}
    recs = with_neg_gap(cr.pair_records(seq_recs, fbi)["TS0"])
    _JOB["sub"] = cr.task_records(recs, "r2_quartile", ())
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(len(STATS)) as pool:
        res = dict(pool.map(_boot, list(STATS)))
    r2 = analysis["R2"]
    if res["d_rho"]["point"] != r2["point"] or res["d_rho"]["interval"] != r2["interval"]:
        raise GuardFailure(f"d_rho does not reproduce R2 ({res['d_rho']} vs {r2['point']}, "
                           f"{r2['interval']}): the draws are not R2's")
    rec = {
        "label": "POST HOC, not a registered test: may only qualify the 2a sentence",
        "rule": {"path": RULES, "commit": rules_commit,
                 "time": cf._git("show", "-s", "--format=%cI", rules_commit).stdout.strip()},
        "module_commit": cf._git("log", "-1", "--format=%H", "--", ME).stdout.strip(),
        "pool": {"set": "TS0", "n_pairs": len(cr._r2_pool(recs)),
                 "what": "MFE pairs with p >= 0.99 carrying a rho_env (R2's pool)"},
        "statistics": res,
        "reproduces_R2": True,
        "reading": reading(res),
        "bootstrap": {"n_boot": N_BOOT, "seed": [cf.SEED, 40],
                      "draws": "paired_bootstrap over the unit universe certfold_run.task_records "
                               "gives R2: R2's own draws, checked by reproducing R2"},
        "provenance": prov, "seconds": round(time.time() - t0, 1),
    }
    out = cf.read_out()
    out["posthoc"] = rec
    cf.write(out)
    print(json.dumps({k: rec[k] for k in ("statistics", "reading")}, indent=1))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"posthoc-gap": posthoc_gap}[ph]())
