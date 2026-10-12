#!/usr/bin/env python3
"""Type B Experiment 2: three EXPLORATORY analyses of the existing fold shards (no folding).

Asked by the user 2026-10-12 after the fold was analysed; none was registered.
At the three published-component points (Turner 2004 plus the nontrivial part of Turner
1999's, Andronescu 2007's and Langdon 2018's difference, each at its own norm):
  1. paired F1-against-reference difference, perturbed minus unperturbed, per sequence,
     mean with a bootstrap 95% CI over sequences ("0.482 and 0.482" were two rounded means)
  2. changed-MFE rate by the unperturbed model's MFE gap (to the best other structure), bins
     0, (0, 0.1], (0.1, 0.5] and > 0.5 kcal/mol (> 0.5 includes no second structure within
     1 kcal/mol), with Wilson 95% intervals and each bin's share of the changed MFEs: is
     the change concentrated in tied or near-tied sequences?
  3. mean F1 between the unperturbed and the perturbed MFE among sequences whose MFE changed,
     with a bootstrap CI and quartiles: does a change lose one pair or many?
The shards are checked against the sha256 recorded by the fold driver.

Writes runs/2026-10-12_turner_typeb_exp2_explore/metrics.json.
Run: .venv/bin/python scripts/turner_typeb_exp2_explore.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_typeb import FOLD, N_WORKERS, SEED, read_out, sha256  # noqa: E402

RUN_ID = "2026-10-12_turner_typeb_exp2_explore"
OUT = R / "runs" / RUN_ID
BOOT = 2000
BINS = [("0", lambda g: g == 0), ("(0, 0.1]", lambda g: 0 < g <= 10),
        ("(0.1, 0.5]", lambda g: 10 < g <= 50), ("> 0.5", lambda g: g > 50)]


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def boot_ci(x, rng):
    x = np.asarray(x, dtype=float)
    b = x[rng.integers(0, len(x), (BOOT, len(x)))].mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def main():
    art = read_out()
    assert art["status"] == "fold analysed", art["status"]
    rec = art["fold"]["shards_sha256"]
    full = FOLD / "full"
    for f, h in rec.items():
        assert sha256(full / f) == h, f"{f} changed since the fold"
    shards = [[json.loads(ln) for ln in (full / f"shard{w}.jsonl").read_text().splitlines()]
              for w in range(N_WORKERS)]
    base = [b for s in shards for b in s[0]["baseline"]]
    gap = np.array([1e9 if b["gap"] is None else b["gap"] for b in base])
    f1ref0 = np.array([b["f1_ref"] for b in base])
    meta = json.loads((FOLD / "points.json").read_text())
    own = [m for m in meta if m["kind"] == "published_own"]
    out = {"run_id": RUN_ID, "status": "running", "label": "EXPLORATORY (asked 2026-10-12)",
           "source_artifact": "runs/2026-10-12_turner_typeb", "n_sequences": len(base),
           **git_provenance(R)}
    res = {}
    for i, m in enumerate(own):
        p = m["p"]
        rs = [next(r for r in s if r.get("p") == p) for s in shards]
        cat = {k: np.array([x for r in rs for x in r[k]]) for k in ("changed", "f1_base",
                                                                    "f1_ref")}
        ch = cat["changed"].astype(bool)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 8, i]))
        d = cat["f1_ref"] - f1ref0
        bins = {}
        for name, test in BINS:
            mask = np.array([test(g) for g in gap])
            k, n = int((ch & mask).sum()), int(mask.sum())
            bins[name] = {"n_sequences": n, "n_changed": k, "rate": k / n if n else None,
                          "rate_wilson95": wilson(k, n),
                          "share_of_changed": k / int(ch.sum()) if ch.sum() else None}
        f1c = cat["f1_base"][ch]
        res[m["name"]] = {
            "norm_kcal": m["norm_kcal"], "n_changed": int(ch.sum()),
            "1_paired_f1_ref_difference": {
                "mean": float(d.mean()), "ci95": boot_ci(d, rng),
                "mean_unperturbed": float(f1ref0.mean()), "mean_perturbed": float(
                    cat["f1_ref"].mean()),
                "among_changed_mean": float(d[ch].mean()) if ch.any() else None,
                "among_changed_ci95": boot_ci(d[ch], rng) if ch.any() else None,
                "n_changed_better": int((d[ch] > 0).sum()),
                "n_changed_worse": int((d[ch] < 0).sum()),
                "n_changed_equal": int((d[ch] == 0).sum())},
            "2_changed_rate_by_gap_kcal": bins,
            "3_f1_old_vs_new_among_changed": {
                "mean": float(f1c.mean()) if ch.any() else None,
                "ci95": boot_ci(f1c, rng) if ch.any() else None,
                "quartiles": [float(np.percentile(f1c, q)) for q in (25, 50, 75)]
                if ch.any() else None,
                "fraction_below_0.5": float((f1c < 0.5).mean()) if ch.any() else None}}
    out["published_points"] = res
    out["gap_bins_population"] = {name: int(np.array([t(g) for g in gap]).sum())
                                  for name, t in BINS}
    out["status"] = "complete"
    write(out)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
