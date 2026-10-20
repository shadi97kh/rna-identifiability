#!/usr/bin/env python3
"""The net split of v.w's change into the Adam step's and decay's signed contributions, per seed.
POST HOC, as specified before any code in docs/advisor_additions_2026-10-20.md (item 1b).

Data: runs/2026-10-12_mxfold2_ablation, arm adamw_wd0.01 (AdamW, lr and l2 as registered there),
seeds 20261006-08, ten epochs each, whose runner logged v.w before and after every optimiser step
(shards/<run>/gauge/vw-epoch-<k>.npz: pre, raw, post, start). Nothing is trained or rerun.

Per step t, exactly as scripts/mxfold2_wd_ablation.py's arm (a):
    decay_t = -lr * wd * pre_t          (torch AdamW's decoupled decay, applied before the step)
    adam_t  = (post_t - pre_t) - decay_t
summed over epoch 1 and over epochs 1 to 10, per seed. Checks 1-5 of the specification stop
the script on any failure. Records runs/2026-10-20_mxfold2_net_split/metrics.json.

Run: .venv/bin/python scripts/mxfold2_net_split.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402

SPEC = "docs/advisor_additions_2026-10-20.md"
ABL = R / "runs" / "2026-10-12_mxfold2_ablation"
ARM = "adamw_wd0.01"
EPOCHS = 10
OUT = R / "runs" / "2026-10-20_mxfold2_net_split"
STEP_TOL = 1e-12        # continuity and endpoint identity: the same float64 values, logged twice
SUM_TOL = 1e-9          # the spec's check 4


class CheckFailed(RuntimeError):
    pass


def _git(*a):
    return subprocess.run(["git", *a], cwd=R, capture_output=True, text=True)


def gate() -> dict:
    if _git("ls-files", "--error-unmatch", SPEC).returncode != 0:
        raise SystemExit(f"{SPEC} is not committed")
    if _git("diff", "--quiet", "HEAD", "--", SPEC).returncode != 0:
        raise SystemExit(f"{SPEC} has uncommitted changes")
    prov = git_provenance(R)
    if prov["git_dirty_excluding_runs"]:
        raise SystemExit("commit first: the tree is dirty outside runs/")
    return prov


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def seed_split(rec: dict, run: dict, lr: float, wd: float) -> dict:
    name = run["name"]
    d = ABL / "shards" / name / "gauge"
    digests = rec["run"]["shards_sha256"][name]
    adam_by_epoch, decay_by_epoch, ends = [], [], []
    prev_post = None
    v_start = None
    for k in range(1, EPOCHS + 1):
        f, g = d / f"vw-epoch-{k}.npz", d / f"gauge-epoch-{k}.json"
        for p in (f, g):
            want = digests.get(f"gauge/{p.name}")
            if want is not None and sha(p) != want:
                raise CheckFailed(f"{name}: {p.name} is not the file the run recorded")
        z = np.load(f)
        gj = json.loads(g.read_text())
        pre, raw, post = (np.asarray(z[a], dtype=np.float64) for a in ("pre", "raw", "post"))
        start = float(np.asarray(z["start"]).ravel()[0])
        if not (np.isfinite(pre).all() and np.isfinite(post).all() and np.isfinite(raw).all()):
            raise CheckFailed(f"{name} epoch {k}: a non-finite v.w")
        if len(pre) != len(post) or len(pre) != len(raw):
            raise CheckFailed(f"{name} epoch {k}: pre/raw/post lengths differ")
        if len(pre) > 1 and float(np.abs(pre[1:] - post[:-1]).max()) > STEP_TOL:
            raise CheckFailed(f"{name} epoch {k}: v.w changed between steps")
        if float(np.abs(raw - post).max()) > STEP_TOL:
            raise CheckFailed(f"{name} epoch {k}: raw != post in an arm that projects nothing")
        if abs(pre[0] - gj["v_dot_w_start"]["v_dot_w"]) > STEP_TOL or \
                abs(post[-1] - gj["v_dot_w"]) > STEP_TOL:
            raise CheckFailed(f"{name} epoch {k}: endpoints differ from the gauge record")
        if abs(start - pre[0]) > STEP_TOL:
            raise CheckFailed(f"{name} epoch {k}: the epoch's measured start is not its first pre")
        if prev_post is not None and abs(start - prev_post) > STEP_TOL:
            raise CheckFailed(f"{name} epoch {k}: start differs from epoch {k - 1}'s last post")
        if v_start is None:
            v_start = float(pre[0])
        decay = -lr * wd * pre
        adam = (post - pre) - decay
        adam_by_epoch.append(float(adam.sum()))
        decay_by_epoch.append(float(decay.sum()))
        ends.append(float(post[-1]))
        prev_post = float(post[-1])

    def horizon(n):
        a, dcy = sum(adam_by_epoch[:n]), sum(decay_by_epoch[:n])
        net = ends[n - 1] - v_start
        if abs((a + dcy) - net) > SUM_TOL:
            raise CheckFailed(f"{name}: contributions do not sum to the net change ({n} epochs)")
        return {"adam_step": a, "decay": dcy, "net_change": net,
                "ratio_adam_over_decay": a / dcy if dcy else None,
                "adam_exceeds_decay_in_magnitude": abs(a) > abs(dcy)}

    return {"run": name, "seed": run["seed"], "v_w_start": v_start, "v_w_by_epoch": ends,
            "adam_step_by_epoch": adam_by_epoch, "decay_by_epoch": decay_by_epoch,
            "epoch_1": horizon(1), "epochs_1_to_10": horizon(EPOCHS)}


def main() -> int:
    prov = gate()
    rec = json.loads((ABL / "metrics.json").read_text())
    lr = float(rec["registered_constants"]["lr"])
    runs = [r for r in rec["registered_constants"]["runs_planned"] if r["arm"] == ARM]
    wd = {float(r["l2"]) for r in runs}
    if len(runs) != 3 or len(wd) != 1:
        raise CheckFailed(f"expected 3 seeds of {ARM} at one l2, got {runs}")
    wd = wd.pop()
    seeds = [seed_split(rec, r, lr, wd) for r in sorted(runs, key=lambda r: r["seed"])]
    passes = all(s[h]["adam_exceeds_decay_in_magnitude"]
                 for s in seeds for h in ("epoch_1", "epochs_1_to_10"))
    out = {
        "run_id": OUT.name, "label": "POST HOC (docs/advisor_additions_2026-10-20.md, item 1b)",
        "specification": {"path": SPEC, "sha256": sha(R / SPEC),
                          "commit": _git("log", "-1", "--format=%H", "--", SPEC).stdout.strip()},
        "source": str(ABL.relative_to(R)), "arm": ARM, "lr": lr, "wd": wd,
        "definition": "decay_t = -lr*wd*pre_t; adam_t = (post_t - pre_t) - decay_t; summed "
                      "over epoch 1 and over epochs 1-10",
        "checks": {"1_finite": True, "2_continuity_and_raw_equals_post": True,
                   "3_endpoints_and_starts": True, "4_contributions_sum_to_net": True,
                   "5_files_match_recorded_sha256": True},
        "by_seed": seeds,
        # the magnitudes across seeds, as the text quotes them (the signs differ by seed)
        "magnitude_ranges": {
            h: {part: {"min": min(abs(s[h][part]) for s in seeds),
                       "max": max(abs(s[h][part]) for s in seeds)}
                for part in ("adam_step", "decay")}
            for h in ("epoch_1", "epochs_1_to_10")},
        "reading": {"rule": "Adam's net contribution exceeds decay's in magnitude in every seed "
                            "at both horizons -> the net figures replace '33 times further'; "
                            "otherwise drop the comparison with decay and keep the no-decay arm",
                    "adam_exceeds_decay_in_every_seed_at_both_horizons": passes,
                    "outcome": ("replace '33 times further' with the net figures" if passes
                                else "drop the comparison with decay; keep the no-decay arm")},
        "provenance": prov}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")
    for s in seeds:
        print(s["seed"], {h: {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in s[h].items()} for h in ("epoch_1", "epochs_1_to_10")})
    print("reading:", out["reading"]["outcome"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
