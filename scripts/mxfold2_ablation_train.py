#!/usr/bin/env python3
"""Run `mxfold2 train`, timed and gauge-hooked, recording v.w around every optimiser step and
optionally projecting the weights off v after every step (A3, scripts/mxfold2_ablation.py).

Modelled on scripts/mxfold2_wd_ablation_train.py (not edited). MXfold2 and the shared gauge
hooks are unchanged: this installs scripts/mxfold2_timed_train.py (timing, at import) and
scripts/mxfold2_gauge_hooks.py (lambda from MXF_GAUGE_LAMBDA, 0 in A3: the hooks only log)
exactly as the wd runner does, then wraps Train.build_optimizer:

  1. v.w of the freshly built (or --init-param) model is recorded UNPROJECTED;
  2. if MXF_PROJECT=1 the weights are projected once (scripts/mxfold2_vproject.project_v_),
     and the record {projected, pre, c, post, bound} is written atomically to
     MXF_VW_OUT/projection-init.json (a resumed process, which builds the optimiser again
     before MXfold2 loads the checkpoint over the fresh weights, writes
     projection-init.again-<n>.json instead and never overwrites the original). The gauge
     hooks record v_dot_w_start at the first Train.train call, i.e. AFTER this projection:
     that is the arm's post-projection start;
  3. torch optimiser step hooks, registered in this order:
       pre-hook   logs v.w before optimizer.step()                               -> pre
       post-hook  logs v.w after the raw optimiser step                           -> raw
       post-hook  (MXF_PROJECT=1 only) projects, then asserts |v.w| <= v_bound live:
                  SystemExit("MXF_PROJECT BOUND VIOLATION ...") on a violation and
                  SystemExit("MXF_PROJECT NON-FINITE v.w ...") on NaN/inf (a diverged run)
       post-hook  logs v.w after everything                                       -> post
     (MXfold2 steps only when the loss is > 0, train.py:51-56; the hooks fire on steps only.)
  4. per epoch, MXF_VW_OUT/vw-epoch-{k}.npz with arrays pre, raw, post and start, and with
     the projection also c (the float64 shift) and bound (v_bound after it), atomically.
     `start` is v.w measured HERE, at the top of Train.train, before the epoch runs. It is
     the only independent per-epoch continuity measurement: the gauge hooks set the next
     epoch's v_dot_w_start from the epoch they have just recorded
     (scripts/mxfold2_gauge_hooks.py:206), so their own start/end equality holds by
     construction within one process and can fail only across a resume. `start` is measured
     from the live parameters, so the launcher can check it against the previous epoch's last
     logged post (and against the hooks' record) for every epoch of every process.

With MXF_PROJECT unset (or "0") the hooks only read: training is bit-identical in effect to
the plain timed runner (checked by the A3 CPU smoke, C1 and C2), and post == raw.

Configuration comes from the environment (extra argv would reach MXfold2's argparse):
  MXF_TIMING_OUT, MXF_GAUGE_OUT, MXF_GAUGE_LAMBDA, MXF_GAUGE_CROSSCHECK  as the gauge runner
  MXF_VW_OUT            directory for projection-init.json and vw-epoch-*.npz (required)
  MXF_PROJECT           "1" projects; unset or "0" does not; anything else is refused
  MXF_PROJECT_PLANTED   SMOKE ONLY: "skip" keeps the live assert but skips every projection
                        after the first step, which must trip the assert (A3 control C4);
                        refused unless MXF_PROJECT=1. The launcher strips every inherited
                        MXF_* variable, so a training run can never carry it.
Usage: .venv-mxfold2/bin/python scripts/mxfold2_ablation_train.py train <mxfold2 train args>
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.mxfold2_timed_train as timed  # noqa: E402  (installs the timing wrappers)
from scripts import mxfold2_gauge_hooks as hooks  # noqa: E402
from scripts.mxfold2_vproject import BOUND_MSG, NONFINITE_MSG  # noqa: E402

VW = {"pre": [], "raw": [], "post": [], "c": [], "bound": []}
STEP = {"n": 0}
HOOK = {}  # "vw": the live v.w reader, set when the optimiser is built (before any epoch)


def _atomic_json(path: pathlib.Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


def _init_path(out: pathlib.Path) -> pathlib.Path:
    p = out / "projection-init.json"
    n = 1
    while p.exists():
        p = out / f"projection-init.again-{n}.json"
        n += 1
    return p


def _check_bound(x: float, bound: float, where: str) -> None:
    if not (math.isfinite(x) and math.isfinite(bound)):
        raise SystemExit(f"{NONFINITE_MSG} {where}: v.w = {x!r}, v_bound = {bound!r} "
                         "(the run diverged)")
    if not abs(x) <= bound:
        raise SystemExit(f"{BOUND_MSG} {where}: |v.w| = {abs(x)!r} > v_bound = {bound!r}")


def install(out_dir: str, project: bool, planted: str | None) -> None:
    import mxfold2.train as train_mod
    import numpy as np
    import torch

    from scripts.mxfold2_gauge import BIAS_KEY, HELIX_W_KEY, v_dot
    from scripts.mxfold2_vproject import project_v_, v_bound

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _build = train_mod.Train.build_optimizer

    def build_optimizer(self, optimizer, model, lr, l2_weight):
        o = _build(self, optimizer, model, lr, l2_weight)
        named = dict(model.named_parameters())
        b, w = named[BIAS_KEY], named[HELIX_W_KEY]

        def vw():
            with torch.no_grad():
                return v_dot({BIAS_KEY: b.detach(), HELIX_W_KEY: w.detach()})

        HOOK["vw"] = vw  # Train.build_optimizer runs before the first Train.train call
        init = {"projected": project, "planted": planted, "pre": vw(), "optimizer":
                type(o).__name__, "lr": lr, "l2_weight": l2_weight}
        if project:
            r = project_v_(b, w)
            post = vw()
            bound = v_bound(b, w, r["c"])
            init.update(c=r["c"], pre_cpu=r["pre"], post=post, bound=bound,
                        within_bound=bool(abs(post) <= bound))
        _atomic_json(_init_path(out), init)
        if project:
            _check_bound(init["post"], init["bound"], "after the initial projection")

        o.register_step_pre_hook(lambda *_: VW["pre"].append(vw()))
        o.register_step_post_hook(lambda *_: VW["raw"].append(vw()))
        if project:
            def projection_hook(*_):
                STEP["n"] += 1
                if planted == "skip" and STEP["n"] > 1:
                    c = 0.0  # the planted failure: no projection, the assert stays
                else:
                    c = project_v_(b, w)["c"]
                x = vw()
                bound = v_bound(b, w, c)
                VW["c"].append(c)
                VW["bound"].append(bound)
                _check_bound(x, bound, f"at optimiser step {STEP['n']} of this process")

            o.register_step_post_hook(projection_hook)
        o.register_step_post_hook(lambda *_: VW["post"].append(vw()))
        return o

    train_mod.Train.build_optimizer = build_optimizer
    _train = train_mod.Train.train

    def train_epoch(self, epoch):
        for v in VW.values():
            v.clear()
        start = HOOK["vw"]()  # before the epoch: the independent continuity measurement
        r = _train(self, epoch)
        arrs = {k: np.array(VW[k], dtype=float) for k in ("pre", "raw", "post")}
        arrs["start"] = np.array([start], dtype=float)
        if project:
            arrs.update(c=np.array(VW["c"], dtype=float), bound=np.array(VW["bound"], dtype=float))
        tmp = out / f"vw-epoch-{epoch}.tmp.npz"
        np.savez(tmp, **arrs)
        os.replace(tmp, out / f"vw-epoch-{epoch}.npz")
        return r

    train_mod.Train.train = train_epoch


def configuration() -> tuple[bool, str | None]:
    p = os.environ.get("MXF_PROJECT", "0")
    if p not in ("0", "1"):
        raise SystemExit(f"MXF_PROJECT must be 0 or 1, got {p!r}")
    planted = os.environ.get("MXF_PROJECT_PLANTED")
    if planted is not None and (planted != "skip" or p != "1"):
        raise SystemExit("MXF_PROJECT_PLANTED=skip is a smoke control and needs MXF_PROJECT=1")
    return p == "1", planted


if __name__ == "__main__":
    project, planted = configuration()
    hooks.install(lam=float(os.environ.get("MXF_GAUGE_LAMBDA", "0")),
                  out_dir=os.environ["MXF_GAUGE_OUT"],
                  crosscheck=int(os.environ.get("MXF_GAUGE_CROSSCHECK", "0")))
    install(os.environ["MXF_VW_OUT"], project, planted)
    timed.main()
