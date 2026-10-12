#!/usr/bin/env python3
"""Run `mxfold2 train`, timed and gauge-hooked, recording v.w around every optimiser step.

The weight-decay ablation's runner (scripts/mxfold2_wd_ablation.py). MXfold2 and the shared
gauge hooks are unchanged: this installs scripts/mxfold2_gauge_hooks.py exactly as
scripts/mxfold2_gauge_train.py does, then adds two things of its own:
  * torch optimiser step pre/post hooks that record v.w (scripts/mxfold2_gauge.v_dot) just
    before and just after every optimizer.step(), saved per epoch as
    MXF_VW_OUT/vw-epoch-{k}.npz (arrays pre, post). With these, AdamW's decoupled decay
    contribution is exactly -lr * wd * pre per step and the preconditioned step is the rest.
  * MXF_WD_OPT=sgd_m0 replaces MXfold2's SGD (hard-coded Nesterov, momentum 0.9) with
    torch.optim.SGD(lr, momentum=0, weight_decay=--l2-weight), so the only things moving v.w
    are the decay and the logged v.g.
Configuration comes from the environment (extra argv would reach MXfold2's argparse):
  MXF_TIMING_OUT, MXF_GAUGE_OUT, MXF_GAUGE_LAMBDA, MXF_GAUGE_CROSSCHECK  as the gauge runner
  MXF_VW_OUT   directory for vw-epoch-*.npz (required)
  MXF_WD_OPT   "sgd_m0" for the momentum-0 SGD override; unset keeps MXfold2's optimiser
Usage: .venv-mxfold2/bin/python scripts/mxfold2_wd_ablation_train.py train <mxfold2 train args>
"""

from __future__ import annotations

import os
import pathlib
import sys

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.mxfold2_timed_train as timed  # noqa: E402  (installs the timing wrappers)
from scripts import mxfold2_gauge_hooks as hooks  # noqa: E402

VW = {"pre": [], "post": []}


def install(out_dir: str, opt: str | None):
    import mxfold2.train as train_mod
    import numpy as np
    import torch

    from scripts.mxfold2_gauge import BIAS_KEY, HELIX_W_KEY, v_dot

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _build = train_mod.Train.build_optimizer

    def build_optimizer(self, optimizer, model, lr, l2_weight):
        if opt == "sgd_m0":
            o = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0,
                                weight_decay=l2_weight)
        else:
            o = _build(self, optimizer, model, lr, l2_weight)
        named = dict(model.named_parameters())
        b, w = named[BIAS_KEY], named[HELIX_W_KEY]

        def vw():
            with torch.no_grad():
                return v_dot({BIAS_KEY: b.detach(), HELIX_W_KEY: w.detach()})

        o.register_step_pre_hook(lambda *_: VW["pre"].append(vw()))
        o.register_step_post_hook(lambda *_: VW["post"].append(vw()))
        return o

    train_mod.Train.build_optimizer = build_optimizer
    _train = train_mod.Train.train

    def train_epoch(self, epoch):
        VW["pre"].clear()
        VW["post"].clear()
        r = _train(self, epoch)
        tmp = out / f"vw-epoch-{epoch}.tmp.npz"
        np.savez(tmp, pre=np.array(VW["pre"]), post=np.array(VW["post"]))
        os.replace(tmp, out / f"vw-epoch-{epoch}.npz")
        return r

    train_mod.Train.train = train_epoch


if __name__ == "__main__":
    hooks.install(lam=float(os.environ.get("MXF_GAUGE_LAMBDA", "0")),
                  out_dir=os.environ["MXF_GAUGE_OUT"],
                  crosscheck=int(os.environ.get("MXF_GAUGE_CROSSCHECK", "0")))
    install(os.environ["MXF_VW_OUT"], os.environ.get("MXF_WD_OPT"))
    timed.main()
