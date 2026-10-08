#!/usr/bin/env python3
"""Run MXfold2's own `mxfold2 train` unchanged, recording where the wall clock goes.

Runs in .venv-mxfold2 (MXfold2 at 51b2136, torch 2.13.0+cu130). Arguments are passed
straight to `mxfold2 train`; nothing about training is altered. Wrappers installed
BEFORE the model is built time, with CUDA synchronisation where GPU work is involved:

  dp_pred     interface.predict_mxfold without a constraint: the loss-augmented
              unconstrained fold (C++, CPU), one per step
  dp_ref      interface.predict_mxfold with the reference as constraint, one per step
  dp_turner   interface.predict_turner: the Turner-only constrained fold that
              hinge_mix adds under no_grad, one per step
  nn_forward  MixedFold.make_param: the network forward pass (and the Turner table read)
  backward    torch.Tensor.backward
  epoch       Train.train, one call per epoch

overhead = epoch - (dp_pred + dp_ref + dp_turner + nn_forward + backward): parameter
copies to the CPU, count bookkeeping, the optimiser step, data loading and Python.

The timing record is written as JSON to the path in MXF_TIMING_OUT.
Usage: .venv-mxfold2/bin/python scripts/mxfold2_timed_train.py train <mxfold2 train args>
"""

from __future__ import annotations

import json
import os
import sys
import time

import mxfold2.interface as interface
import mxfold2.train as train_mod
import torch
from mxfold2.fold.mix import MixedFold

T = {k: 0.0 for k in ("dp_pred_s", "dp_ref_s", "dp_turner_s", "nn_forward_s", "backward_s")}
N = {k: 0 for k in ("dp_pred_calls", "dp_ref_calls", "dp_turner_calls", "nn_forward_calls",
                    "backward_calls")}
EPOCHS = []


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


_mx = interface.predict_mxfold


def predict_mxfold(*a, **k):
    key = "ref" if k.get("constraint") is not None else "pred"
    t = time.perf_counter()
    r = _mx(*a, **k)
    T[f"dp_{key}_s"] += time.perf_counter() - t
    N[f"dp_{key}_calls"] += 1
    return r


_tu = interface.predict_turner


def predict_turner(*a, **k):
    t = time.perf_counter()
    r = _tu(*a, **k)
    T["dp_turner_s"] += time.perf_counter() - t
    N["dp_turner_calls"] += 1
    return r


_mp = MixedFold.make_param


def make_param(self, seq):
    _sync()
    t = time.perf_counter()
    r = _mp(self, seq)
    _sync()
    T["nn_forward_s"] += time.perf_counter() - t
    N["nn_forward_calls"] += 1
    return r


_bw = torch.Tensor.backward


def backward(self, *a, **k):
    _sync()
    t = time.perf_counter()
    r = _bw(self, *a, **k)
    _sync()
    T["backward_s"] += time.perf_counter() - t
    N["backward_calls"] += 1
    return r


_tr = train_mod.Train.train


def train_epoch(self, epoch):
    t = time.perf_counter()
    r = _tr(self, epoch)
    EPOCHS.append({"epoch": epoch, "seconds": time.perf_counter() - t,
                   "n_sequences": len(self.train_loader.dataset)})
    return r


interface.predict_mxfold = predict_mxfold
interface.predict_turner = predict_turner
MixedFold.make_param = make_param
torch.Tensor.backward = backward
train_mod.Train.train = train_epoch


def main():
    from mxfold2.__main__ import main as mx_main

    t0 = time.perf_counter()
    sys.argv = ["mxfold2", *sys.argv[1:]]
    mx_main()
    rec = {"argv": sys.argv[1:], "wall_seconds": time.perf_counter() - t0, "epochs": EPOCHS,
           **T, **N, "torch": torch.__version__,
           "gpu_peak_mib": (torch.cuda.max_memory_allocated() / 2**20
                            if torch.cuda.is_available() else None)}
    ep = sum(e["seconds"] for e in EPOCHS)
    rec["overhead_s"] = ep - sum(T.values())
    with open(os.environ["MXF_TIMING_OUT"], "w") as fh:
        json.dump(rec, fh, indent=1)


if __name__ == "__main__":
    main()
