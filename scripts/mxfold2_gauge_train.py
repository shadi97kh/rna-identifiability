#!/usr/bin/env python3
"""Run `mxfold2 train`, timed and gauge-hooked. MXfold2 itself is unchanged.

Configuration comes from the environment, because extra argv would reach MXfold2's
argparse:
  MXF_TIMING_OUT       timing JSON (scripts/mxfold2_timed_train.py)
  MXF_GAUGE_OUT        directory for gauge-epoch-*.json and steps-epoch-*.npz
  MXF_GAUGE_LAMBDA     null-component regulariser weight; 0 = log only (default 0)
  MXF_GAUGE_CROSSCHECK number of first steps whose R is recomputed on the CPU in
                       float64 by the numpy Geometry and compared (default 0)
Usage: .venv-mxfold2/bin/python scripts/mxfold2_gauge_train.py train <mxfold2 train args>
"""

from __future__ import annotations

import os
import pathlib
import sys

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import scripts.mxfold2_timed_train as timed  # noqa: E402  (installs the timing wrappers)
from scripts import mxfold2_gauge_hooks as hooks  # noqa: E402

if __name__ == "__main__":
    hooks.install(lam=float(os.environ.get("MXF_GAUGE_LAMBDA", "0")),
                  out_dir=os.environ["MXF_GAUGE_OUT"],
                  crosscheck=int(os.environ.get("MXF_GAUGE_CROSSCHECK", "0")))
    timed.main()
