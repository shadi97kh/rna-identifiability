#!/usr/bin/env python3
"""Which TR0 sequences carry the large v.g steps of the converged SGD run?

Asked by the user 2026-10-10, after S1 failed its conservation criterion on SGD seed
20261006 (residual 0.010449 against 1e-2) with ZERO long-helix steps: the logged v.g is
~1.3e-8 on a typical step but reaches 0.21-0.43 on a few steps per epoch.

QUESTION (registered before computing). List the sequence of every update step with
|v.g| > THRESHOLD = 1e-4 in all 10 epochs; count them per epoch; say whether they are the
same sequences every epoch; and intersect them with the sequences whose allowed pairs
admit a run of 31+ stacked pairs (h dropped from G(x); Phase 0 P0.1 counted 13 over
TR0 + TS0). User's hypothesis: if the high steps are exactly those sequences, that is the
whole answer, and the registered S1 exception (a helix > 30 pairs in the PREDICTED or
REFERENCE structure) was stated on the wrong criterion (what the sequence ADMITS).

STEP -> SEQUENCE. The per-step logs hold seq_len, not the file name, so the data order
is reconstructed with MXfold2's own BPseqDataset and a real DataLoader(batch_size=1,
shuffle=True), from the CPU RNG state at the start of each epoch: for epoch 1 the state
after torch.manual_seed(seed), random.seed(seed) and build_model, as mxfold2/train.py
does; for epoch e > 1 the run's RNG sidecar written with the epoch e-1 checkpoint. The
reconstruction is ACCEPTED only if it reproduces all 10,814 logged sequence lengths in
order, in every epoch; any mismatch stops the analysis.

Run (CPU only): CUDA_VISIBLE_DEVICES= .venv-mxfold2/bin/python scripts/mxfold2_sgd_vg_steps.py
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import sys

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402

RUN_ID = "2026-10-10_mxfold2_sgd_vg_steps"
OUT = R / "runs" / RUN_ID
DATA = R / ".cache" / "mxfold2-data"
GRUNS = R / "runs" / "2026-10-08_mxfold2_gauge_runs"
RUN = GRUNS / "seeds" / "sgd_seed20261006_w1.0"
SEED = 20261006
EPOCHS = 10
THRESHOLD = 1e-4
PHASE0_N_LONG_RUN = 13  # runs/2026-10-08_mxfold2_gauge_phase0: P0_1_masks.n_h_dropped_long_helix

QUESTION = {
    "user": "list every step with |v.g| above 1e-4 for all 10 epochs; intersect with the 13 "
            "sequences admitting a helix longer than 30 pairs (h dropped from G(x)); report the "
            "intersection, the count per epoch, and whether the high steps are the same "
            "sequences every epoch. If they are the 13, that is the whole answer and the "
            "registered exception was stated on the wrong criterion.",
    "threshold": THRESHOLD,
    "run": "sgd_seed20261006_w1.0 (the converged SGD seed)",
    "steps_considered": "update steps only (backward = true); every other step leaves the "
                        "weights unchanged",
}


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def long_run_set():
    """Phase 0 P0.1's criterion, sequence by sequence: Masks(seq).h_in_G is False."""
    from scripts.mxfold2_gauge import Masks
    from scripts.mxfold2_head_checks import bpseq_raw, listed

    found = {"TR0": [], "TS0": []}
    for split, lst in (("TR0", "TR0-canonicals.lst"), ("TS0", "TS0-canonicals.lst")):
        for f in listed(lst):
            seq, _ = bpseq_raw(f)
            if not Masks(seq).h_in_G:
                found[split].append(str(f.relative_to(DATA)))
    return found


def epoch_orders(n):
    """The training data order of every epoch, from the recorded CPU RNG states."""
    import torch
    from torch.utils.data import DataLoader

    from scripts.mxfold2_decompose import build_model

    def order():
        return [int(b) for b in DataLoader(list(range(n)), batch_size=1, shuffle=True)]

    orders = {}
    torch.manual_seed(SEED)
    random.seed(SEED)
    build_model(GRUNS / "arch.conf")
    orders[1] = order()
    for e in range(2, EPOCHS + 1):
        side = torch.load(RUN / "checkpoints" / f"rng-epoch-{e - 1}.pt", map_location="cpu",
                          weights_only=False)
        torch.set_rng_state(side["cpu"])
        orders[e] = order()
    return orders


def main():
    import torch

    torch.set_num_threads(2)
    out = {"run_id": RUN_ID, "status": "REGISTERED, not yet computed", "question": QUESTION,
           **git_provenance(R)}
    write(out)

    lr = long_run_set()
    n_lr = len(lr["TR0"]) + len(lr["TS0"])
    out["long_run_sequences"] = {**lr, "n_TR0": len(lr["TR0"]), "n_TS0": len(lr["TS0"]),
                                 "n_total": n_lr, "phase0_recorded_total": PHASE0_N_LONG_RUN,
                                 "matches_phase0": n_lr == PHASE0_N_LONG_RUN}
    write(out)
    if n_lr != PHASE0_N_LONG_RUN:
        out["status"] = f"STOPPED: {n_lr} long-run sequences, Phase 0 recorded {PHASE0_N_LONG_RUN}"
        write(out)
        return 1

    from mxfold2.dataset import BPseqDataset
    cwd = os.getcwd()
    os.chdir(DATA)  # the list holds paths relative to the data root, as in training
    try:
        ds = BPseqDataset(str(DATA / "data" / "TR0-canonicals.lst"))
        names = [ds[i][0] for i in range(len(ds))]
        lens = np.array([len(ds[i][1]) for i in range(len(ds))])
    finally:
        os.chdir(cwd)
    orders = epoch_orders(len(ds))
    lr_tr0 = set(lr["TR0"])

    per_epoch, high_sets, recon = {}, {}, {}
    hist_edges = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, np.inf]
    for e in range(1, EPOCHS + 1):
        z = np.load(RUN / "gauge" / f"steps-epoch-{e}.npz")
        order = np.array(orders[e])
        ok = bool(np.array_equal(lens[order], z["seq_len"]))
        recon[str(e)] = {"reproduces_all_logged_lengths": ok,
                         "n_mismatched_steps": int((lens[order] != z["seq_len"]).sum())}
        if not ok:
            out["order_reconstruction"] = recon
            out["status"] = f"STOPPED: the reconstructed order does not reproduce epoch {e}"
            write(out)
            return 1
        bw, vg, lh = z["backward"], np.abs(z["vg"]), z["long_helix"]
        hi = np.flatnonzero(bw & (vg > THRESHOLD))
        hs = {names[order[i]]: float(vg[i]) for i in hi}
        high_sets[e] = set(hs)
        counts, _ = np.histogram(vg[bw], bins=hist_edges)
        per_epoch[str(e)] = {
            "n_update_steps": int(bw.sum()), "n_long_helix_steps": int(lh.sum()),
            "n_high": len(hs), "high_steps": dict(sorted(hs.items(), key=lambda kv: -kv[1])),
            "high_in_long_run_set": sorted(set(hs) & lr_tr0),
            "high_not_in_long_run_set": sorted(set(hs) - lr_tr0),
            "long_run_not_high": sorted(lr_tr0 - set(hs)),
            "abs_vg_decade_counts": {f"[{a:g}, {b:g})": int(c) for a, b, c in
                                     zip(hist_edges[:-1], hist_edges[1:], counts, strict=True)},
            "sum_abs_vg_high_over_all": float(vg[hi].sum() / vg[bw].sum())}
    out["order_reconstruction"] = recon
    union = set().union(*high_sets.values())
    inter = set.intersection(*high_sets.values())
    out["per_epoch"] = per_epoch
    out["summary"] = {
        "n_high_per_epoch": {str(e): len(s) for e, s in high_sets.items()},
        "same_sequences_every_epoch": all(s == high_sets[1] for s in high_sets.values()),
        "union": sorted(union), "n_union": len(union),
        "in_every_epoch": sorted(inter), "n_in_every_epoch": len(inter),
        "union_cap_long_run_TR0": sorted(union & lr_tr0),
        "union_minus_long_run_TR0": sorted(union - lr_tr0),
        "long_run_TR0_minus_union": sorted(lr_tr0 - union),
        "high_steps_are_exactly_the_long_run_TR0_sequences_every_epoch":
            all(s == lr_tr0 for s in high_sets.values()),
    }
    out["status"] = "complete"
    write(out)
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
