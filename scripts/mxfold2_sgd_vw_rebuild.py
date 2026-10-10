#!/usr/bin/env python3
"""POST-HOC: how much of the converged SGD run's v.w drift do the 11 high-|v.g| steps carry?

Asked by the user 2026-10-10, after the registered S1 failed its conservation criterion on
SGD seed 20261006 (|v.w(10) - v.w_longonly(10)| = 0.010449 > 1e-2, with zero long-helix
steps) and runs/2026-10-10_mxfold2_sgd_vg_steps found that every step with |v.g| > 1e-4 is
one of 11 TR0 sequences admitting a 31+ pair run, the same 11 every epoch. THIS IS NOT A
REGISTERED TEST and changes no verdict: the registered S1 result is recomputed here with
the evaluator's unchanged code and recorded alongside, as it stands.

METHOD. The registered reconstruction (scripts/mxfold2_gauge_eval.s1_sgd): MXfold2 SGD,
Nesterov, momentum 0.9, lr 1e-3; on every update step buf = 0.9 buf + g (buf = g on the
first), v.w -= lr (g + 0.9 buf), accumulated in float64 from the logged v.g. It is linear
in the v.g sequence, so the drift splits EXACTLY into
  high   the 11 sequences' steps alone (every other step's v.g set to 0, momentum kept),
  rest   every other step alone (the 11 steps' v.g set to 0),
with high + rest = full (asserted). Signed quantities throughout: a 99.97% share of sum
|v.g| does not imply the same share of the signed drift, because of momentum and
cancellation. Reported per epoch and over 10 epochs:
  raw signed sum of v.g (high, rest), and the reconstructed signed drift (high, rest, full);
  the actual drift v.w(10) - v.w(0) and actual minus the high steps' contribution, i.e. the
  drift left once the 11 are removed, including the reconstruction mismatch (float32
  weight rounding), set against the full 0.010449 and, for scale only, against the
  registered 1e-2.
The 11 are identified by sequence (the vg_steps artifact's in_every_epoch list, through
its order reconstruction) and checked to be exactly the |v.g| > 1e-4 steps of each epoch.
A second, labelled variant removes all 13 long-run sequences.

Run (CPU only): CUDA_VISIBLE_DEVICES= .venv-mxfold2/bin/python scripts/mxfold2_sgd_vw_rebuild.py
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

RUN_ID = "2026-10-10_mxfold2_sgd_vw_rebuild"
OUT = R / "runs" / RUN_ID
STEPS_ART = R / "runs" / "2026-10-10_mxfold2_sgd_vg_steps" / "metrics.json"
NAME = "sgd_seed20261006_w1.0"
EPOCHS = 10
MU, LR = 0.9, 1e-3   # the registered S1 recurrence (mxfold2_gauge_eval.SGD_MU, SGD_LR)


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def rebuild(vgs, v0):
    """The registered recurrence over epochs of update-step v.g; v.w after each epoch."""
    r, b, first, trace = v0, 0.0, True, {0: v0}
    for e, vg in enumerate(vgs, start=1):
        for g in vg:
            if first:
                b, first = g, False
            else:
                b = MU * b + g
            r -= LR * (g + MU * b)
        trace[e] = r
    return trace


def main():
    import scripts.mxfold2_gauge_eval as ev
    from scripts.mxfold2_seed_study import recorded_seeds
    from scripts.mxfold2_sgd_vg_steps import DATA, RUN, epoch_orders
    from scripts.mxfold2_sgd_vg_steps import EPOCHS as E2

    assert (MU, LR) == (ev.SGD_MU, ev.SGD_LR) and E2 == EPOCHS
    out = {"run_id": RUN_ID, "status": "POST-HOC, not a registered test; not yet computed",
           "post_hoc": True, "asked": "user, 2026-10-10", **git_provenance(R)}
    write(out)

    # the registered S1, recomputed with the evaluator's unchanged code (status-complete SGD
    # runs only, i.e. this seed alone)
    s1 = ev.s1_sgd(recorded_seeds())
    r1 = s1["runs"][NAME]
    out["registered_S1_unchanged"] = {
        "n": len(s1["runs"]), "runs": sorted(s1["runs"]), "S1": s1["S1"],
        "i_reconstruction_ok": r1["i_reconstruction_ok"],
        "max_reconstruction_dev": r1["max_reconstruction_dev"],
        "ii_conservation_ok": r1["ii_conservation_ok"],
        "conservation_residual": r1["conservation_residual"], "bar": 1e-2,
        "verdict": "FAIL (criterion ii); unchanged by this analysis"}
    write(out)

    art = json.loads(STEPS_ART.read_text())
    eleven = set(art["summary"]["in_every_epoch"])
    thirteen = set(art["long_run_sequences"]["TR0"])
    assert len(eleven) == 11 and eleven <= thirteen and len(thirteen) == 13

    from mxfold2.dataset import BPseqDataset
    cwd = os.getcwd()
    os.chdir(DATA)
    try:
        ds = BPseqDataset(str(DATA / "data" / "TR0-canonicals.lst"))
        names = [ds[i][0] for i in range(len(ds))]
        lens = np.array([len(ds[i][1]) for i in range(len(ds))])
    finally:
        os.chdir(cwd)
    orders = epoch_orders(len(ds))

    full, hi11, rest11, rest13, per_epoch, all_abs = [], [], [], [], {}, []
    for e in range(1, EPOCHS + 1):
        z = np.load(RUN / "gauge" / f"steps-epoch-{e}.npz")
        order = np.array(orders[e])
        assert np.array_equal(lens[order], z["seq_len"]), f"order mismatch in epoch {e}"
        bw = z["backward"]
        vg = z["vg"][bw]
        seqs = np.array([names[i] for i in order])[bw]
        in11 = np.isin(seqs, list(eleven))
        in13 = np.isin(seqs, list(thirteen))
        assert np.array_equal(in11, np.abs(vg) > 1e-4), f"epoch {e}: the 11 != the >1e-4 steps"
        full.append(vg)
        all_abs.append(np.abs(vg))
        hi11.append(np.where(in11, vg, 0.0))
        rest11.append(np.where(in11, 0.0, vg))
        rest13.append(np.where(in13, 0.0, vg))
        per_epoch[str(e)] = {"n_update_steps": int(bw.sum()), "n_high": int(in11.sum()),
                             "signed_sum_vg_high": float(vg[in11].sum()),
                             "signed_sum_vg_rest": float(vg[~in11].sum()),
                             "sum_abs_vg_high": float(np.abs(vg[in11]).sum()),
                             "sum_abs_vg_rest": float(np.abs(vg[~in11]).sum()),
                             "median_abs_vg_all_update_steps": float(np.median(np.abs(vg))),
                             "median_abs_vg_rest": float(np.median(np.abs(vg[~in11]))),
                             "max_abs_vg_rest": float(np.abs(vg[~in11]).max())}

    v0 = float(r1["actual"]["0"])
    actual = {int(k): float(v) for k, v in r1["actual"].items()}
    t_full, t_hi, t_rest, t_rest13 = (rebuild(x, v0) for x in (full, hi11, rest11, rest13))
    for e in range(EPOCHS + 1):
        assert abs((t_hi[e] - v0) + (t_rest[e] - v0) - (t_full[e] - v0)) <= 1e-12
        assert abs(t_full[e] - float(r1["reconstruction"][str(e)])) <= 1e-12
    for e in range(1, EPOCHS + 1):
        pe = per_epoch[str(e)]
        pe["drift_high"] = (t_hi[e] - t_hi[e - 1])
        pe["drift_rest"] = (t_rest[e] - t_rest[e - 1])
        pe["drift_full_reconstructed"] = (t_full[e] - t_full[e - 1])
        pe["drift_actual"] = actual[e] - actual[e - 1]

    d_act = actual[EPOCHS] - v0
    d_full, d_hi, d_rest = (t[EPOCHS] - v0 for t in (t_full, t_hi, t_rest))
    left = d_act - d_hi
    out["per_epoch"] = per_epoch
    out["over_10_epochs"] = {
        "v_w_0": v0, "v_w_10_actual": actual[EPOCHS],
        "signed_drift_actual": d_act,
        "signed_drift_reconstructed_full": d_full,
        "signed_drift_from_the_11_high_steps": d_hi,
        "signed_drift_from_every_other_step": d_rest,
        "high_share_of_reconstructed_signed_drift": d_hi / d_full,
        "signed_sum_vg_high": sum(p["signed_sum_vg_high"] for p in per_epoch.values()),
        "signed_sum_vg_rest": sum(p["signed_sum_vg_rest"] for p in per_epoch.values()),
        "median_abs_vg_all_update_steps_pooled": float(np.median(np.concatenate(all_abs))),
        "reconstructed_share_of_actual_drift": d_full / d_act,
        "residual_drift_with_the_11_removed": {
            "from_logged_vg (rest only)": d_rest,
            "actual_minus_high_contribution": left,
            "reconstruction_mismatch (actual - full reconstruction)": d_act - d_full,
            "against_full_0.010449": abs(left) / abs(d_act),
            "against_registered_bar_1e-2_for_scale_only": abs(left) / 1e-2},
        "variant_all_13_long_run_removed": {
            "signed_drift_from_every_other_step": t_rest13[EPOCHS] - v0,
            "actual_minus_the_13_contribution": d_act - (d_full - (t_rest13[EPOCHS] - v0))},
        "trace": {str(e): {"actual": actual[e], "full": t_full[e], "high_only": t_hi[e],
                           "rest_only": t_rest[e]} for e in range(EPOCHS + 1)},
    }
    out["status"] = "complete (POST-HOC; the registered S1 verdict is unchanged: FAIL)"
    write(out)
    print(json.dumps({k: v for k, v in out["over_10_epochs"].items() if k != "trace"},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
