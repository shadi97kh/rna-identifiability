#!/usr/bin/env python3
"""Why does v.g break on 11 of the 13 long-run TR0 sequences? A breaker against a non-breaker.

Asked by the user 2026-10-10, after runs/2026-10-10_mxfold2_sgd_vg_steps found that every
large v.g step of the converged SGD run (seed 20261006) is one of 11 TR0 sequences whose
allowed pairs admit a 31+ pair run, with ZERO steps flagged as having a predicted or
reference helix longer than 30 pairs. Compared: bpRNA_RFAM_22316 (|v.g| 0.396 at epoch 1)
against bpRNA_RFAM_20959 (below 1e-5 every epoch), both among the 13.

For each, at the run's epoch-1 checkpoint, on the CPU (float32 DP, eval mode, d = 0.37 as
registered for Phase 0's P0.2 v-symmetry check):
  theta(w+dv) - theta(w): score_helix_length (all 31 entries, expected -2d(m-1)), the S
    matrix score_helix_stacking (every entry, expected +d), every other positional table
    (expected 0);
  the scored energy of the predicted structure (loss-augmented, as in training) and of the
    reference under both weights, by MXfold2's constrained DP, with the DP's own counts;
  h.count for each scored structure (+1 per counted S entry, -2(m-1) per counted
    helix_length[m]): the energy shift is d * h.count when theta shifts by d h;
  the longest run the allowed pairs admit, and the longest helix of each scored structure,
    both from the dataset structure and from the pair list the DP RETURNS (which is what the
    training hook measured).
Then, for all 13 long-run sequences: the reference's longest helix, h.count of the scored
reference and prediction, and whether the sequence was a breaker.

FINDING THAT SHAPED THE SECOND PART (recorded here because it was found while exploring,
before this script was written): MXfold2's traceback for a helix longer than 30 pairs
(TBType::C_HELIX_E, vendor/mxfold2/mxfold2/src/fold/zuker.cpp:389) loops k < m where the
ordinary helix traceback (C_HELIX, :377) loops k <= m, so the RETURNED structure omits one
pair of every helix longer than 30, while the score and counts include it. A helix of
31-60 pairs therefore comes back as segments of at most 30, and the hook's long-helix flag
cannot fire.

Run (CPU only): CUDA_VISIBLE_DEVICES= .venv-mxfold2/bin/python scripts/mxfold2_vg_breaker.py
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
import sys

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402

RUN_ID = "2026-10-10_mxfold2_vg_breaker"
OUT = R / "runs" / RUN_ID
DATA = R / ".cache" / "mxfold2-data"
GRUNS = R / "runs" / "2026-10-08_mxfold2_gauge_runs"
CKPT = GRUNS / "seeds" / "sgd_seed20261006_w1.0" / "checkpoints" / "epoch-1"
STEPS_ART = R / "runs" / "2026-10-10_mxfold2_sgd_vg_steps" / "metrics.json"
PAIR = {"breaker": "bpRNA_RFAM_22316", "non_breaker": "bpRNA_RFAM_20959"}
D = 0.37  # Phase 0 P0.2's registered delta


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def longest_run(al):
    """The longest run of consecutive stacked pairs the allowed-pair matrix admits."""
    run, k, m = al.copy(), 1, al.shape[0]
    while True:
        nxt = np.zeros_like(run)
        nxt[: m - k, k:] = al[k:, : m - k]
        cand = run & nxt
        if not cand.any():
            return k
        run, k = cand, k + 1


def main():
    import torch
    from mxfold2 import interface
    from mxfold2.dataset import BPseqDataset

    from scripts.mxfold2_decompose import build_model
    from scripts.mxfold2_gauge import (
        BIAS_KEY,
        HELIX_W_KEY,
        MAX_HELIX,
        allowed_matrix,
        v_dot_grad,
    )
    from scripts.mxfold2_gauge_hooks import _longest_helix
    from scripts.mxfold2_head_checks import loss_fn

    torch.set_num_threads(2)
    out = {"run_id": RUN_ID, "status": "running", "checkpoint": str(CKPT.relative_to(R)),
           "d": D, "mode": "eval (no dropout); CPU; float32 DP", **git_provenance(R)}
    write(out)

    model = build_model(GRUNS / "arch.conf")
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True)
                          ["model_state_dict"])
    model.eval()
    m2 = copy.deepcopy(model)
    sd = m2.state_dict()
    sd[BIAS_KEY][0] += D
    sd[HELIX_W_KEY][0, 2:MAX_HELIX + 1] -= 2 * D
    m2.load_state_dict(sd)
    m2.eval()
    w_hl = 2.0 * torch.clamp(torch.arange(MAX_HELIX + 1, dtype=torch.float64) - 1, min=0)

    def dp(mdl, param, seq, **kw):
        p = {"turner": {k: v.detach().cpu() for k, v in param["turner"].items()},
             "positional": {k: v.detach().cpu() for k, v in param["positional"].items()}}
        p = {k: mdl.clear_count(v) for k, v in p.items()}
        v, _, pair = interface.predict_mxfold(seq, p, max_helix_length=MAX_HELIX, **kw)
        return float(v), list(pair), p["positional"]

    def h_dot(cnt):
        nS = float(cnt["count_helix_stacking"].double().sum())
        hl = cnt["count_helix_length"].double()
        return nS - float((hl * w_hl).sum()), nS, {str(int(m)): float(hl[m])
                                                   for m in torch.nonzero(hl).flatten()}

    def analyse(name, full):
        _, seq, pairs = BPseqDataset.read(None, f"data/bpRNA_dataset-canonicals/TR0/{name}.bpseq")
        ref = pairs.tolist()
        L = len(seq)
        r = {"length": L, "longest_run_admitted": longest_run(allowed_matrix(seq))}
        with torch.no_grad():
            a, b = model.make_param([seq])[0], m2.make_param([seq])[0]
        pa, pb = a["positional"], b["positional"]
        _, pred, _ = dp(model, a, seq, max_internal_length=30, reference=ref,
                        loss_pos_paired=0.5, loss_neg_paired=0.005)
        for key, st in (("pred", pred), ("ref", ref)):
            ea, ret, ca = dp(model, a, seq, max_internal_length=L, constraint=st)
            eb, _, _ = dp(m2, b, seq, max_internal_length=L, constraint=st)
            hd, nS, hlc = h_dot(ca)
            dth = sum(float((ca[k].double() * (pb["score_" + k[6:]] - pa["score_" + k[6:]])
                             .double()).sum()) for k in ca if k.startswith("count_"))
            r[key] = {"E_w": ea, "E_w_plus_dv": eb, "dE": eb - ea, "dE_over_d": (eb - ea) / D,
                      "h_dot_count": hd, "dE_predicted_d_h_dot_count": D * hd,
                      "dE_from_counts_times_actual_dtheta": dth,
                      "n_S_entries_counted": nS, "helix_length_counts": hlc,
                      "longest_helix_scored_structure": _longest_helix(st),
                      "longest_helix_returned_by_DP": _longest_helix(ret),
                      "pairs_missing_from_DP_return": [
                          [i, st[i]] for i in range(1, len(st))
                          if st[i] > i and ret[i] != st[i]]}
        model.zero_grad()
        loss_fn(model)([seq], [pairs]).sum().backward()
        r["v_dot_g_eval_mode"] = v_dot_grad(model)
        if full:
            dhl = (pb["score_helix_length"] - pa["score_helix_length"]).double()
            r["theta_shift"] = {
                "score_helix_length": {"actual": dhl.tolist(), "expected": (-D * w_hl).tolist(),
                                       "max_abs_dev": float((dhl + D * w_hl).abs().max())},
                "S": {"shape": list(pa["score_helix_stacking"].shape), "expected": D,
                      "max_abs_dev_every_entry": float(
                          (pb["score_helix_stacking"] - pa["score_helix_stacking"]
                           - D).abs().max())},
                "other_positional_tables_max_abs": {
                    k: float((pb[k] - pa[k]).abs().max()) for k in pa if k.startswith("score_")
                    and k not in ("score_helix_stacking", "score_helix_length")}}
        return r

    steps = json.loads(STEPS_ART.read_text())
    thirteen = [pathlib.Path(p).stem for p in steps["long_run_sequences"]["TR0"]]
    breakers = {pathlib.Path(p).stem for p in steps["summary"]["in_every_epoch"]}
    cwd = os.getcwd()
    os.chdir(DATA)
    try:
        out["pair"] = {role: {"sequence": n, **analyse(n, True)} for role, n in PAIR.items()}
        write(out)
        out["all_13"] = {n: {"breaker": n in breakers, **analyse(n, False)} for n in thirteen}
    finally:
        os.chdir(cwd)
    a13 = out["all_13"]
    long_scored = {n: max(v["pred"]["longest_helix_scored_structure"],
                          v["ref"]["longest_helix_scored_structure"]) > MAX_HELIX
                   for n, v in a13.items()}
    out["summary"] = {
        "breaker_iff_a_scored_structure_has_a_helix_over_30":
            all(long_scored[n] == a13[n]["breaker"] for n in a13),
        "breaker_iff_h_dot_count_nonzero":
            all((abs(a13[n]["pred"]["h_dot_count"]) + abs(a13[n]["ref"]["h_dot_count"]) > 0)
                == a13[n]["breaker"] for n in a13),
        "hook_flag_could_fire_on_any_of_the_13": any(
            max(v["pred"]["longest_helix_returned_by_DP"],
                v["ref"]["longest_helix_returned_by_DP"]) > MAX_HELIX for v in a13.values()),
        "table": {n: {"breaker": v["breaker"], "run_admitted": v["longest_run_admitted"],
                      "ref_helix": v["ref"]["longest_helix_scored_structure"],
                      "ref_helix_returned": v["ref"]["longest_helix_returned_by_DP"],
                      "pred_helix": v["pred"]["longest_helix_scored_structure"],
                      "h_ref": v["ref"]["h_dot_count"], "h_pred": v["pred"]["h_dot_count"]}
                  for n, v in a13.items()},
    }
    out["status"] = "complete"
    write(out)
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
