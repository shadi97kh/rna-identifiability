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
The predicted structure is the one the loss-augmented DP SCORED: its returned pair list is
restored (restore_dropped_pairs) and accepted only if the restored structure's
constrained-DP counts equal the prediction call's counts exactly.
Then, for all 13 long-run sequences: the reference's longest helix, h.count of the scored
reference and prediction, and whether the sequence was a breaker.

WHERE THE PAIR GOES (checked per helix, every long helix of the 13): the C_HELIX_E traceback
records offsets k-2 for k = 2..m-1 with m = max_helix + 1 = 31, i.e. offsets 0-28 from the
outermost pair, and hands off to the E state at offset m-1 = 30, whose traceback records
offset 30 inward. Offset 29, the 30th pair, is never recorded. A scored helix of m >= 31
pairs therefore comes back SPLIT AT THE CAP: an outer run of 29, a one-pair gap, an inner
run of m - 30. Both are <= 30 for every m <= 60, so a flag reading the returned structure
cannot fire for any helix of 31-60 pairs (TR0's longest admitted run is 41).

CONFIGURATION DEPENDENCE (added 2026-10-10): the same scored structures are rescored by the DP
at other --max-helix-length H against the rule read from the DP (uncompensated iff m >
min(H + 1, 30), shift 2d(m - c)), and the TR0 exception-set size is counted for each H.

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


def helices(p):
    """Maximal stacked runs of a 1-based partner list: [(i, j, m)] with (i, j) outermost."""
    out, n = [], len(p) - 1
    for i in range(1, n + 1):
        j = p[i]
        if j <= i or (i > 1 and j < n and p[i - 1] == j + 1):
            continue
        m = 0
        while i + m <= n and p[i + m] == j - m and j - m > i + m:
            m += 1
        out.append((i, j, m))
    return out


def split_of(st, ret, cap):
    """For every helix of the scored structure longer than `cap`: which of its pairs (0-based
    offset from the outermost) the DP's returned structure lacks, and the runs it returns."""
    rows = []
    for i, j, m in helices(st):
        if m <= cap:
            continue
        present = [ret[i + o] == j - o for o in range(m)]
        runs, cur = [], 0
        for x in present:
            if x:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        rows.append({"outer_pair": [i, j], "m": m,
                     "missing_offsets": [o for o, x in enumerate(present) if not x],
                     "returned_runs": runs})
    return rows


def restore_dropped_pairs(ret):
    """Undo the C_HELIX_E traceback drop: a helix of 31+ pairs comes back as a run of exactly
    29 stacked pairs (i..i+28), an unpaired pair of positions (i+29, j-29), then (i+30, j-30)
    paired. The caller accepts the result only if its constrained-DP counts equal the counts
    of the DP call that produced `ret`."""
    p, n, added = list(ret), len(ret) - 1, []
    for i in range(1, n + 1):
        j = p[i]
        if j <= i or (i > 1 and j < n and p[i - 1] == j + 1):
            continue  # not the outermost pair of a helix
        k = 0
        while i + k <= n and p[i + k] == j - k and j - k > i + k:
            k += 1
        a, b = i + 29, j - 29
        if k == 29 and a + 1 < b - 1 and p[a] == 0 and p[b] == 0 and p[a + 1] == b - 1:
            p[a], p[b] = b, a
            added.append([a, b])
    return p, added


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

    def dp(mdl, param, seq, H=MAX_HELIX, **kw):
        p = {"turner": {k: v.detach().cpu() for k, v in param["turner"].items()},
             "positional": {k: v.detach().cpu() for k, v in param["positional"].items()}}
        p = {k: mdl.clear_count(v) for k, v in p.items()}
        v, _, pair = interface.predict_mxfold(seq, p, max_helix_length=H, **kw)
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
        _, pred_ret, c_pred = dp(model, a, seq, max_internal_length=30, reference=ref,
                                 loss_pos_paired=0.5, loss_neg_paired=0.005)
        pred, added = restore_dropped_pairs(pred_ret)
        _, _, c_chk = dp(model, a, seq, max_internal_length=L, constraint=pred)
        scored[name] = {"seq": seq, "a": a, "b": b, "pred": pred, "ref": ref}
        r["pred_restoration"] = {
            "pairs_restored": added,
            "validated_counts_equal_the_prediction_call": all(
                torch.equal(c_chk[k], c_pred[k]) for k in c_pred if k.startswith("count_"))}
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
                      "longest_helix_returned_by_DP": _longest_helix(pred_ret if key == "pred"
                                                                     else ret),
                      "long_helix_split_in_DP_return": split_of(
                          st, pred_ret if key == "pred" else ret, MAX_HELIX),
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

    scored = {}
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
        "all_pred_restorations_validated": all(
            v["pred_restoration"]["validated_counts_equal_the_prediction_call"]
            for v in a13.values()),
        "breaker_iff_a_scored_structure_has_a_helix_over_30":
            all(long_scored[n] == a13[n]["breaker"] for n in a13),
        "breaker_iff_h_dot_count_nonzero":
            all((abs(a13[n]["pred"]["h_dot_count"]) + abs(a13[n]["ref"]["h_dot_count"]) > 0)
                == a13[n]["breaker"] for n in a13),
        "every_long_helix_split_at_offset_29_into_29_and_m_minus_30": all(
            h["missing_offsets"] == [29] and h["returned_runs"] == [29, h["m"] - 30]
            for v in a13.values() for key in ("pred", "ref")
            for h in v[key]["long_helix_split_in_DP_return"]),
        "long_helix_lengths_seen": sorted({h["m"] for v in a13.values() for key in ("pred", "ref")
                                           for h in v[key]["long_helix_split_in_DP_return"]}),
        "n_long_helices_checked": sum(len(v[key]["long_helix_split_in_DP_return"])
                                      for v in a13.values() for key in ("pred", "ref")),
        "hook_flag_could_fire_on_any_of_the_13": any(
            max(v["pred"]["longest_helix_returned_by_DP"],
                v["ref"]["longest_helix_returned_by_DP"]) > MAX_HELIX for v in a13.values()),
        "table": {n: {"breaker": v["breaker"], "run_admitted": v["longest_run_admitted"],
                      "ref_helix": v["ref"]["longest_helix_scored_structure"],
                      "ref_helix_returned": v["ref"]["longest_helix_returned_by_DP"],
                      "pred_helix": v["pred"]["longest_helix_scored_structure"],
                      "pred_helix_returned": v["pred"]["longest_helix_returned_by_DP"],
                      "h_ref": v["ref"]["h_dot_count"], "h_pred": v["pred"]["h_dot_count"]}
                  for n, v in a13.items()},
    }
    # configuration dependence of the exception: the SAME scored structures rescored by the DP
    # at other --max-helix-length H (scoring rule only; nothing is retrained or re-predicted).
    # Reading the DP: a helix of m pairs is uncompensated when m > c(H) = min(H + 1, 30), and
    # shifts by 2d(m - c(H)); so any H >= 29 gives the released 31-pair threshold.
    cfg = {}
    for role, n in PAIR.items():
        sc = scored[n]
        for key in ("pred", "ref"):
            hs = [m for _, _, m in helices(sc[key])]
            for H in (10, 20, 28, 29, 30, 32, 40):
                c = min(H + 1, MAX_HELIX)
                L = len(sc["seq"])
                ea, _, _ = dp(model, sc["a"], sc["seq"], H=H, max_internal_length=L,
                              constraint=sc[key])
                eb, _, _ = dp(m2, sc["b"], sc["seq"], H=H, max_internal_length=L,
                              constraint=sc[key])
                pred_ = 2 * sum(m - c for m in hs if m > c)
                cfg[f"{role}/{key}/H={H}"] = {"c": c, "helix_lengths": sorted(hs, reverse=True),
                                              "dE_over_d": (eb - ea) / D,
                                              "predicted_2_sum_m_minus_c": pred_,
                                              "agree": abs((eb - ea) / D - pred_) <= 1e-3}
    from scripts.mxfold2_head_checks import bpseq_raw, listed
    runs = [longest_run(allowed_matrix(bpseq_raw(f)[0])) for f in listed("TR0-canonicals.lst")]
    runs = np.array(runs)
    out["configuration_dependence"] = {
        "rule_from_the_DP": "a helix of m pairs is uncompensated when m > c(H) = min(H + 1, 30) "
                            "(H = --max-helix-length; 30 = the last index of LengthLayer(31)); "
                            "its score shifts by 2d(m - c(H))",
        "rescoring_check": cfg,
        "rescoring_check_all_agree": all(v["agree"] for v in cfg.values()),
        "TR0_longest_admitted_run": {"n_sequences": int(runs.size), "max": int(runs.max()),
                                     "median": float(np.median(runs))},
        "TR0_sequences_in_the_exception_set_by_H": {
            str(H): {"threshold_run": min(H + 1, MAX_HELIX) + 1,
                     "n": int((runs >= min(H + 1, MAX_HELIX) + 1).sum())}
            for H in (10, 15, 20, 25, 28, 29, 30, 32, 40)},
    }
    write(out)
    out["status"] = "complete"
    write(out)
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
