#!/usr/bin/env python3
"""The shipped MXfold2 model on TS0, scored exactly as the gauge-study arms were.

Asked by the user 2026-10-11: the 10-epoch TR0 retrains score F about 0.52 on TS0. To
separate "our retrain undertrains" from "our scorer differs", the released TrainSetAB
weights are scored on the same 1,305 TS0 sequences (the seed study's registered
evaluation set, sha256-checked) with the gauge evaluator's own code path:
scripts/mxfold2_decompose.read_bpseq and .predict, per-sequence true/false positive and
false negative base pairs, and scripts/mxfold2_gauge_eval.accuracy. Recorded alongside:
the same model scored on 2026-10-07 through the command-line tool and a different scorer
(runs/2026-10-07_mxfold2_ts0_inference), structure by structure; the arms' F from
stage 2b (runs/2026-10-08_mxfold2_gauge_eval); and the published TS0 figure for a
TR0-trained MXfold2 as recorded in that 2026-10-07 artifact (Sato et al. 2021, Table 2).

The shipped weights were trained on TrainSetA + TrainSetB, not TR0, so they bound the
scorer question, not the epoch-budget question; the published TR0-trained figure is the
like-for-like comparator for the retrains.

Run (CPU only): CUDA_VISIBLE_DEVICES= .venv-mxfold2/bin/python scripts/mxfold2_shipped_ts0_f.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402

RUN_ID = "2026-10-11_mxfold2_shipped_ts0_f"
OUT = R / "runs" / RUN_ID
PARAM = R / "vendor" / "mxfold2" / "mxfold2" / "models" / "TrainSetAB.pth"
INFER = R / "runs" / "2026-10-07_mxfold2_ts0_inference"
EVAL = R / "runs" / "2026-10-08_mxfold2_gauge_eval" / "metrics.json"


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def main():
    import torch

    import scripts.mxfold2_gauge_eval as ev
    from scripts.mxfold2_decompose import build_model, dot_bracket, predict, read_bpseq

    torch.set_num_threads(4)
    out = {"run_id": RUN_ID, "status": "running", "model": "shipped TrainSetAB (51b2136)",
           "param_sha256": hashlib.sha256(PARAM.read_bytes()).hexdigest(), **git_provenance(R)}
    write(out)
    model = build_model(ev.LIVE / "arch.conf")
    sd = torch.load(PARAM, map_location="cpu", weights_only=True)
    model.load_state_dict(sd.get("model_state_dict", sd))
    model.eval()
    entries = json.loads((ev.LIVE / "metrics.json").read_text())["evaluation_set"]["entries"]
    tp, fp, fn, dbs = [], [], [], []
    for e in entries:
        seq, partner = read_bpseq(ev.DATA / e["file"])
        assert hashlib.sha256(seq.encode()).hexdigest() == e["sha256"], e["file"]
        with torch.no_grad():
            param = model.make_param([seq])
        _, db, bp = predict(model, seq, param)
        ref = {(i, partner[i]) for i in range(1, len(partner)) if partner[i] > i}
        pred = {(i, bp[i]) for i in range(1, len(bp)) if bp[i] > i}
        tp.append(len(pred & ref))
        fp.append(len(pred - ref))
        fn.append(len(ref - pred))
        dbs.append(db)
        assert db == dot_bracket(bp)
    acc = ev.accuracy(np.array(tp), np.array(fp), np.array(fn))

    # the 2026-10-07 command-line predictions of the same model, structure by structure
    cli = {r["name"]: r["structure"] for r in map(json.loads, (INFER / "pred_A_cpu.txt")
                                                  .read_text().splitlines()) if r}
    names = [pathlib.Path(e["file"]).stem for e in entries]
    infer = json.loads((INFER / "metrics.json").read_text())
    stage = json.loads(EVAL.read_text())["2b"]["results"]
    arms = stage["main"]["arms"]
    pub = infer["published_ts0_for_a_different_model"]
    out.update({
        "shipped_same_scorer_as_arms": acc,
        "shipped_command_line_2026_10_07": infer["accuracy_A_cpu"],
        "command_line_structures_identical": {
            "n_compared": sum(n in cli for n in names), "n_cli": len(cli),
            "n_identical": sum(cli.get(n) == db for n, db in zip(names, dbs, strict=True))},
        "published_TR0_model_TS0": pub,
        "arms_F_mean_per_sequence": {a: {"mean": v["F_mean"], "sd": v["F_sd"],
                                         "n_seeds": v["n_seeds"]} for a, v in arms.items()
                                     if v.get("F_mean") is not None},
        "sgd_seed20261006_F": stage["F_per_model"]["sgd_seed20261006_w1.0"],
    })
    shipped = acc["F_mean_per_sequence"]
    out["gaps_F_mean_per_sequence"] = {
        f"published_minus_{a}": pub["F"] - v["mean"] for a, v in
        out["arms_F_mean_per_sequence"].items()} | {
        f"shipped_minus_{a}": shipped - v["mean"] for a, v in
        out["arms_F_mean_per_sequence"].items()}
    out["status"] = "complete"
    write(out)
    print(json.dumps({k: out[k] for k in ("shipped_same_scorer_as_arms",
                                          "command_line_structures_identical",
                                          "arms_F_mean_per_sequence", "gaps_F_mean_per_sequence")},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
