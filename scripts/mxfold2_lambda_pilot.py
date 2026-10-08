"""Lambda pilot for the MXfold2 null-component regulariser (CPU).

Called by `scripts/mxfold2_gauge_study.py --phase pilot` after Phase 0 has passed. All
settings come from the registered PREREG["pilot"] (`cfg`). It uses the same hooks
module the GPU runs use (scripts/mxfold2_gauge_hooks.py), so the regulariser tested here
is the one that will train.

Start: the live seed study's seed-20261006 w=1.0 epoch-1 checkpoint (model and AdamW
state), read only. Arms: lambda = 0 and lambda_f = f * L0 / R0, every arm re-seeded
identically (same sequence order, same dropout masks; the regulariser draws no random
numbers). The training loop mirrors mxfold2/train.py Train.train (batch 1, backward and
step only when loss > 0). Evaluation in eval mode on a disjoint held-out set: mean R
and mean hinge_mix loss WITHOUT the lambda term.

Rule (registered): the smallest f with rho = R(lambda, 500) / R(0, 500) <= rho_max and
delta = L(lambda, 500) / L(0, 500) - 1 <= delta_max; none -> lambda_star = None and the
regulariser arm is not launched.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from mxfold2.loss import StructuredLossWithTurner  # noqa: E402
from mxfold2.train import Train  # noqa: E402

from scripts import mxfold2_gauge_hooks as hooks  # noqa: E402
from scripts.mxfold2_gauge import Masks, null_ratio  # noqa: E402

DATA = R / ".cache" / "mxfold2-data"


def bpseq_raw(path):
    rows = [r.split() for r in path.read_text().split("\n") if r.strip() and not r.startswith("#")]
    return "".join(r[1] for r in rows), [0] + [int(r[2]) for r in rows]


def build(live):
    p = ArgumentParser(fromfile_prefix_chars="@")
    sub = p.add_subparsers()
    Train.add_args(sub)
    args = p.parse_args(["train", f"@{live / 'arch.conf'}", "x.lst", "--loss-func", "hinge_mix"])
    tr = Train()
    model, _ = tr.build_model(args)
    ck = torch.load(live / "seeds" / "seed20261006_w1.0" / "checkpoints" / "epoch-1",
                    map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    opt = tr.build_optimizer(args.optimizer, model, args.lr, args.l2_weight)
    opt.load_state_dict(ck["optimizer_state_dict"])
    return model, opt, args


def evaluate(model, lf, held):
    model.eval()
    lam = hooks.STATE["lam"]
    hooks.STATE["lam"] = 0.0
    Rs, Ls = [], []
    with torch.no_grad():
        for seq, partner in held:
            pp = model.make_param([seq])[0]["positional"]
            Rs.append(float(null_ratio(pp, Masks(seq))))
            Ls.append(float(lf([seq], [torch.tensor(partner)]).sum()))
    hooks.STATE["lam"] = lam
    model.train()
    return float(np.mean(Rs)), float(np.mean(Ls))


def run(cfg, out_dir, write, prov, live) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(int(os.environ.get("MXF_THREADS", "4")))
    out = {"run_id": out_dir.name, "status": "started", "provenance": prov, "settings": cfg}
    path = out_dir / "metrics.json"
    write(path, out)
    hooks.install(lam=0.0, out_dir=str(out_dir / "hook_scratch"))

    rng = np.random.default_rng(cfg["seed"])
    files = [DATA / ln.split()[0] for ln in (DATA / "data" / "TR0-canonicals.lst").read_text()
             .split("\n") if ln.strip()]
    rng.shuffle(files)
    pool = []
    for f in files:
        s, p = bpseq_raw(f)
        if len(s) <= 200:
            pool.append((s, p, f.name))
        if len(pool) == 700:
            break
    stream, held = pool[:500], [(s, p) for s, p, _ in pool[500:700]]
    out["data"] = {"stream_files": [n for _, _, n in stream], "n_held_out": len(held)}
    write(path, out)

    model0, _, _ = build(live)
    lf0 = StructuredLossWithTurner(model0, loss_pos_paired=0.5, loss_neg_paired=0.005,
                                   l1_weight=0.0, l2_weight=0.0, sl_weight=1.0)
    R0, L0 = evaluate(model0, lf0, held)
    out["step0"] = {"R0": R0, "L0": L0}
    write(path, out)
    arms = [("f0", 0.0)] + [(f"f{f}", f * L0 / R0) for f in cfg["grid_f"]]
    out["arms"] = {}
    t0 = time.time()
    for name, lam in arms:
        model, opt, _ = build(live)
        lf = StructuredLossWithTurner(model, loss_pos_paired=0.5, loss_neg_paired=0.005,
                                      l1_weight=0.0, l2_weight=0.0, sl_weight=1.0)
        hooks.STATE["lam"] = lam
        torch.manual_seed(cfg["seed"])
        model.train()
        curve = {"0": {"R": R0, "L": L0}}
        n_bwd = 0
        for step in range(1, cfg["steps"] + 1):
            seq, partner, _ = stream[(step - 1) % len(stream)]
            opt.zero_grad()
            loss = lf([seq], [torch.tensor(partner)]).sum()
            if loss.item() > 0.0:
                loss.backward()
                opt.step()
                n_bwd += 1
            if step in cfg["eval_steps"]:
                Rm, Lm = evaluate(model, lf, held)
                curve[str(step)] = {"R": Rm, "L": Lm}
                out["arms"][name] = {"lambda": lam, "curve": curve, "n_backward": n_bwd,
                                     "seconds": round(time.time() - t0, 1)}
                write(path, out)
    last = str(cfg["steps"])
    base = out["arms"]["f0"]["curve"][last]
    sel = []
    for f in cfg["grid_f"]:
        a = out["arms"][f"f{f}"]
        c = a["curve"][last]
        rho = c["R"] / base["R"]
        delta = c["L"] / base["L"] - 1.0
        a.update({"rho": rho, "delta": delta,
                  "qualifies": rho <= cfg["rho_max"] and delta <= cfg["delta_max"]})
        if a["qualifies"]:
            sel.append((f, a["lambda"]))
    out["selection"] = {"rule": cfg["rule"], "qualifying_f": [f for f, _ in sel]}
    out["f_star"] = sel[0][0] if sel else None
    out["lambda_star"] = sel[0][1] if sel else None
    out["status"] = "complete"
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    write(path, out)
    print({"f_star": out["f_star"], "lambda_star": out["lambda_star"]})
    return 0
