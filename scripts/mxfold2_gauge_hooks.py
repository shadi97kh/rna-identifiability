"""Training-time hooks for the MXfold2 gauge study: regulariser, v-tracking, RNG sidecars.

Installed by scripts/mxfold2_gauge_train.py AFTER scripts.mxfold2_timed_train (whose
timing wrappers are installed at import), so these wrap the timed versions. MXfold2's
source is not edited. Everything here draws no random numbers, so data order and
dropout masks are those of an unhooked run with the same seed.

What it adds, per training step (batch size 1):
  - R(x) = ||P_G theta(x)||^2 / dim G(x) on the step's positional outputs (train mode,
    dropout included), via scripts/mxfold2_gauge (dense, deterministic on the GPU);
    if lambda > 0 the loss becomes loss + lambda * R, otherwise R is only logged;
  - the hinge term (pred - ref)/L and the sl term w (ref - ref2)^2 / L, read from the
    DP calls, and whether pred or ref contains a helix longer than 30 pairs;
  - v.g after backward (the gradient along the weight-space gauge v).
Per epoch: v.w before the first epoch and after every epoch, the per-epoch means and
per-step arrays, written atomically as gauge-epoch-{k}.json and steps-epoch-{k}.npz in
the run's output directory. Every checkpoint gets an RNG sidecar (torch CPU and CUDA
generators, python random), written atomically; a resume restores it. Restoring keeps
the DATA ORDER exact (it uses the CPU generator), but on the GPU every dropout mask in
every layer diverges after the first post-resume step: a fresh process re-seeds
cuDNN's RNN dropout state from the CUDA generator, which moves its offset. A resumed
GPU run is therefore flagged inexact by the launcher; on the CPU the resume is exact.
Evaluation-mode calls of the loss (Train.test, the pilot's held-out evaluation) bypass
the hook: no lambda term, no logging.
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import time

import mxfold2.interface as interface
import mxfold2.train as train_mod
import numpy as np
import torch
from mxfold2.fold.mix import MixedFold
from mxfold2.loss import StructuredLossWithTurner

from scripts.mxfold2_gauge import MAX_HELIX, Geometry, Masks, null_ratio, v_dot, v_dot_grad

STATE = {"lam": 0.0, "out": None, "crosscheck": 0, "model": None, "steps": None,
         "cur": None, "epoch_start_vw": None, "resumed_from": None, "rng_restored": False,
         "crosscheck_rel": [], "n_seen": 0}


def _new_steps():
    return {k: [] for k in ("hinge", "sl", "R", "lamR", "vg", "long_helix", "backward",
                            "seq_len")}


def _longest_helix(partner):
    best = run = 0
    prev = None
    for i in range(1, len(partner)):
        j = partner[i]
        if j > i:
            run = run + 1 if prev is not None and prev == (i - 1, j + 1) else 1
            best = max(best, run)
            prev = (i, j)
        else:
            pass
    return best


def _atomic_json(path: pathlib.Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def install(lam: float, out_dir: str, crosscheck: int = 0) -> None:
    STATE.update(lam=float(lam), out=pathlib.Path(out_dir), crosscheck=int(crosscheck),
                 steps=_new_steps())
    STATE["out"].mkdir(parents=True, exist_ok=True)

    _mp = MixedFold.make_param

    def make_param(self, seq):
        r = _mp(self, seq)
        self.__dict__["_gauge_param"] = r
        return r

    MixedFold.make_param = make_param

    _mx, _tu = interface.predict_mxfold, interface.predict_turner

    def predict_mxfold(*a, **k):
        r = _mx(*a, **k)
        cur = STATE["cur"]
        if cur is not None:
            key = "ref" if k.get("constraint") is not None else "pred"
            cur[key] = (float(r[0]), list(r[2]))
        return r

    def predict_turner(*a, **k):
        r = _tu(*a, **k)
        if STATE["cur"] is not None:
            STATE["cur"]["ref2"] = float(r[0])
        return r

    interface.predict_mxfold, interface.predict_turner = predict_mxfold, predict_turner

    _fwd = StructuredLossWithTurner.forward

    def forward(self, seq, pairs, fname=None):
        if not self.model.training:  # Train.test / held-out evaluation: plain hinge_mix
            loss = _fwd(self, seq, pairs, fname=fname)
            self.model.__dict__.pop("_gauge_param", None)
            return loss
        STATE["model"] = self.model
        STATE["cur"] = {}
        STATE["n_seen"] += 1
        loss = _fwd(self, seq, pairs, fname=fname)
        p = self.model.__dict__.pop("_gauge_param", None)
        assert p is not None and len(seq) == 1, "expects one make_param per step, batch 1"
        pos = p[0]["positional"]
        mk = Masks(seq[0], device=pos["score_helix_stacking"].device)
        if STATE["lam"] > 0.0:
            R = null_ratio(pos, mk)
            loss = loss + (STATE["lam"] * R).to(loss.dtype)
        else:
            with torch.no_grad():
                R = null_ratio(pos, mk)
        cur, L = STATE["cur"], len(seq[0])
        st = STATE["steps"]
        st["R"].append(R.detach())
        st["lamR"].append(STATE["lam"] * R.detach())
        st["seq_len"].append(L)
        if "pred" in cur and "ref" in cur:
            st["hinge"].append((cur["pred"][0] - cur["ref"][0]) / L)
            st["sl"].append(self.sl_weight * (cur["ref"][0] - cur.get("ref2", float("nan")))
                            ** 2 / L)
            st["long_helix"].append(max(_longest_helix(cur["pred"][1]),
                                        _longest_helix(cur["ref"][1])) > MAX_HELIX)
        else:
            st["hinge"].append(float("nan"))
            st["sl"].append(float("nan"))
            st["long_helix"].append(False)
        st["backward"].append(False)
        st["vg"].append(float("nan"))
        if STATE["n_seen"] <= STATE["crosscheck"]:  # the run's first steps only
            with torch.no_grad():
                g = Geometry(seq[0].upper())  # Geometry is case-sensitive; MXfold2 is not
                th = g.theta({k: v.detach().cpu() for k, v in pos.items()})
                ref = g.proj_norm2(th) / (len(g.I) + int(g.h_in_G))
                STATE["crosscheck_rel"].append(abs(float(R) - ref) / max(abs(ref), 1e-300))
        STATE["cur"] = None
        return loss

    StructuredLossWithTurner.forward = forward

    _bw = torch.Tensor.backward

    def backward(self, *a, **k):
        r = _bw(self, *a, **k)
        m, st = STATE["model"], STATE["steps"]
        if m is not None and st["backward"]:
            st["backward"][-1] = True
            st["vg"][-1] = v_dot_grad(m)
        return r

    torch.Tensor.backward = backward

    _train = train_mod.Train.train

    def train_epoch(self, epoch):
        if STATE["epoch_start_vw"] is None:
            STATE["epoch_start_vw"] = {"epoch": epoch - 1,
                                       "v_dot_w": v_dot(self.model.state_dict())}
        STATE["steps"] = _new_steps()
        t = time.time()
        r = _train(self, epoch)
        st = STATE["steps"]
        R = torch.stack(st["R"]).double().cpu().numpy() if st["R"] else np.zeros(0)
        lamR = torch.stack(st["lamR"]).double().cpu().numpy() if st["lamR"] else np.zeros(0)
        vg = np.array(st["vg"], dtype=float)
        bwd = np.array(st["backward"], dtype=bool)
        lh = np.array(st["long_helix"], dtype=bool)
        rec = {"epoch": epoch, "seconds": time.time() - t,
               "v_dot_w_start": STATE["epoch_start_vw"],
               "v_dot_w": v_dot(self.model.state_dict()),
               "lambda": STATE["lam"], "n_steps": len(st["R"]),
               "n_backward": int(bwd.sum()), "n_long_helix_steps": int(lh.sum()),
               "mean_hinge": float(np.nanmean(st["hinge"])) if st["hinge"] else None,
               "mean_sl": float(np.nanmean(st["sl"])) if st["sl"] else None,
               "mean_R": float(R.mean()) if R.size else None,
               "mean_lamR": float(lamR.mean()) if lamR.size else None,
               "sum_abs_vg": float(np.nansum(np.abs(vg))),
               "max_abs_vg": float(np.nanmax(np.abs(vg))) if np.isfinite(vg).any() else None,
               "max_abs_vg_non_long_helix": (float(np.nanmax(np.abs(vg[~lh])))
                                             if np.isfinite(vg[~lh]).any() else None),
               "resumed_from_epoch": STATE["resumed_from"],
               "rng_restored": STATE["rng_restored"],
               "crosscheck_rel_max": (max(STATE["crosscheck_rel"])
                                      if STATE["crosscheck_rel"] else None)}
        out = STATE["out"]
        tmp = out / f"steps-epoch-{epoch}.tmp.npz"
        np.savez(tmp, R=R, lamR=lamR, vg=vg, backward=bwd, long_helix=lh,
                 hinge=np.array(st["hinge"], dtype=float), sl=np.array(st["sl"], dtype=float),
                 seq_len=np.array(st["seq_len"], dtype=np.int32))
        os.replace(tmp, out / f"steps-epoch-{epoch}.npz")
        _atomic_json(out / f"gauge-epoch-{epoch}.json", rec)
        STATE["epoch_start_vw"] = {"epoch": epoch, "v_dot_w": rec["v_dot_w"]}
        return r

    train_mod.Train.train = train_epoch

    _save = train_mod.Train.save_checkpoint

    def save_checkpoint(self, outdir, epoch):
        r = _save(self, outdir, epoch)
        state = {"cpu": torch.get_rng_state(), "python": random.getstate()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        final = os.path.join(outdir, f"rng-epoch-{epoch}.pt")
        torch.save(state, final + ".tmp")
        os.replace(final + ".tmp", final)
        return r

    train_mod.Train.save_checkpoint = save_checkpoint

    _resume = train_mod.Train.resume_checkpoint

    def resume_checkpoint(self, filename):
        epoch = _resume(self, filename)
        side = pathlib.Path(filename).with_name(f"rng-epoch-{epoch}.pt")
        STATE["resumed_from"] = epoch
        if side.exists():
            try:
                state = torch.load(side, weights_only=False)
                torch.set_rng_state(state["cpu"])
                random.setstate(state["python"])
                if "cuda" in state and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(state["cuda"])
                STATE["rng_restored"] = True
            except Exception as e:  # noqa: BLE001 -- an unreadable sidecar must not stop a resume
                STATE["rng_restored"] = False
                STATE["rng_restore_error"] = repr(e)
        STATE["epoch_start_vw"] = {"epoch": epoch, "v_dot_w": v_dot(self.model.state_dict())}
        return epoch

    train_mod.Train.resume_checkpoint = resume_checkpoint
