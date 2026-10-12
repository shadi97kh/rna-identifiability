#!/usr/bin/env python3
"""Weight-decay ablation: does weight decay fix the MXfold2 gauge, and at what rate?

Plan approved by the user 2026-10-11 (/home/shadi/.claude/plans/wild-kindling-moth.md).
Weight decay was 0.0 in every run of both studies (runs/2026-10-11_mxfold2_weight_decay_check),
so AdamW's v.w drift is not decay-driven; this is the counterfactual. Seed 20261006, TR0
uncapped, hinge_mix, lambda = 0 (hooks log only), through scripts/mxfold2_wd_ablation_train.py,
which records v.w just before and after every optimiser step.

ARMS
  (a) adamw_wd0.01   AdamW (MXfold2's own build_optimizer), --l2-weight 0.01 = torch's
                     decoupled default, score-loss weight 1.0, 1 epoch
  (b) sgd_m0_wd0.01  torch SGD, momentum 0 (NOT the registered SGD arm's Nesterov 0.9, which
                     MXfold2 hard-codes; overridden in the runner), coupled weight decay 0.01,
                     score-loss weight 1.0, 1 epoch
  (c) paper_like     AdamW, --l2-weight 0.01, --score-loss-weight 0.125, 10 epochs: the
                     paper's stated training as closely as the released code permits; its
                     decay is decoupled AdamW weight decay, not the paper's l2 loss term
                     (commented out at loss.py:42-46 and 97-101). Run after (a) and (b) report.
The registered predictions and reported quantities are PRED below, written to the artifact by
--phase preregister before any arm trains.

Phases (.venv-mxfold2):
  preregister   write PRED to runs/2026-10-11_mxfold2_weight_decay_ablation/metrics.json, once
  smoke         arms (a) and (b) on 50 seeded TR0 sequences, one epoch, into the scratch
                directory given as --out; crash-only checks of the logs and the step identity
  run-ab        arms (a) and (b) together on the GPU `make gpu` reports free (launch with setsid)
  analyse-ab    CPU: A1, A2, NC1, NC2, the AR(1) confinement figures
  run-c         arm (c) alone (launch with setsid)
  analyse-c     CPU: TS0 F and the AR(1) validation for arm (c)
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402

RUN_ID = "2026-10-11_mxfold2_weight_decay_ablation"
OUT = R / "runs" / RUN_ID
DATA = R / ".cache" / "mxfold2-data"
PY_MX = R / ".venv-mxfold2" / "bin" / "python"
RUNNER = R / "scripts" / "mxfold2_wd_ablation_train.py"
ARCH_SRC = R / "runs" / "2026-10-08_mxfold2_gauge_runs" / "arch.conf"
EVAL = R / "runs" / "2026-10-08_mxfold2_gauge_eval" / "metrics.json"
SHIPPED = R / "runs" / "2026-10-11_mxfold2_shipped_ts0_f" / "metrics.json"
SGD_RUN = R / "runs" / "2026-10-08_mxfold2_gauge_runs" / "seeds" / "sgd_seed20261006_w1.0"
SEED, LR, WD = 20261006, 1e-3, 0.01
ETA_LAMBDA = LR * WD
BATCHES = (100, 1000)
FLOOR_RATIO = 1.2
BAR = 1e-3
ARMS = {
    "adamw_wd0.01": {"optimizer": "AdamW", "wd_opt": None, "sl": "1.0", "epochs": 1},
    "sgd_m0_wd0.01": {"optimizer": "SGD", "wd_opt": "sgd_m0", "sl": "1.0", "epochs": 1},
    "paper_like": {"optimizer": "AdamW", "wd_opt": None, "sl": "0.125", "epochs": 10},
}

PRED = {
    "A1": {"rule_user": "In arm (b), v.w after one epoch equals (1 - lr*wd)^N times its epoch-0 "
                        "value, N the number of update steps, to within the reconstruction "
                        "residual.",
           "operational": "every update step satisfies vw_post = (1 - lr*wd)*vw_pre - lr*v.g "
                          "(logged v.g); over the epoch the recurrence r(N) = (1 - lr*wd)^N "
                          "v.w(0) - lr sum_t (1 - lr*wd)^(N-1-t) v.g_t matches the observed "
                          "v.w(1) within 1e-3 (S1 criterion (i)'s bar). Reported alongside: the "
                          "pure closed form (1 - lr*wd)^N v.w(0), its miss, and the breaker term "
                          "that accounts for it (the 11 long-helix TR0 sequences move v.w even "
                          "without momentum, about lr * sum(v.g) ~ 1e-4 per epoch)"},
    "A2": {"rule_user": "In arm (a), decay and the preconditioner are separable and both "
                        "present.",
           "operational": "per step, delta = delta_decay + delta_precond with delta_decay = "
                          "-lr*wd*vw_pre exactly (decoupled decay); reported: the two sums and "
                          "their ratio, and arm (a)'s v.w(1) against the wd = 0 AdamW seed-06 "
                          "run's epoch-1 value (stage 2a S2 trace); 'both present and "
                          "separable' = both sums exceed 1e-3 in magnitude",
           "why_not_the_v_g_reconstruction": "reconstruction from logged v.g is SGD's update "
                                             "rule; AdamW's step along v is the "
                                             "preconditioned step, not v.g"},
    "prior_claude": "A1 confirmed (recurrence residual ~1e-5; the pure closed form misses by "
                    "~1e-4 from the breaker steps). A2: the preconditioner dominates, roughly 2.5 "
                    "against at most 0.1 in magnitude; at this rate decay does not fix the gauge "
                    "within an epoch, it contracts v.w toward 0 by ~10% per epoch, time constant "
                    "1/(lr*wd) = 1e5 steps (about 9.2 epochs).",
    "NC1_must_fail": "arm (b)'s recurrence with the decay term omitted misses the observed "
                     "v.w(1) by ~0.1*|v.w(0)| ~ 0.11, far above 1e-3",
    "NC2_must_hold": "the analysis recurrence (torch SGD's rule in v-coordinates), run with the "
                     "registered S1 parameters (wd 0, Nesterov 0.9) on the existing SGD "
                     "seed-06 epoch-1 logs, reproduces the evaluator's registered S1 "
                     "reconstruction at epoch 1 to 1e-12",
    "void_rule": "a NaN loss or a non-finite v.w in an arm voids that arm; no rerun without "
                 "asking",
    "AR1_confinement": {
        "status": "REPORTED quantity, formula fixed in advance; not a prediction (user)",
        "formula": "from arm (a)'s per-step delta_precond: mean mu, SD sigma; under v.w <- "
                   "(1 - eta*lambda) v.w + delta_precond the stationary mean is "
                   "mu/(eta*lambda) and the stationary SD sigma/sqrt(2*eta*lambda) (factors "
                   "1e5 and 223.6 at eta 1e-3, lambda 0.01)",
        "long_run_claude": "increments along v are autocorrelated (beta1 = 0.9, slower drift): "
                           "also reported, sigma_LR from non-overlapping batch means of 100 "
                           "and 1,000 steps, stationary SD sigma_LR/sqrt(2*eta*lambda)",
        "floor_rule_user": "if sigma_LR(1000) exceeds sigma_LR(100) by more than 20%, the "
                           "series has not decorrelated at b = 1000, both are lower bounds and "
                           "the stationary width is a FLOOR; otherwise an ESTIMATE. A floor "
                           "supports 'decay does not confine below the observed spread', "
                           "never the converse",
        "compared_against": "AdamW's unregularised epoch-10 spread across the 8 seeds, read "
                            "from runs/2026-10-08_mxfold2_gauge_eval S2 traces (w = 1.0: SD "
                            "4.97, -6.96 to +8.00; w = 0.125: SD 4.59); the 3.2 quoted with the "
                            "request is the epoch-1 SD",
        "caveat": "epoch-1 increments are early training: an extrapolation"},
    "arm_c": {
        "reported_no_prediction": "TS0 F of the epoch-10 checkpoint with the arms' scorer, "
                                  "against the published 0.575, the shipped 0.5701, the "
                                  "code-default arm 0.5225, the w = 0.125 arm mean 0.5502 and "
                                  "its seed-20261006 model (like-for-like: same seed and C1, "
                                  "wd 0 vs 0.01, n = 1 each); and its v.w trajectory against "
                                  "the wd = 0 seed-06 traces",
        "decay_note": "decoupled AdamW weight decay, not the paper's l2 loss term (commented "
                      "out at loss.py:42-46 and 97-101): the closest reachable configuration",
        "AR1_validation_user": "recompute mu, sigma and sigma_LR (batches 100, 1000; same floor "
                               "rule) from arm (c)'s ten epochs of delta_precond, beside arm "
                               "(a)'s; predicted mean path v.w(t) = (1 - eta*lambda)^t v.w(0) + "
                               "(mu/(eta*lambda)) (1 - (1 - eta*lambda)^t) with arm (c)'s own "
                               "mu; SD(t) = sigma_LR(1000) sqrt((1 - phi^(2t)) / (1 - phi^2)), "
                               "phi = 1 - eta*lambda; TRACKS iff the observed v.w at every one "
                               "of the 10 epoch ends lies within the predicted mean +- 2 SD(t); "
                               "if it tracks, the stationary width is a validated extrapolation "
                               "(still under the floor rule); if not, it is not quoted, only "
                               "the measured trajectory",
        "AR1_validation_note_claude": "the recursion is exact by construction for decoupled "
                                      "decay; with mu fitted from the same run the test checks "
                                      "that the increment mean holds steady over training"},
}


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def read_out():
    return json.loads((OUT / "metrics.json").read_text())


# ----------------------------------------------------------------------------------
# launching
# ----------------------------------------------------------------------------------
def make_gpu_device():
    mg = subprocess.run(["make", "gpu"], cwd=R, capture_output=True, text=True)
    m = re.search(r"OK to launch: CUDA_VISIBLE_DEVICES=(\d+)", mg.stdout)
    return (int(m.group(1)) if mg.returncode == 0 and m else None), mg.stdout[-400:]


def start(name, base, gpu, train_list):
    cfg = ARMS[name]
    d = base / "seeds" / name
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    arch = base / "arch.conf"
    if not arch.exists():
        arch.write_bytes(ARCH_SRC.read_bytes())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               MXF_TIMING_OUT=str(d / "timing.json"), MXF_GAUGE_OUT=str(d / "gauge"),
               MXF_GAUGE_LAMBDA="0", MXF_VW_OUT=str(d / "gauge"))
    if cfg["wd_opt"]:
        env["MXF_WD_OPT"] = cfg["wd_opt"]
    cmd = [str(PY_MX), str(RUNNER), "train", f"@{arch}", "--param", str(d / "param.pth"),
           "--save-config", str(d / "model.conf"), "--log-dir", str(d / "checkpoints"),
           "--epochs", str(cfg["epochs"]), "--gpu", "0", "--seed", str(SEED),
           "--loss-func", "hinge_mix", "--score-loss-weight", cfg["sl"],
           "--optimizer", cfg["optimizer"], "--l2-weight", str(WD),
           "--disable-progress-bar", train_list]
    log = (d / "stdout.log").open("w")
    p = subprocess.Popen(cmd, cwd=DATA, env=env, stdout=log, stderr=subprocess.STDOUT)
    return p, {"cmd": cmd, "env": {k: env[k] for k in env if k.startswith("MXF_")
                                   or k in ("CUDA_VISIBLE_DEVICES",)},
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}


def run_arms(names, key):
    if os.getsid(0) != os.getpid():
        raise SystemExit("launch with setsid so that no session can end the driver")
    out = read_out()
    gpu, mg = make_gpu_device()
    out[f"launch_gate_{key}"] = {"make_gpu_tail": mg, "device": gpu,
                                 "checked": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 **git_provenance(R)}
    if gpu is None or git_provenance(R)["git_dirty_excluding_runs"]:
        out["status"] = f"NOT LAUNCHED ({key}): make gpu not OK or tree dirty"
        write(out)
        return 1
    out["status"] = f"RUNNING {key}"
    write(out)
    procs = {n: start(n, OUT, gpu, "data/TR0-canonicals.lst") for n in names}
    rec = {}
    for n, (p, meta) in procs.items():
        code = p.wait()
        rec[n] = meta | {"exit_code": code, "ended": time.strftime("%Y-%m-%d %H:%M:%S")}
        out.setdefault("runs", {})[n] = rec[n]
        write(out)
    ok = all(v["exit_code"] == 0 for v in rec.values())
    out["status"] = f"{key} trained" if ok else f"{key} FAILED (exit codes)"
    write(out)
    return 0 if ok else 1


# ----------------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------------
def sgd_recurrence(v0, vg, lr, wd, momentum=0.0, nesterov=False):
    """torch.optim.SGD's update in v-coordinates (dampening 0), from the logged v.g of the
    update steps only. Returns the trace after each step."""
    r, buf, trace = v0, None, []
    for g in vg:
        d = g + wd * r
        if momentum:
            buf = d if buf is None else momentum * buf + d
            d = d + momentum * buf if nesterov else buf
        r = r - lr * d
        trace.append(r)
    return np.array(trace)


def load_arm(d, epoch):
    gj = json.loads((d / "gauge" / f"gauge-epoch-{epoch}.json").read_text())
    z = np.load(d / "gauge" / f"steps-epoch-{epoch}.npz")
    vw = np.load(d / "gauge" / f"vw-epoch-{epoch}.npz")
    bw = z["backward"]
    return gj, z, vw, z["vg"][bw]


def void_check(gj, z, vw):
    finite = (np.isfinite(gj["v_dot_w"]) and np.isfinite(vw["pre"]).all()
              and np.isfinite(vw["post"]).all() and np.isfinite(z["hinge"]).all())
    return {"void": not finite, "finite_loss_and_vw": bool(finite)}


def long_run_sd(x, b):
    m = len(x) // b
    means = x[: m * b].reshape(m, b).mean(axis=1)
    return {"batch": b, "n_batches": m, "sigma_LR": float(math.sqrt(b * means.var(ddof=1)))}


def ar1(dp):
    mu, sig = float(dp.mean()), float(dp.std(ddof=1))
    lr_ = {str(b): long_run_sd(dp, b) for b in BATCHES}
    s100, s1000 = lr_["100"]["sigma_LR"], lr_["1000"]["sigma_LR"]
    ratio = s1000 / s100
    floor = ratio > FLOOR_RATIO
    return {"mu": mu, "sigma": sig, "n_steps": int(dp.size),
            "stationary_mean": mu / ETA_LAMBDA,
            "stationary_sd_naive": sig / math.sqrt(2 * ETA_LAMBDA),
            "long_run": lr_, "sigma_LR_ratio_1000_over_100": ratio,
            "stationary_sd_long_run_b1000": s1000 / math.sqrt(2 * ETA_LAMBDA),
            "stationary_sd_long_run_b100": s100 / math.sqrt(2 * ETA_LAMBDA),
            "width_is": "FLOOR" if floor else "ESTIMATE"}


def epoch10_spread():
    s2 = json.loads(EVAL.read_text())["2a"]["results"]["S2"]
    out = {}
    for arm in ("adamw_w1.0", "adamw_w0.125"):
        v10 = [t["10"] for t in s2[arm]["traces"].values()]
        out[arm] = {"seed_sd_epoch10": s2[arm]["seed_sd_by_epoch"]["10"],
                    "min_epoch10": min(v10), "max_epoch10": max(v10),
                    "seed06_trace": s2[arm]["traces"][str(SEED)]}
    return out


def nc2():
    """The registered S1 recurrence (wd 0, Nesterov 0.9) on the existing SGD seed-06 epoch-1
    logs must reproduce the evaluator's recorded reconstruction at epoch 1."""
    s1 = json.loads(EVAL.read_text())["2b"]["results"]["S1"]["runs"]["sgd_seed20261006_w1.0"]
    g0 = json.loads((SGD_RUN / "gauge" / "gauge-epoch-1.json").read_text())
    z = np.load(SGD_RUN / "gauge" / "steps-epoch-1.npz")
    tr = sgd_recurrence(g0["v_dot_w_start"]["v_dot_w"], z["vg"][z["backward"]], LR, 0.0,
                        momentum=0.9, nesterov=True)
    reg = float(s1["reconstruction"]["1"])
    return {"ours": float(tr[-1]), "registered": reg, "abs_diff": abs(float(tr[-1]) - reg),
            "passes": abs(float(tr[-1]) - reg) <= 1e-12}


def analyse_ab():
    out = read_out()
    res = {}
    # arm (b): A1 and NC1
    gj, z, vw, vg = load_arm(OUT / "seeds" / "sgd_m0_wd0.01", 1)
    v0, v1, N = gj["v_dot_w_start"]["v_dot_w"], gj["v_dot_w"], int(len(vg))
    vb = void_check(gj, z, vw)
    step_res = vw["post"] - ((1 - LR * WD) * vw["pre"] - LR * vg)
    full = sgd_recurrence(v0, vg, LR, WD)[-1]
    pure = (1 - LR * WD) ** N * v0
    no_decay = sgd_recurrence(v0, vg, LR, 0.0)[-1]
    res["arm_b"] = vb | {
        "v_w_epoch0": v0, "v_w_epoch1": v1, "N_update_steps": N,
        "n_logged_pre": int(len(vw["pre"])), "n_logged_post": int(len(vw["post"])),
        "per_step_identity_max_abs_residual": float(np.abs(step_res).max()),
        "A1_recurrence": float(full), "A1_residual": float(v1 - full),
        "A1_passes": bool(abs(v1 - full) <= BAR),
        "pure_closed_form": float(pure), "pure_factor": (1 - LR * WD) ** N,
        "pure_closed_form_miss": float(v1 - pure),
        "breaker_term": float(full - pure),
        "NC1_no_decay_recurrence": float(no_decay), "NC1_miss": float(v1 - no_decay),
        "NC1_fails_as_required": bool(abs(v1 - no_decay) > BAR),
        "note": "momentum 0 here; the registered SGD arm used Nesterov 0.9"}
    # arm (a): A2 and the AR(1) figures
    gj, z, vw, vg = load_arm(OUT / "seeds" / "adamw_wd0.01", 1)
    va = void_check(gj, z, vw)
    pre, post = vw["pre"], vw["post"]
    d_decay = -LR * WD * pre
    d_prec = (post - pre) - d_decay
    sp = epoch10_spread()
    res["arm_a"] = va | {
        "v_w_epoch0": gj["v_dot_w_start"]["v_dot_w"], "v_w_epoch1": gj["v_dot_w"],
        "N_update_steps": int(len(pre)),
        "between_steps_max_abs_change": float(np.abs(pre[1:] - post[:-1]).max()),
        "endpoints_match_epoch_record": bool(
            abs(pre[0] - gj["v_dot_w_start"]["v_dot_w"]) < 1e-12
            and abs(post[-1] - gj["v_dot_w"]) < 1e-12),
        "sum_delta_decay": float(d_decay.sum()), "sum_delta_precond": float(d_prec.sum()),
        "ratio_precond_over_decay": float(d_prec.sum() / d_decay.sum()),
        "both_present_and_separable": bool(abs(d_decay.sum()) > BAR
                                           and abs(d_prec.sum()) > BAR),
        "wd0_adamw_seed06_epoch1": sp["adamw_w1.0"]["seed06_trace"]["1"],
        "v_w_epoch1_minus_wd0": gj["v_dot_w"] - sp["adamw_w1.0"]["seed06_trace"]["1"],
        "AR1": ar1(d_prec), "unregularised_epoch10_spread": sp}
    res["NC2"] = nc2()
    res["rate"] = {"per_epoch_contraction_N10814": (1 - ETA_LAMBDA) ** 10814,
                   "time_constant_steps": 1 / ETA_LAMBDA,
                   "time_constant_epochs_at_10814": 1 / ETA_LAMBDA / 10814}
    out["analysis_ab"] = res
    out["status"] = "arms a,b analysed"
    write(out)
    print(json.dumps(res, indent=1, default=str))
    return 0


def ts0_f(checkpoint):
    """TS0 F of a checkpoint with the gauge arms' scorer (the code path of
    scripts/mxfold2_shipped_ts0_f.py: read_bpseq, predict, per-sequence pair counts,
    scripts/mxfold2_gauge_eval.accuracy)."""
    import hashlib

    import torch

    import scripts.mxfold2_gauge_eval as ev
    from scripts.mxfold2_decompose import build_model, predict, read_bpseq

    torch.set_num_threads(4)
    model = build_model(ev.LIVE / "arch.conf")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)
                          ["model_state_dict"])
    model.eval()
    entries = json.loads((ev.LIVE / "metrics.json").read_text())["evaluation_set"]["entries"]
    tp, fp, fn = [], [], []
    for e in entries:
        seq, partner = read_bpseq(ev.DATA / e["file"])
        assert hashlib.sha256(seq.encode()).hexdigest() == e["sha256"], e["file"]
        with torch.no_grad():
            param = model.make_param([seq])
        _, _, bp = predict(model, seq, param)
        ref = {(i, partner[i]) for i in range(1, len(partner)) if partner[i] > i}
        pred = {(i, bp[i]) for i in range(1, len(bp)) if bp[i] > i}
        tp.append(len(pred & ref))
        fp.append(len(pred - ref))
        fn.append(len(ref - pred))
    return ev.accuracy(np.array(tp), np.array(fp), np.array(fn))


def analyse_c():
    out = read_out()
    d = OUT / "seeds" / "paper_like"
    E = ARMS["paper_like"]["epochs"]
    recs = [load_arm(d, e) for e in range(1, E + 1)]
    void = [void_check(gj, z, vw) for gj, z, vw, _ in recs]
    pre = np.concatenate([vw["pre"] for _, _, vw, _ in recs])
    post = np.concatenate([vw["post"] for _, _, vw, _ in recs])
    ends = np.cumsum([len(vw["pre"]) for _, _, vw, _ in recs])
    v0 = recs[0][0]["v_dot_w_start"]["v_dot_w"]
    obs = [gj["v_dot_w"] for gj, _, _, _ in recs]
    d_decay = -LR * WD * pre
    d_prec = (post - pre) - d_decay
    a = ar1(d_prec)
    phi = 1 - ETA_LAMBDA
    s_lr = a["long_run"]["1000"]["sigma_LR"]
    path = []
    for k, t in enumerate(ends, start=1):
        mean = phi ** t * v0 + (a["mu"] / ETA_LAMBDA) * (1 - phi ** t)
        sd = s_lr * math.sqrt((1 - phi ** (2 * t)) / (1 - phi ** 2))
        o = obs[k - 1]
        path.append({"epoch": k, "t_steps": int(t), "observed": o, "predicted_mean": mean,
                     "predicted_sd": sd, "z": (o - mean) / sd,
                     "inside_2sd": bool(abs(o - mean) <= 2 * sd)})
    tracks = all(p["inside_2sd"] for p in path)
    sp = epoch10_spread()
    ev_res = json.loads(EVAL.read_text())["2b"]["results"]
    arms_f = ev_res["main"]["arms"]
    shipped = json.loads(SHIPPED.read_text())
    f_c = ts0_f(d / "checkpoints" / f"epoch-{E}")
    f06 = ev_res["F_per_model"]["adamw_w0.125_seed20261006"]["F_mean_per_sequence"]
    pub = shipped["paper_training"]["table2_TS0_TR0_trained"]["F"]
    res = {
        "void_by_epoch": void, "void": any(v["void"] for v in void),
        "between_steps_max_abs_change": float(np.abs(pre[1:] - post[:-1]).max()),
        "endpoints_match_epoch_records": bool(abs(pre[0] - v0) < 1e-12 and all(
            abs(post[t - 1] - o) < 1e-12 for t, o in zip(ends, obs, strict=True))),
        "v_w_epoch0": v0, "v_w_by_epoch": obs, "N_update_steps": int(ends[-1]),
        "sum_delta_decay": float(d_decay.sum()), "sum_delta_precond": float(d_prec.sum()),
        "AR1": a, "AR1_arm_a_epoch1": out["analysis_ab"]["arm_a"]["AR1"],
        "AR1_validation": {"path": path, "n_epochs_inside_2sd": sum(p["inside_2sd"] for p in path),
                           "tracks": tracks,
                           "stationary_width_quoted": (a["stationary_sd_long_run_b1000"]
                                                       if tracks else None),
                           "stationary_width_is": a["width_is"] if tracks else
                           "not quoted: the AR(1) description failed the registered test"},
        "trajectory_vs_wd0": {"adamw_w0.125_seed06": sp["adamw_w0.125"]["seed06_trace"],
                              "adamw_w1.0_seed06": sp["adamw_w1.0"]["seed06_trace"]},
        "unregularised_epoch10_spread": {k: {x: v[x] for x in ("seed_sd_epoch10",
                                                               "min_epoch10", "max_epoch10")}
                                         for k, v in sp.items()},
        "TS0": {"paper_like_epoch10": f_c, "published_TR0_model": pub,
                "shipped": shipped["shipped_same_scorer_as_arms"]["F_mean_per_sequence"],
                "code_default_arm_mean": arms_f["adamw_w1.0"]["F_mean"],
                "w0.125_arm_mean": arms_f["adamw_w0.125"]["F_mean"],
                "w0.125_seed06_like_for_like": f06,
                "paper_like_minus_w0.125_seed06": f_c["F_mean_per_sequence"] - f06,
                "published_minus_paper_like": pub - f_c["F_mean_per_sequence"],
                "note": "reported with no prediction; n = 1; decay is decoupled AdamW weight "
                        "decay, not the paper's l2 loss term"}}
    out["analysis_c"] = res
    out["status"] = "arm c analysed"
    write(out)
    print(json.dumps({k: res[k] for k in ("void", "v_w_by_epoch", "sum_delta_decay",
                                          "sum_delta_precond", "AR1_validation", "TS0")},
                     indent=1, default=str))
    return 0


def bound_wd0():
    """Per-epoch drift of the no-decay AdamW seed-20261006 runs, as the stationary mean it
    would imply under decay. Requested by the user 2026-10-12 as a pre-arm-(c) bound, but
    computed AFTER arm (c) was analysed (commit 7a9e1d2); recorded with that timing."""
    out = read_out()
    sp = epoch10_spread()
    n = 10814  # update steps per epoch, as in the instrumented runs (seed-study runs have no hooks)
    res = {"timing": "computed 2026-10-12 after arm (c) had been analysed and committed "
                     "(7a9e1d2); requested as a pre-arm-(c) bound, so it is not one",
           "steps_per_epoch_assumed": n}
    for arm in ("adamw_w1.0", "adamw_w0.125"):
        tr = sp[arm]["seed06_trace"]
        vals = [tr[str(e)] for e in range(11)]
        d = np.diff(vals)
        late = d[3:]  # epochs 4-10
        res[arm] = {"per_epoch_delta": d.tolist(), "mu_per_step": (d / n).tolist(),
                    "stationary_mean_implied": (d / n / ETA_LAMBDA).tolist(),
                    "epoch1_mu": float(d[0] / n),
                    "epoch1_stationary_mean": float(d[0] / n / ETA_LAMBDA),
                    "epochs4to10_mean_delta": float(late.mean()),
                    "epochs4to10_mu": float(late.mean() / n),
                    "epochs4to10_stationary_mean": float(late.mean() / n / ETA_LAMBDA),
                    # SE across the 7 epoch deltas, treating them as independent
                    "epochs4to10_stationary_mean_se": float(late.std(ddof=1) / np.sqrt(len(late))
                                                            / n / ETA_LAMBDA)}
    res["arm_a_epoch1_stationary_mean_with_decay"] = out["analysis_ab"]["arm_a"]["AR1"][
        "stationary_mean"]
    if "analysis_c" in out:
        res["arm_c_fitted"] = {k: out["analysis_c"]["AR1"][k] for k in
                               ("mu", "stationary_mean", "stationary_sd_long_run_b1000")}
        res["note"] = ("arm (c) is score-loss weight 0.125, so its like-for-like no-decay "
                       "trace is adamw_w0.125 seed 20261006; arm (a) is weight 1.0")
    out["bound_wd0_late_epochs"] = res
    write(out)
    print(json.dumps(res, indent=1))
    return 0


def lambda_required(width=1.0):
    """The decoupled decay rate that would hold v.w to a stationary SD of `width`, by
    inverting SD = sigma_LR / sqrt(2 eta lambda). Requested by the user 2026-10-12, from data
    on disk; post hoc. Assumes sigma_LR does not itself depend on lambda, which no run here
    tests: the trajectory under a 13-26x stronger decay is different."""
    out = read_out()
    res = {"width": width, "eta": LR, "lambda_conventional": WD,
           "formula": "lambda = sigma_LR^2 / (2 eta width^2); exact AR(1): "
                      "eta lambda = 1 - sqrt(1 - sigma_LR^2 / width^2)",
           "assumption_untested": "sigma_LR measured at lambda = 0.01 holds at the larger "
                                  "lambda; decoupled decay shrinks every parameter, "
                                  "identifiable or not, by (1 - eta lambda) per step"}
    for name, ar in (("arm_c_ten_epochs", out["analysis_c"]["AR1"]),
                     ("arm_a_epoch1", out["analysis_ab"]["arm_a"]["AR1"])):
        for b in ("1000", "100"):
            s = ar["long_run"][b]["sigma_LR"]
            lam = s**2 / (2 * LR * width**2)
            lam_exact = (1 - math.sqrt(1 - s**2 / width**2)) / LR
            res[f"{name}_b{b}"] = {"sigma_LR": s, "lambda": lam, "lambda_exact": lam_exact,
                                   "times_conventional": lam / WD}
    out["lambda_required_for_width"] = res
    write(out)
    print(json.dumps(res, indent=1))
    return 0


def preregister():
    p = OUT / "metrics.json"
    if p.exists():
        raise SystemExit(f"{p.relative_to(R)} exists; a pre-registration is written once")
    write({"run_id": RUN_ID, "status": "PRE-REGISTERED, not yet run", "predictions": PRED,
           "arms": ARMS, "seed": SEED, "lr": LR, "weight_decay": WD,
           "written_at": time.strftime("%Y-%m-%d %H:%M:%S"), **git_provenance(R)})
    print(f"wrote {p.relative_to(R)}; commit it before any run")
    return 0


def smoke(dst):
    """Arms (a) and (b) on 50 seeded TR0 sequences, one epoch, into a scratch directory."""
    base = pathlib.Path(dst)
    base.mkdir(parents=True, exist_ok=True)
    lines = [ln for ln in (DATA / "data" / "TR0-canonicals.lst").read_text().splitlines()
             if ln.strip()]
    pick = np.random.default_rng(SEED).choice(len(lines), 50, replace=False)
    lst = base / "smoke.lst"
    lst.write_text("\n".join(lines[i] for i in sorted(pick)) + "\n")
    gpu, mg = make_gpu_device()
    if gpu is None:
        raise SystemExit(f"make gpu not OK:\n{mg}")
    procs = {n: start(n, base, gpu, str(lst)) for n in ("adamw_wd0.01", "sgd_m0_wd0.01")}
    codes = {n: p.wait() for n, (p, _) in procs.items()}
    rep = {"exit_codes": codes}
    for n in procs:
        d = base / "seeds" / n
        gj, z, vw, vg = load_arm(d, 1)
        rep[n] = {"n_backward": gj["n_backward"], "n_pre": len(vw["pre"]),
                  "n_post": len(vw["post"]), "finite": void_check(gj, z, vw)["finite_loss_and_vw"]}
        if n == "sgd_m0_wd0.01":
            rep[n]["step_identity_max_abs"] = float(np.abs(
                vw["post"] - ((1 - LR * WD) * vw["pre"] - LR * vg)).max())
    print(json.dumps(rep, indent=1))
    return 0 if all(c == 0 for c in codes.values()) else 1


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    if ph == "preregister":
        sys.exit(preregister())
    if ph == "smoke":
        sys.exit(smoke(sys.argv[sys.argv.index("--out") + 1]))
    if ph == "run-ab":
        sys.exit(run_arms(("adamw_wd0.01", "sgd_m0_wd0.01"), "arms a,b"))
    if ph == "analyse-ab":
        sys.exit(analyse_ab())
    if ph == "run-c":
        sys.exit(run_arms(("paper_like",), "arm c"))
    if ph == "analyse-c":
        sys.exit(analyse_c())
    if ph == "bound-wd0":
        sys.exit(bound_wd0())
    if ph == "lambda-required":
        sys.exit(lambda_required())
    raise SystemExit(f"unknown phase {ph}")
