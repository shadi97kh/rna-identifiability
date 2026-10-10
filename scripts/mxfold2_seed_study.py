#!/usr/bin/env python3
"""MXfold2 seed study: does the thermodynamic regulariser weight change the gauge variance?

Approved by the user 2026-10-07 with conditions. This file implements PHASE epoch1 only:
the first epoch of the 8 seeds at the code-default regulariser weight, timed, and then
STOP for review. Later phases (remaining epochs, the second weight, the evaluation
decomposition) are specified here and pre-registered, but not run by this phase.

DESIGN (fixed before any training; written to metrics.json before any launch)

Model     MixC, the architecture of the released model (vendor/mxfold2 at 51b2136,
          models/TrainSetAB.conf without its --param line), built in .venv-mxfold2 with
          torch 2.13.0+cu130 and numpy 2.4.6. MXfold2 pins torch ^1.4 and numpy ^1.18;
          both pins are overridden, and the one-batch smoke test under these versions
          passed before this study was written.
Training  bpRNA-1m TR0 (canonical-pair version, 10,814 sequences) UNCAPPED, from the
          registered MXfold2 datasets (data/SOURCES.yaml: mxfold2_datasets). Chosen over
          TrainSetA+B capped at 400 nt because it is a complete published dataset (no
          subset to defend), it comes with a published test split for a disjoint fixed
          evaluation set, and the cap would save only ~13% of its folding time.
Arms      --loss-func hinge_mix (the loss that contains the thermodynamic regulariser
          sl_weight * (ref - ref2)^2 / L) at --score-loss-weight 1.0 (the code default
          value) and 0.125 (the paper's C1). The weight is the only difference between
          arms. (The code's default LOSS is plain hinge, with no regulariser at all.)
Seeds     the 8 SEEDS of scripts/pooled_alignment.py, read from its source.
Epochs    10 planned, the code default; the paper does not state an epoch count.
Other     every other hyperparameter at the code default: AdamW, lr 1e-3, batch 1,
          loss-pos-paired 0.5 and loss-neg-paired 0.005 (the paper's delta values),
          l1-weight 0, l2 inactive in the code.
Hardware  4 seeds per GPU on both TITAN RTXs, one CPU thread per process.

EVALUATION SET: all 1,305 sequences of TS0 (canonical-pair version), listed in
metrics.json by file name, length and sha256 of the sequence.

DECOMPOSITION (pre-registered; run after training, not in this phase)

For evaluation sequence x (length N), trained model k (seed) at weight w, theta_k(x) is
the vector of MXfold2's positional outputs that some structure of x can use, plus the
global helix_length table:
  S_in(i,j)  = score_helix_stacking[i][j], i<j, (i,j) and (i+1,j-1) both allowed pairs
  S_out(j,i) = score_helix_stacking[j][i], (i,j) and (i-1,j+1) both allowed pairs
  M_in(i,j)  = score_mismatch[i][j], M_out(j,i) = score_mismatch[j][i], (i,j) allowed
  U(a,b)     = the unpaired-interval score, 1 <= a <= b <= N
  helix_length[1..30]
where an allowed pair is canonical (AU, UA, GC, CG, GU, UG) with at least 3 unpaired
nucleotides between, as MXfold2's own constraint builder sets it.
The exact gauge subspace G(x) (scripts/mxfold2_nullspace.py, step 2) is spanned by
  g_p (one per allowed pair p=(i,j)): +1 on S_in(i,j) and M_in(i,j), -1 on S_out(j,i)
       and M_out(j,i), wherever those entries exist;
  h: +1 on every S entry, -2(m-1) on helix_length[m].
These are mutually orthogonal (disjoint supports; h.g_p = 1 - 1 = 0), so the projection
P_G is closed-form. With Delta_k = theta_k(x) - mean over the 8 seeds:
  V_G(x,w)   = (1/7) sum_k ||P_G Delta_k||^2      seed variance in the exact gauge
  V_tot(x,w) = (1/7) sum_k ||Delta_k||^2
  f_G(x,w)   = V_G / V_tot
V_G is a LOWER BOUND on null-space seed variance: step 2 found null directions beyond
G that it could not characterise. The trainable Turner half of MixC is global; its
seed variance is reported separately and not decomposed (that needs MXfold2's own
Turner counting rule, which has not been derived).

Validity checks, required before any statistic is reported:
  V1  the counting rule above, applied to the extracted tables, reproduces MXfold2's
      own positional score (MixC constrained score minus the Turner-only constrained
      score) of each evaluation sequence's reference structure to within
      1e-3 * max(1, |score|);
  V2  adding a random g in G(x) to a trained model's tables changes MXfold2's score of
      the reference and of the predicted structure by less than that tolerance;
  V3  a random direction orthogonal to G(x) changes at least one of them by more.

PRE-REGISTERED PREDICTIONS
  W1 (user). Null-space seed variance is the same at both weights, because step 2
     found the regulariser gauge-invariant. Test: r(x) = V_G(x,0.125) / V_G(x,1.0);
     "the same" means the 95% bootstrap interval (10,000 resamples over evaluation
     sequences, seed 20261006) of the median of log r lies inside [log 0.8, log 1.25].
  W2 (added by Claude, secondary). The per-sequence constant shift c_x (+(b-a+1) on
     U(a,b), +1 on every S and M entry; it moves every structure of x by exactly N) is
     invisible to the hinge loss and pinned only by the regulariser, so its seed
     variance is LARGER at 0.125 than at 1.0: median over x of the ratio > 1.

PHASE epoch1 (this file): weight 1.0 only, one epoch, then stop. Records the measured
per-step overhead and the measured folding factor (MXfold2's unconstrained fold time
over ViennaRNA MFE time on the same TR0 sequences, the latter measured here on an
idle pinned core before launch), replacing both guesses in the earlier estimate.

INTERRUPTION AND FRESH RERUN (2026-10-10). The phase-full driver was killed at about
05:59 on 2026-10-09 with six runs short of epoch 10. PHASE rerun-interrupted records that
as an event in metrics.json (when it stopped, the six runs, the cause as far as known),
moves the six partial directories unchanged to interrupted_2026-10-09/, and reruns the six
FRESH under the registered no-resume policy, in their registered directories and GPUs. It
refuses unless it is its own session leader, so it must be launched with setsid, and
unless the artifact still records the interruption (scripts/predecessor.py: status
RUNNING with no live process is a fault, never completion). Successors wait for the
recorded status FULL_DONE, never for the processes to disappear.

Run: .venv/bin/python scripts/mxfold2_seed_study.py
Rerun the interrupted six (detached from every session):
  setsid nohup .venv/bin/python scripts/mxfold2_seed_study.py --phase rerun-interrupted \\
      > runs/2026-10-07_mxfold2_seed_study/driver_rerun_interrupted.log 2>&1 < /dev/null &
  disown
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.runmeta import git_provenance  # noqa: E402

RUN_ID = "2026-10-07_mxfold2_seed_study"
OUT = R / "runs" / RUN_ID
DATA = R / ".cache" / "mxfold2-data"
CONF = R / "vendor" / "mxfold2" / "mxfold2" / "models" / "TrainSetAB.conf"
PY_MX = R / ".venv-mxfold2" / "bin" / "python"
RUNNER = R / "scripts" / "mxfold2_timed_train.py"
WEIGHTS = (1.0, 0.125)
PHASE_WEIGHT = 1.0
MAX_LOAD = 3.0          # 1-minute load average allowed at launch (16 cores)
MIN_FREE_MIB = 20000    # per GPU
VIENNA_CORE = 15


def recorded_seeds():
    src = (R / "scripts" / "pooled_alignment.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SEEDS"
                                                for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise LookupError("SEEDS")


def read_bpseq(path):
    rows = [r.split() for r in path.read_text().split("\n") if r.strip() and not r.startswith("#")]
    return "".join(r[1] for r in rows).upper().replace("T", "U")


def listed(lst):
    return [ln.split()[0] for ln in (DATA / "data" / lst).read_text().split("\n") if ln.strip()]


def eval_set():
    out = []
    for f in listed("TS0-canonicals.lst"):
        s = read_bpseq(DATA / f)
        out.append({"file": f, "length": len(s),
                    "sha256": hashlib.sha256(s.encode()).hexdigest()})
    return out


def gpus():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,"
                        "utilization.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout
    out = []
    for line in q.strip().split("\n"):
        i, used, tot, util = (int(x) for x in line.split(","))
        out.append({"index": i, "used_mib": used, "free_mib": tot - used, "util": util})
    return out


def vienna_reference(seqs):
    """ViennaRNA MFE wall time over the training sequences, one pinned core."""
    import RNA

    from thermo.vienna_ref import reference_md
    os.sched_setaffinity(0, {VIENNA_CORE})
    md = reference_md()
    t0 = time.perf_counter()
    for s in seqs:
        RNA.fold_compound(s, md).mfe()
    return time.perf_counter() - t0


def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"  # atomic: a driver killed mid-write leaves the last copy
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def main():
    seeds = recorded_seeds()
    assert len(seeds) == 8
    doc = __doc__.split("DESIGN", 1)[1].split("Run:", 1)[0]
    ev = eval_set()
    out = {"run_id": RUN_ID, "status": "PRE-REGISTERED; no training launched yet",
           "protocol": "DESIGN" + doc, "seeds": list(seeds), "weights": list(WEIGHTS),
           "evaluation_set": {"source": "TS0-canonicals.lst (mxfold2_datasets)",
                              "n": len(ev), "entries": ev},
           "training_set": "TR0-canonicals.lst (mxfold2_datasets), uncapped",
           "mxfold2_commit": "51b213676708bebd664f0c40873a46e09353e1ee",
           **git_provenance(R)}
    write(out)

    # launch gates: idle CPU, both GPUs free
    load = os.getloadavg()[0]
    g = gpus()
    out["launch_gate"] = {"load_1min": load, "max_load": MAX_LOAD, "gpus": g,
                          "min_free_mib": MIN_FREE_MIB}
    if load > MAX_LOAD or len(g) < 2 or any(x["free_mib"] < MIN_FREE_MIB for x in g):
        out["status"] = "NOT LAUNCHED: launch gate failed (busy CPU or GPU); nothing trained"
        write(out)
        print(out["status"], json.dumps(out["launch_gate"]))
        return 1

    train = [read_bpseq(DATA / f) for f in listed("TR0-canonicals.lst")]
    out["vienna_reference"] = {"n_sequences": len(train), "core": VIENNA_CORE,
                               "load_1min_before": os.getloadavg()[0]}
    out["vienna_reference"]["mfe_seconds"] = vienna_reference(train)
    os.sched_setaffinity(0, set(range(os.cpu_count())))
    write(out)

    arch = OUT / "arch.conf"
    lines = CONF.read_text().split("\n")
    keep, skip = [], False
    for ln in lines:
        if skip:
            skip = False
            continue
        if ln.strip() == "--param":
            skip = True
            continue
        if ln.strip():
            keep.append(ln)
    arch.write_text("\n".join(keep) + "\n")

    procs = []
    for n, seed in enumerate(seeds):
        d = OUT / "seeds" / f"seed{seed}_w{PHASE_WEIGHT}"
        (d / "checkpoints").mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(n % 2), OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", MXF_TIMING_OUT=str(d / "timing.json"))
        cmd = [str(PY_MX), str(RUNNER), "train", f"@{arch}", "--param", str(d / "param.pth"),
               "--save-config", str(d / "model.conf"), "--log-dir", str(d / "checkpoints"),
               "--epochs", "1", "--gpu", "0", "--seed", str(seed), "--loss-func", "hinge_mix",
               "--score-loss-weight", str(PHASE_WEIGHT), "--disable-progress-bar",
               "data/TR0-canonicals.lst"]
        log = (d / "stdout.log").open("w")
        procs.append((seed, n % 2, subprocess.Popen(cmd, cwd=DATA, env=env, stdout=log,
                                                    stderr=subprocess.STDOUT)))
    out["status"] = "RUNNING phase epoch1"
    out["launched_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write(out)

    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                samples.append({"t": time.time(), "gpus": gpus(), "load": os.getloadavg()[0]})
            except Exception:  # noqa: BLE001 -- a failed sample must not kill the study
                pass
            stop.wait(30)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    codes = {seed: p.wait() for seed, _, p in procs}
    stop.set()
    th.join()

    vienna = out["vienna_reference"]["mfe_seconds"]
    per = {}
    for seed, gpu, _ in procs:
        d = OUT / "seeds" / f"seed{seed}_w{PHASE_WEIGHT}"
        tj = d / "timing.json"
        if codes[seed] != 0 or not tj.exists():
            per[str(seed)] = {"gpu": gpu, "exit_code": codes[seed], "timing": None}
            continue
        t = json.loads(tj.read_text())
        ep = t["epochs"][0]
        steps = ep["n_sequences"]
        dp = t["dp_pred_s"] + t["dp_ref_s"] + t["dp_turner_s"]
        per[str(seed)] = {
            "gpu": gpu, "exit_code": 0, "epoch_seconds": ep["seconds"], "steps": steps,
            "dp_pred_s": t["dp_pred_s"], "dp_ref_s": t["dp_ref_s"],
            "dp_turner_s": t["dp_turner_s"], "nn_forward_s": t["nn_forward_s"],
            "backward_s": t["backward_s"], "overhead_s": t["overhead_s"],
            "per_step_overhead_ms": 1000 * t["overhead_s"] / steps,
            "folding_factor_unconstrained": t["dp_pred_s"] / vienna,
            "folding_factor_all_dp": dp / vienna,
            "gpu_peak_mib": t["gpu_peak_mib"]}
    ok = [v for v in per.values() if v.get("exit_code") == 0 and v.get("epoch_seconds")]
    summ = {}
    if ok:
        for key in ("epoch_seconds", "per_step_overhead_ms", "folding_factor_unconstrained",
                    "folding_factor_all_dp", "dp_pred_s", "nn_forward_s", "backward_s",
                    "gpu_peak_mib"):
            xs = sorted(v[key] for v in ok)
            summ[key] = {"min": xs[0], "median": xs[len(xs) // 2], "max": xs[-1]}
    util = {}
    for i in (0, 1):
        us = [s["gpus"][i]["util"] for s in samples if len(s["gpus"]) > i]
        util[str(i)] = sum(us) / len(us) if us else None
    out["phase_epoch1"] = {
        "weight": PHASE_WEIGHT, "seeds": per, "n_completed": len(ok), "summary": summ,
        "gpu_mean_utilisation_pct": util,
        "load_1min_mean": (sum(s["load"] for s in samples) / len(samples)) if samples else None,
        "notes": ["folding factors divide MXfold2's fold time, measured with 8 training "
                  "processes running, by ViennaRNA's, measured alone on one pinned core: "
                  "contention can only inflate them",
                  "nn_forward and backward include CUDA synchronisation; the first step of "
                  "each process includes CUDA warm-up"]}
    out["status"] = ("epoch1 COMPLETE; stopped by design for review" if len(ok) == 8
                     else f"epoch1 finished with {8 - len(ok)} failed seed(s)")
    out["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write(out)
    print(json.dumps(out["phase_epoch1"]["summary"], indent=1))
    return 0 if len(ok) == 8 else 1


# ==================================================================================
# Phases after epoch1 (user instructions of 2026-10-08)
# ==================================================================================
ALLOC_CONF = "expandable_segments:True"
MAX_PER_GPU = 3
DET_SEED_INDEX = 0          # the seed trained twice for the determinism control
FULL_EPOCHS = 10
FULL_DONE = "full COMPLETE (training)"  # the completion status successors wait for
OOM_RETRIES = 1

DETERMINISM_PREREG = {
    "what": "One seed (the first recorded seed) trained twice under deliberately different "
            "GPU co-location, compared with each other and with its original epoch-1 run.",
    "epoch1_pair": {
        "detA": "GPU 0, sharing the card with the two OOM re-runs (3 processes)",
        "detB": "GPU 1, alone (1 process)",
        "original": "the phase-epoch1 run: GPU 0, 4 processes, default CUDA allocator",
    },
    "ten_epoch_pairs": "the full study adds one duplicate per weight arm on the OTHER GPU, "
                       "in a different wave, so the floor is also measured at epoch 10",
    "mxfold2_determinism_flags_from_source": {
        "file": "vendor/mxfold2/mxfold2/train.py (51b2136), Train.run, when --seed >= 0",
        "sets": ["torch.manual_seed(seed)  (seeds CPU and CUDA generators)",
                 "random.seed(seed)", "torch.backends.cudnn.deterministic = True",
                 "torch.backends.cudnn.benchmark = False"],
        "does_not_set": ["torch.use_deterministic_algorithms(True)",
                         "CUBLAS_WORKSPACE_CONFIG", "numpy's generator (not used for training)"],
        "runtime_record": "each process also records the flags as observed at runtime "
                          "(timing.json: determinism)",
    },
    "measures": [
        "bitwise equality of every model and optimiser tensor after epoch 1 (and epoch 10)",
        "TS0: per-sequence predicted structure and score differences",
        "TS0: the pair's score-table difference decomposed exactly as the seed variance "
        "(V_G and V_tot with n = 2, i.e. ||theta_A - theta_B||^2 / 2 per sequence), by the "
        "evaluator, after the validity checks V1-V3",
    ],
    "rule_user": "If training is bitwise deterministic the floor is exactly zero. If it is "
                 "not, every later seed-variance figure is reported against this floor.",
    "prior_claude": "Bitwise identical across co-location. Reason: batch size 1, seeded CUDA "
                    "generator, cuDNN deterministic with benchmark off, folding DP on the CPU. "
                    "The most likely way it fails: with benchmark off, cuDNN still picks "
                    "algorithms heuristically and that choice can depend on free workspace "
                    "memory, which co-location changes; cuBLAS runs without "
                    "CUBLAS_WORKSPACE_CONFIG; atomics in some backward kernels.",
}

PROJECTION_CORRECTION = (
    "Correction recorded 2026-10-08, BEFORE any decomposition is computed. The protocol states "
    "that h is orthogonal to every g_p (h.g_p = 1 - 1 = 0). That holds only for pairs where "
    "both S_in(i,j) and S_out(j,i) exist. A pair that can stack on one side only, e.g. i = 1 "
    "(no outside stack), has h.g_p = +1 or -1. P_G is therefore computed with h replaced by "
    "h' = h - sum_p (h.g_p / g_p.g_p) g_p (Gram-Schmidt). The span G(x), every prediction and "
    "every other definition are unchanged; this fixes the arithmetic, not the question.")

FULL_DESIGN = {
    "runs": "2 weights x 8 seeds, each a FRESH continuous 10-epoch `mxfold2 train` run, plus "
            "one duplicate per weight (determinism floor at epoch 10): 18 runs",
    "why_fresh_not_resumed": "MXfold2's --resume re-seeds torch at start and restores no RNG "
                             "state (train.py Train.run), so a resumed epoch 2 replays epoch "
                             "1's shuffle order and dropout masks exactly. Continuing the "
                             "epoch-1 checkpoints would not be MXfold2's own 10-epoch run. "
                             "The epoch-1 checkpoints serve timing and the determinism control.",
    "pairing": "data order, initialisation and dropout draws depend only on the seed (the "
               "same number of generator draws per step in both arms), so each seed is a "
               "paired comparison between the two weights",
    "placement": "seed i, weight w -> GPU (i + arm index) mod 2, so each GPU carries 4 runs "
                 "of each arm; duplicates on the GPU opposite their original, queued last",
    "concurrency": f"at most {MAX_PER_GPU} processes per GPU (6 at once); a freed slot starts "
                   "the next queued run on that GPU",
    "allocator": f"PYTORCH_CUDA_ALLOC_CONF={ALLOC_CONF}",
    "oom_policy": f"a run that fails is restarted from scratch at most {OOM_RETRIES} time(s); "
                  "every attempt is recorded",
    "cpu_contention": "the family-matched Rfam control may run concurrently with a thread cap; "
                      "timing is not a deliverable of this phase",
}


def arch_conf():
    arch = OUT / "arch.conf"
    if not arch.exists():
        keep, skip = [], False
        for ln in CONF.read_text().split("\n"):
            if skip:
                skip = False
                continue
            if ln.strip() == "--param":
                skip = True
                continue
            if ln.strip():
                keep.append(ln)
        arch.write_text("\n".join(keep) + "\n")
    return arch


def read_out():
    return json.loads((OUT / "metrics.json").read_text())


def start_run(run, arch):
    d = OUT / "seeds" / run["name"]
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(run["gpu"]), OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTORCH_CUDA_ALLOC_CONF=ALLOC_CONF,
               MXF_TIMING_OUT=str(d / "timing.json"))
    cmd = [str(PY_MX), str(RUNNER), "train", f"@{arch}", "--param", str(d / "param.pth"),
           "--save-config", str(d / "model.conf"), "--log-dir", str(d / "checkpoints"),
           "--epochs", str(run["epochs"]), "--gpu", "0", "--seed", str(run["seed"]),
           "--loss-func", "hinge_mix", "--score-loss-weight", str(run["weight"]),
           "--disable-progress-bar", "data/TR0-canonicals.lst"]
    log = (d / f"stdout.attempt{run['attempt']}.log").open("w")
    return subprocess.Popen(cmd, cwd=DATA, env=env, stdout=log, stderr=subprocess.STDOUT)


def schedule(runs, out, key):
    """Run every entry of `runs` with at most MAX_PER_GPU live processes per GPU."""
    arch = arch_conf()
    queue = {0: [r for r in runs if r["gpu"] == 0], 1: [r for r in runs if r["gpu"] == 1]}
    live, rec = [], {r["name"]: {**{k: r[k] for k in ("seed", "weight", "gpu", "epochs",
                                                     "role")}, "attempts": []} for r in runs}
    while any(queue.values()) or live:
        for g in (0, 1):
            while queue[g] and sum(1 for _, r in live if r["gpu"] == g) < MAX_PER_GPU:
                r = queue[g].pop(0)
                r.setdefault("attempt", 0)
                live.append((start_run(r, arch), r))
                rec[r["name"]]["attempts"].append({"attempt": r["attempt"],
                                                   "started": time.strftime("%H:%M:%S")})
        for p, r in list(live):
            code = p.poll()
            if code is None:
                continue
            live.remove((p, r))
            a = rec[r["name"]]["attempts"][-1]
            a.update({"exit_code": code, "ended": time.strftime("%H:%M:%S")})
            if code != 0 and r["attempt"] < OOM_RETRIES:
                r["attempt"] += 1
                queue[r["gpu"]].insert(0, r)
        for name, v in rec.items():
            ck = OUT / "seeds" / name / "checkpoints"
            v["epochs_checkpointed"] = len(list(ck.glob("epoch-*"))) if ck.exists() else 0
        out[key] = {"runs": rec, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        write(out)
        time.sleep(30)
    return rec


def compare_checkpoints(a, b):
    """Bitwise comparison of two MXfold2 checkpoints (model and optimiser state)."""
    import torch

    ca = torch.load(a, map_location="cpu", weights_only=True)
    cb = torch.load(b, map_location="cpu", weights_only=True)
    out = {"model_tensors": 0, "model_tensors_bitwise_equal": 0, "model_max_abs_diff": 0.0}
    for k, ta in ca["model_state_dict"].items():
        tb = cb["model_state_dict"][k]
        out["model_tensors"] += 1
        eq = torch.equal(ta, tb)
        out["model_tensors_bitwise_equal"] += int(eq)
        if not eq and ta.is_floating_point():
            out["model_max_abs_diff"] = max(out["model_max_abs_diff"],
                                            float((ta - tb).abs().max()))

    def flat(state):
        xs = []
        for st in state["state"].values():
            xs += [v for v in st.values() if torch.is_tensor(v)]
        return xs

    oa, ob = flat(ca["optimizer_state_dict"]), flat(cb["optimizer_state_dict"])
    out["optimizer_tensors"] = len(oa)
    out["optimizer_tensors_bitwise_equal"] = sum(int(torch.equal(x, y))
                                                 for x, y in zip(oa, ob, strict=True))
    out["bitwise_identical"] = (out["model_tensors_bitwise_equal"] == out["model_tensors"]
                                and out["optimizer_tensors_bitwise_equal"] == len(oa))
    return out


def gpu_gate(out, key):
    g = gpus()
    out[key] = {"gpus": g, "min_free_mib": MIN_FREE_MIB, "load_1min": os.getloadavg()[0],
                "checked": time.strftime("%Y-%m-%d %H:%M:%S")}
    ok = len(g) >= 2 and all(x["free_mib"] >= MIN_FREE_MIB for x in g)
    if not ok:
        out["status"] = f"NOT LAUNCHED ({key}): a GPU is not free"
    write(out)
    return ok


def phase_rerun_determinism():
    seeds = recorded_seeds()
    out = read_out()
    out["determinism_control"] = {"preregistered": DETERMINISM_PREREG,
                                  "preregistered_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out["projection_correction"] = PROJECTION_CORRECTION
    out["provenance_phase_rerun_determinism"] = git_provenance(R)
    write(out)
    if not gpu_gate(out, "launch_gate_rerun_determinism"):
        return 1
    s0 = seeds[DET_SEED_INDEX]
    runs = [
        {"name": f"seed{seeds[3]}_w1.0_rerun", "seed": seeds[3], "weight": 1.0, "gpu": 0,
         "epochs": 1, "role": "re-run of an OOM seed"},
        {"name": f"seed{seeds[6]}_w1.0_rerun", "seed": seeds[6], "weight": 1.0, "gpu": 0,
         "epochs": 1, "role": "re-run of an OOM seed"},
        {"name": f"seed{s0}_w1.0_detA", "seed": s0, "weight": 1.0, "gpu": 0, "epochs": 1,
         "role": "determinism A: GPU 0 with the two re-runs"},
        {"name": f"seed{s0}_w1.0_detB", "seed": s0, "weight": 1.0, "gpu": 1, "epochs": 1,
         "role": "determinism B: GPU 1 alone"},
    ]
    out["status"] = "RUNNING phase rerun-determinism"
    write(out)
    rec = schedule(runs, out, "phase_rerun_determinism")
    ck = {k: OUT / "seeds" / k / "checkpoints" / "epoch-1"
          for k in (f"seed{s0}_w1.0", f"seed{s0}_w1.0_detA", f"seed{s0}_w1.0_detB")}
    res = {}
    if all(p.exists() for p in ck.values()):
        res = {"detA_vs_detB": compare_checkpoints(ck[f"seed{s0}_w1.0_detA"],
                                                    ck[f"seed{s0}_w1.0_detB"]),
               "original_vs_detA": compare_checkpoints(ck[f"seed{s0}_w1.0"],
                                                        ck[f"seed{s0}_w1.0_detA"]),
               "original_vs_detB": compare_checkpoints(ck[f"seed{s0}_w1.0"],
                                                        ck[f"seed{s0}_w1.0_detB"])}
    flags = {}
    for name in rec:
        tj = OUT / "seeds" / name / "timing.json"
        if tj.exists():
            t = json.loads(tj.read_text())
            flags[name] = t.get("determinism")
            rec[name]["epoch_seconds"] = t["epochs"][0]["seconds"] if t["epochs"] else None
            rec[name]["gpu_peak_mib"] = t.get("gpu_peak_mib")
    out["determinism_control"]["epoch1_bitwise"] = res
    out["determinism_control"]["runtime_flags"] = flags
    ok = all(v["attempts"] and v["attempts"][-1].get("exit_code") == 0 for v in rec.values())
    out["status"] = ("rerun-determinism COMPLETE" if ok and res
                     else "rerun-determinism finished with failures")
    write(out)
    print(json.dumps(res, indent=1))
    return 0 if ok and res else 1


def phase_full():
    seeds = recorded_seeds()
    out = read_out()
    out["full_study"] = {"design": FULL_DESIGN, "preregistered_at":
                         time.strftime("%Y-%m-%d %H:%M:%S"), **git_provenance(R)}
    runs = []
    for i, s in enumerate(seeds):
        for a, w in enumerate(WEIGHTS):
            runs.append({"name": f"full_seed{s}_w{w}", "seed": s, "weight": w,
                         "gpu": (i + a) % 2, "epochs": FULL_EPOCHS, "role": "study"})
    for a, w in enumerate(WEIGHTS):
        runs.append({"name": f"full_seed{seeds[0]}_w{w}_dup", "seed": seeds[0], "weight": w,
                     "gpu": 1 - (0 + a) % 2, "epochs": FULL_EPOCHS,
                     "role": "determinism duplicate (other GPU, last wave)"})
    out["full_study"]["assignment"] = [{k: r[k] for k in ("name", "seed", "weight", "gpu",
                                                          "role")} for r in runs]
    write(out)
    if not gpu_gate(out, "launch_gate_full"):
        return 1
    out["status"] = "RUNNING phase full"
    write(out)
    rec = schedule(runs, out, "phase_full")
    for w in WEIGHTS:
        o = OUT / "seeds" / f"full_seed{seeds[0]}_w{w}" / "checkpoints" / f"epoch-{FULL_EPOCHS}"
        d = OUT / "seeds" / f"full_seed{seeds[0]}_w{w}_dup" / "checkpoints" / f"epoch-{FULL_EPOCHS}"
        if o.exists() and d.exists():
            out.setdefault("determinism_control", {}).setdefault("epoch10_bitwise", {})[
                f"w{w}"] = compare_checkpoints(o, d)
    ok = all(v["epochs_checkpointed"] == FULL_EPOCHS for v in rec.values())
    out["status"] = FULL_DONE if ok else "full finished with failures"
    write(out)
    return 0 if ok else 1


# ----------------------------------------------------------------------------------
# the runs interrupted on 2026-10-09, rerun fresh
# ----------------------------------------------------------------------------------
INTERRUPTED = ("full_seed20261006_w0.125_dup", "full_seed20261006_w1.0_dup",
               "full_seed20261012_w0.125", "full_seed20261012_w1.0",
               "full_seed20261013_w0.125", "full_seed20261013_w1.0")
INTERRUPTED_DIR = OUT / "interrupted_2026-10-09"
INTERRUPTION_CAUSE = (
    "As far as known: the phase-full driver ran as a background shell of a Claude Code "
    "session, and that session's process ended at about 05:59:22 on 2026-10-09 (its "
    "transcript records a pending question torn down at 09:59:22 UTC), 2 s after the "
    "driver's last artifact write; a process that ends takes its background shells and their "
    "children with it. No reboot (the machine has been up since 2026-09-30 15:08), and the "
    "gauge launcher, started detached with setsid, kept running and passed its gate at "
    "06:00:04. Not established: which signal ended the processes, or why the session ended. "
    "The six runs' stdout logs are empty because their block-buffered output was lost when "
    "they were killed, so the logs hold no evidence either way; their attempts carry no exit "
    "code because the driver died before it could poll them.")
RERUN_POLICY = (
    "The six were rerun FRESH, from initialisation for 10 continuous epochs, with their "
    "registered seeds, weights and GPUs and in their registered run directories, under the "
    "registered no-resume policy (full_study.design.why_fresh_not_resumed); they were NOT "
    "resumed from their partial checkpoints. The partial directories were moved, unchanged, "
    "to interrupted_2026-10-09/. The rerun driver runs as its own session leader (setsid), "
    "so no terminal or agent session can end it.")


def phase_rerun_interrupted():
    """Record the 2026-10-09 interruption as an event, then rerun the six runs it cut short.

    Refuses unless this process is its own session leader (launched with setsid), the
    artifact still records the interruption (status RUNNING phase full with no live
    seed-study process: a fault by scripts/predecessor.py's rule, read from the recorded
    status and confirmed by the process scan), and the runs short of epoch 10 are exactly
    the six recorded in INTERRUPTED."""
    from scripts.predecessor import processes, state

    if os.getsid(0) != os.getpid():
        raise SystemExit("refusing: not a session leader. Launch with setsid (see the "
                         "docstring) so that no terminal or agent session can end this driver")
    out = read_out()
    st = state(OUT / "metrics.json", {FULL_DONE},
               lambda: processes(script="mxfold2_seed_study.py", arg=f"@{OUT / 'arch.conf'}"))
    if not (st["state"] == "fault" and st["status"] == "RUNNING phase full"):
        raise SystemExit(f"refusing: the artifact does not record the interruption ({st})")
    assign = {a["name"]: a for a in out["full_study"]["assignment"]}
    short = sorted(n for n in assign if not (OUT / "seeds" / n / "checkpoints" /
                                             f"epoch-{FULL_EPOCHS}").exists())
    if short != sorted(INTERRUPTED):
        raise SystemExit(f"refusing: the runs short of epoch {FULL_EPOCHS} are {short}, not "
                         f"the six recorded")
    if INTERRUPTED_DIR.exists():
        raise SystemExit(f"refusing: {INTERRUPTED_DIR.name}/ already exists")

    stamp = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))  # noqa: E731
    affected = {}
    for n in INTERRUPTED:
        eps = sorted((OUT / "seeds" / n / "checkpoints").glob("epoch-*"),
                     key=lambda q: int(q.name.split("-")[1]))
        affected[n] = {**{k: assign[n][k] for k in ("seed", "weight", "gpu", "role")},
                       "epochs_checkpointed": len(eps),
                       "last_checkpoint": eps[-1].name if eps else None,
                       "last_checkpoint_written": stamp(eps[-1].stat().st_mtime) if eps else None,
                       "attempts_as_recorded": out["phase_full"]["runs"][n]["attempts"],
                       "moved_to": str((INTERRUPTED_DIR / n).relative_to(OUT))}
    event = {
        "event": "interruption of phase full",
        "stopped": {"last_driver_write": out["phase_full"]["updated"],
                    "bound": "the driver rewrites the artifact every 30 s, so it stopped "
                             "within 30 s after its last write",
                    "status_left_behind": st["status"],
                    "detected_as": st},
        "affected_runs": affected,
        "unaffected": f"the other {len(assign) - len(INTERRUPTED)} runs had reached epoch "
                      f"{FULL_EPOCHS} and are kept as they are",
        "cause": INTERRUPTION_CAUSE,
        "remedy": RERUN_POLICY,
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "driver": {"pid": os.getpid(), "sid": os.getsid(0), "session_leader": True,
                   "ppid": os.getppid()},
        "partial_directories_moved": False,
    }
    out.setdefault("events", []).append(event)
    out["provenance_phase_rerun_interrupted"] = git_provenance(R)
    write(out)
    INTERRUPTED_DIR.mkdir()
    for n in INTERRUPTED:
        os.rename(OUT / "seeds" / n, INTERRUPTED_DIR / n)
    event["partial_directories_moved"] = True
    write(out)

    if not gpu_gate(out, "launch_gate_rerun_interrupted"):
        return 1
    runs = [{**{k: assign[n][k] for k in ("name", "seed", "weight", "gpu", "role")},
             "epochs": FULL_EPOCHS} for n in INTERRUPTED]
    out["status"] = "RUNNING phase rerun-interrupted"
    event["rerun"] = {"phase": "rerun-interrupted", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    write(out)
    rec = schedule(runs, out, "phase_rerun_interrupted")
    seeds = recorded_seeds()
    for w in WEIGHTS:
        o = OUT / "seeds" / f"full_seed{seeds[0]}_w{w}" / "checkpoints" / f"epoch-{FULL_EPOCHS}"
        d = OUT / "seeds" / f"full_seed{seeds[0]}_w{w}_dup" / "checkpoints" / f"epoch-{FULL_EPOCHS}"
        if o.exists() and d.exists():
            out.setdefault("determinism_control", {}).setdefault("epoch10_bitwise", {})[
                f"w{w}"] = compare_checkpoints(o, d)
    ok = all((OUT / "seeds" / n / "checkpoints" / f"epoch-{FULL_EPOCHS}").exists()
             for n in assign)
    event["rerun"].update({"ended": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "six_reached_epoch_10": all(rec[n]["epochs_checkpointed"] ==
                                                       FULL_EPOCHS for n in INTERRUPTED)})
    out["status"] = FULL_DONE if ok else "full finished with failures"
    write(out)
    return 0 if ok else 1


if __name__ == "__main__":
    if "--phase" in sys.argv:
        ph = sys.argv[sys.argv.index("--phase") + 1]
        sys.exit({"rerun-determinism": phase_rerun_determinism, "full": phase_full,
                  "rerun-interrupted": phase_rerun_interrupted}[ph]())
    sys.exit(main())
