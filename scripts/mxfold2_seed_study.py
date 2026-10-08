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

Run: .venv/bin/python scripts/mxfold2_seed_study.py
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
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))


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


if __name__ == "__main__":
    sys.exit(main())
