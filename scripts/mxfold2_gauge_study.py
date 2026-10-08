#!/usr/bin/env python3
"""MXfold2 gauge methods study: pre-registration, Phase 0, lambda pilot, GPU runs.

Plan approved 2026-10-08 (/home/shadi/.claude/plans/wild-kindling-moth.md). Phases:
  preregister  write runs/2026-10-08_mxfold2_gauge_prereg/metrics.json ONCE from PREREG
  phase0       head and runner checks (scripts/mxfold2_head_checks.py), CPU
  pilot        lambda selection (scripts/mxfold2_lambda_pilot.py), CPU
  runs         GPU runs: wait for the live seed study, smoke test, SGD + regulariser arms
Every phase after preregister refuses to run unless the pre-registration artifact is
tracked by git and unmodified and the tree is clean outside runs/. Each phase writes
ONLY its own run directory (path guard in `write`). It never writes the live seed
study's metrics.json, which its own scheduler owns: from that module only the pure
functions recorded_seeds, gpus and compare_checkpoints are imported.

Run (in .venv-mxfold2): .venv-mxfold2/bin/python scripts/mxfold2_gauge_study.py --phase X
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.mxfold2_seed_study import compare_checkpoints, gpus, recorded_seeds  # noqa: E402
from scripts.runmeta import git_provenance  # noqa: E402

PREREG_DIR = R / "runs" / "2026-10-08_mxfold2_gauge_prereg"
PHASE0_DIR = R / "runs" / "2026-10-08_mxfold2_gauge_phase0"
PILOT_DIR = R / "runs" / "2026-10-08_mxfold2_gauge_pilot"
RUNS_DIR = R / "runs" / "2026-10-08_mxfold2_gauge_runs"
OWN_DIRS = (PREREG_DIR, PHASE0_DIR, PILOT_DIR, RUNS_DIR)
LIVE = R / "runs" / "2026-10-07_mxfold2_seed_study"
DATA = R / ".cache" / "mxfold2-data"
PY_MX = R / ".venv-mxfold2" / "bin" / "python"
GAUGE_RUNNER = R / "scripts" / "mxfold2_gauge_train.py"
ALLOC_CONF = "expandable_segments:True"

PREREG = {
    "title": "MXfold2 gauge methods: post-hoc head, SGD on the weight-space gauge v, "
             "null-component regulariser",
    "plan": "approved 2026-10-08; /home/shadi/.claude/plans/wild-kindling-moth.md",
    "established_before_registration": {
        "head_is_training_noop": "An output head theta -> theta - P_G theta leaves the hinge_mix "
            "loss unchanged as a function of the network weights: the loss depends on theta only "
            "through structure totals, and d loss/d theta is a combination of structure count "
            "vectors, orthogonal to G(x). Proved and checked by three independent lenses "
            "(weight gradients agree to 2.4e-7 relative); P0.3 regenerates the check into an "
            "artifact. Network form of docs/gauge.md:116-122.",
        "float32": "MXfold2's C++ DP is float32 (ScoreType float; float64 tensors silently give "
                   "garbage, src/param/util.h), so the defining check is stated in float32.",
        "weight_space_gauge_v": "v = +1 on zuker.net.fc_paired.fc.4.bias[0] (S channel), -2 on "
            "zuker.fc_length.score_helix_length.net.0.weight[0, 2..30]. Moving the weights along "
            "v moves every S entry by delta and helix_length[m] by -2 delta (m-1), i.e. along the "
            "output gauge h; the loss is flat along v on every step whose pred and ref contain no "
            "helix longer than 30 pairs (3 TR0 references do). Under AdamW, v.w moved by -4.8 to "
            "+5.4 across the 8 seeds in epoch 1 (seed SD ~1 at init, ~3.2 after epoch 1).",
        "turner_frozen": "In MixC under hinge_mix the Turner tables receive no gradient.",
        "resume_inexact": "MXfold2 --resume re-seeds and restores no RNG state; cuDNN's RNN "
                          "dropout state cannot be restored. Arm 1 of the original proposal is "
                          "not resumed.",
        "live_study": "Arms 1 (AdamW w=1.0) and 2 (AdamW w=0.125) are the live fresh runs of "
                      "runs/2026-10-07_mxfold2_seed_study, with duplicates for an epoch-10 floor.",
    },
    "phase0": {
        "P0_1_masks": "torch masks equal Geometry (uppercased) on every TR0 and TS0 sequence; "
                      "torch null_norm2 equals Geometry.proj_norm2 on the TS0 theta of the shipped "
                      "TrainSetAB weights to <= 1e-10 relative",
        "P0_1_rel_tol": 1e-10,
        "P0_2_v_symmetry": "30 seeded TR0 sequences, shipped weights, eval mode, delta = 0.37: "
                           "every S entry moves by delta and helix_length[m] by -2 delta (m-1) "
                           "within 1e-5 absolute, M and U are unchanged within 1e-6, the hinge_mix "
                           "loss changes by <= 1e-6 relative and |v.grad L| / (|v| |grad L|) "
                           "<= 1e-6, "
                           "the last two on sequences whose pred and ref have no helix > 30 pairs",
        "P0_2": {"n_sequences": 30, "delta": 0.37, "table_abs_tol": 1e-5, "other_abs_tol": 1e-6,
                 "loss_rel_tol": 1e-6, "cos_tol": 1e-6},
        "P0_3_head_noop": "30 seeded TR0 sequences <= 200 nt, shipped weights: hinge_mix weight "
                          "gradient with the head (canonicalise inserted after make_param) vs "
                          "without, in eval mode and in train mode with identical dropout: "
                          "relative L2 difference <= 1e-5 and identical argmax structures; "
                          "deterministic positive control: projecting out the unit direction of "
                          "the step's own theta-gradient, after removing its G component (a "
                          "non-null direction by construction), must change the weight gradient "
                          "by >= 1e-3 relative on every sequence and mode",
        "P0_3": {"n_sequences": 30, "max_len": 200, "grad_rel_tol": 1e-5,
                 "control_min_change": 1e-3},
        "P0_4_defining_check": "all 1,305 TS0 sequences, shipped TrainSetAB weights: identical "
                               "predicted structures with and without the head, except near-tie "
                               "flips whose score gap under the original parameters is <= "
                               "tie_bound * max(1, |E|) + abs_floor; |dE| <= dE_rel_tol * "
                               "max(1, |E|) + abs_floor for the predicted and (where realisable) "
                               "the reference structure (abs_floor covers float32 summation, "
                               "whose error scales with the partial sums rather than |E|); "
                               "residual null norm of the canonical theta <= 1e-12 * ||theta||^2",
        "P0_4": {"dE_rel_tol": 1e-6, "tie_bound": 1e-5, "abs_floor": 2e-5,
                 "residual_rel_tol": 1e-12},
        "P0_4_negative_controls": {
            "NC1": "theta - ||P_G theta|| r (r a random direction on theta's coordinates, its G "
                   "component removed, unit norm in theta space) must change the predicted-"
                   "structure score by > 1e-3 relative on >= 95% of sequences and change at "
                   "least one predicted structure",
            "NC2": "the head without h (g_p only) must leave a residual null norm >= 100 x the "
                   "full head's on >= 90% of the sequences where h is in G",
            "NC3": "a random G perturbation (norm = ||theta||) applied before the head must not "
                   "survive it: canonical tables equal within 1e-4 absolute, and the predicted "
                   "structure equal to the head's own (s_c) up to near-ties as in P0.4, on every "
                   "sequence",
            "thresholds": {"NC1_rel": 1e-3, "NC1_frac": 0.95, "NC2_ratio": 100.0,
                           "NC2_frac": 0.90, "NC3_abs": 1e-4},
        },
        "P0_5_runner": "CPU, 20 seeded TR0 sequences <= 80 nt: (a) gauge runner with lambda = 0 "
                       "is bitwise equal to the plain timed runner after 1 epoch (AdamW); (b) the "
                       "same for --optimizer SGD; (c) 1 epoch + resume to epoch 2 with the RNG "
                       "sidecar is bitwise equal to 2 uninterrupted epochs; (d) the stdout "
                       "'Train Epoch' loss lines parse; (e) positive control: lambda = 1 changes "
                       "the weights; (f) the float64 CPU crosscheck of R agrees to <= 1e-9",
        "P0_5": {"n_sequences": 20, "max_len": 80, "crosscheck_steps": 5, "crosscheck_tol": 1e-9},
        "stop_rule": "any Phase 0 failure stops everything; nothing trains",
    },
    "pilot": {
        "start": "seeds/seed20261006_w1.0/checkpoints/epoch-1 of the live study (model and AdamW "
                 "state), read only",
        "data": "500 seeded TR0 sequences <= 200 nt as the training stream; 200 disjoint seeded "
                "TR0 sequences <= 200 nt held out; TS0 never used",
        "arms": "lambda = 0 and lambda_f = f * L0 / R0 for f in grid, where L0 and R0 are the "
                "held-out mean hinge_mix loss and mean R at step 0 in eval mode; every arm "
                "re-seeded identically (same order, same dropout)",
        "grid_f": [0.01, 0.03, 0.1, 0.3, 1.0],
        "steps": 500, "eval_steps": [0, 100, 250, 500], "seed": 20261008,
        "rule": "choose the smallest f with rho = R_ho(lambda, 500) / R_ho(0, 500) <= 0.5 and "
                "delta = L_ho(lambda, 500) / L_ho(0, 500) - 1 <= 0.02; if none qualifies, the "
                "regulariser arm is NOT launched and the SGD arm proceeds alone",
        "rho_max": 0.5, "delta_max": 0.02,
    },
    "runs": {
        "arms": {
            "sgd": "seeds 20261006, 20261007, 20261008; --optimizer SGD (MXfold2: Nesterov, "
                   "momentum 0.9, wd 0) at the default lr 1e-3; w = 1.0; lambda = 0 (R logged)",
            "reg": "all 8 recorded seeds; AdamW (code default); w = 1.0; lambda = lambda* from "
                   "the pilot artifact",
            "reg_dup": "a duplicate of reg seed 20261006 on the other GPU (epoch-10 floor; "
                       "fills wave 2's idle slot)",
        },
        "pairing": "every run is seed-paired with the live AdamW w=1.0 run of the same seed "
                   "(same initialisation, data order and dropout draws)",
        "settings": "TR0 uncapped, 10 epochs, checkpoints every epoch, "
                    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "wait": "start only after no live seed-study process remains",
        "smoke": "first: two concurrent 1-epoch runs on 200 seeded TR0 sequences, one per GPU, "
                 "in the reg configuration (or SGD with lambda = 0 if no lambda* qualified): "
                 "checkpoints bitwise identical and the CPU float64 crosscheck of R <= 1e-9; "
                 "peak memory and epoch time recorded; any failure stops",
        "limits": "<= 3 of our processes per GPU; at most one start per GPU per 120 s; a run "
                  "starts only if its GPU has >= 7500 MiB free, the disk >= 5 GB free and the run "
                  "code (runner, hooks, gauge, timed wrapper) is byte-identical to launch",
        "amendments": "any change after registration is appended by --phase amend with a date "
                      "and reason and committed before the step it affects",
        "resume_policy": "a run that dies resumes from its newest loadable checkpoint with the "
                         "RNG sidecar (data order exact; on the GPU every dropout mask diverges "
                         "after the first post-resume step), flagged trajectory_exact = "
                         "false; at most 2 recoveries; NaN = diverged, not retried; every "
                         "analysis also reported with flagged runs excluded; if sgd seed 20261006 "
                         "is flagged, a fresh rerun is queued last and replaces it for A-vs-B",
    },
    "analysis": {
        "data": "TS0 (1,305 sequences, the seed study's pre-registered evaluation set), CPU only",
        "decomposition": "per sequence: theta as in scripts/mxfold2_decompose.py (corrections "
                         "c1-c3); "
                         "null part = P_G, identifiable part = the rest; V1-V3 must pass on every "
                         "model before any statistic is reported",
        "statistics": "seed-level resampling: paired bootstrap over seed indices, 10,000 "
                      "resamples, default_rng(20261006); leave-one-seed-out jackknife reported "
                      "alongside; resamples that draw a single seed are dropped and counted",
        "stage_2a_2b": "2a = live arms when the live study ends; 2b = everything; 2a must "
                       "reproduce bitwise in 2b",
    },
    "predictions": {
        "G1": {"rule_user": "Without the head, A (AdamW seed 20261006, live) vs B (SGD seed "
                            "20261006) differ mostly in the null component.",
               "operational": "more than 50% of ||theta_A - theta_B||^2, pooled over TS0, lies in "
                              "G; compared at matched training loss (e_A = 10, e_B = earliest SGD "
                              "epoch with loss <= L_A(10); if none, e_B = 10 and e_A = the AdamW "
                              "epoch whose loss is closest to L_B(10); labelled descriptive if "
                              "the matched losses differ by > 10%); reported against three "
                              "baselines: dim G / dim theta, 20 random directions per sequence, "
                              "AdamW-AdamW and SGD-SGD seed pairs",
               "prior_claude": "likely refuted: a CPU probe put 16-28% of a theta change in G"},
        "G2": {"rule_user": "With the head, the identifiable component of C vs D is smaller than "
                            "that of A vs B by more than the floor.",
               "status": "REFUTED BY CONSTRUCTION before any training: the head is a training "
                         "no-op (established_before_registration.head_is_training_noop; P0.3)"},
        "G3": {"rule_user": "The floor is small relative to both.",
               "operational": "floor = the difference between two trainings of the SAME seed, "
                              "decomposed exactly like a seed difference (null and identifiable "
                              "parts on TS0): epoch 10 from the live _dup runs and reg_dup; epoch "
                              "1 from every seed with both an original phase-epoch1 checkpoint "
                              "and a live epoch-1 checkpoint",
               "correction_2026_10_08": "The earlier statement 'epoch 1 already bitwise 0' was "
                              "wrong in general. Seed 20261006 is bitwise identical in four "
                              "configurations, but seeds 20261007 and 20261008 differ between "
                              "the original phase-epoch1 runs (4 per GPU, default allocator, two "
                              "co-tenants out of memory) and the live runs (3 per GPU, "
                              "expandable segments), same GPU index: every network tensor "
                              "differs, relative L2 0.34 and 0.31 after one epoch. The floor is "
                              "therefore measured, never assumed, and every seed-variance "
                              "figure is reported against it."},
        "W1": {"rule_user": "Null-space seed variance is the same in arms 1 and 2.",
               "amended_2026_10_08": "before any analysis: the 95% interval for the median over "
                                     "sequences of log(V_G(0.125) / V_G(1.0)) comes from a paired "
                                     "SEED-LEVEL bootstrap (all 1,305 per-sequence variances "
                                     "share the same 8 models, so a sequence bootstrap omits the "
                                     "dominant error); 'same' = interval inside [log 0.8, "
                                     "log 1.25]; the original sequence bootstrap is reported as "
                                     "secondary"},
        "W2": {"rule_user": "The constant-shift direction varies more in arm 2 than arm 1.",
               "operational": "kappa_k(x) = <theta_k, c_x'> / ||c_x'||^2 with c_x = +1 on every S "
                              "and M entry and +(b-a+1) on U(a,b) (0 on U(a,a-1)), c_x' = c_x - "
                              "P_G c_x; statistic = median over x of Var_0.125 / Var_1.0 > 1, "
                              "paired seed bootstrap"},
        "H1": {"rule_user": "Arms 3 and 4 have zero null-space seed variance and smaller "
                            "identifiable seed variance than arms 1 and 2.",
               "status": "split: 'zero null variance' of canonicalised tables is a CHECK; "
                         "'smaller identifiable variance' from the head is REFUTED BY "
                         "CONSTRUCTION; the non-tautological successor is R2"},
        "H2": {"rule_user": "Arms 3 and 4 are indistinguishable because the head already pins "
                            "the constant shift the regulariser was pinning.",
               "status": "WITHDRAWN: c_x is not in G (13-79% of it lies outside), so the head "
                         "does not touch it, and H2 contradicts the filed W2"},
        "S1": {"prior_claude": "SGD conserves v.w exactly except on long-helix steps: the v.w "
                               "trace of every SGD run equals its linear reconstruction from the "
                               "logged per-step v.g under Nesterov momentum (relative error "
                               "<= 1e-6 of the trace's range), and v.w changes only on steps "
                               "whose pred or ref has a helix > 30 pairs"},
        "S2": {"prior_claude": "Under AdamW the seed SD of v.w grows from epoch 0 to epoch 10 "
                               "(live AdamW w=1.0 and w=0.125 checkpoints)"},
        "R1": {"prior_claude": "The regulariser arm has smaller null-space seed variance V_G "
                               "than live AdamW w=1.0 (a check that the penalty works), paired "
                               "seed bootstrap interval of the median log ratio below 0"},
        "R2": {"rule_user_intent": "the method claim (successor of H1)",
               "prior_claude": "The regulariser arm has smaller IDENTIFIABLE seed variance than "
                               "live AdamW w=1.0; paired seed bootstrap interval of the median "
                               "per-sequence log ratio below 0. Uncertain. If refuted, the result "
                               "is the diagnosis alone: gauge drift is real and optimiser-"
                               "dependent, harmless to predictions, and it invalidates reading "
                               "learned tables as energies."},
        "C1": {"prior_claude": "Canonicalised predictions are identical on every evaluated "
                               "model (structures; scores within the P0.4 tolerance)"},
        "F": "TS0 PPV/SEN/F per model and per arm reported with NO prediction",
    },
}


def write(path: pathlib.Path, obj) -> None:
    path = pathlib.Path(path).resolve()
    assert any(str(path).startswith(str(d.resolve()) + os.sep) for d in OWN_DIRS), path
    assert "2026-10-07_mxfold2_seed_study" not in str(path), path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prereg_path() -> pathlib.Path:
    return PREREG_DIR / "metrics.json"


def _registered(stored: dict) -> dict:
    """The registration in force: the last amendment's PREREG, else the original."""
    am = stored.get("amendments") or []
    return am[-1]["prereg"] if am else stored["prereg"]


def check_prereg(require_clean_tree: bool = True) -> dict:
    """Refuse to proceed unless the pre-registration is committed and unmodified, and the
    PREREG in this code equals the registration in force."""
    p = prereg_path()
    rel = str(p.relative_to(R))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=R,
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=R).returncode == 0
    prov = git_provenance(R)
    tree_ok = (not prov["git_dirty_excluding_runs"]) or not require_clean_tree
    if not (p.exists() and tracked and clean and tree_ok):
        raise SystemExit(f"pre-registration not committed/clean (exists={p.exists()}, "
                         f"tracked={tracked}, unmodified={clean}, tree_clean="
                         f"{not prov['git_dirty_excluding_runs']}); refusing to run")
    stored = json.loads(p.read_text())
    assert _registered(stored) == json.loads(json.dumps(PREREG, default=str)), \
        "PREREG in code differs from the registration in force"
    return {"prereg_sha256": sha256(p), "n_amendments": len(stored.get("amendments") or []),
            **prov}


def phase_preregister() -> int:
    p = prereg_path()
    if p.exists():
        raise SystemExit(f"{p} exists; a pre-registration is written once. Use --phase amend.")
    write(p, {"prereg": PREREG, "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              **git_provenance(R)})
    print(f"wrote {p.relative_to(R)}; commit it before any other phase")
    return 0


def phase_amend() -> int:
    """Append a dated amendment {reason, prereg: current PREREG}; the original block is kept.
    Usage: --phase amend --reason "..." ; commit the artifact before the affected step."""
    p = prereg_path()
    stored = json.loads(p.read_text())
    reason = sys.argv[sys.argv.index("--reason") + 1]
    if _registered(stored) == json.loads(json.dumps(PREREG, default=str)):
        raise SystemExit("PREREG is unchanged; nothing to amend")
    stored.setdefault("amendments", []).append(
        {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "reason": reason, "prereg": PREREG,
         **git_provenance(R)})
    write(p, stored)
    print(f"amendment {len(stored['amendments'])} written; commit it")
    return 0


def phase0() -> int:
    from scripts import mxfold2_head_checks as hc
    prov = check_prereg()
    return hc.run(PREREG["phase0"], PHASE0_DIR, write, prov)


def phase0_passed() -> bool:
    p = PHASE0_DIR / "metrics.json"
    return p.exists() and json.loads(p.read_text()).get("all_pass") is True


def pilot() -> int:
    from scripts import mxfold2_lambda_pilot as lp
    prov = check_prereg()
    if not phase0_passed():
        raise SystemExit("Phase 0 has not passed; the pilot does not run")
    return lp.run(PREREG["pilot"], PILOT_DIR, write, prov, LIVE)


# ----------------------------------------------------------------------------------
# GPU runs
# ----------------------------------------------------------------------------------
RUN_CODE = ("scripts/mxfold2_gauge_train.py", "scripts/mxfold2_gauge_hooks.py",
            "scripts/mxfold2_gauge.py", "scripts/mxfold2_timed_train.py")
SETTLE_S = 120          # after a start on a GPU, wait this long before the next start there
RESERVE_MIB = 7500      # memory a just-started process is assumed to take while it settles


def code_hashes() -> dict:
    return {f: sha256(R / f) for f in RUN_CODE}


def _ancestors() -> set:
    pids, pid = set(), os.getpid()
    while pid > 1:
        pids.add(pid)
        try:
            stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def live_study_running() -> bool:
    """A live seed-study process: a python whose script argument is mxfold2_seed_study.py, or
    any process given @<live arch.conf>. Our own ancestors (a launching shell) never count."""
    skip = _ancestors()
    live_arch = "@" + str(LIVE / "arch.conf")
    for d in pathlib.Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in skip:
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [x.decode(errors="ignore") for x in argv if x]
        if not argv:
            continue
        if "python" in pathlib.Path(argv[0]).name and len(argv) > 1 and \
                argv[1].endswith("mxfold2_seed_study.py"):
            return True
        if live_arch in argv:
            return True
    return False


def disk_free_gb() -> float:
    st = os.statvfs(R)
    return st.f_bavail * st.f_frsize / 2**30


def safe_gpus():
    try:
        return gpus()
    except Exception as e:  # noqa: BLE001 -- a failed query closes the gate, it does not crash
        return {"error": repr(e)}


def own_arch() -> pathlib.Path:
    """A byte-identical copy of the live study's arch.conf under our own run dir, so that
    the live-study process scan never matches our own training processes."""
    arch = RUNS_DIR / "arch.conf"
    if not arch.exists():
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        arch.write_bytes((LIVE / "arch.conf").read_bytes())
    assert sha256(arch) == sha256(LIVE / "arch.conf")
    return arch


def run_dir(name):
    return RUNS_DIR / "seeds" / name


def usable_checkpoint(name):
    """The newest checkpoint that loads (a truncated one is skipped), or None."""
    import torch
    cks = sorted(run_dir(name).glob("checkpoints/epoch-*"),
                 key=lambda q: int(q.name.split("-")[1]), reverse=True)
    for ck in cks:
        try:
            torch.load(ck, map_location="cpu", weights_only=True)
            return ck
        except Exception:  # noqa: BLE001
            continue
    return None


def start(run, arch, epochs, train_list, resume_from, hashes):
    d = run_dir(run["name"])
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    att = len(run["attempts"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(run["gpu"]), OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTORCH_CUDA_ALLOC_CONF=ALLOC_CONF,
               MXF_TIMING_OUT=str(d / f"timing.attempt{att}.json"),
               MXF_GAUGE_OUT=str(d / "gauge"), MXF_GAUGE_LAMBDA=str(run["lam"]),
               MXF_GAUGE_CROSSCHECK=str(run.get("crosscheck", 0)))
    cmd = [str(PY_MX), str(GAUGE_RUNNER), "train", f"@{arch}", "--param", str(d / "param.pth"),
           "--save-config", str(d / "model.conf"), "--log-dir", str(d / "checkpoints"),
           "--epochs", str(epochs), "--gpu", "0", "--seed", str(run["seed"]),
           "--loss-func", "hinge_mix", "--score-loss-weight", "1.0",
           "--disable-progress-bar", train_list]
    if run["optimizer"] != "AdamW":
        cmd += ["--optimizer", run["optimizer"]]
    if resume_from is not None:
        cmd += ["--resume", str(resume_from)]
    log = (d / f"stdout.attempt{att}.log").open("w")
    run["attempts"].append({"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "resume_from": str(resume_from) if resume_from else None,
                            "code_sha256": hashes})
    return subprocess.Popen(cmd, cwd=DATA, env=env, stdout=log, stderr=subprocess.STDOUT)


def non_finite(name) -> bool:
    """True if any logged loss is NaN/inf (stdout or the per-epoch gauge records)."""
    import math
    import re
    pat = re.compile(r"loss[:=]\s*(nan|inf|-inf)", re.IGNORECASE)
    if any(pat.search(f.read_text(errors="ignore"))
           for f in run_dir(name).glob("stdout.attempt*.log")):
        return True
    for f in (run_dir(name) / "gauge").glob("gauge-epoch-*.json"):
        try:
            e = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for k in ("mean_hinge", "mean_R", "v_dot_w"):
            v = e.get(k)
            if isinstance(v, float) and not math.isfinite(v):
                return True
    return False


def collect_epochs(run):
    gdir = run_dir(run["name"]) / "gauge"
    run["epochs"] = {}
    for f in sorted(gdir.glob("gauge-epoch-*.json")) if gdir.exists() else []:
        try:
            e = json.loads(f.read_text())
            run["epochs"][str(e["epoch"])] = {k: e.get(k) for k in (
                "v_dot_w", "v_dot_w_start", "mean_hinge", "mean_sl", "mean_R", "mean_lamR",
                "n_backward", "n_long_helix_steps", "max_abs_vg", "max_abs_vg_non_long_helix",
                "seconds", "resumed_from_epoch", "rng_restored", "crosscheck_rel_max")}
        except (OSError, ValueError, KeyError):
            pass


def schedule(runs, out, key, epochs, train_list, hashes, max_recover=2):
    arch = own_arch()
    for r in runs:
        r.setdefault("attempts", [])
        r.setdefault("trajectory_exact", True)
        ck = usable_checkpoint(r["name"]) if not r["attempts"] else None
        if ck is not None:  # a previous launcher died: resume, never silently restart
            r["resume_next"] = ck
            r["trajectory_exact"] = int(ck.name.split("-")[1]) >= epochs and r["trajectory_exact"]
    queue = {0: [r for r in runs if r["gpu"] == 0], 1: [r for r in runs if r["gpu"] == 1]}
    live, last_start = [], {0: 0.0, 1: 0.0}
    blocked = {}
    while any(queue.values()) or live:
        for g in (0, 1):
            if not queue[g] or sum(1 for _, r in live if r["gpu"] == g) >= 3:
                blocked.pop(str(g), None)
                continue
            if time.time() - last_start[g] < SETTLE_S:
                continue
            gs = safe_gpus()
            free = {} if isinstance(gs, dict) else {x["index"]: x["free_mib"] for x in gs}
            if free.get(g, 0) < RESERVE_MIB or disk_free_gb() < 5.0:
                blocked[str(g)] = {"since": blocked.get(str(g), {}).get(
                    "since", time.strftime("%Y-%m-%d %H:%M:%S")),
                    "free_mib": free.get(g), "disk_free_gb": disk_free_gb(),
                    "gpu_query": gs if isinstance(gs, dict) else "ok"}
                continue
            if code_hashes() != hashes:
                blocked[str(g)] = {"since": time.strftime("%Y-%m-%d %H:%M:%S"),
                                   "reason": "run code changed since launch; not starting"}
                continue
            blocked.pop(str(g), None)
            r = queue[g].pop(0)
            live.append((start(r, arch, epochs, train_list, r.pop("resume_next", None), hashes),
                         r))
            last_start[g] = time.time()
        for p, r in list(live):
            code = p.poll()
            if code is None:
                continue
            live.remove((p, r))
            r["attempts"][-1].update({"exit_code": code,
                                      "ended": time.strftime("%Y-%m-%d %H:%M:%S")})
            if non_finite(r["name"]):
                r["status"] = "diverged"
                continue
            if code == 0:
                r["status"] = "complete"
                continue
            if len(r["attempts"]) <= max_recover:
                ck = usable_checkpoint(r["name"])
                if ck is not None:
                    r["resume_next"] = ck
                    if int(ck.name.split("-")[1]) < epochs:
                        r["trajectory_exact"] = False
                queue[r["gpu"]].insert(0, r)
            else:
                r["status"] = "failed"
        for r in runs:
            collect_epochs(r)
        gs = safe_gpus()
        out[key] = {"runs": {r["name"]: {k: v for k, v in r.items() if k != "resume_next"}
                             for r in runs},
                    "blocked": blocked, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "gpus": gs, "disk_free_gb": disk_free_gb()}
        write(RUNS_DIR / "metrics.json", out)
        time.sleep(30)
    return runs


def timing_summary(name):
    rec = []
    for f in sorted(run_dir(name).glob("timing.attempt*.json")):
        try:
            t = json.loads(f.read_text())
            rec.append({"gpu_peak_mib": t.get("gpu_peak_mib"),
                        "epoch_seconds": [e["seconds"] for e in t.get("epochs", [])],
                        "overhead_s": t.get("overhead_s")})
        except (OSError, ValueError):
            pass
    return rec


def phase_runs() -> int:
    import fcntl
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    lock = open(RUNS_DIR / "launcher.lock", "w")  # noqa: SIM115 -- held for the process life
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("another --phase runs launcher holds the lock") from None
    prov = check_prereg()
    if not phase0_passed():
        raise SystemExit("Phase 0 has not passed; nothing launches")
    pil = PILOT_DIR / "metrics.json"
    if not pil.exists() or json.loads(pil.read_text()).get("status") != "complete":
        raise SystemExit("the pilot has not completed; nothing launches")
    lam = json.loads(pil.read_text()).get("lambda_star")  # None: no lambda qualified; SGD only
    hashes = code_hashes()
    seeds = recorded_seeds()
    out = {"run_id": RUNS_DIR.name, "provenance": prov, "pilot_lambda_star": lam,
           "pilot_sha256": sha256(pil), "code_sha256_at_launch": hashes,
           "status": "waiting for the live seed study"}
    write(RUNS_DIR / "metrics.json", out)
    while live_study_running():
        out["waiting_since"] = out.get("waiting_since", time.strftime("%Y-%m-%d %H:%M:%S"))
        write(RUNS_DIR / "metrics.json", out)
        time.sleep(60)
    # re-check after the wait: registration intact and run code unchanged (the rest of the
    # tree may legitimately be mid-edit, e.g. the evaluator)
    out["provenance_after_wait"] = check_prereg(require_clean_tree=False)
    if code_hashes() != hashes:
        out["status"] = "NOT LAUNCHED: run code changed during the wait"
        write(RUNS_DIR / "metrics.json", out)
        return 1
    gs = safe_gpus()
    out["launch_gate"] = {"gpus": gs, "disk_free_gb": disk_free_gb(),
                          "checked": time.strftime("%Y-%m-%d %H:%M:%S")}
    if isinstance(gs, dict) or len(gs) < 2 or any(x["free_mib"] < RESERVE_MIB for x in gs):
        out["status"] = "NOT LAUNCHED: a GPU is not free"
        write(RUNS_DIR / "metrics.json", out)
        return 1

    # smoke test: the registered reg configuration, or SGD with lambda = 0 if no lambda*
    smoke_list = RUNS_DIR / "smoke_tr0.lst"
    if not smoke_list.exists():
        import numpy as np
        lines = [ln for ln in (DATA / "data" / "TR0-canonicals.lst").read_text().split("\n")
                 if ln.strip()]
        idx = np.random.default_rng(20261008).choice(len(lines), 200, replace=False)
        smoke_list.write_text("\n".join(lines[i] for i in sorted(idx)) + "\n")
    sm_opt, sm_lam = ("AdamW", lam) if lam is not None else ("SGD", 0.0)
    sm = [{"name": f"smoke_gpu{g_}", "seed": seeds[0], "gpu": g_, "lam": sm_lam,
           "optimizer": sm_opt, "crosscheck": 5} for g_ in (0, 1)]
    out["status"] = "smoke test"
    schedule(sm, out, "smoke", 1, str(smoke_list), hashes, max_recover=0)
    a_, b_ = (usable_checkpoint(r["name"]) for r in sm)
    cmp = compare_checkpoints(a_, b_) if a_ and b_ else {"bitwise_identical": False}
    xc = [r["epochs"].get("1", {}).get("crosscheck_rel_max") for r in sm]
    ok = bool(cmp.get("bitwise_identical")) and all(x is not None and x <= 1e-9 for x in xc)
    out["smoke_result"] = {"optimizer": sm_opt, "lambda": sm_lam, "bitwise": cmp,
                           "crosscheck_rel_max": xc, "pass": ok,
                           "timing": {r["name"]: timing_summary(r["name"]) for r in sm}}
    if not ok:
        out["status"] = "STOPPED: smoke test failed; nothing else launched"
        write(RUNS_DIR / "metrics.json", out)
        return 1

    runs = []
    for i, s in enumerate(seeds[:3]):
        runs.append({"name": f"sgd_seed{s}_w1.0", "seed": s, "gpu": (i + 1) % 2, "lam": 0.0,
                     "optimizer": "SGD", "arm": "sgd"})
    if lam is not None:
        for i, s in enumerate(seeds):
            runs.append({"name": f"reg_seed{s}_w1.0", "seed": s, "gpu": (i + 1) % 2,
                         "lam": lam, "optimizer": "AdamW", "arm": "reg"})
        runs.append({"name": f"reg_seed{seeds[0]}_w1.0_dup", "seed": seeds[0], "gpu": 0,
                     "lam": lam, "optimizer": "AdamW", "arm": "reg_dup"})
    out["assignment"] = [{k: r[k] for k in ("name", "seed", "gpu", "lam", "optimizer", "arm")}
                         for r in runs]
    out["status"] = "RUNNING"
    write(RUNS_DIR / "metrics.json", out)
    schedule(runs, out, "runs", 10, "data/TR0-canonicals.lst", hashes)
    # registered: a flagged (resumed) SGD seed-0 run gets a fresh rerun that replaces it for A-vs-B
    a_run = next(r for r in runs if r["name"] == f"sgd_seed{seeds[0]}_w1.0")
    if a_run.get("status") == "complete" and not a_run.get("trajectory_exact", True):
        fresh = {"name": f"sgd_seed{seeds[0]}_w1.0_fresh", "seed": seeds[0], "gpu": a_run["gpu"],
                 "lam": 0.0, "optimizer": "SGD", "arm": "sgd_fresh"}
        out["status"] = "RUNNING fresh SGD rerun"
        schedule([fresh], out, "fresh_rerun", 10, "data/TR0-canonicals.lst", hashes,
                 max_recover=0)
        out["replacements"] = {a_run["name"]: fresh["name"]}
        runs.append(fresh)
    if lam is not None:
        o = usable_checkpoint(f"reg_seed{seeds[0]}_w1.0")
        d = usable_checkpoint(f"reg_seed{seeds[0]}_w1.0_dup")
        if o and d and o.name == d.name == "epoch-10":
            out["epoch10_floor_reg"] = compare_checkpoints(o, d)
    done = all(r.get("status") == "complete" for r in runs)
    out["status"] = "complete" if done else "finished with failed or diverged runs"
    write(RUNS_DIR / "metrics.json", out)
    return 0 if done else 1


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    sys.exit({"preregister": phase_preregister, "amend": phase_amend, "phase0": phase0,
              "pilot": pilot, "runs": phase_runs}[ph]())
