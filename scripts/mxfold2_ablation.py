#!/usr/bin/env python3
"""A3: which MXfold2 training choices move the model along its weight-space symmetry, and does a
parameter projection hold it there at no accuracy cost? (Preregistered.)

Plan approved 2026-10-12 (/home/shadi/.claude/plans/wild-kindling-moth.md, section A3), with the
user's three amendments: the projection arm's drift is measured from its POST-projection start
and the one-time initial projection is reported separately per seed; the primary equivalence
test is the paired t-interval over the 3 per-seed differences (df = 2).

THE SYMMETRY. v = +1 on zuker.net.fc_paired.fc.4.bias[0], -2 on
zuker.fc_length.score_helix_length.net.0.weight[0, 2..30] (scripts/mxfold2_gauge.py; |v|^2 =
117). Moving the weights along v moves every output table along the output gauge h, so the
hinge_mix loss is flat along v except on the long-helix breaker sequences. v.w is the
coordinate an optimiser can drift along without the loss noticing.

ROWS OF TABLE 3 (PRED["rows"]): SGD, Adam, Adam + gauge penalty and Adam w = 0.125 are existing
arms; AdamW (weight decay 0.01) and Adam + projection are NEW, three seeds each (20261006-08),
ten epochs, TR0 uncapped, through scripts/mxfold2_ablation_train.py (the gauge hooks at lambda
0, which only log, plus v.w logged around every optimiser step). The projection is
scripts/mxfold2_vproject.project_v_ (w <- w - ((v.w)/117) v, float64 shift, float32 write),
applied once at initialisation and after every optimiser step, with a live rigorous float32
bound assert (v_bound).

REGISTERED TEST (PRED["registered_test_user"]): d_s = F_proj(s) - F_adam(s) for the three
seeds, both scored here; PASS iff mean(d) +- t(0.975, 2) SD(d)/sqrt(3) lies inside
(-0.005, 0.005). Reported beside it: a two-level paired bootstrap (seeds and the 1,304 shared
TS0 sequences), the sequence-only CI, the 3 differences and the duplicate-run floor.

GUARDS. collect-existing, smoke-gpu and run refuse unless the artifact is tracked by git and
unmodified and predictions == PRED (a relaunch of run after a fault therefore needs the
artifact committed first); every phase after registration also refuses unless this file's own
sha256, the registered constants (the run table, the seeds, the epochs, the band, the
bootstrap, the worker count) and the sha256s of the two untracked training inputs (the TR0
list and the seed study's arch.conf) are the ones recorded at registration, and preregister
refuses unless every pinned code file is itself committed. score and analyse re-check PRED and
every shard sha256, and analyse reports no statistic if a must-hold fails (it records the
failing checks, discards any earlier results and leaves the status at `scored`, so that a
cause fixed outside the artifact can be re-analysed). run also refuses unless it is a session
leader (setsid),
the tree is clean outside runs/, collect-existing passed, the GPU smoke G1 passed and `make
gpu` reports both cards free; each start re-checks the GPU's free memory (>= 7500 MiB), the
disk (>= 5 GB), at most 3 of our processes per GPU, one start per GPU per 120 s and the
RUN_CODE hashes. Children run with OMP/OPENBLAS/MKL = 1 and every inherited MXF_* variable
stripped. A NaN marks a run diverged (not rerun); a crash resumes from the newest checkpoint at
most twice and is flagged; a live bound violation stops the run and voids the projection arm.
A failed or bound-stopped run does NOT block the analysis: the run phase ends in a terminal
status that score and analyse accept (and exits nonzero), because only the analysis can report
what PRED registers for those outcomes -- fewer than 3 differences is NOT PASSED, a bound
violation voids the projection arm while the other rows are still reported.
The run code (RUN_CODE and the installed mxfold2/train.py) must be byte-identical to the
code recorded at registration (and CPU-smoked) for smoke-gpu and run, and at every start.
Per-run files live under runs/<RUN_ID>/shards/<run>/ (gitignored); their sha256s are recorded
when the run ends and re-checked by score and analyse. Score workers are CPU only and use the
evaluator's thread count (2b.threads = 2), NOT 1, because the evaluator launched its own
workers with OMP/OPENBLAS/MKL = that number and makes the thread count part of its bitwise
reproduction contract; without it the reported "our F equals the evaluator's F, exactly" check
would not be like for like. Training children run with threads = 1.

DETAILS THE PLAN LEAVES OPEN, fixed here (the conservative option each time):
  * the CPU smoke precedes registration (the plan's schedule: code, CPU smoke, preregister),
    so preregister requires a passing smoke_cpu.json and embeds it with its sha256; the smoke
    report also carries the sha256s of the run code, of this launcher (which drives the smoke)
    and of the training inputs, and preregister refuses unless all three still match;
  * smoke-gpu and run need make gpu to report BOTH cards free (gpucheck.sh's 8000 MiB);
  * G1 runs the projection on 200 seeded TR0 sequences (as the gauge study's smoke); G2 is
    one full TR0 epoch, seed 20261006, projection off, against the seed study's epoch-1
    checkpoint, reported and not a gate (the plan says "reported");
  * a bound violation is not resumed (it is deterministic) and voids the projection arm; a
    projection run that cannot train leaves the registered test NOT PASSED;
  * with a flagged (resumed) projection run, the flagged-excluded variant uses t(0.975, df = 1);
  * collect-existing is written once only when it PASSES: a failing attempt is kept under
    existing_failed_attempts and may be retried (after the artifact is committed again), since
    a must-hold there can fail on a cause outside the artifact -- an input not yet committed,
    a checkpoint not on disk -- and run reads existing.pass, which a failing attempt never sets;
  * the arm a Table 3 row is tabulated as is tied to the configuration that actually ran, from
    both ends: what the process BUILT (projection-init.json's optimiser, lr, l2_weight,
    projected, planted) and what it was ASKED to do (every recorded attempt's argv and MXF_*
    environment). A row whose process disagrees with its arm fails a must-hold;
  * the gauge hooks' per-epoch continuity is true by construction inside one process, so the
    runner measures v.w itself at the top of every epoch and the launcher checks that chain
    independently (PRED must_hold.continuity states both).

Phases (all under .venv-mxfold2):
  smoke-cpu --out DIR  C1-C5 on 20 seeded TR0 sequences <= 80 nt, CPU, into DIR (never into
                       the run directory); writes DIR/smoke_cpu.json; exit 0 iff all pass
  preregister --smoke-cpu DIR/smoke_cpu.json
                       writes PRED (and the passing CPU smoke, with its sha256) once
  collect-existing     the existing rows' v.w and F, with every file's sha256 and the must-holds
  smoke-gpu            G1 (gate) and G2 (reported); refuses unless make gpu reports both free
  run                  the six new runs (launch with setsid)
  score                CPU: per-sequence tp/fp/fn of the 9 epoch-10 models, sha-recorded shards
  analyse              Table 3, the registered test and everything reported beside it
Commit the artifact after preregister, collect-existing and smoke-gpu (run checks that it is
tracked and unmodified). Run:
  .venv-mxfold2/bin/python scripts/mxfold2_ablation.py --phase smoke-cpu --out SCRATCH
  .venv-mxfold2/bin/python scripts/mxfold2_ablation.py --phase preregister \\
      --smoke-cpu SCRATCH/smoke_cpu.json
  .venv-mxfold2/bin/python scripts/mxfold2_ablation.py --phase collect-existing
  setsid nohup .venv-mxfold2/bin/python scripts/mxfold2_ablation.py --phase smoke-gpu > LOG 2>&1 &
  setsid nohup .venv-mxfold2/bin/python scripts/mxfold2_ablation.py --phase run > LOG 2>&1 &
  .venv-mxfold2/bin/python scripts/mxfold2_ablation.py --phase score
  .venv-mxfold2/bin/python scripts/mxfold2_ablation.py --phase analyse
"""

from __future__ import annotations

import hashlib
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

from scripts.mxfold2_vproject import BOUND_MSG, NONFINITE_MSG  # noqa: E402
from scripts.predecessor import RUNNING_PREFIX  # noqa: E402
from scripts.runmeta import git_provenance  # noqa: E402

RUN_ID = "2026-10-12_mxfold2_ablation"
OUT = R / "runs" / RUN_ID
SHARDS = OUT / "shards"  # gitignored (runs/**/shards/); sha256s recorded in the artifact
DATA = R / ".cache" / "mxfold2-data"
PY_MX = R / ".venv-mxfold2" / "bin" / "python"
RUNNER = R / "scripts" / "mxfold2_ablation_train.py"
PLAIN_RUNNER = R / "scripts" / "mxfold2_timed_train.py"
LIVE = R / "runs" / "2026-10-07_mxfold2_seed_study"
GRUNS = R / "runs" / "2026-10-08_mxfold2_gauge_runs"
EVAL = R / "runs" / "2026-10-08_mxfold2_gauge_eval" / "metrics.json"
TRAIN_LIST = "data/TR0-canonicals.lst"  # relative to DATA, the training cwd

SEEDS = (20261006, 20261007, 20261008)
EPOCHS = 10
WD_NEW = 0.01
LR = 1e-3  # MXfold2's default --lr; every arm's, recorded by the runner and checked
N_SCORED = 1304  # TS0 entries with reference pairs (the evaluator's n_scored), PRED's 1,304
ALLOC_CONF = "expandable_segments:True"
MAX_PER_GPU = 3
SETTLE_S = 120
RESERVE_MIB = 7500
GPUCHECK_MIB = 8000  # gpucheck.sh's THRESHOLD: "make gpu" OK on a card
DISK_MIN_GB = 5.0
MAX_RECOVER = 2
POLL_S = 30
REL_TOL = 1e-12  # float64 v.w recomputed on another device or path (as S1's init check)
BAND = 0.005
BOOT = 10_000
BOOT_SEED = (20261006, 70)
T975_DF2 = 0.95 / math.sqrt(2 * 0.975 * 0.025)  # t(0.975, df = 2), closed form
N_SCORE_WORKERS = 4
SMOKE_CPU = {"n_sequences": 20, "max_len": 80, "seed": 20261006, "epochs": 1}
SMOKE_GPU = {"g1_n_sequences": 200, "g1_seed": 20261006, "g1_pick": (20261006, 71),
             "g2_seed": 20261006}
RUN_CODE = ("scripts/mxfold2_ablation_train.py", "scripts/mxfold2_vproject.py",
            "scripts/mxfold2_gauge_hooks.py", "scripts/mxfold2_gauge.py",
            "scripts/mxfold2_timed_train.py")
RUNNING = f"{RUNNING_PREFIX}: A3 training (6 runs)"
REGISTERED = "PRE-REGISTERED, nothing trained"
DONE = "trained"
# every run reached a terminal state, but one failed or was stopped by a bound violation:
# score and analyse still run (PRED's missing_projection_run and void), the exit code is 1
DONE_PARTIAL = "trained, with runs that failed or were stopped"
SCORED = "scored"
ANALYSED = "analysed"
FINAL = ("complete", "diverged", "bound_violation", "failed")

RUNS = (  # arms balanced across the GPUs, as the plan assigns them
    {"name": "adamw_wd0.01_seed20261006", "arm": "adamw_wd0.01", "seed": 20261006, "gpu": 0,
     "project": False, "l2": WD_NEW},
    {"name": "adamw_wd0.01_seed20261008", "arm": "adamw_wd0.01", "seed": 20261008, "gpu": 0,
     "project": False, "l2": WD_NEW},
    {"name": "proj_seed20261007", "arm": "adam_proj", "seed": 20261007, "gpu": 0,
     "project": True, "l2": 0.0},
    {"name": "proj_seed20261006", "arm": "adam_proj", "seed": 20261006, "gpu": 1,
     "project": True, "l2": 0.0},
    {"name": "proj_seed20261008", "arm": "adam_proj", "seed": 20261008, "gpu": 1,
     "project": True, "l2": 0.0},
    {"name": "adamw_wd0.01_seed20261007", "arm": "adamw_wd0.01", "seed": 20261007, "gpu": 1,
     "project": False, "l2": WD_NEW},
)

PRED = {
    "registered": "2026-10-12, after the CPU smoke (embedded in this artifact) and before "
                  "collect-existing, the GPU smoke or any A3 training; nothing trained or scored",
    "plan": "approved 2026-10-12, /home/shadi/.claude/plans/wild-kindling-moth.md section A3, "
            "with the user's amendments: the projection arm's drift from its POST-projection "
            "start, the one-time initial projection reported separately per seed; the primary "
            "test is the paired t-interval over the 3 per-seed differences (df = 2)",
    "question_user": "which MXfold2 training choices move the model along its weight-space "
                     "symmetry v, and does a parameter projection hold it at no accuracy cost?",
    "rows": {
        "sgd": "existing registered arm, runs/2026-10-08_mxfold2_gauge_runs "
               "sgd_seed2026100{6,7,8}_w1.0: MXfold2's SGD (Nesterov, momentum 0.9), lr 1e-3, "
               "wd 0, w = 1.0, lambda 0; 2 of 3 seeds diverged; reported as 'did not train at "
               "the registered LR', n = 1 of 3",
        "adam": "existing adamw_w1.0 arm of runs/2026-10-07_mxfold2_seed_study (8 seeds, "
                "full_seed*_w1.0): MXfold2's code-default AdamW at weight decay 0, which is "
                "torch-identical to Adam; v.w from the evaluator's S2 traces, F from its "
                "F_per_model; seeds 20261006-08 re-scored here for the registered test",
        "adamw_new": "NEW: --optimizer AdamW --l2-weight 0.01 (torch's decoupled weight decay) "
                     "at w = 1.0, seeds 20261006-08, 10 epochs",
        "adam_penalty": "existing reg arm (8 seeds; AdamW wd 0 plus the null-component penalty "
                        "at the pilot's lambda*); its per-epoch v.w lives only in untracked "
                        "gauge-epoch-*.json files, recorded by collect-existing with every "
                        "file's sha256",
        "adam_projection": "NEW: the Adam row's optimiser and flags (AdamW, wd 0) with the "
                           "parameter projection once at initialisation and after every "
                           "optimiser step, seeds 20261006-08, 10 epochs",
        "adam_w0.125": "existing adamw_w0.125 arm (8 seeds): score-loss weight C1 = 0.125, the "
                       "paper's; the paper's l2 term C2 was never in the released code",
        "common": "every new run: the seed study's arch.conf (byte-identical copy, sha256 "
                  "recorded), TR0-canonicals uncapped, hinge_mix, --score-loss-weight 1.0, lr "
                  "1e-3 (default), --seed s, 10 epochs, a checkpoint every epoch, runner "
                  "scripts/mxfold2_ablation_train.py with the gauge hooks at lambda 0 (they "
                  "only log). The Adam baselines ran the plain timed runner; a new run shares "
                  "its seed's initialisation, data order and dropout draws (G2 checks the "
                  "pairing at epoch 1 with the projection off)",
    },
    "projection_spec": "w <- w - ((v.w)/117) v: c = (b[0] - 2 sum W[0, 2:31]) / 117 in float64 "
                       "on the CPU, b[0] -= c, W[0, 2:31] += 2c, written back in float32 "
                       "(scripts/mxfold2_vproject.project_v_); applied once after the model is "
                       "built, before the first step, and after every optimiser step (a torch "
                       "step post-hook between the raw-step log and the post log); Adam's "
                       "moment estimates are not touched",
    "breaker_note": "v is an exact symmetry of the loss only on steps whose scored structures "
                    "have no helix longer than 30 pairs; 11 of the 13 TR0 sequences whose "
                    "allowed pairs admit a 31+ pair run break it (runs/2026-10-10_mxfold2_vg_"
                    "breaker; |v.g| up to ~0.4 on a step). On those steps the projection "
                    "removes a component of the step that the loss does see, so 'no accuracy "
                    "cost' is a measured outcome, not a tautology; every other change the "
                    "projection makes is float32 rounding of 30 entries",
    "registered_test_user": {
        "statement": "projection keeps TS0 F within +-0.005 of Adam",
        "F": "F = mean per-sequence F1 over the 1,304 TS0 sequences with reference pairs "
             "(scripts/mxfold2_gauge_eval.accuracy, F_mean_per_sequence) of the epoch-10 "
             "checkpoint; both arms scored by this script's score phase (the ts0_f code path)",
        "per_seed": "d_s = F_proj(s) - F_adam(s), s in 20261006, 20261007, 20261008; Adam = "
                    "the seed study's full_seed{s}_w1.0",
        "primary": "the paired t-interval over the 3 per-seed differences: mean(d) +- "
                   "t(0.975, df = 2) * SD(d) / sqrt(3), SD with ddof 1, t(0.975, 2) = 0.95 / "
                   f"sqrt(2 * 0.975 * 0.025) = {T975_DF2!r} (closed form for df = 2)",
        "passes_iff": "the interval lies inside the open interval (-0.005, 0.005)",
        "missing_projection_run": "a projection run that diverges or fails after its "
                                  "recoveries leaves fewer than 3 differences: the test is then "
                                  "NOT PASSED and reported so (a projection that cannot train "
                                  "is a cost). A diverged or failed run does not block the "
                                  "analysis: the run phase ends in a terminal status that score "
                                  "and analyse accept, and the row's n falls. It is not retried",
        "void": "a live bound violation voids the projection arm: no statistic is reported for "
                "it (the registered test is VOID) and the result goes to the user. The other "
                "rows of Table 3 are still reported, with the violating runs named",
    },
    "reported_beside_the_test": {
        "two_level_bootstrap": "paired, 10,000 resamples, rng = np.random.default_rng("
                               "np.random.SeedSequence([20261006, 70])): first S = "
                               "rng.integers(0, 3, (10000, 3)) seed indices, then X = "
                               "rng.integers(0, 1304, (10000, 1304)) sequence indices; "
                               "replicate = mean over its 3 drawn seeds of the mean over its "
                               "1,304 drawn sequences of f_proj(s, x) - f_adam(s, x); 95% "
                               "percentile interval",
        "sequence_only": "the same X draws with all three seeds kept once each; 95% percentile "
                         "interval",
        "per_seed_differences": "the 3 d_s",
        "duplicate_floor": "|F(seed 20261006) - F(its duplicate)| from the evaluator's "
                           "F_per_model for adamw_w1.0 (bitwise-identical checkpoints), "
                           "adamw_w0.125 and reg; the projection arm has no duplicate",
        "scorer_check": "a reported check, not a prediction: this script's F for the Adam seeds "
                        "20261006-08 equals the evaluator's F_per_model[adamw_w1.0_seed{s}] "
                        "(every field, exact; differences recorded). For that comparison to be "
                        "like for like the score workers run CUDA_VISIBLE_DEVICES='' on the CPU "
                        "with the EVALUATOR's thread count, read from its artifact (2b.threads "
                        "= 2), not 1: the evaluator launched its own workers with "
                        "OMP/OPENBLAS/MKL = that number and refuses a stage 2b whose --threads "
                        "differs from 2a's, because the thread count is part of its bitwise "
                        "reproduction contract (mxfold2_gauge_eval.py:801-812). The number used "
                        "is recorded. Training children, by contrast, run with threads = 1",
        "flagged": "if any run was resumed (trajectory_exact false) everything is also "
                   "reported with flagged runs excluded",
    },
    "table3_columns": {
        "n": "trained of started (complete; not diverged or failed)",
        "drift": "mean over trained seeds of |v.w(10) - v.w(0)|; v.w(0) is the initialisation "
                 "(existing Adam arms: the evaluator's rebuild, S2 trace '0'; reg, SGD and the "
                 "new AdamW: gauge-epoch-1 v_dot_w_start); for the PROJECTION arm v.w(0) is the "
                 "POST-projection start, gauge-epoch-1 v_dot_w_start, which the hooks record "
                 "after the initial projection (user's amendment)",
        "initial_projection": "projection arm only, per seed, reported separately: |v.w(0)| "
                              "BEFORE the one-time initial projection (projection-init.json "
                              "pre), with the post value and its bound",
        "mean_abs_step": "S2's definition: mean over trained seeds and epochs e = 1..10 of "
                         "|v.w(e) - v.w(e-1)|, with v.w(0) as in drift",
        "sd_epoch10": "across-seed SD of v.w at epoch 10, ddof 1 (none for n = 1)",
        "F": "TS0 F_mean_per_sequence, mean and SD (ddof 1) over trained seeds; existing arms "
             "from the evaluator's F_per_model, the new arms and the Adam seeds 20261006-08 "
             "from this script's scorer",
        "like_for_like": "existing arms are also shown restricted to seeds 20261006-08",
    },
    "check_not_prediction": "|v.w| <= v_bound(b, W, c) after the initial projection and after "
                            "every projected step, asserted live in the runner; the bound is "
                            "derived in scripts/mxfold2_vproject.py",
    "framing_adamw_user": "the AdamW row is reported as measured; there is no claim that weight "
                          "decay fixes or relocates the gauge (the 2026-10-12 ruling)",
    "prior_claude": "projection drift and per-epoch steps at the bound's scale (~1e-6 or below); "
                    "the AdamW row's drift and epoch-10 SD of the same order as Adam's (the "
                    "decay contracts v.w by roughly 10% per epoch while the preconditioned "
                    "drift dominates); the test passes, with per-seed |d| of order the Adam "
                    "arm's seed SD of F (0.0008), SD(d) ~ 0.001 and a t half-width ~ 0.0025. "
                    "Uncertain: with df = 2 an SD(d) above ~0.002 alone fails it",
    "must_hold": {
        "registration": "collect-existing, smoke-gpu and run refuse unless the artifact is "
                        "tracked and unmodified and predictions == PRED; score and analyse "
                        "check PRED and every shard sha256; smoke-gpu and run refuse unless "
                        "the RUN_CODE sha256s (and the installed mxfold2/train.py) equal those "
                        "recorded at registration; run also needs a session leader, a tree "
                        "clean outside runs/, collect-existing passed, G1 passed and make gpu "
                        "reporting both cards free",
        "inputs_pinned": "preregister refuses unless every pinned code file (RUN_CODE and this "
                         "launcher) is tracked and unmodified, and records the sha256 of this "
                         "launcher (analysis_code_sha256), of the registered constants (the run "
                         "table, the seeds, the epochs, the band, the bootstrap, the worker "
                         "count) and of the two untracked training INPUTS: the TR0 list "
                         "data/TR0-canonicals.lst and the seed study's arch.conf. Every later "
                         "phase (collect-existing, smoke-gpu, run, score, analyse) refuses "
                         "unless all three still equal what was registered, and each start "
                         "re-checks the inputs as it re-checks the code",
        "config_matches_arm": "every new run's logged configuration is the arm it is tabulated "
                              "as: projection-init.json records projected, planted, the "
                              "optimiser class, lr and l2_weight as the process built them "
                              "(AdamW, lr 1e-3, l2 0.01 for the AdamW arm and 0 for the "
                              "projection arm, planted null), and every recorded attempt's "
                              "argv and environment carry that arm's flags (--seed, --epochs "
                              "10, hinge_mix, --score-loss-weight 1.0, the TR0 list, "
                              "MXF_PROJECT 0/1, MXF_GAUGE_LAMBDA 0, no MXF_PROJECT_PLANTED, "
                              "threads 1, the device pinned; --optimizer AdamW --l2-weight 0.01 "
                              "on the AdamW arm and neither flag on the projection arm)",
        "raw_step_nonzero": "every epoch of every new run took at least one optimiser step and "
                            "the raw step moved v.w (sum |raw - pre| > 0): a dead optimiser "
                            "would satisfy every bound",
        "pairing_init": "every new run's unprojected initial v.w (projection-init.json pre) "
                        "equals the evaluator's rebuild init_vdot(seed) within 1e-12 * max(1, "
                        "|x|)",
        "hooks_only_read": "the logged v.w does not move between steps within an epoch "
                           "(pre[t+1] == post[t] exactly); for the AdamW runs post == raw "
                           "exactly; each epoch's first pre equals its v_dot_w_start and its "
                           "last post its v_dot_w within 1e-12 * max(1, |x|); epoch 1's "
                           "v_dot_w_start equals projection-init.json's post (projection) or "
                           "pre (AdamW) exactly",
        "continuity": "every run, new or collected: gauge-epoch-e v_dot_w_start == "
                      "gauge-epoch-(e-1) v_dot_w exactly (e = 2..10), and v_dot_w(10) equals "
                      "the epoch-10 checkpoint's v.w (evaluator's ckpt_vdot) within 1e-12 * "
                      "max(1, |x|). Stated exactly: within ONE process the first of those is "
                      "true by construction, because the gauge hooks set the next epoch's "
                      "v_dot_w_start from the epoch they have just recorded "
                      "(mxfold2_gauge_hooks.py:206); it bites across a resume, where the start "
                      "is measured on the loaded checkpoint. The substantive per-epoch check "
                      "is the NEW runs' own measurement: scripts/mxfold2_ablation_train.py "
                      "reads v.w from the live parameters at the top of every epoch "
                      "(vw-epoch-k.npz 'start'), and that value must equal the previous "
                      "epoch's last logged post exactly (the initial pre or post for epoch 1) "
                      "and the hooks' recorded v_dot_w_start exactly",
        "score_inputs": "score refuses unless the evaluator artifact is sha256-identical to "
                        "the one collect-existing read, and records the sha256 of both "
                        "artifacts it reads as inputs (the evaluator's, for the thread count "
                        "and the F compared against, and the seed study's, for the evaluation "
                        "set); analyse refuses unless both are still identical",
        "adam_checkpoints": "the epoch-10 checkpoint this script scores for Adam seeds "
                            "20261006-08 is sha256-identical to the one collect-existing traced "
                            "for the Adam row's v.w, so the row's F and its drift come from the "
                            "same file",
        "bound_logged": "projection runs: every logged post satisfies |post| <= its logged "
                        "bound, and the initial projection's post its bound (re-checked from "
                        "the logs)",
        "existing": "the evaluator artifact AND the gauge-runs artifact "
                    "(runs/2026-10-08_mxfold2_gauge_runs/metrics.json, which supplies the reg "
                    "and SGD rows' status, trajectory_exact and attempts) are tracked and "
                    "unmodified; recomputing S2's SD at "
                    "epoch 10 and mean |step| from its recorded traces and every arm's F mean "
                    "and SD from its F_per_model reproduces the recorded values exactly; the "
                    "Adam traces equal ckpt_vdot of every checkpoint on disk and init_vdot(seed) "
                    "exactly; every reg run is recorded complete with trajectory_exact true and "
                    "has 10 finite gauge records starting at init_vdot(seed) within 1e-12 * "
                    "max(1, |x|); the complete SGD run's records equal the evaluator's S1 trace "
                    "exactly",
        "shards": "every per-run and score shard file is sha256-recorded when written and "
                  "re-checked by score and analyse",
        "scored_set": "every scored TS0 sequence matches the evaluation set's sha256 (asserted "
                      "in the worker; a mismatch exits nonzero and is recorded as "
                      "score_failed); the sequences with reference pairs are the same set for "
                      "every model AND number exactly 1,304, which is also the evaluator's "
                      "n_scored for the Adam models the test compares against",
    },
    "analysis_stopped": "if a must-hold fails, analyse reports no statistic and leaves none in "
                        "the artifact: it records the failing checks under analysis_stopped, "
                        "discards any earlier results and leaves the status at 'scored', so "
                        "that a cause fixed outside the artifact can be re-analysed. Nothing "
                        "about a failing must-hold is reported as a result",
    "void_and_resume": {
        "nan": "a NaN/inf loss or v.w (stdout 'Loss: nan', a non-finite gauge or vw record, or "
               "the runner's NON-FINITE exit) marks a run diverged; it is not rerun without "
               "asking",
        "crash": "any other non-zero exit resumes from the newest loadable checkpoint (the "
                 "gauge hooks restore the RNG sidecar), at most 2 recoveries per run; a resumed "
                 "GPU run is flagged trajectory_exact = false. A run whose recoveries are "
                 "exhausted is 'failed': terminal, not retried, and reported with the row's n",
        "bound_violation": "stops that run (status bound_violation), is not resumed and voids "
                           "the projection arm",
        "terminal_statuses": "the run phase ends with every run in one of complete, diverged, "
                             "failed, bound_violation. If any is failed or bound_violation the "
                             "artifact's status is 'trained, with runs that failed or were "
                             "stopped' (and the phase exits nonzero), which score and analyse "
                             "accept: only the analysis can report what PRED registers for "
                             "those outcomes. The run phase is not relaunched from it",
        "driver_death": "status RUNNING with no live process of this run is a fault "
                        "(scripts/predecessor.py rule), recorded under failed_attempts; after "
                        "the artifact is committed run may be relaunched and resumes unfinished "
                        "runs, the recovery count carried over",
    },
    "smoke_user": {
        "cpu": "C1: the new runner with projection off is bit-identical to the plain timed "
               "runner (wd 0); C2: the same at --l2-weight 0.01; C3: projection on: every "
               "logged post within its bound, Adam's raw v.w step nonzero, checkpoint differs "
               "from C1's; C4 (must fail): a planted skip of the projection exits nonzero with "
               "the bound message; C5: tests/test_mxfold2_vproject.py passes. 20 seeded TR0 "
               "sequences <= 80 nt, seed 20261006, 1 epoch, CPU",
        "gpu": "G1 (must pass before run): two concurrent 1-epoch runs with the projection on, "
               "one per GPU, seed 20261006, 200 seeded TR0 sequences: checkpoints bitwise "
               "identical, v.w logs identical, every logged post within its bound. G2 "
               "(reported): the new runner with the projection off, seed 20261006, full TR0, "
               "1 epoch, GPU 0, against the seed study's full_seed20261006_w1.0 epoch-1 "
               "checkpoint (compare_checkpoints)",
    },
    "seeds": list(SEEDS),
}


# ----------------------------------------------------------------------------------
# artifact
# ----------------------------------------------------------------------------------
def write(out):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    os.replace(tmp, OUT / "metrics.json")


def read_out():
    p = OUT / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {"run_id": RUN_ID}


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _git_clean(path) -> dict:
    if not pathlib.Path(path).resolve().is_relative_to(R):  # scratch tests: never "committed"
        return {"path": str(path), "tracked": False, "unmodified": False}
    rel = str(pathlib.Path(path).resolve().relative_to(R))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=R,
                             capture_output=True).returncode == 0
    unmodified = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=R,
                                capture_output=True).returncode == 0
    return {"path": rel, "tracked": tracked, "unmodified": unmodified}


def registered_constants() -> dict:
    """Everything the registration pins besides PRED's prose and the code sha256s: the run
    table, the seeds, the epochs, the test's band and bootstrap, the scoring width. Recorded
    at registration and re-checked by every later phase, so that editing a constant voids the
    registration instead of silently changing a registered definition."""
    return {"runs_planned": [dict(r) for r in RUNS], "seeds": list(SEEDS), "epochs": EPOCHS,
            "wd_new": WD_NEW, "lr": LR, "n_scored": N_SCORED, "band": BAND, "boot": BOOT,
            "boot_seed": list(BOOT_SEED), "t975_df2": T975_DF2, "rel_tol": REL_TOL,
            "n_score_workers": N_SCORE_WORKERS, "max_per_gpu": MAX_PER_GPU,
            "reserve_mib": RESERVE_MIB, "gpucheck_mib": GPUCHECK_MIB,
            "max_recover": MAX_RECOVER, "run_code": list(RUN_CODE),
            "smoke_cpu": {k: list(v) if isinstance(v, tuple) else v
                          for k, v in SMOKE_CPU.items()},
            "smoke_gpu": {k: list(v) if isinstance(v, tuple) else v
                          for k, v in SMOKE_GPU.items()}}


def data_hashes() -> dict:
    """The two untracked INPUTS every training run reads: the TR0 list and the seed study's
    arch.conf. Neither is in git (.cache/ is ignored, the seed study's arch.conf is not
    committed), so only a sha256 recorded at registration pins them."""
    return {TRAIN_LIST: _sha(DATA / TRAIN_LIST), "arch.conf": _sha(LIVE / "arch.conf")}


def check_registration(out, require_committed: bool = True) -> dict:
    """Refuse unless the artifact's PRED is this PRED, the analysis code, the registered
    constants and the training inputs are the ones registered, and (before and at the run) the
    artifact is tracked by git and unmodified. score and analyse, which follow the run, do not
    require the artifact committed (as scripts/typeb_heldout.analyse does not) but check
    everything else, plus the shard sha256s."""
    g = _git_clean(OUT / "metrics.json")
    if require_committed and not (g["tracked"] and g["unmodified"]):
        raise SystemExit(f"the artifact is not committed and unmodified ({g}); commit it first")
    if out.get("predictions") != json.loads(json.dumps(PRED)):
        raise SystemExit("PRED changed since registration")
    reg_sha = out.get("analysis_code_sha256")
    if reg_sha is not None and reg_sha != _sha(__file__):
        raise SystemExit(f"{_path(__file__)} changed since registration (it carries the "
                         "registered constants, the run table and the whole analysis); the "
                         "registration is void")
    reg_const = out.get("registered_constants")
    if reg_const is not None and reg_const != json.loads(json.dumps(registered_constants())):
        raise SystemExit("the registered constants changed since registration")
    reg_data = out.get("data_sha256_at_registration")
    if reg_data is not None and reg_data != data_hashes():
        raise SystemExit(f"a training input changed since registration: registered {reg_data}, "
                         f"now {data_hashes()}")
    return {"artifact_sha256": _sha(OUT / "metrics.json"), **g}


def _rel(x: float) -> float:
    return REL_TOL * max(1.0, abs(x))


def _path(p) -> str:
    """A path relative to the repository when it lies inside it (scratch tests: absolute)."""
    p = pathlib.Path(p).resolve()
    return str(p.relative_to(R)) if p.is_relative_to(R) else str(p)


# ----------------------------------------------------------------------------------
# training processes
# ----------------------------------------------------------------------------------
def mx_train_file() -> pathlib.Path:
    import importlib.util
    return pathlib.Path(importlib.util.find_spec("mxfold2.train").origin)


def code_hashes() -> dict:
    h = {f: _sha(R / f) for f in RUN_CODE}
    h["mxfold2/train.py (installed)"] = _sha(mx_train_file())
    return h


def own_arch(base: pathlib.Path) -> pathlib.Path:
    """A byte-identical copy of the seed study's arch.conf in our own directory."""
    arch = base / "arch.conf"
    if not arch.exists():
        base.mkdir(parents=True, exist_ok=True)
        arch.write_bytes((LIVE / "arch.conf").read_bytes())
    assert _sha(arch) == _sha(LIVE / "arch.conf"), "arch.conf differs from the seed study's"
    return arch


def train_cmd(runner, arch, d, epochs, seed, gpu_arg, l2, train_list, resume=None):
    """The seed study's command (no --optimizer: code-default AdamW, wd 0) unless l2 > 0."""
    cmd = [str(PY_MX), str(runner), "train", f"@{arch}", "--param", str(d / "param.pth"),
           "--save-config", str(d / "model.conf"), "--log-dir", str(d / "checkpoints"),
           "--epochs", str(epochs), "--gpu", str(gpu_arg), "--seed", str(seed),
           "--loss-func", "hinge_mix", "--score-loss-weight", "1.0", "--disable-progress-bar"]
    if l2:
        cmd += ["--optimizer", "AdamW", "--l2-weight", str(l2)]
    if resume is not None:
        cmd += ["--resume", str(resume)]
    return cmd + [str(train_list)]


def train_env(d, cuda, timing, project=None, planted=None):
    """OMP/OPENBLAS/MKL = 1; every inherited MXF_* stripped; project None = plain runner."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MXF_")}
    env.update(CUDA_VISIBLE_DEVICES=cuda, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTORCH_CUDA_ALLOC_CONF=ALLOC_CONF,
               MXF_TIMING_OUT=str(d / timing))
    if project is not None:
        env.update(MXF_GAUGE_OUT=str(d / "gauge"), MXF_GAUGE_LAMBDA="0",
                   MXF_GAUGE_CROSSCHECK="0", MXF_VW_OUT=str(d / "gauge"),
                   MXF_PROJECT="1" if project else "0")
    if planted is not None:
        env["MXF_PROJECT_PLANTED"] = planted
    return env


def epoch_of(ck) -> int:
    return int(pathlib.Path(ck).name.split("-")[1])


def usable_checkpoint(d: pathlib.Path):
    """The newest checkpoint that loads (a truncated one is skipped), or None."""
    import torch
    for ck in sorted(d.glob("checkpoints/epoch-*"), key=epoch_of, reverse=True):
        try:
            torch.load(ck, map_location="cpu", weights_only=True)
            return ck
        except Exception:  # noqa: BLE001
            continue
    return None


def non_finite(d: pathlib.Path) -> bool:
    """A NaN/inf loss or v.w anywhere in the run's logs (stdout, gauge and vw records)."""
    pat = re.compile(r"loss[:=]\s*(nan|inf|-inf)", re.IGNORECASE)
    for f in d.glob("stdout.attempt*.log"):
        t = f.read_text(errors="ignore")
        if pat.search(t) or NONFINITE_MSG in t:
            return True
    for f in (d / "gauge").glob("gauge-epoch-*.json"):
        try:
            e = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for k in ("mean_hinge", "mean_R", "v_dot_w"):
            v = e.get(k)
            if isinstance(v, float) and not math.isfinite(v):
                return True
    for f in (d / "gauge").glob("vw-epoch-*.npz"):
        try:
            z = np.load(f)
            if not all(np.isfinite(z[k]).all() for k in z.files):
                return True
        except (OSError, ValueError):
            continue
    return False


def gpus_free() -> dict:
    from scripts.mxfold2_seed_study import gpus
    try:
        return {x["index"]: x["free_mib"] for x in gpus()}
    except Exception as e:  # noqa: BLE001 -- a failed query closes the gate, it does not crash
        return {"error": repr(e)}


def disk_free_gb() -> float:
    st = os.statvfs(R)
    return st.f_bavail * st.f_frsize / 2**30


def make_gpu_both() -> tuple[bool, dict]:
    """CLAUDE.md's gate, required on BOTH cards: make gpu exits 0, says OK, never BUSY, and
    reports >= GPUCHECK_MIB free on GPU 0 and GPU 1."""
    mg = subprocess.run(["make", "gpu"], cwd=R, capture_output=True, text=True)
    free = {int(i): int(f) for i, f in re.findall(r"gpu (\d+) free MiB: (\d+)", mg.stdout)}
    ok = (mg.returncode == 0 and "BUSY" not in mg.stdout and "OK to launch" in mg.stdout
          and sorted(free) == [0, 1] and all(v >= GPUCHECK_MIB for v in free.values()))
    return ok, {"make_gpu_exit": mg.returncode, "free_mib": free, "tail": mg.stdout[-600:],
                "checked": time.strftime("%Y-%m-%d %H:%M:%S")}


def our_processes() -> list:
    """Live python processes (not us or our ancestors) whose command line names this run
    directory: a training child or a driver of this run."""
    from scripts.predecessor import ancestors
    skip, mark, found = ancestors(), str(OUT).encode(), []
    for d in pathlib.Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in skip:
            continue
        try:
            raw = (d / "cmdline").read_bytes()
        except OSError:
            continue
        argv = raw.split(b"\0")
        if argv and b"python" in pathlib.Path(argv[0].decode(errors="ignore")).name.encode() \
                and mark in raw:
            found.append(int(d.name))
    return sorted(found)


def live_driver(out) -> list:
    """The recorded driver pid, if a process of THIS script is still running under it.

    our_processes() matches command lines that name the run directory, which a driver's does
    not: `--phase run` names no path. A driver that died before starting any child would
    otherwise look like a fault. Any mismatch returns nothing, so this only ever adds a
    reason to refuse."""
    pid = (out.get("run") or {}).get("driver_pid")
    if not isinstance(pid, int) or pid == os.getpid():
        return []
    try:
        raw = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [pid] if pathlib.Path(__file__).name.encode() in raw else []


def start(r, base, arch, epochs, train_list, resume_from, hashes):
    d = base / r["name"]
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    att = len(r["attempts"])
    env = train_env(d, str(r["gpu"]), f"timing.attempt{att}.json", project=r["project"])
    cmd = train_cmd(RUNNER, arch, d, epochs, r["seed"], 0, r["l2"], train_list, resume_from)
    log = (d / f"stdout.attempt{att}.log").open("w")
    r["attempts"].append({"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "resume_from": str(resume_from) if resume_from else None,
                          "code_sha256": hashes, "cmd": cmd,
                          "env": {k: env[k] for k in sorted(env) if k.startswith("MXF_")
                                  or k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                           "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                           "PYTORCH_CUDA_ALLOC_CONF")}})
    return subprocess.Popen(cmd, cwd=DATA, env=env, stdout=log, stderr=subprocess.STDOUT)


def _queue_recovery(r, d, epochs):
    """Count a recovery and set where it resumes from (fresh if no checkpoint loads)."""
    r["recoveries"] += 1
    ck = usable_checkpoint(d)
    if ck is not None:
        r["resume_next"] = str(ck)
        if epoch_of(ck) < epochs:
            r["trajectory_exact"] = False
    r["status"] = "queued"


def schedule(runs, out, key, base, train_list, epochs, hashes, max_recover, data=None):
    """Copied from scripts/mxfold2_gauge_study.schedule (not edited there) and adapted: own
    run directories, the bound-violation and divergence classes, recoveries carried over a
    driver restart. <= MAX_PER_GPU processes per GPU, one start per GPU per SETTLE_S, each start
    needs RESERVE_MIB free on its GPU, DISK_MIN_GB of disk, unchanged RUN_CODE hashes and, when
    `data` is given, unchanged training inputs (the TR0 list and arch.conf, neither in git)."""
    arch = own_arch(SHARDS)
    for r in runs:
        r.setdefault("attempts", [])
        r.setdefault("trajectory_exact", True)
        r.setdefault("recoveries", 0)
        if r.get("status") in FINAL:
            continue
        r["status"] = "queued"
        if r["attempts"]:  # a previous driver died with this run unfinished
            if r["recoveries"] >= max_recover:
                r["status"] = "failed"
                continue
            _queue_recovery(r, base / r["name"], epochs)
    queue = {g: [r for r in runs if r["gpu"] == g and r["status"] == "queued"] for g in (0, 1)}
    live, last_start, blocked = [], {0: 0.0, 1: 0.0}, {}
    while any(queue.values()) or live:
        for g in (0, 1):
            if not queue[g] or sum(1 for _, r in live if r["gpu"] == g) >= MAX_PER_GPU:
                blocked.pop(str(g), None)
                continue
            if time.time() - last_start[g] < SETTLE_S:
                continue
            free = gpus_free()
            if free.get(g, 0) < RESERVE_MIB or disk_free_gb() < DISK_MIN_GB:
                blocked[str(g)] = {"since": blocked.get(str(g), {}).get(
                    "since", time.strftime("%Y-%m-%d %H:%M:%S")), "free_mib": free.get(g),
                    "disk_free_gb": disk_free_gb(), "gpu_query": free}
                continue
            if code_hashes() != hashes:
                blocked[str(g)] = {"since": time.strftime("%Y-%m-%d %H:%M:%S"),
                                   "reason": "run code changed since launch; not starting"}
                continue
            if data is not None and data_hashes() != data:
                blocked[str(g)] = {"since": time.strftime("%Y-%m-%d %H:%M:%S"),
                                   "reason": "a training input (TR0 list or arch.conf) changed "
                                             "since registration; not starting"}
                continue
            blocked.pop(str(g), None)
            r = queue[g].pop(0)
            live.append((start(r, base, arch, epochs, train_list, r.pop("resume_next", None),
                               hashes), r))
            r["status"] = "running"
            last_start[g] = time.time()
        for p, r in list(live):
            code = p.poll()
            if code is None:
                continue
            live.remove((p, r))
            r["attempts"][-1].update({"exit_code": code,
                                      "ended": time.strftime("%Y-%m-%d %H:%M:%S")})
            d = base / r["name"]
            log = d / f"stdout.attempt{len(r['attempts']) - 1}.log"
            text = log.read_text(errors="ignore") if log.exists() else ""
            if BOUND_MSG in text:
                r["status"] = "bound_violation"
            elif non_finite(d):
                r["status"] = "diverged"
            elif code == 0 and (d / "checkpoints" / f"epoch-{epochs}").exists():
                r["status"] = "complete"
            elif r["recoveries"] < max_recover:
                _queue_recovery(r, d, epochs)
                queue[r["gpu"]].insert(0, r)
            else:
                r["status"] = "failed"
        out[key].update({"runs": {r["name"]: {k: v for k, v in r.items() if k != "resume_next"}
                                  for r in runs},
                         "blocked": blocked, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "gpus_free_mib": gpus_free(), "disk_free_gb": disk_free_gb()})
        write(out)
        if any(queue.values()) or live:
            time.sleep(POLL_S)
    return runs


def run_files(d: pathlib.Path, epochs: int) -> dict:
    """sha256 of every file a run's analysis reads."""
    fs = sorted(d.glob("gauge/*.json")) + sorted(d.glob("gauge/*.npz")) \
        + sorted(d.glob("stdout.attempt*.log")) + sorted(d.glob("timing.attempt*.json"))
    ck = d / "checkpoints" / f"epoch-{epochs}"
    if ck.exists():
        fs.append(ck)
    return {str(f.relative_to(d)): _sha(f) for f in fs}


# ----------------------------------------------------------------------------------
# smoke-cpu (C1-C5): CPU, into a scratch directory, before registration
# ----------------------------------------------------------------------------------
def pick_short_tr0(n, max_len, seed):
    from scripts.mxfold2_decompose import read_bpseq
    lines = [ln for ln in (DATA / TRAIN_LIST).read_text().split("\n") if ln.strip()]
    short = [ln for ln in lines if len(read_bpseq(DATA / ln.split()[0])[0]) <= max_len]
    rng = np.random.default_rng(seed)
    return [short[i] for i in sorted(rng.choice(len(short), n, replace=False))], len(short)


def cpu_run(base, name, runner, l2=0.0, project=None, planted=None):
    d = base / name
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    env = train_env(d, "", "timing.json", project=project, planted=planted)
    cmd = train_cmd(runner, base / "arch.conf", d, SMOKE_CPU["epochs"], SMOKE_CPU["seed"], -1,
                    l2, base / "train.lst")
    with (d / "stdout.log").open("w") as log:
        code = subprocess.run(cmd, cwd=DATA, env=env, stdout=log,
                              stderr=subprocess.STDOUT).returncode
    return d, code


def vw_checks(d: pathlib.Path, epochs: int, projected: bool) -> dict:
    """The runner's v.w logs against its own gauge records (CPU smoke, G1 and analyse).

    `projected` is the arm this run is TABULATED as. The first thing checked is that the
    process agreed: projection-init.json records what the runner was configured to do, so a
    process launched with the wrong MXF_PROJECT fails here instead of raising KeyError on a
    missing 'post'. Two continuity measurements are kept apart:
      * start_continuity compares the HOOKS' recorded v_dot_w_start with the previous epoch's
        recorded end. Within one process the hooks set the next start from the epoch they just
        recorded (mxfold2_gauge_hooks.py:206), so this can only fail across a resume;
      * start_measured_* compare the runner's OWN measurement at the top of each epoch
        (vw-epoch-k.npz 'start') with the previous epoch's last logged post and with the
        hooks' record. Those are independent of the hooks' state and bite every epoch.
    """
    init = json.loads((d / "gauge" / "projection-init.json").read_text())
    rec = {"projection_init": init, "epochs": {}}
    key = "post" if projected else "pre"
    rec["init_projected_matches_arm"] = bool(init.get("projected") is projected)
    rec["init_not_planted"] = init.get("planted") is None
    rec[f"init_has_{key}"] = key in init
    if not (rec["init_projected_matches_arm"] and rec[f"init_has_{key}"]):
        rec["pass"] = False
        rec["reason"] = (f"projection-init.json records projected={init.get('projected')!r} "
                         f"and has no {key!r}: the process was not configured as the arm it "
                         "is tabulated as")
        return rec
    ok = rec["init_not_planted"]
    prev_post = init[key]        # the hooks' chain, through gauge-epoch-*.json
    prev_meas = init[key]        # the runner's own chain, through the npz
    for e in range(1, epochs + 1):
        gj = json.loads((d / "gauge" / f"gauge-epoch-{e}.json").read_text())
        z = np.load(d / "gauge" / f"vw-epoch-{e}.npz")
        pre, raw, post = z["pre"], z["raw"], z["post"]
        n = len(pre)
        r = {"n_steps": n, "n_backward": gj["n_backward"], "lengths_equal": bool(
            len(raw) == n == len(post) == gj["n_backward"]), "v_dot_w_start": gj[
            "v_dot_w_start"]["v_dot_w"], "v_dot_w": gj["v_dot_w"]}
        start_v = gj["v_dot_w_start"]["v_dot_w"]
        r["start_continuity"] = bool(start_v == prev_post)
        meas = float(z["start"][0]) if "start" in z.files and len(z["start"]) == 1 else None
        r["epoch_start_measured"] = meas
        r["start_measured_equals_record"] = bool(meas is not None and meas == start_v)
        r["start_measured_continuity"] = bool(meas is not None and meas == prev_meas)
        r["no_move_between_steps"] = bool(n == 0 or (np.array_equal(pre[1:], post[:-1])
                                                     and abs(pre[0] - start_v) <= _rel(start_v)))
        r["end_matches_record"] = bool(n == 0 or abs(post[-1] - gj["v_dot_w"])
                                       <= _rel(gj["v_dot_w"]))
        r["sum_abs_raw_minus_pre"] = float(np.abs(raw - pre).sum())
        # a dead optimiser would satisfy every bound: the raw step must move v.w
        r["raw_step_nonzero"] = bool(n > 0 and r["sum_abs_raw_minus_pre"] > 0)
        if projected:
            c, bd = z["c"], z["bound"]
            r["bound_lengths_equal"] = bool(len(c) == len(bd) == n)
            r["max_abs_post_over_bound"] = float((np.abs(post) / bd).max()) if n else 0.0
            r["all_post_within_bound"] = bool(n == 0 or (np.abs(post) <= bd).all())
            ok &= r["bound_lengths_equal"] and r["all_post_within_bound"]
        else:
            r["post_equals_raw"] = bool(np.array_equal(post, raw))
            ok &= r["post_equals_raw"]
        ok &= (r["lengths_equal"] and r["start_continuity"] and r["no_move_between_steps"]
               and r["end_matches_record"] and r["raw_step_nonzero"]
               and r["start_measured_equals_record"] and r["start_measured_continuity"])
        rec["epochs"][str(e)] = r
        prev_post = gj["v_dot_w"]
        prev_meas = float(post[-1]) if n else prev_meas
    if projected:
        rec["init_post_within_bound"] = bool(abs(init["post"]) <= init.get("bound", -1.0))
        ok &= rec["init_post_within_bound"]
    rec["pass"] = bool(ok)
    return rec


def smoke_cpu(dst) -> int:
    from scripts.mxfold2_gauge_eval import init_vdot
    from scripts.mxfold2_seed_study import compare_checkpoints
    base = pathlib.Path(dst).resolve()
    assert not str(base).startswith(str(OUT.resolve())), "smoke-cpu never writes the run dir"
    base.mkdir(parents=True, exist_ok=True)
    c = SMOKE_CPU
    pick, n_short = pick_short_tr0(c["n_sequences"], c["max_len"], c["seed"])
    (base / "train.lst").write_text("\n".join(pick) + "\n")
    own_arch(base)
    t0 = time.time()
    rep = {"config": c, "sequences": pick, "n_short_candidates": n_short,
           "train_list_sha256": _sha(base / "train.lst"), "code_sha256": code_hashes(),
           "analysis_code_sha256": _sha(__file__), "data_sha256": data_hashes(),
           "provenance": git_provenance(R)}
    p1, cp1 = cpu_run(base, "c1_plain", PLAIN_RUNNER)
    n1, cn1 = cpu_run(base, "c1_new_proj_off", RUNNER, project=False)
    p2, cp2 = cpu_run(base, "c2_plain_wd0.01", PLAIN_RUNNER, l2=WD_NEW)
    n2, cn2 = cpu_run(base, "c2_new_proj_off_wd0.01", RUNNER, l2=WD_NEW, project=False)
    n3, cn3 = cpu_run(base, "c3_new_proj_on", RUNNER, project=True)
    n4, cn4 = cpu_run(base, "c4_planted_skip", RUNNER, project=True, planted="skip")
    rep["exit_codes"] = {"c1_plain": cp1, "c1_new": cn1, "c2_plain": cp2, "c2_new": cn2,
                         "c3": cn3, "c4_planted": cn4}
    ck = lambda d: d / "checkpoints" / "epoch-1"  # noqa: E731
    ok_runs = cp1 == cn1 == cp2 == cn2 == cn3 == 0
    if ok_runs:
        rep["C1"] = {"compare": compare_checkpoints(ck(p1), ck(n1)),
                     "vw": vw_checks(n1, 1, False)}
        rep["C1"]["pass"] = bool(rep["C1"]["compare"]["bitwise_identical"]
                                 and rep["C1"]["vw"]["pass"])
        rep["C2"] = {"compare": compare_checkpoints(ck(p2), ck(n2)),
                     "vw": vw_checks(n2, 1, False)}
        rep["C2"]["pass"] = bool(rep["C2"]["compare"]["bitwise_identical"]
                                 and rep["C2"]["vw"]["pass"])
        vw3 = vw_checks(n3, 1, True)
        cmp3 = compare_checkpoints(ck(n1), ck(n3))
        init3 = vw3["projection_init"]
        v0 = init_vdot(c["seed"])
        rep["C3"] = {"vw": vw3, "vs_C1": cmp3,
                     "sum_abs_raw_minus_pre": vw3["epochs"]["1"]["sum_abs_raw_minus_pre"],
                     "init_pre": init3["pre"], "init_vdot_rebuild": v0,
                     "init_pre_matches_rebuild": bool(abs(init3["pre"] - v0) <= _rel(v0))}
        rep["C3"]["pass"] = bool(vw3["pass"] and rep["C3"]["sum_abs_raw_minus_pre"] > 0
                                 and not cmp3["bitwise_identical"]
                                 and rep["C3"]["init_pre_matches_rebuild"])
    else:
        for k in ("C1", "C2", "C3"):
            rep[k] = {"pass": False, "reason": "a run exited nonzero"}
    log4 = (n4 / "stdout.log").read_text(errors="ignore")
    rep["C4"] = {"exit_code": cn4, "bound_message": BOUND_MSG in log4,
                 "message": next((ln for ln in log4.splitlines() if BOUND_MSG in ln), None),
                 "pass": bool(cn4 != 0 and BOUND_MSG in log4)}
    ut = subprocess.run([str(PY_MX), "-W", "ignore", "-m", "unittest",
                         "tests.test_mxfold2_vproject", "-v"], cwd=R, capture_output=True,
                        text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    rep["C5"] = {"exit_code": ut.returncode, "tail": ut.stderr[-1500:],
                 "pass": ut.returncode == 0}
    rep["seconds"] = round(time.time() - t0, 1)
    rep["pass"] = all(rep[k]["pass"] for k in ("C1", "C2", "C3", "C4", "C5"))
    (base / "smoke_cpu.json").write_text(json.dumps(rep, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: rep[k]["pass"] for k in ("C1", "C2", "C3", "C4", "C5")}
                     | {"pass": rep["pass"], "report": str(base / "smoke_cpu.json")}, indent=1))
    return 0 if rep["pass"] else 1


# ----------------------------------------------------------------------------------
# preregister
# ----------------------------------------------------------------------------------
def preregister(smoke_report) -> int:
    p = OUT / "metrics.json"
    if p.exists():
        raise SystemExit(f"{_path(p)} exists; a pre-registration is written once")
    if smoke_report is None:
        raise SystemExit("give --smoke-cpu <smoke_cpu.json>: the CPU smoke precedes registration")
    rep = json.loads(pathlib.Path(smoke_report).read_text())
    if rep.get("pass") is not True:
        raise SystemExit("the CPU smoke did not pass; nothing is registered")
    if rep["code_sha256"] != code_hashes():
        raise SystemExit("the run code changed since the CPU smoke; rerun it")
    if rep.get("analysis_code_sha256") != _sha(__file__):
        raise SystemExit(f"{_path(__file__)} changed since the CPU smoke (it drives the "
                         "smoke); rerun it")
    if rep.get("data_sha256") != data_hashes():
        raise SystemExit("a training input changed since the CPU smoke; rerun it")
    # the pinned code must be IN git at registration: a sha256 of an uncommitted file pins
    # nothing a reviewer can recover, and run demands a clean tree anyway
    dirty = [g for g in (_git_clean(R / f) for f in (*RUN_CODE, _path(__file__)))
             if not (g["tracked"] and g["unmodified"])]
    if dirty:
        raise SystemExit(f"commit the pinned code first; not tracked and unmodified: {dirty}")
    write({"run_id": RUN_ID, "status": REGISTERED, "predictions": PRED,
           "runs_planned": [dict(r) for r in RUNS], "epochs": EPOCHS,
           "registered_constants": registered_constants(),
           "smoke_cpu": {"report": rep, "report_sha256": _sha(smoke_report)},
           "code_sha256_at_registration": code_hashes(),
           "analysis_code_sha256": _sha(__file__),
           "data_sha256_at_registration": data_hashes(),
           "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "registration_provenance": git_provenance(R)})
    print(f"wrote {_path(p)}; commit it before any other phase")
    return 0


# ----------------------------------------------------------------------------------
# collect-existing
# ----------------------------------------------------------------------------------
def s2_stats(traces: dict, seeds) -> dict:
    """scripts/mxfold2_gauge_eval.s2_adamw_traces' SD and mean |step|, on {seed: {e: v}}."""
    eps = sorted(set.intersection(*[{int(e) for e in traces[str(s)]} for s in seeds]))
    sd = {str(e): float(np.std([traces[str(s)][str(e)] for s in seeds], ddof=1)) for e in eps}
    step = [abs(traces[str(s)][str(e)] - traces[str(s)][str(e - 1)]) for s in seeds
            for e in eps if e >= 1]
    return {"seed_sd_by_epoch": sd, "mean_abs_epoch_step": float(np.mean(step)) if step else None}


def f_stats(fs) -> dict:
    """The evaluator's core_stats F mean and SD (np.mean, np.std ddof 1, seed order)."""
    return {"F_mean": float(np.mean(fs)), "F_sd": float(np.std(fs, ddof=1))}


def gauge_trace(d: pathlib.Path, seed: int, epochs: int | None = None) -> dict:
    """Per-epoch v.w of a gauge-hooked run, every file's sha256 and the continuity checks."""
    from scripts.mxfold2_gauge_eval import ckpt_vdot, init_vdot
    epochs = EPOCHS if epochs is None else epochs
    files, trace, flags = {}, {}, {}
    prev = None
    cont = True
    for e in range(1, epochs + 1):
        f = d / "gauge" / f"gauge-epoch-{e}.json"
        if not f.exists():
            break
        files[str(f.relative_to(d))] = _sha(f)
        g = json.loads(f.read_text())
        if e == 1:
            trace["0"] = g["v_dot_w_start"]["v_dot_w"]
        elif g["v_dot_w_start"]["v_dot_w"] != prev:
            cont = False
        trace[str(e)] = g["v_dot_w"]
        flags[str(e)] = {"resumed_from_epoch": g.get("resumed_from_epoch"),
                         "rng_restored": g.get("rng_restored")}
        prev = g["v_dot_w"]
    v0 = init_vdot(seed)
    res = {"files_sha256": files, "trace": trace, "epoch_flags": flags,
           "n_epochs": len(trace) - 1 if trace else 0,
           "finite": bool(trace) and all(math.isfinite(v) for v in trace.values()),
           "continuity_exact": cont, "init_vdot_rebuild": v0,
           "start_matches_rebuild": bool(trace) and abs(trace["0"] - v0) <= _rel(v0)}
    ck = d / "checkpoints" / f"epoch-{epochs}"
    if res["n_epochs"] == epochs and ck.exists():
        cv = ckpt_vdot(ck)
        res.update(checkpoint=_path(ck), checkpoint_sha256=_sha(ck),
                   checkpoint_vdot=cv,
                   end_matches_checkpoint=bool(abs(trace[str(epochs)] - cv) <= _rel(cv)))
    else:
        res["end_matches_checkpoint"] = False
    return res


def compute_existing() -> dict:
    """The existing rows (read only). Must-holds in PRED['must_hold']['existing']."""
    from scripts.mxfold2_gauge_eval import ckpt_vdot, init_vdot
    from scripts.mxfold2_seed_study import recorded_seeds
    seeds = list(recorded_seeds())
    ev_git = _git_clean(EVAL)
    ev = json.loads(EVAL.read_text())["2b"]
    res = ev["results"]
    must = {"eval_artifact_committed": ev_git["tracked"] and ev_git["unmodified"],
            "eval_stage_2b_complete": ev["status"] == "complete"}
    out = {"eval_artifact": {**ev_git, "sha256": _sha(EVAL)}, "seeds_all": seeds}
    # Adam arms: S2 traces, recomputed and against every checkpoint on disk
    adam = {}
    for w in ("1.0", "0.125"):
        arm = f"adamw_w{w}"
        s2 = res["S2"][arm]
        tr = s2["traces"]
        again = s2_stats(tr, seeds)
        from_ck, ck_sha = {}, {}
        for s in seeds:
            t = {"0": init_vdot(s)}
            for e in range(1, EPOCHS + 1):
                ck = LIVE / "seeds" / f"full_seed{s}_w{w}" / "checkpoints" / f"epoch-{e}"
                t[str(e)] = ckpt_vdot(ck)
                if e == EPOCHS:
                    ck_sha[str(s)] = _sha(ck)
            from_ck[str(s)] = t
        adam[arm] = {"traces": tr, "recomputed": {"sd_epoch10": again["seed_sd_by_epoch"]["10"],
                                                  "mean_abs_epoch_step":
                                                  again["mean_abs_epoch_step"]},
                     "recorded": {"sd_epoch10": s2["seed_sd_by_epoch"]["10"],
                                  "mean_abs_epoch_step": s2["mean_abs_epoch_step"]},
                     "checkpoint_epoch10_sha256": ck_sha}
        adam[arm]["s2_reproduced_exactly"] = adam[arm]["recomputed"] == adam[arm]["recorded"]
        adam[arm]["traces_match_checkpoints_exactly"] = all(
            from_ck[str(s)][e] == tr[str(s)][e] for s in seeds for e in from_ck[str(s)])
        must[f"{arm}_s2_reproduced"] = adam[arm]["s2_reproduced_exactly"]
        must[f"{arm}_traces_match_checkpoints"] = adam[arm]["traces_match_checkpoints_exactly"]
    out["adam_arms"] = adam
    # F per arm: recomputed from F_per_model exactly as core_stats
    fpm = res["F_per_model"]
    names = {"adamw_w1.0": "adamw_w1.0_seed{s}", "adamw_w0.125": "adamw_w0.125_seed{s}",
             "reg": "reg_seed{s}_w1.0"}
    fa = {}
    for arm, fmt in names.items():
        fs = [fpm[fmt.format(s=s)]["F_mean_per_sequence"] for s in seeds]
        rec = {k: res["main"]["arms"][arm][k] for k in ("F_mean", "F_sd")}
        fa[arm] = {"per_seed": {str(s): f for s, f in zip(seeds, fs, strict=True)},
                   "recomputed": f_stats(fs), "recorded": rec,
                   "reproduced_exactly": f_stats(fs) == rec}
        must[f"F_{arm}_reproduced"] = fa[arm]["reproduced_exactly"]
    out["F_arms"] = fa
    # reg arm and SGD: the gauge runs' records
    meta_p = GRUNS / "metrics.json"
    meta = json.loads(meta_p.read_text())["runs"]["runs"]
    gr_git = _git_clean(meta_p)
    out["gauge_runs_artifact"] = {**gr_git, "sha256": _sha(meta_p)}
    # this artifact supplies the reg and SGD rows' status, trajectory_exact and attempts, so
    # it is an input and must be committed, exactly as the evaluator artifact must
    must["gauge_runs_artifact_committed"] = gr_git["tracked"] and gr_git["unmodified"]
    reg = {}
    for s in seeds:
        n = f"reg_seed{s}_w1.0"
        m = meta[n]
        t = gauge_trace(GRUNS / "seeds" / n, s)
        t.update(status=m["status"], trajectory_exact=m.get("trajectory_exact"),
                 attempts=len(m["attempts"]), F=fpm[n]["F_mean_per_sequence"])
        t["checks_pass"] = bool(m["status"] == "complete" and m.get("trajectory_exact") is True
                                and t["n_epochs"] == EPOCHS and t["finite"]
                                and t["continuity_exact"] and t["start_matches_rebuild"]
                                and t["end_matches_checkpoint"])
        reg[n] = t
        must[f"{n}_checks"] = t["checks_pass"]
    out["reg"] = reg
    sgd = {}
    s1 = res["S1"]["runs"]
    for s in seeds[:3]:
        n = f"sgd_seed{s}_w1.0"
        m = meta[n]
        rec = {"status": m["status"], "trajectory_exact": m.get("trajectory_exact")}
        if m["status"] == "complete":
            t = gauge_trace(GRUNS / "seeds" / n, s)
            t["equals_S1_actual_exactly"] = t["trace"] == s1[n]["actual"]
            t["F"] = fpm[n]["F_mean_per_sequence"]
            t["checks_pass"] = bool(t["n_epochs"] == EPOCHS and t["finite"]
                                    and t["continuity_exact"] and t["start_matches_rebuild"]
                                    and t["end_matches_checkpoint"]
                                    and t["equals_S1_actual_exactly"])
            must[f"{n}_checks"] = t["checks_pass"]
            rec.update(t)
        sgd[n] = rec
    out["sgd"] = sgd
    out["sgd_n_trained"] = sum(1 for r in sgd.values() if r["status"] == "complete")
    must["sgd_one_of_three_trained"] = out["sgd_n_trained"] == 1
    # the duplicate floor of F (evaluator's F_per_model)
    out["duplicate_floor_F"] = {
        arm: {"original": fpm[o]["F_mean_per_sequence"], "duplicate": fpm[d]["F_mean_per_sequence"],
              "abs_diff": abs(fpm[o]["F_mean_per_sequence"] - fpm[d]["F_mean_per_sequence"])}
        for arm, o, d in (("adamw_w1.0", "adamw_w1.0_seed20261006",
                           "adamw_w1.0_seed20261006_dup"),
                          ("adamw_w0.125", "adamw_w0.125_seed20261006",
                           "adamw_w0.125_seed20261006_dup"),
                          ("reg", "reg_seed20261006_w1.0", "reg_seed20261006_w1.0_dup"))}
    out["adam_F_per_model_seeds_06_08"] = {str(s): fpm[f"adamw_w1.0_seed{s}"] for s in SEEDS}
    out["must_hold"] = {k: bool(v) for k, v in must.items()}
    out["pass"] = all(out["must_hold"].values())
    return out


def collect_existing() -> int:
    out = read_out()
    prov = check_registration(out)
    if not out["status"].startswith("PRE-REGISTERED"):
        raise SystemExit(f"collect-existing runs before training; status is {out['status']!r}")
    if out.get("existing", {}).get("pass"):
        raise SystemExit("the existing rows are collected once (a passing record exists)")
    t0 = time.time()
    ex = compute_existing()
    ex.update(seconds=round(time.time() - t0, 1), registration=prov,
              provenance=git_provenance(R))
    if ex["pass"]:
        out["existing"] = ex
    else:  # a must-hold can fail on a cause that is fixed outside this artifact (a checkpoint
        # not on disk, an input not yet committed): keep the failed attempt and allow a retry,
        # but never leave a failing record under "existing", which run reads
        out.pop("existing", None)
        out.setdefault("existing_failed_attempts", []).append(ex)
    write(out)
    print(json.dumps({"must_hold": ex["must_hold"], "pass": ex["pass"]}, indent=1))
    return 0 if ex["pass"] else 1


# ----------------------------------------------------------------------------------
# smoke-gpu (G1 gate, G2 reported)
# ----------------------------------------------------------------------------------
def smoke_gpu() -> int:
    from scripts.mxfold2_seed_study import compare_checkpoints
    out = read_out()
    prov = check_registration(out)
    if out["status"] != REGISTERED:
        raise SystemExit(f"smoke-gpu runs before training; status is {out['status']!r}")
    if code_hashes() != out["code_sha256_at_registration"]:
        raise SystemExit("the run code differs from the code registered (and CPU-smoked)")
    if os.getsid(0) != os.getpid():
        raise SystemExit("launch under setsid (G2 trains a full epoch)")
    earlier = {k: out.pop(k) for k in ("smoke_gpu", "smoke_gpu_g1", "smoke_gpu_g2") if k in out}
    base = SHARDS / "smoke_gpu"
    if earlier or base.exists():  # an earlier smoke: kept, never mixed with this one
        old = base.with_name(f"smoke_gpu.before-{time.strftime('%Y%m%d-%H%M%S')}")
        if base.exists():
            base.rename(old)
        out.setdefault("smoke_gpu_earlier", []).append(
            {"files_moved_to": str(old.relative_to(OUT)) if old.exists() else None, **earlier})
    ok, gate = make_gpu_both()
    out["smoke_gpu"] = {"launch_gate": gate, "registration": prov, "config": SMOKE_GPU,
                        "provenance": git_provenance(R)}
    if not ok:
        out["smoke_gpu"]["status"] = "NOT LAUNCHED: make gpu does not report both cards free"
        write(out)
        return 1
    base.mkdir(parents=True, exist_ok=True)
    lines = [ln for ln in (DATA / TRAIN_LIST).read_text().split("\n") if ln.strip()]
    idx = np.random.default_rng(np.random.SeedSequence(list(SMOKE_GPU["g1_pick"]))).choice(
        len(lines), SMOKE_GPU["g1_n_sequences"], replace=False)
    lst = base / "g1_tr0.lst"
    lst.write_text("\n".join(lines[i] for i in sorted(idx)) + "\n")
    hashes = code_hashes()
    runs = [{"name": f"g1_gpu{g}", "seed": SMOKE_GPU["g1_seed"], "gpu": g, "project": True,
             "l2": 0.0, "list": str(lst)} for g in (0, 1)]
    g2 = {"name": "g2_pairing", "seed": SMOKE_GPU["g2_seed"], "gpu": 0, "project": False,
          "l2": 0.0}
    out["smoke_gpu"].update(status=f"{RUNNING_PREFIX}: smoke-gpu", code_sha256=hashes,
                            data_sha256=data_hashes(), g1_list_sha256=_sha(lst))
    write(out)
    out["smoke_gpu_g1"] = {}
    data = data_hashes()
    schedule(runs, out, "smoke_gpu_g1", base, str(lst), 1, hashes, max_recover=0, data=data)
    out["smoke_gpu_g2"] = {}
    schedule([g2], out, "smoke_gpu_g2", base, TRAIN_LIST, 1, hashes, max_recover=0, data=data)
    a, b = (base / r["name"] / "checkpoints" / "epoch-1" for r in runs)
    res = {"exit_status": {r["name"]: r["status"] for r in runs + [g2]}}
    if all(r["status"] == "complete" for r in runs):
        cmp = compare_checkpoints(a, b)
        va, vb = (vw_checks(base / r["name"], 1, True) for r in runs)
        za, zb = (np.load(base / r["name"] / "gauge" / "vw-epoch-1.npz") for r in runs)
        same_logs = all(np.array_equal(za[k], zb[k]) for k in za.files) and \
            va["projection_init"] == vb["projection_init"]
        res["G1"] = {"compare": cmp, "vw_gpu0": va, "vw_gpu1": vb, "vw_logs_identical": same_logs,
                     "pass": bool(cmp["bitwise_identical"] and va["pass"] and vb["pass"]
                                  and same_logs)}
    else:
        res["G1"] = {"pass": False, "reason": "a G1 run did not complete"}
    ref = LIVE / "seeds" / f"full_seed{SMOKE_GPU['g2_seed']}_w1.0" / "checkpoints" / "epoch-1"
    if g2["status"] == "complete":
        res["G2_reported"] = {"reference": str(ref.relative_to(R)), "reference_sha256": _sha(ref),
                              "compare": compare_checkpoints(base / g2["name"] / "checkpoints" /
                                                             "epoch-1", ref),
                              "vw": vw_checks(base / g2["name"], 1, False)}
    else:
        res["G2_reported"] = {"status": g2["status"]}
    out["smoke_gpu"].update(res, status="G1 passed" if res["G1"]["pass"] else "G1 FAILED",
                            G1_pass=res["G1"]["pass"],
                            files_sha256={r["name"]: run_files(base / r["name"], 1)
                                          for r in runs + [g2]})
    write(out)
    print(json.dumps({"G1_pass": res["G1"]["pass"], "G2_bitwise": res["G2_reported"].get(
        "compare", {}).get("bitwise_identical")}, indent=1))
    return 0 if res["G1"]["pass"] else 1


# ----------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------
def run() -> int:
    import fcntl

    out = read_out()
    st = str(out.get("status", ""))
    if st.startswith(RUNNING_PREFIX):
        live = our_processes() + live_driver(out)
        if live:
            raise SystemExit(f"status {st!r} and processes of this run are live ({live})")
        out.setdefault("failed_attempts", []).append(
            {"status": st, "detected": "driver gone while status RUNNING (fault)",
             "at": time.strftime("%Y-%m-%d %H:%M:%S"), "run": out.get("run")})
        out["status"] = REGISTERED
    elif st.startswith(("FAILED", "STOPPED", "NOT LAUNCHED")):
        out.setdefault("failed_attempts", []).append(
            {"status": st, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "run": out.get("run")})
        out["status"] = REGISTERED
    if out.get("status") != REGISTERED:
        raise SystemExit(f"status is {out.get('status')!r}, not {REGISTERED!r}")
    # registration checks read the COMMITTED artifact: a relaunch after a fault needs a commit
    prov = check_registration(json.loads((OUT / "metrics.json").read_text()))
    if os.getsid(0) != os.getpid():
        raise SystemExit("launch under setsid so that no session can end the driver")
    if git_provenance(R)["git_dirty_excluding_runs"]:
        raise SystemExit("the tree is dirty outside runs/; commit first")
    if not out.get("existing", {}).get("pass"):
        raise SystemExit("collect-existing has not passed")
    if out.get("smoke_gpu", {}).get("G1_pass") is not True:
        raise SystemExit("the GPU smoke G1 has not passed")
    if not (code_hashes() == out["code_sha256_at_registration"]
            == out["smoke_gpu"]["code_sha256"]):
        raise SystemExit("the run code differs from the code registered and GPU-smoked")
    if not (data_hashes() == out["data_sha256_at_registration"]
            == out["smoke_gpu"]["data_sha256"]):
        raise SystemExit("a training input (TR0 list or arch.conf) differs from the one "
                         "registered and GPU-smoked")
    if out["runs_planned"] != json.loads(json.dumps([dict(r) for r in RUNS])):
        raise SystemExit("the run table differs from the one registered")
    if our_processes():
        raise SystemExit(f"processes naming this run directory are live: {our_processes()}")
    SHARDS.mkdir(parents=True, exist_ok=True)
    lock = open(SHARDS / "launcher.lock", "w")  # noqa: SIM115 -- held for the process life
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("another run driver holds the lock") from None
    ok, gate = make_gpu_both()
    prev = out.get("run", {})
    out["run"] = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "driver_pid": os.getpid(),
                  "launch_gate": gate, "registration": prov, "provenance": git_provenance(R),
                  "code_sha256_at_launch": code_hashes(),
                  "data_sha256_at_launch": data_hashes(),
                  "train_list_sha256": _sha(DATA / TRAIN_LIST),
                  "arch_sha256": _sha(LIVE / "arch.conf"),
                  "load_at_start": os.getloadavg(),
                  "runs": prev.get("runs", {})}
    if not ok:
        out["status"] = "NOT LAUNCHED: make gpu does not report both cards free"
        write(out)
        return 1
    hashes = out["run"]["code_sha256_at_launch"]
    runs = [dict(r, **{k: v for k, v in out["run"]["runs"].get(r["name"], {}).items()
                       if k in ("attempts", "trajectory_exact", "recoveries", "status")})
            for r in RUNS]
    out["status"] = RUNNING
    write(out)
    schedule(runs, out, "run", SHARDS, TRAIN_LIST, EPOCHS, hashes, MAX_RECOVER,
             data=out["run"]["data_sha256_at_launch"])
    out["run"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["run"]["shards_sha256"] = {r["name"]: run_files(SHARDS / r["name"], EPOCHS)
                                   for r in runs}
    stat = {r["name"]: r["status"] for r in runs}
    # a failed or bound-stopped run must NOT block score and analyse: PRED registers what each
    # outcome means (missing_projection_run: the test is NOT PASSED; void: the projection arm
    # is void and the result goes to the user), and only the analysis can report that. The
    # status says which case it is and the exit code is still nonzero.
    out["run"]["final_statuses"] = stat
    out["run"]["void"] = any(s == "bound_violation" for s in stat.values())
    out["run"]["partial"] = not all(s in ("complete", "diverged") for s in stat.values())
    if not all(s in FINAL for s in stat.values()):
        out["status"] = f"FAILED: a run is not in a terminal state ({stat})"
    elif out["run"]["partial"]:
        out["status"] = f"{DONE_PARTIAL} ({stat})"
    else:
        out["status"] = DONE
    write(out)
    return 0 if out["status"] == DONE else 1


# ----------------------------------------------------------------------------------
# score (CPU)
# ----------------------------------------------------------------------------------
def score_models(out) -> list:
    m = [{"name": f"adam_seed{s}", "arm": "adam", "seed": s,
          "path": LIVE / "seeds" / f"full_seed{s}_w1.0" / "checkpoints" / f"epoch-{EPOCHS}"}
         for s in SEEDS]
    for r in RUNS:
        if out["run"]["runs"][r["name"]]["status"] == "complete":
            m.append({"name": r["name"], "arm": r["arm"], "seed": r["seed"],
                      "path": SHARDS / r["name"] / "checkpoints" / f"epoch-{EPOCHS}"})
    return m


def score_worker(w, W, dst, models, threads) -> int:
    """Per-sequence tp/fp/fn on TS0 for every model: the code path of
    scripts/mxfold2_wd_ablation.ts0_f (read_bpseq, sha check, make_param, predict, pair sets),
    for sequences x with x % W == w."""
    import torch

    import scripts.mxfold2_gauge_eval as ev
    from scripts.mxfold2_decompose import build_model, predict, read_bpseq

    torch.set_num_threads(threads)
    nets = []
    for m in models:
        net = build_model(ev.LIVE / "arch.conf")
        ck = torch.load(m["path"], map_location="cpu", weights_only=True)
        assert ck["epoch"] == EPOCHS, (m["path"], ck["epoch"])
        net.load_state_dict(ck["model_state_dict"])
        net.eval()
        nets.append(net)
    entries = json.loads((ev.LIVE / "metrics.json").read_text())["evaluation_set"]["entries"]
    xs, tp, fp, fn = [], [], [], []
    t0 = time.time()
    for xi, e in enumerate(entries):
        if xi % W != w:
            continue
        seq, partner = read_bpseq(ev.DATA / e["file"])
        assert hashlib.sha256(seq.encode()).hexdigest() == e["sha256"], e["file"]
        ref = {(i, partner[i]) for i in range(1, len(partner)) if partner[i] > i}
        row = ([], [], [])
        for net in nets:
            with torch.no_grad():
                param = net.make_param([seq])
            _, _, bp = predict(net, seq, param)
            pred = {(i, bp[i]) for i in range(1, len(bp)) if bp[i] > i}
            row[0].append(len(pred & ref))
            row[1].append(len(pred - ref))
            row[2].append(len(ref - pred))
        xs.append(xi)
        tp.append(row[0])
        fp.append(row[1])
        fn.append(row[2])
    dst = pathlib.Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, x=np.array(xs, dtype=np.int64), tp=np.array(tp, dtype=np.int64).reshape(
        len(xs), len(models)), fp=np.array(fp, dtype=np.int64).reshape(len(xs), len(models)),
        fn=np.array(fn, dtype=np.int64).reshape(len(xs), len(models)))
    os.replace(tmp, dst)
    dst.with_suffix(".json").write_text(json.dumps({
        "models": [m["name"] for m in models], "worker": w, "n_workers": W, "threads": threads,
        "scorer_sha256": _sha(__file__), "seconds": round(time.time() - t0, 1)}))
    return 0


def load_score(sd: pathlib.Path, names, n_x, W) -> dict:
    arrs = [np.load(sd / f"w{w}.npz") for w in range(W)]
    for w in range(W):
        meta = json.loads((sd / f"w{w}.json").read_text())
        assert meta["models"] == names, (w, meta["models"])
        assert meta["scorer_sha256"] == _sha(__file__), \
            f"shard w{w} was written by another version of {_path(__file__)}"
    xs = np.concatenate([a["x"] for a in arrs])
    assert np.array_equal(np.sort(xs), np.arange(n_x)), "sequence coverage is not exact"
    order = np.argsort(xs)
    return {k: np.concatenate([a[k] for a in arrs])[order] for k in ("x", "tp", "fp", "fn")}


def score() -> int:
    import scripts.mxfold2_gauge_eval as ev
    out = read_out()
    prov = check_registration(out, require_committed=False)
    st = str(out["status"])
    if not (st in (DONE, SCORED) or st.startswith(DONE_PARTIAL)):
        raise SystemExit(f"status is {st!r}; score follows a finished run")
    for name, files in out["run"]["shards_sha256"].items():
        for f, h in files.items():
            assert _sha(SHARDS / name / f) == h, f"{name}/{f} changed since the run"
    # the two artifacts the scorer reads as INPUTS: the evaluator's (its thread count and the
    # F it is checked against) and the seed study's (the evaluation set and its sha256s)
    assert _sha(EVAL) == out["existing"]["eval_artifact"]["sha256"], \
        f"{_path(EVAL)} changed since collect-existing"
    threads = json.loads(EVAL.read_text())["2b"]["threads"]
    models = score_models(out)
    sd = SHARDS / "score"
    sd.mkdir(parents=True, exist_ok=True)
    for f in sd.glob("w*"):
        f.unlink()
    margs = json.dumps([{**m, "path": str(m["path"])} for m in models])
    tenv = {k: str(threads) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                      "MKL_NUM_THREADS")}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", **tenv)
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, __file__, "--phase", "score-worker", str(w),
                               str(N_SCORE_WORKERS), str(sd / f"w{w}.npz"), margs, str(threads)],
                              env=env) for w in range(N_SCORE_WORKERS)]
    codes = [p.wait() for p in procs]
    if any(codes):
        out["score_failed"] = {"worker_exit": codes, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        write(out)
        return 1
    names = [m["name"] for m in models]
    n_x = len(json.loads((LIVE / "metrics.json").read_text())["evaluation_set"]["entries"])
    cat = load_score(sd, names, n_x, N_SCORE_WORKERS)
    acc = {n: ev.accuracy(cat["tp"][:, i], cat["fp"][:, i], cat["fn"][:, i])
           for i, n in enumerate(names)}
    recorded = out["existing"]["adam_F_per_model_seeds_06_08"]
    check = {str(s): {"ours": acc[f"adam_seed{s}"], "evaluator": recorded[str(s)],
                      "equal_exactly": acc[f"adam_seed{s}"] == recorded[str(s)],
                      "abs_diff_F": abs(acc[f"adam_seed{s}"]["F_mean_per_sequence"]
                                        - recorded[str(s)]["F_mean_per_sequence"])}
             for s in SEEDS}
    # the Adam row's F and its v.w trace must come from the same file on disk: collect-existing
    # sha'd the epoch-10 checkpoint it traced, so compare that with the one just scored
    ck_ex = out["existing"]["adam_arms"]["adamw_w1.0"]["checkpoint_epoch10_sha256"]
    ck_now = {str(s): _sha(LIVE / "seeds" / f"full_seed{s}_w1.0" / "checkpoints"
                           / f"epoch-{EPOCHS}") for s in SEEDS}
    out["score"] = {"models": [{**m, "path": str(m["path"]), "checkpoint_sha256": _sha(m["path"])}
                               for m in models],
                    "adam_checkpoints_sha256": ck_now,
                    "adam_checkpoints_match_existing": all(ck_now[s] == ck_ex[s] for s in ck_now),
                    "input_artifacts_sha256": {_path(EVAL): _sha(EVAL),
                                               _path(LIVE / "metrics.json"):
                                                   _sha(LIVE / "metrics.json")},
                    "workers": N_SCORE_WORKERS, "threads": threads,
                    "threads_note": "the evaluator's thread count (2b.threads), so that the "
                                    "reproduction check is like for like",
                    "shards_sha256": {f.name: _sha(f) for f in sorted(sd.glob("w*"))},
                    "F": acc, "scorer_check_reported": check,
                    "scorer_reproduces_evaluator": all(v["equal_exactly"] for v in check.values()),
                    "seconds": round(time.time() - t0, 1), "registration": prov,
                    "provenance": git_provenance(R)}
    out["status"] = SCORED
    write(out)
    print(json.dumps({"F": {n: a["F_mean_per_sequence"] for n, a in acc.items()},
                      "scorer_reproduces_evaluator": out["score"]["scorer_reproduces_evaluator"]},
                     indent=1))
    return 0


# ----------------------------------------------------------------------------------
# analyse
# ----------------------------------------------------------------------------------
def per_sequence_f(tp, fp, fn):
    """(indices, F) of the sequences with reference pairs, as scripts/mxfold2_gauge_eval.accuracy
    computes its per-sequence list."""
    idx, per = [], []
    for k, (a, b, c) in enumerate(zip(tp, fp, fn, strict=True)):
        if a + c == 0:
            continue
        ppv = a / (a + b) if a + b else 0.0
        sen = a / (a + c)
        idx.append(k)
        per.append(2 * ppv * sen / (ppv + sen) if a else 0.0)
    return np.array(idx), np.array(per)


def t975(df: int) -> float:
    """t(0.975, df) in closed form: df = 2 is the registered test; df = 1 serves only the
    flagged-excluded variant (one projection seed dropped)."""
    if df == 2:
        return T975_DF2
    if df == 1:
        return math.tan(math.pi * 0.475)
    raise ValueError(df)


def t_interval(d) -> dict:
    d = np.asarray(d, dtype=float)
    m, s = float(d.mean()), float(d.std(ddof=1))
    t = t975(len(d) - 1)
    h = t * s / math.sqrt(len(d))
    lo, hi = m - h, m + h
    return {"d": d.tolist(), "n": len(d), "mean": m, "sd": s, "t": t, "df": len(d) - 1,
            "half_width": h, "interval": [lo, hi], "passes": bool(-BAND < lo and hi < BAND)}


def bootstraps(D) -> dict:
    """Two-level paired bootstrap and the sequence-only CI of the mean difference (PRED)."""
    n_s, n_x = D.shape
    rng = np.random.default_rng(np.random.SeedSequence(list(BOOT_SEED)))
    S = rng.integers(0, n_s, (BOOT, n_s))
    X = rng.integers(0, n_x, (BOOT, n_x))
    cs = np.stack([np.bincount(r, minlength=n_s) for r in S]).astype(float)
    M = np.empty((n_s, BOOT))
    for b0 in range(0, BOOT, 1000):  # D @ counts, in blocks
        cx = np.stack([np.bincount(r, minlength=n_x) for r in X[b0:b0 + 1000]]).astype(float)
        M[:, b0:b0 + 1000] = D @ cx.T
    two = (cs * M.T).sum(axis=1) / (n_s * n_x)
    seq = M.mean(axis=0) / n_x
    ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]  # noqa: E731
    return {"two_level_ci95": ci(two), "sequence_only_ci95": ci(seq),
            "point": float(D.mean()), "n_resamples": BOOT, "seed": list(BOOT_SEED)}


def _flag(cmd, flag):
    """The value after `flag` in a recorded argv, or None if the flag is absent."""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def launch_checks(r, attempts) -> dict:
    """Every recorded attempt of this run was launched as the arm it is tabulated as.

    projection-init.json says what the process DID; this says what it was ASKED to do. Both
    are needed: without them a mis-wired l2 or MXF_PROJECT would produce a Table 3 row for an
    arm that never ran, with every other must-hold green."""
    res = {"n_attempts": len(attempts), "per_attempt": []}
    for a in attempts:
        cmd, env = list(a.get("cmd", [])), dict(a.get("env", {}))
        c = {
            "runner": cmd[:3] == [str(PY_MX), str(RUNNER), "train"],
            "seed": _flag(cmd, "--seed") == str(r["seed"]),
            "epochs": _flag(cmd, "--epochs") == str(EPOCHS),
            "loss_func": _flag(cmd, "--loss-func") == "hinge_mix",
            "score_loss_weight": _flag(cmd, "--score-loss-weight") == "1.0",
            "train_list": cmd[-1:] == [TRAIN_LIST],
            "project_env": env.get("MXF_PROJECT") == ("1" if r["project"] else "0"),
            "lambda_zero": env.get("MXF_GAUGE_LAMBDA") == "0",
            "not_planted": "MXF_PROJECT_PLANTED" not in env,
            "threads_one": all(env.get(k) == "1" for k in ("OMP_NUM_THREADS",
                                                           "OPENBLAS_NUM_THREADS",
                                                           "MKL_NUM_THREADS")),
            "device_pinned": env.get("CUDA_VISIBLE_DEVICES") == str(r["gpu"]),
        }
        if r["l2"]:  # the AdamW arm: the decoupled weight decay is explicit
            c["optimizer_flag"] = (_flag(cmd, "--optimizer") == "AdamW"
                                   and _flag(cmd, "--l2-weight") == str(r["l2"]))
        else:        # the Adam and projection arms: the seed study's command, no flags at all
            c["optimizer_flag"] = "--optimizer" not in cmd and "--l2-weight" not in cmd
        res["per_attempt"].append(c)
    res["pass"] = bool(attempts) and all(all(c.values()) for c in res["per_attempt"])
    return res


def new_run_record(r, attempts) -> dict:
    """v.w trace and checks of one new run (the RUNS entry r) from its sha-checked logs."""
    from scripts.mxfold2_gauge_eval import init_vdot
    name, seed, project = r["name"], r["seed"], r["project"]
    d = SHARDS / name
    t = gauge_trace(d, seed)
    vw = vw_checks(d, EPOCHS, project)
    init = vw["projection_init"]
    v0 = init_vdot(seed)
    t["vw_checks"] = vw
    t["init_unprojected"] = init["pre"]
    if project:  # trace "0" is the post-projection start; pairing is checked on init["pre"]
        t["start_matches_rebuild"] = None
    t["pairing_init"] = bool(abs(init["pre"] - v0) <= _rel(v0))
    if project and {"post", "bound", "c"} <= set(init):
        t["initial_projection"] = {"pre_abs": abs(init["pre"]), "post": init["post"],
                                   "bound": init["bound"], "c": init["c"],
                                   "post_within_bound": abs(init["post"]) <= init["bound"]}
    # the configuration the process actually built, against the arm this row is
    t["config"] = {k: init.get(k) for k in ("projected", "planted", "optimizer", "lr",
                                            "l2_weight")}
    t["config_ok"] = bool(init.get("projected") is project and init.get("planted") is None
                          and init.get("optimizer") == "AdamW"
                          and float(init.get("lr", float("nan"))) == LR
                          and float(init.get("l2_weight", float("nan"))) == float(r["l2"]))
    t["launch_checks"] = launch_checks(r, attempts)
    t["checks_pass"] = bool(t["n_epochs"] == EPOCHS and t["finite"] and t["continuity_exact"]
                            and t["end_matches_checkpoint"] and vw["pass"] and t["pairing_init"]
                            and (project or t["start_matches_rebuild"])
                            and t["config_ok"] and t["launch_checks"]["pass"])
    return t


def row_stats(per_seed: dict, seeds=None) -> dict:
    """Table 3 columns from {seed: {"trace": {e: v}, "F": f}} (trained seeds only)."""
    ss = [s for s in (seeds or sorted(per_seed, key=int)) if s in per_seed]
    if not ss:
        return {"n": 0}
    tr = {s: per_seed[s]["trace"] for s in ss}
    drift = [abs(tr[s][str(EPOCHS)] - tr[s]["0"]) for s in ss]
    st = s2_stats(tr, ss) if len(ss) >= 2 else {
        "mean_abs_epoch_step": float(np.mean([abs(tr[ss[0]][str(e)] - tr[ss[0]][str(e - 1)])
                                              for e in range(1, EPOCHS + 1)])),
        "seed_sd_by_epoch": {}}
    fs = [per_seed[s]["F"] for s in ss]
    return {"n": len(ss), "seeds": [int(s) for s in ss], "drift": float(np.mean(drift)),
            "drift_per_seed": dict(zip(ss, drift, strict=True)),
            "mean_abs_step": st["mean_abs_epoch_step"],
            "sd_epoch10": st["seed_sd_by_epoch"].get(str(EPOCHS)),
            "F_mean": float(np.mean(fs)), "F_sd": float(np.std(fs, ddof=1)) if len(fs) > 1
            else None, "F_per_seed": dict(zip(ss, fs, strict=True))}


def analyse() -> int:
    out = read_out()
    prov = check_registration(out, require_committed=False)
    if out["status"] not in (SCORED, ANALYSED):
        raise SystemExit(f"status is {out['status']!r}; analyse follows score")
    for name, files in out["run"]["shards_sha256"].items():
        for f, h in files.items():
            assert _sha(SHARDS / name / f) == h, f"{name}/{f} changed since the run"
    sc = out["score"]
    for f, h in sc["shards_sha256"].items():
        assert _sha(SHARDS / "score" / f) == h, f"score shard {f} changed"
    for f, h in sc["input_artifacts_sha256"].items():
        assert _sha(R / f) == h, f"{f} changed since score"
    names = [m["name"] for m in sc["models"]]
    n_x = len(json.loads((LIVE / "metrics.json").read_text())["evaluation_set"]["entries"])
    cat = load_score(SHARDS / "score", names, n_x, sc["workers"])
    ex = out["existing"]
    runs = out["run"]["runs"]
    checks = {}
    # per-sequence F (registered F and the bootstrap)
    pf, idx0 = {}, None
    for i, n in enumerate(names):
        idx, per = per_sequence_f(cat["tp"][:, i], cat["fp"][:, i], cat["fn"][:, i])
        assert float(np.mean(per)) == sc["F"][n]["F_mean_per_sequence"], n
        idx0 = idx if idx0 is None else idx0
        checks.setdefault("scored_set_identical", True)
        checks["scored_set_identical"] &= bool(np.array_equal(idx, idx0))
        pf[n] = per
    # the scored set is also anchored: the registered F is a mean over 1,304 sequences, which
    # is what the evaluator scored for the Adam models it is compared with
    checks["scored_set_count"] = bool(idx0 is not None and len(idx0) == N_SCORED)
    checks["evaluator_scored_count"] = all(
        ex["adam_F_per_model_seeds_06_08"][str(s)]["n_scored"] == N_SCORED for s in SEEDS)
    checks["adam_checkpoints_match_existing"] = bool(sc["adam_checkpoints_match_existing"])
    checks["scorer_reproduces_evaluator"] = bool(sc["scorer_reproduces_evaluator"])
    # rows
    rows = {}
    new = {}
    for r in RUNS:
        st = runs[r["name"]]["status"]
        rec = {"status": st, "trajectory_exact": runs[r["name"]].get("trajectory_exact", True)}
        if st == "complete":
            rec.update(new_run_record(r, runs[r["name"]].get("attempts", [])))
            rec["F"] = sc["F"][r["name"]]["F_mean_per_sequence"]
            checks[f"{r['name']}_checks"] = rec["checks_pass"]
        new[r["name"]] = rec
    checks["existing_pass"] = bool(ex["pass"])
    if not all(checks.values()):
        # no statistic is reported, and none is LEFT in the artifact either; the status stays
        # SCORED so that a cause fixed outside this artifact (a shard restored, the existing
        # rows recollected) can be re-analysed. PRED's reporting is unchanged: a must-hold
        # failure means no statistic.
        stopped = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": checks,
                   "failed": sorted(k for k, v in checks.items() if not v), "new_runs": new,
                   "registration": prov, "provenance": git_provenance(R),
                   "discarded_results": out.pop("results", None) is not None}
        out.setdefault("analysis_stopped", []).append(stopped)
        out["status"] = SCORED
        write(out)
        print(json.dumps({"analysis_stopped": stopped["failed"], "checks": checks}, indent=1))
        return 1

    def arm_new(arm, exclude_flagged=False):
        return {str(r["seed"]): new[r["name"]] for r in RUNS if r["arm"] == arm
                and new[r["name"]]["status"] == "complete"
                and not (exclude_flagged and not new[r["name"]]["trajectory_exact"])}

    adam_ex = {s: {"trace": ex["adam_arms"]["adamw_w1.0"]["traces"][s],
                   "F": ex["F_arms"]["adamw_w1.0"]["per_seed"][s]}
               for s in ex["adam_arms"]["adamw_w1.0"]["traces"]}
    w0125 = {s: {"trace": ex["adam_arms"]["adamw_w0.125"]["traces"][s],
                 "F": ex["F_arms"]["adamw_w0.125"]["per_seed"][s]}
             for s in ex["adam_arms"]["adamw_w0.125"]["traces"]}
    reg = {k.split("seed")[1].split("_")[0]: {"trace": v["trace"], "F": v["F"]}
           for k, v in ex["reg"].items()}
    sgd = {k.split("seed")[1].split("_")[0]: {"trace": v["trace"], "F": v["F"]}
           for k, v in ex["sgd"].items() if v["status"] == "complete"}
    six = [str(s) for s in SEEDS]
    started = {"sgd": 3, "adam": 8, "adamw_wd0.01": 3, "adam_penalty": 8, "adam_proj": 3,
               "adam_w0.125": 8}
    src = {"sgd": sgd, "adam": adam_ex, "adamw_wd0.01": arm_new("adamw_wd0.01"),
           "adam_penalty": reg, "adam_proj": arm_new("adam_proj"), "adam_w0.125": w0125}
    for k, per in src.items():
        rows[k] = {"n_started": started[k], "n_trained": len(per), **row_stats(per)}
        if k in ("sgd", "adam", "adam_penalty", "adam_w0.125"):
            rows[k]["seeds_20261006_08"] = row_stats(per, six)
    rows["sgd"]["reported_as"] = "did not train at the registered LR (2 of 3 seeds diverged)"
    rows["adamw_wd0.01"]["framing"] = PRED["framing_adamw_user"]
    proj_init = {str(r["seed"]): new[r["name"]].get("initial_projection")
                 for r in RUNS if r["arm"] == "adam_proj"}
    # the registered test
    proj = arm_new("adam_proj")
    d_s = {}
    for s in six:
        if s in proj:
            d_s[s] = sc["F"][f"proj_seed{s}"]["F_mean_per_sequence"] \
                - sc["F"][f"adam_seed{s}"]["F_mean_per_sequence"]
    void = any(new[r["name"]]["status"] == "bound_violation" for r in RUNS)
    if void:
        rows["adam_proj"]["void"] = PRED["registered_test_user"]["void"]
        rows["adam_proj"]["bound_violation_runs"] = sorted(
            r["name"] for r in RUNS if new[r["name"]]["status"] == "bound_violation")
        test = {"status": "VOID: a live bound violation", "passes": False}
    elif len(d_s) < 3:
        test = {"status": f"NOT PASSED: {len(d_s)} of 3 projection runs trained",
                "passes": False, "d": d_s}
    else:
        test = {"status": "evaluated", **t_interval([d_s[s] for s in six]),
                "per_seed": d_s}
    beside = {"per_seed_differences": d_s, "duplicate_floor_F": ex["duplicate_floor_F"],
              "scorer_check_reported": sc["scorer_check_reported"],
              "scorer_reproduces_evaluator": sc["scorer_reproduces_evaluator"]}
    if len(d_s) == 3:
        D = np.stack([pf[f"proj_seed{s}"] - pf[f"adam_seed{s}"] for s in six])
        beside["bootstrap"] = bootstraps(D)
    flagged = sorted(r["name"] for r in RUNS if new[r["name"]]["status"] == "complete"
                     and not new[r["name"]]["trajectory_exact"])
    res = {"table3": rows, "initial_projection_per_seed": proj_init,
           "registered_test": test, "reported_beside_the_test": beside,
           "new_runs": new, "checks": checks, "all_checks_pass": all(checks.values()),
           "run_outcome": {"statuses": {n: v["status"] for n, v in new.items()},
                           "void": bool(void),
                           "partial": bool(out["run"].get("partial"))},
           "flagged_runs": flagged, "registration": prov, "provenance": git_provenance(R)}
    if flagged:
        fx = {"adamw_wd0.01": row_stats(arm_new("adamw_wd0.01", True)),
              "adam_proj": row_stats(arm_new("adam_proj", True))}
        ds = {s: d_s[s] for s in d_s if f"proj_seed{s}" not in flagged}
        fx["registered_test"] = t_interval(list(ds.values())) if len(ds) >= 2 else {
            "n": len(ds)}
        res["flagged_excluded"] = fx
    out["results"] = res
    out["status"] = ANALYSED
    write(out)
    print(json.dumps({"registered_test": test, "checks_pass": res["all_checks_pass"],
                      "table3": {k: {c: v.get(c) for c in ("n_trained", "n_started", "drift",
                                                          "mean_abs_step", "sd_epoch10",
                                                          "F_mean", "F_sd")}
                                 for k, v in rows.items()}}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    if ph == "score-worker":
        a = sys.argv[sys.argv.index("score-worker") + 1:]
        sys.exit(score_worker(int(a[0]), int(a[1]), a[2], json.loads(a[3]), int(a[4])))
    if ph == "smoke-cpu":
        sys.exit(smoke_cpu(sys.argv[sys.argv.index("--out") + 1]))
    if ph == "preregister":
        sys.exit(preregister(sys.argv[sys.argv.index("--smoke-cpu") + 1]
                             if "--smoke-cpu" in sys.argv else None))
    sys.exit({"collect-existing": collect_existing, "smoke-gpu": smoke_gpu, "run": run,
              "score": score, "analyse": analyse}[ph]())
