#!/usr/bin/env python3
"""A1: is the Type B headline an artifact of MFE decoding? Decoder ablation. (Preregistered.)

THE QUESTION. Type B's fold (runs/2026-10-12_turner_typeb, Experiment 2) moved Turner 2004 along
directions that none of the 11,955 TR0+TS0 reference structures can see and counted changed MFE
structures. At the Andronescu 2007 component (point 191, 21.0 kcal/mol) 4.3% of the 11,970
sequences change MFE. The paper's headline, 3.9%, is the real reorderings among them, with the
tie flips set aside (runs/2026-10-12_turner_typeb_tie_split). Null directions also change far
fewer MFEs than contrast directions of the same norm. A reviewer can ask whether both are
properties of MFE decoding. A1 refolds the same written tables and decodes every ensemble three
ways: MFE, MEA (gamma 1) and centroid.

DATA. The 11,970 TR0+TS0 sequences of turner_typeb.fold_rows() and their bpRNA references.
Tables: the fold's written tables, read from fold/points.npz. The sha256 of points.npz and
points.json is checked against the fold artifact's fold.points_sha256, and the read-time
direction assertions of typeb_common.fold_directions are run at BOTH registered test norms,
1 and 16 kcal/mol (19 directions each: 8 null, 3 published, 8 contrast), since that function
only checks the points at the norm it is given. There are 35 points:
  published_own 190-192 (Turner 1999, Andronescu 2007, Langdon 2018 components, own norms);
  null i at 1 kcal/mol (p = 10i+3) and at 16 (p = 10i+7);
  contrast j at 1 (p = 113+10j) and at 16 (p = 117+10j); i, j = 0..7.
Each point's kind, name and norm are asserted from points.json against that index rule. The
Turner 2004 baseline and the rounding control "ctrl" are folded as well.

ROUNDING CONTROL (a registered comparator). MEA and centroid have no genuine/rounding split.
The fold's exact tie split rests on integer MFE energies, and MEA and centroid structures are
functions of real-valued pair probabilities. So a changed MEA structure at point 191 may be
caused by the written table's rounding, which is at most 0.5 dcal per entry. The control
takes exactly the entries that point 191's written table changes (theta_r[191] != t0: 888
entries in int22, int21, int11 and bulge). It moves each by +-1 dcal from Turner 2004, with
signs 2*rng.integers(0, 2, n) - 1 in column order and rng =
default_rng(SeedSequence([20261012, 42])). Its 5 carry rows come from a copy of
turner_typeb.build_points' rule (top 5 generating-set rows by |G (theta - t0)|), and that
copy is first shown to reproduce points.json's carry rows at all 35 points. The indices, the
signs and the sha256 of their int64 bytes go into the registration (the `points` block).
The run re-derives that block and refuses if it differs.

WHAT THE CONTROL DOES AND DOES NOT BOUND (measured at registration, recorded in the points
block and pinned in ROUNDING_SCOPE; a property of the registered tables, not an outcome). On
the 888 entries it touches, the control moves 1 dcal per entry where the written table's
rounding residual theta_r[191] - (t0 + delta_dcal[191]) is at most 0.5, and in total norm the
control (29.80 dcal) exceeds the whole rounding residual (9.23 dcal over all 12,607 columns;
9.20 of it on those 888 entries). It is NOT a superset entry by entry: the written table also
rounds 3,636 entries that it leaves at Turner 2004 (theta_r = t0 while the residual is
nonzero), 5 of them by more than 1e-9 dcal, residual norm 0.710 and at most 0.5 per entry,
and the control does not touch those. So the claim carried in PRED is: a floor at least as
large as the rounding per entry on the entries the written table changes, and larger in total
norm -- not everywhere.

WORKER. A copy of turner_typeb.worker (lines 833-964); that file is not edited. Every guard is
kept:
  - write_par then turner_par.loaded;
  - params_match on the first compound;
  - the 5 carry rows;
  - the Boltzmann carry (a pf constrained to the top carry row's structure gives counts .
    theta_r);
  - per sequence, the reference-energy identity and the MFE-energy identity;
  - the genuine/rounding classification with EXACT_TIE.
Added:
  - md.compute_bpp = 1, set explicitly and read back;
  - in the baseline, assert_turner2004_params and assert_turner2004_boltzmann on the first
    compound;
  - after mfe(), exp_params_rescale(e) and pf(), each compound is decoded by fc.MEA(1.0) and
    fc.centroid(); every structure must be a str of length n over '().', balanced, with only
    AU/CG/GU pairs and j - i - 1 >= 3 for every pair (i, j);
  - the centroid's pair set must equal {(i, j): p_ij > 0.5} in the same compound's float64
    pair probabilities (ViennaRNA's own comparison is in double);
  - for every changed MFE, dE0 = c(s1).theta0 + offset - e0 >= 0 (no new MFE lies below the
    Turner 2004 MFE energy under Turner 2004).
Workers are niced (os.nice(10)) and run single-threaded with no GPU.

MUST-HOLD (any failure stops; nothing is repaired):
  - ViennaRNA is 2.7.2 (VIENNARNA) in every phase and in every worker; the registration's
    recorded version is checked against it too.
  - This script's sha256 is recorded at registration; smoke, mini, run and analyse all refuse
    unless the running file still has it, so the registered T1/T2 code is the code that runs.
    preregister and analyse(record=True) also refuse on a tree dirty outside runs/.
  - Baseline: e0 and the MFE's F1 against the reference equal the fold baseline, joined by
    sequence name.
  - At the 35 fold points the MFE results (changed, genuine, f1_ref, leak) equal the fold
    shards exactly, joined by name. The fold shards are runs/2026-10-12_turner_typeb/fold/full/
    shard{0..11}.jsonl, sha-checked against fold.shards_sha256. The check is made in the worker,
    against an extract (shards/fold_ref.json, sha256 recorded). analyse checks it again: it
    rebuilds the extract from the fold shards and requires it to be identical.
  - analyse: every point and decoder, the control included, carries one result per sequence;
    with record=True the analysis covers all 11,970 sequences (N_SEQUENCES) and the workers'
    baseline names are exactly the fold's, as a set.
  - analyse, on the full set: the MFE tie flips (dE0 = 0) and real reorderings (dE0 > 0) at
    points 190, 191 and 192 equal the registered exact tie split, 43/524, 44/469 and 35/91
    (TIE_SPLIT), both as pinned here and as read from runs/2026-10-12_turner_typeb_tie_split,
    whose status, numbers and sha256 are recorded at registration and re-checked.
  - analyse: _boot_mean's signature is (x, rng, n=1000).
  - analyse: the copied amendment-1 bootstrap reproduces the fold's registered amendment-1
    difference and ci95 EXACTLY at all 10 grid norms, on the fold's own genuine arrays, with
    rng SeedSequence([20261012, 5]) consumed in grid order as the fold did. If not, analyse
    stops and records nothing.

PHASES (each reads and extends runs/<RUN_ID>/metrics.json, written atomically):
  preregister  refuses if the artifact exists, and refuses unless the tree is clean outside
               runs/ (so this script is committed first) and ViennaRNA is 2.7.2; computes and
               records the points block (the 35 points' carry rows, the control's indices,
               signs, sha256 and rounding scope), the tie-split pin, this script's sha256 and
               PRED. Commit the artifact afterwards.
  smoke        2 workers, the first 50 sequences, points 191, 3 and 113. Then three planted
               failures, each on 10 sequences at point 3, that must be caught:
                 - the table load skipped, caught by a carry row;
                 - stale Boltzmann factors (turner_typeb._stale_boltzmann), caught by the
                   Boltzmann carry;
                 - md.compute_bpp = 0, caught by the MEA structure check.
               Then the zero-perturbation control: a table written from Turner 2004 itself
               (write_par of the template's theta) and loaded must leave every MFE, MEA and
               centroid structure, the MFE energy in dcal and every pair probability exactly
               as the built-in table gives them on the same 50 sequences, with params_match
               empty. Without it, a file-versus-built-in artifact would be counted as a
               changed MEA or centroid structure at every point.
               The result is recorded in the artifact.
  mini         every point and the control on the first 12 sequences, 2 workers, then
               analyse(record=False). The result is recorded in the artifact.
               Commit the artifact after smoke and mini, before run.
  run          the driver; launch it under setsid. It refuses unless all of these hold:
                 - the status is PRE-REGISTERED (a FAILED attempt is moved to failed_attempts);
                 - the predictions equal PRED and the points block is re-derived identically;
                 - smoke and mini passed with this exact script (sha256), which is also the
                   sha256 recorded at registration;
                 - ViennaRNA is the registered version;
                 - the artifact is tracked and unmodified;
                 - the tree is clean outside runs/.
               6 niced workers; shard sha256s are recorded.
  analyse      re-checks PRED, the fold inputs' sha256s, the shard and input sha256s and every
               must-hold, then computes the registered tests and the reported and exploratory
               numbers. With record=True it also refuses unless this file's sha256 is the one
               the run recorded, the tree is clean outside runs/, and the analysis covers the
               full 11,970 sequences; re-analysis of a recorded result is allowed only under
               those conditions.

CHOICES THE PLAN LEFT OPEN (the conservative option; also stated in PRED):
  - The control's signs and their sha256 are defined as above; its rng id is 193, the next
    index after the fold's points 0..192.
  - Paired dF1 subtracts the decoder's baseline F1 rounded to 6 dp, as each point's F1 is, so
    an unchanged structure contributes exactly 0.
  - The baseline degeneracy split uses the fold's own baseline gap. No subopt is run here.
  - The centroid margin uses the float64 baseline pair probabilities.
  - smoke and mini must pass, with this script's sha256, before run.
  - A must-hold failure in analyse stops it without recording anything.
  - The zero-perturbation control is not in the plan; it is added because nothing else would
    separate a write_par/loaded round-trip artifact from a real decoder change. It requires
    exact equality (max |dP| = 0, not a tolerance) on the smoke's 50 sequences.
  - The tie-split must-hold is skipped unless the analysis covers all 11,970 sequences, which
    the 12-sequence mini cannot; analyse(record=True) requires the full set, so the pinned
    numbers are always checked on the recorded result.
  - The rounding-control scope numbers are pinned in ROUNDING_SCOPE and recomputed at
    registration; preregister refuses if they differ. "Rounded elsewhere" is counted at a
    1e-9 dcal floor as well as exactly, because t0 + delta is a float sum and ~2.6k entries
    carry residuals at the 1e-12 level that are arithmetic noise, not rounding.

Run: .venv/bin/python scripts/typeb_decoders.py --phase preregister   (then commit)
     .venv/bin/python scripts/typeb_decoders.py --phase smoke
     .venv/bin/python scripts/typeb_decoders.py --phase mini           (then commit)
     setsid nohup .venv/bin/python scripts/typeb_decoders.py --phase run > LOG 2>&1 &
     .venv/bin/python scripts/typeb_decoders.py --phase analyse
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import inspect
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import RNA  # noqa: E402

from scripts.predecessor import RUNNING_PREFIX  # noqa: E402
from scripts.runmeta import git_provenance  # noqa: E402
from scripts.turner_nullspace import Extractor, build_matrix, generating_set  # noqa: E402
from scripts.turner_par import _rebuild, loaded, params_match, template, write_par  # noqa: E402
from scripts.turner_typeb import (  # noqa: E402
    BOOT_DRAWS,
    CARRY_ROWS,
    EXACT_TIE,
    FOLD,
    NULL_TABLES,
    GuardFailure,
    _boot_mean,
    _pairs,
    _stale_boltzmann,
    _vec,
    f1,
    fold_rows,
    load_basis,
)
from scripts.turner_typeb import read_out as fold_out  # noqa: E402
from scripts.typeb_common import fold_directions  # noqa: E402
from thermo.vienna_ref import (  # noqa: E402
    assert_turner2004_boltzmann,
    assert_turner2004_params,
    reference_md,
)

RUN_ID = "2026-10-12_typeb_decoders"
OUT = R / "runs" / RUN_ID
SHARDS = OUT / "shards"  # gitignored (runs/**/shards/); sha256 recorded
SEED = 20261012
VIENNARNA = "2.7.2"  # PRED pins the model; every phase and worker asserts it
N_SEQUENCES = 11970  # the fold's TR0+TS0 sequences; analyse(record=True) requires all of them
N_WORKERS = 6
NICE = 10
DECODERS = ("mfe", "mea", "centroid")  # decoder index d = 0, 1, 2 in every rng stream
MEA_GAMMA = 1.0
CANONICAL = frozenset({"AU", "UA", "GC", "CG", "GU", "UG"})
MIN_HAIRPIN = 3
PUBLISHED = ("Turner1999", "Andronescu2007", "Langdon2018")
OWN = (190, 191, 192)
A07 = 191
TEST_NORMS = (1.0, 16.0)
NULL_AT = {1.0: [10 * i + 3 for i in range(8)], 16.0: [10 * i + 7 for i in range(8)]}
CONTRAST_AT = {1.0: [113 + 10 * j for j in range(8)], 16.0: [117 + 10 * j for j in range(8)]}
FOLD_POINTS = sorted([*OWN, *(p for s in TEST_NORMS for p in NULL_AT[s] + CONTRAST_AT[s])])
CTRL = "ctrl"
CTRL_RNG_ID = 193  # the control's p in rng streams: the next index after the fold's 0..192
CTRL_SEED = [SEED, 42]
T1_DRAWS = 1000
T2_DRAWS = 2000
SPEC = "points_a1.json"
REF = "fold_ref.json"
REF_FIELDS = ("changed", "genuine", "f1_ref", "leak")
BASE_FIELDS = ("name", "tag", "len", "e0", "gap", "f1_ref", "ref_pairs")
SMOKE_POINTS = (191, 3, 113)
SMOKE_N, MINI_N, PLANTED_N, PLANTED_POINT = 50, 12, 10, 3
TIE_SPLIT_RUN = "2026-10-12_turner_typeb_tie_split"
# the registered exact tie split at the three published_own points (an earlier committed run);
# analyse re-checks both the artifact and this pin against its own dE0 counts
TIE_SPLIT = {190: {"n_tie_flips": 43, "n_real_reorderings": 524},
             191: {"n_tie_flips": 44, "n_real_reorderings": 469},
             192: {"n_tie_flips": 35, "n_real_reorderings": 91}}
# the written table's rounding at point 191 against the control, measured 2026-10-12 and
# recomputed by points_spec, which refuses if it differs (norms in dcal, 3 dp)
ROUNDING_SCOPE = {"n_control_entries": 888, "control_norm_dcal": 29.799,
                  "rounding_norm_control_entries_dcal": 9.202,
                  "rounding_norm_other_entries_dcal": 0.71,
                  "rounding_norm_all_entries_dcal": 9.229,
                  "rounding_max_abs_per_entry_dcal": 0.5,
                  "n_other_entries_rounded_nonzero": 3636,
                  "n_other_entries_rounded_above_1e_9": 5}
PLANTED = {"skip-load": "carry row", "stale-boltzmann": "Boltzmann carry row",
           "bpp0": "MEA structure"}
REGISTERED = "PRE-REGISTERED, nothing folded"
RUNNING = f"{RUNNING_PREFIX}: A1 decoder folds"
DONE = "folded (workers)"
assert T2_DRAWS == BOOT_DRAWS, "the copied amendment-1 bootstrap uses the fold's draw count"

PRED = {
    "registered": "2026-10-12, written before any phase of this script ran against runs/"
                  f"{RUN_ID}, and before anything at all was folded on the full "
                  f"{N_SEQUENCES} sequences; the full-set T1, T2, per-point and exploratory "
                  "numbers were unseen. What HAD been seen, in full: (a) the Type B fold's MFE "
                  "results (runs/2026-10-12_turner_typeb) and its exact tie split "
                  "(runs/2026-10-12_turner_typeb_tie_split: 43/524, 44/469 and 35/91 tie "
                  "flips / real reorderings at points 190, 191, 192, now a must-hold here); "
                  "and (b) MEA and centroid outcomes from five scratch harness runs of this "
                  "module against a scratch output directory, not runs/: (1) the worker at "
                  "points 191, 3, 113 and the control on 20 sequences (10 prefix sequences "
                  "plus 4 per point chosen because their MFE changed in the fold), with their "
                  "changed counts, dE0 values and centroid margins printed; (2) the smoke, the "
                  "first 50 sequences at points 191, 3 and 113, plus the three planted "
                  "failures on 10; (3) the mini, all 35 points and the control on the first 12 "
                  "sequences, whose dry-run T1 and T2 were printed for MEA and centroid (T1 "
                  "under MEA at 191: 5 of 12 changed against 0 of 12 at the control; T2 "
                  "differences at both 1 and 16 kcal/mol); (4) baseline decodes on every 30th "
                  "sequence and the worker at 191 and the control on the 8 longest sequences; "
                  "(5) tampered-input negative tests. 12 and 50 sequences decide neither T1 "
                  "nor T2, but the direction of both statistics on those samples was visible "
                  "before registration and this is the record of it",
    "question_user": "is the Type B headline an artifact of MFE decoding? Refold the fold's "
                     "written tables and decode every ensemble by MFE, MEA (gamma 1) and "
                     "centroid",
    "model": f"ViennaRNA {VIENNARNA} (RNA.__version__ asserted in every worker and in "
             "preregister, smoke, mini, run and analyse; the version recorded at registration "
             "is checked against it), Turner 2004, dangles 0, special_hp 0, 37 C "
             "(thermo.vienna_ref.reference_md) with md.compute_bpp = 1 set explicitly and read "
             "back. Per sequence and table one compound: mfe(); exp_params_rescale(e_mfe); "
             "pf(); then fc.MEA(1.0) and fc.centroid() on that same compound",
    "decoders": {"0": "MFE", "1": "MEA (gamma 1)", "2": "centroid"},
    "points_user": "the fold's written tables from fold/points.npz (sha256 checked against the "
                   "fold artifact), 35: published_own 190-192; null i at 1 kcal/mol (p = "
                   "10i+3) and at 16 (10i+7); contrast j at 1 (113+10j) and at 16 (117+10j); "
                   "i, j = 0..7; plus the Turner 2004 baseline and the rounding control",
    "rounding_control_user": "a registered comparator (user's approval): take the entries the "
                             "Andronescu 2007 point's written table changes, i.e. where "
                             "theta_r[191] != t0; add +-1 dcal to exactly those entries, with "
                             "signs drawn from default_rng(SeedSequence([20261012, 42])) and "
                             "recorded with their sha256; write and fold that table under the "
                             "same guards, as point 'ctrl'. It moves reference energies by "
                             "design, so it is not a null point. It is a rounding-noise floor "
                             "at least as large as the written table's rounding, which is at "
                             "most +-0.5 dcal per entry",
    "rounding_control_claude": "details the plan left open, chosen here: signs = "
                               "2 * rng.integers(0, 2, n) - 1 over those entries in column "
                               "order (one draw of n = 888 integers); the recorded sha256 is "
                               "of np.stack([indices, signs]) cast to little-endian int64 "
                               "bytes; the control's carry rows come from a copy of "
                               "turner_typeb.build_points' rule, first shown to reproduce "
                               "points.json's carry rows at all 35 fold points; in the "
                               "reporting rng streams the control's p is 193, the next index "
                               "after the fold's 0..192",
    "rounding_control_scope_claude": "what that floor does and does not bound, measured at "
                                     "registration, recomputed from points.npz and pinned in "
                                     "ROUNDING_SCOPE (a property of the registered tables, not "
                                     "an outcome): on the 888 entries the control touches it "
                                     "moves 1 dcal per entry where the written table's rounding "
                                     "residual theta_r - (t0 + delta_dcal) is at most 0.5 "
                                     "(residual norm 9.202 dcal there, against the control's "
                                     "29.799), and in total norm the control exceeds the whole "
                                     "residual (9.229 dcal over all 12,607 columns). It is not "
                                     "a superset entry by entry: the written table also rounds "
                                     "3,636 entries it leaves at Turner 2004, 5 of them by more "
                                     "than 1e-9 dcal (residual norm 0.710, at most 0.5 per "
                                     "entry), and the control does not touch those. So the "
                                     "claim is a floor at least as large as the rounding per "
                                     "entry on the entries the written table changes, and "
                                     "larger in total norm, not everywhere",
    "no_genuine_split_claude": "MEA and centroid have no genuine/rounding split: the fold's "
                               "exact tie split rests on integer MFE energies, while MEA and "
                               "centroid structures are functions of real-valued pair "
                               "probabilities. Their changes therefore include rounding-caused "
                               "ones; that is why T1 is tested against the control. The MFE "
                               "rounding share at the same points is reported beside them",
    "registered_test_T1_user": "AMENDED by the user before registration (tested against the "
                               "rounding control). Under MEA, at point 191: the paired "
                               "per-sequence difference changed_MEA(191) - changed_MEA(ctrl) "
                               "(values -1, 0, 1 over the 11,970 sequences) has a 95% CI above "
                               "0. CI: scripts.turner_typeb._boot_mean (sequence resampling, "
                               "percentile) with 1000 draws and rng "
                               "default_rng(SeedSequence([20261012, 40, 1, 191])). Passes iff "
                               "the lower bound > 0",
    "registered_test_T1_scope_claude": "the interval resamples sequences only, as the user's "
                                       "amendment specifies. It is conditional on the one "
                                       "registered draw of the control's 888 signs, so "
                                       "sign-draw variability is not in it; a second draw "
                                       "would be a second control, not a wider interval",
    "registered_test_T2_user": "unchanged. Under MEA at 1 AND at 16 kcal/mol: the mean "
                               "changed-MEA fraction over the 8 null directions minus that over "
                               "the 8 contrast directions has a 95% CI below 0 at both norms. "
                               "Method: an exact copy of the fold's amendment-1 bootstrap "
                               "(turner_typeb.fold_analyse): 2000 draws, each resampling "
                               "sequences once (shared by both arms) and, independently, each "
                               "arm's 8 directions with replacement; percentile CI. Passes iff "
                               "the upper bound < 0 at both norms",
    "registered_test_T2_rng_claude": "the plan gives no rng for T2, so this one is chosen: one "
                                     "rng, default_rng(SeedSequence([20261012, 41, 1])) for "
                                     "MEA, consumed at 1 kcal/mol and then at 16, as the fold "
                                     "consumed its single rng in grid order",
    "secondary_registered_claude": "the same two tests for the centroid (decoder index 2): T1 "
                                   "against the same control with SeedSequence([20261012, 40, "
                                   "2, 191]); T2 with SeedSequence([20261012, 41, 2]) at 1 "
                                   "then 16",
    "must_hold_claude": [
        f"RNA.__version__ == {VIENNARNA!r} in every phase and every worker, and the version "
        "recorded at registration equals it",
        "this file's sha256 is recorded at registration and equals the running file's in "
        "smoke, mini, run and analyse; preregister and analyse(record=True) also require a "
        "tree clean outside runs/, and analyse(record=True) requires the run's recorded "
        "script sha256",
        "the fold's points.npz and points.json equal fold.points_sha256; the 12 fold shards "
        "equal fold.shards_sha256; typeb_common.fold_directions' read-time assertions pass; "
        "every point's (kind, name, norm) follows the index rule above; "
        "typeb_common.fold_directions is run at both registered test norms, 1 and 16",
        "the copied carry rule reproduces points.json's carry rows at all 35 points; the "
        "control's rounding scope recomputes to ROUNDING_SCOPE",
        "smoke: a table written from Turner 2004 itself and loaded changes no MFE, MEA or "
        "centroid structure, no MFE energy in dcal and no pair probability (max |dP| = 0) on "
        "the 50 smoke sequences, with params_match empty",
        "per table: write_par then loaded(); params_match empty; the 5 carry rows; the "
        "Boltzmann carry. Per sequence: the reference-energy identity; the MFE-energy "
        "identity; genuine with EXACT_TIE; dE0 >= 0 for every changed MFE",
        "baseline: assert_turner2004_params and assert_turner2004_boltzmann on the first "
        "compound; e0 and the MFE's F1 against the reference equal the fold baseline by name",
        "every MEA and centroid structure: a str of length n over '().', balanced, canonical "
        "pairs only, j - i - 1 >= 3 for every pair; the centroid's pairs equal {p_ij > 0.5} "
        "in the same compound's float64 pair probabilities",
        "the MFE results at the 35 fold points (changed, genuine, f1_ref, leak) equal the "
        "fold shards exactly, joined by name: in the worker against the sha-recorded extract "
        "shards/fold_ref.json, and in analyse against an extract rebuilt from the fold shards",
        "analyse: every point and decoder, the control included, carries one changed, f1_ref "
        f"and f1_base value per sequence; with record=True all {N_SEQUENCES} sequences are "
        "covered and the workers' baseline names are the fold's as a set",
        "analyse on the full set: the MFE tie flips and real reorderings at points 190, 191 "
        "and 192 equal 43/524, 44/469 and 35/91, both as pinned in TIE_SPLIT and as read from "
        f"runs/{TIE_SPLIT_RUN} (status, numbers and sha256 recorded at registration and "
        "re-checked); skipped only when the analysis is not on the full set",
        "analyse: _boot_mean's signature is (x, rng, n=1000); the copied amendment-1 "
        "bootstrap reproduces the fold's registered difference and ci95 exactly at all 10 "
        "grid norms on the fold's genuine arrays (rng SeedSequence([20261012, 5]) in grid "
        "order); otherwise nothing is analysed",
        "any failure stops; nothing is repaired",
    ],
    "reported_without_prediction_user": "per decoder and point, each with CIs: changed "
                                        "fraction; paired dF1 against references; F1 between "
                                        "old and new structure among changed sequences",
    "reported_without_prediction_claude": "how those three are computed, chosen here: all 35 "
                                          "points and the control; each CI from "
                                          "turner_typeb._boot_mean with 1000 draws and rng "
                                          "default_rng(SeedSequence([20261012, 43, decoder, "
                                          "p])), the three CIs drawn from that one rng in the "
                                          "order changed, dF1, F1-old-new, with p = 193 for "
                                          "the control; paired dF1 = the point's F1 minus the "
                                          "decoder's own Turner 2004 baseline F1, both rounded "
                                          "to 6 dp, so an unchanged structure contributes "
                                          "exactly 0; F1 old-new averaged over the changed "
                                          "sequences only. Beside them for MFE: genuine and "
                                          "rounding-caused counts (at the control every MFE "
                                          "change is genuine by construction, its perturbation "
                                          "being exactly integer). Also reported, NOT tests "
                                          "and labelled would_pass_if_tested rather than "
                                          "passes: T1- and T2-shaped statistics on MFE changed "
                                          "(not genuine) fractions, rng [20261012, 40, 0, 191] "
                                          "and [20261012, 41, 0]; the fold's registered MFE "
                                          "result is on genuine changes",
    "exploratory_claude": "changes split by baseline MFE degeneracy (gap = 0 vs > 0, the "
                          "fold's baseline gap; no gap found within 1 kcal/mol counts as > 0); "
                          "the centroid margin min |p0_ij - 0.5| over the flipped pairs (p0 = "
                          "baseline float64 pair probabilities); MFE tie flips (dE0 = 0) vs "
                          "real reorderings (dE0 > 0) at every point including the control",
    "smoke_and_mini_claude": f"smoke: 2 workers, the first {SMOKE_N} sequences, points "
                             f"{list(SMOKE_POINTS)}; three planted failures on {PLANTED_N} "
                             f"sequences at point {PLANTED_POINT} that must be caught: the "
                             f"table load skipped (carry row), stale Boltzmann factors "
                             f"(Boltzmann carry), md.compute_bpp = 0 (MEA structure check); "
                             f"then the zero-perturbation control, a table written from Turner "
                             f"2004 itself, which must change no structure, no MFE energy and "
                             f"no pair probability on those {SMOKE_N} sequences. "
                             f"mini: all points and the control on the first {MINI_N} "
                             f"sequences, 2 workers, then analyse without recording. Both "
                             f"recorded in the artifact; run refuses unless both passed with "
                             f"the same script sha256",
    "execution_claude": f"{N_WORKERS} workers sharded by sequence (i % {N_WORKERS} over "
                        f"fold_rows()), niced by {NICE}, one thread each, no GPU; the driver "
                        f"under setsid; run refuses unless the artifact is tracked and "
                        f"unmodified, the tree is clean outside runs/, this file's sha256 is "
                        f"the registered one and ViennaRNA is the registered version",
    "seed": SEED,
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


def _check_version(out=None):
    """PRED pins the ViennaRNA version: MEA and centroid could move with it without tripping
    any MFE check, so nothing runs or is analysed under another version."""
    if RNA.__version__ != VIENNARNA:
        raise SystemExit(f"ViennaRNA is {RNA.__version__}, the registered {VIENNARNA}")
    if out is not None and out.get("viennarna_version") != VIENNARNA:
        raise SystemExit(f"the registration records ViennaRNA "
                         f"{out.get('viennarna_version')!r}, not {VIENNARNA}")
    return RNA.__version__


def _git_state():
    """Is the artifact tracked and unmodified, and is the tree clean outside runs/?"""
    rel = str((OUT / "metrics.json").relative_to(R))
    ls = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=R,
                        capture_output=True, text=True)
    diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=R,
                          capture_output=True, text=True)
    prov = git_provenance(R)
    return {"artifact_tracked": ls.returncode == 0, "artifact_unmodified": diff.returncode == 0,
            "tree_clean_outside_runs": not prov["git_dirty_excluding_runs"], "provenance": prov}


# ----------------------------------------------------------------------------------
# points: the fold's 35 and the rounding control
# ----------------------------------------------------------------------------------
def fold_inputs():
    """The fold artifact, points.json and points.npz, sha-checked; the directions re-asserted."""
    fo = fold_out()
    if fo.get("status") != "fold analysed":
        raise GuardFailure(f"the fold artifact's status is {fo.get('status')!r}")
    for f, h in fo["fold"]["points_sha256"].items():
        if _sha(FOLD / f) != h:
            raise GuardFailure(f"{f} changed since the fold")
    for s in TEST_NORMS:  # the read-time direction assertions, at BOTH registered test norms
        fold_directions(s)
    meta = json.loads((FOLD / "points.json").read_text())
    return fo, meta, np.load(FOLD / "points.npz")


def chosen_points(meta, fo):
    """The 35 fold points, each asserted against the registered index rule."""
    if [m["p"] for m in meta] != list(range(len(meta))) or len(meta) != 193:
        raise GuardFailure("points.json is not the fold's 193 points in order")
    own = fo["experiment_2_grid"]["published_points_kcal"]
    out = []
    for p in FOLD_POINTS:
        m = meta[p]
        if p in OWN:
            want = ("published_own", PUBLISHED[p - 190], own[PUBLISHED[p - 190]])
        elif p < 80:
            want = ("null", str(p // 10), {3: 1.0, 7: 16.0}[p % 10])
        else:
            want = ("contrast", str((p - 110) // 10), {3: 1.0, 7: 16.0}[p % 10])
        got = (m["kind"], m["name"], m["norm_kcal"])
        if got != want:
            raise GuardFailure(f"point {p}: points.json gives {got}, the rule {want}")
        out.append({"p": p, "kind": got[0], "name": got[1], "norm_kcal": got[2],
                    "n_entries_changed": m["n_entries_changed"], "carry": m["carry"]})
    return out


def _gen_matrix(cols):
    """The generating set's count matrix, as turner_typeb.build_points builds it."""
    T = template()
    gen = generating_set(T["special"])
    ex = Extractor(T["special"], 0, 0)
    return build_matrix([ex.counts(s, d) for _, s, d in gen], cols)


def _carry_rows(G, th, t0, label):
    """turner_typeb.build_points' carry rule (lines 780-783 and 800), copied."""
    ch = G @ (th - t0)
    top = np.argsort(-np.abs(ch), kind="stable")[:CARRY_ROWS]
    if ch[top[0]] == 0:
        raise GuardFailure(f"{label}: no generating-set row sees the written change")
    return [[int(r), int((G[r] @ th)[0])] for r in top]


def _rounding_scope(th, t0, dl, ix):
    """What the +-1 dcal control bounds: the written table's rounding residual
    th - (t0 + dl) on the control's entries ix and on every other column. Recomputed here and
    required to equal the pinned ROUNDING_SCOPE, so PRED's claim cannot drift from the
    tables."""
    res = (th - (t0 + dl)).astype(float)
    other = np.ones(res.size, dtype=bool)
    other[ix] = False
    got = {"n_control_entries": int(ix.size),
           "control_norm_dcal": round(float(np.sqrt(ix.size)), 3),
           "rounding_norm_control_entries_dcal": round(float(np.linalg.norm(res[ix])), 3),
           "rounding_norm_other_entries_dcal": round(float(np.linalg.norm(res[other])), 3),
           "rounding_norm_all_entries_dcal": round(float(np.linalg.norm(res)), 3),
           "rounding_max_abs_per_entry_dcal": round(float(np.abs(res).max()), 3),
           "n_other_entries_rounded_nonzero": int((res[other] != 0).sum()),
           "n_other_entries_rounded_above_1e_9": int((np.abs(res[other]) > 1e-9).sum())}
    if got != ROUNDING_SCOPE:
        raise GuardFailure(f"the control's rounding scope is {got}, pinned {ROUNDING_SCOPE}")
    return got


def tie_split_pins():
    """The registered exact tie split (runs/2026-10-12_turner_typeb_tie_split, an earlier
    committed run), checked against the TIE_SPLIT pin. analyse requires its counts on the full
    set, so the exploratory reordering numbers reported here cannot drift from the paper's."""
    p = R / "runs" / TIE_SPLIT_RUN / "metrics.json"
    art = json.loads(p.read_text())
    if art.get("status") != "complete":
        raise GuardFailure(f"{TIE_SPLIT_RUN}: status {art.get('status')!r}")
    by = {}
    for q, nm in zip(OWN, PUBLISHED, strict=True):
        a = art["published_points"][nm]
        got = {"n_tie_flips": int(a["n_tie_flips"]),
               "n_real_reorderings": int(a["n_real_reorderings"])}
        if got != TIE_SPLIT[q] or int(a["n_sequences"]) != N_SEQUENCES:
            raise GuardFailure(f"{TIE_SPLIT_RUN}: point {q} ({nm}) gives {got} over "
                               f"{a['n_sequences']} sequences, pinned {TIE_SPLIT[q]} over "
                               f"{N_SEQUENCES}")
        by[str(q)] = got | {"name": nm}
    return {"run_id": TIE_SPLIT_RUN, "sha256": _sha(p), "by_point": by,
            "rule": art["rule"]["classification"]}


def _ctrl_sha(ix, sg):
    return hashlib.sha256(np.stack([np.asarray(ix), np.asarray(sg)]).astype("<i8")
                          .tobytes()).hexdigest()


def _ctrl_table(t0, ix, sg):
    th = np.array(t0, dtype=np.int64, copy=True)
    th[ix] += sg
    return th


def points_spec():
    """The registered points block: the 35 fold points (carry rows re-derived and checked
    against points.json) and the rounding control."""
    fo, meta, Z = fold_inputs()
    B = load_basis("tr0ts0")
    cols, used, A = B["cols"], B["used"], B["A"]
    T = template()
    t0 = Z["t0"]
    if not np.array_equal(t0, np.array([T["theta"][k] for k in cols], dtype=np.int64)):
        raise GuardFailure("points.npz's t0 is not Turner 2004 over the basis columns")
    pts = chosen_points(meta, fo)
    G = _gen_matrix(cols)
    for pm in pts:
        c = _carry_rows(G, Z["theta_r"][pm["p"]], t0, f"point {pm['p']}")
        if c != pm["carry"]:
            raise GuardFailure(f"point {pm['p']}: the copied carry rule gives {c}, "
                               f"points.json {pm['carry']}")
    ix = np.flatnonzero(Z["theta_r"][A07] != t0)
    by_table = collections.Counter(cols[i][0] for i in ix)
    if not set(by_table) <= set(NULL_TABLES):
        raise GuardFailure(f"the control would touch {sorted(by_table)}")
    rng = np.random.default_rng(np.random.SeedSequence(CTRL_SEED))
    sg = 2 * rng.integers(0, 2, ix.size) - 1
    th = _ctrl_table(t0, ix, sg)
    if not np.array_equal(np.flatnonzero(th != t0), ix) or np.abs(th - t0).max() != 1:
        raise GuardFailure("the control is not +-1 dcal on exactly point 191's entries")
    ref = np.asarray(A @ (th - t0)[used]).ravel()
    ctrl = {"rule": {k: PRED[f"rounding_control_{k}"]
                     for k in ("user", "claude", "scope_claude")}, "seed": CTRL_SEED,
            "rounding_scope": _rounding_scope(Z["theta_r"][A07], t0, Z["delta_dcal"][A07], ix),
            "indices": [int(i) for i in ix], "signs": [int(s) for s in sg],
            "sha256": _ctrl_sha(ix, sg), "n_entries": int(ix.size),
            "n_plus": int((sg > 0).sum()), "n_minus": int((sg < 0).sum()),
            "by_table": dict(sorted(by_table.items())),
            "carry": _carry_rows(G, th, t0, "ctrl"),
            "n_references": int(ref.size),
            "references_changed_fraction": float(np.mean(ref != 0)),
            "reference_change_dcal_max_abs": int(np.abs(ref).max()),
            "reference_change_dcal_rms": float(np.sqrt(np.mean(ref.astype(float) ** 2)))}
    return {"fold_points": pts, "ctrl": ctrl,
            "fold_points_sha256": fo["fold"]["points_sha256"],
            "fold_shards_sha256": fo["fold"]["shards_sha256"],
            "carry_rule_reproduces_points_json": len(pts)}


# ----------------------------------------------------------------------------------
# the fold's per-sequence MFE results, by name
# ----------------------------------------------------------------------------------
def read_fold_shards(ps, fields=REF_FIELDS):
    """The fold's baseline and, for points ps, the per-sequence fields, in the fold's order
    (shards 0..11 concatenated, as fold_analyse reads them). Every shard sha-checked."""
    rec = fold_out()["fold"]["shards_sha256"]
    nW = len(rec)
    if set(rec) != {f"shard{w}.jsonl" for w in range(nW)}:
        raise GuardFailure(f"unexpected fold shard names {sorted(rec)}")
    want = {int(p) for p in ps}
    base = {k: [] for k in BASE_FIELDS}
    pts = {p: {f: [] for f in fields} for p in want}
    for w in range(nW):
        raw = (FOLD / "full" / f"shard{w}.jsonl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != rec[f"shard{w}.jsonl"]:
            raise GuardFailure(f"fold shard{w}.jsonl changed since the fold")
        lines = raw.decode().splitlines()
        b = json.loads(lines[0])["baseline"]
        for k in BASE_FIELDS:
            base[k].extend(x[k] for x in b)
        seen, done = set(), False
        for ln in lines[1:]:
            if ln == '{"done": true}':
                done = True
                continue
            if not ln.startswith('{"p": '):
                raise GuardFailure(f"fold shard{w}: unexpected line {ln[:40]!r}")
            p = int(ln[6:ln.index(",")])
            if p not in want:
                continue
            r = json.loads(ln)
            if r["p"] != p or p in seen:
                raise GuardFailure(f"fold shard{w}: point {p} repeated or misread")
            seen.add(p)
            for f in fields:
                if len(r[f]) != len(b):
                    raise GuardFailure(f"fold shard{w}: point {p} {f} has the wrong length")
                pts[p][f].extend(r[f])
        if seen != want or not done:
            raise GuardFailure(f"fold shard{w}: points missing or the shard is incomplete")
    if len(set(base["name"])) != len(base["name"]):
        raise GuardFailure("fold baseline names are not unique")
    return json.loads(json.dumps({"names": base["name"],
                                  "baseline": {k: base[k] for k in BASE_FIELDS if k != "name"},
                                  "points": {str(p): pts[p] for p in sorted(want)}}))


def prepare_inputs(dst, spec):
    """The worker's two input files in dst: the points block and the fold extract."""
    dst = pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / SPEC).write_text(json.dumps(spec, sort_keys=True))
    (dst / REF).write_text(json.dumps(read_fold_shards(FOLD_POINTS)))
    return {SPEC: _sha(dst / SPEC), REF: _sha(dst / REF)}


# ----------------------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------------------
def _check_structure(s, seq, what):
    """A str of length n over '().', balanced, canonical pairs, j - i - 1 >= 3 per pair."""
    n = len(seq)
    if not isinstance(s, str) or len(s) != n:
        raise GuardFailure(f"{what}: not a string of length {n}: {str(s)[:60]!r}")
    if set(s) - set("()."):
        raise GuardFailure(f"{what}: characters other than '().'")
    st = []
    for j, c in enumerate(s):
        if c == "(":
            st.append(j)
        elif c == ")":
            if not st:
                raise GuardFailure(f"{what}: unbalanced at {j}")
            i = st.pop()
            if seq[i] + seq[j] not in CANONICAL:
                raise GuardFailure(f"{what}: non-canonical pair ({i}, {j}) {seq[i]}{seq[j]}")
            if j - i - 1 < MIN_HAIRPIN:
                raise GuardFailure(f"{what}: pair ({i}, {j}) encloses fewer than "
                                   f"{MIN_HAIRPIN} nucleotides")
    if st:
        raise GuardFailure(f"{what}: unbalanced, {len(st)} open")


def _decode(fc, seq, where):
    """MEA(1) and centroid on a compound whose pf() was just computed, both checked; the
    centroid against {p_ij > 0.5} in the compound's float64 pair probabilities."""
    n = len(seq)
    sm = fc.MEA(MEA_GAMMA)[0]
    _check_structure(sm, seq, f"{where}: MEA structure")
    sc = fc.centroid()[0]
    _check_structure(sc, seq, f"{where}: centroid structure")
    P = np.array(fc.bpp(), dtype=np.float64)
    if P.shape != (n + 1, n + 1):
        raise GuardFailure(f"{where}: base-pair probabilities missing (shape {P.shape})")
    P = np.triu(P[1:n + 1, 1:n + 1], 1)
    ii, jj = np.nonzero(P > 0.5)
    if set(_pairs(sc)) != set(zip(ii.tolist(), jj.tolist(), strict=True)):
        raise GuardFailure(f"{where}: centroid pairs are not {{p > 0.5}}")
    return sm, sc, P


def worker(w, W, dst, src, spec_sha, ref_sha, limit=None, only=None, planted=None):
    """Shard w of W: the Turner 2004 baseline once, then the 35 fold points and the control,
    each decoded by MFE, MEA and centroid. Writes dst (JSON lines). A copy of
    turner_typeb.worker with all its guards; `planted` exists only for the smoke test's
    must-catch controls."""
    if planted not in (None, *PLANTED):
        raise ValueError(planted)
    if RNA.__version__ != VIENNARNA:
        raise GuardFailure(f"ViennaRNA is {RNA.__version__}, the registered {VIENNARNA}")
    os.nice(NICE)
    src = pathlib.Path(src)
    if _sha(src / SPEC) != spec_sha or _sha(src / REF) != ref_sha:
        raise GuardFailure("the worker's input files are not the recorded ones")
    spec = json.loads((src / SPEC).read_text())
    ref = json.loads((src / REF).read_text())
    if _sha(FOLD / "points.npz") != spec["fold_points_sha256"]["points.npz"]:
        raise GuardFailure("points.npz changed since the fold")
    T = template()
    theta0, special, lxc = T["theta"], T["special"], T["lxc"]
    B = load_basis("tr0ts0")
    cols = B["cols"]
    idx = {k: i for i, k in enumerate(cols)}
    Z = np.load(FOLD / "points.npz")
    t0 = Z["t0"]
    rows = fold_rows()[:limit]
    mine = [r for i, r in enumerate(rows) if i % W == w]
    rpos = {name: k for k, name in enumerate(ref["names"])}
    ex = Extractor(special, 0, 0, lxc=lxc)
    gen = generating_set(special)
    md = reference_md()
    want_bpp = 0 if planted == "bpp0" else 1
    md.compute_bpp = want_bpp
    if md.compute_bpp != want_bpp:
        raise GuardFailure(f"md.compute_bpp reads back {md.compute_bpp}, set {want_bpp}")
    out = open(dst, "w")  # noqa: SIM115

    def emit(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    _rebuild()
    base = []
    for n, (tag, name, seq, db) in enumerate(mine):
        if name not in rpos:
            raise GuardFailure(f"{name}: not in the fold's baseline")
        k = rpos[name]
        fc = RNA.fold_compound(seq, md)
        if n == 0:
            assert_turner2004_params(fc)
        s0, e0 = fc.mfe()
        fc.exp_params_rescale(e0)
        fc.pf()
        if n == 0:
            assert_turner2004_boltzmann(fc)
        sm0, sc0, P0 = _decode(fc, seq, f"baseline {name}")
        c0 = ex.counts(seq, s0)
        off0 = c0.pop(("_offset",), 0)
        cr = ex.counts(seq, db)
        cr.pop(("_offset",), 0)
        e0i = int(round(e0 * 100))
        f1m = f1(s0, db)
        if e0i != ref["baseline"]["e0"][k] or f1m != ref["baseline"]["f1_ref"][k]:
            raise GuardFailure(f"baseline {name}: e0 {e0i}, F1 {f1m} differ from the fold's "
                               f"{ref['baseline']['e0'][k]}, {ref['baseline']['f1_ref'][k]}")
        base.append({"tag": tag, "name": name, "seq": seq, "db": db, "k": k,
                     "s": {"mfe": s0, "mea": sm0, "centroid": sc0}, "e0": e0i, "P0": P0,
                     "c0": _vec(c0, idx), "off0": off0, "cr": _vec(cr, idx),
                     "ref_e0": int(round(fc.eval_structure(db) * 100))})
    emit({"baseline": [{"tag": b["tag"], "name": b["name"], "len": len(b["seq"]),
                        "e0": b["e0"], "gap": ref["baseline"]["gap"][b["k"]],
                        "ref_pairs": len(_pairs(b["db"])), "s": b["s"],
                        "f1_ref": {d: f1(b["s"][d], b["db"]) for d in DECODERS}}
                       for b in base]})
    _rebuild()

    plist = [(pm["p"], pm) for pm in spec["fold_points"]] + [(CTRL, spec["ctrl"])]
    with tempfile.TemporaryDirectory() as t:
        for p, pm in plist:
            if only is not None and p not in only:
                continue
            t1 = time.time()
            if p == CTRL:
                ix = np.array(pm["indices"], dtype=np.int64)
                sg = np.array(pm["signs"], dtype=np.int64)
                if _ctrl_sha(ix, sg) != pm["sha256"]:
                    raise GuardFailure("the control's indices and signs are not the registered")
                th = _ctrl_table(t0, ix, sg)
                dl = (th - t0).astype(float)  # exact: the control has no rounding
                fref = None
            else:
                th, dl = Z["theta_r"][p], Z["delta_dcal"][p]
                fref = ref["points"][str(p)]
            dth = th - t0
            full = dict(theta0) | {k: int(th[i]) for i, k in enumerate(cols)}
            path = write_par(full, pathlib.Path(t) / f"p{p}.par")
            ctx = (contextlib.nullcontext() if planted == "skip-load" else
                   _stale_boltzmann(path) if planted == "stale-boltzmann" else loaded(path))
            with ctx:
                first = RNA.fold_compound(mine[0][2], md)
                bad = params_match(first, full, 0, 0)
                if bad:
                    raise GuardFailure(f"point {p}: params_match {bad}")
                for r, want in pm["carry"]:
                    _, s, d = gen[r]
                    got = int(round(RNA.fold_compound(s, md).eval_structure(d) * 100))
                    if got != want:
                        raise GuardFailure(f"point {p}: carry row {r} evaluates to {got}, the "
                                           f"written table gives {want}")
                # the Boltzmann factors are a separate cache (turner_par V3): a partition
                # function constrained to the top carry row's one structure must give its
                # counts . theta_r free energy
                r, want = pm["carry"][0]
                _, s, d = gen[r]
                fcc = RNA.fold_compound(s, md)
                fcc.hc_add_from_db(d.replace(".", "x"), RNA.CONSTRAINT_DB_DEFAULT
                                   | RNA.CONSTRAINT_DB_ENFORCE_BP)
                _, ec = fcc.mfe()
                fcc.exp_params_rescale(ec)
                gc = int(round(fcc.pf()[1] * 100))
                if gc != want:
                    raise GuardFailure(f"point {p}: Boltzmann carry row {r} gives {gc}, the "
                                       f"written table gives {want}")
                rec = {"p": p}
                for dname in DECODERS:
                    rec[dname] = {"changed": [], "f1_ref": [], "f1_base": [], "new": []}
                rec["mfe"].update({"genuine": [], "leak": [], "dE0": []})
                rec["centroid"]["margin"] = []
                for b in base:
                    seq, db = b["seq"], b["db"]
                    ki, kv = b["cr"]
                    leak = int(kv @ dth[ki])
                    e_ref = int(round(RNA.fold_compound(seq, md).eval_structure(db) * 100))
                    if e_ref != b["ref_e0"] + leak:
                        raise GuardFailure(f"point {p}: {b['name']} reference energy {e_ref} != "
                                           f"{b['ref_e0']} + {leak}")
                    fc = RNA.fold_compound(seq, md)
                    s1, e1 = fc.mfe()
                    fc.exp_params_rescale(e1)
                    fc.pf()
                    sm1, sc1, _ = _decode(fc, seq, f"point {p}: {b['name']}")
                    changed = s1 != b["s"]["mfe"]
                    genuine, dE0 = None, None
                    if not changed:
                        i0, v0 = b["c0"]
                        if int(v0 @ th[i0]) + b["off0"] != int(round(e1 * 100)):
                            raise GuardFailure(f"point {p}: {b['name']} MFE energy is not "
                                               f"counts . theta_r (unchanged structure)")
                    if changed:
                        c1 = ex.counts(seq, s1)
                        off1 = c1.pop(("_offset",), 0)
                        i1, v1 = _vec(c1, idx)
                        i0, v0 = b["c0"]
                        if int(v1 @ th[i1]) + off1 != int(round(e1 * 100)):
                            raise GuardFailure(f"point {p}: {b['name']} MFE energy is not "
                                               f"counts . theta_r")
                        ex1 = v1 @ (t0[i1] + dl[i1]) + off1
                        ex0 = v0 @ (t0[i0] + dl[i0]) + b["off0"]
                        genuine = bool(ex1 <= ex0 + EXACT_TIE)
                        dE0 = int(v1 @ t0[i1]) + off1 - b["e0"]
                        if dE0 < 0:
                            raise GuardFailure(f"point {p}: {b['name']} new MFE lies {dE0} dcal "
                                               f"below the Turner 2004 MFE under Turner 2004")
                    f1r = round(f1(s1, db), 6)
                    if fref is not None:
                        got = [changed, genuine, f1r, leak]
                        want = [fref[f][b["k"]] for f in REF_FIELDS]
                        if got != want:
                            raise GuardFailure(f"point {p}: {b['name']} MFE result {got} "
                                               f"differs from the fold shard's {want}")
                    for dname, new in (("mfe", s1), ("mea", sm1), ("centroid", sc1)):
                        old = b["s"][dname]
                        ch = new != old
                        rd = rec[dname]
                        rd["changed"].append(ch)
                        rd["f1_ref"].append(round(f1(new, db), 6))
                        rd["f1_base"].append(round(f1(new, old), 6))
                        rd["new"].append(new if ch else None)
                    rec["mfe"]["genuine"].append(genuine)
                    rec["mfe"]["leak"].append(leak)
                    rec["mfe"]["dE0"].append(dE0)
                    margin = None
                    if sc1 != b["s"]["centroid"]:
                        flipped = set(_pairs(sc1)) ^ set(_pairs(b["s"]["centroid"]))
                        margin = float(min(abs(b["P0"][i, j] - 0.5) for i, j in flipped))
                    rec["centroid"]["margin"].append(margin)
                rec["seconds"] = round(time.time() - t1, 1)
            emit(rec)
    out.write(json.dumps({"done": True}) + "\n")
    out.close()
    return 0


def worker_cli(argv):
    w, W, dst, src, spec_sha, ref_sha = argv[:6]
    rest = argv[6:]
    limit = int(rest[rest.index("--limit") + 1]) if "--limit" in rest else None
    only = ({x if x == CTRL else int(x) for x in rest[rest.index("--only") + 1].split(",")}
            if "--only" in rest else None)
    planted = rest[rest.index("--planted") + 1] if "--planted" in rest else None
    return worker(int(w), int(W), dst, src, spec_sha, ref_sha, limit=limit, only=only,
                  planted=planted)


# ----------------------------------------------------------------------------------
# preregister, smoke, mini, run
# ----------------------------------------------------------------------------------
def preregister():
    if (OUT / "metrics.json").exists():
        raise SystemExit("the registration exists; it is not rewritten")
    _check_version()
    prov = git_provenance(R)
    if prov["git_dirty_excluding_runs"]:
        raise SystemExit(f"refused: commit this script first; the tree is dirty outside runs/: "
                         f"{prov['git_dirty_paths_outside_runs']}")
    spec = points_spec()
    write({"run_id": RUN_ID, "status": REGISTERED, "predictions": PRED, "points": spec,
           "tie_split_pin": tie_split_pins(), "script_sha256": _sha(__file__),
           "viennarna_version": RNA.__version__, "registration_provenance": prov})
    c = spec["ctrl"]
    print(json.dumps({"status": REGISTERED, "n_fold_points": len(spec["fold_points"]),
                      "ctrl": {k: c[k] for k in ("n_entries", "n_plus", "n_minus", "by_table",
                                                 "sha256", "carry",
                                                 "reference_change_dcal_max_abs",
                                                 "reference_change_dcal_rms")}}, indent=1))
    return 0


def _require_registered(out):
    """Status PRE-REGISTERED; PRED, the points block, the tie-split pin, this file's sha256
    and the ViennaRNA version exactly as registered."""
    if not str(out.get("status", "")).startswith("PRE-REGISTERED"):
        raise SystemExit(f"status is {out.get('status')!r}, not PRE-REGISTERED")
    _check_version(out)
    if out.get("predictions") != json.loads(json.dumps(PRED)):
        raise SystemExit("PRED changed since registration")
    if out.get("points") != json.loads(json.dumps(points_spec())):
        raise SystemExit("the registered points block is not reproduced")
    if out.get("tie_split_pin") != json.loads(json.dumps(tie_split_pins())):
        raise SystemExit("the registered tie-split pin is not reproduced")
    if out.get("script_sha256") != _sha(__file__):
        raise SystemExit("this script is not the one the registration recorded")


def _env():
    return dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")


def _launch(W, dsts, src, shas, extra=()):
    return [subprocess.Popen([sys.executable, __file__, "--phase", "worker", str(w), str(W),
                              str(d), str(src), shas[SPEC], shas[REF], *extra], env=_env())
            for w, d in enumerate(dsts)]


def _done(d):
    d = pathlib.Path(d)
    return d.exists() and d.read_text().rstrip().endswith('{"done": true}')


def zero_control(limit=SMOKE_N):
    """A zero-perturbation control, not in the plan: a table written from Turner 2004 itself
    and loaded must change nothing. Without it, a write_par/loaded round-trip artifact would be
    counted as a changed MEA or centroid structure at every point, and nothing else in this
    script would notice: the MFE must-hold only compares against the fold, which wrote its
    tables the same way. Exact equality is required, max |dP| = 0, not a tolerance."""
    _check_version()
    T = template()
    full = dict(T["theta"])
    md = reference_md()
    md.compute_bpp = 1
    if md.compute_bpp != 1:
        raise GuardFailure(f"md.compute_bpp reads back {md.compute_bpp}, set 1")
    rows = fold_rows()[:limit]
    _rebuild()
    base = []
    for n, (_tag, name, seq, _db) in enumerate(rows):
        fc = RNA.fold_compound(seq, md)
        if n == 0:
            assert_turner2004_params(fc)
        s0, e0 = fc.mfe()
        fc.exp_params_rescale(e0)
        fc.pf()
        if n == 0:
            assert_turner2004_boltzmann(fc)
        sm0, sc0, P0 = _decode(fc, seq, f"zero control, built-in table, {name}")
        base.append((name, seq, s0, int(round(e0 * 100)), sm0, sc0, P0))
    diff = dict.fromkeys(DECODERS, 0)
    dE, dP = 0, 0.0
    with tempfile.TemporaryDirectory() as t:
        path = write_par(full, pathlib.Path(t) / "zero.par")
        with loaded(path):
            bad = params_match(RNA.fold_compound(rows[0][2], md), full, 0, 0)
            if bad:
                raise GuardFailure(f"zero control: params_match {bad}")
            for name, seq, s0, e0i, sm0, sc0, P0 in base:
                fc = RNA.fold_compound(seq, md)
                s1, e1 = fc.mfe()
                fc.exp_params_rescale(e1)
                fc.pf()
                sm1, sc1, P1 = _decode(fc, seq, f"zero control, written table, {name}")
                for d, new, old in (("mfe", s1, s0), ("mea", sm1, sm0),
                                    ("centroid", sc1, sc0)):
                    diff[d] += int(new != old)
                dE = max(dE, abs(int(round(e1 * 100)) - e0i))
                dP = max(dP, float(np.abs(P1 - P0).max()))
    res = {"n_sequences": len(base), "n_changed": diff, "max_abs_dE_dcal": dE,
           "max_abs_dP": dP, "params_match_empty": True,
           "what": "write_par(Turner 2004) under loaded() against the built-in table"}
    res["passes"] = all(v == 0 for v in diff.values()) and dE == 0 and dP == 0.0
    return res


def smoke():
    """2 workers on the first 50 sequences at points 191, 3 and 113, then the three planted
    failures, which must each be caught by the named guard, then the zero-perturbation
    control."""
    out = read_out()
    _require_registered(out)
    if not _git_state()["artifact_tracked"]:
        raise SystemExit("commit the registration first")
    sm = SHARDS / "smoke"
    shas = prepare_inputs(sm, out["points"])
    dsts = [sm / f"shard{w}.jsonl" for w in range(2)]
    rc = [p.wait() for p in _launch(2, dsts, sm, shas, ["--limit", str(SMOKE_N), "--only",
                                                        ",".join(map(str, SMOKE_POINTS))])]
    done = [_done(d) for d in dsts]
    res = {"workers_exit": rc, "shards_done": done, "inputs_sha256": shas, "points": {}}
    if rc == [0, 0] and all(done):
        recs = [json.loads(ln) for d in dsts for ln in d.read_text().splitlines()]
        for p in SMOKE_POINTS:
            rs = [r for r in recs if r.get("p") == p]
            res["points"][str(p)] = {
                "n_seq": sum(len(r["mfe"]["changed"]) for r in rs),
                "n_changed": {d: sum(sum(r[d]["changed"]) for r in rs) for d in DECODERS},
                "mfe_genuine": sum(sum(1 for g in r["mfe"]["genuine"] if g) for r in rs),
                "seconds": [r["seconds"] for r in rs]}
    res["mfe_must_hold"] = "checked in the workers against the fold shards; exit 0 = held"
    res["planted"] = {}
    for kind, phrase in PLANTED.items():
        pr = subprocess.run([sys.executable, __file__, "--phase", "worker", "0", "1",
                             str(sm / f"planted_{kind}.jsonl"), str(sm), shas[SPEC], shas[REF],
                             "--limit", str(PLANTED_N), "--only", str(PLANTED_POINT),
                             "--planted", kind], capture_output=True, text=True, env=_env())
        last = (pr.stderr.strip().splitlines() or [""])[-1]
        caught = (pr.returncode != 0 and "GuardFailure" in last and phrase in last
                  and (kind != "skip-load" or "Boltzmann" not in last))
        res["planted"][kind] = {"exit": pr.returncode, "must_be_caught_by": phrase,
                                "caught": caught, "stderr_tail": last}
    res["zero_control"] = zero_control(SMOKE_N)
    res["passes"] = (rc == [0, 0] and all(done) and len(res["points"]) == len(SMOKE_POINTS)
                     and all(v["n_seq"] == SMOKE_N for v in res["points"].values())
                     and all(v["caught"] for v in res["planted"].values())
                     and res["zero_control"]["passes"])
    out = read_out()
    out["smoke"] = res | {"script_sha256": _sha(__file__), "provenance": git_provenance(R),
                          "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write(out)
    print(json.dumps(res, indent=1))
    return 0 if res["passes"] else 1


def mini():
    """Every point and the control on the first 12 sequences, 2 workers, then the analysis
    as a dry run (nothing recorded but this summary)."""
    out = read_out()
    _require_registered(out)
    if not _git_state()["artifact_tracked"]:
        raise SystemExit("commit the registration first")
    mi = SHARDS / "mini"
    shas = prepare_inputs(mi, out["points"])
    dsts = [mi / f"shard{w}.jsonl" for w in range(2)]
    rc = [p.wait() for p in _launch(2, dsts, mi, shas, ["--limit", str(MINI_N)])]
    res = {"workers_exit": rc, "shards_done": [_done(d) for d in dsts], "inputs_sha256": shas}
    if rc == [0, 0] and all(res["shards_done"]):
        a = analyse(src=mi, n_workers=2, record=False, inputs=shas)
        res.update({"n_sequences": a["n_sequences"],
                    "n_points_analysed": {d: len(a["per_point"][d]) for d in DECODERS},
                    "must_hold": {k: v for k, v in a["must_hold"].items()
                                  if k != "amendment_1_reproduced"}
                    | {"amendment_1_reproduced_all_exact":
                       a["must_hold"]["amendment_1_reproduced"]["all_exact"]}})
    res["passes"] = (rc == [0, 0] and all(res["shards_done"])
                     and res.get("n_points_analysed") == {d: len(FOLD_POINTS) + 1
                                                          for d in DECODERS})
    out = read_out()
    out["mini"] = res | {"script_sha256": _sha(__file__), "provenance": git_provenance(R),
                         "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write(out)
    print(json.dumps(res, indent=1))
    return 0 if res["passes"] else 1


def run():
    out = read_out()
    if str(out.get("status", "")).startswith("FAILED"):
        out.setdefault("failed_attempts", []).append({"status": out["status"], **out["run"]})
        out["status"] = REGISTERED
    _require_registered(out)
    if os.getsid(0) != os.getpid():
        raise SystemExit("launch under setsid")
    st = _git_state()
    if not (st["artifact_tracked"] and st["artifact_unmodified"]
            and st["tree_clean_outside_runs"]):
        raise SystemExit(f"refused: {json.dumps({k: v for k, v in st.items()})}")
    me = _sha(__file__)  # _require_registered has already pinned it to the registration's
    for ph in ("smoke", "mini"):
        if not out.get(ph, {}).get("passes"):
            raise SystemExit(f"the {ph} test has not passed")
        if out[ph].get("script_sha256") != me:
            raise SystemExit(f"the {ph} test ran another version of this script")
    shas = prepare_inputs(SHARDS, out["points"])
    dsts = [SHARDS / f"shard{w}.jsonl" for w in range(N_WORKERS)]
    out.update({"status": RUNNING, "run": {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                           "driver_pid": os.getpid(),
                                           "load_at_start": os.getloadavg(),
                                           "n_workers": N_WORKERS, "nice": NICE,
                                           "inputs_sha256": shas, "script_sha256": me,
                                           "provenance": st["provenance"]}})
    write(out)
    procs = _launch(N_WORKERS, dsts, SHARDS, shas)
    out["run"]["worker_pids"] = [p.pid for p in procs]
    write(out)
    rc = [p.wait() for p in procs]
    done = [_done(d) for d in dsts]
    out = read_out()
    out["run"].update({"workers_exit": rc, "shards_done": done,
                       "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "shards_sha256": {d.name: _sha(d) for d in dsts if d.exists()}})
    ok = rc == [0] * N_WORKERS and all(done)
    out["status"] = DONE if ok else f"FAILED: workers exited {rc}"
    write(out)
    return 0 if ok else 1


# ----------------------------------------------------------------------------------
# analyse
# ----------------------------------------------------------------------------------
def amend1_bootstrap(Gn, Gc, rng, draws=T2_DRAWS):
    """An exact copy of the fold's amendment-1 bootstrap (turner_typeb.fold_analyse, lines
    1184-1194): sequences resampled once per draw and shared by both arms, each arm's
    directions resampled independently; percentile 95% CI. Gn, Gc: directions x sequences."""
    nseq = Gn.shape[1]
    diffs = np.empty(draws)
    for b in range(draws):
        si = rng.integers(0, nseq, nseq)
        dn = rng.integers(0, len(Gn), len(Gn))
        dc = rng.integers(0, len(Gc), len(Gc))
        diffs[b] = Gn[dn][:, si].mean() - Gc[dc][:, si].mean()
    ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    return float(Gn.mean() - Gc.mean()), ci


def reproduce_amendment_1():
    """Must-hold: the copy reproduces the fold's registered amendment-1 numbers exactly, on
    the fold's genuine arrays, at every grid norm, with the fold's rng stream."""
    fo = fold_out()
    meta = json.loads((FOLD / "points.json").read_text())
    grid = fo["experiment_2_grid"]["grid_kcal"]
    reg = fo["experiment_2"]["amendment_1_null_vs_contrast"]["by_scale"]
    F = read_fold_shards([m["p"] for m in meta if m["kind"] in ("null", "contrast")],
                         fields=("genuine",))
    gen = {int(p): np.array([g is True for g in v["genuine"]]) for p, v in F["points"].items()}
    rng_b = np.random.default_rng(np.random.SeedSequence([SEED, 5]))
    rows = []
    for s, r in zip(grid, reg, strict=True):
        Gn = np.stack([gen[m["p"]] for m in meta if m["kind"] == "null"
                       and m["norm_kcal"] == s]).astype(float)
        Gc = np.stack([gen[m["p"]] for m in meta if m["kind"] == "contrast"
                       and m["norm_kcal"] == s]).astype(float)
        d, ci = amend1_bootstrap(Gn, Gc, rng_b, BOOT_DRAWS)
        rows.append({"norm_kcal": s, "difference": d, "ci95": ci,
                     "registered_difference": r["difference"], "registered_ci95": r["ci95"],
                     "exact": r["norm_kcal"] == s and d == r["difference"] and ci == r["ci95"]})
    res = {"n_norms": len(rows), "all_exact": all(x["exact"] for x in rows), "by_scale": rows}
    if not res["all_exact"] or len(rows) != 10:
        raise SystemExit(f"MUST-HOLD FAILED: the copied amendment-1 bootstrap does not "
                         f"reproduce the fold's registered CIs: {json.dumps(rows)}")
    return res


def _check_boot_mean():
    sig = inspect.signature(_boot_mean)
    if list(sig.parameters) != ["x", "rng", "n"] or sig.parameters["n"].default != 1000:
        raise SystemExit(f"MUST-HOLD FAILED: turner_typeb._boot_mean is {sig}")
    return str(sig)


def _rng(*ids):
    return np.random.default_rng(np.random.SeedSequence([SEED, *ids]))


def _pid(p):
    return CTRL_RNG_ID if p == CTRL else int(p)


def check_tie_split(per):
    """Must-hold (full set only): this script's exploratory MFE tie flips and real reorderings
    at points 190-192 equal the registered exact tie split, which is also re-read from its
    artifact. The paper's 3.9% headline is one of those numbers, so a silent disagreement here
    would mean the two runs count different things."""
    for q in OWN:
        row = per["mfe"][str(q)]
        got = {"n_tie_flips": row["exploratory_tie_flips_dE0_0"],
               "n_real_reorderings": row["exploratory_real_reorderings_dE0_pos"]}
        if got != TIE_SPLIT[q]:
            raise SystemExit(f"MUST-HOLD FAILED: point {q} tie split {got}, the registered "
                             f"{TIE_SPLIT[q]}")
    return tie_split_pins() | {"points": [str(q) for q in OWN]}


def _summary(x):
    x = np.asarray(x, dtype=float)
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "min": float(x.min()), "p10": float(np.percentile(x, 10)),
            "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "max": float(x.max()), "fraction_below_0.01": float(np.mean(x < 0.01)),
            "fraction_below_0.05": float(np.mean(x < 0.05))}


def analyse(src=None, n_workers=N_WORKERS, record=True, inputs=None):
    out = read_out()
    _check_version(out)
    if record:
        if out.get("status") not in (DONE, "analysed"):
            raise SystemExit(f"status is {out.get('status')!r}")
        # the recorded results must come from the code that ran, on a committed tree
        if _sha(__file__) != out["run"].get("script_sha256"):
            raise SystemExit("this script is not the one that produced the shards")
        if out.get("script_sha256") != _sha(__file__):
            raise SystemExit("this script is not the one the registration recorded")
        prov = git_provenance(R)
        if prov["git_dirty_excluding_runs"]:
            raise SystemExit(f"refused: the tree is dirty outside runs/: "
                             f"{prov['git_dirty_paths_outside_runs']}")
    if out.get("predictions") != json.loads(json.dumps(PRED)):
        raise SystemExit("PRED changed since registration; nothing is analysed")
    fold_inputs()  # points.npz and points.json sha-checked; the directions re-asserted
    src = SHARDS if src is None else pathlib.Path(src)
    if record:
        for f, h in out["run"]["shards_sha256"].items():
            if _sha(src / f) != h:
                raise SystemExit(f"{f} changed since the run")
        inputs = out["run"]["inputs_sha256"]
    for f, h in inputs.items():
        if _sha(src / f) != h:
            raise SystemExit(f"{f} changed since the run")
    if json.loads((src / SPEC).read_text()) != out["points"]:
        raise SystemExit("the workers' points block differs from the registration")
    ref = json.loads((src / REF).read_text())
    must = {"boot_mean_signature": _check_boot_mean(),
            "amendment_1_reproduced": reproduce_amendment_1()}
    if read_fold_shards(FOLD_POINTS) != ref:
        raise SystemExit("MUST-HOLD FAILED: fold_ref.json differs from the fold shards")
    shards = [[json.loads(ln) for ln in (src / f"shard{w}.jsonl").read_text().splitlines()]
              for w in range(n_workers)]
    for w, s in enumerate(shards):
        if s[-1] != {"done": True}:
            raise SystemExit(f"shard{w} is incomplete")
    base = [b for s in shards for b in s[0]["baseline"]]
    n = len(base)
    names = [b["name"] for b in base]
    if len(set(names)) != n:
        raise SystemExit("MUST-HOLD FAILED: the workers' baseline names are not unique")
    full = n == N_SEQUENCES
    if record and not full:
        raise SystemExit(f"MUST-HOLD FAILED: {n} sequences analysed, not the full "
                         f"{N_SEQUENCES}; nothing is recorded")
    if full and (len(ref["names"]) != N_SEQUENCES or sorted(names) != sorted(ref["names"])):
        raise SystemExit("MUST-HOLD FAILED: the sequences are not the fold's "
                         f"{N_SEQUENCES}")
    keys = [*FOLD_POINTS, CTRL]
    recs = {p: [next(r for r in s if r.get("p") == p) for s in shards] for p in keys}

    def cat(p, d, f):
        return [x for r in recs[p] for x in r[d][f]]

    # coverage: every point and decoder, the control included, carries one value per sequence
    for p in keys:
        for d in DECODERS:
            for f in ("changed", "f1_ref", "f1_base"):
                if len(cat(p, d, f)) != n:
                    raise SystemExit(f"MUST-HOLD FAILED: point {p} {d} {f} covers "
                                     f"{len(cat(p, d, f))} of {n} sequences")
    must["coverage"] = {"n_sequences": n, "full_set": full, "n_points": len(keys),
                        "decoders": list(DECODERS), "fields": ["changed", "f1_ref", "f1_base"],
                        "names_equal_fold": bool(full)}

    # must-hold, again: the MFE results equal the fold's, by name
    rpos = {nm: k for k, nm in enumerate(ref["names"])}
    ix = [rpos[b["name"]] for b in base]
    if ([b["e0"] for b in base] != [ref["baseline"]["e0"][k] for k in ix]
            or [b["f1_ref"]["mfe"] for b in base] != [ref["baseline"]["f1_ref"][k] for k in ix]):
        raise SystemExit("MUST-HOLD FAILED: the baseline differs from the fold's")
    for p in FOLD_POINTS:
        for f in REF_FIELDS:
            if cat(p, "mfe", f) != [ref["points"][str(p)][f][k] for k in ix]:
                raise SystemExit(f"MUST-HOLD FAILED: point {p} MFE {f} differs from the fold")
    must["mfe_reproduces_fold"] = {"points": len(FOLD_POINTS), "sequences": n,
                                   "fields": list(REF_FIELDS), "baseline": ["e0", "f1_ref"]}

    gap0 = np.array([b["gap"] == 0 for b in base])
    ch = {d: {p: np.array(cat(p, d, "changed"), dtype=bool) for p in keys} for d in DECODERS}
    per = {d: {} for d in DECODERS}
    for di, d in enumerate(DECODERS):
        f0 = np.array([round(b["f1_ref"][d], 6) for b in base])
        for p in keys:
            c = ch[d][p]
            dF1 = np.array(cat(p, d, "f1_ref")) - f0
            fb = np.array(cat(p, d, "f1_base"))[c]
            rng = _rng(43, di, _pid(p))
            row = {"n_changed": int(c.sum()), "changed_fraction": float(c.mean()),
                   "changed_fraction_ci95": _boot_mean(c, rng),
                   "paired_dF1": float(dF1.mean()), "paired_dF1_ci95": _boot_mean(dF1, rng),
                   "f1_old_new_among_changed": float(fb.mean()) if fb.size else None,
                   "f1_old_new_among_changed_ci95": _boot_mean(fb, rng),
                   "exploratory_by_baseline_gap": {
                       "gap0": {"n": int(gap0.sum()), "n_changed": int(c[gap0].sum()),
                                "changed_fraction": float(c[gap0].mean())},
                       "gap_above_0": {"n": int((~gap0).sum()),
                                       "n_changed": int(c[~gap0].sum()),
                                       "changed_fraction": float(c[~gap0].mean())}}}
            if d == "mfe":
                g = np.array([x is True for x in cat(p, d, "genuine")])
                e = [x for x in cat(p, d, "dE0") if x is not None]
                if len(e) != int(c.sum()):
                    raise SystemExit(f"point {p}: dE0 recorded for {len(e)} of {c.sum()} "
                                     f"changed MFEs")
                row.update({"n_genuine": int(g.sum()),
                            "n_rounding_caused": int(c.sum() - g.sum()),
                            "rounding_share_of_changed": (float(1 - g.sum() / c.sum())
                                                          if c.sum() else None),
                            "exploratory_tie_flips_dE0_0": int(sum(x == 0 for x in e)),
                            "exploratory_real_reorderings_dE0_pos": int(sum(x > 0 for x in e)),
                            "exploratory_real_reordering_fraction": sum(x > 0 for x in e) / n})
            if d == "centroid":
                row["exploratory_margin"] = _summary(
                    [x for x in cat(p, d, "margin") if x is not None])
            per[d][str(p)] = row

    # must-hold on the full set: the exploratory tie split equals the registered one
    must["tie_split_reproduced"] = (check_tie_split(per) if full else
                                    f"skipped: {n} sequences, not the full {N_SEQUENCES}")

    def mfe_share(ps):
        c = sum(per["mfe"][str(p)]["n_changed"] for p in ps)
        g = sum(per["mfe"][str(p)]["n_genuine"] for p in ps)
        return {"changed": c, "genuine": g, "rounding_share": (c - g) / c if c else None}

    def verdict(di):
        """MFE (di = 0) is reported for comparison and is not a registered test, so its
        outcome is never labelled `passes`."""
        return "passes" if di else "would_pass_if_tested"

    def t1(di):
        d = DECODERS[di]
        x = ch[d][A07].astype(int) - ch[d][CTRL].astype(int)
        ci = _boot_mean(x, _rng(40, di, A07), n=T1_DRAWS)
        return {"decoder": d, "rng": [SEED, 40, di, A07], "draws": T1_DRAWS,
                "changed_fraction_191": float(ch[d][A07].mean()),
                "changed_fraction_ctrl": float(ch[d][CTRL].mean()),
                "difference": float(x.mean()), "ci95": ci, verdict(di): ci[0] > 0,
                "mfe_rounding_share_191": per["mfe"][str(A07)]["rounding_share_of_changed"],
                "mfe_rounding_share_ctrl": per["mfe"][CTRL]["rounding_share_of_changed"]}

    def t2(di):
        d = DECODERS[di]
        rng = _rng(41, di)
        by = {}
        for s in TEST_NORMS:
            Gn = np.stack([ch[d][p] for p in NULL_AT[s]]).astype(float)
            Gc = np.stack([ch[d][p] for p in CONTRAST_AT[s]]).astype(float)
            diff, ci = amend1_bootstrap(Gn, Gc, rng)
            by[f"{s:g}"] = {"null_mean_changed_fraction": float(Gn.mean()),
                            "contrast_mean_changed_fraction": float(Gc.mean()),
                            "difference": diff, "ci95": ci,
                            "ratio_null_over_contrast": (float(Gn.mean() / Gc.mean())
                                                         if Gc.mean() else None),
                            "side": ("below" if ci[1] < 0 else "above" if ci[0] > 0
                                     else "contains_0"),
                            "mfe_rounding_share_null": mfe_share(NULL_AT[s]),
                            "mfe_rounding_share_contrast": mfe_share(CONTRAST_AT[s])}
        return {"decoder": d, "rng": [SEED, 41, di], "draws": T2_DRAWS, "by_norm": by,
                verdict(di): all(v["ci95"][1] < 0 for v in by.values())}

    arms = {"published_own": list(OWN), "ctrl": [CTRL]} | {
        f"{k}@{s:g}": v[s] for k, v in (("null", NULL_AT), ("contrast", CONTRAST_AT))
        for s in TEST_NORMS}
    by_arm = {a: {d: {"mean_changed_fraction": float(np.mean(
        [per[d][str(p)]["changed_fraction"] for p in ps])),
        "mean_paired_dF1": float(np.mean([per[d][str(p)]["paired_dF1"] for p in ps]))}
        for d in DECODERS} | {
        "mfe_rounding": mfe_share(ps),
        "exploratory_mfe_tie_flips": sum(per["mfe"][str(p)]["exploratory_tie_flips_dE0_0"]
                                         for p in ps),
        "exploratory_mfe_real_reorderings": sum(
            per["mfe"][str(p)]["exploratory_real_reorderings_dE0_pos"] for p in ps),
        "exploratory_centroid_margin": _summary(
            [x for p in ps for x in cat(p, "centroid", "margin") if x is not None])}
        for a, ps in arms.items()}
    res = {"n_sequences": n, "must_hold": must,
           "registered": {"T1_MEA": t1(1), "T2_MEA": t2(1)},
           "secondary_registered": {"T1_centroid": t1(2), "T2_centroid": t2(2)},
           "mfe_for_comparison_not_tests": {"T1_shaped_MFE_changed": t1(0),
                                            "T2_shaped_MFE_changed": t2(0)},
           "ctrl": {k: v for k, v in out["points"]["ctrl"].items()
                    if k not in ("indices", "signs", "rule")},
           "by_arm": by_arm, "per_point": per,
           "baseline_f1_ref_mean": {d: float(np.mean([b["f1_ref"][d] for b in base]))
                                    for d in DECODERS},
           "baseline_fraction_gap0": float(gap0.mean()),
           "script_sha256": _sha(__file__)}
    if not record:
        return res
    out["results"] = res | {"provenance": git_provenance(R)}
    out["status"] = "analysed"
    write(out)
    print(json.dumps({"registered": res["registered"],
                      "secondary_registered": res["secondary_registered"],
                      "mfe_for_comparison_not_tests": res["mfe_for_comparison_not_tests"]},
                     indent=1))
    return 0


PHASES = {"preregister": preregister, "smoke": smoke, "mini": mini, "run": run,
          "analyse": analyse}

if __name__ == "__main__":
    if sys.argv[1:3] == ["--phase", "worker"]:
        sys.exit(worker_cli(sys.argv[3:]))
    if len(sys.argv) != 3 or sys.argv[1] != "--phase" or sys.argv[2] not in PHASES:
        raise SystemExit(f"usage: typeb_decoders.py --phase {{{','.join(PHASES)}}}")
    sys.exit(PHASES[sys.argv[2]]())
