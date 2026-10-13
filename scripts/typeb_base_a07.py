#!/usr/bin/env python3
"""A2: is the Type B effect specific to Turner 2004? Repeat it with Andronescu 2007 as theta0.
(Preregistered. RECOMB ablation A2; plan of record docs/PLAN.md, "RECOMB ablations A1-A3".)

THE QUESTION. Type B (runs/2026-10-12_turner_typeb, Experiment 2) moved Turner 2004 along
directions that none of the 11,955 TR0+TS0 reference structures can see, and MFEs changed
anyway. E1 (runs/2026-10-12_typeb_fisher) found that likelihood training at Turner 2004 carries
9.41 times more information about contrast directions than about null ones. Both could be
properties of Turner 2004. A2 repeats both at another published base table, Andronescu 2007
(theta_A07). The directions carry over unchanged: the count matrix A and the null basis N_nt do
not depend on theta, so a direction null at Turner 2004 is null at A07.

Model throughout: ViennaRNA 2.7.2, dangles 0, special_hp 0, 37 C (thermo.vienna_ref.reference_md).

BASE TABLE (phase build; its record is committed by the lead BEFORE preregistration, as
scripts/typeb_heldout.py's build was). theta_A07 is ViennaRNA's shipped Andronescu 2007 set
(params_load_RNA_Andronescu2007, read through params_save by turner_typeb.dump_set), written on
the Turner 2004 template (turner_par.write_par) at the keys the two share. Checks:
  B1  dump: no INF and nothing absent on the 12,607 reachable columns; Ninio (m, cap) = (50,
      300); lxc equal to Turner 2004's; the count model's Ninio linearisation is exact
      (5m <= cap <= 6m).
  B2  key differences: every key not shared is a special hairpin (unused at special_hp 0);
      12,849 shared keys.
  B3  the written table (shards/a07.par, sha256 recorded) parses back to exactly the template
      with A07's values on the shared keys, and lxc round-trips.
  B4  params_match is empty for a compound inside loaded(a07.par), non-empty for a Turner 2004
      compound after it.
  B5  inside loaded(a07.par), eval_structure equals counts . theta_A07 + offset
      (+ typeb_common.ninio_correction(db, 50, 300), which must be 0 on every row) on all
      23,741 generating-set rows (turner_nullspace.generating_set), 0 mismatches.
  B6  80 seeded random sequences (lengths 30-200, SeedSequence([20261012, 50])): MFE
      structure and energy, ensemble G, MEA(1) structure and score, centroid structure and
      distance are identical under loaded(a07.par) and under the shipped loader
      (params_load_RNA_Andronescu2007 + turner_par._rebuild). md.compute_bpp = 1, read back.
  B7  MUST FAIL: the written table with one int22 entry +1 dcal (the first int22 column, in
      basis order, used by TR0+TS0 and by the generating set) fails B5 on exactly the
      generating rows whose count of that entry is nonzero (each int22 entry is built once, so
      one row), and that row's energy rises by exactly its count of the entry.
  B8  the generic Boltzmann check (no Turner 2004 pin): a partition function hard-constrained
      to one generating-set structure returns that structure's counts . theta exactly (dcal),
      on the 5 rows whose energy theta_A07 changes most against Turner 2004; and a compound's
      own Boltzmann factors (exp_params, read after pf() as E1's assert_turner2004_boltzmann
      check (1) reads them) give A07's TerminalAU and ML_cc to 1e-6 dcal. MUST FAIL: both
      with stale Boltzmann factors (A07's energies, Turner 2004's factors).
  B9  the 33 points from the fold's sha-checked points.npz (kinds, names and norms verified
      in points.json). Each written table carries the fold's written integer perturbation:
      theta_r = theta_A07 + (theta_r - theta_T04 of the fold's row), negated for the mirrored
      point (row 191). Each entry is asserted to be a nearest-integer rounding of
      theta_A07 + delta (|theta_r - theta_A07 - delta| <= 0.5 + 1e-9 dcal). This equals
      rint(theta_A07 + delta) except at ties: np.rint rounds half to even, and the mirrored
      component has entries that are half-integers to float precision, so rint at the A07
      base can break a tie the other way than the fold did (the build records every such
      entry). Carrying the fold's integers makes the written perturbations, written errors and
      leaks exactly the fold's (negated for the mirror):
        null      the fold's 8 random null directions at 4 and 16 kcal/mol (rows 10i+5, 10i+7)
        contrast  the fold's 8 contrast directions at 4 and 16 (rows 115+10j, 117+10j)
        mirrored  delta = -delta_dcal[191] = P_nt(theta_T04 - theta_A07), 21.03 kcal/mol,
                  asserted equal (1e-9 relative) to the independent computation through
                  turner_typeb.set_delta and to one from the written A07 table
      |A delta| <= 1e-9 kcal/mol per kcal/mol of ||delta|| on all 11,955 references for the
      null and mirrored points (the contrast moves references by design; it is re-checked
      orthogonal to the null bases and supported in int22/int21/int11/bulge); 5 carry rows
      per point against theta_A07; the written rounding error recorded; every point's
      written integer perturbation theta_r - theta_A07 re-asserted equal to the Type B fold's
      at its row (theta_r - theta_T04; negated for the mirrored point against row 191);
      and per point whether params_match's 12 spot-check entries see the written change at
      all (params_match_can_see; where they cannot, the carry rows and the Boltzmann carry
      are the binding guards). Saved as shards/points_a07.npz and .json, sha256 recorded.

FOLD (phase run-fold; MFE only; a copy of turner_typeb.worker's guard logic, the original is not
edited). Everywhere the fold used theta0, this uses the base table theta_b (theta_A07):
  baseline inside loaded(a07.par): params_match against theta_A07 and the generic Boltzmann
  check on the first compound; c0 . theta_b + off0 = e0 and c(ref) . theta_b + off_ref =
  eval(ref) exactly on every sequence; MUST HOLD: every sequence's f1_ref and A07 reference
  energy reproduce E3's sha-checked full:Andronescu2007 shards (eref0 + dref_dcal; f1_ref as E3
  rounded it), joined by sequence name.
  per point, inside loaded(write_par(point table)): params_match (a 12-entry spot check: at the
  points recorded params_match_can_see = false it cannot see the table, and the carry rows
  bind), the 5 carry rows, the Boltzmann carry (a pf constrained to the top carry row); per
  sequence the reference-energy
  identity e_ref = ref_e0 + c(ref) . (theta_r - theta_b), the MFE-energy identity
  c1 . theta_r + off1 = e1, genuine = c1 . (theta_b + delta) + off1 <= c0 . (theta_b + delta) +
  off0 + EXACT_TIE, dE0 = c1 . theta_b + off1 - e0 >= 0 (tie flip iff dE0 = 0, real reordering
  iff > 0), f1_ref, f1_base, leak = c(ref) . (theta_r - theta_b).

FISHER (phase run-fisher; E1 at A07; a copy of typeb_fisher.worker's sampler loop, the original is
not edited; typeb_fisher.directions and K reused): every TR0+TS0 sequence, K = 2,000 pbacktrack
samples, md.uniq_ML = 1, inside loaded(a07.par), RNA.init_rand(20261014 + sequence index) per
sequence; params_match against theta_A07 and the generic Boltzmann check on the first compound,
which also reads the sampling compound's own exp_params after pf() (TerminalAU and ML_cc
recovered as -kT ln(x) must equal theta_A07's to 1e-6 dcal, as E1's
assert_turner2004_boltzmann check (1) does for Turner 2004);
MUST HOLD: C . N_int = 0 in integers on every sample; reconstruction c . theta_A07 + o =
eval_structure on the first sample of every 4th sequence per shard. Outputs as E1's: Ix (per
sequence Var[c.v]/kT^2 per direction), covF/covA/covB of z = N_nt^T c (/kT^2), kT.

SMOKE (phase smoke, after preregistration, before any run; recorded under smoke; must pass):
  S1  the fold worker in TURNER MODE (base = the written Turner 2004 template, byte-identical to
      it) reproduces the Type B fold shards at points 5, 7 and 191 on the first 60 sequences:
      baseline e0 and f1_ref; per point changed, genuine, f1_base, f1_ref, leak, exactly. The
      point builder reproduces the fold's theta_r and carry rows at those points. The tie
      split, which the fold shards do not store, is checked against an independent oracle,
      the fold's baseline gap (ViennaRNA subopt under Turner 2004: the energy from the MFE to
      the best other structure, None if above max(GAP_STEPS) = 100 dcal): every changed
      sequence has dE0 >= gap (dE0 >= 100 where gap is None), so dE0 = 0 only where gap = 0.
  S2  the sampler in TURNER MODE, seeded as the gradient run was (RNA.init_rand(GRAD_SEED + i),
      GRAD_SEED = typeb_fisher.GRAD_SEED = 20261013), reproduces the Ix rows of
      runs/2026-10-12_typeb_fisher/shards/gradient/shard0.npz bit for bit for sequences 0, 8, 16
      and 24 (shard 0 of 8).
  S3  planted skip-load (the A07 base stays loaded instead of a null point's table) is caught at
      that point by params_match or a carry row (the stderr line must read
      "GuardFailure: point 0: params_match" or "GuardFailure: point 0: carry row"; a baseline
      failure does not count).
  S4  planted stale Boltzmann factors are caught: at a point by the Boltzmann carry
      ("GuardFailure: point 0: Boltzmann carry row"); in the Fisher worker by the generic
      Boltzmann check ("GuardFailure: base table: generic Boltzmann check"), whose exp_params
      part and constrained-pf part must both fire.
  S5  mini run: all 33 points and the sampler on 12 sequences (2 workers each) in A07 mode, then
      the analysis as a dry run (nothing recorded).

REGISTERED TESTS: see PRED. analyse re-checks PRED and every shard's sha256.

TABLE 2 (the plan's A07 MFE row). The approved plan's Table 2 has null/contrast columns at 1 and
16 kcal/mol and a "component changed % minus control" column, while its A2 section (B9) and
docs/PLAN.md fix A2's points at 4 and 16 kcal/mol and give A2 no +-1 dcal control. This script
follows B9 and the plan of record (no 1 kcal/mol point, no control) and states in PRED
(table2_a07_row_claude) what the A07 row reports: the component cell is the mirrored point's
genuine fraction with its registered CI (for MFE the exact genuine/rounding split stands where
A1 needs the control), rounding share beside it; the 16 kcal/mol cells are A2's; the 1 kcal/mol
cells are empty. Adding 1 kcal/mol points and a control instead is the user's call and must be
made before preregistration.

GATES. build refuses unless the tree is clean outside runs/ (so the lead commits this script
before the build) and refuses if a registration exists. preregister refuses unless the build
passed, its record is committed unmodified, the build's recorded provenance is clean outside
runs/, and the code is the build's; it refuses if a registration exists. "The code is the
build's" means: every repository file this script can import (CODE: the static import closure
of this file over scripts/ and thermo/, computed from the source with ast, so lazy imports are
included; recorded) is committed at the build's git_sha and at HEAD, and git diff between the
two is empty on them; with the clean-tree gate, the working copy is then the build's code.
smoke refuses unless the registration is committed and the code is the build's. run-fold and
run-fisher refuse unless: the status is right; the artifact is tracked, unmodified, and its
committed version carries predictions == PRED; the tree is clean outside runs/; the process is a
session leader (setsid); the smoke passed with clean provenance and the code is both the
build's and the smoke's (run-fisher: also the fold run's). So the lead commits the artifact
after smoke and after run-fold. analyse records whether the code is still the build's.

FAILURE HANDLING. The driver polls its workers; on the first nonzero exit it terminates the
rest, records which (terminated_after_first_failure), and sets "FAILED: <phase>s exited [...]".
A failed run is kept under failed_attempts and the status reset when it is re-run. A driver
that is itself killed leaves the status at RUNNING...: no phase resets that, because the
artifact cannot tell a dead driver from a live one (scripts/predecessor.py's rule). The lead
checks that no driver or worker is alive (fold_run/fisher_run record their pids) and resets the
status by hand, keeping the record under failed_attempts.

Phases: build | preregister | smoke | run-fold | run-fisher | analyse.
Run: .venv/bin/python scripts/typeb_base_a07.py --phase build          (then commit)
     .venv/bin/python scripts/typeb_base_a07.py --phase preregister    (then commit)
     .venv/bin/python scripts/typeb_base_a07.py --phase smoke          (then commit)
     setsid nohup .venv/bin/python scripts/typeb_base_a07.py --phase run-fold > LOG 2>&1 &
     (commit)
     setsid nohup .venv/bin/python scripts/typeb_base_a07.py --phase run-fisher > LOG 2>&1 &
     .venv/bin/python scripts/typeb_base_a07.py --phase analyse
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

from scripts.predecessor import RUNNING_PREFIX  # noqa: E402
from scripts.runmeta import git_provenance  # noqa: E402

RUN_ID = "2026-10-12_typeb_base_a07"
OUT = R / "runs" / RUN_ID
SHARDS = OUT / "shards"  # gitignored (runs/**/shards/); every file's sha256 recorded
SEED = 20261012
N_WORKERS = 6
NICE = 10
LOADER = "params_load_RNA_Andronescu2007"
NORMS = (4.0, 16.0)
FOLD_ROW_OFFSET = {4.0: 5, 16.0: 7}  # the fold's grid index of each norm (0.125 * 2^k)
MIRROR_FOLD_P = 191
N_SHARED_KEYS = 12849
SPECIAL_HP = ("Tetraloop", "Triloop", "Hexaloop")
NINIO_A07 = (50, 300)
B6_N, B6_LEN = 80, (30, 200)
CARRY_ROWS = 5
FISHER_SEED = 20261014
BOOT = 2000
S1_POINTS, S1_N = (5, 7, 191), 60
S2_ROWS = (0, 8, 16, 24)
MINI_N = 12
E3_RUN = R / "runs" / "2026-10-12_typeb_full_tables"
E3_TABLE = "full:Andronescu2007"
REGISTERED = "PRE-REGISTERED, nothing folded or sampled"
RUNNING_FOLD = f"{RUNNING_PREFIX}: A2 fold (Andronescu 2007 base)"
RUNNING_FISHER = f"{RUNNING_PREFIX}: A2 Fisher sampling (Andronescu 2007 base)"
FOLDED = "folded (workers)"
SAMPLED = "sampled (workers)"

PRED = {
    "registered": "2026-10-12, after the build (theta_A07 written on the Turner 2004 template and "
                  "verified, B1-B9; the 33 points fixed in shards/points_a07.npz) and before any "
                  "smoke, fold or sampling; written before any run",
    "question_user": "is the Type B effect specific to Turner 2004? Repeat it with Andronescu "
                     "2007 as theta0",
    "model": "ViennaRNA 2.7.2, dangles 0, special_hp 0, 37 C (thermo.vienna_ref.reference_md)",
    "base_table": "theta_A07 = ViennaRNA's shipped Andronescu 2007 set "
                  "(params_load_RNA_Andronescu2007, read through params_save) written on the "
                  "Turner 2004 template at the 12,849 shared keys (shards/a07.par, sha256 in the "
                  "build record); the non-shared keys are special hairpins, unused at "
                  "special_hp 0. Its reproduction of the shipped set is the build's B1-B9",
    "points": "33 written tables at the fold's directions: the fold's 8 random null "
              "directions at 4 and 16 kcal/mol (its delta_dcal rows 10i+5, 10i+7); the fold's 8 "
              "contrast directions at the same norms (rows 115+10j, 117+10j; reported without "
              "prediction); the mirrored component delta = -delta_dcal[191] = "
              "P_nt(theta_T04 - theta_A07), 21.03 kcal/mol. The directions stay null at A07 "
              "because A and N_nt do not depend on theta",
    "points_written_claude": "each written table carries the fold's written integer "
                             "perturbation, theta_r = theta_A07 + (theta_r - theta_T04 at the "
                             "fold's row), negated for the mirrored point (row 191); every entry "
                             "is a nearest-integer rounding of theta_A07 + delta (|theta_r - "
                             "theta_A07 - delta| <= 0.5 + 1e-9 dcal, asserted). It equals "
                             "rint(theta_A07 + delta) except at half-integer ties, where np.rint "
                             "(half to even) at the A07 base could break the tie the other way "
                             "than the fold did; the fold's tie-break is kept and every such "
                             "entry is recorded in the build, so the written perturbations, "
                             "written errors and leaks are exactly the fold's (negated for the "
                             "mirror)",
    "registered_test_mirrored_user": "the mirrored component's genuine changed-MFE fraction has "
                                     "a 95% CI that excludes 0 (turner_typeb._boot_mean)",
    "registered_test_mirrored_claude": "_boot_mean as the fold uses it: percentile 95% CI, 1000 "
                                       "draws over the 11,970 sequences; rng "
                                       "default_rng(SeedSequence([20261012, 60, 0]))",
    "registered_test_information_user": "the contrast/null information ratio at A07 (E1's "
                                        "statistic and directions) has a 95% CI lower bound > 1 "
                                        "(2000-draw sequence bootstrap)",
    "registered_test_information_claude": "the ratio is mean I over the 8 contrast directions "
                                          "over mean I over the 8 random null directions "
                                          "(typeb_fisher.directions); the CI is the percentile "
                                          "interval of the 2000 sequence-bootstrap draws; rng "
                                          "default_rng(SeedSequence([20261012, 61]))",
    "definitions": {
        "genuine": "for a sequence whose MFE changed (s0 -> s1): c1 . (theta_A07 + delta) + off1 "
                   "<= c0 . (theta_A07 + delta) + off0 + EXACT_TIE (exact unrounded energies; "
                   "turner_typeb.EXACT_TIE = 1e-9 dcal is float round-off, not a tolerance); "
                   "rounding-caused otherwise",
        "tie_split": "dE0 = c1 . theta_A07 + off1 - e0 >= 0 (must hold; s0 is A07's MFE): a "
                     "tie flip iff dE0 = 0, a real reordering iff dE0 > 0",
        "information": "I(v) = sum_x Var_{s~P_A07(.|x)}[c(s).v] / kT^2 at theta_A07, kT from the "
                       "compound's own Boltzmann factors; K = 2,000 pbacktrack samples per "
                       "sequence (md.uniq_ML = 1), ViennaRNA's sampler seeded "
                       "RNA.init_rand(20261014 + sequence index); per-sequence sample variance "
                       "ddof 1",
        "paired_dF1_claude": "mean over sequences of F1(point MFE, reference) - F1(A07 baseline "
                             "MFE, reference), both rounded to 6 places as the fold stores "
                             "them, so an unchanged MFE contributes exactly 0",
        "changed": "the MFE structure under the point table differs from A07's baseline MFE",
    },
    "table2_a07_row_claude": "the plan's Table 2 has null/contrast columns at 1 and 16 kcal/mol "
                             "and a 'component changed % minus control' column; its A2 section "
                             "(B9) and docs/PLAN.md fix A2's points at 4 and 16 kcal/mol with no "
                             "+-1 dcal control, and A2 follows them. The A07 MFE row therefore "
                             "reports: component = the mirrored point's genuine changed-MFE "
                             "fraction with the registered test's CI (for MFE the exact "
                             "genuine/rounding split takes the place of A1's +-1 dcal control, "
                             "which A1 needs because MEA and centroid have no exact split), its "
                             "rounding share beside it; dF1 = the mirrored point's paired dF1; "
                             "null/contrast at 16 kcal/mol = A2's 16 kcal/mol values; the 1 "
                             "kcal/mol cells are empty (not measured at A07; not replaced by the "
                             "4 kcal/mol values, which go to the supplement)",
    "reported_without_prediction_user": "per point: changed (all), genuine and "
                                        "real-reordering fractions, dF1 and rounding share; null "
                                        "means at 4 and 16 kcal/mol beside the Turner fold's; the "
                                        "contrast directions at the same norms; per-direction I "
                                        "at A07, with intrinsic exactly 0 by the integer "
                                        "must-hold",
    "reported_without_prediction_claude": "per-point CIs by _boot_mean (1000 draws, rng "
                                          "SeedSequence([20261012, 64, p, k])); rounding share = "
                                          "(changed - genuine) / changed; leak = c(ref) . "
                                          "(theta_r - theta_A07) as median, 95th percentile and "
                                          "max (kcal/mol, pair-bearing references); the Turner "
                                          "fold's values are its registered experiment_2.points; "
                                          "null minus contrast genuine fraction at 4 and 16 by the "
                                          "fold's amendment-1 bootstrap (2000 draws, one sequence "
                                          "resample shared by both arms, each arm's 8 directions "
                                          "resampled; rng SeedSequence([20261012, 62])); I with "
                                          "CIs from the registered test's draws and Cramer-Rao "
                                          "SEs 1/sqrt(I)",
    "exploratory_user": "paired-by-direction differences, A07 minus Turner base, and the ratio "
                        "of the two information ratios",
    "exploratory_claude": "the fold: genuine changed-MFE fraction per point minus the Turner "
                          "fold's at the same direction and norm (the mirrored point against "
                          "point 191), on the same sequences from the fold's sha-checked shards, "
                          "sequence bootstrap (2000 draws, rng SeedSequence([20261012, 65])); "
                          "E1: I per direction against E1's registered sha-checked shards on the "
                          "same sequences; ratio of ratios (A07 over Turner 2004) with a paired "
                          "sequence bootstrap (2000 draws, rng SeedSequence([20261012, 63]))",
    "must_hold": "fold baseline: c0 . theta_A07 + off0 = e0 and c(ref) . theta_A07 + off_ref = "
                 "eval(ref) on every sequence; every sequence's f1_ref and A07 reference energy "
                 "reproduce E3's sha-checked full:Andronescu2007 shards (eref0 + dref_dcal), "
                 "joined by name. Per point: params_match (spot check; the carry rows bind where "
                 "it cannot see the table, recorded per point as params_match_can_see), 5 carry "
                 "rows, the Boltzmann carry; per sequence the reference-energy identity, the "
                 "MFE-energy identity and dE0 >= 0. Fisher: params_match and the generic "
                 "Boltzmann check on the first compound (including the sampling compound's own "
                 "exp_params: TerminalAU and ML_cc to 1e-6 dcal); C . N_int = 0 in integers on "
                 "every sample; reconstruction c . theta_A07 + o = eval_structure on the first "
                 "sample of every 4th sequence per shard. Any failure stops the run; nothing is "
                 "repaired",
    "planted_controls": "B7 one int22 entry +1 dcal fails B5 on exactly its rows; B8 stale "
                        "Boltzmann factors fail the generic check (constrained pf and "
                        "exp_params); B4 params_match non-empty against Turner 2004; S3 "
                        "skip-load caught at the point (params_match or a carry row, not a "
                        "baseline guard); S4 stale Boltzmann caught at a point (Boltzmann carry) "
                        "and in the sampler (generic check, both its exp_params and its "
                        "constrained-pf parts)",
    "smoke": "S1 Turner-mode fold reproduces the fold shards at points 5, 7, 191 on 60 "
             "sequences, and every changed sequence's dE0 >= the fold's baseline gap (dE0 >= "
             "100 dcal where the gap is above 100); S2 Turner-mode sampler reproduces the "
             "gradient run's Ix rows bit for bit (sequences 0, 8, 16, 24); S3, S4 planted "
             "controls; S5 mini run. Must pass before run-fold",
    "execution": f"{N_WORKERS} niced workers per run (nice {NICE}), sharded by sequence index "
                 f"mod {N_WORKERS}; OMP/OPENBLAS/MKL = 1, CUDA_VISIBLE_DEVICES=''; drivers under "
                 "setsid; about 20 min for the fold (E3's measured MFE rate) and 10-15 min for "
                 "the sampler (E1's)",
    "conservative_choices_claude": "where the plan left a detail open: (1) every theta0 of the "
                                   "fold and of E1 is theta_A07, including the carry rows and "
                                   "the Boltzmann probe (the generating row whose energy A07 "
                                   "changes most against Turner 2004); (2) the Ninio "
                                   "linearisation is asserted exact for every table loaded; "
                                   "(3) the point-level planted controls (S3, S4 at a point) "
                                   "leave the closest stale table loaded, the A07 base, not "
                                   "Turner 2004 (B8 and the sampler's S4 use Turner 2004's "
                                   "factors, the only other table loaded at that stage); (4) "
                                   "the run phases require the committed artifact, so the lead "
                                   "commits after smoke and after the fold; (5) the contrast "
                                   "directions are re-checked "
                                   "orthogonal to both null bases and supported in the four "
                                   "tables at their own norms; (6) the build must run on a "
                                   "tree clean outside runs/, and every later phase requires "
                                   "the code (this script's static import closure over "
                                   "scripts/ and thermo/) to be the build's, the run phases "
                                   "also the smoke's; (7) every point carries the Type B "
                                   "fold's written integer perturbation, keeping its "
                                   "half-integer tie-breaks (points_written_claude), and "
                                   "the build re-asserts the equality; (8) the driver "
                                   "terminates the other "
                                   "workers on the first failure; a RUNNING status left by a "
                                   "killed driver is reset by hand only",
    "seed": SEED,
}


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


def _pred_ok(out):
    return out.get("predictions") == json.loads(json.dumps(PRED))


# ----------------------------------------------------------------------------------
# shared helpers (generic: no Turner 2004 pin)
# ----------------------------------------------------------------------------------
def _ninio_exact(theta, label):
    """The count model linearises MIN2(cap, asym * m) as asym * m up to asymmetry 5 and cap
    from 6 (turner_nullspace.Extractor._ninio); exact for this table iff 5m <= cap <= 6m."""
    from scripts.turner_typeb import GuardFailure
    m, cap = theta[("ninio_m",)], theta[("ninio_max",)]
    if not 5 * m <= cap <= 6 * m:
        raise GuardFailure(f"{label}: Ninio (m, cap) = ({m}, {cap}); the count model is not exact")
    return m, cap


def _generating(special, cols):
    """The 23,741 generating-set rows, their counts (no lxc, so no offsets) and the matrix."""
    from scripts.turner_nullspace import Extractor, build_matrix, generating_set
    gen = generating_set(special)
    ex = Extractor(special, 0, 0)
    cnts = []
    for _, s, d in gen:
        c = ex.counts(s, d)
        assert ("_offset",) not in c, "generating rows have no loops beyond 30"
        cnts.append(c)
    return gen, cnts, build_matrix(cnts, cols)


def _vec_of(theta, cols):
    return np.array([theta[k] for k in cols], dtype=np.int64)


def _constrained_dcal(md, seq, db):
    """Free energy (dcal) of a partition function hard-constrained to the one structure db, on
    a compound with model details md (so it reads the Boltzmann table md's compounds use)."""
    import RNA
    fcc = RNA.fold_compound(seq, md)
    fcc.hc_add_from_db(db.replace(".", "x"), RNA.CONSTRAINT_DB_DEFAULT
                       | RNA.CONSTRAINT_DB_ENFORCE_BP)
    _, ec = fcc.mfe()
    fcc.exp_params_rescale(ec)
    return int(round(fcc.pf()[1] * 100))


EXP_KEYS = (("TerminalAU", ("TerminalAU",)), ("expMLclosing", ("ML_cc",)))
EXP_TOL = 1e-6  # dcal; assert_turner2004_boltzmann's tolerance on -kT ln(x), float round-off


def _exp_params_dcal(fc):
    """TerminalAU and ML closing (dcal) recovered from the compound's own Boltzmann factors as
    -kT ln(x), as assert_turner2004_boltzmann check (1) reads them. Call after pf()."""
    import math
    ep = fc.exp_params
    return {"TerminalAU": -math.log(ep.expTermAU) * ep.kT / 10,
            "expMLclosing": -math.log(ep.expMLclosing) * ep.kT / 10}


def _boltzmann_check(md, gen, G, tb, t_ref, label, fc=None, theta=None):
    """The generic Boltzmann check (no Turner 2004 pin): on the generating row whose energy tb
    changes most against t_ref (against 0 when t_ref is None), eval_structure and the
    constrained partition function must both give counts . tb exactly. With fc (a compound
    after pf()) and theta, also fc's own exp_params must give theta's TerminalAU and ML_cc to
    EXP_TOL. Every part is evaluated before one GuardFailure names all that failed."""
    import RNA

    from scripts.turner_typeb import GuardFailure
    d = G @ (tb - t_ref) if t_ref is not None else G @ tb
    r = int(np.argsort(-np.abs(d), kind="stable")[0])
    want = int(G[r].dot(tb)[0])
    _, s, db = gen[r]
    got_e = int(round(RNA.fold_compound(s, md).eval_structure(db) * 100))
    got_g = _constrained_dcal(md, s, db)
    bad = []
    if got_e != want:
        bad.append(f"eval {got_e} != counts . theta {want} (generating row {r})")
    if got_g != want:
        bad.append(f"constrained pf {got_g} != counts . theta {want} (generating row {r})")
    rec = {"row": r, "dcal": want, "change_vs_reference_dcal": int(d[r])}
    if fc is not None:
        got = _exp_params_dcal(fc)
        off = {f: (theta[k], round(got[f], 6)) for f, k in EXP_KEYS
               if abs(got[f] - theta[k]) > EXP_TOL}
        if off:
            bad.append(f"exp_params {off} (field: (theta, got) dcal)")
        rec["exp_params_dcal"] = {f: round(v, 9) for f, v in got.items()}
    if bad:
        raise GuardFailure(f"{label}: generic Boltzmann check: " + "; ".join(bad))
    return rec


def _params_match_keys(md):
    """The entries turner_par.params_match spot-checks, read off params_match itself: against
    the template with every entry shifted by 10^6 it reports exactly those entries (the model
    details are the reference ones, so nothing else)."""
    import RNA

    from scripts.turner_par import params_match, template
    th = {k: v + 10**6 for k, v in template()["theta"].items()}
    bad = params_match(RNA.fold_compound("GGGGAAACCCC", md), th, 0, 0)
    keys = [k for k in th if repr(k) in bad]
    assert len(keys) == len(bad) == 12, bad
    return keys


def _code_paths():
    """CODE: this script and every repository module it can import, as the static import
    closure over scripts/ and thermo/ (ast over the source, so imports inside functions count;
    package __init__ files included). Paths relative to the repository."""
    import ast
    me = pathlib.Path(__file__).resolve().relative_to(R)
    todo, seen = [me], set()

    def target(name):
        p = pathlib.Path(*name.split("."))
        for f in (p.with_suffix(".py"), p / "__init__.py"):
            if (R / f).exists():
                return f
        return None

    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        for node in ast.walk(ast.parse((R / rel).read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for nm in names:
                parts = nm.split(".")
                if parts[0] not in ("scripts", "thermo"):
                    continue
                for k in range(1, len(parts) + 1):  # the packages on the way count too
                    f = target(".".join(parts[:k]))
                    if f is not None and f not in seen:
                        todo.append(f)
    return sorted(str(p) for p in seen)


@contextlib.contextmanager
def _stale_boltzmann(path, prev, md):
    """PLANTED CONTROL ONLY. Inside, the energy table is `path`'s and the Boltzmann factors for
    model details md are `prev`'s (None: whatever is loaded now): `prev` is loaded with the full
    rebuild, a model-md partition function builds its Boltzmann table, then `path` is loaded
    with the old MFE-only sentinel (turner_par V3). Turner 2004 is restored after."""
    import RNA

    from scripts.turner_par import _rebuild
    from thermo.vienna_ref import assert_turner2004_params, reference_md
    if prev is not None:
        if not RNA.params_load(str(prev)):
            raise RuntimeError(f"ViennaRNA refused {prev}")
        _rebuild()
    fc = RNA.fold_compound("GGGGAAACCCC", md)
    fc.mfe()
    fc.pf()
    if not RNA.params_load(str(path)):
        raise RuntimeError(f"ViennaRNA refused {path}")
    sent = reference_md()
    sent.temperature = 25.0
    RNA.fold_compound("GGGGAAACCCC", sent)
    try:
        yield
    finally:
        RNA.params_load_RNA_Turner2004()
        _rebuild()
        assert_turner2004_params(RNA.fold_compound("GGGGAAACCCC", reference_md()))


@contextlib.contextmanager
def _shipped(loader):
    """ViennaRNA's own shipped set inside; Turner 2004 restored and asserted after."""
    import RNA

    from scripts.turner_par import _rebuild
    from thermo.vienna_ref import assert_turner2004_params, reference_md
    getattr(RNA, loader)()
    _rebuild()
    try:
        yield
    finally:
        RNA.params_load_RNA_Turner2004()
        _rebuild()
        assert_turner2004_params(RNA.fold_compound("GGGGAAACCCC", reference_md()))


def _fold_points_checked():
    """The Type B fold's points.npz/json, sha-checked against its artifact."""
    from scripts.turner_typeb import FOLD
    from scripts.turner_typeb import read_out as typeb_out
    rec = typeb_out()["fold"]["points_sha256"]
    for f, h in rec.items():
        assert _sha(FOLD / f) == h, f"fold {f} changed since the fold"
    return np.load(FOLD / "points.npz"), json.loads((FOLD / "points.json").read_text())


def build_points(t_base, specs, B, G):
    """Written tables for specs [{kind, name, norm_kcal, fold_p, delta_dcal[, dint]}]:
    theta_r = rint(t_base + delta) (the fold's construction), or, with dint (an integer
    perturbation carried from the fold), theta_r = t_base + dint, which must be a
    nearest-integer rounding of t_base + delta (|theta_r - t_base - delta| <= 0.5 + TOL dcal
    per entry; ties are the fold's). With the fold's build_points assertions (copied;
    turner_typeb is not edited): the 5 carry rows against t_base, |A delta| <= TOL ||delta||
    for every non-contrast point, and for contrast points orthogonality to both null bases and
    support in the four tables."""
    from scripts.turner_typeb import NULL_TABLES, TOL, GuardFailure
    cols, used, A, NT, NI = B["cols"], B["used"], B["A"], B["NT"], B["I"]
    n = len(cols)
    mask = np.array([cols[i][0] in NULL_TABLES for i in used])
    inside = np.zeros(n, dtype=bool)
    inside[used[mask]] = True
    meta = []
    TH, DL = np.zeros((len(specs), n), np.int64), np.zeros((len(specs), n))
    for p, sp_ in enumerate(specs):
        kind, s = sp_["kind"], float(sp_["norm_kcal"])
        dl = np.asarray(sp_["delta_dcal"], dtype=float)
        got = float(np.linalg.norm(dl) / 100.0)
        if abs(got - s) > TOL * s:
            raise GuardFailure(f"point {p}: ||delta|| = {got} kcal/mol, expected {s}")
        th_rint = np.rint(t_base + dl).astype(np.int64)
        if sp_.get("dint") is None:
            th = th_rint
        else:
            th = t_base + np.asarray(sp_["dint"], dtype=np.int64)
            worst = float(np.abs(th - t_base - dl).max())
            if worst > 0.5 + TOL:
                raise GuardFailure(f"point {p}: the carried integer perturbation is not a "
                                   f"nearest-integer rounding of delta (max residual {worst})")
        TH[p], DL[p] = th, dl
        ch = G @ (th - t_base)
        top = np.argsort(-np.abs(ch), kind="stable")[:CARRY_ROWS]
        if ch[top[0]] == 0:
            raise GuardFailure(f"point {p}: no generating-set row sees the written change")
        ref_change = A @ (th - t_base)[used]
        null_check = float(np.abs(A @ (dl[used] / 100.0)).max())
        if kind != "contrast" and null_check > TOL * s:
            raise GuardFailure(f"point {p}: {kind} {sp_['name']} is not null on the references "
                               f"(max |A delta| = {null_check:.3g} kcal/mol at ||delta|| = {s})")
        overlap = None
        if kind == "contrast":
            u = dl / np.linalg.norm(dl)
            overlap = max(float(np.abs(NT.T @ u[used]).max()), float(np.abs(NI.T @ u[used]).max()))
            if overlap > TOL:
                raise GuardFailure(f"point {p}: contrast {sp_['name']} overlaps the null space "
                                   f"({overlap:.3g})")
            if np.any(dl[~inside]):
                raise GuardFailure(f"point {p}: contrast {sp_['name']} has support outside "
                                   f"{list(NULL_TABLES)}")
        res = th - t_base - dl
        meta.append({"p": p, "kind": kind, "name": sp_["name"], "norm_kcal": s,
                     "fold_p": sp_["fold_p"],
                     "written_rel_error": float(np.linalg.norm(res) / np.linalg.norm(dl)),
                     "written_max_abs_residual_dcal": float(np.abs(res).max()),
                     "n_entries_changed": int((th != t_base).sum()),
                     "carried_integer_perturbation": sp_.get("dint") is not None,
                     "entries_differing_from_rint": [
                         {"col": int(i), "delta_dcal": float(dl[i]),
                          "written_minus_base": int(th[i] - t_base[i]),
                          "rint_minus_base": int(th_rint[i] - t_base[i])}
                         for i in np.nonzero(th != th_rint)[0]],
                     "reference_change_dcal_max_abs": int(np.abs(ref_change).max()),
                     "max_abs_null_check_kcal": null_check,
                     "null_check_bar_kcal": TOL * s if kind != "contrast" else None,
                     "contrast_max_overlap_with_null_bases": overlap,
                     "carry": [[int(r), int(G[r].dot(th)[0])] for r in top]})
    return meta, TH, DL


# ----------------------------------------------------------------------------------
# build (B1-B9)
# ----------------------------------------------------------------------------------
def build():
    import tempfile

    import RNA

    from scripts.turner_nullspace import INF, theta_from_par
    from scripts.turner_par import loaded, params_match, template, write_par
    from scripts.turner_typeb import NULL_TABLES, dump_set, load_basis, set_delta
    from scripts.typeb_common import ninio_correction
    from thermo.vienna_ref import reference_md

    out = read_out()
    if "predictions" in out:
        raise SystemExit("preregistered: the base table and the points are fixed")
    if git_provenance(R)["git_dirty_excluding_runs"]:
        raise SystemExit("the tree is not clean outside runs/: commit the code before the build "
                         "(preregister requires the build's code)")
    if "build" in out:
        out.setdefault("build_history", []).append(
            {"status": out.get("status"), "files_sha256": out["build"].get("files_sha256"),
             "provenance": out["build"].get("provenance")})
    t_start = time.time()
    SHARDS.mkdir(parents=True, exist_ok=True)
    out.update({"status": "building", "build": {}})
    write(out)
    bld = out["build"]
    T = template()
    t0d = T["theta"]
    B = load_basis("tr0ts0")
    cols, used = B["cols"], B["used"]
    t0 = _vec_of(t0d, cols)
    md = reference_md()

    def done(key, rec):
        bld[key] = rec
        write(out)
        if not rec["passes"]:
            out["status"] = f"BUILD FAILED: {key}"
            write(out)
            print(json.dumps({key: rec}, indent=1, default=str))
        return rec["passes"]

    # B1: dump
    P, shipped_sha = dump_set(LOADER)
    raw = P["theta"]
    absent = [k for k in cols if k not in raw]
    inf_reach = [k for k in cols if k in raw and raw[k] >= INF]
    ninio = (raw[("ninio_m",)], raw[("ninio_max",)])
    m_, cap_ = ninio
    if not done("B1_dump", {
            "loader": LOADER, "shipped_par_sha256": shipped_sha,
            "reachable_columns": len(cols), "reachable_absent": [repr(k) for k in absent],
            "reachable_inf": [repr(k) for k in inf_reach], "ninio": list(ninio),
            "ninio_expected": list(NINIO_A07), "lxc": P["lxc"], "lxc_turner2004": T["lxc"],
            "ninio_linearisation_exact": 5 * m_ <= cap_ <= 6 * m_,
            "passes": (not absent and not inf_reach and ninio == NINIO_A07
                       and P["lxc"] == T["lxc"] and 5 * m_ <= cap_ <= 6 * m_)}):
        return 1

    # B2: key differences
    extra, missing = set(raw) - set(t0d), set(t0d) - set(raw)
    shared = set(raw) & set(t0d)

    def by_table(keys):
        c = {}
        for k in keys:
            c[k[0]] = c.get(k[0], 0) + 1
        return dict(sorted(c.items()))

    differ = [k for k in shared if raw[k] != t0d[k]]
    reach = set(cols)
    try:
        write_par(raw, SHARDS / "refused.par")
        refuses = False
    except KeyError:
        refuses = True
    (SHARDS / "refused.par").unlink(missing_ok=True)
    if not done("B2_keys", {
            "n_shipped_keys": len(raw), "n_template_keys": len(t0d), "n_shared": len(shared),
            "n_shared_expected": N_SHARED_KEYS,
            "extra_in_A07_by_table": by_table(extra), "missing_in_A07_by_table": by_table(missing),
            "shared_differing_by_table": by_table(differ),
            "shared_differing_unreachable_by_table": by_table([k for k in differ
                                                               if k not in reach]),
            "write_par_refuses_the_shipped_keys": refuses,
            "passes": (all(k[0] in SPECIAL_HP for k in extra | missing)
                       and len(shared) == N_SHARED_KEYS and refuses)}):
        return 1

    # B3: write, round trip
    thA = dict(t0d) | {k: raw[k] for k in shared}
    a07 = write_par(thA, SHARDS / "a07.par")
    P2 = theta_from_par(a07)
    tA = _vec_of(thA, cols)
    if not done("B3_round_trip", {
            "a07_par_sha256": _sha(a07),
            "parses_to_template_with_a07_on_shared_keys": P2["theta"] == thA,
            "shared_values_equal_shipped": all(P2["theta"][k] == raw[k] for k in shared),
            "lxc_round_trip": P2["lxc"] == P["lxc"],
            "reachable_columns_changed_vs_turner2004": int((tA != t0).sum()),
            "passes": (P2["theta"] == thA and all(P2["theta"][k] == raw[k] for k in shared)
                       and P2["lxc"] == P["lxc"])}):
        return 1

    # B4: params_match inside and against Turner 2004
    with loaded(a07):
        inside = params_match(RNA.fold_compound("GGGGAAACCCC", md), thA, 0, 0)
    outside = params_match(RNA.fold_compound("GGGGAAACCCC", md), thA, 0, 0)
    if not done("B4_params_match", {
            "inside_loaded_a07": inside, "turner2004_compound_vs_a07": outside,
            "passes": inside == {} and len(outside) > 0}):
        return 1

    # B5: generating set
    gen, cnts, G = _generating(T["special"], cols)

    def energies(theta):
        assert not params_match(RNA.fold_compound(gen[0][1], md), theta, 0, 0)
        return [int(round(RNA.fold_compound(s, md).eval_structure(d) * 100)) for _, s, d in gen]

    want = [sum(thA[k] * v for k, v in c.items()) for c in cnts]
    nc = [ninio_correction(d, m_, cap_) for _, _, d in gen]
    with loaded(a07):
        eA = energies(thA)
    mism = [i for i in range(len(gen)) if eA[i] != want[i] + nc[i]]
    if not done("B5_generating_set", {
            "n_rows": len(gen), "n_mismatch": len(mism), "ninio_correction_nonzero": sum(
                x != 0 for x in nc),
            "first_mismatches": [{"row": i, "vienna": eA[i], "counts_dot_theta": want[i]}
                                 for i in mism[:5]],
            "passes": len(gen) == 23741 and not mism and not any(nc)}):
        return 1

    # B6: the written table reproduces the shipped set
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 50]))
    seqs = ["".join("ACGU"[x] for x in rng.integers(0, 4, int(rng.integers(B6_LEN[0],
                                                                          B6_LEN[1] + 1))))
            for _ in range(B6_N)]
    mdb = reference_md()
    mdb.compute_bpp = 1
    assert mdb.compute_bpp == 1

    def decode(theta, label):
        recs, pm, fc = [], None, None
        for n, s in enumerate(seqs):
            fc = RNA.fold_compound(s, mdb)
            if n == 0:
                pm = params_match(fc, theta, 0, 0)
            st, e = fc.mfe()
            fc.exp_params_rescale(e)
            _, g = fc.pf()
            mea = fc.MEA(1.0)
            cen = fc.centroid()
            recs.append((st, e, g, mea[0], mea[1], cen[0], cen[1]))
        # the last compound's own Boltzmann factors (after its pf) are checked too
        probe = _boltzmann_check(mdb, gen, G, tA, t0, label, fc=fc, theta=theta)
        return recs, pm, probe

    with loaded(a07):
        rw, pm_w, pr_w = decode(thA, "B6 written")
    with _shipped(LOADER):
        rs, pm_s, pr_s = decode(thA, "B6 shipped")
    fields = ("mfe_structure", "mfe_energy", "ensemble_G", "mea_structure", "mea_score",
              "centroid_structure", "centroid_distance")
    ident = {f: sum(a[i] == b[i] for a, b in zip(rw, rs, strict=True))
             for i, f in enumerate(fields)}
    lens = [len(s) for s in seqs]
    if not done("B6_shipped_set", {
            "n_sequences": B6_N, "length_min_max": [min(lens), max(lens)],
            "seed": [SEED, 50], "identical_by_field": ident,
            "params_match_written": pm_w, "params_match_shipped": pm_s,
            "boltzmann_check_written": pr_w, "boltzmann_check_shipped": pr_s,
            "sequences_sha256": hashlib.sha256("\n".join(seqs).encode()).hexdigest(),
            "passes": all(v == B6_N for v in ident.values()) and not pm_w and not pm_s}):
        return 1

    # B7: must fail, one int22 entry +1 dcal
    colnnz = np.asarray((G != 0).sum(axis=0)).ravel()
    used_set = set(int(i) for i in used)
    j = next(i for i, k in enumerate(cols) if k[0] == "int22" and i in used_set and colnnz[i] > 0)
    k7 = cols[j]
    th7 = dict(thA)
    th7[k7] += 1
    with tempfile.TemporaryDirectory() as t:
        with loaded(write_par(th7, pathlib.Path(t) / "b7_planted.par")):
            e7 = energies(th7)
    fail = {i for i in range(len(gen)) if e7[i] != want[i] + nc[i]}
    expect = {i for i, c in enumerate(cnts) if c.get(k7, 0) != 0}
    # every int22 entry is used by exactly one generating row (the set builds each once), so
    # the shift itself is also required: that row's energy rises by its count of the entry
    shift = all(e7[i] - want[i] - nc[i] == cnts[i][k7] for i in fail & expect)
    if not done("B7_planted_int22", {
            "entry": repr(k7), "a07_value": thA[k7], "rows_expected_to_fail": len(expect),
            "rows_failing": len(fail), "same_rows": fail == expect,
            "failing_rows_shift_by_their_count_dcal": shift,
            "passes": fail == expect and len(fail) > 0 and shift}):
        return 1

    # B8: generic Boltzmann check and its must-fail plant
    d = G @ (tA - t0)
    top = np.argsort(-np.abs(d), kind="stable")[:CARRY_ROWS]
    rows8 = []

    def exp_of(s_):
        """A compound's own exp_params after its pf() (dcal), and their misses against A07."""
        fce = RNA.fold_compound(s_, md)
        _, e_ = fce.mfe()
        fce.exp_params_rescale(e_)
        fce.pf()
        got = _exp_params_dcal(fce)
        return ({f: round(v, 9) for f, v in got.items()},
                [f for f, k in EXP_KEYS if abs(got[f] - thA[k]) > EXP_TOL])

    with loaded(a07):
        for r in top:
            _, s, db = gen[int(r)]
            w_ = int(G[int(r)].dot(tA)[0])
            rows8.append({"row": int(r), "counts_dot_theta": w_, "change_vs_turner_dcal": int(d[r]),
                          "constrained_pf": _constrained_dcal(md, s, db),
                          "eval": int(round(RNA.fold_compound(s, md).eval_structure(db) * 100))})
        exp_in, exp_in_miss = exp_of(gen[int(top[0])][1])
    r0 = int(top[0])
    _, s, db = gen[r0]
    with _stale_boltzmann(a07, None, md):
        stale = _constrained_dcal(md, s, db)
        stale_eval = int(round(RNA.fold_compound(s, md).eval_structure(db) * 100))
        exp_st, exp_st_miss = exp_of(s)
    ok8 = all(x["constrained_pf"] == x["counts_dot_theta"] == x["eval"] for x in rows8)
    caught = stale != rows8[0]["counts_dot_theta"]
    want_exp = {f: thA[k] for f, k in EXP_KEYS}
    if not done("B8_boltzmann", {
            "rows": rows8,
            "exp_params_inside_loaded_a07": {"dcal": exp_in, "theta_a07": want_exp,
                                             "missed": exp_in_miss},
            "planted_stale": {
                "row": r0, "constrained_pf": stale, "eval": stale_eval,
                "turner2004_counts_dot_theta": int(G[r0].dot(t0)[0]),
                "a07_counts_dot_theta": rows8[0]["counts_dot_theta"], "caught": caught,
                "exp_params_dcal": exp_st, "exp_params_missed": exp_st_miss,
                "exp_params_caught": len(exp_st_miss) == len(EXP_KEYS)},
            "passes": (ok8 and caught and stale_eval == rows8[0]["counts_dot_theta"]
                       and not exp_in_miss and len(exp_st_miss) == len(EXP_KEYS))}):
        return 1

    # B9: the points
    Zf, meta_f = _fold_points_checked()
    if not np.array_equal(Zf["t0"], t0):
        raise SystemExit("the fold's t0 is not the Turner 2004 template")
    look = {(m["kind"], m["name"], m["norm_kcal"]): m["p"] for m in meta_f}
    specs = []
    for s_ in NORMS:
        for kind, base_row in (("null", 0), ("contrast", 110)):
            for i in range(8):
                p = look[(kind, str(i), s_)]
                if p != base_row + 10 * i + FOLD_ROW_OFFSET[s_]:
                    raise SystemExit(f"fold point {kind} {i}@{s_} is row {p}, not the expected")
                specs.append({"kind": kind, "name": str(i), "norm_kcal": s_, "fold_p": p,
                              "delta_dcal": Zf["delta_dcal"][p],
                              "dint": Zf["theta_r"][p] - Zf["t0"]})
    mf = meta_f[MIRROR_FOLD_P]
    if (mf["p"], mf["kind"], mf["name"]) != (MIRROR_FOLD_P, "published_own", "Andronescu2007"):
        raise SystemExit(f"fold row {MIRROR_FOLD_P} is not Andronescu 2007's own point: {mf}")
    mirror = -Zf["delta_dcal"][MIRROR_FOLD_P]
    NT = B["NT"]
    dX, rec_X, support = set_delta("Andronescu2007", cols, t0d)
    indep = np.zeros(len(cols))
    indep[used] = -100.0 * (NT @ (NT.T @ dX[used]))
    dW = (tA - t0) / 100.0
    indep_w = np.zeros(len(cols))
    indep_w[used] = -100.0 * (NT @ (NT.T @ dW[used]))
    rel = float(np.linalg.norm(mirror - indep) / np.linalg.norm(indep))
    rel_w = float(np.linalg.norm(mirror - indep_w) / np.linalg.norm(indep_w))
    # every point carries the fold's written integer perturbation (negated for the mirrored
    # point), not rint(theta_A07 + delta): np.rint rounds half to even and the mirrored
    # component has entries that are half-integers to float precision, so rint at the A07 base
    # can break a tie the other way than the fold did at Turner 2004's. Either is a
    # nearest-integer rounding of theta_A07 + delta (build_points asserts it); carrying the
    # fold's makes the written perturbations, written errors and leaks the fold's exactly
    specs.append({"kind": "mirrored", "name": "Andronescu2007", "norm_kcal": mf["norm_kcal"],
                  "fold_p": MIRROR_FOLD_P, "delta_dcal": mirror,
                  "dint": -(Zf["theta_r"][MIRROR_FOLD_P] - Zf["t0"])})
    meta, TH, DL = build_points(tA, specs, B, G)
    # asserted independently of the specs: the written integer perturbation is the fold's at
    # the same row (negated for the mirrored point against row 191)
    same_int = {}
    for q, m in enumerate(meta):
        want_int = Zf["theta_r"][m["fold_p"]] - Zf["t0"]
        if m["kind"] == "mirrored":
            want_int = -want_int
        same_int[f"{m['kind']}:{m['name']}@{m['norm_kcal']:g}"] = bool(
            np.array_equal(TH[q] - tA, want_int))
    # where params_match's 12 spot-check entries do not change, it cannot see the point table
    idx_c = {k: i for i, k in enumerate(cols)}
    pm_cols = [idx_c[k] for k in _params_match_keys(md) if k in idx_c]
    for q, m in enumerate(meta):
        m["params_match_can_see"] = bool(np.any(TH[q][pm_cols] != tA[pm_cols]))
    np.savez(SHARDS / "points_a07.npz", theta_r=TH, delta_dcal=DL, t_base=tA)
    (SHARDS / "points_a07.json").write_text(json.dumps(meta))
    nonc = [m for m in meta if m["kind"] != "contrast"]
    if not done("B9_points", {
            "n_points": len(meta),
            "by_kind": {k: sum(m["kind"] == k for m in meta)
                        for k in ("null", "contrast", "mirrored")},
            "fold_rows": [m["fold_p"] for m in meta],
            "mirrored": {"norm_kcal": float(np.linalg.norm(mirror) / 100),
                         "rel_diff_vs_set_delta": rel, "rel_diff_vs_written_a07": rel_w,
                         "set_delta_shared_support": rec_X["shared_support"],
                         "set_delta_support_complete": bool(support.all()),
                         "weight_in_null_tables": float(
                             sum(mirror[i] ** 2 for i, k in enumerate(cols)
                                 if k[0] in NULL_TABLES) / (mirror @ mirror))},
            "null_check_max_over_bar": max(m["max_abs_null_check_kcal"] / m["null_check_bar_kcal"]
                                           for m in nonc),
            "written_rel_error_by_point": {f"{m['kind']}:{m['name']}@{m['norm_kcal']:g}":
                                           m["written_rel_error"] for m in meta},
            "written_max_abs_residual_dcal": max(m["written_max_abs_residual_dcal"]
                                                 for m in meta),
            "integer_perturbation_equals_fold": {
                "n_equal": sum(same_int.values()), "n_points": len(same_int),
                "unequal": [k for k, v in same_int.items() if not v]},
            "entries_where_carried_differs_from_rint_a07": {
                f"{m['kind']}:{m['name']}@{m['norm_kcal']:g}": [
                    e | {"key": repr(cols[e["col"]])} for e in m["entries_differing_from_rint"]]
                for m in meta if m["entries_differing_from_rint"]},
            "params_match_can_see": {
                "n_entries_checked_in_reachable_columns": len(pm_cols),
                "n_points_can_see": sum(m["params_match_can_see"] for m in meta),
                "points_cannot_see": [f"{m['kind']}:{m['name']}@{m['norm_kcal']:g}"
                                      for m in meta if not m["params_match_can_see"]]},
            "passes": (len(meta) == 33 and rel <= 1e-9 and rel_w <= 1e-9
                       and bool(support.all()) and all(same_int.values()))}):
        return 1
    out["build"]["files_sha256"] = {f: _sha(SHARDS / f) for f in
                                    ("a07.par", "points_a07.npz", "points_a07.json")}
    out["build"].update({"all_pass": True, "viennarna_version": RNA.__version__,
                         "seconds": round(time.time() - t_start, 1),
                         "provenance": git_provenance(R), "code_paths": _code_paths()})
    out["status"] = "built"
    write(out)
    print(json.dumps({k: v["passes"] for k, v in bld.items() if k[:1] == "B" and k[1].isdigit()}
                     | {"files_sha256": out["build"]["files_sha256"]}, indent=1))
    return 0


def _check_files(out):
    for f, h in out["build"]["files_sha256"].items():
        assert _sha(SHARDS / f) == h, f"{f} changed since the build"


# ----------------------------------------------------------------------------------
# gates, preregister
# ----------------------------------------------------------------------------------
def _git(*args):
    return subprocess.run(["git", *args], cwd=R, capture_output=True, text=True)


def _committed(out, unmodified):
    """The artifact is tracked and its committed version carries this registration; with
    `unmodified`, the working copy equals the committed one."""
    rel = str((OUT / "metrics.json").relative_to(R))
    if _git("ls-files", "--error-unmatch", rel).returncode != 0:
        raise SystemExit(f"{rel} is not tracked: commit it first")
    if unmodified and _git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"{rel} has uncommitted changes: commit it first")
    return json.loads(_git("show", f"HEAD:{rel}").stdout)


def _gate(out, *, unmodified):
    if not _pred_ok(out):
        raise SystemExit("PRED changed since registration")
    head = _committed(out, unmodified)
    if head.get("predictions") != json.loads(json.dumps(PRED)):
        raise SystemExit("the committed artifact does not carry this registration")
    if git_provenance(R)["git_dirty_excluding_runs"]:
        raise SystemExit("the tree is not clean outside runs/")


def _code_diff(sha):
    """CODE files that are not committed at `sha` or at HEAD, or that differ between the two
    (empty: HEAD carries `sha`'s code; with a clean tree, so does the working copy)."""
    paths = _code_paths()
    bad = [f"{p} (absent at {c[:12]})" for c in (sha, "HEAD") for p in paths
           if _git("cat-file", "-e", f"{c}:{p}").returncode != 0]
    d = _git("diff", "--name-only", sha, "HEAD", "--", *paths)
    if d.returncode != 0:
        return bad + [f"git diff failed: {d.stderr.strip()}"]
    return bad + d.stdout.split()


def _same_code(prov, label):
    """Refuse unless the phase `label` ran on a tree clean outside runs/ and HEAD carries its
    code (the CODE files)."""
    if not prov or prov.get("git_dirty_excluding_runs") is not False:
        raise SystemExit(f"the {label} did not run on a tree clean outside runs/: "
                         f"{(prov or {}).get('git_dirty_paths_outside_runs')}")
    diff = _code_diff(prov["git_sha"])
    if diff:
        raise SystemExit(f"the code is not the {label}'s ({prov['git_sha'][:12]}): {diff}")


def preregister():
    out = read_out()
    if "predictions" in out:
        raise SystemExit("the registration exists")
    assert out.get("status") == "built" and out["build"].get("all_pass"), out.get("status")
    head = _committed(out, unmodified=True)
    if head.get("build", {}).get("files_sha256") != out["build"]["files_sha256"]:
        raise SystemExit("the committed artifact does not carry this build")
    _check_files(out)
    if git_provenance(R)["git_dirty_excluding_runs"]:
        raise SystemExit("the tree is not clean outside runs/: commit the code first")
    _same_code(out["build"].get("provenance"), "build")
    out.update({"status": REGISTERED, "predictions": PRED,
                "registration_provenance": git_provenance(R)})
    write(out)
    return 0


# ----------------------------------------------------------------------------------
# workers
# ----------------------------------------------------------------------------------
def _base(cfg):
    """The base table: its written file (sha-checked), parsed theta and lxc."""
    from scripts.turner_nullspace import theta_from_par
    from scripts.turner_par import template
    path = pathlib.Path(cfg["base"])
    if _sha(path) != cfg["base_sha"]:
        raise SystemExit(f"{path}: sha256 differs from the record")
    P = theta_from_par(path)
    if cfg["mode"] == "turner":
        assert P["theta"] == template()["theta"], "turner mode: the base is not Turner 2004"
    elif cfg["mode"] != "a07":
        raise SystemExit(f"mode {cfg['mode']}")
    _ninio_exact(P["theta"], "base table")
    return path, P["theta"], P["lxc"]


def _e3_reference(mode):
    """{name: (reference energy dcal, f1_ref, digits)} under the base table, from E3's shards
    (sha-checked against its artifact). a07: E3's full:Andronescu2007 record (eref0 + dref_dcal;
    f1_ref as E3 rounded it, 6 places). turner: E3's Turner 2004 baseline (eref0, f1_ref0)."""
    art = json.loads((E3_RUN / "metrics.json").read_text())
    rec = art["run"]["shards_sha256"]
    ref = {}
    for f, h in sorted(rec.items()):
        p = E3_RUN / "shards" / f
        assert _sha(p) == h, f"E3 {f} changed"
        sh = [json.loads(ln) for ln in p.read_text().splitlines()]
        base = sh[0]["baseline"]
        if mode == "a07":
            t = next(r for r in sh if r.get("table") == E3_TABLE)
            for b, f1r, dr in zip(base, t["f1_ref"], t["dref_dcal"], strict=True):
                assert dr is not None, f"{b['name']}: forbidden under A07 in E3"
                ref[b["name"]] = (b["eref0"] + dr, f1r, 6)
        else:
            for b in base:
                ref[b["name"]] = (b["eref0"], b["f1_ref0"], None)
    assert len(ref) == 11970, len(ref)
    return ref


def fold_worker(w, W, dst, cfg):
    """Shard w of W: the base-table baseline once, then every point. JSON lines to dst. A copy
    of turner_typeb.worker's guard logic with theta0 -> the base table (MFE only)."""
    import tempfile

    import RNA

    from scripts.turner_nullspace import Extractor
    from scripts.turner_par import loaded, params_match, template, write_par
    from scripts.turner_typeb import (
        EXACT_TIE,
        GuardFailure,
        _pairs,
        _vec,
        f1,
        fold_rows,
        load_basis,
    )
    from thermo.vienna_ref import reference_md

    T = template()
    special = T["special"]
    base, theta_b, lxc = _base(cfg)
    cols = load_basis("tr0ts0")["cols"]
    idx = {k: i for i, k in enumerate(cols)}
    tb = _vec_of(theta_b, cols)
    pts = pathlib.Path(cfg["points"])
    pj = pts.with_suffix(".json")
    if _sha(pts) != cfg["points_sha"] or _sha(pj) != cfg["points_json_sha"]:
        raise SystemExit("points file sha256 differs from the record")
    Z = np.load(pts)
    meta = json.loads(pj.read_text())
    if not np.array_equal(Z["t_base"], tb):
        raise GuardFailure("the points were built on another base table")
    rows = fold_rows()[:cfg.get("limit")]
    mine = [r for i, r in enumerate(rows) if i % W == w]
    ex = Extractor(special, 0, 0, lxc=lxc)
    gen, _, G = _generating(special, cols)
    t_ref = _vec_of(T["theta"], cols) if cfg["mode"] == "a07" else None
    e3 = _e3_reference(cfg["mode"])
    md = reference_md()
    only = set(cfg["only"]) if cfg.get("only") is not None else None
    fh = open(dst, "w")  # noqa: SIM115

    def emit(rec):
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    base_recs, probe = [], None
    with loaded(base):
        for n, (tag, name, seq, db) in enumerate(mine):
            fc = RNA.fold_compound(seq, md)
            if n == 0:
                bad = params_match(fc, theta_b, 0, 0)
                if bad:
                    raise GuardFailure(f"base table: params_match {bad}")
                probe = _boltzmann_check(md, gen, G, tb, t_ref, "base table")
            s0, e0f = fc.mfe()
            e0 = int(round(e0f * 100))
            c0 = ex.counts(seq, s0)
            off0 = c0.pop(("_offset",), 0)
            i0, v0 = _vec(c0, idx)
            if int(v0 @ tb[i0]) + off0 != e0:
                raise GuardFailure(f"{name}: baseline MFE energy is not counts . theta_b")
            cr = ex.counts(seq, db)
            offr = cr.pop(("_offset",), 0)
            ri, rv = _vec(cr, idx)
            ref_e0 = int(round(fc.eval_structure(db) * 100))
            if int(rv @ tb[ri]) + offr != ref_e0:
                raise GuardFailure(f"{name}: baseline reference energy is not counts . theta_b")
            f1r = f1(s0, db)
            want_e, want_f1, digits = e3[name]
            if ref_e0 != want_e or (round(f1r, digits) if digits else f1r) != want_f1:
                raise GuardFailure(f"{name}: baseline does not reproduce E3 (reference energy "
                                   f"{ref_e0} vs {want_e}, f1_ref {f1r} vs {want_f1})")
            base_recs.append({"tag": tag, "name": name, "seq": seq, "db": db, "s0": s0, "e0": e0,
                              "c0": (i0, v0), "off0": off0, "cr": (ri, rv), "ref_e0": ref_e0,
                              "f1_ref": f1r})
    emit({"baseline": [{k: b[k] for k in ("tag", "name", "e0", "ref_e0", "f1_ref")}
                       | {"len": len(b["seq"]), "ref_pairs": len(_pairs(b["db"]))}
                       for b in base_recs], "boltzmann_probe": probe, "mode": cfg["mode"]})

    with tempfile.TemporaryDirectory() as t:
        for pm in meta:
            p = pm["p"]
            if only is not None and p not in only:
                continue
            th, dl = Z["theta_r"][p], Z["delta_dcal"][p]
            dth = th - tb
            full = dict(theta_b) | {k: int(th[i]) for i, k in enumerate(cols)}
            _ninio_exact(full, f"point {p}")
            path = write_par(full, pathlib.Path(t) / f"p{p}.par")
            ctx = (loaded(base) if cfg.get("skip_load") else
                   _stale_boltzmann(path, base, md) if cfg.get("stale_boltzmann") else
                   loaded(path))
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
                r, want = pm["carry"][0]
                _, s, d = gen[r]
                gc = _constrained_dcal(md, s, d)
                if gc != want:
                    raise GuardFailure(f"point {p}: Boltzmann carry row {r} gives {gc}, the "
                                       f"written table gives {want}")
                rec = {"p": p, "changed": [], "genuine": [], "dE0": [], "f1_base": [],
                       "f1_ref": [], "leak": []}
                for b in base_recs:
                    seq, db = b["seq"], b["db"]
                    ri, rv = b["cr"]
                    leak = int(rv @ dth[ri])
                    e_ref = int(round(RNA.fold_compound(seq, md).eval_structure(db) * 100))
                    if e_ref != b["ref_e0"] + leak:
                        raise GuardFailure(f"point {p}: {b['name']} reference energy {e_ref} != "
                                           f"{b['ref_e0']} + {leak}")
                    s1, e1f = RNA.fold_compound(seq, md).mfe()
                    e1 = int(round(e1f * 100))
                    changed = s1 != b["s0"]
                    i0, v0 = b["c0"]
                    genuine = dE0 = None
                    if not changed:
                        if int(v0 @ th[i0]) + b["off0"] != e1:
                            raise GuardFailure(f"point {p}: {b['name']} MFE energy is not "
                                               f"counts . theta_r (unchanged structure)")
                    else:
                        c1 = ex.counts(seq, s1)
                        off1 = c1.pop(("_offset",), 0)
                        i1, v1 = _vec(c1, idx)
                        if int(v1 @ th[i1]) + off1 != e1:
                            raise GuardFailure(f"point {p}: {b['name']} MFE energy is not "
                                               f"counts . theta_r")
                        ex1 = v1 @ (tb[i1] + dl[i1]) + off1
                        ex0 = v0 @ (tb[i0] + dl[i0]) + b["off0"]
                        genuine = bool(ex1 <= ex0 + EXACT_TIE)
                        dE0 = int(v1 @ tb[i1]) + off1 - b["e0"]
                        if dE0 < 0:
                            raise GuardFailure(f"point {p}: {b['name']} s1 lies below the base "
                                               f"MFE ({dE0} dcal)")
                    rec["changed"].append(changed)
                    rec["genuine"].append(genuine)
                    rec["dE0"].append(dE0)
                    rec["f1_base"].append(round(f1(s1, b["s0"]), 6))
                    rec["f1_ref"].append(round(f1(s1, db), 6))
                    rec["leak"].append(leak)
            emit(rec)
    fh.write(json.dumps({"done": True}) + "\n")
    fh.close()
    return 0


def fisher_worker(w, W, dst, cfg):
    """E1's sampler at the base table (a copy of typeb_fisher.worker's loop; the original is not
    edited). Shard w of W, or the explicit sequence indices cfg['rows']."""
    import RNA
    import scipy.sparse as sp

    from scripts.turner_nullspace import Extractor
    from scripts.turner_par import loaded, params_match, template
    from scripts.turner_typeb import GuardFailure, fold_rows
    from scripts.typeb_fisher import RECON_EVERY, K, directions
    from thermo.vienna_ref import reference_md

    T = template()
    names, D, Nint, NTf, B = directions()
    cols = B["cols"]
    idx = {k: i for i, k in enumerate(cols)}
    base, theta_b, lxc = _base(cfg)
    tb = _vec_of(theta_b, cols)
    ex = Extractor(T["special"], 0, 0, lxc=lxc)
    md = reference_md()
    md.uniq_ML = 1
    rows = fold_rows()
    if cfg.get("rows") is not None:
        mine = [(i, rows[i]) for i in cfg["rows"]]
    else:
        mine = [(i, r) for i, r in enumerate(rows[:cfg.get("limit")]) if i % W == w]
    gen, _, G = _generating(T["special"], cols)
    t_ref = _vec_of(T["theta"], cols) if cfg["mode"] == "a07" else None
    seed_base = int(cfg["seed_base"])
    k_nt = NTf.shape[1]
    Ix = np.zeros((len(mine), D.shape[1]))
    covF, covA, covB = (np.zeros((k_nt, k_nt)) for _ in range(3))
    recon, kT, probe = 0, None, None
    seq_index = []
    ctx = _stale_boltzmann(base, None, md) if cfg.get("stale_boltzmann") else loaded(base)
    with ctx:
        for n, (i, (_, name, seq, _db)) in enumerate(mine):
            fc = RNA.fold_compound(seq, md)
            if n == 0:
                bad = params_match(fc, theta_b, 0, 0)
                if bad:
                    raise GuardFailure(f"base table: params_match {bad}")
            _, e = fc.mfe()
            fc.exp_params_rescale(e)
            fc.pf()
            if n == 0:
                # the sampling compound's own Boltzmann factors too (E1's check (1), generic)
                probe = _boltzmann_check(md, gen, G, tb, t_ref, "base table", fc=fc,
                                         theta=theta_b)
                kT = fc.exp_params.kT / 1000.0  # kcal/mol
            RNA.init_rand(seed_base + i)
            ss = fc.pbacktrack(K)
            assert len(ss) == K, (name, len(ss))
            ri, ci, va = [], [], []
            for r, s in enumerate(ss):
                c = ex.counts(seq, s)
                off = c.pop(("_offset",), 0)
                for key, v in c.items():
                    ri.append(r)
                    ci.append(idx[key])
                    va.append(v)
                if r == 0 and n % RECON_EVERY == 0:
                    cv = np.array([c[k] for k in c], dtype=np.int64)
                    ix = np.array([idx[k] for k in c], dtype=np.int64)
                    want = int(round(fc.eval_structure(s) * 100))
                    if int(cv @ tb[ix]) + off != want:
                        raise GuardFailure(f"reconstruction failed: {name} {s}")
                    recon += 1
            C = sp.csr_matrix((np.array(va, dtype=np.int64), (ri, ci)), shape=(K, len(cols)))
            if np.any(C @ Nint != 0):
                raise GuardFailure(f"must-hold failed: c(s).N != 0 on {name}")
            P = np.asarray(C @ D)
            Ix[n] = P.var(axis=0, ddof=1) / kT**2
            Z = np.asarray(C @ NTf)
            for M_, rows_ in ((covF, slice(None)), (covA, slice(0, K // 2)),
                              (covB, slice(K // 2, K))):
                z = Z[rows_] - Z[rows_].mean(axis=0)
                M_ += (z.T @ z) / (z.shape[0] - 1)
            seq_index.append(i)
    np.savez(dst, Ix=Ix, seq_index=np.array(seq_index), covF=covF / kT**2, covA=covA / kT**2,
             covB=covB / kT**2, names=np.array(names), kT=kT, recon_checked=recon,
             seed_base=seed_base, probe_row=probe["row"], probe_dcal=probe["dcal"],
             mode=cfg["mode"])
    return 0


def _env():
    return dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                CUDA_VISIBLE_DEVICES="")


def _cmd(phase, w, W, dst, cfg):
    return ["nice", "-n", str(NICE), sys.executable, __file__, "--phase", phase, str(w), str(W),
            str(dst), json.dumps(cfg)]


def _launch(phase, dsts, cfg):
    return [subprocess.Popen(_cmd(phase, w, len(dsts), d, cfg), env=_env())
            for w, d in enumerate(dsts)]


def _a07_cfg(out):
    f = out["build"]["files_sha256"]
    return {"mode": "a07", "base": str(SHARDS / "a07.par"), "base_sha": f["a07.par"],
            "points": str(SHARDS / "points_a07.npz"), "points_sha": f["points_a07.npz"],
            "points_json_sha": f["points_a07.json"]}


def _fisher_cfg(out):
    c = _a07_cfg(out)
    return {"mode": "a07", "base": c["base"], "base_sha": c["base_sha"],
            "seed_base": FISHER_SEED}


# ----------------------------------------------------------------------------------
# smoke
# ----------------------------------------------------------------------------------
def _fold_lines(path, ps):
    """A Type B fold shard: its baseline and the records of points ps (line 1 + p)."""
    lines = pathlib.Path(path).read_text().splitlines()
    recs = {p: json.loads(lines[1 + p]) for p in ps}
    assert all(recs[p]["p"] == p for p in ps)
    return json.loads(lines[0])["baseline"], recs


def _fold_shards_by_name(ps):
    """{name: {'baseline': rec, p: {field: value}}} from the Type B fold shards, sha-checked."""
    from scripts.turner_typeb import FOLD
    from scripts.turner_typeb import read_out as typeb_out
    rec = typeb_out()["fold"]["shards_sha256"]
    by = {}
    for f, h in rec.items():
        assert _sha(FOLD / "full" / f) == h, f"fold {f} changed since the fold"
        base, recs = _fold_lines(FOLD / "full" / f, ps)
        for n, b in enumerate(base):
            by[b["name"]] = {"baseline": b} | {
                p: {k: v[n] for k, v in recs[p].items() if isinstance(v, list)} for p in ps}
    return by


def _s1(sm):
    from scripts.turner_par import template, write_par
    from scripts.turner_typeb import GAP_STEPS, load_basis
    T = template()
    tw = write_par(T["theta"], sm / "turner2004.par")
    identical = tw.read_text() == T["text"]
    Zf, meta_f = _fold_points_checked()
    B = load_basis("tr0ts0")
    gen, _, G = _generating(T["special"], B["cols"])
    specs = [{"kind": meta_f[p]["kind"], "name": meta_f[p]["name"],
              "norm_kcal": meta_f[p]["norm_kcal"], "fold_p": p,
              "delta_dcal": Zf["delta_dcal"][p]} for p in S1_POINTS]
    meta, TH, DL = build_points(Zf["t0"], specs, B, G)
    builder = {str(p): bool(np.array_equal(TH[q], Zf["theta_r"][p])
                            and meta[q]["carry"] == meta_f[p]["carry"])
               for q, p in enumerate(S1_POINTS)}
    np.savez(sm / "points_turner.npz", theta_r=TH, delta_dcal=DL, t_base=Zf["t0"])
    (sm / "points_turner.json").write_text(json.dumps(meta))
    cfg = {"mode": "turner", "base": str(tw), "base_sha": _sha(tw),
           "points": str(sm / "points_turner.npz"), "points_sha": _sha(sm / "points_turner.npz"),
           "points_json_sha": _sha(sm / "points_turner.json"), "limit": S1_N}
    dsts = [sm / f"s1_shard{w}.jsonl" for w in range(2)]
    rc = [p.wait() for p in _launch("fold-worker", dsts, cfg)]
    res = {"workers_exit": rc, "written_turner_is_template": identical,
           "builder_reproduces_fold_points": builder, "n_sequences": S1_N,
           "fold_points": list(S1_POINTS)}
    if rc != [0, 0]:
        return res | {"passes": False}
    fold = _fold_shards_by_name(S1_POINTS)
    mine = [[json.loads(ln) for ln in d.read_text().splitlines()] for d in dsts]
    fields = ("changed", "genuine", "f1_base", "f1_ref", "leak")
    mism = {f: 0 for f in ("e0", "f1_ref_baseline", *fields)}
    n_cmp, changed = 0, {str(p): 0 for p in S1_POINTS}
    # the tie split against an independent oracle: the fold's baseline gap (subopt under
    # Turner 2004) bounds the base energy of any other structure, so dE0 >= gap for every
    # changed sequence (dE0 >= max(GAP_STEPS) where no other structure lies within it)
    tie = {"n_changed_checked": 0, "n_dE0_zero": 0, "n_gap_zero": 0, "n_gap_above_steps": 0,
           "violations": []}
    for sh in mine:
        base = sh[0]["baseline"]
        recs = {r["p"]: r for r in sh[1:] if "p" in r}
        for n, b in enumerate(base):
            fb = fold[b["name"]]
            mism["e0"] += b["e0"] != fb["baseline"]["e0"]
            mism["f1_ref_baseline"] += b["f1_ref"] != fb["baseline"]["f1_ref"]
            n_cmp += 1
            for q, p in enumerate(S1_POINTS):
                r = recs[q]
                changed[str(p)] += bool(r["changed"][n])
                for f in fields:
                    mism[f] += r[f][n] != fb[p][f]
                if r["changed"][n]:
                    d_, g_ = r["dE0"][n], fb["baseline"]["gap"]
                    tie["n_changed_checked"] += 1
                    tie["n_dE0_zero"] += d_ == 0
                    tie["n_gap_zero"] += g_ == 0
                    tie["n_gap_above_steps"] += g_ is None
                    if d_ is None or not (d_ >= g_ if g_ is not None else d_ >= max(GAP_STEPS)):
                        tie["violations"].append({"name": b["name"], "fold_point": p,
                                                  "dE0": d_, "gap": g_})
    return res | {"n_compared": n_cmp, "mismatches": mism, "changed_by_fold_point": changed,
                  "tie_split_vs_baseline_gap": tie,
                  "passes": (identical and all(builder.values()) and n_cmp == S1_N
                             and not any(mism.values()) and not tie["violations"])}


def _s2(sm):
    from scripts.typeb_fisher import GRAD_SEED, GRAD_SHARDS
    from scripts.typeb_fisher import read_out as e1_out
    h = e1_out()["exploratory_gradient"]["run"]["shards_sha256"]["shard0.npz"]
    assert _sha(GRAD_SHARDS / "shard0.npz") == h, "the gradient shard changed"
    g = np.load(GRAD_SHARDS / "shard0.npz")
    tw = sm / "turner2004.par"
    cfg = {"mode": "turner", "base": str(tw), "base_sha": _sha(tw), "seed_base": GRAD_SEED,
           "rows": list(S2_ROWS)}
    dst = sm / "s2.npz"
    pr = subprocess.run(_cmd("fisher-worker", 0, 1, dst, cfg), env=_env(), capture_output=True,
                        text=True)
    res = {"exit": pr.returncode, "rows": list(S2_ROWS), "seed_base": GRAD_SEED}
    if pr.returncode != 0:
        return res | {"stderr_tail": pr.stderr.strip().splitlines()[-1:], "passes": False}
    z = np.load(dst)
    pos = [int(np.nonzero(g["seq_index"] == i)[0][0]) for i in S2_ROWS]
    same_names = [str(x) for x in z["names"]] == [str(x) for x in g["names"]]
    bitwise = z["Ix"].tobytes() == g["Ix"][pos].tobytes()
    return res | {"names_equal": same_names, "kT_equal": float(z["kT"]) == float(g["kT"]),
                  "Ix_bitwise_equal": bitwise,
                  "max_abs_diff": float(np.abs(z["Ix"] - g["Ix"][pos]).max()),
                  "passes": same_names and bitwise and float(z["kT"]) == float(g["kT"])}


def _plant(phase, cfg, dst, any_of, all_of=()):
    """A planted control must stop the worker with a GuardFailure whose message (the last
    stderr line; the traceback's source lines quote every guard) contains one of `any_of`
    (each a full 'GuardFailure: <where>: <guard>' prefix, so a failure elsewhere, e.g. in the
    baseline, does not count) and every one of `all_of`."""
    pr = subprocess.run(_cmd(phase, 0, 1, dst, cfg), env=_env(), capture_output=True, text=True)
    last = (pr.stderr.strip().splitlines() or [""])[-1]
    hit = [x for x in any_of if x in last]
    parts = {x: x in last for x in all_of}
    return {"exit": pr.returncode, "caught_by": hit, "parts_fired": parts,
            "stderr_last_line": last,
            "caught": pr.returncode != 0 and bool(hit) and all(parts.values())}


def smoke():
    out = read_out()
    assert str(out.get("status", "")).startswith("PRE-REGISTERED"), out.get("status")
    _gate(out, unmodified=False)
    _same_code(out["build"].get("provenance"), "build")
    _check_files(out)
    if "smoke" in out and not out["smoke"].get("passes"):
        out.setdefault("smoke_failed_attempts", []).append(out["smoke"])
    sm = SHARDS / "smoke"
    sm.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {"S1_turner_fold": _s1(sm), "S2_turner_sampler": _s2(sm)}
    a07 = _a07_cfg(out)
    p0 = json.loads((SHARDS / "points_a07.json").read_text())[0]
    assert (p0["kind"], p0["name"], p0["norm_kcal"]) == ("null", "0", 4.0), p0
    res["S3_planted_skip_load"] = _plant(
        "fold-worker", a07 | {"limit": 10, "only": [0], "skip_load": True}, sm / "s3.jsonl",
        ("GuardFailure: point 0: params_match", "GuardFailure: point 0: carry row"))
    res["S4_planted_stale_boltzmann_point"] = _plant(
        "fold-worker", a07 | {"limit": 10, "only": [0], "stale_boltzmann": True},
        sm / "s4.jsonl", ("GuardFailure: point 0: Boltzmann carry row",))
    res["S4_planted_stale_boltzmann_sampler"] = _plant(
        "fisher-worker", _fisher_cfg(out) | {"rows": [0, 1], "stale_boltzmann": True},
        sm / "s4.npz", ("GuardFailure: base table: generic Boltzmann check",),
        all_of=("exp_params", "constrained pf"))
    mf, mi = sm / "mini_fold", sm / "mini_fisher"
    mf.mkdir(exist_ok=True)
    mi.mkdir(exist_ok=True)
    rcf = [p.wait() for p in _launch("fold-worker", [mf / f"shard{w}.jsonl" for w in range(2)],
                                     a07 | {"limit": MINI_N})]
    rci = [p.wait() for p in _launch("fisher-worker", [mi / f"shard{w}.npz" for w in range(2)],
                                     _fisher_cfg(out) | {"limit": MINI_N})]
    s5 = {"fold_workers_exit": rcf, "fisher_workers_exit": rci}
    if rcf == [0, 0] and rci == [0, 0]:
        r = analyse(fold_src=mf, fisher_src=mi, n_workers=2, record=False, expect_n=MINI_N)
        s5 |= {"n_points_analysed": len(r["fold"]["points"]),
               "registered_tests_dry_run": r["registered_tests"], "passes": True}
    else:
        s5["passes"] = False
    res["S5_mini"] = s5
    passes = (res["S1_turner_fold"]["passes"] and res["S2_turner_sampler"]["passes"]
              and res["S3_planted_skip_load"]["caught"]
              and res["S4_planted_stale_boltzmann_point"]["caught"]
              and res["S4_planted_stale_boltzmann_sampler"]["caught"] and s5["passes"])
    out["smoke"] = res | {"passes": bool(passes), "seconds": round(time.time() - t0, 1),
                          "provenance": git_provenance(R)}
    write(out)
    print(json.dumps(out["smoke"], indent=1, default=str))
    return 0 if passes else 1


# ----------------------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------------------
POLL_S = 2.0


def _wait_all(procs):
    """Wait for every worker; on the first nonzero exit terminate the ones still running (a
    guard failure in one shard stops the run, so the rest is wasted CPU). Returns the exit codes
    and the indices terminated."""
    rc, killed = [None] * len(procs), []
    while any(r is None for r in rc):
        rc = [p.poll() for p in procs]
        if any(r not in (None, 0) for r in rc):
            for k, p in enumerate(procs):
                if p.poll() is None:
                    p.terminate()
                    killed.append(k)
            rc = [p.wait() for p in procs]
            break
        if any(r is None for r in rc):
            time.sleep(POLL_S)
    return rc, killed


def _drive(out, key, running, done_status, phase, dsts, cfg, done_check):
    procs = _launch(phase, dsts, cfg)
    out.update({"status": running, key: {"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                         "driver_pid": os.getpid(),
                                         "worker_pids": [p.pid for p in procs],
                                         "load_at_start": os.getloadavg(),
                                         "n_workers": len(dsts), "nice": NICE,
                                         "provenance": git_provenance(R)}})
    write(out)
    rc, killed = _wait_all(procs)
    done = [d.exists() and done_check(d) for d in dsts]
    out = read_out()
    out[key].update({"workers_exit": rc, "shards_done": done,
                     "terminated_after_first_failure": killed,
                     "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "shards_sha256": {d.name: _sha(d) for d in dsts if d.exists()}})
    ok = rc == [0] * len(dsts) and all(done)
    out["status"] = done_status if ok else f"FAILED: {phase}s exited {rc}"
    write(out)
    return 0 if ok else 1


def run_fold():
    out = read_out()
    if os.getsid(0) != os.getpid():
        raise SystemExit("launch under setsid")
    _gate(out, unmodified=True)
    st = str(out.get("status", ""))
    if st.startswith("FAILED: fold-worker"):
        out.setdefault("failed_attempts", []).append({"status": st, **out.get("fold_run", {})})
        out["status"] = REGISTERED
    assert out["status"].startswith("PRE-REGISTERED"), out["status"]
    assert out.get("smoke", {}).get("passes"), "the smoke test has not passed"
    _same_code(out["build"].get("provenance"), "build")
    _same_code(out["smoke"].get("provenance"), "smoke")
    _check_files(out)
    d = SHARDS / "fold"
    d.mkdir(parents=True, exist_ok=True)
    dsts = [d / f"shard{w}.jsonl" for w in range(N_WORKERS)]
    return _drive(out, "fold_run", RUNNING_FOLD, FOLDED, "fold-worker", dsts, _a07_cfg(out),
                  lambda p: p.read_text().rstrip().endswith('{"done": true}'))


def run_fisher():
    out = read_out()
    if os.getsid(0) != os.getpid():
        raise SystemExit("launch under setsid")
    _gate(out, unmodified=True)
    st = str(out.get("status", ""))
    if st.startswith("FAILED: fisher-worker"):
        out.setdefault("failed_attempts", []).append({"status": st, **out.get("fisher_run", {})})
        out["status"] = FOLDED
    assert out["status"] == FOLDED, out["status"]
    assert out.get("smoke", {}).get("passes"), "the smoke test has not passed"
    _same_code(out["build"].get("provenance"), "build")
    _same_code(out["smoke"].get("provenance"), "smoke")
    _same_code(out["fold_run"].get("provenance"), "fold run")
    _check_files(out)
    d = SHARDS / "fisher"
    d.mkdir(parents=True, exist_ok=True)
    dsts = [d / f"shard{w}.npz" for w in range(N_WORKERS)]
    return _drive(out, "fisher_run", RUNNING_FISHER, SAMPLED, "fisher-worker", dsts,
                  _fisher_cfg(out), lambda p: True)


# ----------------------------------------------------------------------------------
# analyse
# ----------------------------------------------------------------------------------
def _ci(x):
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def analyse(fold_src=None, fisher_src=None, n_workers=N_WORKERS, record=True, expect_n=11970):
    from scripts.turner_typeb import _boot_mean
    from scripts.turner_typeb import read_out as typeb_out
    from scripts.typeb_fisher import OUT as E1_OUT
    from scripts.typeb_fisher import SHARDS as E1_SHARDS
    from scripts.typeb_fisher import K
    from scripts.typeb_fisher import read_out as e1_out

    out = read_out()
    if record:
        assert out["status"] in (SAMPLED, "analysed"), out["status"]
        for key, sub, ext in (("fold_run", "fold", "jsonl"), ("fisher_run", "fisher", "npz")):
            rec = out[key]["shards_sha256"]
            assert set(rec) == {f"shard{w}.{ext}" for w in range(n_workers)}, rec
            for f, h in rec.items():
                assert _sha(SHARDS / sub / f) == h, f"{sub}/{f} changed"
    if not _pred_ok(out):
        raise SystemExit("the artifact's predictions differ from PRED; nothing is analysed")
    _check_files(out)
    fold_src = SHARDS / "fold" if fold_src is None else pathlib.Path(fold_src)
    fisher_src = SHARDS / "fisher" if fisher_src is None else pathlib.Path(fisher_src)
    meta = json.loads((SHARDS / "points_a07.json").read_text())

    # ---- fold
    sh = [[json.loads(ln) for ln in (fold_src / f"shard{w}.jsonl").read_text().splitlines()]
          for w in range(n_workers)]
    assert all(s[-1] == {"done": True} for s in sh), "a fold shard is incomplete"
    base = [b for s in sh for b in s[0]["baseline"]]
    names = [b["name"] for b in base]
    n = len(base)
    assert n == expect_n and len(set(names)) == n, n
    e3 = _e3_reference("a07")
    e3_ok = all(b["ref_e0"] == e3[b["name"]][0] and round(b["f1_ref"], 6) == e3[b["name"]][1]
                for b in base)
    assert e3_ok, "the baseline does not reproduce E3 (the workers' must-hold)"
    # paired against the baseline rounded as the points' f1_ref are (6 places, Python round),
    # so an unchanged MFE contributes exactly 0 to dF1
    f0 = np.array([round(b["f1_ref"], 6) for b in base])
    pb = np.array([b["ref_pairs"] > 0 for b in base])
    per, rows = {}, []
    for pm in meta:
        p = pm["p"]
        rs = [next(r for r in s if r.get("p") == p) for s in sh]
        cat = {k: [x for r in rs for x in r[k]] for k in ("changed", "genuine", "dE0",
                                                          "f1_base", "f1_ref", "leak")}
        ch = np.array(cat["changed"])
        gen = np.array([g is True for g in cat["genuine"]])
        dE0 = np.array([-1 if x is None else x for x in cat["dE0"]])
        real = ch & (dE0 > 0)
        tie = ch & (dE0 == 0)
        dF1 = np.array(cat["f1_ref"]) - f0
        leak = np.abs(np.array(cat["leak"])[pb]) / 100.0

        def rng(k, p=p):
            return np.random.default_rng(np.random.SeedSequence([SEED, 64, p, k]))

        row = {k: pm[k] for k in ("p", "kind", "name", "norm_kcal", "fold_p",
                                  "written_rel_error", "n_entries_changed",
                                  "params_match_can_see",
                                  "reference_change_dcal_max_abs")} | {
            "n_changed": int(ch.sum()), "n_genuine": int(gen.sum()),
            "n_tie_flips": int(tie.sum()), "n_real_reorderings": int(real.sum()),
            "changed_fraction": float(ch.mean()), "changed_ci95": _boot_mean(ch, rng(0)),
            "genuine_fraction": float(gen.mean()), "genuine_ci95": _boot_mean(gen, rng(1)),
            "real_reordering_fraction": float(real.mean()),
            "real_reordering_ci95": _boot_mean(real, rng(2)),
            "rounding_share_of_changed": (float((ch.sum() - gen.sum()) / ch.sum())
                                          if ch.sum() else None),
            "paired_dF1": float(dF1.mean()), "paired_dF1_ci95": _boot_mean(dF1, rng(3)),
            "f1_base_mean": float(np.mean(cat["f1_base"])),
            "leak_kcal": {"median": float(np.median(leak)), "p95": float(np.percentile(leak, 95)),
                          "max": float(leak.max())}}
        rows.append(row)
        per[p] = (row, ch, gen)
    mir = next(m["p"] for m in meta if m["kind"] == "mirrored")
    t1 = _boot_mean(per[mir][2], np.random.default_rng(np.random.SeedSequence([SEED, 60, 0])))
    test1 = {"rule": {"user": PRED["registered_test_mirrored_user"],
                      "claude": PRED["registered_test_mirrored_claude"]},
             "genuine_fraction": per[mir][0]["genuine_fraction"], "ci95": t1,
             "passes": bool(t1[0] > 0 or t1[1] < 0)}
    # null and contrast means beside the Turner fold's registered values
    tb_pts = {(r["kind"], r["name"], r["norm_kcal"]): r
              for r in typeb_out()["experiment_2"]["points"]}
    means, nvc = {}, {}
    rng_b = np.random.default_rng(np.random.SeedSequence([SEED, 62]))
    for s in NORMS:
        means[f"{s:g}"] = {}
        for kind in ("null", "contrast"):
            rr = [r for r in rows if r["kind"] == kind and r["norm_kcal"] == s]
            tt = [tb_pts[(kind, r["name"], s)] for r in rr]
            means[f"{s:g}"][kind] = {
                f"{m}_mean_a07": float(np.mean([r[m] for r in rr])) for m in
                ("changed_fraction", "genuine_fraction", "real_reordering_fraction",
                 "paired_dF1")} | {
                f"{m}_mean_turner_fold": float(np.mean([t[m] for t in tt])) for m in
                ("changed_fraction", "genuine_fraction")} | {"n_directions": len(rr)}
        Gn = np.stack([per[m["p"]][2] for m in meta if m["kind"] == "null"
                       and m["norm_kcal"] == s]).astype(float)
        Gc = np.stack([per[m["p"]][2] for m in meta if m["kind"] == "contrast"
                       and m["norm_kcal"] == s]).astype(float)
        diffs = np.empty(BOOT)
        for b in range(BOOT):
            si = rng_b.integers(0, n, n)
            dn = rng_b.integers(0, len(Gn), len(Gn))
            dc = rng_b.integers(0, len(Gc), len(Gc))
            diffs[b] = Gn[dn][:, si].mean() - Gc[dc][:, si].mean()
        ci = _ci(diffs)
        nvc[f"{s:g}"] = {"difference": float(Gn.mean() - Gc.mean()), "ci95": ci,
                         "side": "below" if ci[1] < 0 else "above" if ci[0] > 0
                         else "contains_0"}
    # exploratory: paired by direction and sequence against the Turner fold's shards
    fps = sorted({m["fold_p"] for m in meta})
    fold_by = _fold_shards_by_name(fps)
    bi = np.random.default_rng(np.random.SeedSequence([SEED, 65])).integers(0, n, (BOOT, n))
    paired = {}
    for m in meta:
        gT = np.array([fold_by[nm][m["fold_p"]]["genuine"] is True for nm in names], dtype=float)
        dlt = per[m["p"]][2].astype(float) - gT
        paired[f"{m['kind']}:{m['name']}@{m['norm_kcal']:g}"] = {
            "fold_p": m["fold_p"], "genuine_a07": float(per[m["p"]][2].mean()),
            "genuine_turner_same_sequences": float(gT.mean()),
            "difference": float(dlt.mean()), "difference_ci95": _ci(dlt[bi].mean(axis=1))}
    del bi, fold_by

    # ---- Fisher
    fz = [np.load(fisher_src / f"shard{w}.npz") for w in range(n_workers)]
    fnames = [str(x) for x in fz[0]["names"]]
    assert all([str(x) for x in z["names"]] == fnames for z in fz)
    Ix = np.concatenate([z["Ix"] for z in fz])
    sidx = np.concatenate([z["seq_index"] for z in fz])
    kTs = {float(z["kT"]) for z in fz}
    assert len(kTs) == 1 and len(set(sidx.tolist())) == len(sidx) == expect_n
    assert {str(z["mode"]) for z in fz} == {"a07"}
    assert {int(z["seed_base"]) for z in fz} == {FISHER_SEED}
    Itot = Ix.sum(axis=0)
    bf = np.random.default_rng(np.random.SeedSequence([SEED, 61])).integers(
        0, len(sidx), (BOOT, len(sidx)))
    Ib = np.stack([Ix[b].sum(axis=0) for b in bf])
    del bf
    grp = {g: [j for j, nm in enumerate(fnames) if nm.startswith(g + ":")]
           for g in ("intrinsic", "null", "published", "contrast", "coverage")}
    ratio = Ib[:, grp["contrast"]].mean(axis=1) / Ib[:, grp["null"]].mean(axis=1)
    r_hat = float(Itot[grp["contrast"]].mean() / Itot[grp["null"]].mean())
    r_ci = _ci(ratio)
    test2 = {"rule": {"user": PRED["registered_test_information_user"],
                      "claude": PRED["registered_test_information_claude"]},
             "contrast_over_null_mean_I": r_hat, "ci95": r_ci, "passes": bool(r_ci[0] > 1)}
    perI = {}
    for j, nm in enumerate(fnames):
        lo, hi = np.percentile(Ib[:, j], [2.5, 97.5])
        perI[nm] = {"I": float(Itot[j]), "ci95": [float(lo), float(hi)],
                    "cr_se_kcal": float(1 / np.sqrt(Itot[j])) if Itot[j] > 0 else None}
    for j in grp["intrinsic"]:
        perI[fnames[j]] = {"I_float": perI[fnames[j]]["I"], "I_exact": 0,
                           "I_exact_basis": "c(s).N = 0 in integers on every sample (worker "
                                            "must-hold)", "cr_se_kcal": None}
    # exploratory: E1 (Turner 2004) on the same sequences, paired
    e1 = e1_out()
    for f, h in e1["run"]["shards_sha256"].items():
        assert _sha(E1_SHARDS / f) == h, f"E1 {f} changed"
    ez = [np.load(E1_SHARDS / f) for f in sorted(e1["run"]["shards_sha256"])]
    assert all([str(x) for x in z["names"]] == fnames for z in ez)
    eI = np.concatenate([z["Ix"] for z in ez])
    epos = {int(i): k for k, i in enumerate(np.concatenate([z["seq_index"] for z in ez]))}
    IxT = eI[[epos[int(i)] for i in sidx]]
    ItT = IxT.sum(axis=0)
    bp = np.random.default_rng(np.random.SeedSequence([SEED, 63])).integers(
        0, len(sidx), (BOOT, len(sidx)))
    rr_b = np.empty(BOOT)
    for k, b in enumerate(bp):
        a_, t_ = Ix[b].sum(axis=0), IxT[b].sum(axis=0)
        rr_b[k] = ((a_[grp["contrast"]].mean() / a_[grp["null"]].mean())
                   / (t_[grp["contrast"]].mean() / t_[grp["null"]].mean()))
    del bp
    r_T = float(ItT[grp["contrast"]].mean() / ItT[grp["null"]].mean())
    explo_I = {nm: {"I_a07": float(Itot[j]), "I_turner_same_sequences": float(ItT[j]),
                    "difference": float(Itot[j] - ItT[j]),
                    "ratio": float(Itot[j] / ItT[j]) if ItT[j] > 0 else None}
               for j, nm in enumerate(fnames) if not nm.startswith("intrinsic:")}

    res = {
        "n_sequences": n,
        "registered_tests": {"mirrored_genuine_fraction": test1,
                             "information_ratio": test2},
        "fold": {"baseline_f1_ref_mean_a07": float(f0.mean()),
                 "baseline_reproduces_e3": e3_ok, "points": rows,
                 "null_and_contrast_means_beside_turner_fold": means,
                 "null_minus_contrast_genuine": nvc},
        "fisher": {"kT_kcal": kTs.pop(), "K": K, "per_direction": dict(sorted(perI.items())),
                   "reconstruction_checked": int(sum(int(z["recon_checked"]) for z in fz)),
                   "intrinsic_exactly_zero": True,
                   "intrinsic_exactly_zero_basis": "the integer must-hold; the run reaches "
                                                   f"'{SAMPLED}' only if it held"},
        "exploratory": {
            "label": "EXPLORATORY (registered as exploratory in PRED)",
            "fold_paired_vs_turner": paired,
            "information_per_direction_vs_turner": explo_I,
            "information_ratio_turner_same_sequences": r_T,
            "information_ratio_registered_e1": e1["results"]["registered_test"],
            "ratio_of_ratios_a07_over_turner": r_hat / r_T,
            "ratio_of_ratios_ci95": _ci(rr_b)},
        "inputs": {"e1_artifact": str(E1_OUT.relative_to(R)),
                   "e3_artifact": str(E3_RUN.relative_to(R))},
    }
    if not record:
        return res
    code_diff = _code_diff(out["build"]["provenance"]["git_sha"])
    out["results"] = res | {"provenance": git_provenance(R),
                            "code_is_the_builds": not code_diff, "code_diff_vs_build": code_diff}
    out["status"] = "analysed"
    write(out)
    print(json.dumps({"registered_tests": res["registered_tests"],
                      "null_minus_contrast": nvc,
                      "ratio_of_ratios": [res["exploratory"]["ratio_of_ratios_a07_over_turner"],
                                          res["exploratory"]["ratio_of_ratios_ci95"]]},
                     indent=1))
    return 0


PHASES = {"build": build, "preregister": preregister, "smoke": smoke, "run-fold": run_fold,
          "run-fisher": run_fisher, "analyse": analyse}

if __name__ == "__main__":
    ph = sys.argv[sys.argv.index("--phase") + 1]
    if ph in ("fold-worker", "fisher-worker"):
        a = sys.argv[sys.argv.index(ph) + 1:]
        fn = fold_worker if ph == "fold-worker" else fisher_worker
        sys.exit(fn(int(a[0]), int(a[1]), a[2], json.loads(a[3])))
    if ph not in PHASES:
        raise SystemExit(f"usage: typeb_base_a07.py --phase {{{','.join(PHASES)}}}")
    sys.exit(PHASES[ph]())
