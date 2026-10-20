# The advisor's additions of 2026-10-20: two post hoc items

Specified on 2026-10-20, before any code for either item, at the user's decision of the same day
to lift the analysis freeze for these two items only (docs/PLAN.md). Both are POST HOC; every
number from them is labeled post hoc where it appears.

**Deadline and drop rule.** Results must be committed by 2026-10-22, end of day. An item whose
checks fail, or that misses the deadline, is dropped, and nothing about it enters the paper. The
abstract changes only as the advisor's item 3 says (a text change, not specified here).

**Item 1a (a records check, already done).** "2.066 against 0.062" and "decay +1.39 vs optimizer
-2.92" are NET SIGNED changes in v.w. In runs/2026-10-11_mxfold2_weight_decay_ablation,
scripts/mxfold2_wd_ablation.py sums signed per-step contributions:
- decay's is -lr * wd * (v.w before the step), which is exact for torch's decoupled AdamW decay;
- the Adam step's is the rest of each step's change.

The two sums add to the change in v.w between the endpoints:
- one epoch, arm (a): -1.068 to 1.060;
- ten epochs, analysis_c: -1.068 to -2.596.

Both are single-seed figures.

## Item 1b: the net split per seed

**Data.** runs/2026-10-12_mxfold2_ablation, arm adamw_wd0.01: AdamW, lr 1e-3, l2 0.01, seeds
20261006, 20261007 and 20261008, ten epochs each. Its runner (scripts/mxfold2_ablation_train.py)
logged v.w before and after every optimiser step, in shards/<run>/gauge/vw-epoch-<k>.npz (arrays
pre, raw, post, start). These logs exist for all ten epochs of all three seeds, so no rerun is
needed and none is made.

**Computation.** For every step t, with lr = 1e-3 and wd = 0.01 as registered for the arm:
- decay's contribution is -lr * wd * pre_t;
- the Adam step's contribution is (post_t - pre_t) minus decay's contribution.

These are the definitions of scripts/mxfold2_wd_ablation.py's arm (a). Both are summed over epoch
1, and over epochs 1 to 10, per seed. Also reported: the net change in v.w over each horizon, and
the ratio of the two contributions.

**Checks (any failure drops item 1b).**
1. Every logged value is finite.
2. Within each epoch, pre_{t+1} = post_t for every step; raw = post, since the arm projects
   nothing.
3. Each epoch's first pre and last post equal the gauge record's v_dot_w_start and v_dot_w for
   that epoch, and each epoch's start equals the previous epoch's last post.
4. The two contributions sum to the endpoint change to 1e-9.
5. Every logs file used is one the run's record lists, with the recorded sha256 (where the
   record lists it).

**Reading, fixed now.**
- If the Adam step's net contribution exceeds decay's in magnitude in every seed, at BOTH
  horizons (one epoch and ten epochs), Section 4 and the introduction replace "33 times further"
  with the net figures.
- Otherwise they drop the comparison with decay and keep only the no-decay arm: the SD of v.w
  from 1.04 to 4.97 with no decay.
- Post hoc label either way.

## Item 2: the Type B dimension against the number of structures

Linear algebra only; nothing is folded.

**Function and settings.** scripts/turner_nullspace_rfam.analyse, imported unchanged, with the
same parameter vector theta_from_par(dump_params()), at dangles 0 and special hairpins off. That
is the function, and those are the tolerances, of the registered runs that gave 426 (TR0+TS0,
runs/2026-10-08_turner_nullspace_tr0) and 214 (Rfam 15.1, runs/2026-10-07_turner_nullspace_rfam_r2).

**Population.** TR0+TS0's 11,955 distinct pair-bearing rows, built exactly as the 426 run built
them: tr0_ts0_rows, distinct_rows, pair-bearing filter.

**Subsamples.**
- Sizes 1,000, 2,000, 4,000 and 8,000, with 8 seeds each.
- Each draw is uniform without replacement, from
  rng = default_rng(SeedSequence([20261007, 92, size, k])) for k = 0 to 7.
- analyse gets its rng from SeedSequence([20261007, 92, size, k, 1]).
- **Seed tag 92 is new.**

**Anchors and the union.**
- The full set (11,955) must reproduce 426.
- The Rfam 15.1 rows alone (load_rows, distinct pair-bearing, as the 214 run built them) must
  reproduce 214.
- The union is TR0+TS0's rows and the Rfam rows, distinct by (sequence, structure).

**Checks (any failure drops item 2).**
1. For every analysis, the dimension methods agree: QR/SVD, Gram eigendecomposition, reduced
   eigendecomposition and exact rank mod p, as analyse's "agree" flag.
2. The per-row reconstruction has no failures, and control C2 passes.
3. The full set reproduces 426, and the Rfam rows reproduce 214.

**Reading, fixed now.**
- Report the curve (each size's 8 dimensions, their mean and range) and the union's dimension N
  with its M distinct structures.
- Section 3 gets one sentence: "Within the data the Type B dimension does not reach 0: N
  directions remain for the union of M structures (post hoc)." Beside it, the statement that by
  Proposition 1 enough distinct structures would leave only the gauge.
- No asymptote claim, and no extrapolation.
- The curve is plotted with Fig. S1's Rfam curve. It goes in the main text as a panel only if
  the body stays at most 9.95 pages; otherwise it stays in the supplement.

## Scripts and records

- Item 1b: scripts/mxfold2_net_split.py writes runs/2026-10-20_mxfold2_net_split/metrics.json.
- Item 2: scripts/typeb_dimension_curve.py writes runs/2026-10-20_typeb_dimension_curve/metrics.json.
- Both refuse to run unless this file is committed and unmodified, and the tree is clean.
