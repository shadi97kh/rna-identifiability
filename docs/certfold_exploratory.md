# Certified folding: exploratory analyses beside registration 1

**Status: EXPLORATORY. Not registered tests.** Reported as exploratory, beside R1 to R3, never in
their place. Specified by the authors on 2026-10-14 and written here before any test-set
accuracy exists.

## When this was written, exactly

The authors asked for this file to be committed before the test-set run was launched. It was
not: the run (`scripts/certfold_run.py --phase run`, commit 31e8c8a) was launched at 01:39 on
2026-10-14, with the user's approval, before these instructions arrived, and its folds were
complete by about 12:30. This file is committed after those folds and **before the analysis
phase**: no prediction has been scored against any test-set reference structure, and no PPV,
AURC or coverage number exists for TS0, the held-out set or bpRNA-new. That property, which is
what the timing rule protects, holds. The commit of this file is recorded in the exploratory
record and in the analysis record.

## Why

rho falls steeply with sequence length: on the TR0 pilot the envelope at lambda = 1 certified
64 to 76% of MFE pairs in the shortest length strata and 12% in the longest. MFE accuracy also
tends to fall with length. So the pairs with the lowest rho_env may simply be the pairs of long
sequences, and R2 (lowest rho_env quartile less accurate than the rest, among pairs with
p >= 0.99) can pass on length alone. A pass would then not show that the certificate catches
errors that p misses. These analyses ask whether the effect survives once length is held fixed.

**Reading.** If the within-stratum effect holds (E1's pooled interval lies entirely below 0),
the claim stands. If it does not, the paper says that among confident pairs rho_env mostly
tracks length. E3 says how much of rho_env's ranking a length score alone reproduces.

## E1. R2 within length strata (TS0)

- Pool: R2's pool. TS0 Turner 2004 MFE pairs with p >= 0.99 that carry a rho_env.
- Strata: the ten TR0 length strata of the pilot (`certfold_data.TR0_STRATA`; a length on a
  boundary goes to the lower stratum), each sequence in exactly one.
- Within each stratum s with n_s pairs: the lowest rho_env quartile cut at that stratum's own
  boundary, with R2's rules exactly. The reversed ranking is rho_env ascending, then the nominal
  gap ascending. The lowest quartile is its top k_low = the smallest k with k / n_s >= 1/4, and
  the rest is the top n_s - k_low of the forward ranking. A block tied on both keys across the
  boundary is shared by fractional weights. This is `certfold_run.stat_r2_quartile` applied to
  the stratum's pairs. d_s = PPV(lowest) - PPV(rest), with one-nucleotide slip.
- Pooled: sum_s n_s d_s / sum_s n_s over the strata where d_s is defined on the full data. A
  stratum where it is not (fewer than 2 pairs) is listed and left out.
- Capped fraction per stratum reported beside d_s. R2's capped-versus-uncapped fallback is a
  rule of the registered test and is not applied here.

## E2. D1 within length strata (TS0)

- Pool: TS0 MFE pairs with p >= 0.99 (labels are complete, so no rho_env is needed).
- Within each stratum: d_s = PPV(D1_s) - PPV(C_s), where D1_s is the pairs not certified by the
  envelope at lambda = 1 (D1 as registered) and C_s the pairs certified at lambda = 1.
- Pooled: weighted by each stratum's p >= 0.99 pair count, over the strata where both sets are
  non-empty on the full data.

## Intervals for E1 and E2

10,000 resamples. In each, every stratum's sequences are drawn with replacement within that
stratum, as many as it holds, so stratum sizes are fixed and length cannot leak between strata.
A sequence drawn more than once counts each time. Every d_s and the pooled value are recomputed
on the resample, with the pooled weights taken from the resample's own pair counts. Draws come
from `default_rng(SeedSequence([20261013, 41]))`, strata in order 0 to 9 and the sequences
within a stratum sorted by index. The interval is the 95% percentile interval. If a statistic
is undefined on some resample (an empty lowest quartile, an empty D1_s), its interval is
reported as undefined with the number of such resamples. The resample is never dropped. E1 and
E2 use the same draws, so they are paired.

This stratified bootstrap is new code, in `scripts/certfold_exploratory.py`. The frozen
`certfold_metrics.paired_bootstrap` resamples units over a whole set and cannot hold strata
fixed. Every PPV inside it still comes from the frozen metrics (`risk_coverage`,
`ppv_at_coverage`, `pooled_ppv`, `disagreement_sets`, `slip_correct`).

## E3. A length baseline in the matched-coverage comparison (all three sets)

- Score: -n_nt, so a shorter sequence counts as more confident. Ties go by the nominal gap and
  then fractional weights, the registered tiebreak of every ranking.
- Compared with rho_env exactly as R3 compares rho_env with p and the nominal gap
  (`certfold_metrics.matched_coverage_comparison`). Same pool (MFE pairs carrying a rho_env),
  same coverage (the p >= 0.99 set's), same weighting (families equally on the held-out set,
  pooled on TS0 and bpRNA-new), same paired bootstrap (`paired_bootstrap`, seed tag 40, the
  draws R3 uses). It is reported beside p and the nominal gap.

## Wording (the authors, 2026-10-14)

The held-out set's references are Rfam consensus structures projected onto each sequence, not
structures determined per molecule. TS0 is the primary accuracy set, and R2 runs on TS0.

## Code and order

`scripts/certfold_exploratory.py`, phase `exploratory`. It is written and committed before
`certfold_run.py --phase analyse` runs, and run after it, so no exploratory code is written
once accuracy exists. It reads the run's shards through `certfold_run.read_shards`, checks
their digests against the committed run record, builds records with
`certfold_run.pair_records`, and writes `metrics.json["exploratory"]` with this file's commit.
