# A post hoc addition: two certificate-free rankings of the reached entries

Specified before any code for this item was written. The analysis freeze was lifted for
this one item (docs/PLAN.md). It is POST HOC: it comes after both registrations and every earlier
addition were seen, and every number from it is labeled post hoc wherever it appears. No post hoc
result enters the abstract.

**What is already known at this point.** Registration 2's numbers: the certificate's ranking
gains 6.73 points of TS0's MFE pairs at K = 50, and random sets of 50 reach 2.12 at their 95th
percentile. Neither ranking below has been computed, on the pilot or on TS0.

**Question.** Does the certificate's ranking of what to measure gain more than rankings that need
no certificate, built from the same pilot?

**Script and record.** `scripts/certfold_frequency_rankings.py`, one phase per step, each refusing
to run unless this file is committed unmodified and the previous step's record is committed.
Record: `runs/2026-11-03_certfold_frequency_rankings/metrics.json`, shards under its `shards/`
(gitignored, sha256 recorded). Environment: .venv-certfold, the registered environment with the
compiled callback.

## The rankings

Both rank the 923 reached entries (u_e > 0 in the registered envelope) from the first pilot's 300
TR0 sequences (`certfold.pilot_sequences`), the pilot behind registration 2's ranking. Both are
fixed, and committed, before any TS0 fold under them.

- **(a) mfe_reads.** For each pilot sequence, its Turner 2004 MFE structure under the pinned model.
  An entry's count is the number of that structure's interior loops reading it, summed over the
  pilot.
- **(b) admissible_reads.** The competitor-side exposure. For each pilot sequence, every admissible
  interior loop: (i, j) canonical with at least three unpaired bases inside, a shape (u1, u2) among
  the shapes the reached entries have, and (k, l) = (i + u1 + 1, j - u2 - 1) canonical with at
  least three unpaired bases inside. This is the enumeration of `certfold.exposure` and of
  registration 2's `readable_entries`. An entry's count is the number of those loops reading it,
  summed over the pilot. Must hold on every pilot sequence: the number of loops reading a reached
  entry equals `certfold.exposure`.

Ties, in both, by the entry key as a string, as in registration 2. The top 50 of each is its
measured set. Reported beside them: each top 50 by table, and its overlap with the certificate's
top 50 and with the other baseline's.

## The gate

G2 on each top 50 before any TS0 fold, as registration 2's `gate_g2_measured`: the same 200
sequences of at most 20 nt, lambda = 1, the measured u (u_e = 0 on the 50 entries, every other u_e
kept). Zero false certificates are required, and halving the weights on the first 50 sequences
must still produce false certificates. A failure stops the item.

## The labels

Exactly as registration 2's `measured_labels_worker`: for each of TS0's 1,288 sequences and each
set whose zeroed entries it can read, the pairs uncertain at lambda = 1 in registration 1 are
refolded under the measured set, and each refolded margin must be at least registration 1's. Gain
at K = 50 = (pairs newly certified) / (TS0's 53,248 MFE pairs).

Control: the certificate's own top 50 is run through this code as well. Must hold: on every
sequence its newly certified pairs equal registration 2's, and its gain equals the record's 6.73
points. If not, the item stops.

## The comparison

For each baseline, the certificate ranking's gain minus the baseline's, with a paired sequence
bootstrap: TS0's 1,288 sequences drawn with replacement, sorted by index, 10,000 resamples, one set
of draws for both comparisons, from `SeedSequence([20261013, 98])`. On each resample the statistic
is (newly certified under the certificate's set - under the baseline's) / (MFE pairs), counted over
the resample. Percentile interval, numpy linear, 2.5 and 97.5.

The rankings are held fixed, so the interval covers TS0's sampling and not the pilot's.

**Reading, per baseline.** The certificate's ranking is "higher" if the interval lies entirely above
0, "lower" if it lies entirely below 0, and "comparable" otherwise.

**Stop rule.** If either baseline reads comparable, or the certificate's ranking reads lower, no
wording about what to measure changes without a further decision. The abstract does not change in any
case.
