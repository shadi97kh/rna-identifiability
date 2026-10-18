# Certified folding: the reviewer's additions (post hoc), specified before any of them runs

Requested 2026-10-18 by the authors ("do all these") after an outside reviewer read
`docs/certfold_experiment_report.pdf`. Every analysis here comes after both registrations'
results were seen, so every one is **post hoc**. None is a registered test, and none may
strengthen a registered claim; each may only describe, qualify or extend one. This file is
committed before any of them runs, and the code that runs them (`scripts/certfold_reviewer.py`,
records in `runs/2026-10-18_certfold_reviewer/metrics.json`) refuses to run unless this file is
committed unmodified.

Data: TS0 (1,288 sequences, 53,248 Turner 2004 MFE pairs), registration 1's labels and records,
registration 2's folds and measured-set labels. New seed tags, all `SeedSequence([20261013, tag])`:

| Tag | Use |
|---|---|
| 80 | leave-one-table-out R4 statistics, stratified bootstrap |
| 81 | R4 statistic with certification at lambda = 0.5 and 2, stratified bootstrap |
| 82 | bootstrap of the TR0 pilot's 300 sequences (ranking stability) |
| 83 | an independent second TR0 pilot of 300 sequences |
| 84 | random orderings for the measurement-efficiency curve |
| 85 | R4 statistic under the alternative basis, stratified bootstrap |
| 86 | G2 sequences for the alternative sets are registration 1's (no new draw); unused |

Every stratified bootstrap is `certfold_exploratory.stratified_bootstrap` with 10,000 resamples,
strata 0 and 1 merged as for E2 and R4, unit and p >= 0.99 MFE records only (R4's own pool).

## 1. Reproduce R4 and R5's intervals independently

A fresh-context agent writes `scripts/certfold_reproduce_r4r5.py` from the shard files, without
importing any bootstrap, metric or record-building code from the certfold modules, and recomputes
R4 (primary, secondary, tertiary) and R5 (3 tables x 3 baselines) with the registered seeds (60,
61) and resampling schemes as their docstrings describe them. Expected: every point and interval
equal to the committed record to 1e-12. Any difference is reported, and if it is not exact, both
implementations are rerun on one fresh tag (87) to show the difference is Monte Carlo, or it is
treated as a defect.

## 2. Stability, made interpretable (descriptive, plus exploratory intervals)

Pool: R4's (TS0 MFE pairs with p >= 0.99; certified = registration 1's label at lambda = 1).

- **Absolute table.** For each of the three whole tables and for retention over all three:
  kept fraction of certified and of uncertified pairs, the absolute difference in points, pairs
  lost in each group (count), and the relative reduction in pair loss, 1 - loss(certified) /
  loss(uncertified). Raw (pooled over TS0) and by length stratum.
- **Leave one table out** (tag 80). R4's statistic with retention over each pair of tables, and
  with kept under each single table. Reading, fixed now: if every interval lies above 0, the text
  may say the difference "holds with any two of the three tables"; otherwise it says the result
  depends on which tables are included, naming the case.
- **Wording, fixed now** (the reviewer's): "Certified pairs show modestly higher empirical
  retention under the three evaluated parameter-table substitutions, although the certified
  radius does not rank pair survival better than base-pair probability." No claim of
  generalization to other energy models.

## 3. The uncertainty set: scale, coverage of the published tables, and the held-out basis

**3a. Scale (descriptive and exploratory).** Certified-present fraction by stratum at lambda =
0.5, 1, 2 (registration 1's labels). R4's statistic with certification at lambda = 0.5 and at
lambda = 2 (tag 81); a stratum where either group is empty is reported undefined, never dropped.

**3b. Coverage of the published tables (descriptive).** For Turner 1999, Andronescu 2007 and
Langdon 2018: delta_X (INF entries set to a zero shift, as everywhere) split into the part inside
the set's span (Pi delta), and outside it the part on columns no TR0 reference reads, the gauge
part and the rest; each part's norm broken down by parameter family (stack, int11, int21, int22,
bulge, hairpin, interior-loop initiation, mismatches, multiloop, terminal AU, Ninio). Also, per
table: whether Pi delta lies inside E_1 (max per-entry shift against delta, norm against r_max).

**3c. The held-out basis (exploratory).** B is built from TR0 alone; on TS0, 49 of its 448
directions are ones TS0's own references would pin. The alternative excludes them: B_excl = the
part of span(B_TR0) that TS0's references cannot see either (expected dimension 399 = 448 - 49),
restricted to the four tables and re-orthonormalized exactly as B_four is. Two envelopes on it:
- (i) the same r_max and delta as the registered envelope, so only the directions change;
- (ii) r_max and delta recomputed from the three published components projected onto B_excl,
  by the registered definitions.
Before any TS0 fold under them: G2's n <= 20 enumeration (registration 1's 200 sequences) on
both, 0 false certificates required, and the halved-weights check must produce false
certificates. Then every TS0 MFE pair is labeled at lambda = 1 under each (one forbidden fold per
pair; no pair is skipped, so monotonicity is a check rather than an assumption: under (i) every
pair certified under E must stay certified). Reported: certified-present fraction overall and by
stratum, transitions against E, and R4's statistic with each alternative's labels (tag 85).
Reading, fixed now: if R4 under an alternative has an interval above 0, the text may say the
stability result "does not depend on the 49 TS0-pinnable directions" for that variant; otherwise
it says it does.

## 4. Targeted measurement: the efficiency curve and the ranking's stability

**4a. Efficiency curve (exploratory).** For the frozen top ordering, the certified-present gain at
every K = 1, ..., 50. Lowering u on more entries can only add certificates, so along a nested
ordering each pair has a smallest K at which it becomes certified; that K is found by bisection,
only for the pairs certified at K = 50 (registration 2's measured labels), every other pair being
uncertain at every K <= 50. Control: 20 random orderings, each a seeded permutation (tag 84) of
one registered random-50 draw's entries, so each random curve ends at that draw's registered K =
50 gain. Reported: gain(K) for the top and the 20 random curves (median and 5th to 95th
percentile band), the marginal gain per entry, and K_half, the smallest K reaching half of the
top-50 gain. Before TS0: G2 on top-1, top-5 and top-25 (registration 1's 200 sequences).

**4b. Ranking stability.**
- *Bootstrap of the pilot* (descriptive, tag 82): the 300 pilot sequences resampled within their
  length strata 1,000 times, the ranking recomputed from the recorded witnesses (no new folds).
  Reported: for each frozen top-50 entry, the fraction of resamples keeping it in the top 50;
  the distribution of |top-50 overlap|; the same for the top 10. Reading, fixed now: the ranking
  is called "stable under resampling of the pilot" only if at least 80% of the frozen top-50
  entries stay in the top 50 in at least 80% of resamples; otherwise the text reports the overlap
  without that word.
- *An independent second pilot* (exploratory, tag 83): 300 TR0 sequences disjoint from the first,
  drawn with the same per-stratum design, their witnesses folded, the ranking rebuilt; its top-10
  and top-50 overlaps with the frozen ones; then its top-50 measured set evaluated on TS0 exactly
  as registration 2's (uncertain pairs only, sequences that cannot read a zeroed entry skipped,
  margins checked against registration 1's), after G2 on that set. Reported: its K = 50 gain
  beside the frozen ranking's 6.73 and the random-50 95th percentile 2.12.

## 5. Conservativeness, and what an uncertain label means (descriptive)

From existing records only. Panel A: at each exactly enumerable scale (G2: ball r = 1, 5, 20 and
the envelope, n <= 20; G4: ball r = 1, 5 and the envelope, n <= 150), exactly robust pairs, how
many the relaxation certifies, how many it misses, and the fraction missed (on the envelope G4 is
a lower bound, as recorded). Panel B: TS0 MFE pairs at lambda = 1 split into certified,
uncertain and observed to move under one of the 200 envelope draws, and uncertain and not
observed to move, labeled **unresolved, not stable**; by stratum. The text states the guarantee
is one-sided: certification establishes robustness, failure to certify does not establish
fragility.

## 6. The radius against the nominal gap (descriptive)

TS0 MFE pairs carrying a finite rho_env. Spearman correlation of rho_env and the nominal gap,
overall and by stratum; agreement of their quartiles; "substantial disagreement", fixed now: a
pair in the lowest quartile of one and the upper half of the other. Where it concentrates: by
stratum, by competitor-side exposure quartile, and by whether the pair is adjacent in s* to an
interior loop the set reaches. For each disagreement direction, the two most extreme TS0 pairs
(by the difference in percentile rank, ties by index) as worked cases: the pair, p, the gap,
rho_env, and the witness at lambda = rho_env (one fold each) with its differing loops' weight
D(s). The point to show, if the data show it: rho_env = min N(s)/D(s) weighs a competitor's gap
by how much of the set its differing loops can use, which the gap alone does not.

## 7. Reproducibility

A reproduction section in `artifact/README.md`: environments, the order of phases, the frozen
configurations (PRED, PRED2, this file), what each records, and the data's own licences; a
licence audit of everything the export would publish, written to `artifact/LICENSE_AUDIT.md`.
Pushing to the artifact repository, and making it public, wait for the authors' explicit go.

## 8. The paper

Restructured as the reviewer proposes, within 10 Letter pages: the guarantee and its theorem
with the computational assumptions stated in the main text; coverage, conservativeness and
whole-table stability with absolute effects; targeted measurement with the efficiency curve; the
accuracy ranking kept in the main text as a limitation; exhaustive checks and full tables in the
supplement. R4's secondary is reported as an interpretation the registered rules do not cover,
labeled so. Every number from this file is labeled post hoc where it appears.

## Correction, 2026-10-18, before any phase ran: 67 directions, not 49

Smoke-testing section 3c's code (on TS0's reference structures only: no label, accuracy or
fold of a test sequence) found that TS0's references have rank **67** on B_TR0, with a clean
gap in the singular values (0.339, then 7e-13). Principal angles between B_TR0 (448) and the
TR0+TS0 basis (426) agree independently: they share exactly 381 directions. So B_excl, as
defined above, has dimension **381**, and TS0's own references would pin **67** of the set's
directions, not 49.

The 49 is a different quantity, recorded by build-basis under `principal_angles /
dimension_arithmetic`: B_TR0's dimension beyond the span of the TR0+TS0 directions *restricted to
TR0's entries* (448 - 399). Restricting a direction drops its part on entries only TS0 reads, so 18
of those 399 restricted directions are not invisible to TS0's references. The paper's sentence
"on TS0 the set includes 49 directions that TS0's own references would pin" is therefore wrong
and is corrected in the paper and the experiment report: 448 = 381 + 67, and 426 = 381 + 45
(45 of the TR0+TS0 directions use entries no TR0 reference reads, 27 of them only such entries).

Nothing else changes. Section 3c's definition stands (B_excl = the part of span(B_TR0) TS0's
references cannot see either); its "expected dimension 399" reads 381, and its reading's
"49 TS0-pinnable directions" reads "67".
