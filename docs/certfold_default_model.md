# Certified folding: the default model, and a bound on decoupling (post hoc, two items)

Specified on 2026-10-19, before any code for either item was written. This follows the authors'
decision of the same evening to lift the analysis freeze one last time for these two items,
under this one specification (docs/PLAN.md). Both items are POST HOC: they come after both
registrations, the reviewer's additions and the unread-entries addition were seen. Every number
from them is labeled post hoc wherever it appears.

**Drop rule.** Results must be committed by 2026-10-21, end of day. If an item misses that, or
any check listed for it below fails, that item is dropped. The failure is reported to the
authors, and nothing about the dropped item enters the paper. **The abstract does not change,
whatever either result is.**

**Script and record.**
- Script: `scripts/certfold_default_model.py`. It imports the frozen code: certfold.py,
  certfold_run.py, certfold_reg2.py and certfold_exploratory.py. It defines no label, weight,
  statistic or bootstrap of its own, and it refuses to run unless this file is committed
  unmodified.
- Record: `runs/2026-10-19_certfold_default_model/metrics.json`.
- Environment: .venv-certfold, the registered environment.
- Seed: tag 91, SeedSequence([20261013, 91]), the stratified bootstrap of item (a). Nothing else
  is random.

## (a) Default-model transfer

**The question.** Do the certified labels, computed at dangles 0 with special hairpins off,
predict which Turner 2004 MFE pairs survive the change to ViennaRNA's default model?

**The default model.** thermo.vienna_ref.reference_md() with `dangles = 2` and `special_hp = 1`,
each read back. Every other field stays as pinned; the record lists every field in which this
model differs from a bare RNA.md().

**The folds.**
- Each of TS0's 1,288 sequences is folded once under the default model, with Turner 2004
  parameters and nothing perturbed.
- No certificate is computed at dangles 2, and no guarantee is claimed there.

**The statistic.** R4's registered statistic (registration 2), unchanged except for one
substitution. Retention over the three whole tables is replaced by kept(p) = 1 if the
default-model MFE structure contains the pair, else 0.
- The pool: TS0's Turner 2004 MFE pairs at the pinned model (dangles 0) with p >= 0.99, from
  registration 1's records.
- Certified means registration 1's label at the envelope, lambda = 1.
- The statistic is the mean kept among certified pairs minus the mean among uncertified ones,
  within the ten TR0 length strata (strata 0 and 1 merged), pooled by the p >= 0.99 pair count.
- The interval comes from 10,000 stratified resamples, seed tag 91.
- The code path is registration 2's own: certfold_run.pair_records for the records,
  certfold_reg2._r4_spec for the statistic, and certfold_exploratory.strata_groups and
  stratified_bootstrap for the bootstrap.

**Descriptive.**
- Absolute retention by label: among the p >= 0.99 pairs, the share kept by the default-model
  MFE for certified and uncertified pairs, the numbers lost, and the relative reduction in loss,
  pooled and by stratum.
- The share of all TS0 MFE pairs the default-model MFE keeps, and the number of sequences whose
  MFE structure changes.

**Checks (any failure drops item a).**
1. The default model's dangles and special_hp read back as 2 and 1, and every compound passes
   assert_turner2004_params.
2. Every default-model fold's MFE energy equals the count model's energy exactly, in integer
   dcal. That energy is Extractor(special, 2, 1, lxc) counts dotted with clip_d2(theta0),
   plus the long-loop offset and the Ninio correction. The clip in clip_d2 is turner_par's
   validated d2 clip.
3. Each sequence's dangles 0 MFE equals registration 1's s* for that sequence.
4. The same code path, run on registration 2's own records with its seed tag 60, must reproduce
   registration 2's R4 primary point and interval exactly before tag 91 is drawn.

**Reading, fixed now.**
- If the pooled interval lies above 0, Limitations' "and no claim here covers it" becomes "and
  no guarantee here covers it, though certified pairs are kept more often in its MFE (post hoc,
  S7)".
- Otherwise, Limitations gains "and the dangles 0 labels carry no detectable signal into its
  MFE (post hoc, S7)".
- In either case S7 gets one paragraph with the statistic, its interval and the absolute
  retention, since both Limitations clauses point there.
- No other text, and nothing in the main text beyond that clause.

## (b) Decoupled-coverage bound (no folding)

**The question.** How much does coupling buy? A decoupled certificate charges each structure in
full, every weighted loop of the competitor and of s*, instead of only the loops where they
differ.

**Its necessary condition.** Such a certificate certifies an MFE pair p only if
min over s without p of [E0(s) - W(s)] > E0(s*) + W(s*). Here W(s) is the summed lambda = 1
charge on s's weighted loops. The left side is at most X0(p), the nominal energy of the best
structure without p. So the certificate can certify p only if the nominal gap X0(p) - E0(s*)
exceeds W(s*). The share of MFE pairs with gap > W(s*) is therefore an upper bound on what a
decoupled certificate with the same s* could certify.

**Computed from stored records.**
- The nominal gap: registration 1's nominal_gap_kcal per pair, as integer dcal.
- W(s*): certfold.eprime_offset(s*) under certfold.Weights(u_env, cols, 1.0), the frozen
  envelope weights at lambda = 1, integer dcal.
- The comparison is strict, in integer dcal.
- Reported pooled and by registration 1's ten length strata, beside the coupled certificate's
  certified share (40.9% pooled, and its strata).

**Checks (any failure drops item b).**
1. Every nominal gap is >= 0 and every W(s*) is >= 0.
2. The MFE pairs are exactly registration 1's.

**Reading, fixed now.**
- S7 says "a decoupled certificate could certify at most X% (post hoc)" for the strata, and
  pooled, where the bound is below the coupled share, and calls the other strata uninformative.
- Supplement only.

## Order

Each phase refuses unless its predecessor's record is committed:
1. `fold` (item a's folds and checks 1 to 3);
2. `analyse` (check a4, item a's statistic and tables, item b).

Then the text by the fixed readings, CLAIM lines and pins for every number, the four manuscript
tests, the 10-page check and a fresh-context review of the changed passages.
