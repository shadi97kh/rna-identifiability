# Certified folding: reading rules for registration 1

Reading rules, from the authors, committed before any test-set accuracy is seen. Copied
verbatim. Committed while the analysis phase was still running (it started at 13:14 on
2026-10-14 and writes nothing until it finishes); the commit's time is recorded beside the
analysis's finish time in the exploratory record and in the supplement's disclosure.

---

The headline claim comes from TS0 only. The held-out set (references are Rfam consensus
projections) and bpRNA-new are secondary: a result in the same direction as TS0 is reported as
"consistent"; one in the opposite direction is reported as such and never pooled away.

1. Soundness: R1 holds (0 false certificates across G2 to G5). Stated as measured, with the
   attack counts.
2. R2 and the within-strata check together decide the error-detection claim:
   a. R2 passes and the within-strata pooled interval is also entirely below 0: "among pairs
      Turner 2004 is confident about (p >= 0.99), those with the smallest certified radius are
      less accurate, also within length strata."
   b. R2 passes but the within-strata interval includes 0: R2 is reported as passed, and the
      text says the difference is explained by length. No claim that rho flags errors beyond
      length.
   c. R2 fails: reported as failed. The certificate is presented as a guarantee with a stated
      meaning, not as an error detector.
3. R3 and the length baseline: a difference is called higher or lower only when its paired
   interval excludes 0; otherwise "comparable". If rho_env does not beat the length baseline
   at matched coverage, the text says rho's ranking mostly carries length information.
4. D1: its PPV against the certified p >= 0.99 pairs, raw and within strata, is read by rule 2's
   logic. D2 gets its PPV and interval, and no claim beyond that.
5. Nothing is called better or worse on a point estimate alone.

---

## Addendum, 2026-10-15: one post hoc check (the authors)

Written and committed before the check runs. It is **post hoc**: it comes after R2, E1 and R3
were read (`docs/certfold_reading_report.md`), and it may only **qualify** the rule-2a sentence,
never strengthen it.

**The check.** R2's statistic with the nominal gap $X_0(p)-E_0(s^*)$ in place of rho_env:
PPV(lowest gap quartile) $-$ PPV(rest), among the TS0 MFE pairs with $p\ge0.99$ that carry a
rho_env, which is R2's own pool of 13,878 pairs. It uses R2's tie rules: the reversed ranking at
coverage 1/4, the forward ranking for the rest, and a block tied across the boundary shared by
fractional weights. With the gap as both score and tiebreak, ties on it are fractional. It uses
the same 10,000 bootstrap draws: the frozen `paired_bootstrap`, seed tag 40, over the same unit
universe (every TS0 sequence). The paired difference between the two statistics is also
reported:

$\Delta = d_\rho - d_{\text{gap}}$, on each resample.

Both $d$ are negative when the lower-ranked pairs are less accurate. $\Delta<0$ means rho's
effect is the larger.

**The reading.** By rule 5, "larger" is decided by $\Delta$'s paired interval, not by point
estimates.

- **$\Delta$'s interval lies entirely below 0** (rho's difference is larger): report it in the
  supplement as a post hoc observation, and change nothing in the main text.
- **Otherwise, with $d_{\text{gap}}$'s own interval entirely below 0** (the gap's difference is
  as large or larger): the main text adds *"the energy gap alone shows the same pattern, so this
  signal is not specific to the certificate"*, and the abstract adds *"a pattern the energy gap
  alone also shows"* after "within length strata".
- **Otherwise** ($d_{\text{gap}}$'s interval includes 0 and $\Delta$'s does not lie below 0):
  - This case is not in the authors' instruction. The agent added it, qualifying only, for the
    authors to confirm.
  - The main text adds *"whether the signal is specific to the certificate is unresolved: the
    energy gap's pattern cannot be told apart from it"*. The supplement reports both.

**Confirmed by the authors, 2026-10-15** (committed before the result landed). This supersedes the
wording of the three branches above; it is read by intervals only.

- **(a)** The paired difference's interval favors rho (it excludes 0 on rho's side): report it in
  the supplement only, as a post hoc observation, and leave the main text unchanged.
- **(b)** Otherwise, if the gap's own interval lies entirely below 0:
  - Section 6 adds *"a pattern the energy gap alone also shows, so this signal is not specific to
    the certificate"*.
  - The abstract adds *", a pattern the energy gap alone also shows"* after "within length
    strata".
- **(c)** Otherwise:
  - Section 6 and Limitations say *"whether this signal is specific to the certificate is
    unresolved"*.
  - The abstract is unchanged, since its sentence claims no specificity.

`scripts/certfold_posthoc.py` was committed before this confirmation, so the reading text it
writes into its record carries the earlier wording. The branch it selects is the same one, and the
paper follows the wording here.

**Check before reading.** The bootstrap of $d_\rho$ must reproduce R2's committed point and
interval exactly. That is the evidence that the draws are R2's.
