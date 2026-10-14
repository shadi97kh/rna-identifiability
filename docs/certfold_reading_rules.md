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
