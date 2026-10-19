# Certified folding: the four tables' unread entries (post hoc addition)

Specified 2026-10-19, before any code for it was written, at the authors' decision of the same
day, which lifts the analysis freeze for this addition and nothing else (docs/PLAN.md). It is
POST HOC: it comes after both registrations' results and the reviewer's additions were seen, and
every number from it is labeled post hoc wherever it appears.

**Deadline and drop rule.** Its results must be committed by 2026-10-21, end of day. If they are
not, or if any soundness check below finds a false certificate or a violation, the addition is
dropped: the failure is reported to the authors, and nothing about the addition enters the paper
or its supplement.

## 1. The set

The registered primary set is the published envelope at lambda = 1,
E_1 = {theta0 + B t : ||t|| <= r_max, |(B t)_e| <= delta for every entry e}, with B = B_four
from TR0's references alone (registration 1, runs/2026-10-13_certified_folding). It moves only
int11, int21, int22 and bulge entries that TR0's references read.

The addition lets the other entries of those four tables move too:

    E+_1 = {theta0 + B t + v : t in E_1, v in V},
    V    = {v : v_e = 0 off U, |v_e| <= m_e on U}.

- **Columns.** Exactly E's: the count model's parameter-file entries (B["cols"], 12,607),
  untied. An entry and its orientation twin are separate columns, as they are in B. That is the
  tie map E uses, and so the one used here.
- **U**, the unread entries: the columns of the int11, int21, int22 and bulge tables that no TR0
  reference structure reads, that is, those four tables' columns outside B["used"], the columns
  B_TR0 was built on. B_four is exactly zero on U, so the supports of B t and v are disjoint.
- **m_e**, the bound on an unread entry: the largest |theta_X[e] - theta0[e]| over
  X in {Turner 1999, Andronescu 2007, Langdon 2018}, in kcal/mol. These are the three published
  tables as typeb_common.full_tables loads them, and theta0 is Turner 2004. A table's forbidden
  (INF) entries count as a zero difference, as they do for E and its radii (Langdon 2018 forbids
  some loops). The script refuses if any column of U is INF in Turner 2004 itself.
- **lambda = 1 only.** There is no radius, no scale arm and no ball.

**What E+_1 contains.** For each published table X, the projection of its difference
delta_X (INF-zeroed) onto span(B_four) + R^U. The two subspaces are orthogonal, since their
supports are disjoint, so the projection is Pi delta_X + delta_X|U. Pi delta_X lies in E_1
(registration 1 checks this on every entry), and delta_X|U lies in V by the definition of m_e.
E+_1 does not contain the published tables themselves: the parts the data can see and the
gauge parts stay outside it.

## 2. The certificate, and why it is sound

The certificate is the frozen coupled certificate of registration 1, unchanged except for its
per-entry bound. The envelope uses u_e = min(delta, r_max ||a_e||) on the reached entries and 0
elsewhere. The addition uses

    u+_e = u_e + m_e   (u_e on the reached entries, m_e on U, 0 elsewhere),

passed as certfold.Weights(u+, B["cols"], 1.0). The callback, its shape filter (derived from
the nonzero entries of the bound vector), the ceiling to integer dcal, the forbidden-pair folds,
the Zuker pass, the strict comparisons, the recursion-side checks and the compiled backend are
all the frozen code's.

**Soundness.** For theta = theta0 + B t + v in E+_1 and any structure s,
E_theta(s) - E_theta(s*) = N(s) + sum over loops L of s of c_L.(Bt + v) - the same over s*.
Loops the two structures share cancel term for term. For every other loop, the triangle
inequality gives

    |c_L.(Bt + v)| <= |c_L.Bt| + |c_L.v| <= sum_e |c_{L,e}| u_e + sum_e |c_{L,e}| m_e
                    = sum_e |c_{L,e}| u+_e = w+_L.

The first bound is registration 1's (|(Bt)_e| <= u_e on E_1). The second is the box V. They add
because the supports are disjoint. So E_theta(s) - E_theta(s*) >= N(s) - sum over the
symmetric difference of w+_L, and the ceiling only lowers that bound further. That bound is
exactly what the frozen certificate compares with zero, so every label it issues with u+ holds
for every theta in E+_1. Type A directions change no energy, so the labels are gauge invariant
as before.

Every interior loop reads at most one entry of the four tables, with count 1
(certfold.il_keys). Loops of every other kind read none of them. So w+_L is the bound of the
single entry the loop reads.

**Monotonicity, a must-hold.** u+ >= u entry by entry, and the ceiling is monotone. So for every
TS0 MFE pair the margin under E+_1 is at most its registered margin under E_1 (integer dcal,
registration 1's run records). Every pair certified under E+_1 is certified under E_1, and per
sequence the certified-absent counts are at most the registered ones. Any exception stops the
run as a code error.

## 3. The script

New script: `scripts/certfold_unread.py`.

- It imports certfold.py and changes only u: every label and every gate check is certfold's own
  function, called with `dict(B, u_env=u+)` or `Weights(u+, ...)`. G4's check is inline in
  certfold.gate_g4, so its envelope branch is mirrored line for line with u+ in place of u.
- It refuses to run unless this file is committed and unmodified, and each phase refuses unless
  its predecessor's record is committed.
- It runs in .venv-certfold with the compiled callback, the registered environment, on 8 workers
  at nice 10, single-threaded.
- Records: `runs/2026-10-19_certfold_unread/metrics.json`, plus shards.
- **Seed tag 90:** SeedSequence([20261013, 90]) draws the signs of G5's vertex moves on U, one
  independent vector of signs per envelope draw. Nothing else here is random, and every other
  seed is registration 1's.

## 4. Phases, in order

**a. describe (before any folding).**
- The size of U by table, and the distribution of m_e: n nonzero, median, max, the entry and
  table attaining the max.
- The number of U entries each table forbids.
- For each published table, the share of its squared difference (INF-zeroed, in the same
  coordinates as the reviewer's coverage table) that E+'s span holds:
  (||Pi delta_X||^2 + ||delta_X|U||^2) / ||delta_X||^2, beside the registered span's
  3.3 / 3.3 / 1.4%.
- **Cross-check:** this must equal the reviewer's committed record. That is, "inside the set"
  plus the int11, int21, int22 and bulge parts of "outside: columns no TR0 reference reads"
  (runs/2026-10-18_certfold_reviewer, /descriptive/table_coverage) must agree to 1e-12.
- **Containment check:** each Pi delta_X + delta_X|U lies in E+_1, entry by entry.

**b. gates (G1, G2 and G4, on registration 1's own gate sequences).**
- **G1, the callback identity, with the extended weights.**
  - Sampled half: the 20 TR0 sequences (n from 50 to 500), each with 1,000 stochastic
    structures plus s*.
  - Constructed half: every row of the generating set, under s* = its own structure and under
    s* = all-unpaired, so that every int11, int21 and int22 entry, U included, is charged with
    both signs.
  - Both callback backends. An exact integer match is required.
- **G2 (n <= 20).** The 200 registered sequences, through certfold.g2_one with u+. A certified
  pair must be robust over the box {|x_e| <= u+_e}, which contains E+_1. X' and Z' must match
  the enumerated minimum.
- **G2 power check.** At half weights, on the registered first 50 sequences, the check must
  produce at least one false certificate.
- **G4 (n <= 150).** The 100 registered TR0 sequences: the Wuchty band under E' with u+, each
  structure tested by the box bound.
  - The registered band cap stands (600 dcal): a sequence whose band exceeds it is skipped and
    counted, never truncated.
  - The relaxation loss is reported as for E.

Any false certificate, power-check failure or identity mismatch: report it and stop.

**c. attacks (G5 on TS0, after the gates are committed).**
- Three component attacks: Pi delta_X + delta_X|U for each table.
- 200 draws: registration 1's G5 envelope draws (seed tag 50), each plus a vertex of V with signs
  from tag 90.
- For each of the 20 lowest-E' alternatives (under u+), registration 1's worst-case envelope
  shift plus the worst vertex of V, v_e = -sign(dc_e) m_e on U (dc = c(s) - c(s*)). That vertex
  is the exact minimiser of the gap over V.
- A violation counts only if exact float energies from counts put a structure lacking a
  certified pair, or holding a certified-absent one, at or below E(s*).
- The phase records **violation counts only**, plus a sha256 of each sequence's certified sets,
  so that the run can confirm it labels exactly what was attacked.
- Any violation: report it and stop.

**d. run (TS0, 1,288 sequences, only after G5 passed and is committed).**
- Labels under E+_1 through certfold.certify, for the same rows, stars and strata as
  registration 1's run: present for every MFE pair, absent over the admissible pairs outside s*.
- The contested pairs (Turner 2004 p >= 0.01) come from the plain compound, as in
  certfold.certify_sequence.
- The run checks the monotonicity must-holds of Section 2 and the G5 digests.
- No reference structure is read. No accuracy, no radius, no interval.

**e. analyse.**
- The certified-present share of TS0's MFE pairs, pooled and by the registered TR0 length strata,
  beside registration 1's 40.9% and its strata.
- The certified-absent share of the contested pairs, beside 38.7% (S10 only), and of all
  admissible non-MFE pairs.
- Whole structures certified.
- The fractions use registration 1's definitions (certfold_run.describe's label part). The same
  function is first run on registration 1's own TS0 records and must reproduce 21,776 of 53,248
  and 40,291 of 104,231 exactly.

## 5. Reading, fixed now

Descriptive.

- **Section 6** gets one clause after the coverage sentence: "Letting the entries of the four
  tables that no TR0 reference reads move too, each by up to the largest published difference
  on it, certifies X% of TS0's predicted pairs (post hoc)."
- **The conclusion's 1.4 to 3.3% sentence** gets the same X appended.
- No adjective, and no claim beyond placing 40.9% beside X.
- **The abstract does not change, whatever X is.**
- **S10** gets the details: the set, the soundness paragraph, the gates' counts, the share of each
  published difference that E+'s span holds, the strata table, and the certified-absent share.
- Every number gets a CLAIM line and a pin.
