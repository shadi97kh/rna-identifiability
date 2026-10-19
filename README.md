# Same Energies, Different Folds

Code and run artifacts for *Same Energies, Different Folds: Certifying RNA Base Pairs Against
Parameters the Data Cannot Fix* (S. Khodagholi and N. Yousefi; submitted to RECOMB 2027). Every
experimental number the paper reports is read from a `runs/<id>/metrics.json` in this repository,
and each of those files records the commit and the code that produced it.

## What the paper shows

Nearest-neighbor energy models predict RNA secondary structure from tables of loop parameters
that are refit, learned and compared entry by entry, yet structure data cannot pin down every
parameter. The paper describes three kinds of change the data cannot detect, and introduces
certified folding, which certifies for each base pair of a predicted structure that it stays present
or stays absent under every data-restricted move within stated bounds, or reports it as uncertain.

| | the change | code |
|---|---|---|
| **Type A** | *Gauge directions* leave every structure's energy unchanged. ViennaRNA's Turner 2004 model has exactly two with dangles off. | `scripts/turner_nullspace.py` |
| **Type B** | *Data-restricted directions* leave the energies of a set of reference structures unchanged but not those of others: 426 for the 11,955 bpRNA structures in TR0 and TS0. | `scripts/turner_nullspace_tr0.py`, `scripts/turner_typeb*.py`, `scripts/typeb_*.py` |
| **Type C** | Learned models such as MXfold2 add output gauges of their own; the optimizer, not the data, moves MXfold2 along a weight symmetry. | `scripts/mxfold2_*.py` |
| **Certified folding** | For each base pair of the Turner 2004 MFE structure: certified present, certified absent or uncertain, for every table in a set built from TR0's Type B directions and bounded by the published tables' data-invisible differences, with a verified certified radius. | `scripts/certfold*.py` |

## Where each headline number comes from

| claim in the abstract | artifact |
|---|---|
| exactly two gauge directions; 426 data-restricted directions for the 11,955 TR0 and TS0 structures | `runs/2026-10-07_turner_nullspace_r2/`, `runs/2026-10-08_turner_nullspace_tr0/` |
| adding to Turner 2004 only the part of Andronescu 2007's difference the data cannot see reorders 3.9% of predicted structures, with no detectable change in in-sample accuracy (a small, detectable loss on held-out families) | `runs/2026-10-12_turner_typeb_tie_split/`, `runs/2026-10-12_typeb_decoders/`, `runs/2026-10-12_typeb_heldout/` |
| no false certificate in 1.2 million checks on 5,429 test sequences, including 108,580 targeted worst-case moves | `runs/2026-10-13_certified_folding/` (`gate`) |
| for moves along TR0's data-restricted directions up to the size of the published tables' differences along them, 40.9% of TS0's predicted pairs certified, 70% in short sequences to 12% in long ones | `runs/2026-10-13_certified_folding/` (`analysis`) |
| the certified radius ranks pairs less accurately than base-pair probability, and its signal among confident pairs is shared with the energy gap | `runs/2026-10-13_certified_folding/` (`analysis`, `posthoc`) |
| measuring the 50 top-ranked motifs exactly would raise the certified share from 40.9% to 47.6%; 20 random sets of 50 raise it by at most 2.5 points; a second training sample gains nearly as much | `runs/2026-10-15_certfold_registration2/` (`analysis_d`), `runs/2026-10-18_certfold_reviewer/` (`analysis`) |
| in MXfold2 the optimizer, not the data, moves the model along a weight symmetry, and fixing it costs no accuracy | `runs/2026-10-11_mxfold2_weight_decay_ablation/`, `runs/2026-10-12_mxfold2_ablation/` |

The other directories under `runs/` back the remaining numbers in the paper and its
supplement, or are the registrations and inputs those runs name. The paper's TeX sources
(`main_letter.tex`, `supplement_letter.tex`) are not part of this repository; every number in
them is tied to its artifact by a pin, checked by `tests/test_recomb_pins.py` in the authors'
working repository.

## Layout

```
scripts/   every experiment, one driver per run; each run's PRED block is the registration
thermo/    vienna_ref.py, the one place the ViennaRNA model is pinned (see thermo/ORACLES.md)
tests/     unit and invariant tests
runs/      one directory per run: metrics.json, written only by the script that ran it
data/      SOURCES.yaml only: where each dataset comes from, its digest and its licence
docs/      remote_commit_map.tsv (see Provenance)
```

## Requirements

Python 3.11, and the versions every result here was produced with:

| package | version | used by |
|---|---|---|
| ViennaRNA (`pip install ViennaRNA`) | **2.7.2, exactly** | everything thermodynamic; `thermo/vienna_ref.py` refuses any other model |
| numpy | 2.4.6 | everything |
| scipy | 1.17.1 | null-space and Fisher computations |
| torch | 2.13.0 | the MXfold2 experiments |
| pyyaml | 6.0.3 | the data ingest scripts |
| pytest | 9.1.1 | the tests |
| numba | 0.68.0 | certified folding's compiled callback, in `.venv-certfold` only |

The MXfold2 experiments additionally need [MXfold2](https://github.com/mxfold/mxfold2) at
commit `51b2136`, installed in its own environment. It is an external dependency and is not
included here.

## Data

No dataset is redistributed. The scripts download each one from its original source and check
it against the digest recorded in `data/SOURCES.yaml`:

| dataset | used for | fetch with | licence |
|---|---|---|---|
| TR0, TS0 and bpRNA-new (bpRNA-derived, from MXfold2's data release) | the reference structures; the certified-folding test sets TS0 and bpRNA-new | `scripts/ingest_mxfold2_data.py` | CC BY 4.0 |
| Rfam 12.2 and 15.1 | the 1,641 held-out families | `scripts/ingest_rfam.py`, `scripts/ingest_rfam122_families.py` | CC0 1.0 |

`data/SOURCES.yaml` here holds only these three entries, each with its download URL, sha256 and
licence.

### Attribution

- **MXfold2 datasets** (used here: TR0, TS0 and bpRNA-new). Kengo Sato, *The datasets used in
  "RNA secondary structure prediction using deep learning with thermodynamic integration"*,
  Zenodo, 2021, <https://doi.org/10.5281/zenodo.4430150>, licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The paper that introduced them:
  K. Sato, M. Akiyama and Y. Sakakibara, *Nature Communications* 12:941 (2021),
  <https://doi.org/10.1038/s41467-021-21194-4>. The structures derive from bpRNA-1m:
  P. Danaee et al., *Nucleic Acids Research* 46:5381–5394 (2018),
  <https://doi.org/10.1093/nar/gky285>.
  - **Changes:** the archive is downloaded and checked against its digest, never modified or
    redistributed. The scripts read it and filter the rows: sequences with letters other than
    ACGU are dropped, and pairs closing hairpins of fewer than three unpaired bases are removed.
    bpRNA-new is subsampled with a recorded seed.
- **Rfam** (the 15.1 seed alignments and the 12.2 family table). CC0 1.0, so no attribution is
  required. Please cite N. Ontiveros-Palacios et al., *Nucleic Acids Research* 53:D258–D267
  (2025), <https://doi.org/10.1093/nar/gkae1023>, and, for release 12, E. P. Nawrocki et al.,
  *Nucleic Acids Research* 43:D130–D137 (2015).

## Reproducing a result

Each experiment is a script run in phases, for example

```
python scripts/turner_nullspace_tr0.py
python scripts/certfold.py --phase build-basis
```

The registered experiments refuse to run out of order. A phase checks that the tree is clean,
that its predecessor's record is committed, and that the predictions block (`PRED`) in the
script still equals the one committed with the registration; the registration is always
committed before the analyses it governs (for certified folding, before any test sequence was
certified; earlier experiments had folded test sequences). Large intermediate files (bases, shards) are not
committed: each `metrics.json` records their sha256, and rerunning the phase rebuilds them.

`make test` runs the tests. Some tests read a run's intermediate files, so they need that run
rebuilt first; for example `tests/test_certfold.py` needs
`python scripts/certfold.py --phase build-basis`.

## Provenance

Every `metrics.json` records the git commit it was produced at (`git_sha`). Those are commits
of the authors' working repository; this repository is a filtered copy of its history that
keeps only the code and results the paper uses, so its commits have different hashes.
`docs/remote_commit_map.tsv` maps each working-repository commit to the commit here that
carries its code.

## Certified folding

For each base pair of the Turner 2004 MFE structure, certified folding returns a label
(certified present, certified absent or uncertain) and a verified certified radius. Each label
holds for every parameter table θ0 + Bt in a set built from the data, by a proof that assumes ViennaRNA's recursions are exact (checked, not proved): the directions TR0's
reference structures cannot see, restricted to the int11, int21, int22 and bulge tables, and
bounded by the envelope of the published tables' data-invisible differences (or by a ball,
for comparison).

The certificate charges each interior loop by its worst-case shift: +⌈w⌉ on the loops of the
prediction and −⌈w⌉ on every other loop. It does this through a compiled ViennaRNA
soft-constraint callback, so one constrained fold per pair decides that pair.
`docs/certfold_reading_rules.md` and the paper's supplement give the guarantee and its proof.

| file | what it does |
|---|---|
| `scripts/certfold.py` | the basis, the callback, the certificate, the radius; the soundness gate (G1 to G6) and the registration (PRED) |
| `scripts/certfold_data.py` | the test sets: TS0, one sequence per held-out Rfam family, a seeded bpRNA-new sample |
| `scripts/certfold_baselines.py` | base-pair probability, MEA, γ-centroid, pair frequencies under envelope and noise draws |
| `scripts/certfold_metrics.py` | the registered accuracy metrics (slip-tolerant PPV, matched coverage, risk-coverage, paired bootstrap) |
| `scripts/certfold_run.py` | the test-set run and the registered analysis |
| `scripts/certfold_exploratory.py` | the exploratory analyses: the registered test within length strata, a length baseline |
| `scripts/certfold_posthoc.py` | the one post hoc check, against the energy gap |

**Environment.** The callback is compiled with numba, which lives in its own environment,
`.venv-certfold`: Python 3.11.15, numba 0.68.0 and llvmlite 0.50.0, every other package at the
versions above, and the same ViennaRNA 2.7.2 build. The recipe is at the top of `.gitignore`.
`--phase env-check` verifies the build against the main environment.

**Order.** Every phase refuses to run unless its predecessor's record is committed:

```
.venv-certfold/bin/python scripts/certfold.py --phase build-basis
.venv-certfold/bin/python scripts/certfold.py --phase gate-g1      # likewise gate-g2, g3, g6, g4
.venv-certfold/bin/python scripts/certfold.py --phase env-check
.venv-certfold/bin/python scripts/certfold.py --phase preregister  # PRED: before any test sequence is certified
.venv-certfold/bin/python scripts/certfold.py --phase gate-attacks # G5, on the test sets
.venv-certfold/bin/python scripts/certfold_run.py --phase run
.venv-certfold/bin/python scripts/certfold_run.py --phase analyse
.venv-certfold/bin/python scripts/certfold_exploratory.py --phase exploratory
.venv-certfold/bin/python scripts/certfold_posthoc.py --phase posthoc-gap
```

**Where the results are.** Everything is in `runs/2026-10-13_certified_folding/metrics.json`:

| key | contents | status |
|---|---|---|
| `gate` | G1 to G6 | gate |
| `predictions` | the registration | registered |
| `run` | labels, radii, baselines | registered |
| `analysis` | R1 to R3, the disagreement sets, fractions by length stratum | registered |
| `exploratory` | the within-strata checks, the length baseline | exploratory |
| `posthoc` | the energy-gap check | post hoc |

**Reading and order of events.**
- `docs/certfold_exploratory.md` specified the exploratory analyses before any accuracy existed.
- `docs/certfold_reading_rules.md` holds the authors' rules for reading the results, with a
  dated addendum for the post hoc check.
- The test-set run was launched before those rules arrived. The paper's supplement discloses
  the order of events.

**Registration 2, the additions and the reviewer's analyses.** Three later scripts follow the
same discipline: a registration or specification committed before the folds it governs, phases
that refuse to run out of order, and records written only by the scripts.

| file | what it does | record | status |
|---|---|---|---|
| `scripts/certfold_additions.py` | A: how many uncertain pairs move under envelope draws; B: runtime | `runs/2026-10-13_certified_folding/metrics.json` → `additions` | descriptive |
| `scripts/certfold_reg2.py` | C: stability under whole published tables (R4, R5); D: which measurements would certify the most (R6) | `runs/2026-10-15_certfold_registration2/metrics.json` | registered (PRED2) |
| `scripts/certfold_reviewer.py` | the reviewer's additions: absolute stability, leave one table out, scale, coverage of the published tables, the held-out basis, the measurement-efficiency curve, the ranking's stability, conservativeness, the radius against the gap | `runs/2026-10-18_certfold_reviewer/metrics.json` | post hoc |
| `scripts/certfold_unread.py` | the four tables' unread entries: the envelope with the int11/int21/int22/bulge entries no TR0 reference reads also free, each by up to the largest published difference on it; G1, G2, G4 and G5 rerun; TS0 labels | `runs/2026-10-19_certfold_unread/metrics.json` | post hoc |
| `scripts/certfold_reproduce_r4r5.py` | an independent reimplementation of R4 and R5 from the raw shards | `runs/2026-10-18_certfold_reviewer/reproduce_r4r5/` | check (117 of 117 bit-identical) |

```
.venv-certfold/bin/python scripts/certfold_additions.py --phase uncertain-change  # then runtime
.venv-certfold/bin/python scripts/certfold_reg2.py --phase rank-entries            # TR0 only
.venv-certfold/bin/python scripts/certfold_reg2.py --phase preregister2            # PRED2
.venv-certfold/bin/python scripts/certfold_reg2.py --phase gate-g2-measured
.venv-certfold/bin/python scripts/certfold_reg2.py --phase fold-tables             # then analyse-c
.venv-certfold/bin/python scripts/certfold_reg2.py --phase measured-labels         # then analyse-d
.venv-certfold/bin/python scripts/certfold_reviewer.py --phase descriptive        # then pilot2-rank,
                                                    # gates, ts0-folds, analyse
.venv-certfold/bin/python scripts/certfold_unread.py --phase describe          # then gates,
                                                    # attacks, run, analyse
.venv-certfold/bin/python scripts/certfold_reproduce_r4r5.py OUTDIR
```

**Frozen configurations.** Registration 1: `PRED` in `scripts/certfold.py`, committed in its
record under `predictions`. Registration 2: `PRED2` in `scripts/certfold_reg2.py`, with the
frozen measurement ranking and the sha256 of the MXfold2 predictions it reads. The reviewer's
additions: `docs/certfold_reviewer_additions.md`, committed before any of them ran, with a dated
correction. The unread-entries addition: `docs/certfold_unread_addition.md`, committed before
any of its code. Each record carries the commit it ran at, its seeds and the sha256 of every
intermediate file it read.

Registration 2's tertiary outcome reads the released MXfold2's TS0 predictions, which ship here:
`runs/2026-10-07_mxfold2_ts0_inference/pred_A_cpu.txt` (sha256 `b846b1ab...`, pinned in PRED2),
one line per TS0 sequence with its bpRNA identifier and predicted dot-bracket structure, and no
sequences. They are the output of MXfold2 at commit 51b2136 (MIT licence, K. Sato) with its
released TrainSetAB model (`models/TrainSetAB.conf`, `TrainSetAB.pth`), run on CPU on TS0 from
the MXfold2 data release (Sato, Zenodo 2021, <https://doi.org/10.5281/zenodo.4430150>, CC BY 4.0;
structures derived from bpRNA-1m, Danaee et al. 2018). Rerunning that model on TS0 regenerates
them, checked against the pinned sha256.

**Cost.** The test-set run (5,429 sequences, seven uncertainty sets) took 78 CPU-hours on 8
cores. `make test` covers the certified-folding modules; `tests/test_certfold.py` needs the
build-basis shard, and its compiled-callback test needs `.venv-certfold`.

## Licence

MIT, see `LICENSE`. ViennaRNA, MXfold2 and the datasets are used under their own licences and
are not redistributed here.
