# Same Energies, Different Folds

Code and run artifacts for *Same Energies, Different Folds: Parameter Non-Identifiability in
Thermodynamic and Learned RNA Folding Models* (S. Khodagholi and N. Yousefi; manuscript under
review). Every experimental number the paper reports is read from a `runs/<id>/metrics.json` in this
repository, and each of those files records the commit and the code that produced it.

## What the paper shows

Nearest-neighbor energy models score RNA secondary structures with tables of loop parameters
that are refit, learned and compared entry by entry. The paper describes three kinds of
parameter change that data cannot detect, and an algorithm that turns the second kind into
per-pair guarantees.

| | the change | code |
|---|---|---|
| **Type A** | *Gauge directions* leave every structure's energy unchanged. ViennaRNA's Turner 2004 model has exactly two with dangles off, and any loss computed from structures has zero gradient along them. | `scripts/turner_nullspace.py` |
| **Type B** | *Data-restricted directions* leave the energies of a given set of reference structures unchanged but not those of others: 426 for the 11,955 bpRNA structures in TR0 and TS0. | `scripts/turner_nullspace_tr0.py`, `scripts/turner_typeb*.py`, `scripts/typeb_*.py` |
| **Type C** | In MXfold2, per-sequence changes to the output scores leave every structure's score unchanged, and the loss is almost flat along a weight symmetry. | `scripts/mxfold2_*.py` |
| **Certified folding** | For each base pair of the minimum free energy structure, a label that holds for *every* parameter table in a ball of Type B directions: certified present, certified absent, or uncertain. | `scripts/certfold.py` |

## Where each headline number comes from

| claim in the abstract | artifact |
|---|---|
| 426 data-restricted directions for the 11,955 TR0 and TS0 structures | `runs/2026-10-08_turner_nullspace_tr0/` |
| adding to Turner 2004 only the part of Andronescu 2007 that lies in the 426 directions changes the predicted MFE structure of 3.9% of the 11,970 sequences | `runs/2026-10-12_turner_typeb_tie_split/` |
| likelihood training carries about 9 times less information along these directions | `runs/2026-10-12_typeb_fisher/` |
| on 1,641 held-out Rfam families: 2.9% of predictions change, mean F1 falls by 0.0004 | `runs/2026-10-12_typeb_heldout/` |
| Adam moves MXfold2 along the weight symmetry 33 times more than weight decay | `runs/2026-10-11_mxfold2_weight_decay_ablation/` |
| penalizing the output gauge cuts its seed-to-seed variance about 1,250-fold | `runs/2026-10-08_mxfold2_gauge_eval/` |

The other directories under `runs/` back the remaining numbers in the paper and its
supplement, or are the registrations and inputs those runs name.

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
committed before any test-set data is folded. Large intermediate files (bases, shards) are not
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

## Certified folding: status

`scripts/certfold.py` carries a soundness gate that must pass before any certificate is
reported, and `runs/2026-10-13_certified_folding/metrics.json` records each part as it passes:
the callback identity against ViennaRNA's own interior-loop energy (G1), exhaustive enumeration
at n ≤ 20 with zero false certificates and a planted-bug power check (G2), the pair tables at
realistic lengths (G3), and gauge invariance and monotonicity in the radius (G6). This part of
the work is in progress; the artifact says which gates have passed.

## Licence

MIT, see `LICENSE`. ViennaRNA, MXfold2 and the datasets are used under their own licences and
are not redistributed here.
