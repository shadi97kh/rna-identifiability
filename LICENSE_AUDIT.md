# Licence audit of the public artifact

Audited 2026-10-18 against the allow-lists in this directory (`keep_files.txt`, `keep_runs.txt`,
`keep_sources.txt`), which decide everything `export.sh` publishes. The artifact is still
private; making it public waits for the authors' go.

**Verdict: nothing in the export carries third-party code or redistributes a dataset; one
decision is open** (whether to add the MXfold2 TS0 predictions, below).

## Our code and records

| What | Licence | Checked |
|---|---|---|
| Every script, test and doc on the allow-list | MIT (`LICENSE`) | no kept file carries a third-party copyright, licence or SPDX header; the only match for "copyright / GPL / licensed under / SPDX" is `data/SOURCES.yaml`, which records the datasets' own licences |
| `runs/*/metrics.json` and the other committed run files | MIT, as our own outputs | no dataset sequence is embedded; the only sequences in kept records are constructed test hairpins (`runs/2026-10-10_mxfold2_release_repro`, `runs/2026-10-12_turner_par_writer`). Records name dataset sequences (bpRNA identifiers) and give pair indices and statistics |
| No code copied from a GPL project | n/a | the rule of the repository; `vendor/` (MXfold2, JAX-RNAfold) is not on the allow-list |

## Dependencies, used and not redistributed

| Dependency | Licence | How it is used |
|---|---|---|
| ViennaRNA 2.7.2 | its own: research, educational and commercial use and modification permitted, no redistribution for a fee, credit to the authors required | installed from PyPI; never copied. Its parameter tables are loaded through ViennaRNA itself; perturbed tables are written to temporary files and deleted. Credited by citation (Lorenz et al. 2011) |
| MXfold2 at commit 51b2136 | MIT | an external dependency in its own environment; not included |
| JAX-RNAfold | its own, academic use only | a reference implementation in a local environment; not on the allow-list, not imported by any exported script |
| numpy, scipy, numba, llvmlite, torch, pyyaml, pytest | BSD-style or MIT | installed; not included |

## Data, fetched and checked, not redistributed

| Dataset | Licence | Obligation | Met by |
|---|---|---|---|
| TR0, TS0, bpRNA-new (MXfold2's data release, from bpRNA-1m) | CC BY 4.0 | attribution and an indication of changes | the README's Attribution section (Zenodo record, the MXfold2 and bpRNA papers; the filtering steps) |
| Rfam 12.2 and 15.1 | CC0 1.0 | none (citation requested) | cited in the README |

## Open decision for the authors

**The MXfold2 TS0 predictions** (`runs/2026-10-07_mxfold2_ts0_inference/pred_A_cpu.txt`) are not
committed, yet registration 2's tertiary outcome reads them, and PRED2 pins their sha256
(`b846b1ab...`). They are the released MXfold2 model's (MIT) predictions on TS0 (CC BY 4.0):
dot-bracket structures keyed by bpRNA identifier, with no sequences. Both licences allow
redistributing them with the attribution the README already gives. Two options:
1. add the file to the export (recommended: the tertiary outcome then reproduces from the
   artifact alone); or
2. leave it out and document regeneration with MXfold2 51b2136 and its released TrainSetAB
   model on TS0, checked against the pinned sha256.

Also not on the allow-list, by choice rather than licence: `docs/certfold_reading_report.md` and
`docs/certfold_experiment_report.*`, which carry the authors' draft text and open decisions.
