#!/usr/bin/env python3
"""Pool the published stack tables into one gauge-aligned set. Which alignment transfers?

THE QUESTION

`docs/gauge.md` establishes that each published nearest-neighbour table sits in
its OWN gauge: a uniform shift of every stack by c, absorbed by initiation at
-n_bar*c, moves no predicted duplex free energy, so the shift is unobservable
within a table and arbitrary between tables. Pooling tables for a single encoder
therefore hands the same motif targets that disagree by a quantity which is
convention rather than chemistry.

So: can an atom-level encoder trained on the pooled tables predict a chemistry
family it has never seen, and which way of removing the per-table shift makes
that work? The primary held-out target is the RNA:DNA hybrid family.

A DIFFERENT MODEL CLASS FROM THE PAPER'S, DELIBERATELY

`thermo.nn_params.LearnedNNParams` predicts a RESIDUAL against a family baseline
and multiplies it by a hard modification mask, so Delta_phi is exactly zero on
canonical chemistry. That is the right architecture for the paper's claim and the
wrong one here: with the hybrid held out, only 16 masked-in stacks would remain
(the 2'-OMe table), and the experiment would measure sample size. The head below
predicts ABSOLUTE (dH, dS) with no mask and no baseline. Nothing under `models/`,
`datasets/` or `thermo/` is modified; this is a new evaluation path only.

The head emits two channels and dG37 is DERIVED as dH - T*dS, per
`.claude/rules/validation.md`: "never add dG as an independent head". Arm (d)
subtracts each table's coordinate in each channel separately, and the per-channel
coordinates reconstruct the dG37 coordinate to 1e-15 (checked below, with a
negative control).

THE ALIGNMENT ARMS

  a         raw published targets, pooled as published
  b         plus one learned (dH, dS) offset per training table
  c-joint   plus per-table PER-NUCLEOTIDE offsets, learned jointly with the
            encoder. A stack's offset is the sum over its four nucleoside slots.
  c-frozen  the same token offsets, but fitted by least squares on each training
            table's own targets FIRST and then frozen, with the encoder trained
            on the residual. This is the AtomRef recipe (Huang et al. 2025) and
            SevenNet-Omni's shift initialisation.
  d         targets minus that table's SVD gauge coordinate, per channel, in four
            variants: {c_free, c_16} x {every table at its lowest-c, at its
            highest-c configuration}. c_native is excluded because
            `runs/2026-08-25_conditioning_table/` marks it not comparable across
            tables.
  e         one learned embedding per training table, width 8, concatenated to
            the encoder features before the head (UMA-style task conditioning).
            A held-out table gets the MEAN of the training embeddings. This
            deliberately breaks docs/PLAN.md P4's "no learned context embedding"
            rule, in a model class that is not the paper's.

A held-out table receives NO learned offset and NO embedding of its own, because
it has no training rows to fit one from. That is the whole difficulty.

THE METRIC IS CENTRED, BECAUSE A RAW METRIC WOULD SCORE THE GAUGE

Centred dG37 RMSE: the mean over the held-out table's own stacks is subtracted
from prediction and target alike, so a uniform per-table shift -- which IS the
gauge -- cannot be scored either way. Raw RMSE is reported beside it and labelled
gauge-dependent; it is not what any verdict below rests on. Spearman is reported
because it is invariant to shift and scale together.

PREDICTIONS, RECORDED BEFORE THE FIRST FULL RUN AND NOT EDITED AFTERWARDS

  1. b and d tie on centred RMSE, because a per-table uniform shift IS the gauge
     and the two remove it by different means (learned vs measured by SVD).
  2. c-joint transfers WORSE than d, because per-nucleotide offsets absorb real
     chemistry and not only the convention.
  3. a is worst within a family, judged against {b, c-joint, d}, the set the
     prediction was made against. Its rank among all arms is reported too.
  4. c-joint's held-out centred RMSE has a seed SD more than 2x the larger of
     b's and d's. The reason is that a shared GC slope trades freely against the
     per-table GC offsets, a flat direction the conditioning layer itself
     creates -- the token-count matrix of a homoduplex table has rank 2, spanning
     {1, n_GC}, so only a constant and a GC slope are identifiable.
  5. c-frozen is STABLE across seeds but loses the GC effect: its held-out
     centred RMSE is at least the RMS of the held-out table's centred component
     along the G-C pair count.
  6. e transfers worse than d, by the stated rule.

DECISION RULES, FIXED IN ADVANCE

  - "d's seed SD" means the LARGEST seed SD among the four d variants, which is
    the conservative choice for prediction 4.
  - a tie means |delta mean| < the larger of the two seed SDs
  - "worse" means delta mean > the larger of the two seed SDs
  - "stable" (prediction 5) means seed SD <= the larger of b's and d's
  - verdicts are issued on the PRIMARY (hybrid) split; the leave-one-out splits
    are reported as "k of 8 consistent"
  Nothing is tuned to make a prediction hold.

PRE-REGISTERED CONDITIONAL ON OFFSET INITIALISATION

`TableStackOffset` in `scripts/reference_head_experiment.py` initialises to
exactly zero, and c-joint's token offsets here do the same, so every seed starts
its offsets at the same point and seed variation enters only through the encoder
and head weights. The head's last layer is `nn.Linear(64, 2)` with PyTorch's
default initialisation, U(-1/sqrt(64), +1/sqrt(64)) = U(-0.125, 0.125).

IF PREDICTION 4 IS REFUTED, c-joint is re-run once with its offsets drawn from
U(-0.125, 0.125) under the same 8 seeds and 9 splits (72 fits) BEFORE any
conclusion is written about the flat direction, and both runs are reported
whatever they show. If prediction 4 is confirmed, no re-run is done and the
artifact says so.

Run:        PYTHONPATH=. .venv/bin/python scripts/pooled_alignment.py
Smoke:      PYTHONPATH=. .venv/bin/python scripts/pooled_alignment.py --smoke
Time 1 fit: PYTHONPATH=. .venv/bin/python scripts/pooled_alignment.py --time-one
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from scipy import stats
from torch_geometric.data import Batch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from models.encoder import N_STACK_SLOTS, StackEncoder  # noqa: E402
from models.stacks import TABLES_BY_KEY  # noqa: E402
from models.supervision import (  # noqa: E402
    INCREMENT_TABLES,
    _canonical_key,
    baseline_channels,
    parameter_rows,
)
from scripts.nn_projection import ALL16, exact_rank  # noqa: E402
from scripts.robustness_checks import DISTINCT_TOL  # noqa: E402
from scripts.runmeta import git_provenance  # noqa: E402
from thermo.nn_params import T37, delta_g  # noqa: E402

R = pathlib.Path(__file__).resolve().parents[1]
RUN_ID = "2026-10-06_pooled_alignment"
RAW = R / "data" / "raw" / "thermo"
CONDITIONING = R / "runs" / "2026-08-25_conditioning_table" / "metrics.json"

DTYPE = torch.float64
# Hyperparameters copied from scripts/reference_head_experiment.py so this run is
# comparable with it rather than separately tuned.
HIDDEN = 64
FEATURE_DIM = 32
LAYERS = 3
STEPS = 600
LR = 3e-3
EMBED_DIM = 8
SEEDS = (20261006, 20261007, 20261008, 20261009, 20261010, 20261011, 20261012, 20261013)
# The head's last layer initialisation, for the pre-registered conditional.
OFFSET_RAND_HALFWIDTH = 1.0 / (HIDDEN**0.5)

# The 10 tables with a defined gauge coordinate. Asserted against the
# conditioning artifact rather than trusted from this list.
POOL_KEYS = (
    "allawi_santalucia1997_dna_nn",
    "breslauer1986_dna_nn",
    "santalucia1996_dna_nn",
    "santalucia_dna_nn",
    "sugimoto1996_dna_nn",
    "tanaka2004_dna_nn",
    "freier1986_rna_nn",
    "turner_nndb",
    "kierzek2006_2ome_nn",
    "sugimoto_hybrid_nn",
)
PRIMARY_HELD_OUT = "sugimoto_hybrid_nn"
HETERODUPLEX = ("kierzek2006_2ome_nn", "sugimoto_hybrid_nn")

ARMS = (
    ("a", None),
    ("b", None),
    ("c-joint", None),
    ("c-frozen", None),
    ("d", "c_free_low"),
    ("d", "c_free_high"),
    ("d", "c_16_low"),
    ("d", "c_16_high"),
    ("e", None),
)

PREDICTIONS = {
    "1": {
        "claim": "b and d tie on held-out centred RMSE",
        "reason": "a per-table uniform shift is the gauge; b learns it, d measures it by SVD",
        "rule": "tie if |delta mean| < max(seed SD of b, seed SD of d)",
    },
    "2": {
        "claim": "c-joint transfers worse than d",
        "reason": "per-nucleotide offsets absorb real chemistry, not only convention",
        "rule": "worse if delta mean > max(seed SD of the two)",
    },
    "3": {
        "claim": "a is worst within a family, against {b, c-joint, d}",
        "reason": "the same motif carries targets differing by the gauge",
        "rule": "worst if its centred RMSE mean exceeds every member of that set",
    },
    "4": {
        "claim": "c-joint's seed SD is more than 2x the larger of b's and d's",
        "reason": (
            "a shared GC slope trades freely against per-table GC offsets, a flat "
            "direction the conditioning layer creates; a homoduplex token-count "
            "matrix has rank 2 spanning {1, n_GC}"
        ),
        "rule": "confirmed if sd(c-joint) > 2 * max(sd(b), largest sd over the four d variants)",
    },
    "5": {
        "claim": (
            "c-frozen is stable across seeds but loses the GC effect: its centred "
            "RMSE is at least the RMS of the held-out table's centred component "
            "along the G-C pair count"
        ),
        "reason": "the frozen offsets are absent for the held-out table, GC slope included",
        "rule": "stable if sd <= max(sd(b), sd(d)); floor is the GC-projection RMS",
    },
    "6": {
        "claim": "e transfers worse than d",
        "reason": "a mean embedding is not the held-out table's embedding",
        "rule": "worse if delta mean > max(seed SD of the two)",
    },
    "conditional_on_prediction_4": {
        "offset_initialisation": (
            "arm b and arm c-joint offsets are initialised to exactly zero, matching "
            "scripts/reference_head_experiment.py:87, so seed variation enters only "
            "through the encoder and head weights"
        ),
        "head_last_layer_initialisation": (
            f"nn.Linear({HIDDEN}, 2) default, U(-{OFFSET_RAND_HALFWIDTH}, "
            f"+{OFFSET_RAND_HALFWIDTH})"
        ),
        "rule": (
            "if prediction 4 is refuted, re-run c-joint once with offsets drawn from "
            "that same uniform scheme over the same 8 seeds x 9 splits (72 fits) "
            "before writing any conclusion about the flat direction, and report both"
        ),
    },
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class LeakageError(RuntimeError):
    """A held-out table reached a training batch."""


def pool_records():
    """Buildable published stacks for the 10 gauge-bearing tables, by table."""
    records, skipped = parameter_rows()
    for key in POOL_KEYS:
        if key in INCREMENT_TABLES:
            raise AssertionError(
                f"{key} publishes increments, not absolute stacks; it cannot enter a "
                "pool of absolute targets without its baseline, which would reintroduce "
                "the family term this experiment is measuring"
            )
    by_table: dict[str, list] = {k: [] for k in POOL_KEYS}
    for rec in records:
        if rec.source_key in by_table:
            by_table[rec.source_key].append(rec)
    for key, rows in by_table.items():
        if not rows:
            raise AssertionError(f"{key} contributed no buildable stack")
    return by_table, skipped


def coordinates():
    """Per-channel gauge coordinates per table, at its lowest-c and highest-c config.

    Read from runs/2026-08-25_conditioning_table/, never recomputed: the point of
    arm (d) is that the quantity was measured by a singular value decomposition
    that never saw this model.
    """
    data = json.loads(CONDITIONING.read_text())
    out: dict[str, dict] = {}
    recon: list[dict] = []
    for entry in data["tables"]:
        key = entry["registry_key"]
        variants = entry.get("gauge_coordinate_variants") or []
        if not variants or key not in POOL_KEYS or key in out:
            continue
        ch_h = entry["channels"]["delta_h_kcal"]["variants"]
        ch_s = entry["channels"]["delta_s_cal_per_k"]["variants"]
        per_variant = []
        for i, v in enumerate(variants):
            row = {
                "configuration": v["configuration"],
                "c_free_dg": v["c_arity_free"],
                "c_16_dg": v["c_projection_16"],
                "c_free_dh": ch_h[i]["c_arity_free"],
                "c_free_ds": ch_s[i]["c_arity_free"],
                "c_16_dh": ch_h[i]["c_projection_16"],
                "c_16_ds": ch_s[i]["c_projection_16"],
            }
            per_variant.append(row)
            for name in ("c_free", "c_16"):
                recon.append(
                    {
                        "registry_key": key,
                        "configuration": v["configuration"],
                        "coordinate": name,
                        "residual": row[f"{name}_dh"]
                        - T37 * row[f"{name}_ds"] / 1000.0
                        - row[f"{name}_dg"],
                    }
                )
        out[key] = {
            "variants": per_variant,
            "c_free_low": min(per_variant, key=lambda r: r["c_free_dg"]),
            "c_free_high": max(per_variant, key=lambda r: r["c_free_dg"]),
            "c_16_low": min(per_variant, key=lambda r: r["c_16_dg"]),
            "c_16_high": max(per_variant, key=lambda r: r["c_16_dg"]),
        }
    missing = sorted(set(POOL_KEYS) - set(out))
    if missing:
        raise AssertionError(f"no gauge coordinate in the artifact for {missing}")
    return out, recon


def coordinate_checks(coords):
    """cH - T*cS/1000 == c_dG37, with a cross-table control that MUST fail."""
    recon = []
    for entry in coords.values():
        for v in entry["variants"]:
            for name in ("c_free", "c_16"):
                recon.append(
                    abs(v[f"{name}_dh"] - T37 * v[f"{name}_ds"] / 1000.0 - v[f"{name}_dg"])
                )
    worst = max(recon)
    keys = sorted(coords)
    control = []
    for i, ki in enumerate(keys):
        kj = keys[(i + 1) % len(keys)]
        vi = coords[ki]["variants"][0]
        vj = coords[kj]["variants"][0]
        control.append(abs(vi["c_free_dh"] - T37 * vj["c_free_ds"] / 1000.0 - vi["c_free_dg"]))
    return {
        "max_abs_reconstruction_residual": worst,
        "tolerance": 1e-12,
        "passes": worst < 1e-12,
        "control_cross_table_max": max(control),
        "control_fails_as_required": max(control) > 1e-12,
        "n_checked": len(recon),
    }


def token_matrix(rows):
    """(n_stacks, n_tokens) counts over the four NUCLEOSIDE slots, and the tokens."""
    tokens = sorted({t for r in rows for t in r.spec.slots()[:4]})
    index = {t: i for i, t in enumerate(tokens)}
    m = np.zeros((len(rows), len(tokens)), dtype=np.int64)
    for i, r in enumerate(rows):
        for t in r.spec.slots()[:4]:
            m[i, index[t]] += 1
    return m, tokens


def gc_count(rec) -> int:
    """G/C pairs in the stack: the two top residues, in RNA letters."""
    top = _canonical_key(rec.motif).split("/")[0]
    return sum(1 for b in top if b in "GC")


def dg37(dh, ds) -> float:
    return float(delta_g(dh, ds, T37))


# ---------------------------------------------------------------------------
# the deterministic table: no training anywhere in here
# ---------------------------------------------------------------------------

def variance_share(values: np.ndarray, basis: np.ndarray) -> float:
    """Share of centred variance of `values` lying in the column span of `basis`."""
    y = values - values.mean()
    if not np.any(y):
        return 0.0
    b = basis - basis.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(b)
    proj = q @ (q.T @ y)
    return float((proj @ proj) / (y @ y))


def deterministic_gc_table(by_table):
    """Share of centred dG37 variance in span{1, n_GC}, with a permutation control.

    The null expectation for one regressor on n points is 1/(n-1), NOT zero, so
    that is reported beside the control rather than left for the reader to supply.
    """
    out = {}
    for key in POOL_KEYS:
        rows = by_table[key]
        listed = {_canonical_key(r.motif): (dg37(r.target_dh, r.target_ds), gc_count(r))
                  for r in rows}
        weightings = {}

        # As listed. For a canonical table that is the 10 flip orbits; for the two
        # heteroduplex tables it is already the native 16, so "10-stack" does not
        # exist and is recorded as null rather than faked.
        n_listed = len(listed)
        if n_listed == 10:
            weightings["10_stack"] = listed
        else:
            weightings["10_stack"] = None

        # Flip-expanded to 16 for canonical tables, native 16 for the others. The
        # expansion duplicates orbit partners, which inflates n without adding
        # information; the null expectation below is what makes that visible.
        sixteen = {}
        for motif in ALL16:
            top, bottom = motif.split("/")
            flip = f"{bottom[1]}{bottom[0]}/{top[1]}{top[0]}"
            hit = listed.get(motif, listed.get(flip))
            if hit is not None:
                sixteen[motif] = (hit[0], sum(1 for b in top if b in "GC"))
        weightings["16_stack"] = sixteen if len(sixteen) == 16 else None

        entry = {"n_listed": n_listed}
        for name, table in weightings.items():
            if table is None:
                entry[name] = None
                continue
            vals = np.array([v[0] for v in table.values()], dtype=np.float64)
            gc = np.array([[v[1]] for v in table.values()], dtype=np.float64)
            n = len(vals)
            shares = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                shares.append(variance_share(vals, gc[rng.permutation(n)]))
            entry[name] = {
                "n": n,
                "share_in_span_1_nGC": variance_share(vals, gc),
                "permuted_control_mean": float(np.mean(shares)),
                "permuted_control_max": float(np.max(shares)),
                "null_expectation_one_regressor": 1.0 / (n - 1),
            }

        # What c-frozen actually removes: the full token span. For a homoduplex
        # table that span IS {1, n_GC} at rank 2, so the column above already is
        # this quantity; for the two heteroduplex tables it is rank 4 and larger.
        m, tokens = token_matrix(rows)
        vals = np.array([dg37(r.target_dh, r.target_ds) for r in rows], dtype=np.float64)
        entry["token_span"] = {
            "tokens": tokens,
            "exact_rank": exact_rank(m),
            "numpy_rank": int(np.linalg.matrix_rank(m)),
            "share_in_full_token_span": variance_share(vals, m.astype(np.float64)),
            "equals_span_1_nGC_by_construction": exact_rank(m) == 2,
        }
        out[key] = entry
    return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class AbsoluteStackHead(torch.nn.Module):
    """Encoder features -> absolute (dH, dS). No mask, no baseline, by design.

    Same width and activation as thermo.nn_params.LearnedNNParams, so the only
    deliberate difference is that nothing multiplies the output by a modification
    mask and nothing adds a published baseline to it.
    """

    def __init__(self, in_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class TableTokenOffset(torch.nn.Module):
    """Per-table, per-nucleoside-token (dH, dS) offset; a stack sums its four slots.

    Initialised to exactly zero unless `randomise` is set, which is the
    pre-registered conditional for prediction 4 and nothing else.
    """

    def __init__(self, tokens_by_table, dtype=DTYPE, randomise=False, generator=None):
        super().__init__()
        self.index = {}
        flat = []
        for key, tokens in sorted(tokens_by_table.items()):
            for t in tokens:
                self.index[(key, t)] = len(flat)
                flat.append((key, t))
        self.weight = torch.nn.Parameter(torch.zeros(len(flat), 2, dtype=dtype))
        if randomise:
            with torch.no_grad():
                self.weight.uniform_(
                    -OFFSET_RAND_HALFWIDTH, OFFSET_RAND_HALFWIDTH, generator=generator
                )

    def design(self, records) -> torch.Tensor:
        """(n_stacks, n_params) token counts, zero for any token not in training."""
        m = torch.zeros(len(records), self.weight.shape[0], dtype=self.weight.dtype)
        for i, r in enumerate(records):
            for t in r.spec.slots()[:4]:
                j = self.index.get((r.source_key, t))
                if j is not None:
                    m[i, j] += 1.0
        return m

    def forward(self, design: torch.Tensor) -> torch.Tensor:
        return design @ self.weight


class TableEmbedding(torch.nn.Module):
    """One learned vector per training table; a held-out table gets the mean."""

    def __init__(self, keys, dim=EMBED_DIM, dtype=DTYPE):
        super().__init__()
        self.keys = list(dict.fromkeys(keys))
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.weight = torch.nn.Parameter(torch.zeros(len(self.keys), dim, dtype=dtype))

    def forward(self, source_keys) -> torch.Tensor:
        mean = self.weight.mean(dim=0, keepdim=True)
        rows = [
            self.weight[self.index[k]].unsqueeze(0) if k in self.index else mean
            for k in source_keys
        ]
        return torch.cat(rows, dim=0)


class BatchedStacks:
    """The graph batch for a fixed stack list, built once instead of per step.

    112 stacks name 560 slots, but those slots draw on a vocabulary of about
    seventeen distinct molecules, and the encoder is a pure function of the
    molecule. So each DISTINCT molecule is message-passed once per step and the
    result is gathered back to the slots, which is the same arithmetic over a
    thirty-third of the graphs. `scripts/encoder_gauge_seeds.py` takes the same
    route for the same reason.

    Verified against StackEncoder.forward on construction and again after every
    fit: a fast path that is not the same model is worse than no fast path.
    """

    def __init__(self, encoder: StackEncoder, records):
        self.records = list(records)
        tokens = [t for r in self.records for t in r.spec.slots()]
        self.vocab = sorted(set(tokens))
        index = {t: i for i, t in enumerate(self.vocab)}
        self.batch = Batch.from_data_list([encoder.graph_for(t) for t in self.vocab])
        self.gather = torch.tensor([index[t] for t in tokens], dtype=torch.long)

    def features(self, encoder: StackEncoder) -> torch.Tensor:
        embedded = encoder.nucleoside(self.batch)[self.gather]
        stacked = embedded.reshape(
            len(self.records), N_STACK_SLOTS * encoder.nucleoside.hidden
        )
        return encoder.combine(stacked)

    def verify(self, encoder: StackEncoder) -> float:
        with torch.no_grad():
            fast = self.features(encoder)
            slow = encoder([r.spec for r in self.records])
        return float((fast - slow).abs().max())


# ---------------------------------------------------------------------------
# splits and leakage
# ---------------------------------------------------------------------------

def splits():
    """The primary family hold-out, then leave-one-table-out inside each family."""
    out = [("hybrid", PRIMARY_HELD_OUT)]
    by_family: dict[str, list[str]] = {}
    for key in POOL_KEYS:
        by_family.setdefault(TABLES_BY_KEY[key].family, []).append(key)
    for family, keys in sorted(by_family.items()):
        if len(keys) < 2:
            continue
        for key in keys:
            out.append((f"loo:{family}", key))
    return out


def assert_no_heldout_in_training(train_records, held_key, *, where):
    bad = sorted({r.source_key for r in train_records} & {held_key})
    if bad:
        raise LeakageError(f"held-out table {bad} reached {where}")


def leakage_control(by_table):
    """The leakage check must FIRE on an injected held-out row, or it is vacuous."""
    train = [r for k, rows in by_table.items() if k != PRIMARY_HELD_OUT for r in rows]
    assert_no_heldout_in_training(train, PRIMARY_HELD_OUT, where="control/clean")
    tripped = False
    try:
        assert_no_heldout_in_training(
            train + [by_table[PRIMARY_HELD_OUT][0]], PRIMARY_HELD_OUT, where="control/injected"
        )
    except LeakageError:
        tripped = True
    return {"clean_passes": True, "injected_row_trips_the_check": tripped}


# ---------------------------------------------------------------------------
# one fit
# ---------------------------------------------------------------------------

def targets_for(records, arm, variant, coords):
    """(dh, ds) targets after whatever arm (d) subtracts. Other arms are untouched."""
    dh = np.array([r.target_dh for r in records], dtype=np.float64)
    ds = np.array([r.target_ds for r in records], dtype=np.float64)
    if arm != "d":
        return dh, ds
    name = "c_free" if variant.startswith("c_free") else "c_16"
    for i, r in enumerate(records):
        cfg = coords[r.source_key][variant]
        dh[i] -= cfg[f"{name}_dh"]
        ds[i] -= cfg[f"{name}_ds"]
    return dh, ds


def frozen_token_offsets(train_records, tokens_by_table):
    """Per-table least-squares token offsets, minimum norm, fitted then frozen."""
    offsets = {}
    for key, tokens in tokens_by_table.items():
        rows = [r for r in train_records if r.source_key == key]
        m, _ = token_matrix(rows)
        order = {t: i for i, t in enumerate(sorted({t for r in rows
                                                   for t in r.spec.slots()[:4]}))}
        cols = np.zeros((len(rows), len(tokens)), dtype=np.float64)
        for j, t in enumerate(tokens):
            if t in order:
                cols[:, j] = m[:, order[t]]
        y = np.stack(
            [
                np.array([r.target_dh for r in rows], dtype=np.float64),
                np.array([r.target_ds for r in rows], dtype=np.float64),
            ],
            axis=1,
        )
        sol, *_ = np.linalg.lstsq(cols, y, rcond=None)
        offsets[key] = (cols, sol, rows)
    return offsets


def fit_one(by_table, coords, arm, variant, held_key, seed, *, steps=STEPS,
            randomise_offsets=False):
    """One fit. Returns held-out predicted (dh, ds) and the training loss."""
    train = [r for k, rows in by_table.items() if k != held_key for r in rows]
    held = list(by_table[held_key])
    assert_no_heldout_in_training(train, held_key, where=f"{arm}/{variant}/{held_key}")

    train_keys = sorted({r.source_key for r in train})
    tokens_by_table = {
        k: sorted({t for r in train if r.source_key == k for t in r.spec.slots()[:4]})
        for k in train_keys
    }

    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    in_dim = FEATURE_DIM + (EMBED_DIM if arm == "e" else 0)
    encoder = StackEncoder(hidden=HIDDEN, out_dim=FEATURE_DIM, layers=LAYERS).to(DTYPE)
    head = AbsoluteStackHead(in_dim).to(DTYPE)
    params = list(encoder.parameters()) + list(head.parameters())

    tr_batch = BatchedStacks(encoder, train)
    he_batch = BatchedStacks(encoder, held)
    # The vocabulary fast path must BE StackEncoder.forward, not merely resemble
    # it. Checked on the untrained weights of every fit, so a divergence cannot
    # ride along unnoticed; the bound is float64 scatter-order round-off.
    for batch in (tr_batch, he_batch):
        agreement = batch.verify(encoder)
        if not agreement < 1e-10:
            raise AssertionError(
                f"the batched encoder path is not the same model: max abs diff {agreement}"
            )

    offset = embed = None
    tr_design = he_design = None
    if arm in ("b", "c-joint"):
        if arm == "b":
            tokens_b = {k: ["__table__"] for k in train_keys}
            offset = TableTokenOffset(tokens_b, randomise=randomise_offsets, generator=gen)
            tr_design = torch.zeros(len(train), offset.weight.shape[0], dtype=DTYPE)
            for i, r in enumerate(train):
                tr_design[i, offset.index[(r.source_key, "__table__")]] = 1.0
            he_design = torch.zeros(len(held), offset.weight.shape[0], dtype=DTYPE)
        else:
            offset = TableTokenOffset(
                tokens_by_table, randomise=randomise_offsets, generator=gen
            )
            tr_design = offset.design(train)
            he_design = offset.design(held)
        params += list(offset.parameters())
    elif arm == "e":
        embed = TableEmbedding(train_keys)
        with torch.no_grad():
            embed.weight.uniform_(-OFFSET_RAND_HALFWIDTH, OFFSET_RAND_HALFWIDTH,
                                  generator=gen)
        params += list(embed.parameters())

    dh, ds = targets_for(train, arm, variant, coords)
    if arm == "c-frozen":
        frozen = frozen_token_offsets(train, tokens_by_table)
        for key, (cols, sol, rows) in frozen.items():
            idx = {id(r): i for i, r in enumerate(rows)}
            pred = cols @ sol
            for i, r in enumerate(train):
                if r.source_key == key:
                    dh[i] -= pred[idx[id(r)], 0]
                    ds[i] -= pred[idx[id(r)], 1]

    t_dh = torch.tensor(dh, dtype=DTYPE)
    t_ds = torch.tensor(ds, dtype=DTYPE)
    sh = t_dh.std().clamp(min=1e-6)
    ss = t_ds.std().clamp(min=1e-6)
    tr_keys = [r.source_key for r in train]
    he_keys = [r.source_key for r in held]

    opt = torch.optim.Adam(params, lr=LR)

    def predict(batch, design, keys):
        feats = batch.features(encoder)
        if embed is not None:
            feats = torch.cat([feats, embed(keys)], dim=1)
        out = head(feats)
        p_dh, p_ds = out[:, 0], out[:, 1]
        if offset is not None:
            o = offset(design)
            p_dh, p_ds = p_dh + o[:, 0], p_ds + o[:, 1]
        return p_dh, p_ds

    loss_value = float("nan")
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        p_dh, p_ds = predict(tr_batch, tr_design, tr_keys)
        loss = (((p_dh - t_dh) / sh) ** 2).mean() + (((p_ds - t_ds) / ss) ** 2).mean()
        loss.backward()
        opt.step()
        loss_value = float(loss.detach())

    with torch.no_grad():
        p_dh, p_ds = predict(he_batch, he_design, he_keys)
    return p_dh.numpy(), p_ds.numpy(), loss_value, tr_batch.verify(encoder)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def centred(x: np.ndarray) -> np.ndarray:
    return x - x.mean()


def centred_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    d = centred(pred) - centred(target)
    return float(np.sqrt((d * d).mean()))


def centred_metric_checks(rng_seed=SEEDS[0]):
    """Centring must kill a per-table constant and must NOT kill per-stack noise."""
    rng = np.random.default_rng(rng_seed)
    pred = rng.normal(size=16)
    target = rng.normal(size=16)
    base = centred_rmse(pred, target)
    shifted = centred_rmse(pred + 3.7, target)
    noised = centred_rmse(pred + rng.normal(scale=0.5, size=16), target)
    return {
        "uniform_shift_change": abs(shifted - base),
        "tolerance": 1e-12,
        "passes": abs(shifted - base) < 1e-12,
        "control_per_stack_noise_change": abs(noised - base),
        "control_changes_as_required": abs(noised - base) > 1e-12,
    }


def heldout_targets(records, arm, variant, coords):
    """Held-out dG37 in the arm's own target space, and in the published space."""
    dh = np.array([r.target_dh for r in records], dtype=np.float64)
    ds = np.array([r.target_ds for r in records], dtype=np.float64)
    published = np.array([dg37(a, b) for a, b in zip(dh, ds, strict=True)])
    if arm != "d":
        return published, published, 0.0
    name = "c_free" if variant.startswith("c_free") else "c_16"
    cfg = coords[records[0].source_key][variant]
    shift = cfg[f"{name}_dg"]
    return published - shift, published, shift


def baselines(by_table, held_key, coords):
    """Everything the encoder must beat on this split."""
    held = by_table[held_key]
    target = np.array([dg37(r.target_dh, r.target_ds) for r in held])
    out = {}

    train = [r for k, rows in by_table.items() if k != held_key for r in rows]
    const = float(np.mean([dg37(r.target_dh, r.target_ds) for r in train]))
    pred = np.full_like(target, const)
    out["training_mean_null"] = {
        "centred_rmse": centred_rmse(pred, target),
        "raw_rmse": float(np.sqrt(((pred - target) ** 2).mean())),
        "spearman": None,
        "spearman_note": "undefined: a constant has no rank order",
    }

    if held_key == PRIMARY_HELD_OUT:
        dna = baseline_channels("dna_dna")
        rna = baseline_channels("rna_rna")
        vals, keep = [], []
        for i, r in enumerate(held):
            k = _canonical_key(r.motif)
            if k in dna and k in rna:
                vals.append(0.5 * (dg37(*dna[k]) + dg37(*rna[k])))
                keep.append(i)
        if len(keep) == len(held):
            p = np.array(vals)
            out["dna_rna_mean"] = {
                "centred_rmse": centred_rmse(p, target),
                "raw_rmse": float(np.sqrt(((p - target) ** 2).mean())),
                "spearman": float(stats.spearmanr(p, target).statistic),
                "n": len(keep),
            }
        else:
            out["dna_rna_mean"] = {"blocked_reason": f"only {len(keep)}/{len(held)} matched"}

    family = TABLES_BY_KEY[held_key].family
    siblings = [k for k in POOL_KEYS
                if k != held_key and TABLES_BY_KEY[k].family == family]
    if siblings:
        vals, keep = [], []
        for i, r in enumerate(held):
            k = _canonical_key(r.motif)
            others = [
                dg37(o.target_dh, o.target_ds)
                for s in siblings
                for o in by_table[s]
                if _canonical_key(o.motif) == k
            ]
            if others:
                vals.append(float(np.mean(others)))
                keep.append(i)
        if len(keep) == len(held):
            p = np.array(vals)
            out["motif_mean_same_family"] = {
                "centred_rmse": centred_rmse(p, target),
                "raw_rmse": float(np.sqrt(((p - target) ** 2).mean())),
                "spearman": float(stats.spearmanr(p, target).statistic),
                "n_sibling_tables": len(siblings),
            }
        else:
            out["motif_mean_same_family"] = {
                "blocked_reason": f"only {len(keep)}/{len(held)} motifs matched"
            }

    gc = np.array([gc_count(r) for r in held], dtype=np.float64)
    y = centred(target)
    b = (gc - gc.mean()).reshape(-1, 1)
    q, _ = np.linalg.qr(b)
    proj = q @ (q.T @ y)
    out["heldout_gc_projection_rms"] = float(np.sqrt((proj * proj).mean()))
    out["heldout_gc_slope_of_target"] = float(np.polyfit(gc, target, 1)[0])
    return out


def evaluate(by_table, coords, arm, variant, held_key, *, steps=STEPS,
             randomise_offsets=False):
    held = by_table[held_key]
    arm_target, published, shift = heldout_targets(held, arm, variant, coords)
    gc = np.array([gc_count(r) for r in held], dtype=np.float64)

    per_seed, readouts = [], []
    verify_worst = 0.0
    for seed in SEEDS:
        p_dh, p_ds, loss, verify = fit_one(
            by_table, coords, arm, variant, held_key, seed,
            steps=steps, randomise_offsets=randomise_offsets,
        )
        verify_worst = max(verify_worst, verify)
        pred = np.array([dg37(a, b) for a, b in zip(p_dh, p_ds, strict=True)])
        readouts.append(pred)
        per_seed.append(
            {
                "seed": seed,
                "train_loss": loss,
                "centred_rmse": centred_rmse(pred, arm_target),
                "raw_rmse_published_space": float(
                    np.sqrt(((pred + shift - published) ** 2).mean())
                ),
                "spearman": float(stats.spearmanr(pred, arm_target).statistic),
                "gc_slope_of_prediction": float(np.polyfit(gc, centred(pred), 1)[0]),
            }
        )

    rmse = np.array([s["centred_rmse"] for s in per_seed])
    stacked = np.array(readouts)
    distinct = []
    for i in range(len(stacked)):
        if not any(np.abs(stacked[i] - stacked[j]).max() < DISTINCT_TOL for j in distinct):
            distinct.append(i)
    return {
        "arm": arm if variant is None else f"{arm}:{variant}",
        "held_out": held_key,
        "n_held_out_stacks": len(held),
        "centred_rmse_mean": float(rmse.mean()),
        "centred_rmse_sd": float(rmse.std(ddof=1)),
        "raw_rmse_published_space_mean": float(
            np.mean([s["raw_rmse_published_space"] for s in per_seed])
        ),
        "raw_rmse_is_gauge_dependent": True,
        "spearman_mean": float(np.mean([s["spearman"] for s in per_seed])),
        "gc_slope_of_prediction_mean": float(
            np.mean([s["gc_slope_of_prediction"] for s in per_seed])
        ),
        "gc_slope_of_prediction_sd": float(
            np.std([s["gc_slope_of_prediction"] for s in per_seed], ddof=1)
        ),
        "n_distinct_seed_solutions": len(distinct),
        "n_seeds": len(SEEDS),
        "coordinate_shift_added_back_for_raw": shift,
        "batched_path_max_abs_diff": verify_worst,
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------

def verdicts(primary: dict[str, dict]) -> dict:
    """The six predictions, judged by the rules fixed in the docstring."""
    def mean(name):
        return primary[name]["centred_rmse_mean"]

    def sd(name):
        return primary[name]["centred_rmse_sd"]

    d_names = [f"d:{v}" for _, v in ARMS if _ == "d"]
    d_best = min(d_names, key=mean)
    d_sd_max = max(sd(n) for n in d_names)
    out = {"d_variant_used_for_comparison": d_best,
           "d_variant_largest_seed_sd": d_sd_max,
           "d_variant_spread_of_means": max(mean(n) for n in d_names)
                                        - min(mean(n) for n in d_names)}

    delta = mean("b") - mean(d_best)
    tol = max(sd("b"), d_sd_max)
    out["1"] = {"delta_mean": delta, "tolerance": tol,
                "verdict": "confirmed" if abs(delta) < tol else "refuted"}

    delta = mean("c-joint") - mean(d_best)
    tol = max(sd("c-joint"), d_sd_max)
    out["2"] = {"delta_mean": delta, "tolerance": tol,
                "verdict": "confirmed" if delta > tol else "refuted"}

    against = {n: mean(n) for n in ("b", "c-joint", d_best)}
    out["3"] = {
        "a_mean": mean("a"),
        "compared_against": against,
        "rank_among_all_arms": 1 + sum(1 for n in primary if mean(n) < mean("a")),
        "n_arms": len(primary),
        "verdict": "confirmed" if all(mean("a") > v for v in against.values()) else "refuted",
    }

    floor = 2.0 * max(sd("b"), d_sd_max)
    out["4"] = {"c_joint_sd": sd("c-joint"), "threshold": floor,
                "verdict": "confirmed" if sd("c-joint") > floor else "refuted"}

    out["6"] = None
    delta = mean("e") - mean(d_best)
    tol = max(sd("e"), d_sd_max)
    out["6"] = {"delta_mean": delta, "tolerance": tol,
                "verdict": "confirmed" if delta > tol else "refuted"}
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="one split, one seed, few steps; writes nothing")
    ap.add_argument("--time-one", action="store_true",
                    help="time a single full-length fit; writes nothing")
    args = ap.parse_args()

    by_table, skipped = pool_records()
    coords, _ = coordinates()
    print("=" * 96)
    print("POOLED GAUGE ALIGNMENT  --  can an encoder transfer to an unseen chemistry family?")
    print("=" * 96)
    print(f"\n  pool: {len(POOL_KEYS)} gauge-bearing tables, "
          f"{sum(len(v) for v in by_table.values())} buildable stacks")
    for key in POOL_KEYS:
        print(f"    {key:30} {TABLES_BY_KEY[key].family:9} {len(by_table[key]):3} stacks")

    if args.smoke:
        print("\nSMOKE: one split, one seed, 5 steps, nothing written")
        for arm, variant in ARMS:
            p_dh, p_ds, loss, verify = fit_one(
                by_table, coords, arm, variant, PRIMARY_HELD_OUT, SEEDS[0], steps=5
            )
            name = arm if variant is None else f"{arm}:{variant}"
            print(f"    {name:16} loss {loss:10.4f}  "
                  f"batched-vs-slow {verify:.2e}  pred[0] dG37 "
                  f"{dg37(p_dh[0], p_ds[0]):+.4f}")
        print("\n  smoke ok")
        return 0

    if args.time_one:
        t0 = time.perf_counter()
        _, _, loss, verify = fit_one(
            by_table, coords, "c-joint", None, PRIMARY_HELD_OUT, SEEDS[0]
        )
        dt = time.perf_counter() - t0
        total = len(ARMS) * len(splits()) * len(SEEDS)
        print(f"\n  one fit ({STEPS} steps): {dt:.2f} s   final loss {loss:.4f}"
              f"   batched-vs-slow {verify:.2e}")
        print(f"  projected for {total} fits: {total * dt / 60:.1f} min")
        print(f"  plus the 72-fit conditional re-run: {72 * dt / 60:.1f} min")
        return 0

    dst = R / "runs" / RUN_ID
    dst.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": RUN_ID,
        "dtype": "float64",
        "hidden": HIDDEN, "feature_dim": FEATURE_DIM, "layers": LAYERS,
        "steps": STEPS, "lr": LR, "embed_dim": EMBED_DIM,
        "seeds": list(SEEDS),
        "pool_keys": list(POOL_KEYS),
        "primary_held_out": PRIMARY_HELD_OUT,
        "arms": [a if v is None else f"{a}:{v}" for a, v in ARMS],
        "splits": [f"{kind}/{key}" for kind, key in splits()],
    }
    # The predictions land in the artifact BEFORE any training, so they cannot be
    # edited to match what came out.
    (dst / "metrics.json").write_text(
        json.dumps({"config": config, "predictions": PREDICTIONS}, indent=2, sort_keys=True)
        + "\n"
    )
    print(f"\n  pre-registered predictions written to {dst / 'metrics.json'}")

    out = {"config": config, "predictions": PREDICTIONS, "skipped_stacks": skipped}
    out["checks"] = {
        "coordinate_reconstruction": coordinate_checks(coords),
        "centred_metric_invariance": centred_metric_checks(),
        "leakage": leakage_control(by_table),
    }
    for name, chk in out["checks"].items():
        print(f"    check {name}: {chk}")
    out["deterministic_gc_table"] = deterministic_gc_table(by_table)
    out["coordinates_used"] = coords

    results: dict[str, dict[str, dict]] = {}
    t0 = time.perf_counter()
    for kind, held_key in splits():
        split_name = f"{kind}/{held_key}"
        results[split_name] = {"baselines": baselines(by_table, held_key, coords), "arms": {}}
        for arm, variant in ARMS:
            name = arm if variant is None else f"{arm}:{variant}"
            res = evaluate(by_table, coords, arm, variant, held_key)
            results[split_name]["arms"][name] = res
            print(f"  {split_name:34} {name:16} "
                  f"centred {res['centred_rmse_mean']:.4f} +/- {res['centred_rmse_sd']:.4f}"
                  f"  rho {res['spearman_mean']:+.3f}"
                  f"  distinct {res['n_distinct_seed_solutions']}/{res['n_seeds']}")
    out["results"] = results
    out["elapsed_seconds"] = time.perf_counter() - t0

    primary = results[f"hybrid/{PRIMARY_HELD_OUT}"]["arms"]
    v = verdicts(primary)
    floor = results[f"hybrid/{PRIMARY_HELD_OUT}"]["baselines"]["heldout_gc_projection_rms"]
    v["5"] = {
        "c_frozen_sd": primary["c-frozen"]["centred_rmse_sd"],
        "stability_threshold": max(
            primary["b"]["centred_rmse_sd"], v["d_variant_largest_seed_sd"]
        ),
        "c_frozen_mean": primary["c-frozen"]["centred_rmse_mean"],
        "gc_projection_rms_floor": floor,
        "verdict": "confirmed" if (
            primary["c-frozen"]["centred_rmse_sd"]
            <= max(primary["b"]["centred_rmse_sd"], v["d_variant_largest_seed_sd"])
            and primary["c-frozen"]["centred_rmse_mean"] >= floor
        ) else "refuted",
    }
    out["verdicts"] = v
    print("\n  verdicts on the primary split:")
    for k in ("1", "2", "3", "4", "5", "6"):
        print(f"    prediction {k}: {v[k]['verdict']}")

    if v["4"]["verdict"] == "refuted":
        print("\n  prediction 4 refuted -> running the pre-registered 72-fit re-run")
        rerun = {}
        for kind, held_key in splits():
            rerun[f"{kind}/{held_key}"] = evaluate(
                by_table, coords, "c-joint", None, held_key, randomise_offsets=True
            )
        out["conditional_rerun_c_joint_random_offsets"] = {
            "triggered": True,
            "results": rerun,
            "primary_sd": rerun[f"hybrid/{PRIMARY_HELD_OUT}"]["centred_rmse_sd"],
        }
    else:
        out["conditional_rerun_c_joint_random_offsets"] = {
            "triggered": False,
            "reason": "prediction 4 was confirmed, so the re-run condition did not fire",
        }

    out.update(git_provenance(R))
    (dst / "metrics.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    if out["predictions"] != PREDICTIONS:
        raise AssertionError("the predictions block changed during the run")
    print(f"\nwrote {dst / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
