# Reference implementations, and what each one is actually a model of

Two external partition functions are available to check `thermo/mccaskill.py`
against. They implement *different* energy models, and neither matches the
other out of the box. This file records what each one is, so that a
disagreement can be read as a real bug rather than a configuration difference.

## ViennaRNA 2.7.2 — the correctness gate

Pinned in [`vienna_ref.py`](vienna_ref.py); go through `ensemble_free_energy`,
never through a bare `RNA.md()`. The model and the reasoning are in that
module's docstring. The three settings worth repeating here:

- **`dangles = 0`** — the default is 2, and on a 73mer the two differ by
  0.31 kcal/mol, six times the gate.
- **`special_hp = 0`** — the default is 1. Pinned off to match a first-version
  DP with no tetraloop/triloop/hexaloop tables. Worth 0.89 kcal/mol on a UUCG
  tetraloop.
- **Turner 2004** — verified per call by fingerprinting `fold_compound.params`,
  because `RNA.params_load_*` is process-global and order-dependent.

Also note that `RNA.pf_fold(seq)` ignores `model_details` entirely and reads
global `RNA.cvar` state, so it is always the dangles=2 model and is never a
valid oracle here. `test_reference_is_not_the_global_pf_fold` keeps that true.

## jax-rnafold — a differentiability reference, not a drop-in second gate

Vendored at `vendor/jax-rnafold` (JAX-RNAfold 2.0.0-beta, commit `13f08b7`).
CPU-only, in `.venv-jax`; never import torch there.

The McCaskill-style entry point is `d0/ss.py::get_ss_partition_fn(em, seq_len)`,
which sums over **all structures** and returns `Q` for a one-hot `p_seq` of
shape `[n, 4]`. Do not confuse it with `d2/seq_pf.py::get_seq_partition_fn`,
which is the opposite object: a fixed dot-bracket structure summed over
*sequences*. Only `ss.py` is a folding oracle. `probe_jax_grad.py` deliberately
uses `seq_pf`, because it is testing gradient flow, not folding.

What its NN model implements:

| Term | Status |
| --- | --- |
| Hairpin, stack, bulge, internal loop | full Turner, `MAX_LOOP = 30` (matches ViennaRNA) |
| Internal loop special cases | `int11`, `int21`, `int22`, Ninio asymmetry |
| Multiloop | linear, `a + b·branches`, no per-unpaired-nt cost |
| Terminal AU/GU penalty | yes |
| Dangles / terminal mismatches in ext+multi loops | `d2` only; `d0` has none |
| Coaxial stacking | **not implemented** |
| All-C hairpin penalty | **not implemented** (matches ViennaRNA's own omission) |
| Special hairpins | **always on, no flag** |
| Lonely pairs | always allowed, i.e. `noLP = 0` |
| Temperature | **37 °C only** — see below |

Three constraints that decide how it can be used:

1. **Special hairpins cannot be switched off.** The bonus tables are baked into
   the `Model` base class and applied by the DP recursion itself. This directly
   conflicts with our `special_hp = 0` pin. Until `mccaskill.py` implements the
   tables, jax-rnafold is only comparable on sequences containing none of the 22
   Turner 2004 special loops — so it is not a substitute for the ViennaRNA gate.
2. **37 °C only.** The `dH` enthalpy columns of the `.par` files are never
   parsed; `temp` only sets `beta`. At any other temperature it computes
   `exp(-dG37/RT)` rather than ViennaRNA's enthalpy extrapolation. The library's
   own test suite carries a `FIXME: fails for t_celsius != 37.0`.
3. **Prefer `d0`; `d2` is the weaker reference.** Measured against ViennaRNA on
   n = 25–30, `d0` agrees to a relative error of 1e-9…1e-7, but `d2` reaches
   ~3e-5 on some sequences. The vendored suite never validates the `d2`
   partition function against ViennaRNA at all — only its per-structure energy
   evaluator, and only to 4 decimal places — so that gap is unguarded upstream.

Operational notes: `seq_len` is compiled into the returned function, so build
one per length. `JaxNNModel()` allocates a 4000×4000 float64 asymmetry matrix
(~128 MB) unless you pass a smaller `max_precompute`. `d0` takes a `scale`
argument (use `-1.5 * seq_len`) to avoid overflow and you must undo it;
`d2` has no rescaling and will overflow float64 on long sequences.

## Consequence for the gate

`test_torch_partition_function_matches_vienna` compares against ViennaRNA only.
If jax-rnafold is later promoted to a second gate, `special_hp` has to become 1
in `REFERENCE_MODEL` *and* the special-hairpin tables have to exist in
`mccaskill.py`. Change both together, and never the tolerance.
