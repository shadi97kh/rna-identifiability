"""The pinned ViennaRNA reference model for the partition-function correctness gate.

Nothing in this project may call ViennaRNA for a reference value except through
`ensemble_free_energy` here. The reason is that a bare `RNA.md()` is not a
specification -- it is a snapshot of process-global state, and several of its
fields move the ensemble free energy by far more than the 0.05 kcal/mol gate.

The model this file pins, and which the torch DP must implement:

    dangles          0      no dangling-end contributions at all
    temperature      37 C
    parameter set    Turner 2004 (ViennaRNA's compiled-in default)
    special_hp       0      no tetraloop/triloop/hexaloop bonus tables
    noLP             0      lonely (helix-of-one) pairs are allowed
    noGU             0      GU wobble pairs are allowed
    noGUclosure      0      GU pairs may close hairpins and interior loops
    gquad            0      no G-quadruplex module
    circ             0      linear, not circular, RNA
    max_bp_span      -1     unlimited base-pair span
    min_loop_size    3      standard hairpin minimum
    salt             1.021  the reference molarity, i.e. no salt correction
    betaScale        1.0
    pf_smooth        0      no smoothing of the dangle tables
    logML            0      linear multiloop model
    energy_set       0      full ACGU alphabet with standard pairs

Two of these are the classic ways this comparison silently goes wrong:

`dangles` defaults to 2, and the torch DP should implement 0 for its first
version. On a 73mer the two differ by 0.31 kcal/mol -- six times the gate --
which reads exactly like an off-by-one in the DP recursion rather than a model
mismatch. Note also that ViennaRNA's partition function silently treats
dangles=1 and dangles=3 as dangles=2; only 0 and 2 are distinct here.

`special_hp` defaults to 1, adding sequence-specific bonuses for a fixed list of
tetraloops, triloops and hexaloops. On a UUCG tetraloop that is worth
0.89 kcal/mol. A first-pass DP will not have those tables, so the reference is
pinned to 0. Flip it to 1 here, not at the call site, once the DP grows them.

Three process-global hazards this module actively defends against:

1. `RNA.pf_fold(seq)` ignores any `md` you construct -- it reads the global
   `RNA.cvar` settings instead. It can only ever produce the default dangles=2
   model, so it is never a valid oracle here. Use `RNA.fold_compound(seq, md)`.

2. A fresh `RNA.md()` inherits global state. Setting `RNA.cvar.temperature = 25`
   anywhere in the process makes the next unpinned `RNA.md()` a 25 C model, and
   on a 21mer that shifts the answer by 0.74 kcal/mol. Every field above is
   therefore assigned explicitly rather than left at its default, even where the
   default already happens to be the value we want.

3. `RNA.params_load_RNA_Turner1999()` and friends mutate a process-global table,
   and whether a later fold actually picks the change up depends on call order
   within the process. So we never trust the load order: `ensemble_free_energy`
   fingerprints `fold_compound.params`, which is a faithful witness of what that
   specific compound computed with, and refuses to return a number if it is not
   Turner 2004.
"""

from __future__ import annotations

import RNA

__all__ = [
    "ModelNotPinned",
    "REFERENCE_MODEL",
    "ENERGY_NEUTRAL_FIELDS",
    "TOLERANCE_KCAL",
    "reference_md",
    "assert_model_fields_classified",
    "assert_turner2004_params",
    "ensemble_free_energy",
]

#: The correctness gate from CLAUDE.md. Never loosen this to make a test pass.
TOLERANCE_KCAL = 0.05

#: Every model field that can change the ensemble free energy, assigned
#: explicitly. See the module docstring for what the choices mean.
REFERENCE_MODEL: dict[str, object] = {
    "dangles": 0,
    "temperature": 37.0,
    "special_hp": 0,
    "noLP": 0,
    "noGU": 0,
    "noGUclosure": 0,
    "gquad": 0,
    "circ": 0,
    "max_bp_span": -1,
    "min_loop_size": 3,
    "salt": 1.021,
    "betaScale": 1.0,
    "pf_smooth": 0,
    "logML": 0,
    "energy_set": 0,
    "uniq_ML": 0,
}

#: Fields that provably cannot move the ensemble free energy, each with the
#: reason. Kept explicit so that `assert_model_fields_classified` can tell a
#: field we have thought about from one a ViennaRNA upgrade has just added.
ENERGY_NEUTRAL_FIELDS: dict[str, str] = {
    "alias": "comparative folding only (alignment column alphabet)",
    "backbone_length": "G-quadruplex geometry, unused while gquad=0",
    "backtrack": "selects whether a structure is traced back, not its energy",
    "backtrack_type": "as backtrack",
    "circ_alpha0": "circular RNA only, unused while circ=0",
    "circ_penalty": "circular RNA only, unused while circ=0",
    "compute_bpp": "controls base-pair probability output, not the free energy",
    "cv_fact": "comparative folding covariance weight",
    "helical_rise": "G-quadruplex geometry, unused while gquad=0",
    "nc_fact": "comparative folding non-compatible-pair weight",
    "nonstandards": "only consulted when energy_set selects a restricted alphabet",
    "oldAliEn": "comparative folding energy convention",
    "pair": "derived pairing table, regenerated from energy_set/noGU",
    "ribo": "comparative folding ribosum scoring",
    "rtype": "derived reverse-pair-type table",
    "saltDPXInit": "salt correction terms, inert at the reference salt of 1.021 M",
    "saltDPXInitFact": "as saltDPXInit",
    "saltMLLower": "as saltDPXInit",
    "saltMLUpper": "as saltDPXInit",
    "sfact": "numerical scaling of the partition function, cancels in G",
    "this": "SWIG pointer to the underlying vrna_md_t",
    "thisown": "SWIG ownership flag",
    "window_size": "local-folding window, unused outside the *_window entry points",
}

# Turner 2004 dG37 values in dacal/mol, read back from the fold_compound that
# actually computed a number. Turner 1999 differs in hairpin[3] (570), MLclosing
# (340) and MLintern (40), so this discriminates the two parameter sets.
_TURNER_2004_FINGERPRINT: dict[str, int] = {
    "stack[2][2]": -340,
    "stack[1][2]": -330,
    "hairpin[3]": 540,
    "hairpin[4]": 560,
    "bulge[1]": 380,
    "internal_loop[4]": 110,
    "TerminalAU": 50,
    "MLclosing": 930,
    "MLbase": 0,
    "MLintern[1]": -90,
}


class ModelNotPinned(RuntimeError):
    """Raised when ViennaRNA is not demonstrably in the pinned reference model.

    Deliberately not an AssertionError: this must survive `python -O` and must
    never be mistaken for an ordinary failed comparison.
    """


def _fingerprint(params) -> dict[str, int]:
    """Read the discriminating energy terms out of a live parameter table."""
    return {
        "stack[2][2]": params.stack[2][2],
        "stack[1][2]": params.stack[1][2],
        "hairpin[3]": params.hairpin[3],
        "hairpin[4]": params.hairpin[4],
        "bulge[1]": params.bulge[1],
        "internal_loop[4]": params.internal_loop[4],
        "TerminalAU": params.TerminalAU,
        "MLclosing": params.MLclosing,
        "MLbase": params.MLbase,
        "MLintern[1]": params.MLintern[1],
    }


def reference_md() -> RNA.md:
    """Build a model_details with every energy-relevant field explicitly assigned.

    Read back and verify each assignment: some SWIG fields silently ignore
    out-of-range values, and a field that did not take would otherwise leave the
    reference quietly running on a global default.
    """
    md = RNA.md()
    for field, value in REFERENCE_MODEL.items():
        setattr(md, field, value)

    mismatched = {
        field: (value, getattr(md, field))
        for field, value in REFERENCE_MODEL.items()
        if getattr(md, field) != value
    }
    if mismatched:
        detail = ", ".join(
            f"{f}: set {w!r} but read back {g!r}" for f, (w, g) in mismatched.items()
        )
        raise ModelNotPinned(f"ViennaRNA ignored a model_details assignment -- {detail}")

    return md


def assert_model_fields_classified() -> None:
    """Fail if this ViennaRNA exposes a model field we have neither pinned nor cleared.

    This is the version-drift guard. A ViennaRNA upgrade that adds a new energy
    knob would otherwise be picked up silently at its default, and the first
    symptom would be the correctness gate drifting for no visible reason.
    """
    exposed = {
        name
        for name in dir(RNA.md())
        if not name.startswith("_") and not callable(getattr(RNA.md(), name, None))
    }
    unclassified = exposed - set(REFERENCE_MODEL) - set(ENERGY_NEUTRAL_FIELDS)
    if unclassified:
        raise ModelNotPinned(
            f"ViennaRNA {RNA.__version__} exposes model fields this project has not "
            f"classified: {sorted(unclassified)}. Decide for each whether it can change "
            f"the ensemble free energy, then add it to REFERENCE_MODEL (and to the torch "
            f"DP) or to ENERGY_NEUTRAL_FIELDS with the reason."
        )

    stale = (set(REFERENCE_MODEL) | set(ENERGY_NEUTRAL_FIELDS)) - exposed
    if stale:
        raise ModelNotPinned(
            f"ViennaRNA {RNA.__version__} no longer exposes {sorted(stale)}. The pinned "
            f"model refers to fields that do not exist, so it is no longer describing "
            f"the model actually being used."
        )


def assert_turner2004_params(fold_compound) -> None:
    """Fail if the compound did not use Turner 2004 at the pinned temperature.

    Checks `fold_compound.params` rather than the global `RNA.param()`, because a
    `params_load_*` call elsewhere in the process can leave those two disagreeing
    -- and it is the compound's own table that produced the number.
    """
    params = fold_compound.params
    got = _fingerprint(params)
    if got != _TURNER_2004_FINGERPRINT:
        differing = {
            term: (_TURNER_2004_FINGERPRINT[term], got[term])
            for term in _TURNER_2004_FINGERPRINT
            if got[term] != _TURNER_2004_FINGERPRINT[term]
        }
        detail = ", ".join(f"{t}: expected {w}, got {g}" for t, (w, g) in differing.items())
        raise ModelNotPinned(
            f"fold_compound is not using Turner 2004 dG37 parameters -- {detail}. "
            f"Something in this process called RNA.params_load_*; parameter loading is "
            f"global and order-dependent, so the reference cannot be trusted."
        )

    pinned_temperature = REFERENCE_MODEL["temperature"]
    if params.temperature != pinned_temperature:
        raise ModelNotPinned(
            f"fold_compound is at {params.temperature} C, not the pinned "
            f"{pinned_temperature} C. A bare RNA.md() inherits RNA.cvar.temperature, "
            f"so some code has changed the global and the md did not override it."
        )


def ensemble_free_energy(sequence: str) -> float:
    """Reference ensemble free energy G = -RT ln Z in kcal/mol, in the pinned model.

    The only sanctioned way to get a ViennaRNA number out of this project. Every
    guard runs on every call: the cost is a few microseconds against a DP, and
    the failure it prevents is a silent one.
    """
    if not sequence or set(sequence) - set("ACGU"):
        raise ValueError(
            f"expected a non-empty uppercase ACGU sequence, got {sequence!r}. "
            f"ViennaRNA maps unknown characters to unpaired N rather than erroring."
        )

    assert_model_fields_classified()
    fc = RNA.fold_compound(sequence, reference_md())
    assert_turner2004_params(fc)
    _structure, free_energy = fc.pf()
    return free_energy
