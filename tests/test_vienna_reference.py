"""Lock down the ViennaRNA reference before it is used to judge the torch DP.

The correctness gate in CLAUDE.md says the torch partition function must match
ViennaRNA to 0.05 kcal/mol on unmodified sequences of length 21 to 120. That
sentence only means something once "ViennaRNA" names one specific model, so
these tests pin it and keep it pinned. The reference model, defined in
thermo/vienna_ref.py:

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

dangles=0 and special_hp=0 are the two that are pinned away from ViennaRNA's own
defaults, because they describe what a first-version torch DP will implement.
`test_defaults_would_break_the_gate` measures what leaving them at their
defaults would cost, so that nobody later "fixes" the reference back to the
defaults and spends a week hunting an implementation bug that is not there.
Both should be revisited together with the DP: when the DP grows dangles or the
special-hairpin tables, change REFERENCE_MODEL, never the tolerance.
"""

import importlib.util

import pytest

from thermo.vienna_ref import (
    ENERGY_NEUTRAL_FIELDS,
    REFERENCE_MODEL,
    TOLERANCE_KCAL,
    ModelNotPinned,
    assert_model_fields_classified,
    ensemble_free_energy,
    reference_md,
)

# Ensemble free energies in kcal/mol under the pinned model, on ViennaRNA 2.7.2.
# These are a tripwire, not ground truth: they exist so that a ViennaRNA upgrade
# or a parameter-file change announces itself here rather than as a mysterious
# regression in the DP comparison. If one moves, work out why before touching it.
GOLDEN = {
    "sirna_guide_21": ("UUCUCCGAACGUGUCACGUAA", -1.390904),
    "sirna_passenger_21": ("UUACGUGACACGUUCGGAGAA", -1.512949),
    "uucg_tetraloop_14": ("GGGGCUUCGGCCCC", -8.711866),
    "random_30": ("CGCAUACGUGACGGCCUGCAAUUUCUUUUA", -1.891809),
    "random_60": (
        "GAACGCGGUACGAGAUGUAGGUUGCUAGGUCUUCGUGUACAUACUAAUCUAGUGAUUUGC",
        -14.185592,
    ),
    "trna_73": (
        "GGGCUAUUAGCUCAGUUGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAAUUCAGCAUAGCCC",
        -29.440468,
    ),
    "random_120": (
        "GGUCUUGAGAAAAUGGGACAGUGAGACACAGAUCGUGCACGUACAGAAGUAGGCAUGCUAGUCCGCUGUCUGG"
        "CAGCCCCUCCCAAGUCUAAAGCUCGUUGGGUAAGAACAUAACAUACA",
        -29.339783,
    ),
}


def test_reference_md_sets_every_pinned_field():
    """The md we hand to ViennaRNA really carries the model documented above."""
    md = reference_md()
    for field, expected in REFERENCE_MODEL.items():
        assert getattr(md, field) == expected, f"{field} did not take"


def test_every_model_field_is_classified():
    """Version-drift guard: a new ViennaRNA knob must be triaged, not inherited."""
    assert_model_fields_classified()


def test_pinned_and_neutral_field_sets_are_disjoint():
    overlap = set(REFERENCE_MODEL) & set(ENERGY_NEUTRAL_FIELDS)
    assert not overlap, f"fields claimed both pinned and energy-neutral: {sorted(overlap)}"


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_ensemble_free_energies(name):
    sequence, expected = GOLDEN[name]
    got = ensemble_free_energy(sequence)
    assert got == pytest.approx(expected, abs=1e-4), (
        f"{name}: reference moved from {expected} to {got}. The oracle changed "
        f"underneath the project -- check the ViennaRNA version and parameter files "
        f"before updating this number."
    )


def test_reference_is_not_the_global_pf_fold():
    """RNA.pf_fold ignores model_details, so it can only ever be the dangles=2 model.

    Anything that reaches for it instead of ensemble_free_energy is silently
    comparing the torch DP against a different model.
    """
    import RNA

    sequence = GOLDEN["trna_73"][0]
    assert RNA.pf_fold(sequence)[1] != pytest.approx(
        ensemble_free_energy(sequence), abs=TOLERANCE_KCAL
    ), "RNA.pf_fold now agrees with the pinned model; re-check which model each one uses"


@pytest.mark.parametrize(
    ("field", "value", "min_effect"),
    [
        ("dangles", 2, 0.3),
        ("special_hp", 1, 0.5),
    ],
)
def test_defaults_would_break_the_gate(field, value, min_effect):
    """Leaving these at ViennaRNA's defaults costs far more than the 0.05 gate.

    Recorded as a measurement rather than a comment so the cost stays true. The
    sequences are chosen to be sensitive to the field in question: the tRNA has
    many helix ends for dangles, the UUCG hairpin hits the tetraloop table.
    """
    import RNA

    sequence = GOLDEN["trna_73" if field == "dangles" else "uucg_tetraloop_14"][0]

    md = reference_md()
    setattr(md, field, value)
    perturbed = RNA.fold_compound(sequence, md).pf()[1]

    difference = abs(perturbed - ensemble_free_energy(sequence))
    assert difference > TOLERANCE_KCAL, (
        f"{field}={value} no longer changes the answer; the pin may be pointless now"
    )
    assert difference > min_effect, (
        f"{field}={value} shifts G by {difference:.4f} kcal/mol, less than the "
        f"{min_effect} kcal/mol previously measured"
    )


def test_params_poisoning_is_caught():
    """A stray params_load_* elsewhere in the process must fail loudly, not silently.

    Parameter loading is global and its effect depends on call order, so this
    runs in a subprocess with the load first, where it does take effect.
    """
    import pathlib
    import subprocess
    import sys

    program = (
        "import RNA; RNA.params_load_RNA_Turner1999()\n"
        "from thermo.vienna_ref import ensemble_free_energy, ModelNotPinned\n"
        "try:\n"
        f"    ensemble_free_energy({GOLDEN['trna_73'][0]!r})\n"
        "except ModelNotPinned:\n"
        "    print('CAUGHT')\n"
    )
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_root,
    )
    assert "CAUGHT" in result.stdout, (
        f"Turner 1999 parameters were not rejected.\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


@pytest.mark.parametrize("sequence", ["", "ACGTACGT", "acgu", "ACGUNNNN"])
def test_rejects_sequences_vienna_would_quietly_accept(sequence):
    """ViennaRNA maps T, lowercase and N to unpaired N rather than erroring."""
    with pytest.raises(ValueError):
        ensemble_free_energy(sequence)


def test_model_not_pinned_survives_optimised_mode():
    """The guards must not be assert statements: `python -O` would strip them."""
    assert issubclass(ModelNotPinned, RuntimeError)
    assert not issubclass(ModelNotPinned, AssertionError)


@pytest.mark.skipif(
    importlib.util.find_spec("thermo.mccaskill") is None,
    reason="thermo/mccaskill.py does not exist yet; this is the gate it must pass",
)
def test_torch_partition_function_matches_vienna():
    """The correctness gate itself: torch McCaskill vs the pinned reference.

    Lengths 21 to 120 as specified in CLAUDE.md. Deliberately compares against
    `ensemble_free_energy` rather than a raw ViennaRNA call, so the comparison
    cannot drift onto an unpinned model.
    """
    import random

    from thermo.mccaskill import ensemble_free_energy as torch_free_energy

    rng = random.Random(20260730)
    lengths = [21, 30, 45, 60, 80, 100, 120]
    failures = []

    for length in lengths:
        for _ in range(3):
            sequence = "".join(rng.choice("ACGU") for _ in range(length))
            reference = ensemble_free_energy(sequence)
            torch_value = float(torch_free_energy(sequence))
            if abs(torch_value - reference) > TOLERANCE_KCAL:
                failures.append(
                    f"  len={length} {sequence}\n"
                    f"    vienna={reference:.6f} torch={torch_value:.6f} "
                    f"delta={torch_value - reference:+.6f}"
                )

    assert not failures, (
        f"{len(failures)} sequence(s) outside the {TOLERANCE_KCAL} kcal/mol gate:\n"
        + "\n".join(failures)
    )
