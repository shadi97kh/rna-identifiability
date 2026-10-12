"""Shared by the Type B follow-up experiments (E1 scripts/typeb_fisher.py, E2
scripts/typeb_heldout.py): the fold's directions, rebuilt and re-checked on read.

The fold (runs/2026-10-12_turner_typeb) stored every point's unrounded perturbation
`delta_dcal` and the written table `theta_r` in fold/points.npz (gitignored; sha256 recorded in
the artifact). A unit direction is delta/||delta|| at the 1.0 kcal/mol point. Because the
written table rounds most entries to zero at that scale, using theta_r by mistake would give
wrong directions and nothing downstream would notice. So every direction is RE-ASSERTED on read
(the user's instruction, 2026-10-12):
  null and published-component directions  |M v|_inf <= 1e-9 on all 11,955 references
  contrast directions                       orthogonal to the nontrivial and intrinsic bases
                                            (<= 1e-9) with support only in int22, int21, int11
                                            and bulge
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from scripts.turner_typeb import FOLD, NULL_TABLES, TOL, load_basis, read_out, sha256


def fold_directions(norm: float = 1.0):
    """{(kind, name): unit vector over the basis columns (kcal/mol coordinates)}, kinds null,
    published, contrast; plus the basis it was checked against."""
    art = read_out()
    for f, h in art["fold"]["points_sha256"].items():
        assert sha256(FOLD / f) == h, f"{f} changed since the fold"
    meta = json.loads((FOLD / "points.json").read_text())
    Z = np.load(FOLD / "points.npz")
    B = load_basis("tr0ts0")
    used, A, NT, NI = B["used"], B["A"], B["NT"], B["I"]
    mask = np.array([B["cols"][i][0] in NULL_TABLES for i in used])
    out = {}
    for m in meta:
        if m["kind"] not in ("null", "published", "contrast") or m["norm_kcal"] != norm:
            continue
        d = Z["delta_dcal"][m["p"]]
        v = d / np.linalg.norm(d)
        vu = v[used]
        if m["kind"] in ("null", "published"):
            worst = float(np.abs(A @ vu).max())
            assert worst <= TOL, f"{m['kind']} {m['name']}: |M v| = {worst} > {TOL}"
        else:
            o = max(float(np.abs(NT.T @ vu).max()), float(np.abs(NI.T @ vu).max()))
            assert o <= TOL, f"contrast {m['name']}: overlap with the null space {o}"
            outside = np.ones(len(v), dtype=bool)
            outside[used[mask]] = False
            assert not np.any(v[outside]), f"contrast {m['name']}: support outside the tables"
        out[(m["kind"], m["name"])] = v
    counts = {k: sum(1 for kk in out if kk[0] == k) for k in ("null", "published", "contrast")}
    assert counts == {"null": 8, "published": 3, "contrast": 8}, counts
    return out, B


def tracked_sha(path: pathlib.Path) -> str:
    return sha256(path)
