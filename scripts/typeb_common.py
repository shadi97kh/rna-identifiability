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


# ----------------------------------------------------------------------------------
# folding a set of rows under a list of tables (E2 held-out folds; E3 full published tables)
# ----------------------------------------------------------------------------------
INF_DCAL = 10_000_000


def point_tables(names_norms):
    """Written fold points as tables: [(id, theta dict, lxc)] for (kind, name, norm) triples,
    re-checked on read (fold_directions) and taken from the sha-verified points.npz."""
    from scripts.turner_par import template

    fold_directions()  # the read-time assertions
    meta = json.loads((FOLD / "points.json").read_text())
    Z = np.load(FOLD / "points.npz")
    B = load_basis("tr0ts0")
    T = template()
    out = []
    for kind, name, norm in names_norms:
        m = next(mm for mm in meta if mm["kind"] == kind and mm["name"] == name
                 and abs(mm["norm_kcal"] - norm) < 1e-9)
        th = Z["theta_r"][m["p"]]
        full = dict(T["theta"]) | {k: int(th[i]) for i, k in enumerate(B["cols"])}
        out.append({"id": f"{kind}:{name}@{norm:g}", "kind": "point", "theta": full,
                    "lxc": T["lxc"], "p": m["p"], "norm_kcal": m["norm_kcal"]})
    return out


def full_tables():
    """The shipped published tables, whole: [(id, theta dict, lxc, loader)]."""
    from scripts.turner_typeb import SETS, dump_set

    out = []
    for X, loader in SETS.items():
        P, h = dump_set(loader)
        out.append({"id": f"full:{X}", "kind": "full", "theta": P["theta"], "lxc": P["lxc"],
                    "loader": loader, "par_sha256": h})
    return out


def _carry(table, gen_counts, cols, theta0):
    """The 5 constructed structures whose energy the table changes most, avoiding INF entries."""
    th = np.array([table["theta"][k] for k in cols], dtype=np.int64)
    t0 = np.array([theta0[k] for k in cols], dtype=np.int64)
    inf = th >= INF_DCAL
    G = gen_counts
    usable = np.asarray((G[:, inf] != 0).sum(axis=1)).ravel() == 0 if inf.any() else \
        np.ones(G.shape[0], dtype=bool)
    ch = np.where(usable, np.abs(G @ (np.where(inf, 0, th) - t0)), -1)
    top = np.argsort(-ch, kind="stable")[:5]
    assert ch[top[0]] > 0, f"{table['id']}: no constructed structure sees the table"
    return [(int(r), int(G[r].dot(th)[0])) for r in top]


def ninio_correction(db: str, m: int, cap: int) -> int:
    """Exact Ninio term minus the count model's linearisation, summed over a structure's
    interior loops (dcal/mol).

    ViennaRNA adds MIN2(cap, asym * m) to every interior loop except bulges and the 1x1, 1x2
    and 2x2 table loops. The count model (turner_nullspace.Extractor) linearises the term at
    Turner 2004's m = 60, cap = 300: asym * m for asymmetry up to 5, the cap from 6. That is
    exact for Turner 2004. For a table whose cap binds at another asymmetry (DNA m = 40;
    Langdon 2018 m = 80) the difference is this correction."""
    n = len(db)
    pt = [0] * (n + 2)
    st = []
    for i, c in enumerate(db, start=1):
        if c == "(":
            st.append(i)
        elif c == ")":
            j = st.pop()
            pt[i], pt[j] = j, i
    corr = 0
    for i in range(1, n + 1):
        j = pt[i]
        if j <= i:
            continue
        inner, k = [], i + 1
        while k < j:
            if pt[k] > k:
                inner.append((k, pt[k]))
                k = pt[k] + 1
            else:
                k += 1
        if len(inner) != 1:
            continue
        k, l_ = inner[0]
        ul, ur = k - i - 1, j - l_ - 1
        if ul == 0 or ur == 0:
            continue  # stack or bulge
        ns, nl = min(ul, ur), max(ul, ur)
        if (ns, nl) in ((1, 1), (1, 2), (2, 2)):
            continue
        asym = nl - ns
        corr += min(cap, asym * m) - (asym * m if asym <= 5 else cap)
    return corr


def fold_rows_under_tables(rows, tables, dst, w, W):
    """Shard w of W of `rows` [(group, name, seq, db)]: the Turner 2004 baseline, then every
    table, each loaded with the fold's guards. JSON lines to dst: one baseline record, then one
    record per table with per-sequence changed / f1_ref / reference-energy change (dcal)."""
    import contextlib
    import tempfile

    import RNA

    from scripts.turner_nullspace import Extractor, build_matrix, generating_set
    from scripts.turner_par import _rebuild, loaded, params_match, template, write_par
    from scripts.turner_typeb import _vec, f1
    from thermo.vienna_ref import assert_turner2004_params, reference_md

    T = template()
    theta0, special = T["theta"], T["special"]
    cols = load_basis("tr0ts0")["cols"]
    idx = {k: i for i, k in enumerate(cols)}
    gen = generating_set(special)
    G = build_matrix([Extractor(special, 0, 0).counts(s, d) for _, s, d in gen], cols)
    md = reference_md()
    mine = [r for i, r in enumerate(rows) if i % W == w]
    fh = open(dst, "w")  # noqa: SIM115

    def emit(rec):
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    def counts(ex, seq, s):
        c = ex.counts(seq, s)
        off = c.pop(("_offset",), 0)
        return _vec(c, idx), off

    ex0 = Extractor(special, 0, 0, lxc=T["lxc"])
    _rebuild()
    base = []
    for n, (grp, name, seq, db) in enumerate(mine):
        fc = RNA.fold_compound(seq, md)
        if n == 0:
            assert_turner2004_params(fc)
        s0, _ = fc.mfe()
        base.append({"group": grp, "name": name, "seq": seq, "db": db, "s0": s0,
                     "eref0": int(round(fc.eval_structure(db) * 100)),
                     "cref": counts(ex0, seq, db), "f1_ref0": f1(s0, db)})
    emit({"baseline": [{k: b[k] for k in ("group", "name", "eref0", "f1_ref0")}
                       | {"len": len(b["seq"])} for b in base]})

    @contextlib.contextmanager
    def named(loader):
        getattr(RNA, loader)()
        _rebuild()
        try:
            yield
        finally:
            RNA.params_load_RNA_Turner2004()
            _rebuild()
            assert_turner2004_params(RNA.fold_compound("GGGGAAACCCC", reference_md()))

    with tempfile.TemporaryDirectory() as t:
        for tb in tables:
            th = tb["theta"]
            thv = np.array([th[k] for k in cols], dtype=np.int64)
            t0v = np.array([theta0[k] for k in cols], dtype=np.int64)
            ex = Extractor(special, 0, 0, lxc=tb["lxc"])
            nm, ncap = th[("ninio_m",)], th[("ninio_max",)]
            carry = _carry(tb, G, cols, theta0)
            ctx = (loaded(write_par(th, pathlib.Path(t) / "x.par")) if tb["kind"] == "point"
                   else named(tb["loader"]))
            with ctx:
                first = RNA.fold_compound(mine[0][2], md)
                bad = params_match(first, th, 0, 0)
                if bad:
                    raise SystemExit(f"{tb['id']}: params_match {bad}")
                for r, want in carry:
                    _, s, d = gen[r]
                    got = int(round(RNA.fold_compound(s, md).eval_structure(d) * 100))
                    if got != want:
                        raise SystemExit(f"{tb['id']}: carry row {r}: {got} != {want}")
                rec = {"table": tb["id"], "changed": [], "f1_ref": [], "dref_dcal": [],
                       "n_reference_forbidden": 0}
                for b in base:
                    seq, db = b["seq"], b["db"]
                    (ri, rv), roff = b["cref"]
                    (qi, qv), qoff = counts(ex, seq, db)
                    forbidden = bool(np.any(thv[qi] >= INF_DCAL))
                    eref = int(round(RNA.fold_compound(seq, md).eval_structure(db) * 100))
                    # the reference energy under this table, exactly from counts (with the
                    # table's lxc); a reference using an INF entry is a forbidden motif there
                    if not forbidden and eref != (int(qv @ thv[qi]) + qoff
                                                  + ninio_correction(db, nm, ncap)):
                        raise SystemExit(f"{tb['id']}: {b['name']} reference-energy identity")
                    rec["n_reference_forbidden"] += forbidden
                    s1, e1 = RNA.fold_compound(seq, md).mfe()
                    (si, sv), soff = counts(ex, seq, s1)
                    if (int(sv @ thv[si]) + soff + ninio_correction(s1, nm, ncap)
                            != int(round(e1 * 100))):
                        raise SystemExit(f"{tb['id']}: {b['name']} MFE-energy identity")
                    rec["changed"].append(s1 != b["s0"])
                    rec["f1_ref"].append(round(f1(s1, db), 6))
                    rec["dref_dcal"].append(None if forbidden else eref - b["eref0"])
                    # the change equals c(ref).(theta_table - theta_0) exactly (same lxc for
                    # points; for full tables lxc may differ, so only the identity above holds)
                    if tb["kind"] == "point" and eref - b["eref0"] != int(rv @ (thv[ri] - t0v[ri])):
                        raise SystemExit(f"{tb['id']}: {b['name']} reference-change identity")
                emit(rec)
    fh.write(json.dumps({"done": True}) + "\n")
    fh.close()
    return 0
