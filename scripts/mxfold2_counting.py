"""MXfold2's positional counting rule, with no dependencies beyond the standard library.

Shared by scripts/mxfold2_nullspace.py (step 2, run in .venv) and
scripts/mxfold2_decompose.py (seed-study evaluator, run in .venv-mxfold2, which has
neither ViennaRNA nor scipy). The rule was derived from MXfold2's source at 51b2136;
see the docstring of scripts/mxfold2_nullspace.py. Moved here verbatim on 2026-10-08,
then corrected the same day: type C now counts the empty-interval entry U(a, a-1),
which step 2's rule omitted (see the CORRECTION note in scripts/mxfold2_nullspace.py).
"""

from __future__ import annotations


def pair_table(db: str) -> list[int]:
    pt = [0] * (len(db) + 1)
    st = []
    for k, c in enumerate(db, start=1):
        if c == "(":
            st.append(k)
        elif c == ")":
            i = st.pop()
            pt[i], pt[k] = k, i
    assert not st
    return pt


class MX:
    """Count vector of MXfold2's positional score (model_type 'C' or 'M')."""

    def __init__(self, model_type: str, exterior_stem_term: bool = True):
        assert model_type in ("C", "M")
        self.t = model_type
        self.ext = exterior_stem_term

    def counts(self, seq: str, db: str) -> dict:
        n = len(seq)
        pt = pair_table(db)
        c: dict[tuple, int] = {}

        def add(k, v=1):
            c[k] = c.get(k, 0) + v

        def unpaired(a, b):  # interval a..b inclusive, 1-based
            if b < a:
                # an EMPTY interval (b == a - 1): a bulge's empty side, or a multiloop's
                # first branch adjacent to its closing pair. positional.cpp reads
                # score_base_*(a, a-1) there with no guard. In type C that is a real
                # matrix entry (strictly lower triangle); in type M the matrix is
                # triu'd, so it is 0 and carries no parameter. (Correction 2026-10-08.)
                if self.t == "C" and b == a - 1:
                    add(("U", a, b))
                return
            if self.t == "C":
                add(("U", a, b))
            else:
                for k in range(a, b + 1):
                    add(("u", k))

        # exterior
        k = 1
        while k <= n:
            if pt[k] > k:
                if self.ext:
                    add(("M", pt[k], k))
                k = pt[k] + 1
            else:
                unpaired(k, k)
                k += 1

        # helices: maximal stack runs, each from its outermost pair
        for i in range(1, n + 1):
            j = pt[i]
            if j <= i:
                continue
            if i > 1 and j < n and pt[i - 1] == j + 1:
                continue  # not the outermost pair of its helix
            m = 1
            while pt[i + m] == j - m and i + m < j - m:
                m += 1
            for kk in range(1, m):
                add(("S", i + kk - 1, j - kk + 1))
                add(("S", j - kk, i + kk))
            add(("helix_length", min(m, 30)))
            # the loop closed by the innermost pair
            a, b = i + m - 1, j - m + 1
            self._loop(a, b, pt, add, unpaired)
        return c

    def _loop(self, i, j, pt, add, unpaired):
        inner = []
        p = i + 1
        while p < j:
            if pt[p] > p:
                inner.append((p, pt[p]))
                p = pt[p] + 1
            else:
                p += 1
        add(("M", i, j))
        if not inner:
            add(("hairpin_length", min(j - i - 1, 30)))
            unpaired(i + 1, j - 1)
            return
        if len(inner) == 1:
            p, q = inner[0]
            l1, l2 = p - i - 1, j - q - 1
            ls, ll = min(l1, l2), max(l1, l2)
            assert ll > 0, "a stack inside a loop position means the helix walk is wrong"
            add(("M", q, p))
            unpaired(i + 1, p - 1)
            unpaired(q + 1, j - 1)
            if ls == 0:
                add(("bulge_length", min(ll, 30)))
            else:
                add(("internal_length", min(ls + ll, 30)))
                add(("internal_explicit", min(ls, 4), min(ll, 4)))
                if ls == ll:
                    add(("internal_symmetry", min(ll, 15)))
                add(("internal_asymmetry", min(ll - ls, 28)))
            return
        # multiloop
        first = inner[0][0]
        unpaired(i + 1, first - 1)  # leading run as one interval
        for idx, (u, v) in enumerate(inner):
            add(("M", v, u))
            nxt = inner[idx + 1][0] if idx + 1 < len(inner) else j
            for kk in range(v + 1, nxt):
                unpaired(kk, kk)  # later runs one nucleotide at a time
