"""Provenance stamped into every run artifact.

WHY `git_dirty` ALONE IS THE WRONG QUESTION

An artifact records whether the CODE AND INPUTS that produced it were committed.
Its own output directory is not an input. A bare `git status --porcelain` also
reports `runs/`, so a long run can stamp itself irreproducible merely because a
stale artifact from a previous run was sitting modified, or because anything
else touched the tree during the minutes the run took. Neither says anything
about reproducibility.

`git_dirty` is still recorded, unchanged, so nothing is hidden. What downstream
checks should assert on is `git_dirty_excluding_runs`, which is the property that
actually matters: every tracked file outside `runs/` was committed at the moment
the artifact was produced.

This is a correction, not a relaxation. Excluding outputs from a "were the inputs
committed" check makes it answer its own question.
"""

from __future__ import annotations

import pathlib
import subprocess


def _git(args, root: pathlib.Path, *, strip: bool = True) -> str:
    out = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    ).stdout
    return out.strip() if strip else out


def _porcelain_path(line: str) -> str:
    """The path out of one `git status --porcelain` line.

    The status field is exactly two characters followed by a space, so the path
    begins at index 3 -- but ONLY if the line still has its leading space. An
    unmodified-in-index entry starts with one (" M path"), and calling .strip()
    on the whole command output silently eats it from the FIRST line alone.
    That shifted every first path by one character, and it was not merely
    cosmetic: a leading " M runs/..." then failed the runs/ test and reported the
    artifact as generated from a dirty tree. Rebuilt from the right-hand side so
    the leading space cannot matter.

    A rename is recorded as "R  old -> new"; the new path is what exists on disk.
    """
    body = line[2:].strip() if len(line) > 2 else line.strip()
    if " -> " in body:
        body = body.split(" -> ", 1)[1]
    return body.strip('"')


def git_provenance(root: pathlib.Path) -> dict:
    porcelain = _git(["status", "--porcelain"], root, strip=False)
    paths = [_porcelain_path(ln) for ln in porcelain.splitlines() if ln.strip()]
    outside = [q for q in paths if not q.startswith("runs/")]
    return {
        "git_sha": _git(["rev-parse", "HEAD"], root),
        "git_dirty": bool(paths),
        "git_dirty_excluding_runs": bool(outside),
        "git_dirty_paths_outside_runs": outside[:20],
    }
