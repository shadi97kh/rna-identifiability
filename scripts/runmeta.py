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


def _git(args, root: pathlib.Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    ).stdout.strip()


def git_provenance(root: pathlib.Path) -> dict:
    porcelain = _git(["status", "--porcelain"], root)
    lines = [ln for ln in porcelain.splitlines() if ln.strip()]
    outside = [ln for ln in lines if not ln[3:].lstrip('"').startswith("runs/")]
    return {
        "git_sha": _git(["rev-parse", "HEAD"], root),
        "git_dirty": bool(lines),
        "git_dirty_excluding_runs": bool(outside),
        "git_dirty_paths_outside_runs": [ln[3:] for ln in outside][:20],
    }
