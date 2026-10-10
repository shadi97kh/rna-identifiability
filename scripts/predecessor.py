#!/usr/bin/env python3
"""Wait for a predecessor run by its RECORDED STATUS, never by process presence.

Adopted 2026-10-10. The MXfold2 seed study's driver was killed at about 05:59 on
2026-10-09 with its artifact still saying RUNNING, and the gauge launcher, which waited
for the study's processes to disappear, read the disappearance as completion. It did no
harm only because the GPUs happened to be idle. The rule here:

  recorded status is a completion status          -> done: proceed
  recorded status is RUNNING..., a process is live -> keep waiting
  recorded status is RUNNING..., no process live   -> FAULT: report it, do not proceed
  any other recorded status, or none readable      -> FAULT

Process presence is used only to detect the fault, never as a completion signal. Before a
fault is declared the status is read again, so a predecessor that writes its final status
and exits between the two reads is seen as done.

Standard library only: imported from .venv and from .venv-mxfold2.

CLI (blocks; exit 0 when done, 2 on a fault; one JSON line on stdout either way):
  .venv/bin/python scripts/predecessor.py ARTIFACT --done STATUS [--done STATUS ...]
      [--script NAME] [--arg VALUE] [--poll SECONDS]
  --script  a python process whose script argument ends with NAME counts as live
  --arg     any process given VALUE as an argument counts as live
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

RUNNING_PREFIX = "RUNNING"


def ancestors() -> set:
    """This process and its ancestors: a launching shell never counts as the predecessor."""
    pids, pid = set(), os.getpid()
    while pid > 1:
        pids.add(pid)
        try:
            stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def processes(script=None, arg=None) -> list:
    """Live pids, other than this process and its ancestors, that are a python running a
    script whose path ends with `script`, or that were given `arg` as an argument."""
    skip, found = ancestors(), []
    for d in pathlib.Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in skip:
            continue
        try:
            argv = [x.decode(errors="ignore") for x in (d / "cmdline").read_bytes().split(b"\0")
                    if x]
        except OSError:
            continue
        if not argv:
            continue
        if script and "python" in pathlib.Path(argv[0]).name and len(argv) > 1 \
                and argv[1].endswith(script):
            found.append(int(d.name))
        elif arg and arg in argv:
            found.append(int(d.name))
    return sorted(found)


def _status(artifact):
    try:
        return json.loads(pathlib.Path(artifact).read_text()).get("status"), None
    except (OSError, ValueError) as e:
        return None, f"artifact unreadable: {e!r}"


def state(artifact, done, alive) -> dict:
    """One reading. `done` is the set of completion statuses; `alive()` returns the live
    predecessor processes (a list of pids, or a bool)."""
    status, err = _status(artifact)
    st = {"artifact": str(artifact), "status": status,
          "checked": time.strftime("%Y-%m-%d %H:%M:%S")}
    if err:
        return st | {"state": "fault", "detail": err}
    if status in done:
        return st | {"state": "done"}
    if isinstance(status, str) and status.startswith(RUNNING_PREFIX):
        live = alive()
        if live:
            return st | {"state": "running", "live": live}
        again, err = _status(artifact)  # it may have finished between the two reads
        if again in done:
            return st | {"status": again, "state": "done"}
        return st | {"state": "fault", "status": again,
                     "detail": "processes gone while status is RUNNING"}
    return st | {"state": "fault",
                 "detail": "status is neither RUNNING nor a completion status"}


def wait(artifact, done, alive, poll_s=60, on_poll=None) -> dict:
    """Block until the predecessor is done or faulted; return the final reading.
    `on_poll(reading)` is called after every reading that keeps us waiting."""
    while True:
        st = state(artifact, done, alive)
        if st["state"] != "running":
            return st
        if on_poll:
            on_poll(st)
        time.sleep(poll_s)


def main(argv) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("artifact")
    ap.add_argument("--done", action="append", required=True)
    ap.add_argument("--script")
    ap.add_argument("--arg")
    ap.add_argument("--poll", type=float, default=60)
    a = ap.parse_args(argv)
    if not (a.script or a.arg):
        ap.error("give --script or --arg: process presence is how a fault is detected")
    st = wait(a.artifact, set(a.done), lambda: processes(a.script, a.arg), a.poll)
    print(json.dumps(st))
    return 0 if st["state"] == "done" else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
