"""scripts/predecessor.py: completion is read from the recorded status, never from process
presence. Synthetic artifacts only; stdlib unittest so it also runs in .venv-mxfold2."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.predecessor import processes, state, wait  # noqa: E402

DONE = {"full COMPLETE (training)"}


class TestPredecessor(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.art = pathlib.Path(self.dir.name) / "metrics.json"

    def tearDown(self):
        self.dir.cleanup()

    def put(self, status):
        self.art.write_text(json.dumps({"status": status}))

    def test_done(self):
        self.put("full COMPLETE (training)")
        self.assertEqual(state(self.art, DONE, lambda: [])["state"], "done")

    def test_running_with_live_process(self):
        self.put("RUNNING phase full")
        st = state(self.art, DONE, lambda: [123])
        self.assertEqual((st["state"], st["live"]), ("running", [123]))

    def test_processes_gone_while_running_is_a_fault_not_completion(self):
        # the 2026-10-09 case: the driver was killed, the artifact still says RUNNING
        self.put("RUNNING phase full")
        st = state(self.art, DONE, lambda: [])
        self.assertEqual(st["state"], "fault")
        self.assertEqual(st["detail"], "processes gone while status is RUNNING")

    def test_finished_between_the_two_reads_is_done(self):
        self.put("RUNNING phase full")

        def alive():  # the predecessor writes its final status and exits mid-reading
            self.put("full COMPLETE (training)")
            return []

        self.assertEqual(state(self.art, DONE, alive)["state"], "done")

    def test_failure_status_is_a_fault(self):
        self.put("full finished with failures")
        self.assertEqual(state(self.art, DONE, lambda: [1])["state"], "fault")

    def test_unreadable_artifact_is_a_fault(self):
        self.assertEqual(state(self.art, DONE, lambda: [1])["state"], "fault")
        self.art.write_text("{not json")
        self.assertEqual(state(self.art, DONE, lambda: [1])["state"], "fault")

    def test_wait_polls_until_the_status_completes(self):
        self.put("RUNNING phase full")
        polls = []

        def on_poll(st):
            polls.append(st["state"])
            if len(polls) == 3:
                self.put("full COMPLETE (training)")

        st = wait(self.art, DONE, lambda: [1], poll_s=0, on_poll=on_poll)
        self.assertEqual((st["state"], polls), ("done", ["running"] * 3))

    def test_wait_returns_the_fault_instead_of_proceeding(self):
        self.put("RUNNING phase full")
        self.assertEqual(wait(self.art, DONE, lambda: [], poll_s=0)["state"], "fault")

    def test_gauge_launcher_waits_for_the_seed_studys_own_completion_status(self):
        from scripts.mxfold2_gauge_study import SEED_STUDY_DONE
        from scripts.mxfold2_seed_study import FULL_DONE

        self.assertEqual(SEED_STUDY_DONE, FULL_DONE)

    def test_process_scan_never_counts_this_process(self):
        me = pathlib.Path(sys.argv[0]).name or "python"
        self.assertNotIn(os.getpid(), processes(script=me, arg=sys.argv[0]))


if __name__ == "__main__":
    unittest.main()
