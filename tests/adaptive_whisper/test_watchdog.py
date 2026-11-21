"""Cross-platform synthetic process-progress timeout tests (no Whisper model)."""
from __future__ import annotations
from pathlib import Path
import os
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.watchdog import WatchdogPolicy, run_watched


class NativeProgressWatchdogTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_script(self,program,policy,hard=3.0,name="run"):
        folder=self.root/name
        folder.mkdir()
        return run_watched([sys.executable,"-u","-c",program],
            cwd=folder,env=os.environ.copy(),
            stdout_raw=folder/"out.raw",
            stderr_raw=folder/"err.raw",
            hard_timeout_s=hard,policy=policy)

    def test_advanced_progress_and_raw_byte_logs_pass(self):
        program=("import sys,time\n"
          "for pct in (5,25,70,100):\n"
          " sys.stderr.write(f'whisper_print_progress_callback: progress =  {pct}%\\r')\n"
          " sys.stderr.flush()\n"
          " time.sleep(.08)\n"
          "sys.stdout.buffer.write(b'\\xff\\x00DATA');sys.stdout.flush()\n")
        policy=WatchdogPolicy(startup_timeout_s=1,idle_progress_timeout_s=.6,poll_s=.02)
        r=self.run_script(program,policy)
        self.assertFalse(r["timed_out"])
        self.assertEqual(r["exit_code"],0)
        self.assertEqual(r["highest_progress_pct"],100)
        self.assertGreaterEqual(len(r["progress_events"]),4)
        self.assertEqual((self.root/"run/out.raw").read_bytes(),b"\xff\x00DATA")
        self.assertIn(b"progress =  100%",(self.root/"run/err.raw").read_bytes())

    def test_genuine_no_progress_startup_timeout(self):
        r=self.run_script("import time;time.sleep(8)",
                          WatchdogPolicy(startup_timeout_s=.5,
                          idle_progress_timeout_s=2,poll_s=.025),hard=5)
        self.assertTrue(r["timed_out"])
        self.assertEqual(r["watchdog_termination_reason"],"startup_no_progress")
        self.assertIsNone(r["highest_progress_pct"])
        self.assertLess(r["elapsed_seconds"],3)

    def test_no_advanced_progress_after_first_callback(self):
        program=("import sys,time\n"
           "sys.stderr.write('whisper_print_progress_callback: progress =  12%\\n');sys.stderr.flush()\n"
           "time.sleep(8)\n")
        r=self.run_script(program,WatchdogPolicy(startup_timeout_s=2,
                  idle_progress_timeout_s=.5,poll_s=.02),hard=5)
        self.assertEqual(r["watchdog_termination_reason"],"progress_idle_timeout")
        self.assertEqual(r["highest_progress_pct"],12)

    def test_hard_wall_timeout_even_with_repeated_updates(self):
        program=("import sys,time\n"
          "for k in range(100):\n"
          " sys.stderr.write(f'progress = {min(k,99)}%\\n');sys.stderr.flush();time.sleep(.09)\n")
        r=self.run_script(program,WatchdogPolicy(startup_timeout_s=2,
                 idle_progress_timeout_s=2,poll_s=.02),hard=.6)
        self.assertTrue(r["timed_out"])
        self.assertEqual(r["watchdog_termination_reason"],"hard_wall_timeout")

    def test_clean_no_progress_fast_process_not_a_stall(self):
        r=self.run_script("print('no CLI progress lines')",
                          WatchdogPolicy(startup_timeout_s=1,
                             idle_progress_timeout_s=1,poll_s=.02))
        self.assertEqual(r["exit_code"],0)
        self.assertFalse(r["timed_out"])
        self.assertFalse(r["observed_first_progress"])

    def test_raw_logs_are_never_overwritten(self):
        p=WatchdogPolicy(startup_timeout_s=1,idle_progress_timeout_s=1)
        self.run_script("print('finished')",p,name="locked")
        folder=self.root/"locked"
        with self.assertRaises(FileExistsError):
            run_watched([sys.executable,"-c","print('second')"],
                         cwd=folder,env=os.environ.copy(),
                         stdout_raw=folder/"out.raw",
                         stderr_raw=folder/"err.raw",
                         hard_timeout_s=2,policy=p)


if __name__=="__main__":
    unittest.main()
