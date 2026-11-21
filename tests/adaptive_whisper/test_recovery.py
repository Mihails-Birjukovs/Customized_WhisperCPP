"""Safety and coverage tests for backend-neutral recovery planner."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.diagnostics import RepetitionCandidate
from adaptive_whisper.recovery import RetryPolicy, plan_retries

def candidate(start,end):
    return RepetitionCandidate(start,end,13,65,end-start)

class RetryPlanningTests(unittest.TestCase):
    def test_long_loop_is_partial_and_bounded(self):
        policy=RetryPolicy()
        plan=plan_retries(candidate(907720,1387030),1397250,policy)
        self.assertEqual(len(plan.windows),2)
        self.assertFalse(plan.full_coverage)
        self.assertEqual(plan.status,"PARTIAL_PROBE_ONLY_NO_STITCH")
        self.assertEqual(plan.windows[0].global_start_ms,904720)
        self.assertEqual(plan.windows[0].global_end_ms,964720)
        self.assertEqual(plan.windows[1].global_start_ms,961720)
        self.assertEqual(plan.windows[1].global_end_ms,1021720)
        self.assertGreater(plan.remaining_uncovered_ms,350000)
        self.assertLessEqual(sum(w.duration_ms for w in plan.windows),policy.max_audio_ms)

    def test_short_candidate_covered_in_one_window(self):
        plan=plan_retries(candidate(50000,65000),100000)
        self.assertTrue(plan.full_coverage)
        self.assertEqual(len(plan.windows),1)
        self.assertEqual(plan.windows[0].global_start_ms,47000)
        self.assertEqual(plan.windows[0].global_end_ms,68000)
        self.assertEqual(plan.remaining_uncovered_ms,0)

    def test_global_time_offset(self):
        w=plan_retries(candidate(20000,40000),80000).windows[0]
        self.assertEqual(w.to_global(5000),22000)
        with self.assertRaises(ValueError):
            w.to_global(-1)
        with self.assertRaises(ValueError):
            w.to_global(w.duration_ms+1)

    def test_rejects_bad_policy_and_out_of_audio_candidate(self):
        with self.assertRaises(ValueError):
            RetryPolicy(window_ms=5000,overlap_ms=5000)
        with self.assertRaises(ValueError):
            RetryPolicy(max_jobs=0)
        with self.assertRaises(ValueError):
            plan_retries(candidate(30000,100000),90000)
if __name__=="__main__":
    unittest.main()
