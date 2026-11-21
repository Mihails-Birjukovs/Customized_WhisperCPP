"""Synthetic cost-aware scheduling and full-source VAD fallback acceptance."""
from __future__ import annotations
from pathlib import Path
import json
import sys
import tempfile
import unittest
import wave

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.backend import DecodeProfile
from adaptive_whisper.controller import (
    AdaptivePolicy,RetryStrategy,Trigger,estimated_strategy_order,run_adaptive,_planning_candidate
)
from test_controller import FakeAdapter,synthetic_cues


class SchedulingTests(unittest.TestCase):
    @staticmethod
    def strategies():
        return (
          RetryStrategy("many_short",20000,2000,2000,2000,6,(),10),
          RetryStrategy("fewer_joins",45000,2000,2000,2000,3,(),10),
          RetryStrategy("full_vad",90000,0,0,0,1,
                        (("temperature",.2),("vad_threshold",.65)),10,
                        scope="full_source"),
        )

    def test_estimates_are_heuristics_and_prefer_fewer_joins(self):
        trigger=Trigger("long_loop",20000,39000,"synthetic")
        policy=AdaptivePolicy(strategies=self.strategies(),cost_aware=True,
                              max_total_decode_ms=160000,
                              assumed_native_rtf=.1,boundary_review_penalty_s=30)
        steps,est=estimated_strategy_order(trigger,60000,policy)
        self.assertEqual([x.name for x in steps],
                         ["fewer_joins","full_vad","many_short"])
        self.assertTrue(all(x["not_measured_wall_time"] for x in est))
        self.assertTrue(all(x["full_coverage_feasible"] for x in est))
        self.assertEqual(next(x for x in est if x["strategy"]=="full_vad")["estimated_jobs"],1)
        self.assertEqual(next(x for x in est if x["strategy"]=="full_vad")["estimated_redecode_ms"],60000)

    def test_native_subtitle_endpoint_rounding_does_not_disable_local_backtracking(self):
        trigger=Trigger("micro_loop",63790,75010,"single_cue_ngram")
        short=RetryStrategy("local",25000,2000,2500,2500,4,(),180)
        full=RetryStrategy("full",75000,0,0,0,1,(),180,scope="full_source")
        plan=_planning_candidate(trigger,75000,short)
        self.assertEqual(plan.start_ms,63790)
        self.assertEqual(plan.end_ms,75000)
        p=AdaptivePolicy(strategies=(short,full),cost_aware=True,
            max_total_decode_ms=140000,max_total_jobs=6)
        ordered,est=estimated_strategy_order(trigger,75000,p)
        self.assertEqual(ordered[0].name,"local")
        self.assertTrue(est[0]["full_coverage_feasible"])
        with self.assertRaises(ValueError):
            _planning_candidate(Trigger("micro_loop",63790,75200,"overhang"),75000,short)

    def test_disabling_cost_model_preserves_strategy_order(self):
        trigger=Trigger("micro_loop",10000,25000,"synthetic")
        policy=AdaptivePolicy(strategies=self.strategies(),cost_aware=False)
        steps,_=estimated_strategy_order(trigger,60000,policy)
        self.assertEqual([x.name for x in steps],
                         ["many_short","fewer_joins","full_vad"])

    def test_infeasible_full_source_does_not_preempt_good_short_plan(self):
        trigger=Trigger("long_loop",20000,39000,"synthetic")
        policy=AdaptivePolicy(strategies=self.strategies(),cost_aware=True,
                              max_total_decode_ms=45000,
                              max_total_jobs=6)
        steps,est=estimated_strategy_order(trigger,60000,policy)
        self.assertFalse(next(x for x in est if x["strategy"]=="full_vad")["full_coverage_feasible"])
        self.assertEqual(steps[-1].name,"full_vad")

    def test_real_controller_escalates_to_costly_full_source_only_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            audio=root/"clip.wav"
            with wave.open(str(audio),"wb") as w:
                w.setnchannels(1);w.setframerate(16000);w.setsampwidth(2)
                w.writeframes(b"\x01\x00"*16000*60)
            old=root/"original.srt"
            repeated="The same prolonged phrase repeats through decoding."
            old.write_text(synthetic_cues([(20000,26000,repeated),
                           (26000,32000,repeated),(32000,39000,repeated)]),encoding="utf-8")
            fake=FakeAdapter(fail_default=True)
            policy=AdaptivePolicy(
                strategies=self.strategies(),cost_aware=True,
                max_attempts=2,max_total_jobs=3,
                max_total_decode_ms=100000,
                stop_at_conflicts=0)
            result=run_adaptive(root,fake,audio,old,False,root/"adaptive",
                                "model.bin","vad.bin",DecodeProfile(),policy)
            self.assertEqual(result["cost_planning"]["ordered_strategies"][0],"fewer_joins")
            self.assertEqual(result["attempts"][0]["status"],"RETRY_REQUIRED_WINDOW_FAILURE")
            self.assertEqual(result["attempts"][1]["scope"],"full_source")
            self.assertEqual(result["attempts"][1]["planned_audio_ms"],60000)
            self.assertEqual(result["attempts"][1]["status"],"PATHOLOGY_CLEARED_STRUCTURAL")
            self.assertEqual(result["selected_provisional_attempt"],2)
            self.assertTrue(result["attempts"][1]["energy_coverage_proxy"]["not_ground_truth"])
            self.assertTrue(result["attempts"][1]["machine_hypothesis_consistency"]["baseline_pathological"])
            self.assertFalse(result["human_audio_accuracy_verified"])
            self.assertTrue(old.is_file())

if __name__=="__main__":
    unittest.main()
