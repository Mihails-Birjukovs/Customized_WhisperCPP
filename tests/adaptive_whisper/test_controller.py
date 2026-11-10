"""Offline synthetic pathology/recovery tests; no model, GPU or private audio."""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import json
import sys
import tempfile
import unittest
import wave

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.backend import DecodeProfile
from adaptive_whisper.controller import (
    AdaptivePolicy, RetryStrategy, DetectionPolicy, detect_pathologies, run_adaptive,
)
from adaptive_whisper.stitch import fmt


def synthetic_cues(lines):
    return "".join(
        f"{i}\n{fmt(a)} --> {fmt(b)}\n{t}\n\n"
        for i,(a,b,t) in enumerate(lines,1)
    )


class FakeAdapter:
    """Writes only instrumented synthetic output. Enables exact branch testing."""
    def __init__(self, fail_default=False, malform_default=False):
        self.fail_default=fail_default
        self.malform_default=malform_default
        self.calls=[]

    def run_once(self,audio,model_name,vad_name,output_dir,profile,timeout_s=600):
        if output_dir.exists():
            raise FileExistsError(output_dir)
        output_dir.mkdir()
        self.calls.append({"profile":profile,"audio":audio,"timeout":timeout_s})
        import hashlib
        hashfile=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        (output_dir/"stdout.log").write_text("",encoding="utf-8")
        (output_dir/"stderr.log").write_text("",encoding="utf-8")
        failing=self.fail_default and profile.temperature==0.0
        if not failing:
            tag=output_dir.name.replace("w","")
            text=f"Synthetic nonrepeating word sequence from window {tag}."
            raw=synthetic_cues([(2000,3200,text)])
            (output_dir/"transcript.srt").write_text(raw,encoding="utf-8")
            (output_dir/"transcript.txt").write_text(text+"\n",encoding="utf-8")
            if self.malform_default and profile.temperature==0.2:
                bad=(b'{"transcription":[{"text":"\xc4\xb7",'
                     b'"tokens":[{"text":"\xc4"},{"text":"\xb7"}]}]}')
                (output_dir/"transcript.json").write_bytes(bad)
            else:
                (output_dir/"transcript.json").write_text(
                    json.dumps({"transcription":[{"text":text,"tokens":[{"text":text}]}]}),
                    encoding="utf-8")
        return {
            "inference_passed":not failing,"exit_code":None if failing else 0,
            "elapsed_seconds":0.001,
            "native_json_parse_valid":not failing and not (self.malform_default and profile.temperature==.2),
            "hashes":{"model":"synthetic-model-sha","binary":"synthetic-exe-sha"}
        }


class ControllerUnitTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.ws=Path(self.temp.name)
        self.source=self.ws/"speech.wav"
        with wave.open(str(self.source),"wb") as w:
            w.setframerate(16000);w.setsampwidth(2);w.setnchannels(1)
            w.writeframes(b"\x01\x00"*16000*60)
        self.baseline=self.ws/"historical.srt"
        p="The decoder repeats an artificially long phrase."
        self.baseline.write_text(synthetic_cues([
            (20000,26000,p),(26000,32000,p),(32000,39000,p)]),encoding="utf-8")
        self.profile=DecodeProfile(threads=4,vad_pad_ms=50)
        self.clean=self.ws/"clean.srt"
        self.clean.write_text(synthetic_cues([(1000,4000,"Short opening statement."),
             (30000,34000,"Different statement.")]),encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def strategy(name,window=20000,update=()):
        return RetryStrategy(name,window,2000,2000,2000,6,update,20)

    def invoke(self,adapter,policy,baseline=None,name="new_run",failed=False):
        return run_adaptive(self.ws,adapter,self.source,
                            baseline or self.baseline,failed,self.ws/name,
                            "model.bin","vad.bin",self.profile,policy)

    def test_exact_three_cue_micro_loop_detected(self):
        triggers=detect_pathologies(self.baseline.read_text(encoding="utf-8"),60000)
        self.assertEqual(triggers[0].kind,"micro_loop")
        self.assertEqual((triggers[0].start_ms,triggers[0].end_ms),(20000,39000))

    def test_long_loop_and_intra_cue_micro_detection(self):
        p="Long repeated sentence with several distinctive words."
        data=synthetic_cues([(3000,17000,p),(17000,31000,p),(31000,45000,p)])
        kinds=[x.kind for x in detect_pathologies(data,60000)]
        self.assertIn("long_loop",kinds)
        phrase="the same unusual phrase"
        raw=synthetic_cues([(2000,6000," ".join([phrase]*4))])
        self.assertTrue(any(x.signal=="repeated_ngram_within_single_cue"
                            for x in detect_pathologies(raw,60000)))

    def test_clean_baseline_makes_no_native_calls(self):
        fake=FakeAdapter()
        report=self.invoke(fake,AdaptivePolicy(
            strategies=(self.strategy("reset"),),max_total_decode_ms=140000),
            baseline=self.clean)
        self.assertEqual(report["status"],"NO_CONFIRMED_TRIGGER")
        self.assertEqual(report["native_jobs_executed"],0)
        self.assertFalse(report["historical_baseline_modified"])

    def test_context_reset_recovers_synthetic_loop_without_word_truth(self):
        fake=FakeAdapter()
        report=self.invoke(fake,AdaptivePolicy(strategies=(self.strategy("reset"),),
                            max_total_decode_ms=160000,max_total_jobs=6,
                            max_attempts=1))
        self.assertTrue(report["status"].startswith("PATHOLOGY_CLEARED"))
        self.assertTrue(report["attempts"][0]["planned_full_coverage"])
        self.assertIn(report["attempts"][0]["status"],
            ("PATHOLOGY_CLEARED_STRUCTURAL","PATHOLOGY_CLEARED_JOIN_REVIEW"))
        self.assertGreater(report["native_jobs_executed"],0)
        self.assertFalse(report["human_audio_accuracy_verified"])
        self.assertEqual(report["original_srt_sha256"],
            __import__("hashlib").sha256(self.baseline.read_bytes()).hexdigest())

    def test_failed_decode_escalates_temperature_and_backtracks(self):
        fake=FakeAdapter(fail_default=True)
        policy=AdaptivePolicy(strategies=(
            self.strategy("reset"),self.strategy("temperature_backtrack",
                                               update=(("temperature",0.2),))),
            max_total_decode_ms=200000,max_total_jobs=12,max_attempts=2)
        report=self.invoke(fake,policy)
        self.assertEqual(len(report["attempts"]),2)
        self.assertEqual(report["attempts"][0]["status"],"RETRY_REQUIRED_WINDOW_FAILURE")
        self.assertTrue(report["attempts"][1]["status"].startswith("PATHOLOGY_CLEARED"))
        self.assertEqual(report["attempts"][1]["parameter_adjustments"]["temperature"],0.2)
        self.assertEqual(fake.calls[0]["profile"].temperature,0.0)
        self.assertEqual(fake.calls[-1]["profile"].temperature,0.2)

    def test_lossless_native_json_fragment_fallback_not_silent_replacement(self):
        fake=FakeAdapter(fail_default=True,malform_default=True)
        policy=AdaptivePolicy(strategies=(
            self.strategy("reset"),self.strategy("temperature_retry",
                                                update=(("temperature",.2),))),
            max_total_decode_ms=200000,max_total_jobs=12,max_attempts=2)
        report=self.invoke(fake,policy)
        self.assertTrue(report["status"].startswith("PATHOLOGY_CLEARED"))
        self.assertTrue(all(w["json_quality"]["status"]=="reversible_token_fragment_derived"
            for w in report["attempts"][1]["windows"]))
        self.assertTrue((self.ws/"new_run/attempt_02_temperature_retry/jobs/w001/"
                         "transcript.derived.lossless-adaptive-v1.json").is_file())

    def test_global_budget_blocks_partial_coverage_and_does_not_mislabel_success(self):
        fake=FakeAdapter()
        policy=AdaptivePolicy(strategies=(self.strategy("reset"),),
                    max_total_decode_ms=22000,max_total_jobs=6,max_attempts=1)
        report=self.invoke(fake,policy)
        self.assertEqual(report["status"],"NO_RETRY_BUDGET_FOR_COMPLETE_COVERAGE")
        self.assertEqual(report["attempts"][0]["status"],"SKIPPED_INCOMPLETE_WINDOW_COVERAGE")
        self.assertEqual(fake.calls,[])

    def test_stall_trigger_requires_native_failure_or_explicit_approval(self):
        no_srt=detect_pathologies(None,60000,failure_or_timeout=False)
        self.assertEqual(no_srt[0].kind,"no_output_unverified")
        stalled=detect_pathologies(None,60000,failure_or_timeout=True)
        self.assertEqual(stalled[0].kind,"decoder_stall")
        fake=FakeAdapter()
        report=run_adaptive(self.ws,fake,self.source,None,False,self.ws/"no_output_run",
                            "model.bin","vad.bin",self.profile,
                            AdaptivePolicy(strategies=(self.strategy("reset",window=60000),),
                                           max_total_decode_ms=90000))
        self.assertEqual(report["status"],"NO_CONFIRMED_TRIGGER")
        self.assertEqual(fake.calls,[])

    def test_does_not_overwrite_completed_runs(self):
        fake=FakeAdapter()
        policy=AdaptivePolicy(strategies=(self.strategy("reset"),),max_total_decode_ms=140000)
        self.invoke(fake,policy)
        with self.assertRaises(FileExistsError):
            self.invoke(fake,policy)

    def test_boundary_conflicts_trigger_second_longer_context_strategy(self):
        fake=FakeAdapter()
        policy=AdaptivePolicy(
            strategies=(self.strategy("short",window=20000),
                        self.strategy("long",window=45000)),
            max_attempts=2,max_total_decode_ms=150000,max_total_jobs=10)
        report=self.invoke(fake,policy)
        self.assertEqual(report["attempts"][0]["status"],"PATHOLOGY_CLEARED_JOIN_REVIEW")
        self.assertEqual(report["attempts"][1]["status"],"PATHOLOGY_CLEARED_STRUCTURAL")
        self.assertEqual(report["selected_provisional_attempt"],2)
        self.assertTrue(report["status"].endswith("STRUCTURAL_UNVERIFIED"))

    def test_short_true_repeated_utterance_not_treated_as_pathology(self):
        # Ordinary emphatic speech can repeat a short word without ASR malfunction.
        normal=synthetic_cues([(0,1100,"Yes."),(1100,2200,"Yes."),
                               (2200,3300,"Yes.")])
        self.assertEqual(detect_pathologies(normal,60000),())

    def test_native_timeout_with_partial_progress_is_stall_candidate(self):
        partial=synthetic_cues([(0,10000,"Speech before native CLI stalled.")])
        result=detect_pathologies(partial,60000,failure_or_timeout=True)
        self.assertEqual(result[0].kind,"decoder_stall")
        self.assertEqual(result[0].start_ms,5000)
        self.assertEqual(result[0].end_ms,60000)

    def test_per_strategy_backend_fallback_uses_same_shared_controller(self):
        failing_cuda=FakeAdapter(fail_default=True)
        working_vulkan=FakeAdapter()
        policy=AdaptivePolicy(
            strategies=(self.strategy("cuda_reset"),
                        self.strategy("vulkan_retry")),
            max_attempts=2,max_total_decode_ms=160000,max_total_jobs=10)
        report=run_adaptive(self.ws,failing_cuda,self.source,self.baseline,False,
                            self.ws/"fallback_new_run","model.bin","vad.bin",
                            self.profile,policy,
                            adapters_by_strategy={"vulkan_retry":working_vulkan})
        self.assertEqual(len(failing_cuda.calls),1)
        self.assertGreater(len(working_vulkan.calls),0)
        self.assertEqual(report["attempts"][0]["status"],"RETRY_REQUIRED_WINDOW_FAILURE")
        self.assertTrue(report["attempts"][1]["status"].startswith("PATHOLOGY_CLEARED"))

    def test_unsafe_profile_field_rejected(self):
        with self.assertRaises(ValueError):
            self.strategy("unsafe",update=(("gpu_device",4),))
        with self.assertRaises(ValueError):
            AdaptivePolicy(max_attempts=0)

if __name__=="__main__":
    unittest.main()
