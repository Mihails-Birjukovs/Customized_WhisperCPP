"""Private-data-free multi-region regression tests."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adaptive_whisper.backend import DecodeProfile
from adaptive_whisper.controller import AdaptivePolicy, RetryStrategy, Trigger, run_adaptive
from adaptive_whisper.regions import plan_regions, run_adaptive_regions
from adaptive_whisper.stitch import fmt


def srt(lines):
    return "".join(f"{i}\n{fmt(a)} --> {fmt(b)}\n{text}\n\n"
                   for i, (a, b, text) in enumerate(lines, 1))


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def run_once(self, audio, model_name, vad_name, output_dir, profile, timeout_s=600):
        output_dir.mkdir()
        self.calls.append(audio)
        text = f"Different unique synthetic utterance {len(self.calls)}."
        (output_dir / "transcript.srt").write_text(
            srt([(1000, 1900, text)]), encoding="utf-8")
        (output_dir / "transcript.json").write_text(
            json.dumps({"transcription": [{"text": text}]}), encoding="utf-8")
        return {"inference_passed": True, "exit_code": 0, "elapsed_seconds": .001,
                "native_json_parse_valid": True,
                "hashes": {"model": "fake-model", "binary": "fake-executable"}}


class MultiRegionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        self.wav = self.ws / "original.wav"
        with wave.open(str(self.wav), "wb") as w:
            w.setframerate(16000)
            w.setnchannels(1)
            w.setsampwidth(2)
            w.writeframes(b"\x01\x00" * 16000 * 60)
        phrase = "This distinct and unusually repetitive error phrase."
        other = "A second separate artificial hallucination repeats elsewhere."
        self.baseline = self.ws / "baseline.srt"
        self.baseline.write_text(srt([
            (5000, 8000, phrase), (8000, 11000, phrase),
            (11000, 14000, phrase),
            (35000, 38000, other), (38000, 41000, other),
            (41000, 44000, other),
        ]), encoding="utf-8")
        self.profile = DecodeProfile(threads=4)
        self.adapter = FakeAdapter()

    def policy(self, **changes):
        options = dict(strategies=(
            RetryStrategy("single_reset", 20000, 2000, 2000, 2000, 4),
        ), max_regions=2, max_attempts=1,
            max_total_decode_ms=70000, max_total_jobs=4,
            compose_full_timeline_preview=True)
        options.update(changes)
        return AdaptivePolicy(**options)

    def invoke(self, policy=None, name="results", **kwargs):
        return run_adaptive_regions(self.ws, self.adapter, self.wav, self.baseline,
                                    False, self.ws / name, "fake-model", "fake-vad",
                                    self.profile, policy or self.policy(), **kwargs)

    def test_two_disjoint_regions_are_processed_with_separate_ledgers(self):
        original = self.baseline.read_bytes()
        report = self.invoke()
        self.assertEqual(report["regions_detected"], 2)
        self.assertEqual(report["regions_structurally_cleared"], 2)
        self.assertEqual(report["status"], "ALL_REGIONS_FULL_TIMELINE_REVIEW_REQUIRED")
        self.assertEqual(len(self.adapter.calls), 2)
        self.assertEqual(report["consumed_native_jobs"], 2)
        self.assertTrue(report["full_timeline_splice_performed"])
        self.assertTrue((self.ws / "results/provisional_multi_full_timeline.srt").is_file())
        self.assertTrue((self.ws / "results/provisional_multi_full_timeline_review.json").is_file())
        self.assertEqual(report["full_timeline"]["status"], "PROVISIONAL_REVIEW_REQUIRED")
        self.assertFalse(report["auto_release_authorized"])
        self.assertEqual(self.baseline.read_bytes(), original)
        self.assertEqual([r["source_interval_ms"] for r in report["regions"]],
                         [[5000, 14000], [35000, 44000]])
        for i in (1, 2):
            folder = self.ws / "results" / f"region_{i:03d}"
            self.assertTrue((folder / "adaptive_decision_ledger.json").exists())
            self.assertFalse((folder / "provisional_full_timeline.srt").exists())

    def test_shared_decode_budget_defers_second_region(self):
        report = self.invoke(self.policy(max_total_decode_ms=24000), "budgeted")
        self.assertEqual(report["regions_detected"], 2)
        self.assertEqual(report["regions_structurally_cleared"], 1)
        self.assertEqual(report["status"], "PARTIAL_REGIONS_UNRESOLVED")
        self.assertEqual(report["full_timeline"]["status"],
                         "SKIPPED_UNRESOLVED_OR_DEFERRED_REGIONS")
        self.assertFalse(report["full_timeline_splice_performed"])
        self.assertEqual(report["regions"][1]["status"],
                         "NO_RETRY_BUDGET_FOR_COMPLETE_COVERAGE")
        self.assertLessEqual(report["consumed_decode_ms"], 24000)
        self.assertEqual(len(self.adapter.calls), 1)

    def test_max_regions_defers_without_falsely_claiming_all_recovered(self):
        report = self.invoke(self.policy(max_regions=1), "one")
        self.assertEqual(report["regions"][1]["status"], "DEFERRED_REGION_LIMIT")
        self.assertEqual(report["regions_deferred"], 1)
        self.assertEqual(report["status"], "PARTIAL_REGIONS_UNRESOLVED")

    def test_overlap_merges_and_deduplicates_detected_triggers(self):
        t = (Trigger("micro_loop", 1000, 5000, "a"),
             Trigger("micro_loop", 4000, 8000, "b"),
             Trigger("long_loop", 20000, 28000, "c"))
        regions = plan_regions(t, 60000)
        self.assertEqual([(x.start_ms, x.end_ms, len(x.members)) for x in regions],
                         [(1000, 8000, 2), (20000, 28000, 1)])
        bridged = plan_regions(t, 60000, guard_ms=12000)
        self.assertEqual(len(bridged), 1)

    def test_empty_srt_without_native_failure_is_not_eligible(self):
        self.baseline.write_text("", encoding="utf-8")
        result = self.invoke(name="silent")
        self.assertEqual(result["status"], "NO_ELIGIBLE_REGIONS")
        self.assertEqual(result["regions_detected"], 0)
        self.assertEqual(len(self.adapter.calls), 0)

    def test_full_source_strategies_are_rejected_before_output(self):
        p = self.policy(strategies=(RetryStrategy(
            "full", 60000, 0, 0, 0, 1, scope="full_source"),))
        with self.assertRaises(ValueError):
            self.invoke(p, "unsafe_full")
        self.assertFalse((self.ws / "unsafe_full").exists())

    def test_invalid_override_is_rejected(self):
        with self.assertRaises(ValueError):
            run_adaptive(self.ws, self.adapter, self.wav, self.baseline, False,
                         self.ws / "override", "model", "vad", self.profile,
                         self.policy(), trigger_override=Trigger(
                             "micro_loop", 47000, 50000, "invented"))
        self.assertEqual(self.adapter.calls, [])

    def test_opt_out_retains_only_independent_region_previews(self):
        report = self.invoke(self.policy(compose_full_timeline_preview=False), "optout")
        self.assertEqual(report["status"], "ALL_REGIONS_PROVISIONAL_UNVERIFIED")
        self.assertFalse(report["full_timeline_splice_performed"])
        self.assertEqual(report["full_timeline"]["status"], "NOT_REQUESTED")
        self.assertFalse((self.ws / "optout/provisional_multi_full_timeline.srt").exists())

    def test_two_separate_triggers_with_overlapping_retry_windows_block_splice(self):
        # Disjoint detector intervals have overlapping *padded retry audio*.
        first = "This distinct and unusually repetitive error phrase."
        other = "A second separate artificial hallucination repeats elsewhere."
        self.baseline.write_text(srt([
            (5000, 8000, first), (8000, 11000, first),
            (11000, 14000, first),
            (17500, 19500, other), (19500, 21500, other),
            (21500, 23500, other),
        ]), encoding="utf-8")
        report = self.invoke(name="overlapping")
        self.assertEqual(report["regions_detected"], 2)
        self.assertEqual(report["regions_structurally_cleared"], 2)
        self.assertEqual(report["full_timeline"]["status"],
                         "BLOCKED_OVERLAPPING_RECOVERY_WINDOWS")
        self.assertEqual(report["full_timeline"]["overlap_ms"], [15500, 16000])
        self.assertFalse(report["full_timeline_splice_performed"])
        self.assertFalse((self.ws / "overlapping/provisional_multi_full_timeline.srt").exists())

    def test_signal_specific_strategy_selection_preserves_shared_budget(self):
        # Two different detector signal families, on one natural-like
        # source timeline. Explicitly choose different window contexts.
        first = "An artificial internal repeated phrase. " * 3
        second = "Another distinct consecutive synthetic error phrase."
        self.baseline.write_text(srt([
            (5000, 14000, first),
            (35000, 38000, second),
            (38000, 41000, second),
            (41000, 44000, second),
        ]), encoding="utf-8")
        options = {
            "repeated_ngram_within_single_cue": (
                RetryStrategy("inside_cue_extended", 30000, 0, 8000, 8000, 1),
            ),
            "consecutive_normalized_cues": (
                RetryStrategy("consecutive_short", 20000, 0, 2000, 2000, 1),
            ),
        }
        # First uses 22 s, leaving 14 s of a 36-s global cap.
        # Second has a nominal 20-s maximum but needs only 13 s.
        # This must run rather than falsely fail the budget check.
        report = self.invoke(self.policy(max_total_decode_ms=36000),
                             name="signal_contexts",
                             strategies_by_signal=options)
        self.assertEqual(report["consumed_decode_ms"], 35000)
        self.assertEqual([r["redecoded_audio_ms"] for r in report["regions"]],
                         [22000,13000])
        self.assertEqual(report["regions_detected"], 2)
        self.assertEqual(report["regions_structurally_cleared"], 2)
        self.assertEqual(report["consumed_native_jobs"], 2)
        self.assertLessEqual(report["consumed_decode_ms"], 36000)
        self.assertTrue(report["signal_specific_strategy_selection_is_opt_in"])
        self.assertEqual(
            report["signal_specific_strategy_override_keys"],
            sorted(options),
        )
        self.assertEqual(
            [r["strategy_policy_source"] for r in report["regions"]],
            ["repeated_ngram_within_single_cue", "consecutive_normalized_cues"],
        )
        self.assertEqual(
            [r["planned_strategy_names"][0] for r in report["regions"]],
            ["inside_cue_extended", "consecutive_short"],
        )
        self.assertFalse(report["auto_release_authorized"])

    def test_unmatched_signal_policy_key_fails_before_creating_output(self):
        mapping = {"accidental_typo": (
            RetryStrategy("unused", 30000, 0, 0, 0, 1),
        )}
        with self.assertRaisesRegex(ValueError, "no detected region"):
            self.invoke(name="invalid_key", strategies_by_signal=mapping)
        self.assertFalse((self.ws / "invalid_key").exists())
        self.assertFalse(self.adapter.calls)

    def test_full_source_signal_strategy_is_rejected_before_output(self):
        mapping = {"consecutive_normalized_cues": (
            RetryStrategy("unsafe_scope", 60000, 0, 0, 0, 1, scope="full_source"),
        )}
        with self.assertRaisesRegex(ValueError, "nonempty local retry"):
            self.invoke(name="unsafe_override", strategies_by_signal=mapping)
        self.assertFalse((self.ws / "unsafe_override").exists())

    def test_mixed_signal_group_rejects_ambiguous_strategy_override(self):
        # A cue itself repeats three times and is also repeated across
        # three separate captions, creating two different signal families.
        phrase = "A nested artificial repetitive phrase. " * 3
        self.baseline.write_text(srt([
            (5000, 8000, phrase),
            (8000, 11000, phrase),
            (11000, 14000, phrase),
        ]), encoding="utf-8")
        mapping = {"repeated_ngram_within_single_cue": (
            RetryStrategy("ambiguous", 30000, 0, 1000, 1000, 1),
        )}
        with self.assertRaisesRegex(ValueError, "mixed-signal"):
            self.invoke(name="mixed_signal", strategies_by_signal=mapping)
        self.assertFalse((self.ws / "mixed_signal").exists())

    def test_collision_is_refused(self):
        self.invoke()
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(len(self.adapter.calls), 2)


if __name__ == "__main__":
    unittest.main()
