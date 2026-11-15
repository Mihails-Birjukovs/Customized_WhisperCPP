"""Controlled audio/subtitle gap detector and fail-closed retry tests.

Noise/tone can trigger this energy proxy; true speech and WER require
independent human verification, which these synthetic tests do not claim.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import struct
import sys
import wave
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.uncaptioned_recovery import (
    GapSearchPolicy,find_uncaptioned_activity,plan_dual_gap_windows,
    caption_gap_envelope,machine_gap_consensus,machine_gap_fragment_proposals
)
from adaptive_whisper.stitch import fmt


def wav(path:Path,blocks):
    """blocks: [(milliseconds, PCM amplitude),...]"""
    with wave.open(str(path),"wb") as w:
        w.setframerate(16000);w.setnchannels(1);w.setsampwidth(2)
        for ms,amplitude in blocks:
            samples=int(ms*16)
            if not amplitude:
                w.writeframes(b"\x00\x00"*samples)
            else:
                # Definitely *not* speech: alternating electronic tone.
                w.writeframes(
                    b"".join(struct.pack("<h",amplitude if i%4<2 else -amplitude)
                             for i in range(samples))
                )


def srt(rows):
    return "".join(
        f"{i}\n{fmt(a)} --> {fmt(b)}\n{text}\n\n"
        for i,(a,b,text) in enumerate(rows,1)
    )


class UncaptionedCandidateTests(unittest.TestCase):
    def setUp(self):
        td=TemporaryDirectory();self.addCleanup(td.cleanup)
        self.audio=Path(td.name)/"fake.wav"

    def test_silent_file_produces_no_candidates(self):
        wav(self.audio,[(5000,0)])
        result=find_uncaptioned_activity(self.audio,"")
        self.assertEqual(result["candidate_count_total"],0)
        self.assertFalse(result["auto_release_authorized"])

    def test_noise_without_speech_produces_candidate_not_truth(self):
        wav(self.audio,[(1000,0),(2500,4000),(1500,0)])
        found=find_uncaptioned_activity(self.audio,"")
        self.assertEqual(found["candidate_count_total"],1)
        c=found["candidates"][0]
        self.assertEqual(c["source_interval_ms"],[1000,3500])
        self.assertTrue(c["machine_word_omission_unverified"])
        self.assertTrue(found["candidate_is_not_speech_ground_truth"])

    def test_caption_coverage_suppresses_false_alarm_on_tone(self):
        wav(self.audio,[(1000,0),(2500,4000),(1500,0)])
        covered=find_uncaptioned_activity(
            self.audio,srt([(1000,3500,"Machine caption, not acoustic truth")]))
        self.assertEqual(covered["candidate_count_total"],0)

    def test_known_removed_caption_triggers_without_human_reference(self):
        wav(self.audio,[(1000,0),(2500,4000),(1500,0)])
        base=srt([(800,3600,"Machine caption")])
        full=find_uncaptioned_activity(self.audio,base)
        missing=find_uncaptioned_activity(self.audio,"")
        self.assertEqual(full["candidate_count_total"],0)
        self.assertGreater(missing["candidate_count_total"],0)
        self.assertTrue(missing["human_words_not_used"])

    def test_min_two_threshold_support_blocks_low_energy(self):
        wav(self.audio,[(1000,0),(2500,200),(1500,0)])
        report=find_uncaptioned_activity(self.audio,"")
        self.assertEqual(report["candidate_count_total"],0)

    def test_malformed_srt_rejected(self):
        wav(self.audio,[(5000,0)])
        with self.assertRaisesRegex(ValueError,"Malformed"):
            find_uncaptioned_activity(self.audio,"not a subtitle")

    def test_out_of_range_cue_rejected(self):
        wav(self.audio,[(5000,0)])
        with self.assertRaisesRegex(ValueError,"Out-of-source"):
            find_uncaptioned_activity(
                self.audio,srt([(1000,7000,"out of range")]))

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            GapSearchPolicy(thresholds_dbfs=(-40.,-40.))
        with self.assertRaises(ValueError):
            GapSearchPolicy(min_threshold_support=4)

    def test_caption_gap_expands_energy_core_to_neighbor_boundaries(self):
        text=srt([(0,1000,"before"),(4000,5000,"after")])
        c={"source_interval_ms":[1500,3400]}
        x=caption_gap_envelope(text,c,source_duration_ms=5000)
        self.assertEqual(x["source_interval_ms"],[1000,4000])
        self.assertFalse(x["automatic_release_authorized"])

    def test_caption_gap_rejects_existing_cue(self):
        text=srt([(0,2000,"intersects")])
        with self.assertRaisesRegex(ValueError,"overlaps"):
            caption_gap_envelope(text,{"source_interval_ms":[1500,3400]},
                                 source_duration_ms=5000)

    def test_excessively_long_caption_gap_not_accepted(self):
        text=srt([(0,1000,"before"),(35000,36000,"after")])
        with self.assertRaisesRegex(ValueError,"Unbounded"):
            caption_gap_envelope(text,{"source_interval_ms":[19000,22000]},
                                 source_duration_ms=36000)

    def test_two_source_windows_and_clipping(self):
        plans=plan_dual_gap_windows(
            {"source_interval_ms":[15000,17000]},75000
        )
        self.assertEqual(plans[0]["source_interval_ms"],[11000,21000])
        self.assertEqual(plans[1]["source_interval_ms"],[3000,29000])
        self.assertFalse(plans[0]["no_human_word_truth"] is False)

    def test_identical_clipped_windows_refused(self):
        with self.assertRaisesRegex(ValueError,"identical"):
            plan_dual_gap_windows(
                {"source_interval_ms":[0,2000]},4000,
                context_padding_ms=(4000,12000)
            )

    def test_strict_machine_consensus_review_not_release(self):
        gap={"source_interval_ms":[2000,4000]}
        # Different context windows, same words in gap, no surrounding cues.
        retry={
            "short":srt([(2000,3500,"Some words here")]),
            "long":srt([(2000,3500,"some words here")]),
        }
        r=machine_gap_consensus(
            srt([(0,1000,"before"),(4500,5000,"after")]),
            gap,retry,{"short":0,"long":0},
            original_duration_ms=6000)
        self.assertTrue(r["machine_consensus_token_sequence_identical"])
        self.assertEqual(r["candidate_words_count"],3)
        self.assertFalse(r["automatic_full_timeline_splice_authorized"])
        self.assertFalse(r["automatic_release_authorized"])

    def test_conflicting_machine_words_cause_abstention(self):
        gap={"source_interval_ms":[2000,4000]}
        r=machine_gap_consensus(
            "",gap,{
                "short":srt([(2000,3500,"some correct words")]),
                "long":srt([(2000,3500,"different words")])
            },{"short":0,"long":0},original_duration_ms=6000)
        self.assertFalse(r["machine_consensus_token_sequence_identical"])
        self.assertEqual(r["result_status"],"ABSTAIN_UNRESOLVED_MACHINE_WORDS")

    def test_partial_clip_crossing_yields_no_consensus(self):
        gap={"source_interval_ms":[2000,4000]}
        r=machine_gap_consensus(
            "",gap,{
                "short":srt([(1500,4100,"some words")]),
                "long":srt([(1500,4100,"some words")])
            },{"short":0,"long":0},original_duration_ms=6000)
        self.assertEqual(r["candidate_words_count"],0)
        self.assertFalse(r["machine_consensus_token_sequence_identical"])

    def test_matching_inside_words_with_crossing_cues_must_still_abstain(self):
        gap={"source_interval_ms":[2000,4000]}
        r=machine_gap_consensus(
            "",gap,{
                "short":srt([(1500,2100,"unowned prefix"),
                             (2100,3800,"agreed inner words")]),
                "long":srt([(1500,2100,"unowned prefix"),
                            (2100,3800,"agreed inner words")])
            },{"short":0,"long":0},original_duration_ms=6000)
        self.assertGreater(r["per_retry"]["short"]["native_boundary_crossing_cues"],0)
        self.assertFalse(r["machine_consensus_token_sequence_identical"])
        self.assertEqual(r["result_status"],"ABSTAIN_UNRESOLVED_MACHINE_WORDS")

    def test_single_owned_native_gap_fragment_is_private_review_only(self):
        original=srt([(0,2000,"left"),(4000,6000,"right")])
        result=machine_gap_fragment_proposals(
            original,{"source_interval_ms":[2000,4000]},
            {"short":srt([(1500,4100,"crossing short phrase")]),
             "long":srt([(1000,2100,"long crossing left"),
                         (2200,3800,"possibly recovered words")])},
            {"short":0,"long":0},original_duration_ms=6000)
        self.assertFalse(result["fragments_by_context"]["short"])
        self.assertIn("possibly recovered words",result["fragments_by_context"]["long"])
        self.assertEqual(result["candidate_diagnostics"]["long"]["strictly_owned_cues"],1)
        self.assertEqual(result["candidate_diagnostics"]["short"]["crossing_cues_excluded_from_preview"],1)
        self.assertTrue(result["any_fragment_available"])
        self.assertFalse(result["auto_release_authorized"])

    def test_fragment_proposals_do_not_trim_crossing_cues(self):
        result=machine_gap_fragment_proposals(
            "",{"source_interval_ms":[2000,4000]},
            {"short":srt([(1500,4500,"outside")]),
             "long":srt([(1500,4100,"outside too")])},
            {"short":0,"long":0},original_duration_ms=6000)
        self.assertFalse(result["any_fragment_available"])
        self.assertEqual(result["fragments_by_context"]["short"],"")
        self.assertEqual(result["fragments_by_context"]["long"],"")

    def test_cannot_rewrite_existing_original_cue(self):
        with self.assertRaisesRegex(ValueError,"not an empty"):
            machine_gap_consensus(
                srt([(2500,3500,"existing")]),
                {"source_interval_ms":[2000,4000]},
                {"short":"","long":""},
                {"short":0,"long":0},
                original_duration_ms=6000)

    def test_redecode_source_origin_is_checked(self):
        with self.assertRaisesRegex(ValueError,"source-clock"):
            machine_gap_consensus(
                "",{"source_interval_ms":[2000,4000]},
                {"short":"","long":""},
                {"short":3000,"long":0},
                original_duration_ms=6000)


if __name__=="__main__":
    unittest.main()
