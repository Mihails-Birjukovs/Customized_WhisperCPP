"""Conservative full-timeline splice tests; no private audio or transcript text."""
from pathlib import Path
from array import array
import tempfile
import sys
import unittest
import wave

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.full_timeline import (
    TimelinePolicy,assemble_full_timeline,write_full_timeline_preview
)
from adaptive_whisper.diagnostics import parse_srt,diagnose_srt
from adaptive_whisper.stitch import fmt


def srt(events):
    return "".join(f"{i}\n{fmt(a)} --> {fmt(b)}\n{t}\n\n"
                   for i,(a,b,t) in enumerate(events,1))


class FullTimelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.p=Path(self.tmp.name)
        with wave.open(str(self.p/"sample.wav"),"wb") as w:
            w.setframerate(16000);w.setnchannels(1);w.setsampwidth(2)
            w.writeframes(b"\0\0"*16000*4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_disjoint_replacement_retains_original_prefix_suffix_verbatim(self):
        baseline=srt([(0,900,"Original opening."),(1100,2100,"Original lost text."),
                      (2200,2900,"Original failed passage."),(3200,3800,"Original tail.")])
        retry=srt([(1000,1900,"Repaired passage A."),
                   (2100,2800,"Repaired passage B.")])
        merged,report=assemble_full_timeline(baseline,retry,
                                original_duration_ms=4000,
                                repaired_audio_window_ms=(1000,3000))
        cues=parse_srt(merged)
        self.assertEqual([x.text for x in cues],[
           "Original opening.","Repaired passage A.","Repaired passage B.","Original tail."])
        self.assertEqual(report["status"],"STRUCTURALLY_COMPOSED_UNVERIFIED")
        self.assertEqual(report["hazards_count"],0)
        self.assertEqual(report["unchanged_baseline_prefix_cues"],1)
        self.assertEqual(report["unchanged_baseline_suffix_cues"],1)
        self.assertEqual(report["discarded_baseline_cue_count"],2)
        self.assertTrue(report["source_clock_coverage_not_speech_coverage"])

    def test_overlapping_boundary_requires_review_and_keeps_both_source_hypotheses(self):
        baseline=srt([(0,1700,"First long spoken thought."),
                      (1900,3000,"Broken repeated repeated repeated.")])
        retry=srt([(1200,2500,"Alternative speech from original audio.")])
        merged,report=assemble_full_timeline(baseline,retry,
                              original_duration_ms=4000,
                              repaired_audio_window_ms=(1000,3500),
                              policy=TimelinePolicy(max_cross_source_overlap_ms=400))
        self.assertEqual(report["status"],"PROVISIONAL_REVIEW_REQUIRED")
        self.assertIn("BASELINE_PREFIX_CUE_CROSSES_RETRY_START",
                      {h["code"] for h in report["hazards"]})
        self.assertIn("CROSS_SOURCE_DIFFERENT_HYPOTHESES_OVERLAP",
                      {h["code"] for h in report["hazards"]})
        self.assertIn("First long spoken thought.",merged)
        self.assertIn("Alternative speech from original audio.",merged)
        self.assertFalse(report["human_audio_verification"])

    def test_nonadjacent_cross_source_overlap_is_also_reported(self):
        # Historical prefix spans two smaller alternative captions. Both
        # overlaps need review, even though the source order is A, B, C.
        baseline=srt([(0,3100,"Long historical hypothesis crosses both.")])
        retry=srt([(1700,2200,"Recovered first hypothesis."),
                   (2400,2900,"Recovered second hypothesis.")])
        _,r=assemble_full_timeline(baseline,retry,
                    original_duration_ms=4000,
                    repaired_audio_window_ms=(1600,3500),
                    policy=TimelinePolicy(max_cross_source_overlap_ms=400))
        hazards=[x for x in r["hazards"]
                 if x["code"]=="CROSS_SOURCE_DIFFERENT_HYPOTHESES_OVERLAP"]
        self.assertEqual(len(hazards),2)
        self.assertEqual([x["interval_ms"] for x in hazards],
                         [[1700,2200],[2400,2900]])
        self.assertEqual(r["status"],"PROVISIONAL_REVIEW_REQUIRED")

    def test_small_native_rounding_is_flagged_not_silently_clamped(self):
        baseline=srt([(0,2000,"Original lead.")])
        retry=srt([(2100,4020,"Recovered final text.")])
        out,r=assemble_full_timeline(baseline,retry,
                         original_duration_ms=4000,
                         repaired_audio_window_ms=(2000,4000))
        self.assertTrue(any(x["code"]=="RECOVERED_NATIVE_TIMESTAMP_ROUNDING"
                            for x in r["hazards"]))
        self.assertIn("00:00:04,020",out)
        self.assertEqual(r["status"],"PROVISIONAL_REVIEW_REQUIRED")

    def test_large_out_of_source_cue_rejected(self):
        baseline=srt([(0,2000,"Original lead.")])
        retry=srt([(2100,4300,"Decoder time is far past audio end.")])
        with self.assertRaises(ValueError):
            assemble_full_timeline(baseline,retry,
                         original_duration_ms=4000,
                         repaired_audio_window_ms=(2000,4000))

    def test_source_window_must_cover_selected_retry_and_have_captions(self):
        baseline=srt([(0,1500,"Original lead.")])
        with self.assertRaises(ValueError):
            assemble_full_timeline(baseline,srt([(2900,3900,"Too late.")]),
                    original_duration_ms=4000,
                    repaired_audio_window_ms=(1500,2700))
        with self.assertRaises(ValueError):
            assemble_full_timeline(baseline,"",original_duration_ms=4000,
                    repaired_audio_window_ms=(2000,4000))

    def test_boundary_gap_detected_without_automatic_caption_fabrication(self):
        baseline=srt([(0,900,"Opening."),(3600,3900,"Tail.")])
        retry=srt([(1800,2200,"Repaired speech.")])
        merged,r=assemble_full_timeline(baseline,retry,
                   original_duration_ms=4000,
                   repaired_audio_window_ms=(1000,3300),
                   policy=TimelinePolicy(min_gap_to_review_ms=500))
        self.assertEqual(r["status"],"PROVISIONAL_REVIEW_REQUIRED")
        self.assertTrue(any(x["code"]=="CROSS_SOURCE_CAPTION_GAP" for x in r["hazards"]))
        self.assertEqual(len(parse_srt(merged)),3)

    def test_write_wrapper_checks_original_hash_and_never_overwrites(self):
        a=self.p/"base.srt"
        b=self.p/"retry.srt"
        a.write_text(srt([(0,900,"Opening."),(1100,2100,"Bad text."),
                          (3200,3800,"Closing.")]),encoding="utf-8")
        b.write_text(srt([(1100,1800,"Corrected option."),
                          (2200,2900,"Different words.")]),encoding="utf-8")
        out=self.p/"preview.srt"
        ledger=self.p/"ledger.json"
        result=write_full_timeline_preview(a,b,out,ledger,
            source_audio=self.p/"sample.wav",
            repaired_audio_window_ms=(1000,3000))
        self.assertTrue(out.exists())
        self.assertEqual(result["input_file_sha256"]["original_audio"],
                         __import__("hashlib").sha256((self.p/"sample.wav").read_bytes()).hexdigest())
        self.assertEqual(len(result["acoustic_energy_proxy_near_boundaries"]),6)
        with self.assertRaises(FileExistsError):
            write_full_timeline_preview(a,b,out,ledger,
              source_audio=self.p/"sample.wav",
              repaired_audio_window_ms=(1000,3000))


if __name__=="__main__":
    unittest.main()
