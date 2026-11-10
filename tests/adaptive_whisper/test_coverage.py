"""Synthetic sound-energy transcript-timeline diagnostic tests, no ASR truth."""
from pathlib import Path
from array import array
import sys
import tempfile
import unittest
import wave
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.coverage import CoveragePolicy,coverage_report,multi_threshold_report,compare_machine_hypotheses,region_output_diversity,region_energy_coverage
from adaptive_whisper.stitch import fmt

def srt(bounds):
    return "".join(f"{n}\n{fmt(a)} --> {fmt(b)}\nSynthetic words only.\n\n"
                   for n,(a,b) in enumerate(bounds,1))

class EnergyCoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/"synthetic.wav"
        with wave.open(str(self.path),"wb") as w:
            w.setframerate(16000);w.setnchannels(1);w.setsampwidth(2)
            # 1000ms synthetic *sound* (not guaranteed speech), 1000ms silence.
            vals=array("h", [10000 if i%2 else -10000 for i in range(16000)])
            if sys.byteorder!="little": vals.byteswap()
            w.writeframes(vals.tobytes()+b"\0\0"*16000)

    def tearDown(self):
        self.tmp.cleanup()

    def test_padded_cue_encloses_audio_energy(self):
        x=coverage_report(self.path,srt([(0,1000)]),
                    CoveragePolicy(cue_pad_ms=0))
        self.assertEqual(x["high_energy_duration_ms"],1000)
        self.assertEqual(x["high_energy_cue_coverage_fraction"],1.0)
        self.assertEqual(x["uncaptioned_high_energy_intervals_count"],0)
        self.assertTrue(x["not_ground_truth"])

    def test_no_caption_reports_uncaptioned_sound_not_missing_speech(self):
        x=coverage_report(self.path,"",CoveragePolicy(cue_pad_ms=0))
        self.assertEqual(x["high_energy_not_in_padded_cue_ms"],1000)
        self.assertEqual(x["longest_uncaptioned_high_energy_ms"],1000)
        self.assertEqual(x["uncaptioned_high_energy_intervals_ms"],[(0,1000)])
        self.assertTrue(x["cannot_certify_no_speech_omission"])

    def test_partial_caption_counts_uncovered_energy(self):
        x=coverage_report(self.path,srt([(0,500)]),
                    CoveragePolicy(cue_pad_ms=0,long_uncovered_ms=200))
        self.assertEqual(x["high_energy_cue_coverage_fraction"],0.5)
        self.assertEqual(x["uncaptioned_high_energy_intervals_ms"],[(500,1000)])

    def test_invalid_cue_timing_is_recorded(self):
        x=coverage_report(self.path,srt([(200,100)]))
        self.assertEqual(x["invalid_or_nonmonotone_cue_count"],1)

    def test_threshold_sensitivity_and_zero_energy(self):
        x=multi_threshold_report(self.path,srt([(0,1000)]))
        self.assertEqual(len(x["measurements"]),3)
        self.assertTrue(all(y["high_energy_duration_ms"]==1000
                            for y in x["measurements"]))
        with self.assertRaises(ValueError):
            CoveragePolicy(energy_dbfs=-100)

    def test_identical_machine_outputs_never_generate_divergence_warning(self):
        ref=srt([(100,1000)])
        m=compare_machine_hypotheses(ref,ref)
        self.assertEqual(m["token_multiset_f1_machine_agreement"],1)
        self.assertEqual(m["review_warnings"],[])
        self.assertTrue(m["not_reference_truth"])

    def test_large_machine_deletion_flags_review_when_reference_healthy(self):
        baseline=("1\n00:00:00,000 --> 00:00:03,000\n"
                  "Here are many words in an explanation of numerical methods "
                  "and finite differences for continuum physics.\n\n")
        candidate=("1\n00:00:00,000 --> 00:00:03,000\nHere are words.\n\n")
        m=compare_machine_hypotheses(baseline,candidate)
        self.assertIn("token_count_ratio_large_change",m["review_warnings"])
        self.assertIn("machine_token_bag_large_disagreement",m["review_warnings"])

    def test_pathological_machine_reference_cannot_define_truth(self):
        baseline=("1\n00:00:00,000 --> 00:00:03,000\n"
                  "repeat repeat repeat repeat repeat repeat repeat repeat\n\n")
        candidate=("1\n00:00:00,000 --> 00:00:03,000\nreal speech\n\n")
        m=compare_machine_hypotheses(baseline,candidate,baseline_pathological=True)
        self.assertEqual(m["review_warnings"],[])
        self.assertTrue(m["word_error_rate_unavailable"])

    def test_fixed_region_identical_run_vs_varied_output(self):
        old=srt([(0,1000),(1000,2000),(2000,3000)])
        # Identical captions are expected because the helper uses one text.
        a=region_output_diversity(old,0,3000)
        self.assertEqual(a["intersecting_cues"],3)
        self.assertEqual(a["unique_normalized_caption_count"],1)
        self.assertEqual(a["longest_identical_caption_run_ms"],3000)
        self.assertEqual(a["longest_identical_caption_run_cues"],3)
        new=("1\n00:00:00,000 --> 00:00:01,000\nFirst text.\n\n"
             "2\n00:00:01,000 --> 00:00:02,000\nSecond text.\n\n"
             "3\n00:00:02,000 --> 00:00:03,000\nThird text.\n\n")
        b=region_output_diversity(new,0,3000)
        self.assertEqual(b["unique_normalized_caption_count"],3)
        self.assertEqual(b["longest_identical_caption_run_ms"],0)
        self.assertTrue(b["not_truth"])

    def test_fixed_region_empty_and_invalid_intervals(self):
        a=region_output_diversity(srt([(0,1000)]),5000,8000)
        self.assertEqual(a["intersecting_cues"],0)
        self.assertIsNone(a["unique_caption_fraction"])
        with self.assertRaises(ValueError):
            region_output_diversity(srt([(0,1000)]),2000,1000)

    def test_region_sound_energy_coverage_is_localized(self):
        roi=region_energy_coverage(self.path,srt([(0,500)]),0,1000,
             CoveragePolicy(cue_pad_ms=0,long_uncovered_ms=200))
        self.assertEqual(roi["high_energy_ms"],1000)
        self.assertEqual(roi["high_energy_caption_coverage_fraction"],0.5)
        self.assertEqual(roi["uncaptioned_high_energy_ms"],500)
        self.assertEqual(roi["uncaptioned_intervals_ms"],[(500,1000)])
        self.assertTrue(roi["not_proof_of_speech_retention"])

    def test_region_sound_energy_rejects_oob(self):
        with self.assertRaises(ValueError):
            region_energy_coverage(self.path,"",1000,2500)

    def test_rejects_nonstandard_sample_format(self):
        nonmono=Path(self.tmp.name)/"stereo.wav"
        with wave.open(str(nonmono),"wb") as w:
            w.setframerate(16000);w.setnchannels(2);w.setsampwidth(2)
            w.writeframes(b"\0"*6400)
        with self.assertRaises(ValueError):
            coverage_report(nonmono,"")
if __name__=="__main__":
    unittest.main()
