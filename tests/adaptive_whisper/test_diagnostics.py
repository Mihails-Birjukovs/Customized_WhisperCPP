"""No private audio or transcript contents in repository tests."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.diagnostics import diagnose_srt, parse_srt, time_ms


def cue(i: int, start: str, stop: str, text: str) -> str:
    return f"{i}\n{start} --> {stop}\n{text}\n\n"


class SrtDiagnosticTests(unittest.TestCase):
    def test_long_exact_repetition_run(self):
        phrase = "An illustrative artificial caption repeated by a broken decoder."
        s = "".join(cue(i + 1, f"00:00:{i:02d},000", f"00:00:{i+1:02d},000", phrase) for i in range(4))
        d = diagnose_srt(s)
        self.assertEqual(d.cue_count, 4)
        self.assertEqual(len(d.candidates), 1)
        self.assertEqual(d.candidates[0].segment_count, 4)
        self.assertEqual(d.candidates[0].duration_ms, 4000)
        self.assertEqual(d.invalid_timing_cues, 0)

    def test_short_repetition_not_flagged(self):
        p = "Repeated speech can be intentional; do not infer acoustic error."
        s = cue(1, "00:00:00,000", "00:00:01,000", p)
        s += cue(2, "00:00:01,000", "00:00:02,000", p)
        self.assertEqual(len(diagnose_srt(s).candidates), 0)

    def test_non_monotone_and_negative_interval_reported(self):
        s = cue(1, "00:00:04,000", "00:00:03,000", "First line")
        s += cue(2, "00:00:01,000", "00:00:02,000", "Second line")
        d = diagnose_srt(s)
        self.assertEqual(d.invalid_timing_cues, 1)
        self.assertEqual(d.backwards_start_cues, 1)

    def test_invalid_timestamp_is_error(self):
        with self.assertRaises(ValueError):
            parse_srt("1\n00:01:65,123 --> 00:01:66,123\nText\n")

    def test_case_and_whitespace_normalized_for_detection(self):
        phrase = "The same phrase should be grouped after whitespace normalization."
        s = cue(1, "00:00:00,000", "00:00:01,000", phrase)
        s += cue(2, "00:00:01,000", "00:00:02,000", phrase.upper())
        s += cue(3, "00:00:02,000", "00:00:03,000", "  " + phrase)
        self.assertEqual(diagnose_srt(s).candidates[0].segment_count, 3)


if __name__ == "__main__":
    unittest.main()
