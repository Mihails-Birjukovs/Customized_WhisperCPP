"""Synthetic byte-fidelity and negative-control tests for Stage 10."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.natural_gate import plan_natural_prefix, prepare_natural_prefix


def srt(cues):
    def time(ms):
        h, r = divmod(ms, 3600000)
        m, r = divmod(r, 60000)
        s, r = divmod(r, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{r:03d}"
    return "".join(f"{i}\r\n{time(a)} --> {time(b)}\r\n{text}\r\n\r\n"
                   for i, (a, b, text) in enumerate(cues, 1))


class NaturalFixtureGateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Path(tmp.name)
        self.wav = self.ws / "source.wav"
        with wave.open(str(self.wav), "wb") as w:
            w.setframerate(16000)
            w.setnchannels(1)
            w.setsampwidth(2)
            w.writeframes(b"\x07\x00" * (16000 * 90))
        first = "Synthetic hallucination phrase one repeated identically."
        second = "Separate hallucination phrase two repeated identically."
        self.baseline_bytes = srt([
            (0, 5000, "Natural start of recording."),
            (10000, 12000, first),
            (12000, 14000, first),
            (14000, 16000, first),
            (20000, 21000, "Distinct middle dialogue."),
            (40000, 42000, second),
            (42000, 44000, second),
            (44000, 46000, second),
            (55000, 60000, "Last cue exactly at planned test boundary."),
            (65000, 70000, "Later healthy utterance, intentionally excluded."),
        ]).encode("utf-8")
        self.srt = self.ws / "historical.srt"
        self.srt.write_bytes(self.baseline_bytes)
        self.expected = ((10000, 16000), (40000, 46000))

    def test_byte_exact_historical_srt_prefix_preserves_crlf_and_all_words(self):
        result, report = plan_natural_prefix(
            self.baseline_bytes, prefix_end_ms=60000,
            expected_regions_ms=self.expected
        )
        marker = self.baseline_bytes.index(b"10\r\n")
        self.assertEqual(result, self.baseline_bytes[:marker])
        self.assertEqual(report["retained_srt_cue_count"], 9)
        self.assertEqual(report["region_bounds_ms"], [[10000,16000],[40000,46000]])
        self.assertIn(b"\r\n", result)
        self.assertEqual(report["prefix_srt_sha256"],
                         hashlib.sha256(result).hexdigest())

    def test_complete_wave_prefix_and_original_hashes_preserved(self):
        wav_sha = hashlib.sha256(self.wav.read_bytes()).hexdigest()
        srt_sha = hashlib.sha256(self.srt.read_bytes()).hexdigest()
        folder = self.ws / "derived"
        result = prepare_natural_prefix(
            self.ws, original_wav=self.wav, original_srt=self.srt,
            destination=folder, prefix_end_ms=60000,
            expected_regions_ms=self.expected
        )
        self.assertEqual(result["derived_prefix_pcm_frames"], 60000*16)
        with wave.open(str(folder/"natural_prefix_pcm16.wav"), "rb") as w:
            self.assertEqual(w.getnframes(), 60000*16)
            self.assertEqual(w.getframerate(),16000)
            self.assertEqual(w.readframes(2), b"\x07\x00\x07\x00")
        self.assertEqual(hashlib.sha256(self.wav.read_bytes()).hexdigest(),wav_sha)
        self.assertEqual(hashlib.sha256(self.srt.read_bytes()).hexdigest(),srt_sha)
        self.assertEqual((folder/"historical_prefix_exact_original_bytes.srt").read_bytes(),
                         self.baseline_bytes[:self.baseline_bytes.index(b"10\r\n")])
        self.assertEqual(json.loads((folder/"natural_prefix_manifest.json").read_text(
            encoding="utf-8"))["original_wav_sha256"],wav_sha)
        self.assertFalse(result["original_files_modified"])

    def test_cutting_through_caption_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"cut through"):
            plan_natural_prefix(
                self.baseline_bytes, prefix_end_ms=57000,
                expected_regions_ms=self.expected)

    def test_cutting_inside_caption_is_refused_before_any_output(self):
        folder=self.ws/"bad_cut"
        with self.assertRaises(ValueError):
            prepare_natural_prefix(
                self.ws,original_wav=self.wav,original_srt=self.srt,
                destination=folder,prefix_end_ms=57000,
                expected_regions_ms=self.expected)
        self.assertFalse(folder.exists())

    def test_missing_exact_final_subtitle_is_refused(self):
        with self.assertRaisesRegex(ValueError,"exact last selected subtitle"):
            plan_natural_prefix(
                self.baseline_bytes,prefix_end_ms=61000,
                expected_regions_ms=self.expected)

    def test_wrong_expected_pathology_count_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError,"Expected exactly"):
            plan_natural_prefix(
                self.baseline_bytes,prefix_end_ms=60000,
                expected_regions_ms=((10000,16000),))

    def test_cannot_overwrite_existing_fixture(self):
        folder=self.ws/"already"
        folder.mkdir()
        sentinel=folder/"keep.txt"
        sentinel.write_text("untouched")
        with self.assertRaises(FileExistsError):
            prepare_natural_prefix(
                self.ws,original_wav=self.wav,original_srt=self.srt,
                destination=folder,prefix_end_ms=60000,
                expected_regions_ms=self.expected)
        self.assertEqual(sentinel.read_text(),"untouched")

    def test_rejects_short_wav_before_writing(self):
        short=self.ws/"short.wav"
        with wave.open(str(short),"wb") as w:
            w.setframerate(16000);w.setnchannels(1);w.setsampwidth(2)
            w.writeframes(b"\x00\x00"*16000*5)
        out=self.ws/"too_short"
        with self.assertRaisesRegex(ValueError,"not shorter"):
            prepare_natural_prefix(
                self.ws,original_wav=short,original_srt=self.srt,
                destination=out,prefix_end_ms=60000,
                expected_regions_ms=self.expected)
        self.assertFalse(out.exists())

    def test_rejects_non_mono_pcm16_before_writing(self):
        stereo=self.ws/"stereo.wav"
        with wave.open(str(stereo),"wb") as w:
            w.setframerate(16000);w.setnchannels(2);w.setsampwidth(2)
            w.writeframes(b"\x00\x00"*16000*90*2)
        out=self.ws/"invalid"
        with self.assertRaisesRegex(ValueError,"mono PCM16"):
            prepare_natural_prefix(
                self.ws,original_wav=stereo,original_srt=self.srt,
                destination=out,prefix_end_ms=60000,
                expected_regions_ms=self.expected)
        self.assertFalse(out.exists())

    def test_corrupt_srt_header_fails_preservation_gate(self):
        corrupted = self.baseline_bytes.replace(
            b"10\r\n01:05:00,000", b"00x10\r\n01:05:00,000"
        )
        # Corrupt a cue header that is actually present, and reject it.
        corrupted = self.baseline_bytes.replace(
            b"10\r\n00:01:05,000",b"invalid_header\r\n00:01:05,000"
        )
        with self.assertRaises(ValueError):
            plan_natural_prefix(
                corrupted,prefix_end_ms=60000,
                expected_regions_ms=self.expected)

    def test_rejects_unsupported_extra_unmatched_cue_layout(self):
        invalid=self.baseline_bytes.replace(b"9\r\n",b"nine\r\n")
        with self.assertRaises(ValueError):
            plan_natural_prefix(
                invalid,prefix_end_ms=60000,
                expected_regions_ms=self.expected)


if __name__ == "__main__":
    unittest.main()
