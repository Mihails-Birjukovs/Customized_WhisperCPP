"""Synthetic offline audio-first word-alignment UI tests."""
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.alignment_workbench import (
    create_offline_workbench, wave_envelope, sha256
)


class OfflineAlignmentWorkbenchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.wav = self.root / "fake.wav"
        self.review = self.root / "review.json"
        self.html = self.root / "local_workbench.html"
        with wave.open(str(self.wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(b"\x40\x00"*(16000//2))
        self.evidence = {
            "schema":"whispercpp/human-audio-boundary-review/v1",
            "source_audio_sha256":hashlib.sha256(b"full original").hexdigest(),
            "locked_reference_sha256":hashlib.sha256(b"human CSV").hexdigest(),
            "start_ms":1000,"end_ms":1500,
            "speech_confirmed_by_human":True,
            "reviewer_count":1,
            "words":[
                {"index":1,"transcribed_word":"Latvian",
                 "audibility":"fully_heard","heard_fragment":None,
                 "word_start_ms":None,"word_end_ms":None,
                 "human_timing_verified":False},
                {"index":2,"transcribed_word":"vērtību",
                 "audibility":"partial_context_completed","heard_fragment":"vēr",
                 "word_start_ms":None,"word_end_ms":None,
                 "human_timing_verified":False},
            ],
        }
        self.review.write_text(json.dumps(self.evidence), encoding="utf-8")

    def build(self):
        return create_offline_workbench(
            self.review, self.wav, self.html,
            expected_review_sha256=sha256(self.review),
            expected_clip_sha256=sha256(self.wav),
            expected_locked_human_sha256=self.evidence["locked_reference_sha256"],
            original_source_duration_ms=10000
        )

    def test_generates_offline_local_playback_and_unfilled_words(self):
        report = self.build()
        self.assertEqual(report["human_word_count"],2)
        self.assertEqual(report["word_times_prepopulated"],0)
        self.assertFalse(report["auto_release_authorized"])
        html=self.html.read_text(encoding="utf-8")
        self.assertIn('src="fake.wav"',html)
        self.assertIn("Download review JSON",html)
        self.assertIn("word_start_ms",html)
        self.assertIn("word_end_ms",html)
        self.assertNotIn("https://",html)
        payload=re.search(
            r'<script id="data" type="application/json">(.*?)</script>',
            html,re.S
        ).group(1)
        blob=json.loads(payload)
        self.assertEqual(len(blob["waveform"]),100)
        self.assertEqual(blob["review"]["words"][1]["transcribed_word"],"vērtību")
        self.assertEqual(blob["review"]["words"][1]["word_start_ms"],None)
        self.assertEqual(blob["review"]["words"][1]["human_timing_verified"],False)

    def test_playhead_uses_animation_frames_and_supports_fine_scrubbing(self):
        self.build()
        page=self.html.read_text(encoding="utf-8")
        self.assertIn("requestAnimationFrame(animationTick)",page)
        self.assertIn('audio.addEventListener("play",beginAnimation)',page)
        self.assertIn('audio.addEventListener("pause",stopAnimation)',page)
        self.assertIn('audio.addEventListener("seeked",refreshPlayhead)',page)
        self.assertIn('id="back10"',page)
        self.assertIn('id="forward10"',page)
        self.assertIn('id="speed"',page)
        self.assertIn('value="0.5"',page)
        self.assertIn('audio.currentTime+ms/1000',page)
        self.assertNotIn('audio.addEventListener("timeupdate",()=>{\n document.getElementById("time")',page)
        self.assertIn("human_timing_verified",page)
        self.assertFalse(self.evidence["words"][0]["human_timing_verified"])

    def test_non_destructive_and_no_overwrites(self):
        a=sha256(self.wav);b=sha256(self.review)
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(sha256(self.wav),a)
        self.assertEqual(sha256(self.review),b)

    def test_hash_gate_blocks_changed_audio(self):
        h=sha256(self.wav)
        self.wav.write_bytes(self.wav.read_bytes()+b"extra")
        with self.assertRaisesRegex(ValueError,"SHA"):
            create_offline_workbench(
                self.review,self.wav,self.html,
                expected_review_sha256=sha256(self.review),
                expected_clip_sha256=h,
                expected_locked_human_sha256=self.evidence["locked_reference_sha256"],
                original_source_duration_ms=10000)
        self.assertFalse(self.html.exists())

    def test_reference_hash_gate(self):
        with self.assertRaisesRegex(ValueError,"reference"):
            create_offline_workbench(
                self.review,self.wav,self.html,
                expected_review_sha256=sha256(self.review),
                expected_clip_sha256=sha256(self.wav),
                expected_locked_human_sha256=hashlib.sha256(b"wrong").hexdigest(),
                original_source_duration_ms=10000)

    def test_wrong_review_hash_blocks_without_output(self):
        with self.assertRaisesRegex(ValueError,"hash"):
            create_offline_workbench(
                self.review,self.wav,self.html,
                expected_review_sha256=hashlib.sha256(b"wrong").hexdigest(),
                expected_clip_sha256=sha256(self.wav),
                expected_locked_human_sha256=self.evidence["locked_reference_sha256"],
                original_source_duration_ms=10000)
        self.assertFalse(self.html.exists())

    def test_bad_wave_format_fails_closed(self):
        with wave.open(str(self.wav),"wb") as f:
            f.setnchannels(2);f.setsampwidth(2);f.setframerate(16000)
            f.writeframes(b"\x00\x00"*16000)
        with self.assertRaisesRegex(ValueError,"mono"):
            self.build()
        self.assertFalse(self.html.exists())

    def test_not_full_duration_fails_closed(self):
        self.evidence["end_ms"] = 1600
        self.review.write_text(json.dumps(self.evidence),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"duration"):
            self.build()
        self.assertFalse(self.html.exists())

    def test_script_injection_is_serialized_inertly(self):
        # Model/user words must never be interpreted as HTML/JS.
        self.evidence["words"][0]["transcribed_word"] = "Latvian"
        self.evidence["private_note"] = "</script><script>alert('x')</script>"
        self.review.write_text(json.dumps(self.evidence),encoding="utf-8")
        self.build()
        html=self.html.read_text(encoding="utf-8")
        self.assertNotIn("</script><script>alert",html)
        self.assertIn("\\u003c/script",html)

    def test_verified_word_timing_requires_new_version(self):
        self.evidence["words"][0].update(
            word_start_ms=1050,word_end_ms=1130,human_timing_verified=True)
        self.review.write_text(json.dumps(self.evidence),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"new version"):
            self.build()

    def test_waveform_not_word_inference(self):
        values=wave_envelope(
            self.wav,clip_sha256=sha256(self.wav),duration_ms=500)
        self.assertEqual(len(values),100)
        self.assertTrue(all(a<=b for a,b in values))
        self.assertEqual(values[0],[0.00195,0.00195])


if __name__=="__main__":
    unittest.main()
