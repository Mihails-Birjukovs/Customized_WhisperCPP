"""Synthetic regression tests: a speech gap cannot be cleared by SRT/energy.

Human word times are never inferred; source and output stay immutable.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.boundary_review import (
    assess_human_boundary, validate_human_interval, write_boundary_assessment
)


def h(raw: bytes = b"seed") -> str:
    return hashlib.sha256(raw).hexdigest()


def write_pcm(path):
    with wave.open(str(path), "wb") as wav:
        wav.setframerate(16000)
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.writeframes(b"\x00\x00" * (16000 * 4))


def srt(cues):
    def time(v):
        hour, rest = divmod(v, 3600000)
        minute, rest = divmod(rest, 60000)
        second, ms = divmod(rest, 1000)
        return f"{hour:02}:{minute:02}:{second:02},{ms:03}"
    return "".join(
        f"{i}\n{time(a)} --> {time(b)}\n{text}\n\n"
        for i, (a, b, text) in enumerate(cues, 1)
    )


def review(start=1000, end=3200):
    return {
        "schema": "whispercpp/human-audio-boundary-review/v1",
        "start_ms": start, "end_ms": end,
        "speech_confirmed_by_human": True,
        "reviewer_count": 1,
        "source_audio_sha256": h(b"sourceaudio"),
        "locked_reference_sha256": h(b"reviewer"),
        "words": [
            {
                "index": 1, "transcribed_word": "Clear",
                "audibility": "fully_heard",
                "heard_fragment": None,
                "word_start_ms": None, "word_end_ms": None,
                "human_timing_verified": False,
            },
            {
                "index": 2, "transcribed_word": "completion",
                "audibility": "partial_context_completed",
                "heard_fragment": "comp",
                "word_start_ms": None, "word_end_ms": None,
                "human_timing_verified": False,
            }
        ],
    }


class HumanBoundaryGateTests(unittest.TestCase):
    def setUp(self):
        self.evidence = review()

    def assess(self, captions, evidence=None):
        return assess_human_boundary(
            srt(captions), evidence or self.evidence,
            original_duration_ms=4000
        )

    def test_unambiguous_gap_is_confirmed_omission_not_release(self):
        result = self.assess([(0, 500, "Intro"), (3500, 3900, "Outro")])
        self.assertEqual(result["candidate_status"],
                         "HUMAN_CONFIRMED_SPEECH_WITHOUT_CAPTION")
        self.assertEqual(result["intersecting_cue_count"], 0)
        self.assertIn("WORD_TIMESTAMPS_NOT_HUMAN_VERIFIED",
                      result["blockers"])
        self.assertIn("CONTEXT_COMPLETED_WORD_NOT_FULLY_AUDIBLE",
                      result["blockers"])
        self.assertFalse(result["ready_for_editorial_word_review"])
        self.assertFalse(result["auto_release_authorized"])
        self.assertTrue(result["report_contains_no_transcript_text"])
        self.assertNotIn("completion", str(result))

    def test_three_crossing_cues_cannot_certify_words(self):
        result = self.assess([
            (0, 1300, "Wrong before boundary"),
            (1300, 2600, "Entirely wrong text"),
            (2600, 3600, "Wrong after boundary")
        ])
        self.assertEqual(result["candidate_status"],
                         "CAPTION_PRESENT_TEXT_UNVERIFIED")
        self.assertEqual(result["intersecting_cue_count"], 3)
        self.assertEqual(result["crossing_cue_count"], 2)
        self.assertIn("CUE_OVERLAPS_REVIEW_CLIP_BOUNDARY",
                      result["blockers"])
        self.assertFalse(result["machine_words_compared_to_human_reference"])
        self.assertFalse(result["auto_release_authorized"])

    def test_all_manual_timestamps_are_only_editorial_ready(self):
        evidence = deepcopy(self.evidence)
        evidence["words"][0].update(
            word_start_ms=1200, word_end_ms=1500,
            human_timing_verified=True
        )
        evidence["words"][1].update(
            audibility="fully_heard", heard_fragment=None,
            word_start_ms=1700, word_end_ms=2100,
            human_timing_verified=True
        )
        result = self.assess([(1000, 3200, "Some candidate")], evidence)
        self.assertTrue(result["ready_for_editorial_word_review"])
        self.assertFalse(result["auto_release_authorized"])
        self.assertIn("EXPLICIT_EDITORIAL_DECISION_REQUIRED", result["blockers"])

    def test_timed_partially_audible_word_remains_blocked(self):
        evidence = deepcopy(self.evidence)
        for index, row in enumerate(evidence["words"]):
            row.update(word_start_ms=1100+index*500,
                       word_end_ms=1300+index*500,
                       human_timing_verified=True)
        result = self.assess([(1000, 3200, "Some candidate")], evidence)
        self.assertFalse(result["ready_for_editorial_word_review"])
        self.assertEqual(result["human_evidence"]["human_timed_word_count"], 2)

    def test_missing_human_speech_attestation_fails_closed(self):
        evidence=deepcopy(self.evidence)
        evidence["speech_confirmed_by_human"]=False
        with self.assertRaises(ValueError):
            self.assess([], evidence)

    def test_hashes_and_reviewer_count_are_strict(self):
        for key,value in (
            ("source_audio_sha256", "z"*64),
            ("locked_reference_sha256", "bad"), ("reviewer_count", 2),
        ):
            evidence=deepcopy(self.evidence)
            evidence[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.assess([], evidence)

    def test_rejects_false_verified_word_times(self):
        evidence=deepcopy(self.evidence)
        evidence["words"][0]["human_timing_verified"]=True
        with self.assertRaisesRegex(ValueError,"paired"):
            self.assess([], evidence)

    def test_rejects_word_outside_audio_clip(self):
        evidence=deepcopy(self.evidence)
        evidence["words"][0].update(word_start_ms=3200, word_end_ms=3300,
                                    human_timing_verified=True)
        with self.assertRaisesRegex(ValueError,"outside|exceeds"):
            self.assess([], evidence)

    def test_rejects_nonmonotone_human_times(self):
        evidence=deepcopy(self.evidence)
        evidence["words"][0].update(word_start_ms=2000, word_end_ms=2100,
                                    human_timing_verified=True)
        evidence["words"][1].update(word_start_ms=1500, word_end_ms=1700,
                                    human_timing_verified=True)
        with self.assertRaisesRegex(ValueError,"monotonic"):
            self.assess([], evidence)

    def test_rejects_partial_fragment_that_is_not_word_onset(self):
        evidence=deepcopy(self.evidence)
        evidence["words"][1]["heard_fragment"]="other"
        with self.assertRaisesRegex(ValueError,"onset"):
            self.assess([], evidence)

    def test_rejects_nonconsecutive_word_ids_and_sentence_tokens(self):
        evidence=deepcopy(self.evidence)
        evidence["words"][1]["index"]=3
        with self.assertRaises(ValueError):
            self.assess([], evidence)
        evidence=deepcopy(self.evidence)
        evidence["words"][0]["transcribed_word"]="Clear two words"
        with self.assertRaises(ValueError):
            self.assess([], evidence)

    def test_rejects_candidate_out_of_source_clock(self):
        with self.assertRaisesRegex(ValueError,"timing"):
            self.assess([(3200, 4500, "Bad beyond original source")])

    def test_boolean_reviewers_and_indices_cannot_bypass_schema(self):
        evidence=deepcopy(self.evidence)
        evidence["reviewer_count"]=True
        with self.assertRaisesRegex(ValueError,"reviewer"):
            self.assess([],evidence)
        evidence=deepcopy(self.evidence)
        evidence["words"][0]["index"]=True
        with self.assertRaisesRegex(ValueError,"index"):
            self.assess([],evidence)

    def test_nonempty_malformed_srt_is_not_misdiagnosed_as_missing_caption(self):
        with self.assertRaisesRegex(ValueError,"invalid candidate SRT"):
            assess_human_boundary(
                "This is not a subtitle file.",self.evidence,
                original_duration_ms=4000
            )

    def test_no_human_timestamps_can_be_derived_from_cue_clock(self):
        evidence=deepcopy(self.evidence)
        result=self.assess([(1000,3200,"Clear completion")],evidence)
        self.assertEqual(result["human_evidence"]["human_timed_word_count"],0)
        self.assertFalse(result["ready_for_editorial_word_review"])
        self.assertFalse(result["automatic_word_alignment_performed"])

    def test_binary_file_wrapper_hash_gated_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            audio=root/"src.bin"
            write_pcm(audio)
            self.evidence["source_audio_sha256"] = h(audio.read_bytes())
            candidate=root/"old.srt"
            candidate.write_bytes(srt([(0,500,"Outside"),(3300,3900,"After")]).encode())
            evidence=root/"human.json"
            evidence.write_text(json.dumps(self.evidence),encoding="utf-8")
            result_file=root/"private_audit.json"
            report=write_boundary_assessment(
                candidate,evidence,audio,result_file,
                source_duration_ms=4000,
                expected_candidate_sha256=h(candidate.read_bytes()),
                expected_review_sha256=h(evidence.read_bytes()))
            self.assertEqual(report["candidate_status"],
                             "HUMAN_CONFIRMED_SPEECH_WITHOUT_CAPTION")
            self.assertFalse(report["auto_release_authorized"])
            self.assertEqual(json.loads(result_file.read_text())["human_evidence"]["human_word_count"],2)
            with self.assertRaises(FileExistsError):
                write_boundary_assessment(
                    candidate,evidence,audio,result_file,
                    source_duration_ms=4000,
                    expected_candidate_sha256=h(candidate.read_bytes()),
                    expected_review_sha256=h(evidence.read_bytes()))
            self.assertEqual(h(audio.read_bytes()),self.evidence["source_audio_sha256"])

    def test_wrapper_refuses_inaccurate_source_audio_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio=root/"audio.wav";write_pcm(audio)
            evidence=deepcopy(self.evidence)
            evidence["source_audio_sha256"]=h(audio.read_bytes())
            reviewfile=root/"review.json"
            reviewfile.write_text(json.dumps(evidence),encoding="utf-8")
            candidate=root/"candidate.srt"
            candidate.write_text("",encoding="utf-8")
            target=root/"should_not_exist.json"
            with self.assertRaisesRegex(ValueError, "duration differs"):
                write_boundary_assessment(
                    candidate,reviewfile,audio,target,
                    source_duration_ms=4500,
                    expected_candidate_sha256=h(candidate.read_bytes()),
                    expected_review_sha256=h(reviewfile.read_bytes()),
                )
            self.assertFalse(target.exists())

    def test_wrapper_refuses_wrong_source_pcm_format(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            audio=root/"bad_rate.wav"
            with wave.open(str(audio),"wb") as wav:
                wav.setframerate(8000);wav.setnchannels(1);wav.setsampwidth(2)
                wav.writeframes(b"\x00\x00"*(8000*4))
            evidence=deepcopy(self.evidence)
            evidence["source_audio_sha256"]=h(audio.read_bytes())
            reviewfile=root/"review.json"
            reviewfile.write_text(json.dumps(evidence),encoding="utf-8")
            candidate=root/"candidate.srt"
            candidate.write_text("",encoding="utf-8")
            target=root/"must_not_exist.json"
            with self.assertRaisesRegex(ValueError, "16-kHz"):
                write_boundary_assessment(
                    candidate,reviewfile,audio,target,
                    source_duration_ms=4000,
                    expected_candidate_sha256=h(candidate.read_bytes()),
                    expected_review_sha256=h(reviewfile.read_bytes()),
                )
            self.assertFalse(target.exists())

    def test_wrapper_does_not_emit_when_review_hash_is_wrong(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            audio=root/"src.bin";write_pcm(audio)
            self.evidence["source_audio_sha256"] = h(audio.read_bytes())
            candidate=root/"old.srt";candidate.write_text("",encoding="utf-8")
            evidence=root/"human.json";evidence.write_text(json.dumps(self.evidence))
            output=root/"must_not_exist.json"
            with self.assertRaisesRegex(ValueError,"hash"):
                write_boundary_assessment(
                    candidate,evidence,audio,output,
                    source_duration_ms=4000,
                    expected_candidate_sha256=h(candidate.read_bytes()),
                    expected_review_sha256=h(b"incorrect"))
            self.assertFalse(output.exists())


if __name__=="__main__":
    unittest.main()
