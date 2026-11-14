"""Synthetic tests for human-timed but unapproved transcript fragments."""
from copy import deepcopy
import hashlib
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.editorial_fragment import build_review_only_fragment
from adaptive_whisper.diagnostics import parse_srt
from adaptive_whisper.stitch import fmt


def srt(items):
    return "".join(f"{i}\n{fmt(a)} --> {fmt(b)}\n{t}\n\n"
                   for i,(a,b,t) in enumerate(items,1))


def evidence():
    make=lambda i,text,a,b,audibility,fragment=None: {
        "index":i,"transcribed_word":text,"audibility":audibility,
        "heard_fragment":fragment,"word_start_ms":a,"word_end_ms":b,
        "human_timing_verified":True,
    }
    return {
        "schema":"whispercpp/human-audio-boundary-review/v1",
        "start_ms":1000,"end_ms":3000,
        "speech_confirmed_by_human":True,
        "reviewer_count":1,
        "source_audio_sha256":hashlib.sha256(b"fake original").hexdigest(),
        "locked_reference_sha256":hashlib.sha256(b"fake human").hexdigest(),
        "words":[
            make(1,"cilvēks",1200,1500,"fully_heard"),
            make(2,"tur",1520,1790,"fully_heard"),
            make(3,"optimizēt",1850,3000,"partial_context_completed","opti"),
        ],
    }


def candidates():
    return {
        "original":srt([(100,1000,"outside left"),(3000,3700,"outside right")]),
        "extended":srt([(0,1600,"untrusted left"),
                        (1600,1700,"untrusted middle"),
                        (1700,3100,"untrusted right")]),
        "historical":srt([(400,3000,"long prior machine hypothesis")])
    }


class EditorialFragmentTests(unittest.TestCase):
    def build(self,review=None,alts=None):
        return build_review_only_fragment(
            review if review is not None else evidence(),
            alts if alts is not None else candidates(),
            original_duration_ms=5000
        )

    def test_only_fully_heard_words_appear_in_review_srt(self):
        r=self.build()
        cues=parse_srt(r.word_srt)
        self.assertEqual([(x.start_ms,x.end_ms,x.text) for x in cues],
                         [(1200,1500,"cilvēks"),(1520,1790,"tur")])
        self.assertNotIn("optimizēt",r.word_srt)
        self.assertEqual(r.review["fully_heard_word_cues"],2)
        self.assertEqual(r.review["partial_context_completed_words_excluded"],1)
        self.assertFalse(r.review["publication_ready"])
        self.assertFalse(r.review["auto_release_authorized"])

    def test_partial_final_token_stays_right_censored_in_sidecar(self):
        r=self.build()
        p=r.review["partially_heard_word_evidence"][0]
        self.assertEqual(p["audible_fragment"],"opti")
        self.assertEqual(p["context_completed_word"],"optimizēt")
        self.assertTrue(p["right_censored_by_review_clip"])
        self.assertFalse(p["entire_word_acoustically_confirmed"])

    def test_competing_machine_cue_boundary_splits_known_word(self):
        r=self.build()
        word=r.review["human_word_to_machine_cue_ownership"][1]
        ext=word["candidate_cue_intersections"]["extended"]
        self.assertTrue(ext["machine_cues_split_human_word"])
        self.assertEqual(ext["cue_boundaries_inside_human_word_ms"],[1600,1700])
        self.assertFalse(word["editorial_word_ownership_confirmed"])

    def test_machine_srt_sha_is_of_each_machine_not_preview(self):
        r=self.build()
        for label,raw in candidates().items():
            self.assertEqual(
                r.review["source_hypotheses"][label]["candidate_srt_sha256"],
                hashlib.sha256(raw.encode()).hexdigest()
            )
        self.assertNotEqual(
            r.review["source_hypotheses"]["extended"]["candidate_srt_sha256"],
            r.review["fully_heard_word_preview_srt_sha256"])

    def test_baseline_gap_preserves_precise_neighbor_positions(self):
        r=self.build()
        old=r.review["source_hypotheses"]["original"]
        self.assertEqual(old["intersecting_review_cues"],[])
        self.assertEqual(old["nearest_preceding_cue_end_ms"],1000)
        self.assertEqual(old["nearest_following_cue_start_ms"],3000)

    def test_unicode_source_word_not_normalized_or_modified(self):
        e=evidence()
        e["words"][0]["transcribed_word"]="īstenībā"
        r=self.build(review=e)
        self.assertIn("īstenībā",r.word_srt)
        self.assertNotIn("isteniba",r.word_srt)

    def test_incomplete_fully_heard_word_time_is_rejected(self):
        e=evidence()
        e["words"][0]["word_start_ms"]=None
        e["words"][0]["human_timing_verified"]=False
        with self.assertRaisesRegex(ValueError,"Missing paired word times|missing human timing"):
            self.build(review=e)

    def test_unverified_timing_cannot_generate_cue(self):
        e=evidence()
        e["words"][0]["human_timing_verified"]=False
        with self.assertRaisesRegex(ValueError,"not confirmed by human|human-verified"):
            self.build(review=e)

    def test_time_overlap_between_human_words_is_rejected(self):
        e=evidence()
        e["words"][1]["word_start_ms"]=1450
        with self.assertRaisesRegex(ValueError,"Overlapping"):
            self.build(review=e)

    def test_multiple_machine_sources_are_immutable(self):
        given=candidates()
        before=deepcopy(given)
        e=evidence()
        self.build(review=e,alts=given)
        self.assertEqual(given,before)
        self.assertEqual(e,evidence())

    def test_malformed_nonempty_machine_srt_cannot_masquerade_as_gap(self):
        c=candidates();c["original"]="this is not SRT"
        with self.assertRaisesRegex(ValueError,"invalid candidate SRT"):
            self.build(alts=c)

    def test_candidate_clock_bounds_must_fit_original_source(self):
        c=candidates();c["extended"]=srt([(0,6000,"invalid")])
        with self.assertRaisesRegex(ValueError,"original source clock"):
            self.build(alts=c)

    def test_machine_text_cannot_change_human_word_cues(self):
        c=candidates();c["extended"]=srt([(1000,3000,"completely wrong")])
        a=self.build(alts=c)
        b=self.build()
        self.assertEqual(a.word_srt,b.word_srt)
        self.assertNotIn("completely wrong",a.word_srt)
        self.assertFalse(a.review["machine_caption_texts_not_used_to_create_review_words"] is False)

    def test_all_partial_words_cannot_build_fully_audible_preview(self):
        e=evidence()
        for w in e["words"]:
            w["audibility"]="partial_context_completed"
            w["heard_fragment"]=w["transcribed_word"][:2]
        with self.assertRaisesRegex(ValueError,"No fully audible"):
            self.build(review=e)


if __name__=="__main__":
    unittest.main()
