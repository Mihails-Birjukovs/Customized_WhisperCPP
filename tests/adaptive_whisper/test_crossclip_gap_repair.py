"""Synthetic regression gates for cross-clip human-corroborated gap previews."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.crossclip_gap_repair import build_crossclip_gap_preview
from adaptive_whisper.diagnostics import parse_srt
from adaptive_whisper.stitch import fmt


def srt(rows):
    return "".join(f"{i}\n{fmt(a)} --> {fmt(b)}\n{t}\n\n"
                   for i,(a,b,t) in enumerate(rows,1))


def evidence():
    def w(i,text,a,b,kind="fully_heard",frag=None):
        return dict(index=i,transcribed_word=text,
                    word_start_ms=a,word_end_ms=b,
                    audibility=kind,heard_fragment=frag,human_timing_verified=True)
    return {
        "schema":"whispercpp/human-audio-boundary-review/v1",
        "speech_confirmed_by_human":True, "reviewer_count":1,
        "source_audio_sha256":hashlib.sha256(b"synthetic source").hexdigest(),
        "locked_reference_sha256":hashlib.sha256(b"synthetic reference").hexdigest(),
        "start_ms":1000,"end_ms":4000,
        "words":[w(1,"Viņš",1200,1640),w(2,"nevar",1800,2250),
                 w(3,"optimizēt",2450,4000,"partial_context_completed","opti")],
    }


def corroboration():
    sha=hashlib.sha256(b"frozen human CSV").hexdigest()
    return [
        dict(clip_id="LONG01",source_clock_ms=[500,5500],
             human_transcript="Viņš nevar optimizēt",audio_first_review=True,
             full_word_audibly_heard=True,locked_csv_sha256=sha),
        dict(clip_id="LONG02",source_clock_ms=[2000,4900],
             human_transcript="nevar optimizēt; vēl kas",audio_first_review=True,
             full_word_audibly_heard=True,locked_csv_sha256=sha)
    ]


def original():
    return srt([(100,1000,"No edits before"),(4000,4900,"No edits after"),
                (5500,6300,"Other content intact")])


class CorroboratedGapTests(unittest.TestCase):
    def build(self,base=None,words=None,refs=None):
        return build_crossclip_gap_preview(
            original() if base is None else base,
            evidence() if words is None else words,
            corroboration() if refs is None else refs,
            original_duration_ms=9000,expected_empty_gap_ms=(1000,4000),
        )

    def test_complete_word_from_context_is_used_as_provisional_identity(self):
        result=self.build()
        cues=parse_srt(result.proposed_full_srt)
        self.assertEqual([(x.start_ms,x.end_ms,x.text) for x in cues],
                         [(100,1000,"No edits before"),
                          (1200,4000,"Viņš nevar optimizēt"),
                          (4000,4900,"No edits after"),
                          (5500,6300,"Other content intact")])
        audit=result.amendment_ledger
        self.assertEqual(audit["preview_caption_count"],4)
        self.assertEqual(audit["fully_audible_within_short_clip"],2)
        self.assertTrue(audit["final_word_acoustic_end_ms_unmeasured"])
        self.assertTrue(audit["final_word_last_short_clip_timestamp_is_right_censored"])
        self.assertFalse(audit["auto_release_authorized"])

    def test_corroboration_requires_two_longer_audio_first_references(self):
        for refs in [corroboration()[:1], []]:
            with self.subTest(refs=refs),self.assertRaisesRegex(ValueError,"At least two"):
                self.build(refs=refs)
        refs=corroboration()
        refs[0]["audio_first_review"]=False
        with self.assertRaisesRegex(ValueError,"audio-first"):
            self.build(refs=refs)

    def test_sources_must_contain_complete_word_in_human_text(self):
        refs=corroboration()
        refs[1]["human_transcript"]="only opti fragment"
        with self.assertRaisesRegex(ValueError,"does not contain"):
            self.build(refs=refs)

    def test_context_clip_must_cover_end_of_short_clip(self):
        refs=corroboration()
        refs[0]["source_clock_ms"]=[500,3999]
        with self.assertRaisesRegex(ValueError,"cover source word"):
            self.build(refs=refs)

    def test_same_clip_cannot_be_counted_twice(self):
        refs=corroboration()
        refs[1]["clip_id"]="LONG01"
        with self.assertRaisesRegex(ValueError,"duplicated"):
            self.build(refs=refs)

    def test_right_censored_word_not_assumed_fully_timed(self):
        audit=self.build().amendment_ledger
        self.assertTrue(audit["final_word_right_edge_is_a_display_cutoff_not_full_word_end"])
        self.assertFalse(audit["complete_full_timeline_transcript_approved"])
        self.assertFalse(audit["automatic_word_end_inference"])

    def test_cannot_replace_nonempty_machine_caption(self):
        with self.assertRaisesRegex(ValueError,"Refusing"):
            self.build(base=srt([(500,1600,"Existing machine says something"),
                                 (4000,4900,"After")]))

    def test_requires_source_interval_identity(self):
        with self.assertRaisesRegex(ValueError,"Gap source clock"):
            build_crossclip_gap_preview(original(),evidence(),corroboration(),
                                        original_duration_ms=9000,
                                        expected_empty_gap_ms=(1300,4000))

    def test_rejects_unverified_word_times(self):
        e=evidence()
        e["words"][0]["human_timing_verified"]=False
        with self.assertRaisesRegex(ValueError,"not confirmed by human"):
            self.build(words=e)

    def test_rejects_word_time_overlap(self):
        e=evidence()
        e["words"][1]["word_start_ms"]=1500
        with self.assertRaisesRegex(ValueError,"Overlapping"):
            self.build(words=e)

    def test_requires_final_partial_not_other_partial(self):
        e=evidence()
        e["words"][0].update(audibility="partial_context_completed",heard_fragment="Vi")
        with self.assertRaisesRegex(ValueError,"one censored final"):
            self.build(words=e)

    def test_final_right_clock_must_touch_clip_boundary(self):
        e=evidence()
        e["words"][2]["word_end_ms"]=3900
        with self.assertRaisesRegex(ValueError,"right-censored"):
            self.build(words=e)

    def test_longer_word_reference_has_sha_and_human_audio_attestation(self):
        refs=corroboration()
        refs[0]["locked_csv_sha256"]="invalid"
        with self.assertRaisesRegex(ValueError,"reference hash"):
            self.build(refs=refs)
        refs=corroboration()
        refs[0]["full_word_audibly_heard"]=False
        with self.assertRaisesRegex(ValueError,"not attested"):
            self.build(refs=refs)

    def test_diacritics_must_match_word_not_ascii_variant(self):
        refs=corroboration()
        refs[0]["human_transcript"]="vins nevar optimizet"
        with self.assertRaisesRegex(ValueError,"does not contain"):
            self.build(refs=refs)

    def test_base_order_and_original_text_unchanged(self):
        source=original()
        result=self.build()
        old=[(x.start_ms,x.end_ms,x.text) for x in parse_srt(source)]
        new=[(x.start_ms,x.end_ms,x.text) for x in parse_srt(result.proposed_full_srt)]
        self.assertEqual([t for t in new if t not in old],
                         [(1200,4000,"Viņš nevar optimizēt")])
        self.assertEqual([t for t in new if t in old],old)
        self.assertEqual(result.amendment_ledger["base_caption_count"],3)

    def test_malformed_SRT_cannot_be_treated_as_gap(self):
        with self.assertRaisesRegex(ValueError,"Invalid source SRT"):
            self.build(base="Not SRT")

    def test_no_private_data_or_asr_default_changes(self):
        result=self.build()
        ledger=result.amendment_ledger
        self.assertNotIn("Viņš",str(ledger))
        self.assertTrue(ledger["machine_caption_text_used_to_restore_phrase"] is False)
        self.assertFalse(ledger["auto_release_authorized"])


if __name__=="__main__":
    unittest.main()
