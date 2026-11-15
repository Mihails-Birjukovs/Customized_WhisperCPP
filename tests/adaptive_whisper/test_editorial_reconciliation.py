"""Deterministic, non-acoustic lexical reconciliation of disputed words."""
import hashlib
import sys
from pathlib import Path
from copy import deepcopy
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.editorial_reconciliation import (
    semiglobal_lexical_alignment, reconciliation_evidence,
    tokens_with_cue_provenance, write_unapproved_editorial_worklist
)
from adaptive_whisper.stitch import fmt


def srt(cues):
    return "".join(f"{i}\n{fmt(a)} --> {fmt(b)}\n{text}\n\n"
                   for i,(a,b,text) in enumerate(cues,1))


def ref():
    def word(ix,text,heard=True,fragment=None):
        return {
            "index":ix, "transcribed_word":text,
            "audibility":"fully_heard" if heard else "partial_context_completed",
            "heard_fragment":fragment,
            "word_start_ms":None, "word_end_ms":None,
            "human_timing_verified":False,
        }
    return {
        "schema":"whispercpp/human-audio-boundary-review/v1",
        "start_ms":1000, "end_ms":3500,
        "reviewer_count":1, "speech_confirmed_by_human":True,
        "source_audio_sha256":hashlib.sha256(b"source").hexdigest(),
        "locked_reference_sha256":hashlib.sha256(b"human").hexdigest(),
        "words":[word(1,"Human"),word(2,"audio"),word(3,"words"),
                 word(4,"corrected",False,"corr")]
    }


class ReconciliationTests(unittest.TestCase):
    def test_empty_machine_means_missing_human_words(self):
        result=reconciliation_evidence(ref(),"",original_duration_ms=4000)
        self.assertEqual(result["machine_tokens_from_intersecting_cues"],0)
        self.assertEqual(result["lexical_alignment"]["minimum_sequence_edit_cost"],3)
        self.assertTrue(all(row["lexical_relation"]=="human_word_not_in_machine"
                            for row in result["word_review_rows"][:3]))
        self.assertEqual(result["word_review_rows"][3]["lexical_relation"],
                         "context_completed_partial_word_unscored")
        self.assertFalse(result["automatic_transcript_release_authorized"])

    def test_machine_crossing_cues_not_word_timing(self):
        transcript=srt([(100,1900,"Before Human audio"),(2000,3700,"words more after")])
        result=reconciliation_evidence(ref(),transcript,original_duration_ms=4000)
        self.assertEqual(result["fully_heard_reference_words_compared"],3)
        self.assertTrue(result["machine_cue_boundary_crossings_not_resolved"])
        self.assertEqual([r["lexical_relation"] for r in result["word_review_rows"][:3]],
                         ["exact","exact","exact"])
        self.assertTrue(result["word_review_rows"][0]["machine_source_cue_crosses_review_interval"])
        self.assertFalse(result["word_review_rows"][0]["acoustic_correctness_certified"])
        self.assertFalse(result["human_timestamps_complete"])
        self.assertTrue(result["not_validated_WER"])

    def test_semiglobal_skips_unknown_machine_lead_and_tail(self):
        align=semiglobal_lexical_alignment(
            ["brown","fox"],["a","quick","brown","fox","ran"]
        )
        self.assertEqual(align["minimum_sequence_edit_cost"],0)
        self.assertEqual(align["candidate_subsequence_start_index_zero_based"],2)
        self.assertEqual(align["candidate_subsequence_end_index_exclusive"],4)
        self.assertEqual([x["machine_index"] for x in align["assignments"]],[3,4])
        self.assertTrue(align["free_candidate_prefix_and_suffix"])
        self.assertTrue(align["not_acoustic_alignment"])

    def test_mismatch_does_not_get_certified(self):
        align=semiglobal_lexical_alignment(["one","two"],["one","wrong"])
        self.assertEqual(align["minimum_sequence_edit_cost"],1)
        # With free candidate suffix, a 1-edit omission and substitution
        # are equally cheap. Either is an unresolved human review, not exact.
        self.assertEqual(align["assignments"][0]["relation"],"exact")
        self.assertIn(align["assignments"][1]["relation"],
                      ("substituted","human_word_not_in_machine"))

    def test_repeated_machine_word_tie_is_deterministic(self):
        first=semiglobal_lexical_alignment(["hello"],["hello","hello"])
        second=semiglobal_lexical_alignment(["hello"],["hello","hello"])
        self.assertEqual(first,second)
        self.assertEqual(first["assignments"][0]["machine_index"],1)

    def test_human_diacritics_preserved_and_casefolded(self):
        evidence=ref()
        evidence["words"][0]["transcribed_word"]="Rīga"
        x=reconciliation_evidence(
            evidence,srt([(1000,3500,"RĪGA audio words")]),
            original_duration_ms=4000
        )
        self.assertEqual(x["lexical_alignment"]["minimum_sequence_edit_cost"],0)

    def test_diacritic_loss_is_not_silently_normalized(self):
        evidence=ref()
        evidence["words"][0]["transcribed_word"]="Rīga"
        x=reconciliation_evidence(
            evidence,srt([(1000,3500,"Riga audio words")]),
            original_duration_ms=4000
        )
        self.assertEqual(x["lexical_alignment"]["minimum_sequence_edit_cost"],1)

    def test_partial_word_cannot_claim_exact_match(self):
        x=reconciliation_evidence(
            ref(),srt([(1000,3500,"Human audio words corrected")]),
            original_duration_ms=4000
        )
        self.assertEqual(x["partially_heard_context_completed_words_excluded"],1)
        self.assertEqual(x["word_review_rows"][-1]["lexical_relation"],
                         "context_completed_partial_word_unscored")
        self.assertFalse(x["automatic_transcript_release_authorized"])

    def test_empty_reference_rejected(self):
        with self.assertRaises(ValueError):
            semiglobal_lexical_alignment([],["foo"])

    def test_machine_token_budget_and_malformed_srt_refused(self):
        transcript=srt([(1000,3500,"one two three four")])
        with self.assertRaisesRegex(ValueError,"Too many"):
            tokens_with_cue_provenance(transcript,1000,3500,max_tokens=2)
        with self.assertRaisesRegex(ValueError,"Malformed"):
            tokens_with_cue_provenance("not valid SRT",1000,3500)

    def test_invalid_review_source_time_rejected(self):
        evidence=ref();evidence["end_ms"]=9000
        with self.assertRaises(ValueError):
            reconciliation_evidence(
                evidence,srt([(0,1000,"Some cue")]),original_duration_ms=4000)

    def test_report_does_not_contain_any_human_or_machine_plain_words(self):
        x=reconciliation_evidence(
            ref(),srt([(1000,3500,"Human audio words bizarre")]),
            original_duration_ms=4000
        )
        self.assertNotIn("bizarre",str(x))
        self.assertNotIn("corrected",str(x))
        self.assertTrue(x["no_private_text_in_report"])


    def test_private_editorial_worklist_is_unapproved_non_destructive(self):
        import csv
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            human=ref()
            reports={
                "original":reconciliation_evidence(
                    human,"",original_duration_ms=4000),
                "extended_tail":reconciliation_evidence(
                    human,srt([(800,3800,"Human audio words")]),
                    original_duration_ms=4000),
            }
            dest=Path(folder)/"editorial_private.csv"
            summary=write_unapproved_editorial_worklist(
                human,reports,dest,original_duration_ms=4000)
            self.assertEqual(summary["editorial_choices_completed"],0)
            self.assertFalse(summary["auto_release_authorized"])
            with dest.open("r",encoding="utf-8-sig",newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),4)
            self.assertEqual(rows[0]["human_reference_word"],"Human")
            self.assertEqual(rows[0]["original_lexical_relation"],
                             "human_word_not_in_machine")
            self.assertEqual(rows[3]["human_audibility"],
                             "partial_context_completed")
            self.assertEqual(rows[0]["editorial_choice_UNFILLED"],"")
            with self.assertRaises(FileExistsError):
                write_unapproved_editorial_worklist(
                    human,reports,dest,original_duration_ms=4000)

    def test_editorial_worklist_rejects_different_human_text(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            human=ref()
            result=reconciliation_evidence(
                human,srt([(1000,3500,"Human audio words")]),
                original_duration_ms=4000
            )
            wrong=deepcopy(result)
            wrong["word_review_rows"][0]["human_word_sha256"]="a"*64
            path=Path(folder)/"must_not_exist.csv"
            with self.assertRaisesRegex(ValueError,"different human token"):
                write_unapproved_editorial_worklist(
                    human,{"original":result,"extended_tail":wrong},path,
                    original_duration_ms=4000)
            self.assertFalse(path.exists())

    def test_editorial_worklist_rejects_incorrect_source_interval(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            human=ref()
            result=reconciliation_evidence(
                human,srt([(1000,3500,"Human audio words")]),
                original_duration_ms=4000)
            wrong=deepcopy(result)
            wrong["interval_ms"]=[1100,3500]
            path=Path(folder)/"must_not_exist.csv"
            with self.assertRaisesRegex(ValueError,"source interval"):
                write_unapproved_editorial_worklist(
                    human,{"original":result,"extended_tail":wrong},path,
                    original_duration_ms=4000)
            self.assertFalse(path.exists())


if __name__=="__main__":
    unittest.main()
