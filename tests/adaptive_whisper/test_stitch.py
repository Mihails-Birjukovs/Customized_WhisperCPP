"""Synthetic boundary safety tests: never use private transcript text."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.recovery import RetryWindow
from adaptive_whisper.stitch import merge_windows,fmt
from adaptive_whisper.diagnostics import parse_srt

def cues(*triples):
    return "".join(f"{i}\n{fmt(a)} --> {fmt(b)}\n{text}\n\n"
                   for i,(a,b,text) in enumerate(triples,1))

W1=RetryWindow(1,100000,160000)
W2=RetryWindow(2,157000,217000)

class StitchSafetyTests(unittest.TestCase):
    def test_absolute_timestamps_and_order(self):
        a=cues((1000,2500,"First statement."),(54000,55000,"Prior boundary line."))
        b=cues((4500,6000,"Next statement."))
        result=merge_windows((W1,W2),(a,b))
        self.assertEqual([x.start_ms for x in result.segments],[101000,154000,161500])
        self.assertEqual(len(parse_srt(result.text_srt)),3)
        self.assertEqual(result.status,"STRUCTURALLY_STITCHED_UNVERIFIED")
        self.assertIn("00:01:41,000",result.text_srt)

    def test_duplicates_around_cut_choose_one_owner(self):
        a=cues((56700,59000,"Unique shared text."))  # global 156700..159000
        b=cues((0,2000,"Unique shared text."),(5000,6100,"Next line.")) # global 157000..159000
        r=merge_windows((W1,W2),(a,b))
        self.assertEqual(sum("Unique shared" in x.text for x in r.segments),1)
        self.assertLessEqual(len(r.segments),2)

    def test_divergent_boundary_hypotheses_require_review(self):
        a=cues((57200,59300,"A long first language sentence here."))
        b=cues((300,2500,"A very different sentence about another topic."))
        r=merge_windows((W1,W2),(a,b))
        self.assertEqual(r.status,"PROVISIONAL_REVIEW_REQUIRED")
        self.assertIn("DIVERGENT_BOUNDARY_HYPOTHESES",
                      [x["kind"] for x in r.conflicts])

    def test_unilateral_boundary_speech_requires_review(self):
        a=cues((57200,59500,"Only the old window hears this."))
        b=cues((7500,9000,"Speech later in the next window."))
        r=merge_windows((W1,W2),(a,b))
        self.assertEqual(r.status,"PROVISIONAL_REVIEW_REQUIRED")
        self.assertIn("UNILATERAL_SPEECH_AT_BOUNDARY",
                      [x["kind"] for x in r.conflicts])

    def test_rejects_gap_in_audio_coverage(self):
        b=RetryWindow(2,160001,220001)
        with self.assertRaises(ValueError):
            merge_windows((W1,b),("", ""))

    def test_rejects_native_invalid_or_out_of_window_timing(self):
        with self.assertRaises(ValueError):
            merge_windows((W1,W2),(cues((5000,4000,"Negative interval.")), ""))
        with self.assertRaises(ValueError):
            merge_windows((W1,W2),(cues((60500,62000,"Too far past window.")), ""))

    def test_joint_caption_gap_avoids_divergent_midpoint(self):
        left=RetryWindow(1,0,30000)
        right=RetryWindow(2,10000,40000)
        # Different captions span the nominal midpoint at 20s.
        a=cues((17000,23000,"The left backend suggests one sentence."))
        b=cues((7000,13500,"The right backend suggests different words."))
        at_mid=merge_windows((left,right),(a,b))
        self.assertEqual(at_mid.status,"PROVISIONAL_REVIEW_REQUIRED")
        fixed=merge_windows((left,right),(a,b),cut_strategy="joint_caption_gap")
        self.assertEqual(fixed.boundaries[0]["cut_method"],"joint_caption_gap")
        self.assertTrue(fixed.boundaries[0]["joint_caption_gap_ms"]>=800)
        self.assertEqual(fixed.status,"STRUCTURALLY_STITCHED_UNVERIFIED")

    def test_joint_gap_fallback_does_not_hide_conflict(self):
        left=RetryWindow(1,0,30000)
        right=RetryWindow(2,10000,40000)
        a=cues((10000,30000,"Entire overlap described in the left view."))
        b=cues((0,20000,"Entire overlap described very differently."))
        r=merge_windows((left,right),(a,b),cut_strategy="joint_caption_gap")
        self.assertEqual(r.boundaries[0]["cut_method"],"fallback_midpoint_NO_JOINT_GAP")
        self.assertEqual(r.status,"PROVISIONAL_REVIEW_REQUIRED")

    def test_does_not_rewrite_source_text(self):
        txt="Technical letters: α + β = γ."
        a=cues((1000,2500,txt))
        r=merge_windows((W1,W2),(a,""))
        self.assertEqual(r.segments[0].text,txt)
        self.assertEqual(len(r.segments),1)
        self.assertEqual(r.segments[0].source_job,1)

if __name__=="__main__":
    unittest.main()
