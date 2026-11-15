"""Synthetic safety/clock/provenance tests for multi-region SRT assembly."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.diagnostics import parse_srt
from adaptive_whisper.multi_timeline import (
    OwnershipOverlapError, RecoveryReplacement, RecoveryFile,
    assemble_multi_timeline, write_multi_timeline_preview,
)
from adaptive_whisper.stitch import fmt


def srt(lines):
    return "".join(
        f"{i}\n{fmt(a)} --> {fmt(b)}\n{text}\n\n"
        for i, (a, b, text) in enumerate(lines, 1)
    )


class MultiTimelineTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        self.audio = self.folder / "source.wav"
        with wave.open(str(self.audio), "wb") as w:
            w.setframerate(16000)
            w.setnchannels(1)
            w.setsampwidth(2)
            w.writeframes(b"\x00\x00" * 16000 * 4)
        self.baseline = srt([
            (0, 700, "Historical opening."),
            (1000, 1500, "Historical damaged region one."),
            (1850, 2150, "Historical middle untouched."),
            (2400, 2800, "Historical damaged region two."),
            (3300, 3800, "Historical end."),
        ])
        self.first = RecoveryReplacement(
            1, (1000, 1700), srt([(1100, 1500, "Corrected option one.")])
        )
        self.second = RecoveryReplacement(
            2, (2300, 3000), srt([(2450, 2800, "Corrected option two.")])
        )

    def test_separate_regions_preserve_historical_middle_and_source_cue_hashes(self):
        output, ledger = assemble_multi_timeline(
            self.baseline, (self.second, self.first), original_duration_ms=4000
        )
        cues = parse_srt(output)
        self.assertEqual([c.text for c in cues], [
            "Historical opening.", "Corrected option one.",
            "Historical middle untouched.", "Corrected option two.",
            "Historical end.",
        ])
        self.assertEqual(ledger["status"], "STRUCTURALLY_COMPOSED_UNVERIFIED")
        self.assertEqual(ledger["hazards_count"], 0)
        self.assertEqual(ledger["retained_baseline_cue_count"], 3)
        self.assertEqual(ledger["discarded_baseline_cue_count"], 2)
        self.assertEqual([p["region_index"] for p in ledger["cue_source_provenance"]],
                         [None, 1, None, 2, None])
        self.assertEqual(ledger["cue_source_provenance"][1]["source_text_sha256"],
                         hashlib.sha256(b"Corrected option one.").hexdigest())
        self.assertEqual(ledger["source_window_ownership"][2]["interval_ms"],
                         [1700, 2300])
        self.assertFalse(ledger["human_audio_verification"])
        self.assertFalse(ledger["auto_release_authorized"])

    def test_exactly_touching_source_windows_do_not_imply_safe_caption_join(self):
        left = RecoveryReplacement(
            1, (1000, 2000), srt([(1500, 2300, "First disputed continuation.")])
        )
        right = RecoveryReplacement(
            2, (2000, 3000), srt([(2050, 2500, "Alternative second continuation.")])
        )
        _, ledger = assemble_multi_timeline(
            "", (left, right), original_duration_ms=4000
        )
        codes = {h["code"] for h in ledger["hazards"]}
        self.assertIn("CROSS_REGION_RECOVERY_CAPTION_OVERLAP", codes)
        self.assertEqual(ledger["status"], "PROVISIONAL_REVIEW_REQUIRED")
        self.assertEqual(ledger["recovery_cue_count"], 2)

    def test_overlapping_recovery_ownership_fails_closed(self):
        right = RecoveryReplacement(
            2, (1500, 3000), srt([(2000, 2600, "Competing owner.")])
        )
        with self.assertRaises(OwnershipOverlapError) as ctx:
            assemble_multi_timeline(
                self.baseline, (self.first, right), original_duration_ms=4000
            )
        self.assertEqual(ctx.exception.overlap_ms, (1500, 1700))

    def test_nonadjacent_baseline_hypothesis_overlaps_two_recovered_cues(self):
        old = srt([(0, 1950, "Original phrase spans two alternatives.")])
        first = RecoveryReplacement(
            1, (1000, 1700), srt([
                (1100, 1300, "First recovery hypothesis."),
                (1400, 1600, "Second recovery hypothesis."),
            ])
        )
        _, ledger = assemble_multi_timeline(
            old, (first,), original_duration_ms=4000
        )
        hazards = [h for h in ledger["hazards"]
                   if h["code"] == "CROSS_SOURCE_CAPTION_OVERLAP"]
        self.assertEqual(len(hazards), 2)
        self.assertIn("RETAINED_BASELINE_CUE_CROSSES_REGION_START",
                      {h["code"] for h in ledger["hazards"]})
        self.assertTrue(all(h["overlap_ms"] > 0 for h in hazards))

    def test_native_rounding_reported_and_hard_overhang_rejected(self):
        near = RecoveryReplacement(
            2, (3000, 4000), srt([(3600, 4020, "Native timing roundoff.")])
        )
        _, ledger = assemble_multi_timeline(
            self.baseline, (near,), original_duration_ms=4000
        )
        self.assertIn("RECOVERED_NATIVE_TIMESTAMP_ROUNDING",
                      {x["code"] for x in ledger["hazards"]})
        distant = RecoveryReplacement(
            2, (3000, 4000), srt([(3600, 4300, "Invalid beyond source.")])
        )
        with self.assertRaises(ValueError):
            assemble_multi_timeline(
                self.baseline, (distant,), original_duration_ms=4000
            )

    def test_recovery_with_no_cue_midpoint_in_own_region_rejected(self):
        detached = RecoveryReplacement(
            1, (1000, 1700), srt([(2200, 2500, "Outside region.")])
        )
        with self.assertRaises(ValueError):
            assemble_multi_timeline(
                self.baseline, (detached,), original_duration_ms=4000
            )

    def test_wrapper_hashes_output_and_refuses_collision(self):
        base = self.folder / "base.srt"
        left = self.folder / "left.srt"
        right = self.folder / "right.srt"
        base.write_text(self.baseline, encoding="utf-8")
        left.write_text(self.first.source_srt, encoding="utf-8")
        right.write_text(self.second.source_srt, encoding="utf-8")
        recoveries = (
            RecoveryFile(1, self.first.window_ms, left),
            RecoveryFile(2, self.second.window_ms, right),
        )
        dest = self.folder / "combined.srt"
        ledger_path = self.folder / "combined.json"
        audit = write_multi_timeline_preview(
            base, recoveries, dest, ledger_path, source_audio=self.audio
        )
        self.assertTrue(dest.exists())
        self.assertEqual(audit["input_file_sha256"]["baseline_srt"],
                         hashlib.sha256(base.read_bytes()).hexdigest())
        self.assertEqual(len(audit["acoustic_energy_proxy_near_boundaries"]), 12)
        self.assertEqual(json.loads(ledger_path.read_text(encoding="utf-8"))["output_srt_sha256"],
                         hashlib.sha256(dest.read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            write_multi_timeline_preview(
                base, recoveries, dest, ledger_path, source_audio=self.audio
            )

    def test_inherited_window_join_conflict_is_reported_in_standalone_manifest(self):
        base = self.folder / "conflict_base.srt"
        left = self.folder / "conflict_recovery.srt"
        base.write_text(self.baseline, encoding="utf-8")
        left.write_text(self.first.source_srt, encoding="utf-8")
        result = write_multi_timeline_preview(
            base, (RecoveryFile(1, self.first.window_ms, left),),
            self.folder / "conflicted.srt", self.folder / "conflicted.json",
            source_audio=self.audio, upstream_window_join_conflicts={1: 2},
        )
        self.assertIn("UPSTREAM_WINDOW_JOIN_CONFLICTS",
                      {h["code"] for h in result["hazards"]})
        self.assertEqual(result["status"], "PROVISIONAL_REVIEW_REQUIRED")
        self.assertEqual(result["upstream_window_join_conflicts"],
                         [{"region_index": 1, "count": 2}])
        self.assertFalse(result["auto_release_authorized"])

    def test_invalid_join_metadata_rejected_before_creating_preview(self):
        base = self.folder / "bad_metadata_base.srt"
        left = self.folder / "bad_metadata_recovery.srt"
        base.write_text(self.baseline, encoding="utf-8")
        left.write_text(self.first.source_srt, encoding="utf-8")
        output = self.folder / "should_not_exist.srt"
        ledger = self.folder / "should_not_exist.json"
        for metadata in ({1: -1}, {42: 1}):
            with self.assertRaises(ValueError):
                write_multi_timeline_preview(
                    base, (RecoveryFile(1, self.first.window_ms, left),),
                    output, ledger, source_audio=self.audio,
                    upstream_window_join_conflicts=metadata,
                )
        self.assertFalse(output.exists())
        self.assertFalse(ledger.exists())

    def test_wrapper_blocks_overlapping_windows_without_creating_outputs(self):
        base = self.folder / "base.srt"
        left = self.folder / "left.srt"
        right = self.folder / "right.srt"
        base.write_text(self.baseline, encoding="utf-8")
        left.write_text(self.first.source_srt, encoding="utf-8")
        right.write_text(self.second.source_srt, encoding="utf-8")
        outputs = (self.folder / "never.srt", self.folder / "never.json")
        with self.assertRaises(OwnershipOverlapError):
            write_multi_timeline_preview(
                base, (RecoveryFile(1, (1000, 1700), left),
                       RecoveryFile(2, (1500, 3000), right)),
                *outputs, source_audio=self.audio
            )
        self.assertFalse(outputs[0].exists())
        self.assertFalse(outputs[1].exists())


if __name__ == "__main__":
    unittest.main()
