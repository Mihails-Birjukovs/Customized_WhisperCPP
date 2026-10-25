"""Phase 0 public corpus registry schema/integrity checks; no network or media required."""
import json
import pathlib
import re
import unittest

MANIFEST = pathlib.Path(__file__).resolve().parents[2] / "benchmarks" / "phase0_corpus_manifest.json"

class CorpusManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.records = cls.data["records"]

    def test_inventory_and_locators(self):
        self.assertEqual(len(self.records), 40)
        self.assertEqual(len({r["record_id"] for r in self.records}), 40)
        self.assertEqual(sum(r["group"] == "meeting" for r in self.records), 35)
        self.assertEqual(sum(r["group"] == "lecture" for r in self.records), 5)
        self.assertEqual(sum("suspected_repetition" in r for r in self.records), 8)
        self.assertEqual(sum("encoding_issue" in r for r in self.records), 1)
        for r in self.records:
            self.assertEqual(r["baseline_output_status"], "present_not_reproduced")
            self.assertEqual(r["transcript_reference_status"], "not_human_verified")
            self.assertEqual(set(r["source_assets"]), {"audio", "legacy_txt", "legacy_srt", "legacy_config"})
            for asset in r["source_assets"].values():
                self.assertIn(asset["folder_key"], self.data["folder_keys"])
                self.assertGreater(asset["bytes"], 0)
                self.assertFalse(asset["filename_glob"].startswith("http"))
                if r["group"] == "meeting":
                    self.assertIn(r["recording_date"], asset["filename_glob"])
                else:
                    self.assertIn(str(r["lecture_index"]), asset["filename_glob"])

    def test_srt_repetition_intervals(self):
        def seconds(s):
            m = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})", s)
            self.assertIsNotNone(m)
            return 3600*int(m[1])+60*int(m[2])+int(m[3])+int(m[4])/1000
        for r in self.records:
            if "suspected_repetition" in r:
                run = r["suspected_repetition"]
                self.assertEqual(run["source"], "legacy SRT")
                self.assertGreaterEqual(run["segments"], 3)
                self.assertGreater(seconds(run["end"]), seconds(run["start"]))

    def test_no_sensitive_payloads_or_links(self):
        text = MANIFEST.read_text(encoding="utf-8")
        self.assertNotIn("drive.google.com", text)
        self.assertNotIn("file/d/", text)
        self.assertNotIn("Didzis-Rihards", text)

if __name__ == "__main__":
    unittest.main()
