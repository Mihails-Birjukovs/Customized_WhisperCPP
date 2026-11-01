"""Read-only Phase 0 integrity checks. python -m unittest discover -s tests/phase0 -v"""
import hashlib
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]

class LegacyIntegrityTests(unittest.TestCase):
    def test_sha256(self):
        expected = json.loads((ROOT / "legacy/SHA256SUMS.json").read_text(encoding="utf-8"))
        for relative, digest in expected.items():
            with self.subTest(file=relative):
                path = ROOT / relative
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_baseline_is_documented_and_safe(self):
        data = json.loads((ROOT / "configs/baseline_2026-10-17.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "documented-not-reproduced")
        self.assertFalse(data["disable_fallback"])
        self.assertEqual(data["max_context"], 128)
        self.assertEqual(data["vad"]["padding_ms_by_backend"]["cuda"], 50)
        self.assertEqual(data["vad"]["padding_ms_by_backend"]["cpu"], 30)

if __name__ == "__main__":
    unittest.main()
