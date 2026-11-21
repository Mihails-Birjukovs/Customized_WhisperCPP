"""Synthetic token byte-split tests; no private transcription text."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.native_json import (
    NativeJsonIntegrityError,prepare_lossless_canonical,write_canonical_separate
)

def sample():
    # Valid UTF-8 segment text; two invalid individual token fragments.
    # C4 B7 is a valid two-byte UTF-8 code point only when concatenated.
    return (b'{"transcription":[{"text":"\xc4\xb7",'
            b'"tokens":[{"id":1,"text":"\xc4"},{"id":2,"text":"\xb7"}]}]}')

class TokenFragmentTests(unittest.TestCase):
    def test_lossless_fragments_and_segment_text(self):
        data=sample()
        canonical,audit=prepare_lossless_canonical(data)
        self.assertEqual(audit["converted_token_fields"],2)
        self.assertEqual(audit["affected_segments"],1)
        self.assertEqual(canonical["transcription"][0]["text"],data.decode("utf-8",errors="surrogateescape").split('text":"')[1].split('"')[0])
        toks=canonical["transcription"][0]["tokens"]
        self.assertEqual(toks[0]["text_raw_bytes_hex"],"c4")
        self.assertEqual(toks[1]["text_raw_bytes_hex"],"b7")
        self.assertIsNone(toks[0]["text"])
        self.assertIsNone(toks[1]["text"])
        encoded=json.dumps(canonical,ensure_ascii=False).encode("utf-8","strict")
        self.assertEqual(json.loads(encoded.decode("utf-8")),canonical)

    def test_unrelated_invalid_bytes_rejected(self):
        data=b'{"other":"\xff","transcription":[{"text":"ok","tokens":[{"text":"ok"}]}]}'
        with self.assertRaises(NativeJsonIntegrityError):
            prepare_lossless_canonical(data)

    def test_unresolved_invalid_join_rejected(self):
        data=b'{"transcription":[{"text":"ok","tokens":[{"text":"\xc4"}]}]}'
        with self.assertRaises(NativeJsonIntegrityError):
            prepare_lossless_canonical(data)

    def test_segment_text_invalid_rejected(self):
        data=b'{"transcription":[{"text":"\xff","tokens":[{"text":"ok"}]}]}'
        with self.assertRaises(NativeJsonIntegrityError):
            prepare_lossless_canonical(data)

    def test_new_artifacts_no_overwrite_and_original_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            original=p/"native.json"
            original.write_bytes(sample())
            canonical=p/"derived.json"
            audit=p/"audit.json"
            before=original.read_bytes()
            report=write_canonical_separate(original,canonical,audit)
            self.assertEqual(original.read_bytes(),before)
            self.assertTrue(report["canonical_sha256"])
            obj=json.loads(canonical.read_text(encoding="utf-8"))
            self.assertEqual(obj["_whispercpp_derived_provenance"]["converted_token_fields"],2)
            with self.assertRaises(FileExistsError):
                write_canonical_separate(original,canonical,audit)

if __name__=="__main__":
    unittest.main()
