"""Portable, offline adapter invariants. No real ASR execution in tests."""
from __future__ import annotations
import sys
import os
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adaptive_whisper.backend import Engine, DecodeProfile, WhisperCppAdapter, ensure_inside

SANDBOX = Path(os.environ.get("WHISPER_TEST_TMP_ROOT") or tempfile.gettempdir()).resolve()
SANDBOX.mkdir(parents=True, exist_ok=True)


class BackendContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=SANDBOX)
        self.root = Path(self.tmp.name)
        self.audio = self.root / "fixture.wav"
        self.audio.write_bytes(b"RIFF-test")
        for k in ("CPU", "CUDA", "Vulkan"):
            p = self.root / "runtimes" / k
            (p / "models").mkdir(parents=True)
            (p / "whisper-cli.exe").write_bytes(b"fixture")
            for name in ("tiny.bin", "vad.bin"):
                (p / "models" / name).write_bytes(b"model")
        self.output = self.root / "results"
        self.output.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_default_same_engine_independent_settings(self):
        self.assertEqual(DecodeProfile().context, 128)
        self.assertEqual(DecodeProfile().beams, 8)
        self.assertEqual(DecodeProfile().vad_pad_ms, 50)

    def test_device_switch_is_vulkan_only(self):
        cpu = WhisperCppAdapter(self.root, Engine.CPU)
        cuda = WhisperCppAdapter(self.root, Engine.CUDA)
        vk = WhisperCppAdapter(self.root, Engine.VULKAN)
        profile = DecodeProfile()
        args = [a.args(self.audio, "tiny.bin", "vad.bin", self.output / a.engine.value, profile)
                for a in (cpu, cuda, vk)]
        self.assertNotIn("-dev", args[0])
        self.assertNotIn("-dev", args[1])
        self.assertIn("-dev", args[2])
        for arglist in args:
            for flag in ("-mc", "-bs", "-bo", "-tp", "-tpi", "--vad", "-vm",
                         "-vt", "-vsd", "-vspd", "-vp", "-nth", "-otxt", "-osrt"):
                self.assertIn(flag, arglist)

    def test_optional_native_json_is_shared(self):
        for engine in (Engine.CPU, Engine.CUDA, Engine.VULKAN):
            adapter = WhisperCppAdapter(self.root, engine)
            args = adapter.args(self.audio, "tiny.bin", "vad.bin",
                                self.output / engine.value, DecodeProfile(native_json_full=True))
            self.assertIn("-ojf", args)
        without_json = WhisperCppAdapter(self.root, Engine.CPU).args(
            self.audio, "tiny.bin", "vad.bin", self.output / "nojson", DecodeProfile())
        self.assertNotIn("-ojf", without_json)

    def test_vulkan_gpu_not_confused_with_cpu_vad(self):
        a = WhisperCppAdapter(self.root, Engine.VULKAN)
        s = ("ggml_vulkan: 0 = NVIDIA GeForce RTX 3070 (NVIDIA)\n"
             "whisper_backend_init_gpu: using Vulkan0 backend\n"
             "whisper_vad_init_context: compute buffer (VAD) = 1.60 MB\n"
             "whisper_backend_init_gpu: no GPU found\n")
        e = a.identify_backend(s)
        self.assertTrue(e.actual_gpu_selected)
        self.assertTrue(e.vad_cpu_execution_observed)

    def test_vulkan_missing_evidence_rejected(self):
        a = WhisperCppAdapter(self.root, Engine.VULKAN)
        self.assertFalse(a.identify_backend("no GPU found\n").actual_gpu_selected)

    def test_path_cannot_escape_workspace(self):
        sibling = self.root.parent / "other-fixture"
        sibling.mkdir(exist_ok=True)
        self.addCleanup(lambda: sibling.rmdir())
        with self.assertRaises(ValueError):
            ensure_inside(sibling, self.root)
        with self.assertRaises(ValueError):
            WhisperCppAdapter(self.root, Engine.CPU).args(
                self.audio, "../outside.bin", "vad.bin", self.output / "foo", DecodeProfile())

    def test_no_output_overwrite(self):
        a = WhisperCppAdapter(self.root, Engine.CPU)
        occupied = self.output / "existing"
        occupied.mkdir()
        with self.assertRaises(FileExistsError):
            a.run_once(self.audio, "tiny.bin", "vad.bin", occupied, DecodeProfile(), timeout_s=2)


if __name__ == "__main__":
    unittest.main()
