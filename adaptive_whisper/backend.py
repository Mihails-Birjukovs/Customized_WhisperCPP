"""Prototype shared whisper.cpp adapter: CPU, CUDA and Vulkan.

Not a release. Only the backed native executable and device-evidence policy
vary by engine. No adaptive retry controller is implemented in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import time
from typing import Any


class Engine(str, Enum):
    CPU = "CPU"
    CUDA = "CUDA"
    VULKAN = "Vulkan"


@dataclass(frozen=True)
class DecodeProfile:
    threads: int = 12
    language: str = "auto"
    context: int = 128
    beams: int = 8
    best_of: int = 3
    temperature: float = 0.0
    temperature_inc: float = 0.10
    vad_threshold: float = 0.80
    vad_silence_ms: int = 400
    vad_speech_ms: int = 350
    vad_pad_ms: int = 50
    no_speech_threshold: float = 0.995
    gpu_device: int = 0
    native_json_full: bool = False


@dataclass(frozen=True)
class BackendEvidence:
    engine: Engine
    native_backend_registered: bool
    actual_gpu_selected: bool
    device_name: str | None
    vad_cpu_execution_observed: bool
    not_verified_reason: str | None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_inside(path: Path, root: Path) -> Path:
    """Canonical path guard, including symlinks/junctions."""
    absolute = path.resolve(strict=True)
    base = root.resolve(strict=True)
    if not absolute.is_relative_to(base):
        raise ValueError(f"Path outside WhisperCPP workspace: {path}")
    return absolute


class WhisperCppAdapter:
    def __init__(self, workspace: Path, engine: Engine, runtime: Path | None = None):
        self.workspace = workspace.resolve(strict=True)
        self.engine = engine
        self.runtime = ensure_inside(runtime or self.workspace / "runtimes" / engine.value, self.workspace)
        self.exe = ensure_inside(self.runtime / "whisper-cli.exe", self.workspace)

    def args(self, audio: Path, model_name: str, vad_name: str, output_base: Path,
             profile: DecodeProfile) -> list[str]:
        audio = ensure_inside(audio, self.workspace)
        output_parent = ensure_inside(output_base.parent, self.workspace)
        if output_parent != output_base.parent.resolve():
            raise ValueError("Unstable output parent")
        for name in (model_name, vad_name):
            if Path(name).name != name or name in ("", ".", ".."):
                raise ValueError("Model filenames must be direct names under runtime/models")
            ensure_inside(self.runtime / "models" / name, self.workspace)
        if profile.threads < 1 or profile.vad_pad_ms < 0:
            raise ValueError("Invalid profile")
        result = [
            str(self.exe), "-m", str(Path("models") / model_name),
            "-f", str(audio), "-l", profile.language, "-t", str(profile.threads),
            "-mc", str(profile.context), "-bs", str(profile.beams), "-bo", str(profile.best_of),
            "-tp", str(profile.temperature), "-tpi", str(profile.temperature_inc),
            "-pp", "--vad", "-vm", str(Path("models") / vad_name),
            "-vt", str(profile.vad_threshold), "-vsd", str(profile.vad_silence_ms),
            "-vspd", str(profile.vad_speech_ms), "-vp", str(profile.vad_pad_ms),
            "-nth", str(profile.no_speech_threshold), "-otxt", "-osrt", "-of", str(output_base)
        ]
        if profile.native_json_full:
            result.insert(result.index("-of"), "-ojf")
        if self.engine == Engine.VULKAN:
            result[1:1] = ["-dev", str(profile.gpu_device)]
        return result

    def identify_backend(self, stderr: str) -> BackendEvidence:
        vad_cpu = "whisper_vad_init_context: compute buffer" in stderr
        if self.engine == Engine.VULKAN:
            match = re.search(r"ggml_vulkan:\s*0\s*=\s*([^\r\n]+)", stderr)
            selected = re.search(r"whisper_backend_init_gpu:\s*using Vulkan\d+ backend", stderr) is not None
            return BackendEvidence(self.engine, match is not None, selected,
                                   match.group(1).strip() if match else None, vad_cpu,
                                   None if selected else "No explicit ASR Vulkan backend selection")
        if self.engine == Engine.CUDA:
            match = re.search(r"Device\s+0:\s*([^\r\n]+)", stderr)
            registered = "ggml_cuda_init: found" in stderr
            gpu = "use gpu    = 1" in stderr and registered
            return BackendEvidence(self.engine, registered, gpu,
                                   match.group(1).strip() if match else None, vad_cpu,
                                   None if gpu else "No positive CUDA initialization evidence")
        return BackendEvidence(self.engine, True, False, None, vad_cpu, None)

    def run_once(self, audio: Path, model_name: str, vad_name: str,
                 output_dir: Path, profile: DecodeProfile, timeout_s: int = 600) -> dict[str, Any]:
        """Run without overwriting. Preserves raw stderr/stdout and command metadata."""
        if timeout_s < 1:
            raise ValueError("Timeout must be positive")
        # Resolve existing parent; create exactly one new run folder.
        parent = ensure_inside(output_dir.parent, self.workspace)
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir}")
        output_dir.mkdir()
        output_dir = ensure_inside(output_dir, self.workspace)
        command = self.args(audio, model_name, vad_name, output_dir / "transcript", profile)
        paths = {"model": self.runtime / "models" / model_name,
                 "vad": self.runtime / "models" / vad_name, "audio": audio, "binary": self.exe}
        hashes = {k: sha256_file(ensure_inside(v, self.workspace)) for k, v in paths.items()}
        env = os.environ.copy()
        for variable in ("TMP", "TEMP", "TMPDIR"):
            env[variable] = str(self.workspace / "temp")
        env["CUDA_CACHE_PATH"] = str(self.workspace / "cache" / "cuda")
        env["HF_HOME"] = str(self.workspace / "cache" / "huggingface")
        begin = time.perf_counter()
        try:
            result = subprocess.run(command, cwd=self.runtime, env=env, timeout=timeout_s,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, errors="replace", check=False)
            exit_code: int | None = result.returncode
            stdout, stderr = result.stdout, result.stderr
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            exit_code = None
            timed_out = True
            stdout = str(exc.stdout or b"")
            stderr = str(exc.stderr or b"")
        elapsed = round(time.perf_counter() - begin, 3)
        (output_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (output_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        evidence = self.identify_backend(stderr)
        txt = (output_dir / "transcript.txt").is_file()
        srt = (output_dir / "transcript.srt").is_file()
        native_json = (output_dir / "transcript.json").is_file()
        record = {
            "schema": "whispercpp/backend-adapter-prototype/v1",
            "engine": self.engine.value, "command": command,
            "profile": asdict(profile), "hashes": hashes,
            "elapsed_seconds": elapsed, "exit_code": exit_code, "timed_out": timed_out,
            "txt_exists": txt, "srt_exists": srt,
            "native_json_full_requested": profile.native_json_full,
            "native_json_full_exists": native_json,
            "evidence": {**asdict(evidence), "engine": evidence.engine.value},
            "accepted": exit_code == 0 and not timed_out and txt and srt
                        and (not profile.native_json_full or native_json)
                        and (self.engine == Engine.CPU or evidence.actual_gpu_selected),
        }
        (output_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record
