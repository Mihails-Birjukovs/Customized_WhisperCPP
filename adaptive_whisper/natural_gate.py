"""Strict natural-recording, exact-caption-boundary prefix gate (Stage 10).

Preserves complete historical SRT cue bytes and source WAV. A cropped prefix
is derived only when its endpoint coincides with a historical caption end
and a separate next-caption start; no preexisting source files are changed.

This is a fixture PREPARATION step, not a decoder run or audio truth label.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import wave
from typing import Any

from .backend import ensure_inside, sha256_file
from .controller import detect_pathologies
from .diagnostics import parse_srt
from .regions import plan_regions


# Exact original SRT byte boundaries, without normalizing CRLF or text.
_SRT_HEADER_BYTES = re.compile(
    rb"(?m)^[0-9]+\r?\n[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}"
    rb" --> [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}\r?\n"
)


def plan_natural_prefix(
    historical_srt_bytes: bytes,
    *,
    prefix_end_ms: int,
    expected_regions_ms: tuple[tuple[int, int], ...],
) -> tuple[bytes, dict[str, Any]]:
    """SRT-only preflight; no private audio is needed to validate the plan.

    The prefix must retain *all* original captions through an exact end
    boundary. A caption crossing that boundary or a missing final caption
    causes refusal rather than silent text/timestamp normalization.
    """
    if not isinstance(prefix_end_ms, int) or isinstance(prefix_end_ms, bool) or prefix_end_ms <= 0:
        raise ValueError("Invalid prefix end")
    if not expected_regions_ms or len(set(expected_regions_ms)) != len(expected_regions_ms):
        raise ValueError("Expected distinct, explicitly declared regions")
    original = historical_srt_bytes.decode("utf-8-sig", errors="strict")
    cues = parse_srt(original)
    if not cues:
        raise ValueError("Original SRT has no captions")
    if any(c.start_ms < 0 or c.end_ms < c.start_ms for c in cues):
        raise ValueError("Unusable original cue timing")
    if any(c.start_ms < prefix_end_ms < c.end_ms for c in cues):
        raise ValueError("Prefix would cut through a historical caption")
    keep = tuple(c for c in cues if c.end_ms <= prefix_end_ms)
    if not keep or keep[-1].end_ms != prefix_end_ms:
        raise ValueError("Prefix must end at the exact last selected subtitle boundary")
    if len(keep) >= len(cues):
        raise ValueError("Requested test prefix does not exclude any later cue")
    # Anchored numeric SRT headers support the same ordinary SRT dialect as
    # our input corpus. We fail closed when the byte-level parse disagrees.
    markers = tuple(_SRT_HEADER_BYTES.finditer(historical_srt_bytes))
    if len(markers) != len(cues):
        raise ValueError("Cannot preserve byte-exact SRT prefix unambiguously")
    boundary = markers[len(keep)].start()
    prefix_bytes = historical_srt_bytes[:boundary]
    prefix_cues = parse_srt(prefix_bytes.decode("utf-8-sig", errors="strict"))
    if prefix_cues != keep:
        raise ValueError("SRT prefix byte boundary failed roundtrip verification")
    triggers = detect_pathologies(prefix_bytes.decode("utf-8-sig"), prefix_end_ms)
    regions = plan_regions(triggers, prefix_end_ms)
    bounds = tuple((r.start_ms, r.end_ms) for r in regions)
    if bounds != expected_regions_ms:
        raise ValueError(
            f"Expected exactly {expected_regions_ms}, detector found {bounds}"
        )
    report = {
        "schema": "whispercpp/natural-prefix-preflight/v1",
        "cut_ms": prefix_end_ms,
        "source_historical_srt_sha256": hashlib.sha256(historical_srt_bytes).hexdigest(),
        "source_srt_cue_count": len(cues),
        "retained_srt_cue_count": len(keep),
        "prefix_srt_sha256": hashlib.sha256(prefix_bytes).hexdigest(),
        "prefix_srt_exact_original_byte_prefix": True,
        "original_srt_byte_length": len(historical_srt_bytes),
        "prefix_srt_byte_length": len(prefix_bytes),
        "region_bounds_ms": [list(b) for b in bounds],
        "regions": [
            {"start_ms": r.start_ms, "end_ms": r.end_ms,
             "triggers": [asdict(t) for t in r.members]}
            for r in regions
        ],
        "source_audio_not_necessarily_present": True,
        "historical_transcripts_are_not_ground_truth": True,
        "human_audio_accuracy_not_verified": True,
        "no_automatic_release": True,
    }
    return prefix_bytes, report


def prepare_natural_prefix(
    workspace: Path,
    *,
    original_wav: Path,
    original_srt: Path,
    destination: Path,
    prefix_end_ms: int,
    expected_regions_ms: tuple[tuple[int, int], ...],
) -> dict[str, Any]:
    """Write one new, isolated WAV+SRT+manifest, never overwrite inputs.

    Exact PCM16 mono 16 kHz input is mandatory. Cut at 16 samples per
    millisecond; preserve all PCM samples up to the cut. Reproducible hashes
    enable re-verification before any GPU inference.
    """
    root = workspace.resolve(strict=True)
    source = ensure_inside(original_wav, root)
    baseline = ensure_inside(original_srt, root)
    output_parent = ensure_inside(destination.parent, root)
    if destination.resolve().parent != output_parent:
        raise ValueError("Unstable destination parent")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite natural-case fixture")
    # All source structure and expected-region checks before creating output.
    historical_srt_bytes = baseline.read_bytes()
    srt_prefix, report = plan_natural_prefix(
        historical_srt_bytes,
        prefix_end_ms=prefix_end_ms,
        expected_regions_ms=expected_regions_ms,
    )
    with wave.open(str(source), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate(),
                w.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError("Original source must be 16-kHz mono PCM16")
        original_frames = w.getnframes()
        frames_wanted = prefix_end_ms * 16
        if frames_wanted >= original_frames:
            raise ValueError("The requested prefix is not shorter than the WAV")
    original_audio_hash = sha256_file(source)
    destination.mkdir(exist_ok=False)
    wav_out = destination / "natural_prefix_pcm16.wav"
    srt_out = destination / "historical_prefix_exact_original_bytes.srt"
    with wave.open(str(source), "rb") as reader, wav_out.open("xb") as handle:
        with wave.open(handle, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            left = frames_wanted
            while left:
                block = reader.readframes(min(left, 65536))
                if not block:
                    raise IOError("Unexpected EOF during prefix PCM extraction")
                if len(block) % 2:
                    raise IOError("Unaligned PCM16 source chunk")
                writer.writeframesraw(block)
                left -= len(block) // 2
            writer.writeframes(b"")
    with wave.open(str(wav_out), "rb") as recorded:
        if (recorded.getnframes() != frames_wanted or
                recorded.getframerate() != 16000 or
                recorded.getnchannels() != 1 or
                recorded.getsampwidth() != 2):
            raise RuntimeError("Extracted WAV layout mismatch")
    with srt_out.open("xb") as file:
        file.write(srt_prefix)
    if sha256_file(srt_out) != report["prefix_srt_sha256"]:
        raise RuntimeError("SRT prefix hash mismatch")
    if sha256_file(source) != original_audio_hash:
        raise RuntimeError("Original WAV hash changed during preparation")
    if sha256_file(baseline) != report["source_historical_srt_sha256"]:
        raise RuntimeError("Original SRT hash changed during preparation")
    report.update({
        "source_audio_not_necessarily_present": False,
        "original_wav_sha256": original_audio_hash,
        "original_wav_frames": original_frames,
        "derived_prefix_wav_sha256": sha256_file(wav_out),
        "derived_prefix_pcm_frames": frames_wanted,
        "derived_pcm_duration_ms": prefix_end_ms,
        "destination_files": {
            "wav": wav_out.name, "srt": srt_out.name,
            "manifest": "natural_prefix_manifest.json"
        },
        "original_files_modified": False,
        "human_reference_not_available_for_this_natural_recording": True,
    })
    with (destination / "natural_prefix_manifest.json").open(
            "x", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, indent=2, ensure_ascii=True)
        file.write("\n")
    return report
