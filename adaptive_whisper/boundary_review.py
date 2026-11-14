"""Fail-closed, human-audio boundary evidence gate (no transcript publication).

Cue timing/energy is not proof of correct words. This module validates
human audibility and explicitly reviewed word timestamps as separate facts.
Private real-word reviews stay off public GitHub; tests use synthetic text.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import wave
from typing import Any

from .diagnostics import parse_srt

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_WORD = re.compile(r"\w+", re.UNICODE)
_AUDIBILITY = {"fully_heard", "partial_context_completed"}


def _sha(value: object, key: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("Invalid SHA256: " + key)
    return value


def _integer(value: object, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Expected integer: " + key)
    return value


def _single_word(value: object, key: str) -> str:
    if (not isinstance(value, str) or not value
            or _WORD.fullmatch(value) is None):
        raise ValueError("Expected one Unicode word: " + key)
    return value


def validate_human_interval(review: dict[str, Any], *,
                            source_duration_ms: int) -> dict[str, Any]:
    """Check explicit human data; do not invent missing source-word timings."""
    if not isinstance(review, dict) or review.get("schema") != (
            "whispercpp/human-audio-boundary-review/v1"):
        raise ValueError("Unknown human boundary review schema")
    if review.get("speech_confirmed_by_human") is not True:
        raise ValueError("Audible speech requires explicit human confirmation")
    if type(review.get("reviewer_count")) is not int or review["reviewer_count"] != 1:
        raise ValueError("Single-reviewer schema requires reviewer_count=1")
    source_duration_ms = _integer(source_duration_ms, "source_duration_ms")
    start = _integer(review.get("start_ms"), "start_ms")
    end = _integer(review.get("end_ms"), "end_ms")
    if not (0 <= start < end <= source_duration_ms):
        raise ValueError("Invalid reviewed source-clock interval")
    audio_sha = _sha(review.get("source_audio_sha256"), "source_audio")
    human_sha = _sha(review.get("locked_reference_sha256"), "locked_reference")
    words = review.get("words")
    if not isinstance(words, list) or not words:
        raise ValueError("Expected nonempty ordered human word inventory")
    full, partial, timed = 0, 0, 0
    prev_start = start
    for ix, word in enumerate(words, 1):
        if (not isinstance(word, dict) or type(word.get("index")) is not int
                or word["index"] != ix):
            raise ValueError("Word index mismatch")
        text = _single_word(word.get("transcribed_word"), "transcribed_word")
        audibility = word.get("audibility")
        if not isinstance(audibility, str) or audibility not in _AUDIBILITY:
            raise ValueError("Unknown or missing audibility class")
        if audibility == "fully_heard":
            full += 1
            if word.get("heard_fragment") is not None:
                raise ValueError("Fully heard word cannot specify a fragment")
        else:
            partial += 1
            fragment = _single_word(word.get("heard_fragment"), "heard_fragment")
            if not text.casefold().startswith(fragment.casefold()):
                raise ValueError("Audible fragment must match the word onset")
        confirmed = word.get("human_timing_verified")
        if type(confirmed) is not bool:
            raise ValueError("Word timing verification must be explicit boolean")
        a, b = word.get("word_start_ms"), word.get("word_end_ms")
        if a is None or b is None:
            if a is not None or b is not None or confirmed:
                raise ValueError("Missing paired word times or false verification")
        else:
            a = _integer(a, "word_start_ms")
            b = _integer(b, "word_end_ms")
            if not (start <= a < b <= end):
                raise ValueError("Word timing exceeds reviewed interval")
            if a < prev_start:
                raise ValueError("Word start order is not monotonic")
            if not confirmed:
                raise ValueError("Word timestamps not confirmed by human")
            prev_start = a
            timed += 1
    return {
        "source_audio_sha256": audio_sha,
        "locked_reference_sha256": human_sha,
        "start_ms": start, "end_ms": end,
        "human_word_count": len(words),
        "fully_heard_word_count": full,
        "partially_heard_context_completed_word_count": partial,
        "human_timed_word_count": timed,
        "human_timestamps_complete": timed == len(words),
        "fully_audible_words_only": partial == 0,
    }


def assess_human_boundary(
    candidate_srt: str,
    review: dict[str, Any],
    *,
    original_duration_ms: int,
) -> dict[str, Any]:
    """Return text-free status only; never accept a transcript for release."""
    check = validate_human_interval(
        review, source_duration_ms=original_duration_ms
    )
    cues = parse_srt(candidate_srt)
    if candidate_srt.strip() and not cues:
        raise ValueError("Nonempty invalid candidate SRT is not an empty transcript")
    if any(c.start_ms < 0 or c.end_ms < c.start_ms
           or c.end_ms > original_duration_ms + 50 for c in cues):
        raise ValueError("Invalid source-clock cue timing")
    start, end = check["start_ms"], check["end_ms"]
    overlapping = [c for c in cues if c.start_ms < end and c.end_ms > start]
    crossing = sum(c.start_ms < start or c.end_ms > end for c in overlapping)
    state = ("HUMAN_CONFIRMED_SPEECH_WITHOUT_CAPTION" if not overlapping
             else "CAPTION_PRESENT_TEXT_UNVERIFIED")
    blockers = []
    if not check["human_timestamps_complete"]:
        blockers.append("WORD_TIMESTAMPS_NOT_HUMAN_VERIFIED")
    if not check["fully_audible_words_only"]:
        blockers.append("CONTEXT_COMPLETED_WORD_NOT_FULLY_AUDIBLE")
    if crossing:
        blockers.append("CUE_OVERLAPS_REVIEW_CLIP_BOUNDARY")
    blockers.append("EXPLICIT_EDITORIAL_DECISION_REQUIRED")
    return {
        "schema": "whispercpp/human-boundary-assessment/v1",
        "human_review_status": "SINGLE_REVIEWER_SPEECH_CONFIRMED",
        "candidate_status": state,
        "interval_ms": [start, end],
        "human_evidence": check,
        "candidate_srt_text_sha256": hashlib.sha256(
            candidate_srt.encode("utf-8")).hexdigest(),
        "intersecting_cue_count": len(overlapping),
        "crossing_cue_count": crossing,
        "ready_for_editorial_word_review": (
            check["human_timestamps_complete"]
            and check["fully_audible_words_only"]
        ),
        "blockers": blockers,
        "machine_words_compared_to_human_reference": False,
        "automatic_word_alignment_performed": False,
        "audio_energy_is_not_speech_ground_truth": True,
        "independently_adjudicated": False,
        "auto_release_authorized": False,
        "report_contains_no_transcript_text": True,
    }


def write_boundary_assessment(
    candidate_srt_path: Path,
    review_path: Path,
    source_audio_path: Path,
    report_path: Path,
    *,
    source_duration_ms: int,
    expected_candidate_sha256: str,
    expected_review_sha256: str,
) -> dict[str, Any]:
    """Exclusive write with original/source/human SHA checks before and after."""
    if report_path.exists():
        raise FileExistsError("Boundary report already exists")
    _sha(expected_candidate_sha256, "candidate")
    _sha(expected_review_sha256, "review")
    if any(report_path.resolve() == p.resolve()
           for p in (candidate_srt_path, review_path, source_audio_path)):
        raise ValueError("Output cannot be an input path")

    def file_sha(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1048576), b""):
                h.update(chunk)
        return h.hexdigest()

    if file_sha(candidate_srt_path) != expected_candidate_sha256:
        raise ValueError("Candidate SRT hash changed")
    if file_sha(review_path) != expected_review_sha256:
        raise ValueError("Locked review hash changed")
    evidence = json.loads(review_path.read_text(encoding="utf-8"))
    if file_sha(source_audio_path) != evidence.get("source_audio_sha256"):
        raise ValueError("Source audio hash changed")
    # A caller-supplied duration must agree with actual source PCM frames:
    # arbitrary clock arguments must not create false speech/word evidence.
    with wave.open(str(source_audio_path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                wav.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("Expected 16-kHz mono PCM16 original source")
        measured_ms = round(1000 * wav.getnframes() / wav.getframerate())
    if measured_ms != source_duration_ms:
        raise ValueError("Original source clock duration differs from WAV frames")
    result = assess_human_boundary(
        candidate_srt_path.read_text(encoding="utf-8-sig", errors="strict"),
        evidence, original_duration_ms=source_duration_ms
    )
    result["source_audio_sha256_on_disk"] = evidence["source_audio_sha256"]
    result["candidate_srt_sha256_on_disk"] = expected_candidate_sha256
    result["review_json_sha256_on_disk"] = expected_review_sha256
    with report_path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
        f.write("\n")
    if (file_sha(candidate_srt_path) != expected_candidate_sha256
            or file_sha(review_path) != expected_review_sha256
            or file_sha(source_audio_path) != evidence["source_audio_sha256"]):
        raise RuntimeError("Immutable input changed during boundary assessment")
    return result
