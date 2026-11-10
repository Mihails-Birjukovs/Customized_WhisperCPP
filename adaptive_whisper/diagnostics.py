"""Shared, backend-neutral SRT output diagnostics.

Output-only anomaly candidates; never claim acoustic error or WER from SRT.
Raw text remains in memory; public manifests should contain counts/times only.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class Cue:
    index: int
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class RepetitionCandidate:
    start_ms: int
    end_ms: int
    segment_count: int
    text_length: int
    duration_ms: int
    label: str = "suspected_output_repetition"


@dataclass(frozen=True)
class SrtDiagnostics:
    cue_count: int
    invalid_timing_cues: int
    backwards_start_cues: int
    candidates: tuple[RepetitionCandidate, ...]


_TIMING = re.compile(
    r"^\s*(\d{1,3}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(\d{1,3}:\d{2}:\d{2}[,.]\d{3})(?:\s+.*)?$"
)


def time_ms(value: str) -> int:
    match = re.fullmatch(r"(\d{1,3}):(\d{2}):(\d{2})[,.](\d{3})", value)
    if not match:
        raise ValueError(f"Malformed SRT time: {value!r}")
    h, m, s, ms = (int(x) for x in match.groups())
    if m > 59 or s > 59:
        raise ValueError(f"Invalid minute or second in {value!r}")
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def parse_srt(raw: str) -> tuple[Cue, ...]:
    cues = []
    for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")):
        lines = block.strip().split("\n")
        ix = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if ix is None:
            continue
        match = _TIMING.fullmatch(lines[ix])
        if not match:
            raise ValueError(f"Invalid cue timestamp: {lines[ix]!r}")
        text = " ".join(line.strip() for line in lines[ix+1:] if line.strip()).strip()
        if not text:
            continue
        cues.append(Cue(len(cues)+1, time_ms(match.group(1)), time_ms(match.group(2)), text))
    return tuple(cues)


def _key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def diagnose_srt(raw: str, min_repeat: int = 3, min_chars: int = 30) -> SrtDiagnostics:
    if min_repeat < 2 or min_chars < 1:
        raise ValueError("Invalid repetition thresholds")
    cues = parse_srt(raw)
    negatives = sum(cue.end_ms < cue.start_ms for cue in cues)
    backwards = sum(cues[i].start_ms < cues[i-1].start_ms for i in range(1, len(cues)))
    runs = []
    start = 0
    for end in range(1, len(cues)+1):
        if end < len(cues) and _key(cues[end].text) == _key(cues[start].text):
            continue
        count = end-start
        if count >= min_repeat and len(_key(cues[start].text)) >= min_chars:
            first, last = cues[start], cues[end-1]
            runs.append(RepetitionCandidate(
                first.start_ms,last.end_ms,count,len(_key(first.text)),
                last.end_ms-first.start_ms
            ))
        start = end
    return SrtDiagnostics(len(cues), negatives, backwards, tuple(runs))
