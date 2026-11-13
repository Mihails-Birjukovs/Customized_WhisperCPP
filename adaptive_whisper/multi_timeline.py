"""Provisional, provenance-preserving SRT assembly for disjoint recovery regions.

This is NOT automatic transcript correction. Source caption hypotheses remain
whole, cross-origin disagreements are retained and flagged, and no recovered
speech is asserted without listening. Overlapping *ownership windows* fail
closed because there is no independently validated choice of which region owns
their shared audio.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import wave
from typing import Any

from .coverage import CoveragePolicy, region_energy_coverage
from .diagnostics import Cue, diagnose_srt, parse_srt
from .full_timeline import TimelinePolicy, _sha, _hash_file, _validate
from .stitch import fmt


@dataclass(frozen=True)
class RecoveryReplacement:
    region_index: int
    window_ms: tuple[int, int]
    source_srt: str


@dataclass(frozen=True)
class RecoveryFile:
    region_index: int
    window_ms: tuple[int, int]
    srt_path: Path


class OwnershipOverlapError(ValueError):
    """Two recovery source windows own some of the same original audio."""

    def __init__(self, first: int, second: int, overlap: tuple[int, int]):
        self.first_region = first
        self.second_region = second
        self.overlap_ms = overlap
        super().__init__(
            f"Recovery ownership overlap between regions {first} and {second}: {overlap}"
        )


def _checked_replacements(
    replacements: tuple[RecoveryReplacement, ...], duration_ms: int
) -> tuple[RecoveryReplacement, ...]:
    if not replacements or duration_ms < 1:
        raise ValueError("Need at least one recovery and a positive source duration")
    ids = [r.region_index for r in replacements]
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 1 for i in ids):
        raise ValueError("Recovery region indexes must be positive integers")
    if len(ids) != len(set(ids)):
        raise ValueError("Recovery region indexes must be unique")
    ordered = tuple(sorted(replacements, key=lambda r: (r.window_ms[0], r.region_index)))
    for r in ordered:
        start, end = r.window_ms
        if not (0 <= start < end <= duration_ms):
            raise ValueError("Recovery window exceeds original source clock")
    for left, right in zip(ordered, ordered[1:]):
        if right.window_ms[0] < left.window_ms[1]:
            raise OwnershipOverlapError(
                left.region_index, right.region_index,
                (right.window_ms[0], min(left.window_ms[1], right.window_ms[1]))
            )
    return ordered


def assemble_multi_timeline(
    baseline_srt: str,
    replacements: tuple[RecoveryReplacement, ...],
    *,
    original_duration_ms: int,
    audio: Path | None = None,
    policy: TimelinePolicy = TimelinePolicy(),
) -> tuple[str, dict[str, Any]]:
    """Produce a *private, provisional* full source-clock subtitle candidate.

    Recovered cues must use original source time, not clip-relative time.
    Baseline is retained between and outside windows; recovered text owns
    disjoint windows by whole-cue midpoint. Nothing is trimmed or inferred.
    If any ownership windows overlap the operation rejects the splice before
    emitting a preview, so priority between competing recovery regions is
    never silently invented.
    """
    ordered = _checked_replacements(replacements, original_duration_ms)
    baseline = parse_srt(baseline_srt)
    _validate(baseline, original_duration_ms,
              policy.max_out_of_audio_rounding_ms, "baseline")
    inputs: list[tuple[RecoveryReplacement, tuple[Cue, ...], tuple[int, ...]]] = []
    for r in ordered:
        cues = parse_srt(r.source_srt)
        overhang = _validate(
            cues, original_duration_ms, policy.max_out_of_audio_rounding_ms,
            f"recovery_region_{r.region_index}",
        )
        inputs.append((r, cues, overhang))

    hazards: list[dict[str, Any]] = []

    def flag(code: str, bounds: tuple[int, int], *,
             severity: str = "review_required", **fields: Any) -> None:
        hazards.append({
            "code": code, "interval_ms": list(bounds),
            "severity": severity, **fields,
        })

    ownership: list[dict[str, Any]] = []
    cursor = 0
    for r in ordered:
        start, end = r.window_ms
        if cursor < start:
            ownership.append({"owner": "baseline", "interval_ms": [cursor, start]})
        ownership.append({
            "owner": "recovery", "region_index": r.region_index,
            "interval_ms": [start, end],
        })
        cursor = end
    if cursor < original_duration_ms:
        ownership.append({
            "owner": "baseline", "interval_ms": [cursor, original_duration_ms],
        })

    # Midpoints assign intact historical captions to exactly one owner.
    retained: list[Cue] = []
    discarded: list[tuple[Cue, int]] = []
    for cue in baseline:
        mid = (cue.start_ms + cue.end_ms) // 2
        selected_region = next((
            r.region_index for r in ordered
            if r.window_ms[0] <= mid < r.window_ms[1]
            or (r.window_ms[1] == original_duration_ms and mid == original_duration_ms)
        ), None)
        if selected_region is None:
            retained.append(cue)
        else:
            discarded.append((cue, selected_region))

    # Preserve all historical captions straddling a replacement boundary in
    # the hazard ledger, including discarded captions spanning *two* regions.
    for cue in retained:
        for r in ordered:
            start, end = r.window_ms
            if cue.start_ms < start < cue.end_ms:
                flag("RETAINED_BASELINE_CUE_CROSSES_REGION_START",
                     (start, cue.end_ms), region_index=r.region_index,
                     source_cue_index=cue.index)
            if cue.start_ms < end < cue.end_ms:
                flag("RETAINED_BASELINE_CUE_CROSSES_REGION_END",
                     (cue.start_ms, end), region_index=r.region_index,
                     source_cue_index=cue.index)
    for cue, owner in discarded:
        for r in ordered:
            start, end = r.window_ms
            if cue.start_ms < start < cue.end_ms or cue.start_ms < end < cue.end_ms:
                flag("DISCARDED_BASELINE_CUE_CROSSES_REGION_BOUNDARY",
                     (cue.start_ms, cue.end_ms),
                     selected_region_index=owner, boundary_region_index=r.region_index,
                     source_cue_index=cue.index)

    # The origin tuple is (cue, source-label, region-ID). It is the sole
    # authority for emitted provenance.
    selected: list[tuple[Cue, str, int | None]] = [
        (cue, "baseline", None) for cue in retained
    ]
    per_region: list[dict[str, Any]] = []
    for r, cues, overhang in inputs:
        start, end = r.window_ms
        emitted = []
        ignored = []
        for cue in cues:
            mid = (cue.start_ms + cue.end_ms) // 2
            if start <= mid < end or (end == original_duration_ms and mid == end):
                emitted.append(cue)
                selected.append((cue, "recovery", r.region_index))
                if cue.start_ms < start:
                    flag("RECOVERY_CUE_CROSSES_REGION_START",
                         (cue.start_ms, start), region_index=r.region_index,
                         source_cue_index=cue.index)
                if cue.end_ms > end:
                    flag("RECOVERY_CUE_CROSSES_REGION_END",
                         (end, cue.end_ms), region_index=r.region_index,
                         source_cue_index=cue.index)
            else:
                ignored.append(cue.index)
        if not emitted:
            raise ValueError(f"No captions assigned to recovery region {r.region_index}")
        if overhang:
            flag("RECOVERED_NATIVE_TIMESTAMP_ROUNDING",
                 (original_duration_ms, max(c.end_ms for c in cues if c.index in overhang)),
                 severity="rounding_review", region_index=r.region_index,
                 count=len(overhang))
        per_region.append({
            "region_index": r.region_index,
            "source_window_ms": list(r.window_ms),
            "input_srt_sha256": _sha(r.source_srt),
            "source_candidate_cue_count": len(cues),
            "selected_cue_count": len(emitted),
            "excluded_outside_window_cue_indices": ignored,
            "native_rounding_cue_indices": list(overhang),
        })

    selected.sort(key=lambda item: (
        item[0].start_ms, item[0].end_ms, item[1],
        -1 if item[2] is None else item[2],
    ))
    # Scan ALL cross-origin pairs, not only adjacent caption pairs.
    for i, (a, a_source, a_region) in enumerate(selected):
        for b, b_source, b_region in selected[i + 1:]:
            if b.start_ms >= a.end_ms:
                break
            if (a_source, a_region) == (b_source, b_region):
                continue
            overlap = min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms)
            if overlap > 0:
                flag(
                    "CROSS_REGION_RECOVERY_CAPTION_OVERLAP"
                    if a_source == "recovery" and b_source == "recovery"
                    else "CROSS_SOURCE_CAPTION_OVERLAP",
                    (max(a.start_ms, b.start_ms), min(a.end_ms, b.end_ms)),
                    overlap_ms=overlap,
                    exceeds_configured_overlap_tolerance=(
                        overlap > policy.max_cross_source_overlap_ms
                    ),
                    sources=[a_source, b_source],
                    region_indices=[a_region, b_region],
                    source_cue_indices=[a.index, b.index],
                )
    for (a, a_source, a_region), (b, b_source, b_region) in zip(
        selected, selected[1:]
    ):
        if (a_source, a_region) == (b_source, b_region):
            continue
        gap = max(0, b.start_ms - a.end_ms)
        if gap >= policy.min_gap_to_review_ms:
            flag(
                "CROSS_OWNER_CAPTION_GAP", (a.end_ms, b.start_ms),
                gap_ms=gap,
                sources=[a_source, b_source],
                region_indices=[a_region, b_region],
            )

    output = "".join(
        f"{i}\n{fmt(cue.start_ms)} --> {fmt(cue.end_ms)}\n{cue.text}\n\n"
        for i, (cue, source, region) in enumerate(selected, 1)
    )
    diag = diagnose_srt(output)
    if diag.invalid_timing_cues or diag.backwards_start_cues:
        raise ValueError("Multi-region preview contains invalid SRT timing")
    provenance = [
        {
            "output_cue_index": i,
            "source": source,
            "region_index": region,
            "source_cue_index": cue.index,
            "start_ms": cue.start_ms,
            "end_ms": cue.end_ms,
            "source_text_sha256": _sha(cue.text),
        }
        for i, (cue, source, region) in enumerate(selected, 1)
    ]
    acoustic: list[dict[str, Any]] = []
    if audio is not None:
        # Sound energy is not a voice-activity or word-accuracy label.
        for r in ordered:
            for boundary in r.window_ms:
                begin = max(0, boundary - policy.context_guard_ms)
                finish = min(original_duration_ms, boundary + policy.context_guard_ms)
                if finish - begin < 100:
                    continue
                for threshold in policy.coverage_dbfs:
                    acoustic.append(region_energy_coverage(
                        audio, output, begin, finish,
                        CoveragePolicy(energy_dbfs=threshold)
                    ))
    report = {
        "schema": "whispercpp/conservative-multi-timeline-assembly/v1",
        "status": ("PROVISIONAL_REVIEW_REQUIRED" if hazards
                   else "STRUCTURALLY_COMPOSED_UNVERIFIED"),
        "original_source_duration_ms": original_duration_ms,
        "baseline_input_srt_sha256": _sha(baseline_srt),
        "recovery_region_inputs": per_region,
        "output_srt_sha256": _sha(output),
        "baseline_cue_count": len(baseline),
        "retained_baseline_cue_count": len(retained),
        "discarded_baseline_cue_count": len(discarded),
        "output_cue_count": len(selected),
        "recovery_cue_count": len(selected) - len(retained),
        "source_window_ownership": ownership,
        "source_clock_window_coverage_full": True,
        "cue_source_provenance": provenance,
        "hazards": hazards,
        "hazards_count": len(hazards),
        "repeated_caption_run_count": len(diag.candidates),
        "acoustic_energy_proxy_near_boundaries": acoustic,
        "historical_source_caption_text_retained_verbatim": True,
        "human_audio_verification": False,
        "recognition_accuracy_not_measured": True,
        "source_clock_coverage_not_speech_coverage": True,
        "auto_release_authorized": False,
        "do_not_overwrite_original": True,
    }
    return output, report


def write_multi_timeline_preview(
    baseline_path: Path,
    recoveries: tuple[RecoveryFile, ...],
    destination_srt: Path,
    destination_manifest: Path,
    *,
    source_audio: Path,
    policy: TimelinePolicy = TimelinePolicy(),
    upstream_window_join_conflicts: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Check source format, hashes and output collisions before writing.

    Do not emit any preview on invalid or overlapping region ownership.
    The caller must apply trusted-workspace path guards before this wrapper.
    """
    if destination_srt == destination_manifest:
        raise ValueError("SRT and manifest must have different destinations")
    if destination_srt.exists() or destination_manifest.exists():
        raise FileExistsError("Refusing to overwrite a multi-region preview")
    with wave.open(str(source_audio), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth(),
                w.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("Expected 16-kHz mono PCM16 original source")
        duration_ms = round(1000 * w.getnframes() / w.getframerate())
    baseline_srt = baseline_path.read_text(encoding="utf-8-sig", errors="strict")
    replacements = tuple(
        RecoveryReplacement(
            region_index=r.region_index,
            window_ms=r.window_ms,
            source_srt=r.srt_path.read_text(encoding="utf-8-sig", errors="strict"),
        )
        for r in recoveries
    )
    text, ledger = assemble_multi_timeline(
        baseline_srt, replacements, original_duration_ms=duration_ms,
        audio=source_audio, policy=policy,
    )
    upstream = upstream_window_join_conflicts or {}
    indexes = {r.region_index for r in recoveries}
    if (any(index not in indexes for index in upstream) or
            any(not isinstance(count, int) or isinstance(count, bool) or count < 0
                for count in upstream.values())):
        raise ValueError("Invalid upstream window-join conflict provenance")
    ledger["upstream_window_join_conflicts"] = [
        {"region_index": index, "count": upstream.get(index, 0)}
        for index in sorted(indexes)
    ]
    for r in recoveries:
        count = upstream.get(r.region_index, 0)
        if count:
            ledger["hazards"].append({
                "code": "UPSTREAM_WINDOW_JOIN_CONFLICTS",
                "interval_ms": list(r.window_ms),
                "interval_is_recovery_window_not_localized_join": True,
                "severity": "review_required",
                "region_index": r.region_index, "count": count,
            })
    ledger["hazards_count"] = len(ledger["hazards"])
    if ledger["hazards"]:
        ledger["status"] = "PROVISIONAL_REVIEW_REQUIRED"
    ledger["input_file_sha256"] = {
        "original_audio": _hash_file(source_audio),
        "baseline_srt": _hash_file(baseline_path),
        "recovery_region_srt": [
            {"region_index": r.region_index,
             "sha256": _hash_file(r.srt_path)}
            for r in recoveries
        ],
    }
    # Exclusive creation plus best-effort rollback avoids silently overwriting
    # prior runs if an output appears while the analysis was in progress.
    wrote_srt = False
    wrote_manifest = False
    try:
        with destination_srt.open("x", encoding="utf-8", newline="\n") as f:
            wrote_srt = True
            f.write(text)
        with destination_manifest.open("x", encoding="utf-8", newline="\n") as f:
            wrote_manifest = True
            json.dump(ledger, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception:
        # Only files opened exclusively by this call are eligible for cleanup.
        # A pre-existing file is never unlinked by a collision error.
        if wrote_manifest:
            destination_manifest.unlink(missing_ok=True)
        if wrote_srt:
            destination_srt.unlink(missing_ok=True)
        raise
    return ledger
