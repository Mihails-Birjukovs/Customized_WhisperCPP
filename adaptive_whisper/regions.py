"""Experimental bounded multi-region orchestration for output-level anomalies.

Each region runs through the existing single-region controller and receives its
own audit. Opt-in full-clock composition only occurs when every detected region
is structurally cleared, with disjoint source-window ownership and explicit
boundary risk records; there is never automatic publication.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Any

from .backend import DecodeProfile, ensure_inside, sha256_file
from .controller import (
    AdaptivePolicy, NativeAdapter, Trigger, _source_pcm16,
    detect_pathologies, run_adaptive,
)
from .multi_timeline import (
    RecoveryFile, OwnershipOverlapError, write_multi_timeline_preview,
)


@dataclass(frozen=True)
class Region:
    start_ms: int
    end_ms: int
    members: tuple[Trigger, ...]


def plan_regions(triggers: tuple[Trigger, ...], audio_ms: int, *,
                 guard_ms: int = 0, allow_unverified: bool = False
                 ) -> tuple[Region, ...]:
    """Merge pathologies whose source intervals (plus guard) intersect.

    Guard only influences grouping, not the reported original anomaly bounds.
    Return all eligible regions chronologically, without truncation by budget.
    """
    if audio_ms <= 0 or guard_ms < 0:
        raise ValueError("Invalid source duration or region guard")
    eligible = [t for t in triggers if
                t.kind != "no_output_unverified" or allow_unverified]
    for t in eligible:
        if not (0 <= t.start_ms < audio_ms and
                t.start_ms < t.end_ms <= audio_ms + 100):
            raise ValueError("Trigger exceeds source PCM and rounding tolerance")
    eligible.sort(key=lambda t: (t.start_ms, t.end_ms, t.kind, t.signal))
    groups: list[list[Trigger]] = []
    for t in eligible:
        if groups and t.start_ms <= max(x.end_ms for x in groups[-1]) + guard_ms:
            groups[-1].append(t)
        else:
            groups.append([t])
    return tuple(Region(min(t.start_ms for t in group),
                        max(t.end_ms for t in group), tuple(group))
                 for group in groups)


def run_adaptive_regions(
    workspace: Path,
    adapter: NativeAdapter,
    source_audio: Path,
    baseline_srt: Path | None,
    baseline_failed: bool,
    output_parent: Path,
    model_name: str,
    vad_name: str,
    initial_profile: DecodeProfile,
    policy: AdaptivePolicy = AdaptivePolicy(),
    adapters_by_strategy: dict[str, NativeAdapter] | None = None,
    *,
    group_guard_ms: int = 0,
) -> dict[str, Any]:
    """Process up to max_regions with one shared decode/job/wall budget.

    Multi-region mode only accepts local trigger-scoped strategies. Whole-source
    changes require separate experiments; repeated whole-source retries would
    undermine both isolation and the shared audio budget.

    Do not interpret a structurally cleared region as verified speech. An
    opt-in full-clock *provisional preview* is never a releasable transcript.
    """
    if any(step.scope != "trigger" for step in policy.strategies):
        raise ValueError("Multi-region mode requires trigger-scoped strategies")
    ws = workspace.resolve(strict=True)
    src = ensure_inside(source_audio, ws)
    base = ensure_inside(baseline_srt, ws) if baseline_srt is not None else None
    parent = ensure_inside(output_parent.parent, ws)
    if output_parent.resolve().parent != parent:
        raise ValueError("Output must be a direct child of trusted workspace parent")
    if output_parent.exists():
        raise FileExistsError("Refusing to overwrite multi-region output")
    _, audio_ms = _source_pcm16(src)
    original = base.read_text(encoding="utf-8-sig", errors="strict") if base else None
    detected = detect_pathologies(original, audio_ms,
                                  failure_or_timeout=baseline_failed,
                                  policy=policy.detection)
    regions = plan_regions(detected, audio_ms, guard_ms=group_guard_ms,
                           allow_unverified=policy.allow_unverified_no_output_trigger)
    output_parent.mkdir(exist_ok=False)
    started = time.perf_counter()
    consumed_ms = 0
    jobs = 0
    rows: list[dict[str, Any]] = []
    selected_recoveries: list[RecoveryFile] = []
    for index, region in enumerate(regions, 1):
        row: dict[str, Any] = {
            "region_index": index, "source_interval_ms": [region.start_ms, region.end_ms],
            "member_triggers": [asdict(t) for t in region.members],
            "status": "NOT_PROCESSED", "output_relative_dir": None,
        }
        rows.append(row)
        if index > policy.max_regions:
            row["status"] = "DEFERRED_REGION_LIMIT"
            continue
        remaining_ms = policy.max_total_decode_ms - consumed_ms
        remaining_jobs = policy.max_total_jobs - jobs
        remaining_wall = int(policy.max_wall_s - (time.perf_counter() - started))
        if remaining_ms < 1000 or remaining_jobs < 1 or remaining_wall < 1:
            row["status"] = "DEFERRED_GLOBAL_BUDGET"
            continue
        # Single-region artifacts and strategy loops remain independent.
        local_policy = replace(
            policy, max_total_decode_ms=remaining_ms,
            max_total_jobs=remaining_jobs, max_wall_s=remaining_wall,
            max_regions=1, compose_full_timeline_preview=False,
        )
        selected = Trigger(
            kind=region.members[0].kind if len(region.members) == 1 else "grouped_anomalies",
            start_ms=region.start_ms, end_ms=region.end_ms,
            signal=region.members[0].signal if len(region.members) == 1 else "merged_detected_intervals",
        )
        child = output_parent / f"region_{index:03d}"
        result = run_adaptive(
            ws, adapter, src, base, baseline_failed, child, model_name,
            vad_name, initial_profile, local_policy, adapters_by_strategy,
            trigger_override=selected,
        )
        consumed_ms += result["total_transcribed_audio_ms"]
        jobs += result["native_jobs_executed"]
        if consumed_ms > policy.max_total_decode_ms or jobs > policy.max_total_jobs:
            raise RuntimeError("Shared decode budget violated")
        row.update({
            "status": result["status"],
            "output_relative_dir": child.name,
            "native_jobs_executed": result["native_jobs_executed"],
            "redecoded_audio_ms": result["total_transcribed_audio_ms"],
            "selected_provisional_attempt": result.get("selected_provisional_attempt"),
            "ledger_sha256": sha256_file(child / "adaptive_decision_ledger.json"),
            "human_audio_accuracy_verified": False,
        })
        if (result["status"].startswith("PATHOLOGY_CLEARED") and
                result.get("selected_provisional_attempt") is not None):
            selected_id = result["selected_provisional_attempt"]
            chosen = next(a for a in result["attempts"] if a["index"] == selected_id)
            work = child / f"attempt_{selected_id:02d}_{chosen['strategy']}"
            recovery_path = ensure_inside(work / "provisional_recovery.srt", ws)
            if (not recovery_path.is_file() or
                    sha256_file(recovery_path) != chosen["private_preview_sha256"]):
                raise ValueError("Selected region SRT does not match decision-ledger hash")
            source_window = (
                min(w["source_start_ms"] for w in chosen["windows"]),
                max(w["source_end_ms"] for w in chosen["windows"]),
            )
            selected_recoveries.append(RecoveryFile(index, source_window, recovery_path))
            row["selected_audio_window_ms"] = list(source_window)
            row["selected_join_conflicts"] = chosen.get("join_conflicts", 0)
            row["selected_preview_relative_path"] = (
                f"{child.name}/{work.name}/provisional_recovery.srt"
            )
    completed = sum(r["status"].startswith("PATHOLOGY_CLEARED") for r in rows)
    deferred = sum(r["status"].startswith("DEFERRED") for r in rows)
    status = (
        "NO_ELIGIBLE_REGIONS" if not regions else
        "ALL_REGIONS_PROVISIONAL_UNVERIFIED"
        if completed == len(regions) else
        "PARTIAL_REGIONS_UNRESOLVED"
    )
    full_timeline: dict[str, Any] = {
        "status": "NOT_REQUESTED", "preview_written": False,
        "auto_release_authorized": False,
    }
    preview_written = False
    if policy.compose_full_timeline_preview:
        if not regions:
            full_timeline["status"] = "SKIPPED_NO_ELIGIBLE_REGIONS"
        elif base is None:
            full_timeline["status"] = "SKIPPED_NO_HISTORICAL_BASELINE"
        elif completed != len(regions) or len(selected_recoveries) != len(regions):
            full_timeline["status"] = "SKIPPED_UNRESOLVED_OR_DEFERRED_REGIONS"
            full_timeline["remaining_region_indices"] = [
                r["region_index"] for r in rows
                if not r["status"].startswith("PATHOLOGY_CLEARED")
            ]
        else:
            preview_path = output_parent / "provisional_multi_full_timeline.srt"
            review_path = output_parent / "provisional_multi_full_timeline_review.json"
            try:
                combined = write_multi_timeline_preview(
                    base, tuple(selected_recoveries), preview_path, review_path,
                    source_audio=src,
                    upstream_window_join_conflicts={
                        r["region_index"]: r.get("selected_join_conflicts", 0)
                        for r in rows
                    },
                )
                preview_written = True
                remaining = detect_pathologies(
                    preview_path.read_text(encoding="utf-8-sig", errors="strict"),
                    audio_ms, policy=policy.detection,
                )
                upstream_conflicts = sum(r.get("selected_join_conflicts", 0) for r in rows)
                full_timeline = {
                    "status": combined["status"],
                    "preview_written": True,
                    "preview_relative_path": preview_path.name,
                    "manifest_relative_path": review_path.name,
                    "preview_sha256": sha256_file(preview_path),
                    "hazards_count": combined["hazards_count"],
                    "upstream_window_join_conflicts": upstream_conflicts,
                    "residual_output_anomalies": [asdict(t) for t in remaining],
                    "source_window_ownership": combined["source_window_ownership"],
                    "auto_release_authorized": False,
                }
                if remaining or combined["hazards_count"] or upstream_conflicts:
                    status = "ALL_REGIONS_FULL_TIMELINE_REVIEW_REQUIRED"
                else:
                    status = "ALL_REGIONS_FULL_TIMELINE_STRUCTURAL_UNVERIFIED"
            except OwnershipOverlapError as exc:
                full_timeline = {
                    "status": "BLOCKED_OVERLAPPING_RECOVERY_WINDOWS",
                    "preview_written": False,
                    "first_region_index": exc.first_region,
                    "second_region_index": exc.second_region,
                    "overlap_ms": list(exc.overlap_ms),
                    "auto_release_authorized": False,
                }
                status = "ALL_REGIONS_FULL_TIMELINE_BLOCKED_REVIEW_REQUIRED"
            except (ValueError, UnicodeError, OSError) as exc:
                full_timeline = {
                    "status": "FAILED_REVIEW_REQUIRED",
                    "reason_type": type(exc).__name__,
                    "preview_written": False,
                    "auto_release_authorized": False,
                }
                status = "ALL_REGIONS_FULL_TIMELINE_ASSEMBLY_FAILED"
    report = {
        "schema": "whispercpp/adaptive-multiregion-prototype/v1",
        "status": status,
        "source_sha256": sha256_file(src),
        "baseline_srt_sha256": sha256_file(base) if base else None,
        "source_duration_ms": audio_ms,
        "baseline_detected": [asdict(t) for t in detected],
        "regions_detected": len(regions),
        "regions_structurally_cleared": completed,
        "regions_deferred": deferred,
        "max_regions": policy.max_regions,
        "group_guard_ms": group_guard_ms,
        "shared_limits": {
            "decode_ms": policy.max_total_decode_ms,
            "jobs": policy.max_total_jobs, "wall_s": policy.max_wall_s,
        },
        "consumed_decode_ms": consumed_ms,
        "consumed_native_jobs": jobs,
        "wall_elapsed_s": round(time.perf_counter() - started, 3),
        "regions": rows,
        "full_timeline": full_timeline,
        "full_timeline_splice_performed": preview_written,
        "human_audio_accuracy_verified": False,
        "auto_release_authorized": False,
        "historical_baseline_modified": False,
        "remaining_regions_must_not_be_marked_recovered": True,
    }
    with (output_parent / "multiregion_decision_ledger.json").open(
            "x", encoding="utf-8", newline="\n") as ledger:
        json.dump(report, ledger, indent=2, ensure_ascii=True)
        ledger.write("\n")
    return report
