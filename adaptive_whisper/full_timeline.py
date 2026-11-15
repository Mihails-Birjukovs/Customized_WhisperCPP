"""Conservative *provisional* full-timeline SRT assembly after local retry.

Keep original SRT/WAV immutable. A caption cannot be sliced at a phonetic
boundary without listening; this tool selects whole source cues by midpoint,
retains all provenance, and refuses to call a conflicted join "recovered".
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
from typing import Any

from .diagnostics import Cue, diagnose_srt, parse_srt
from .stitch import fmt
from .coverage import CoveragePolicy, region_energy_coverage


@dataclass(frozen=True)
class TimelinePolicy:
    max_out_of_audio_rounding_ms: int = 100
    max_cross_source_overlap_ms: int = 750
    min_gap_to_review_ms: int = 1200
    context_guard_ms: int = 1500
    coverage_dbfs: tuple[float,...] = (-48.,-42.,-36.)

    def __post_init__(self) -> None:
        if (self.max_out_of_audio_rounding_ms < 0 or
            self.max_cross_source_overlap_ms < 0 or
            self.min_gap_to_review_ms < 1 or self.context_guard_ms < 0 or
            not self.coverage_dbfs):
            raise ValueError("Invalid timeline policy")


def _sha(text:str)->str:
    return sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path:Path)->str:
    hasher=sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate(cues:tuple[Cue,...],duration_ms:int,
              overhang_ms:int,source:str)->tuple[int,...]:
    bad=[]
    for c in cues:
        if (c.start_ms < 0 or c.end_ms <= c.start_ms or
            c.start_ms >= duration_ms + overhang_ms or
            c.end_ms > duration_ms + overhang_ms):
            bad.append(c.index)
    if bad:
        raise ValueError(f"{source} subtitle cue outside source PCM tolerance: {bad[:5]}")
    if any(x.start_ms>y.start_ms for x,y in zip(cues,cues[1:])):
        raise ValueError(f"{source} cue start order regressed")
    return tuple(c.index for c in cues if c.end_ms>duration_ms)


def assemble_full_timeline(
    baseline_srt:str,
    repaired_srt:str,
    *,
    original_duration_ms:int,
    repaired_audio_window_ms:tuple[int,int],
    audio:Path | None = None,
    policy:TimelinePolicy = TimelinePolicy(),
) -> tuple[str,dict[str,Any]]:
    """Assemble source-preserving whole-cue preview and structured risk ledger.

    The recovered SRT is expected in ORIGINAL source coordinates, not
    relative-to-window. The full original source is always the clock authority.

    The owner of the prefix and suffix is the unchanged historical hypothesis.
    The retry owns the center by cue temporal midpoint. A cross-boundary cue
    is never truncated, silently merged or replaced word-by-word.
    """
    start,end=repaired_audio_window_ms
    if not (original_duration_ms>0 and 0<=start<end<=original_duration_ms):
        raise ValueError("Retry window exceeds original-audio clock")
    base=parse_srt(baseline_srt)
    retry=parse_srt(repaired_srt)
    _validate(base,original_duration_ms,policy.max_out_of_audio_rounding_ms,"baseline")
    overhang=_validate(retry,original_duration_ms,
                        policy.max_out_of_audio_rounding_ms,"recovery")
    # Only cues whose midpoint belongs to the retry window may be used.
    selected_repair=[]
    ignored_retry=[]
    for c in retry:
        midpoint=(c.start_ms+c.end_ms)//2
        if start<=midpoint<=end:
            selected_repair.append(c)
        else:
            ignored_retry.append(c.index)
    if not selected_repair:
        raise ValueError("No recovery captions assigned to selected retry window")
    prefix=[];suffix=[];discarded=[]
    for c in base:
        midpoint=(c.start_ms+c.end_ms)//2
        if midpoint<start:
            prefix.append(c)
        elif midpoint>=end:
            suffix.append(c)
        else:
            discarded.append(c)
    selected=[(c,"baseline_prefix") for c in prefix]+[
        (c,"recovery") for c in selected_repair]+[(c,"baseline_suffix") for c in suffix]
    selected.sort(key=lambda x:(x[0].start_ms,x[0].end_ms,x[1]))
    hazards=[]
    def flag(code:str,bounds:tuple[int,int],severity:str="review_required",**metadata:Any)->None:
        hazards.append({"code":code,"interval_ms":list(bounds),
                        "severity":severity,**metadata})
    if overhang:
        flag("RECOVERED_NATIVE_TIMESTAMP_ROUNDING",(original_duration_ms,
             max(c.end_ms for c in selected_repair)),
             "rounding_review",count=len(overhang))
    for c in prefix:
        if c.end_ms>start:
            flag("BASELINE_PREFIX_CUE_CROSSES_RETRY_START",(start,c.end_ms),
                 source_cue_index=c.index)
    for c in selected_repair:
        if c.start_ms<start:
            flag("RECOVERY_CUE_CROSSES_RETRY_START",(c.start_ms,start),
                 "rounding_review",source_cue_index=c.index)
        if c.end_ms>end:
            flag("RECOVERY_CUE_CROSSES_RETRY_END",(end,c.end_ms),
                 "rounding_review",source_cue_index=c.index)
    for c in suffix:
        if c.start_ms<end:
            flag("BASELINE_SUFFIX_CUE_CROSSES_RETRY_END",(c.start_ms,end),
                 source_cue_index=c.index)
    # Scan *all* overlapping cross-source cue pairs, not only adjacent cues.
    # A long baseline caption may overlap two or more shorter recovery cues;
    # an adjacency-only inspection can miss the second disputed hypothesis.
    for i,(a,s0) in enumerate(selected):
        for b,s1 in selected[i+1:]:
            if b.start_ms >= a.end_ms:
                break
            if s0==s1:
                continue
            overlap=max(0,min(a.end_ms,b.end_ms)-max(a.start_ms,b.start_ms))
            if overlap>policy.max_cross_source_overlap_ms:
                flag("CROSS_SOURCE_DIFFERENT_HYPOTHESES_OVERLAP",
                     (max(a.start_ms,b.start_ms),min(a.end_ms,b.end_ms)),
                     overlap_ms=overlap,sources=[s0,s1],
                     source_cue_indices=[a.index,b.index])
    for (a,s0),(b,s1) in zip(selected,selected[1:]):
        if s0==s1:
            continue
        gap=max(0,b.start_ms-a.end_ms)
        if gap>=policy.min_gap_to_review_ms:
            flag("CROSS_SOURCE_CAPTION_GAP",(a.end_ms,b.start_ms),
                 gap_ms=gap,sources=[s0,s1])
    if any(c.start_ms<start or c.end_ms>end for c in discarded):
        crossed=[c for c in discarded if c.start_ms<start or c.end_ms>end]
        for c in crossed:
            flag("DISCARDED_BASELINE_CUE_CROSSES_RETRY_REGION",
                 (c.start_ms,c.end_ms),source_cue_index=c.index)
    provenance=[]
    text="".join(
        f"{i}\n{fmt(c.start_ms)} --> {fmt(c.end_ms)}\n{c.text}\n\n"
        for i,(c,source) in enumerate(selected,1)
    )
    diag=diagnose_srt(text)
    if diag.invalid_timing_cues or diag.backwards_start_cues:
        raise ValueError("Preview has invalid or regressing SRT timestamps")
    for i,(c,source) in enumerate(selected,1):
        provenance.append({
            "output_cue_index":i,"source":source,
            "source_cue_index":c.index,"start_ms":c.start_ms,
            "end_ms":c.end_ms,"source_text_sha256":_sha(c.text)
        })
    acoustic=[]
    if audio is not None:
        rois=[(max(0,start-policy.context_guard_ms),
               min(original_duration_ms,start+policy.context_guard_ms))]
        if end < original_duration_ms:
            rois.append((max(0,end-policy.context_guard_ms),
                         min(original_duration_ms,end+policy.context_guard_ms)))
        for roi in rois:
            if roi[1]-roi[0]<100:
                continue
            for threshold in policy.coverage_dbfs:
                acoustic.append(region_energy_coverage(
                    audio,text,roi[0],roi[1],CoveragePolicy(energy_dbfs=threshold)))
    status=("PROVISIONAL_REVIEW_REQUIRED" if hazards else
            "STRUCTURALLY_COMPOSED_UNVERIFIED")
    report={
       "schema":"whispercpp/conservative-full-timeline-assembly/v1",
       "status":status,
       "source_window_ownership_ms":{
           "original_prefix":[0,start],"recovery":[start,end],
           "original_suffix":[end,original_duration_ms]
       },
       "original_source_duration_ms":original_duration_ms,
       "baseline_input_srt_sha256":_sha(baseline_srt),
       "recovery_input_srt_sha256":_sha(repaired_srt),
       "output_srt_sha256":_sha(text),
       "baseline_cue_count":len(base),
       "recovery_candidate_cue_count":len(retry),
       "output_cue_count":len(selected),
       "unchanged_baseline_prefix_cues":len(prefix),
       "unchanged_baseline_suffix_cues":len(suffix),
       "recovery_cues":len(selected_repair),
       "discarded_baseline_cue_count":len(discarded),
       "excluded_outside_window_recovery_cue_indices":ignored_retry,
       "existing_source_caption_texts_retained_verbatim":True,
       "source_clock_window_coverage_full":True,
       "cue_source_provenance":provenance,
       "hazards":hazards,
       "hazards_count":len(hazards),
       "repeated_caption_run_count":len(diag.candidates),
       "acoustic_energy_proxy_near_boundaries":acoustic,
       "human_audio_verification":False,
       "recognition_accuracy_not_measured":True,
       "source_clock_coverage_not_speech_coverage":True,
       "do_not_auto_publish_or_overwrite_original":True,
    }
    return text,report


def write_full_timeline_preview(
    baseline_path:Path,recovery_path:Path,destination_srt:Path,
    destination_manifest:Path,*,source_audio:Path,
    repaired_audio_window_ms:tuple[int,int],
    policy:TimelinePolicy=TimelinePolicy()
)->dict[str,Any]:
    """Local file wrapper, output collision denied and input hashes recorded."""
    if destination_srt.exists() or destination_manifest.exists():
        raise FileExistsError("No overwriting of preview or manifest")
    import wave
    with wave.open(str(source_audio),"rb") as w:
        if (w.getframerate(),w.getnchannels(),w.getsampwidth(),
            w.getcomptype())!=(16000,1,2,"NONE"):
            raise ValueError("Expected original 16k mono PCM16 WAV")
        duration_ms=round(1000*w.getnframes()/w.getframerate())
    baseline=baseline_path.read_text(encoding="utf-8-sig",errors="strict")
    retry=recovery_path.read_text(encoding="utf-8-sig",errors="strict")
    srt,manifest=assemble_full_timeline(
        baseline,retry,original_duration_ms=duration_ms,
        repaired_audio_window_ms=repaired_audio_window_ms,
        audio=source_audio,policy=policy)
    manifest["input_file_sha256"]={
      "original_audio":_hash_file(source_audio),
      "baseline_srt":_hash_file(baseline_path),
      "recovery_srt":_hash_file(recovery_path)
    }
    destination_srt.write_text(srt,encoding="utf-8")
    destination_manifest.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",
                                    encoding="utf-8")
    return manifest
