"""Conservative timestamp-preserving multiwindow SRT assembly.

This generates an EXPERIMENTAL PRIVATE PREVIEW, never a ground-truth transcript.
Conflict and source-origin ledgers are mandatory, not optional.
"""
from __future__ import annotations
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
import unicodedata
from .diagnostics import Cue,parse_srt
from .recovery import RetryWindow

@dataclass(frozen=True)
class SourceCue:
    start_ms: int
    end_ms: int
    text: str
    source_job: int

@dataclass(frozen=True)
class StitchReport:
    status: str
    text_srt: str
    segments: tuple[SourceCue,...]
    conflicts: tuple[dict,...]
    boundaries: tuple[dict,...]
    selection_count: int
    discarded_outside_ownership: int

def norm(s:str)->str:
    return " ".join(unicodedata.normalize("NFKC",s).casefold().split())

def fmt(ms:int)->str:
    if ms<0:raise ValueError("Negative SRT time")
    h,rem=divmod(ms,3600000)
    m,rem=divmod(rem,60000)
    s,ms=divmod(rem,1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def merge_windows(windows:tuple[RetryWindow,...], native_srts:tuple[str,...],
                  max_cross_window_overlap_ms:int=750,
                  cut_strategy:str="midpoint")->StitchReport:
    if not windows or len(windows)!=len(native_srts):
        raise ValueError("Need one SRT per ordered window")
    if tuple(w.job for w in windows)!=tuple(range(1,len(windows)+1)):
        raise ValueError("Windows must be consecutively numbered 1..N")
    if max_cross_window_overlap_ms<0:raise ValueError("Invalid overlap tolerance")
    if cut_strategy not in ("midpoint","joint_caption_gap"):
        raise ValueError("Invalid ownership cut strategy")
    for i,w in enumerate(windows):
        if w.duration_ms<=0 or w.global_start_ms<0:
            raise ValueError("Invalid window")
        if i and not (windows[i-1].global_start_ms < w.global_start_ms
                    < windows[i-1].global_end_ms < w.global_end_ms):
            raise ValueError("Windows must progress with positive adjacent overlap")
    boundaries=[]
    for left,right in zip(windows,windows[1:]):
        cut=(left.global_end_ms+right.global_start_ms)//2
        boundaries.append({
            "left_job":left.job,"right_job":right.job,
            "overlap_start_ms":right.global_start_ms,
            "overlap_end_ms":left.global_end_ms,
            "overlap_ms":left.global_end_ms-right.global_start_ms,
            "ownership_cut_ms":cut
        })
    all_cues=[]
    for w,raw in zip(windows,native_srts):
        cues=parse_srt(raw)
        absolute=[]
        previous=-1
        for cue in cues:
            if cue.start_ms<0 or cue.end_ms<=cue.start_ms or cue.end_ms>w.duration_ms+1000:
                raise ValueError(f"Window {w.job} invalid/out-of-bounds native cue")
            if cue.start_ms<previous:raise ValueError(f"Window {w.job} out-of-order cue")
            previous=cue.start_ms
            absolute.append(SourceCue(w.global_start_ms+cue.start_ms,
                                      w.global_start_ms+cue.end_ms,
                                      cue.text,w.job))
        all_cues.append(absolute)
    # Prefer a handoff inside a true *joint gap in ASR cue coverage* when one
    # exists, rather than cutting in the middle of an utterance. Such a gap is
    # NOT evidence of silence without waveform inspection.
    if cut_strategy=="joint_caption_gap":
        for b in boundaries:
            left=all_cues[b["left_job"]-1]
            right=all_cues[b["right_job"]-1]
            start,end=b["overlap_start_ms"],b["overlap_end_ms"]
            center=(start+end)//2
            guard=min(3000,(end-start)//4)
            start_allowed,end_allowed=start+guard,end-guard
            occupied=[]
            for cue in left+right:
                a=max(start_allowed,cue.start_ms)
                z=min(end_allowed,cue.end_ms)
                if a<z:occupied.append((a,z))
            occupied.sort()
            gaps=[]
            cursor=start_allowed
            for a,z in occupied:
                if a>cursor:gaps.append((cursor,a))
                cursor=max(cursor,z)
            if cursor<end_allowed:gaps.append((cursor,end_allowed))
            permitted=[g for g in gaps if g[1]-g[0]>=800]
            if permitted:
                # Use the closest joint gap to the window-overlap midpoint.
                chosen=min(permitted,key=lambda g:(abs((g[0]+g[1])//2-center),-(g[1]-g[0])))
                b["ownership_cut_ms"]=(chosen[0]+chosen[1])//2
                b["cut_method"]="joint_caption_gap"
                b["joint_caption_gap_ms"]=chosen[1]-chosen[0]
            else:
                b["cut_method"]="fallback_midpoint_NO_JOINT_GAP"
                b["joint_caption_gap_ms"]=0
    selected=[]
    ignored=0
    for i,(w,cues) in enumerate(zip(windows,all_cues)):
        low=boundaries[i-1]["ownership_cut_ms"] if i else w.global_start_ms
        high=boundaries[i]["ownership_cut_ms"] if i<len(boundaries) else w.global_end_ms
        for cue in cues:
            # Deterministic responsibility by temporal midpoint.
            midpoint=(cue.start_ms+cue.end_ms)//2
            if low<=midpoint<high or (i==len(windows)-1 and midpoint==high):
                selected.append(cue)
            else:
                ignored+=1
    selected.sort(key=lambda c:(c.start_ms,c.end_ms,c.source_job))
    conflicts=[]
    dedup=[]
    for cue in selected:
        if dedup:
            last=dedup[-1]
            shared=max(0,min(cue.end_ms,last.end_ms)-max(cue.start_ms,last.start_ms))
            minspan=min(cue.end_ms-cue.start_ms,last.end_ms-last.start_ms)
            if shared>0 and norm(cue.text)==norm(last.text) and shared/minspan>=.6:
                # Record exact duplicate removal; do not repeat it in final SRT.
                conflicts.append({
                    "kind":"DUPLICATE_CUE_SUPPRESSED",
                    "jobs":[last.source_job,cue.source_job],
                    "start_ms":min(last.start_ms,cue.start_ms),
                    "end_ms":max(last.end_ms,cue.end_ms)
                })
                continue
            if cue.source_job!=last.source_job and shared>max_cross_window_overlap_ms:
                conflicts.append({
                    "kind":"CROSS_WINDOW_OVERLAPPING_DIFFERENT_CUES",
                    "jobs":[last.source_job,cue.source_job],
                    "start_ms":max(last.start_ms,cue.start_ms),
                    "end_ms":min(last.end_ms,cue.end_ms),
                    "overlap_ms":shared
                })
        dedup.append(cue)
    # Independently inspect competing source hypotheses around each handoff cut.
    for b in boundaries:
        cut=b["ownership_cut_ms"]
        a=all_cues[b["left_job"]-1]
        c=all_cues[b["right_job"]-1]
        # For a joint-gap join, inspect only the unoccupied area around
        # the selected cut. The gap itself is the evidence for this handoff.
        band= min(350,b["joint_caption_gap_ms"]//4) if b.get("cut_method")=="joint_caption_gap" else 1500
        band_start=cut-band
        band_end=cut+band
        left=[x for x in a if x.start_ms<band_end and x.end_ms>band_start]
        right=[x for x in c if x.start_ms<band_end and x.end_ms>band_start]
        ratios=[SequenceMatcher(None,norm(x.text),norm(y.text),autojunk=False).ratio()
                for x in left for y in right]
        b["local_left_cues"]=len(left)
        b["local_right_cues"]=len(right)
        b["max_hypothesis_similarity"]=round(max(ratios),4) if ratios else None
        if left and right and (not ratios or max(ratios)<.75):
            conflicts.append({
                "kind":"DIVERGENT_BOUNDARY_HYPOTHESES",
                "jobs":[b["left_job"],b["right_job"]],
                "cut_ms":cut,"best_similarity":round(max(ratios),4) if ratios else None
            })
        elif bool(left)!=bool(right):
            conflicts.append({
                "kind":"UNILATERAL_SPEECH_AT_BOUNDARY",
                "jobs":[b["left_job"],b["right_job"]],"cut_ms":cut
            })
    # Do not falsely assert continuous speech or word correctness from window coverage.
    problems=[x for x in conflicts if x["kind"]!="DUPLICATE_CUE_SUPPRESSED"]
    status="PROVISIONAL_REVIEW_REQUIRED" if problems else "STRUCTURALLY_STITCHED_UNVERIFIED"
    srt="".join(f"{idx}\n{fmt(c.start_ms)} --> {fmt(c.end_ms)}\n{c.text}\n\n"
                for idx,c in enumerate(dedup,start=1))
    return StitchReport(status,srt,tuple(dedup),tuple(conflicts),
                        tuple(boundaries),len(selected),ignored)
