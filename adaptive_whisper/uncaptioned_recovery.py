"""Reference-free *candidate* detection and review-only bounded ASR retries.

Energy without captions is not proof of speech. This module never uses
human-transcribed words to find or reconstruct gaps. Independent native
redecodes may provide competing text hypotheses, not ground truth.
All outputs remain experimental and must not automatically replace SRT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import math
import re
import unicodedata
from typing import Any

from .coverage import CoveragePolicy, energy_mask
from .diagnostics import parse_srt
from .stitch import fmt


@dataclass(frozen=True)
class GapSearchPolicy:
    frame_ms: int = 100
    thresholds_dbfs: tuple[float, ...] = (-48.,-42.,-36.)
    min_active_ms: int = 500
    bridge_gap_ms: int = 200
    cue_pad_ms: int = 250
    min_uncovered_ms: int = 1200
    min_threshold_support: int = 2
    max_candidates: int = 15

    def __post_init__(self):
        if (type(self.frame_ms) is not int or self.frame_ms<10 or
                1000%self.frame_ms or
                not self.thresholds_dbfs or
                len(set(self.thresholds_dbfs))!=len(self.thresholds_dbfs) or
                not all(-90<=x<=0 for x in self.thresholds_dbfs) or
                self.min_active_ms<self.frame_ms or
                self.min_uncovered_ms<self.frame_ms or
                self.bridge_gap_ms<0 or self.cue_pad_ms<0 or
                not (1<=self.min_threshold_support<=len(self.thresholds_dbfs)) or
                self.max_candidates<1):
            raise ValueError("Invalid uncaptioned-activity search policy")


def _runs(mask: list[bool], frame_ms: int) -> list[tuple[int,int]]:
    result=[]
    first=None
    for i,active in enumerate(mask+[False]):
        if active and first is None:first=i
        if not active and first is not None:
            result.append((first*frame_ms,i*frame_ms))
            first=None
    return result


def find_uncaptioned_activity(
    audio: Path,
    baseline_srt: str,
    policy: GapSearchPolicy = GapSearchPolicy(),
) -> dict[str, Any]:
    """Return prioritized audio-energy/subtitle-gap *suspicions*, not speech.

    The fixed multithreshold policy is not calibrated on human review.
    Silence, music, collisions, microphone clicks and noise can trigger it.
    """
    cues=parse_srt(baseline_srt)
    if baseline_srt.strip() and not cues:
        raise ValueError("Malformed nonempty baseline SRT")
    masks=[]
    lengths=[]
    for threshold in policy.thresholds_dbfs:
        mask,duration=energy_mask(
            audio,CoveragePolicy(
                frame_ms=policy.frame_ms,energy_dbfs=threshold,
                min_active_ms=policy.min_active_ms,
                bridge_gap_ms=policy.bridge_gap_ms,
                cue_pad_ms=policy.cue_pad_ms,
                long_uncovered_ms=policy.min_uncovered_ms))
        masks.append(mask);lengths.append(duration)
    if len(set(lengths))!=1 or len(set(map(len,masks)))!=1:
        raise RuntimeError("Inconsistent PCM masks")
    duration=lengths[0]
    n=len(masks[0])
    covered=[False]*n
    for cue in cues:
        if (cue.start_ms<0 or cue.end_ms<=cue.start_ms or
                cue.end_ms>duration+100):
            raise ValueError("Out-of-source baseline cue")
        left=max(0,(cue.start_ms-policy.cue_pad_ms)//policy.frame_ms)
        right=min(n,math.ceil((cue.end_ms+policy.cue_pad_ms)/policy.frame_ms))
        covered[left:right]=[True]*(right-left)
    support=[sum(bool(mask[i]) for mask in masks) for i in range(n)]
    uncovered=[support[i]>=policy.min_threshold_support and not covered[i]
               for i in range(n)]
    candidates=[]
    for start,end in _runs(uncovered,policy.frame_ms):
        end=min(end,duration)
        if end-start<policy.min_uncovered_ms:continue
        a=start//policy.frame_ms
        b=math.ceil(end/policy.frame_ms)
        candidate={
            "source_interval_ms":[start,end],
            "uncaptioned_activity_ms":end-start,
            "minimum_threshold_support":min(support[a:b]),
            "max_threshold_support":max(support[a:b]),
            "source_clip_edge_uncertain":start==0 or end==duration,
            "candidate_kind":"UNCAPTIONED_AUDIO_ACTIVITY_NOT_SPEECH_TRUTH",
            "audio_speech_presence_unverified":True,
            "machine_word_omission_unverified":True,
        }
        candidates.append(candidate)
    # Rank by uncovered duration then time, independent of human transcript.
    candidates.sort(key=lambda row:(-row["uncaptioned_activity_ms"],
                                   row["source_interval_ms"][0]))
    count=len(candidates)
    return {
        "schema":"whispercpp/reference-free-multithreshold-gap-detector/v1",
        "source_duration_ms":duration,
        "baseline_cue_count":len(cues),
        "policy":asdict(policy),
        "candidate_count_total":count,
        "candidate_count_returned":min(count,policy.max_candidates),
        "candidate_truncated":count>policy.max_candidates,
        "candidates":candidates[:policy.max_candidates],
        "high_energy_frames_by_threshold":[sum(x) for x in masks],
        "candidate_is_not_speech_ground_truth":True,
        "human_words_not_used":True,
        "auto_repair_authorized":False,
        "auto_release_authorized":False,
    }


def plan_dual_gap_windows(
    candidate: dict[str,Any],
    audio_duration_ms: int,
    context_padding_ms: tuple[int,int]=(4000,12000),
) -> tuple[dict[str,Any],...]:
    """Compute two distinct source-clock ASR contexts for review only."""
    if len(context_padding_ms)!=2 or not(0<context_padding_ms[0]<context_padding_ms[1]):
        raise ValueError("Expected two strictly increasing positive contexts")
    a,b=candidate.get("source_interval_ms",[None,None])
    if (type(a) is not int or type(b) is not int
            or type(audio_duration_ms) is not int or
            not (0<=a<b<=audio_duration_ms)):
        raise ValueError("Invalid uncaptioned audio candidate")
    jobs=[]
    for label,pad in zip(("short","long"),context_padding_ms):
        begin=max(0,a-pad)
        stop=min(audio_duration_ms,b+pad)
        if stop-begin<1000:raise ValueError("Native window under 1000ms")
        jobs.append({
            "label":label,"source_interval_ms":[begin,stop],
            "gap_interval_ms":[a,b],
            "duration_ms":stop-begin,
            "audio_context_padding_ms":pad,
            "no_human_word_truth":True,
        })
    if jobs[0]["source_interval_ms"]==jobs[1]["source_interval_ms"]:
        raise ValueError("Two clipped retries would be identical")
    return tuple(jobs)


def _tokens(cues)->list[str]:
    return [unicodedata.normalize("NFKC",x).casefold()
            for cue in cues for x in re.findall(r"\w+",cue.text,flags=re.UNICODE)]


def machine_gap_consensus(
    original_srt: str,
    candidate: dict[str,Any],
    retry_srt_by_label: dict[str,str],
    retry_source_start_ms: dict[str,int],
    *,
    original_duration_ms:int,
) -> dict[str,Any]:
    """Strict optional *suggestion*: two machine hypotheses agree exactly.

    Only cues entirely inside the previously caption-free gap qualify.
    Uncertainties, boundary-crossing cues, divergent outputs, or no text
    cause abstention. Even exact machine agreement is not human truth.
    """
    a,b=candidate.get("source_interval_ms",[None,None])
    if (type(a) is not int or type(b) is not int
            or not(0<=a<b<=original_duration_ms)):
        raise ValueError("Invalid candidate time")
    if set(retry_srt_by_label)!={"short","long"} or set(retry_source_start_ms)!={"short","long"}:
        raise ValueError("Expected the two pre-registered native context labels")
    original=parse_srt(original_srt)
    if original_srt.strip() and not original:
        raise ValueError("Malformed original")
    if any(c.start_ms<b and c.end_ms>a for c in original):
        raise ValueError("Original candidate is not an empty-caption interval")
    results={}
    for label in ("short","long"):
        start=retry_source_start_ms[label]
        if type(start) is not int or start<0 or start>a:
            raise ValueError("Bad retry source-clock origin")
        raw=retry_srt_by_label[label]
        cues=parse_srt(raw)
        if raw.strip() and not cues:
            raise ValueError("Malformed native retry SRT")
        if any(c.start_ms<0 or c.end_ms<=c.start_ms or start+c.end_ms>original_duration_ms+100
               for c in cues):
            raise ValueError("Native retry source clock invalid")
        overlapping=[c for c in cues if start+c.start_ms<b and start+c.end_ms>a]
        crossing=[c for c in overlapping
                  if start+c.start_ms<a or start+c.end_ms>b]
        inside=[c for c in overlapping
                if start+c.start_ms>=a and start+c.end_ms<=b]
        tokens=_tokens(inside)
        results[label]={
            "native_intersecting_cue_count":len(overlapping),
            "native_boundary_crossing_cues":len(crossing),
            "native_strictly_inside_cue_count":len(inside),
            "candidate_word_tokens":tokens,
            "inside_source_intervals_ms":[[start+c.start_ms,start+c.end_ms]
                                          for c in inside],
            "source_start_ms":start,
        }
    t0=results["short"]["candidate_word_tokens"]
    t1=results["long"]["candidate_word_tokens"]
    consistent=(bool(t0) and t0==t1 and
                all(not results[label]["native_boundary_crossing_cues"]
                    for label in ("short","long")))
    if consistent:
        # Keep only the original SRT punctuation/case as untrusted text.
        # It may be entirely inaccurate, or missing split speech.
        from .diagnostics import diagnose_srt
        inside0=[
            c for c in parse_srt(retry_srt_by_label["short"])
            if (results["short"]["source_start_ms"]+c.start_ms>=a and
                results["short"]["source_start_ms"]+c.end_ms<=b)
        ]
        if len(diagnose_srt(retry_srt_by_label["short"]).candidates)>0:
            consistent=False
        candidate_text=" ".join(c.text for c in inside0) if consistent else None
    else:
        candidate_text=None
    return {
        "schema":"whispercpp/two-context-machine-gap-consensus/v1",
        "source_gap_ms":[a,b],
        "machine_consensus_token_sequence_identical":consistent,
        "candidate_text_for_human_review_only":candidate_text,
        "candidate_words_count":len(t0) if consistent else 0,
        "per_retry":{
            k:{key:value for key,value in item.items() if key!="candidate_word_tokens"}
            for k,item in results.items()
        },
        "result_status":"CONSISTENT_MACHINE_TEXT_UNVERIFIED"
                        if consistent else "ABSTAIN_UNRESOLVED_MACHINE_WORDS",
        "original_caption_preserved":True,
        "human_word_accuracy_not_established":True,
        "automatic_release_authorized":False,
        "automatic_full_timeline_splice_authorized":False,
    }



def caption_gap_envelope(
    baseline_srt: str,
    activity_candidate: dict[str,Any],
    *,
    source_duration_ms: int,
    max_gap_ms: int = 20000,
) -> dict[str,Any]:
    """Expand detected sound to nearest unchanged SRT cue boundaries.

    No human words, no missing-word estimates. This prevents dropping
    potentially audible words at the edges of the energy-only core.
    A candidate inside a caption, or a huge/ambiguous gap, is refused.
    """
    if type(source_duration_ms) is not int or source_duration_ms<1:
        raise ValueError("Bad source duration")
    if type(max_gap_ms) is not int or max_gap_ms<1000:
        raise ValueError("Bad maximum caption gap")
    core=activity_candidate.get("source_interval_ms")
    if (not isinstance(core,(tuple,list)) or len(core)!=2 or
            any(type(x) is not int for x in core)):
        raise ValueError("Bad audio-energy candidate")
    a,b=core
    if not (0<=a<b<=source_duration_ms):
        raise ValueError("Energy candidate outside source")
    cues=parse_srt(baseline_srt)
    if baseline_srt.strip() and not cues:
        raise ValueError("Malformed baseline")
    if any(c.start_ms<b and c.end_ms>a for c in cues):
        raise ValueError("Candidate overlaps pre-existing subtitle")
    before=[c.end_ms for c in cues if c.end_ms<=a]
    after=[c.start_ms for c in cues if c.start_ms>=b]
    left=max(before,default=0)
    right=min(after,default=source_duration_ms)
    if not (0<=left<=a<b<=right<=source_duration_ms and right-left<=max_gap_ms):
        raise ValueError("Unbounded or invalid empty caption gap")
    return {
        "schema":"whispercpp/cue-fenced-uncaptioned-candidate/v1",
        "activity_core_ms":[a,b],
        "source_interval_ms":[left,right],
        "original_caption_gap_ms":right-left,
        "expansion_based_only_on_unchanged_srt_neighbors":True,
        "word_presence_and_accuracy_unverified":True,
        "human_reference_accessed":False,
        "automatic_release_authorized":False,
    }



def machine_gap_fragment_proposals(
    original_srt: str,
    candidate: dict[str,Any],
    retry_srt_by_label: dict[str,str],
    retry_source_start_ms: dict[str,int],
    *,
    original_duration_ms:int,
) -> dict[str,Any]:
    """Return private preview fragments from *fully owned* native SRT cues.

    In the absence of two-context consensus, one owned cue can still be
    shown for HUMAN review, labelled unverified. Crossing cues are never
    trimmed or placed into the gap by guesswork. No full SRT is repaired.
    """
    a,b=candidate.get("source_interval_ms",[None,None])
    if (type(a) is not int or type(b) is not int
            or not (0<=a<b<=original_duration_ms)):
        raise ValueError("Bad gap source clock")
    existing=parse_srt(original_srt)
    if original_srt.strip() and not existing:
        raise ValueError("Malformed original SRT")
    if any(c.start_ms<b and c.end_ms>a for c in existing):
        raise ValueError("Original gap is not empty")
    if set(retry_srt_by_label)!=set(("short","long")) or set(retry_source_start_ms)!=set(("short","long")):
        raise ValueError("Two named native contexts required")
    descriptions={}
    fragments={}
    for label in ("short","long"):
        start=retry_source_start_ms[label]
        if type(start) is not int or start<0 or start>a:
            raise ValueError("Invalid native context source start")
        source=retry_srt_by_label[label]
        cues=parse_srt(source)
        if source.strip() and not cues:
            raise ValueError("Malformed native SRT")
        if any(c.start_ms<0 or c.end_ms<=c.start_ms or
               start+c.end_ms>original_duration_ms+100 for c in cues):
            raise ValueError("Native cue falls outside source clock")
        crossing=[c for c in cues if start+c.start_ms<b and
                  start+c.end_ms>a and (start+c.start_ms<a or
                                       start+c.end_ms>b)]
        inside=[c for c in cues if start+c.start_ms>=a and
                start+c.end_ms<=b]
        fragments[label]="".join(
            f"{i}\n{fmt(start+c.start_ms)} --> {fmt(start+c.end_ms)}\n{c.text}\n\n"
            for i,c in enumerate(inside,1))
        descriptions[label]={
            "strictly_owned_cues":len(inside),
            "strictly_owned_word_tokens":len(_tokens(inside)),
            "crossing_cues_excluded_from_preview":len(crossing),
            "native_source_start_ms":start,
            "private_fragment_sha256":hashlib.sha256(
                fragments[label].encode("utf-8")).hexdigest(),
            "candidate_incomplete_when_crossing_cues_present":bool(crossing),
        }
    return {
        "schema":"whispercpp/private-review-only-machine-gap-fragments/v1",
        "source_gap_ms":[a,b],
        "fragments_by_context":fragments,
        "candidate_diagnostics":descriptions,
        "any_fragment_available":any(fragments.values()),
        "word_identity_and_acoustic_timing_not_human_verified":True,
        "no_automatic_full_timeline_splice":True,
        "auto_release_authorized":False,
    }
