"""Backend-neutral, reference-free audio/SRT *coverage proxies*.

PCM intensity is not a speech detector. This reports sound-energy intervals
without nearby subtitle timestamps; it cannot prove omissions or WER.
Preserves source files and emits no transcript text.
"""
from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import sys
import wave

from .diagnostics import Cue,parse_srt


@dataclass(frozen=True)
class CoveragePolicy:
    frame_ms: int = 100
    energy_dbfs: float = -42.0
    min_active_ms: int = 300
    bridge_gap_ms: int = 200
    cue_pad_ms: int = 250
    long_uncovered_ms: int = 1000

    def __post_init__(self) -> None:
        if (self.frame_ms <= 0 or 1000%self.frame_ms
                or self.min_active_ms < self.frame_ms
                or self.bridge_gap_ms<0 or self.cue_pad_ms<0
                or self.long_uncovered_ms<self.frame_ms
                or not -90.0 <= self.energy_dbfs <= 0.0):
            raise ValueError("Invalid coverage policy")


def _intervals(mask:list[bool],frame_ms:int)->list[tuple[int,int]]:
    result=[]
    start=None
    for i,yes in enumerate(mask+[False]):
        if yes and start is None:
            start=i
        elif not yes and start is not None:
            result.append((start*frame_ms,i*frame_ms))
            start=None
    return result


def energy_mask(pcm16_wav:Path,policy:CoveragePolicy)->tuple[list[bool],int]:
    """Return fixed-width RMS energy mask, streaming and standard-library only."""
    with wave.open(str(pcm16_wav),"rb") as src:
        if (src.getnchannels(),src.getsampwidth(),
            src.getframerate(),src.getcomptype()) != (1,2,16000,"NONE"):
            raise ValueError("Expected mono 16 kHz PCM16 WAV")
        frame_samples=16*policy.frame_ms
        total=src.getnframes()
        result=[]
        limit=10.0 ** (policy.energy_dbfs/20.0) * 32768.0
        limit2=limit*limit
        while True:
            payload=src.readframes(frame_samples)
            if not payload:
                break
            values=array("h")
            values.frombytes(payload)
            if sys.byteorder!="little":
                values.byteswap()
            if not values:
                break
            power=sum(x*x for x in values)/len(values)
            result.append(power>=limit2)
        exact_ms=round(total*1000/16000)
    # Merge subthreshold gaps with activity on both sides to avoid subword
    # holes, then reject isolated sounds shorter than the minimum length.
    maxgap=policy.bridge_gap_ms//policy.frame_ms
    if maxgap:
        active=_intervals(result,policy.frame_ms)
        for (_,old_end),(next_start,_) in zip(active,active[1:]):
            if 0<next_start-old_end<=policy.bridge_gap_ms:
                a=old_end//policy.frame_ms
                b=next_start//policy.frame_ms
                result[a:b]=[True]*(b-a)
    for a,b in _intervals(result,policy.frame_ms):
        if b-a<policy.min_active_ms:
            result[a//policy.frame_ms:b//policy.frame_ms]=[False]*((b-a)//policy.frame_ms)
    return result,exact_ms


def coverage_report(audio:Path,srt_text:str,policy:CoveragePolicy=CoveragePolicy())->dict:
    """SRT coverage of high-energy frames (not independently verified speech)."""
    mask,duration=energy_mask(audio,policy)
    cues=parse_srt(srt_text)
    n=len(mask)
    covered=[False]*n
    invalid=0
    overlaps=0
    for i,c in enumerate(cues):
        if c.end_ms<c.start_ms or c.start_ms<0 or c.end_ms>duration+policy.frame_ms:
            invalid+=1
            continue
        if i and c.start_ms<cues[i-1].start_ms:
            invalid+=1
        if i and c.start_ms<cues[i-1].end_ms:
            overlaps+=1
        a=max(0,(c.start_ms-policy.cue_pad_ms)//policy.frame_ms)
        b=min(n,math.ceil((c.end_ms+policy.cue_pad_ms)/policy.frame_ms))
        covered[a:b]=[True]*(b-a)
    active_count=sum(mask)
    supported_count=sum(active and covered[i] for i,active in enumerate(mask))
    missing=[x and not covered[i] for i,x in enumerate(mask)]
    uncaptioned=_intervals(missing,policy.frame_ms)
    longest=max((b-a for a,b in uncaptioned),default=0)
    long=[(a,min(b,duration)) for a,b in uncaptioned
          if b-a>=policy.long_uncovered_ms]
    # Strict activity over captions = *not* absence of words; a music clip can
    # be energetic throughout and correctly yield no speech transcript.
    return {
        "schema":"whispercpp/audio-energy-srt-coverage-proxy/v1",
        "not_ground_truth":True,
        "cannot_certify_no_speech_omission":True,
        "method":"PCM RMS sound-energy per frame, cue timestamps padded; no audio ASR",
        "policy":asdict(policy),
        "duration_ms":duration,
        "frame_count":n,
        "cue_count":len(cues),
        "invalid_or_nonmonotone_cue_count":invalid,
        "overlapping_cue_count":overlaps,
        "high_energy_duration_ms":active_count*policy.frame_ms,
        "high_energy_in_padded_cue_ms":supported_count*policy.frame_ms,
        "high_energy_not_in_padded_cue_ms":sum(missing)*policy.frame_ms,
        "high_energy_cue_coverage_fraction":round(supported_count/active_count,5)
                                              if active_count else None,
        "longest_uncaptioned_high_energy_ms":longest,
        "uncaptioned_high_energy_intervals_ms":long,
        "uncaptioned_high_energy_intervals_count":len(long),
    }


def compare_machine_hypotheses(baseline_srt:str,candidate_srt:str,
                               *,baseline_pathological:bool=False,
                               low_token_ratio:float=.65,
                               low_bag_f1:float=.55)->dict:
    """Flag major output divergence WITHOUT any reference-transcript truth.

    This is an engineering review gate for pairs of machine outputs.
    A damaged looping baseline must never be used as word accuracy truth.
    """
    import collections
    import re
    from difflib import SequenceMatcher

    if not (0<low_token_ratio<=1 and 0<low_bag_f1<=1):
        raise ValueError("Invalid machine-consistency thresholds")
    base=parse_srt(baseline_srt)
    new=parse_srt(candidate_srt)
    tokens=lambda cues:[x.casefold() for cue in cues
             for x in re.findall(r"\w+",cue.text,flags=re.UNICODE)]
    t0=tokens(base)
    t1=tokens(new)
    a,b=len(t0),len(t1)
    ratio=min(a,b)/max(a,b) if max(a,b) else 1.0
    c0=collections.Counter(t0)
    c1=collections.Counter(t1)
    shared=sum((c0 & c1).values())
    bag_f1=2*shared/(a+b) if a+b else 1.0
    # Avoid quadratic worst-case on long recordings while keeping a
    # reproducible ordering-sensitive score for bounded snippets.
    seq=(round(SequenceMatcher(None,t0,t1,autojunk=False).ratio(),5)
         if a+b<=1500 else None)
    warnings=[]
    if not baseline_pathological and ratio<low_token_ratio:
        warnings.append("token_count_ratio_large_change")
    if not baseline_pathological and bag_f1<low_bag_f1:
        warnings.append("machine_token_bag_large_disagreement")
    return {
      "schema":"whispercpp/machine-hypothesis-consistency/v1",
      "not_reference_truth":True,
      "word_error_rate_unavailable":True,
      "baseline_pathological":baseline_pathological,
      "baseline_cue_count":len(base),
      "candidate_cue_count":len(new),
      "baseline_token_count":a,
      "candidate_token_count":b,
      "shorter_to_longer_token_count_ratio":round(ratio,5),
      "token_multiset_f1_machine_agreement":round(bag_f1,5),
      "ordered_token_similarity_if_small":seq,
      "review_warnings":warnings,
      "no_warning_does_not_certify_completeness":True,
    }


def region_output_diversity(srt_text:str,start_ms:int,end_ms:int)->dict:
    """Output-only caption diversity on a *predefined* original-audio region.

    The region MUST be the same when comparing hypotheses. Repetition-free
    output is not evidence that all speech is present or correctly heard.
    """
    import collections
    import re
    import unicodedata
    if start_ms<0 or end_ms<=start_ms:
        raise ValueError("Invalid original-audio analysis interval")
    cues=tuple(c for c in parse_srt(srt_text)
               if c.end_ms>start_ms and c.start_ms<end_ms)
    keys=[" ".join(unicodedata.normalize("NFKC",c.text).casefold().split())
          for c in cues]
    counts=collections.Counter(keys)
    length=len(keys)
    max_count=max(counts.values(),default=0)
    run_start=0
    longest_run_ms=0
    longest_run_cues=0
    for i in range(1,length+1):
        if i<length and keys[i]==keys[run_start]:
            continue
        run_count=i-run_start
        if run_count>=2:
            duration=(min(end_ms,cues[i-1].end_ms)
                      -max(start_ms,cues[run_start].start_ms))
            if duration>longest_run_ms:
                longest_run_ms=duration
                longest_run_cues=run_count
        run_start=i
    tokens=sum(len(re.findall(r"\w+",c.text,flags=re.UNICODE))
               for c in cues)
    return {
      "schema":"whispercpp/fixed-interval-output-diversity/v1",
      "not_truth":True,
      "source_interval_ms":[start_ms,end_ms],
      "interval_duration_ms":end_ms-start_ms,
      "intersecting_cues":length,
      "token_count":tokens,
      "unique_normalized_caption_count":len(counts),
      "unique_caption_fraction":round(len(counts)/length,5) if length else None,
      "highest_single_caption_fraction":round(max_count/length,5) if length else None,
      "longest_identical_caption_run_ms":longest_run_ms,
      "longest_identical_caption_run_cues":longest_run_cues,
      "all_speech_retained_not_verified":True,
    }


def region_energy_coverage(audio:Path,srt_text:str,start_ms:int,end_ms:int,
                           policy:CoveragePolicy=CoveragePolicy())->dict:
    """Compare an identical, fixed original-audio ROI across ASR variants.

    100-ms frame discretization discards partial ROI-edge bins; it does not
    certify that retained captions contain the correct words.
    """
    mask,duration=energy_mask(audio,policy)
    if start_ms<0 or end_ms>duration or start_ms>=end_ms:
        raise ValueError("Invalid original source ROI")
    first=math.ceil(start_ms/policy.frame_ms)
    last=end_ms//policy.frame_ms
    if first>=last:
        raise ValueError("Source ROI shorter than complete analysis frame")
    cues=parse_srt(srt_text)
    selected=mask[first:last]
    covered=[False]*len(selected)
    for c in cues:
        if c.end_ms<c.start_ms:
            continue
        a=max(first,(c.start_ms-policy.cue_pad_ms)//policy.frame_ms)
        b=min(last,math.ceil((c.end_ms+policy.cue_pad_ms)/policy.frame_ms))
        if b>a:
            covered[a-first:b-first]=[True]*(b-a)
    active=sum(selected)
    intersection=sum(v and covered[i] for i,v in enumerate(selected))
    uncaptioned=[v and not covered[i] for i,v in enumerate(selected)]
    intervals=[(first*policy.frame_ms+a,
                first*policy.frame_ms+b)
               for a,b in _intervals(uncaptioned,policy.frame_ms)
               if b-a>=policy.long_uncovered_ms]
    return {
       "schema":"whispercpp/fixed-interval-energy-coverage-proxy/v1",
       "not_ground_truth":True,
       "source_interval_ms":[start_ms,end_ms],
       "frame_aligned_evaluated_ms":[first*policy.frame_ms,last*policy.frame_ms],
       "frame_edge_time_uncertainty_ms":policy.frame_ms,
       "threshold_dbfs":policy.energy_dbfs,
       "high_energy_ms":active*policy.frame_ms,
       "high_energy_caption_covered_ms":intersection*policy.frame_ms,
       "high_energy_caption_coverage_fraction":round(intersection/active,5) if active else None,
       "uncaptioned_high_energy_ms":sum(uncaptioned)*policy.frame_ms,
       "uncaptioned_intervals_ms":intervals,
       "not_proof_of_speech_retention":True,
    }


def multi_threshold_report(audio:Path,srt_text:str,
                           thresholds:tuple[float,...]=(-48.,-42.,-36.))->dict:
    return {
        "schema":"whispercpp/multi-threshold-energy-coverage/v1",
        "thresholds_dbfs":list(thresholds),
        "measurements":[coverage_report(audio,srt_text,
                    CoveragePolicy(energy_dbfs=t)) for t in thresholds],
        "interpretation":"Sensitivity analysis; cannot distinguish speech from noise or music",
    }
