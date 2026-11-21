"""Experimental error-adaptive recovery controller for whisper.cpp.

Not a general ASR accuracy optimizer, streaming decoder, or final transcript publisher.
The controller detects output-level pathologies, resets context by sliding PCM
windows, and escalates through bounded parameter/backtracking strategies.
It retains explicit source/variant provenance and never replaces a baseline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol
import hashlib
import json
import re
import time
import wave

from .backend import DecodeProfile, ensure_inside, sha256_file
from .diagnostics import diagnose_srt, parse_srt, RepetitionCandidate
from .recovery import RetryPolicy, RetryWindow, plan_retries
from .stitch import merge_windows
from .native_json import write_canonical_separate, NativeJsonIntegrityError
from .watchdog import WatchdogPolicy
from .coverage import coverage_report,compare_machine_hypotheses
from .full_timeline import write_full_timeline_preview


@dataclass(frozen=True)
class Trigger:
    kind: str
    start_ms: int
    end_ms: int
    signal: str
    confidence: str = "output_pattern_not_audio_truth"


@dataclass(frozen=True)
class DetectionPolicy:
    micro_min_chars: int = 12
    micro_min_cues: int = 3
    micro_min_span_ms: int = 4000
    long_min_span_ms: int = 30000
    long_min_cues: int = 5
    within_cue_min_repetitions: int = 3
    within_cue_min_ngram_words: int = 2
    within_cue_max_ngram_words: int = 7

    def __post_init__(self):
        if self.micro_min_cues < 2 or self.micro_min_chars < 1 or self.micro_min_span_ms < 1:
            raise ValueError("Invalid detector policy")
        if self.within_cue_min_repetitions < 2 or self.within_cue_min_ngram_words < 1:
            raise ValueError("Invalid micro-loop policy")


def _intra_cue_loop(text: str, policy: DetectionPolicy) -> bool:
    words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    for n in range(policy.within_cue_min_ngram_words,
                   min(policy.within_cue_max_ngram_words, len(words) // policy.within_cue_min_repetitions)+1):
        for begin in range(len(words)-n*policy.within_cue_min_repetitions+1):
            phrase = words[begin:begin+n]
            if len(" ".join(phrase)) < policy.micro_min_chars:
                continue
            if all(words[begin+k*n:begin+(k+1)*n] == phrase
                   for k in range(policy.within_cue_min_repetitions)):
                return True
    return False


def detect_pathologies(
    srt: str | None,
    audio_duration_ms: int,
    *,
    failure_or_timeout: bool = False,
    policy: DetectionPolicy = DetectionPolicy()
) -> tuple[Trigger, ...]:
    """Only output-level evidence. A failure/timeout is a genuine execution problem.

    Do NOT infer that speech was absent/present from ASR output alone.
    """
    if audio_duration_ms < 1:
        raise ValueError("Source duration must be positive")
    triggers: list[Trigger] = []
    cues = ()
    if srt:
        try:
            cues = parse_srt(srt)
            scan = diagnose_srt(srt, min_repeat=policy.micro_min_cues,
                                min_chars=policy.micro_min_chars)
            for candidate in scan.candidates:
                if candidate.duration_ms < policy.micro_min_span_ms:
                    continue
                kind = ("long_loop" if candidate.duration_ms >= policy.long_min_span_ms
                        or candidate.segment_count >= policy.long_min_cues else "micro_loop")
                triggers.append(Trigger(kind, candidate.start_ms,
                                        candidate.end_ms, "consecutive_normalized_cues"))
            for cue in cues:
                if _intra_cue_loop(cue.text, policy):
                    triggers.append(Trigger("micro_loop",cue.start_ms,cue.end_ms,
                                            "repeated_ngram_within_single_cue"))
            if scan.invalid_timing_cues or scan.backwards_start_cues:
                triggers.append(Trigger("invalid_timing",0,audio_duration_ms,
                                        "negative_or_backward_srt_timestamps"))
        except ValueError:
            triggers.append(Trigger("malformed_srt",0,audio_duration_ms,
                                    "unparseable_srt"))
    if failure_or_timeout:
        tail = cues[-1].end_ms if cues else 0
        if tail < audio_duration_ms:
            triggers.append(Trigger("decoder_stall",
                                    max(0,tail-5000),audio_duration_ms,
                                    "native_process_failure_or_timeout"))
        else:
            triggers.append(Trigger("decoder_stall",max(0,audio_duration_ms-10000),
                                    audio_duration_ms,"failed_process_despite_output"))
    elif not cues:
        # This is not *confirmed* decoder pathology: audio could be silent.
        triggers.append(Trigger("no_output_unverified",0,audio_duration_ms,
                                "no_cues_without_native_failure"))
    priority={"decoder_stall":0,"long_loop":1,"micro_loop":2,
              "invalid_timing":3,"malformed_srt":4,"no_output_unverified":5}
    unique={}
    for x in triggers:
        key=(x.kind,x.start_ms,x.end_ms)
        unique[key]=x
    return tuple(sorted(unique.values(),key=lambda x:(priority.get(x.kind,9),
                                                       -(x.end_ms-x.start_ms),x.start_ms)))


@dataclass(frozen=True)
class RetryStrategy:
    name: str
    window_ms: int
    overlap_ms: int
    lead_ms: int
    tail_ms: int
    max_jobs: int
    profile_updates: tuple[tuple[str, float | int], ...] = ()
    timeout_s: int = 240
    scope: str = "trigger"  # or full_source; source coverage must remain complete

    def __post_init__(self):
        if not isinstance(self.name,str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",self.name):
            raise ValueError("Retry strategy name must be a safe slug")
        if (self.window_ms < 1000 or self.overlap_ms < 0 or
                self.overlap_ms >= self.window_ms or self.max_jobs < 1 or
                self.timeout_s < 1 or self.lead_ms < 0 or self.tail_ms < 0):
            raise ValueError("Invalid retry strategy")
        if self.scope not in ("trigger","full_source"):
            raise ValueError("Invalid strategy scope")
        allowed={"temperature","temperature_inc","beams","best_of",
                 "vad_threshold","vad_pad_ms","vad_silence_ms","vad_speech_ms"}
        if any(k not in allowed for k,_ in self.profile_updates):
            raise ValueError("Unexpected decoding parameter update")


@dataclass(frozen=True)
class AdaptivePolicy:
    strategies: tuple[RetryStrategy,...] = (
        RetryStrategy("short_context_reset",60000,3000,3000,3000,12),
        RetryStrategy("longer_context_fewer_joins",300000,30000,3000,3000,3),
        RetryStrategy("temperature_backtrack",90000,10000,8000,8000,10,
                      (("temperature",0.20),)),
        RetryStrategy("vad_backtrack",60000,5000,8000,8000,12,
                      (("temperature",0.20),("vad_threshold",0.65),
                       ("vad_pad_ms",75))),
    )
    max_attempts: int = 4
    max_total_decode_ms: int = 1_700_000
    max_total_jobs: int = 32
    max_wall_s: int = 1500
    max_regions: int = 1
    stop_at_conflicts: int = 0
    min_cues_per_non_silent_window: int = 1
    allow_unverified_no_output_trigger: bool = False
    detection: DetectionPolicy = DetectionPolicy()
    cost_aware: bool = False
    assumed_native_rtf: float = .10
    estimated_startup_s: float = 3.0
    boundary_review_penalty_s: float = 30.0
    watchdog: WatchdogPolicy | None = None
    compose_full_timeline_preview: bool = False

    def __post_init__(self):
        if (self.assumed_native_rtf<=0 or self.estimated_startup_s<0 or
                self.boundary_review_penalty_s<0):
            raise ValueError("Invalid heuristic cost parameters")
        if (not self.strategies or self.max_attempts<1 or self.max_total_decode_ms<1000
                or self.max_total_jobs<1 or self.max_wall_s<1 or self.max_regions<1
                or self.stop_at_conflicts<0):
            raise ValueError("Invalid adaptive limits")


class NativeAdapter(Protocol):
    def run_once(self, audio: Path, model_name: str, vad_name: str,
                 output_dir: Path, profile: DecodeProfile,
                 timeout_s: int=600,
                 watchdog: WatchdogPolicy | None = None) -> dict[str,Any]: ...


def _source_pcm16(path:Path)->tuple[int,int]:
    with wave.open(str(path),"rb") as w:
        if (w.getnchannels(),w.getsampwidth(),w.getframerate(),
            w.getcomptype()) != (1,2,16000,"NONE"):
            raise ValueError("Prototype only accepts 16 kHz mono PCM16 source")
        return w.getnframes(),round(w.getnframes()*1000/16000)


def _exact_window(source:Path,destination:Path,window:RetryWindow)->str:
    if destination.exists():
        raise FileExistsError("Audio segment would be overwritten")
    start_sample=window.global_start_ms*16
    n_samples=window.duration_ms*16
    with wave.open(str(source),"rb") as wav:
        if start_sample+n_samples>wav.getnframes():
            raise ValueError("Window exceeds source PCM")
        wav.setpos(start_sample)
        data=wav.readframes(n_samples)
        if len(data)!=n_samples*2:
            raise IOError("Incomplete source PCM read")
    with wave.open(str(destination),"wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(data)
    with wave.open(str(destination),"rb") as check:
        if check.getnframes()!=n_samples:
            raise IOError("Derived WAV sample count mismatched")
    return sha256_file(destination)


def _safe_native_json_artifact(run_dir:Path,record:dict[str,Any])->dict[str,Any]:
    if record.get("native_json_parse_valid"):
        return {"status":"strict_native_valid","derived_sha256":None}
    original=run_dir/"transcript.json"
    if not original.is_file():
        return {"status":"missing_json","derived_sha256":None}
    try:
        info=write_canonical_separate(
            original,run_dir/"transcript.derived.lossless-adaptive-v1.json",
            run_dir/"transcript.derived.lossless-adaptive-v1.provenance.json")
        return {"status":"reversible_token_fragment_derived",
                "derived_sha256":info["canonical_sha256"]}
    except (NativeJsonIntegrityError,ValueError,UnicodeError) as e:
        return {"status":"invalid_unrecoverable_json",
                "error_type":type(e).__name__,"derived_sha256":None}


def _new_span_bounds(triggers:tuple[Trigger,...],source_len:int,policy:AdaptivePolicy)->tuple[Trigger|None,int]:
    eligible=[t for t in triggers if t.kind not in
              ("no_output_unverified",) or policy.allow_unverified_no_output_trigger]
    return (eligible[0] if eligible else None,len(eligible))


def _planning_candidate(trigger:Trigger,audio_ms:int,step:RetryStrategy)->RepetitionCandidate:
    if step.scope == "full_source":
        return RepetitionCandidate(0,audio_ms,1,0,audio_ms)
    # The native SRT timebase can round its final cue a few milliseconds past
    # the PCM end. Permit at most 100 ms of end rounding, and record original
    # trigger times unchanged elsewhere in the decision ledger.
    if (trigger.start_ms < 0 or trigger.start_ms >= audio_ms or
            trigger.end_ms <= trigger.start_ms or
            trigger.end_ms > audio_ms+100):
        raise ValueError("Detector time interval outside source PCM tolerance")
    end=min(trigger.end_ms,audio_ms)
    return RepetitionCandidate(trigger.start_ms,end,1,0,
                               end-trigger.start_ms)


def estimated_strategy_order(trigger:Trigger,audio_ms:int,
                             policy:AdaptivePolicy)->tuple[tuple[RetryStrategy,...],list[dict[str,Any]]]:
    """Pre-run heuristic scheduling; NEVER claim these times were measured.

    Cost is estimated native seconds plus per-process startup and an explicit
    per-window-boundary review penalty. Plans without complete audio coverage
    are ineligible. Source RTF must be calibrated per backend/device and must
    not be misinterpreted as actual benchmark data.
    """
    if audio_ms < 1000:
        raise ValueError("Recording shorter than minimum retry window")
    candidates=[]
    for ordinal,step in enumerate(policy.strategies):
        window_ms=min(step.window_ms,audio_ms)
        feasible=False
        jobs=0
        covered_ms=0
        boundary_count=0
        if window_ms<=policy.max_total_decode_ms:
            try:
                planning=RetryPolicy(
                    window_ms=window_ms,overlap_ms=min(step.overlap_ms,window_ms-1),
                    lead_ms=step.lead_ms if step.scope=="trigger" else 0,
                    tail_ms=step.tail_ms if step.scope=="trigger" else 0,
                    max_jobs=min(step.max_jobs,policy.max_total_jobs),
                    max_audio_ms=policy.max_total_decode_ms)
                plan=plan_retries(_planning_candidate(trigger,audio_ms,step),
                                  audio_ms,planning)
                jobs=len(plan.windows)
                covered_ms=sum(w.duration_ms for w in plan.windows)
                feasible=plan.full_coverage and bool(jobs)
                boundary_count=max(0,jobs-1)
            except ValueError:
                feasible=False
        score=round(covered_ms/1000*policy.assumed_native_rtf
                    +jobs*policy.estimated_startup_s
                    +boundary_count*policy.boundary_review_penalty_s,3) if feasible else None
        candidates.append({
            "strategy":step.name,"ordinal":ordinal,"scope":step.scope,
            "estimated_jobs":jobs,"estimated_redecode_ms":covered_ms,
            "estimated_boundaries":boundary_count,
            "full_coverage_feasible":feasible,
            "estimated_cost_units_seconds_heuristic":score,
            "not_measured_wall_time":True,
        })
    order=sorted(range(len(candidates)),
                 key=lambda i:(not candidates[i]["full_coverage_feasible"],
                               candidates[i]["estimated_cost_units_seconds_heuristic"]
                               if candidates[i]["full_coverage_feasible"] else float("inf"),
                               i))
    if not policy.cost_aware:
        order=list(range(len(candidates)))
    return tuple(policy.strategies[i] for i in order),candidates


def run_adaptive(
    workspace:Path,
    adapter:NativeAdapter,
    source_audio:Path,
    baseline_srt:Path | None,
    baseline_failed:bool,
    output_parent:Path,
    model_name:str,
    vad_name:str,
    initial_profile:DecodeProfile,
    policy:AdaptivePolicy=AdaptivePolicy(),
    adapters_by_strategy:dict[str,NativeAdapter] | None = None
)->dict[str,Any]:
    """Run automatically; generate only *provisional* recovery outputs.

    The whole source is not overwritten. Each step has a strict budget,
    independent result path and immutable decision audit. The decoder is
    invoked only after the input source and output-parent guards pass.
    """
    ws=workspace.resolve(strict=True)
    src=ensure_inside(source_audio,ws)
    root_parent=ensure_inside(output_parent.parent,ws)
    if output_parent.resolve().parent != root_parent:
        raise ValueError("Output destination is not a direct child of its trusted parent")
    if baseline_srt is not None:
        reference=ensure_inside(baseline_srt,ws)
        text=reference.read_text(encoding="utf-8-sig",errors="strict")
        baseline_hash=sha256_file(reference)
    else:
        text=None
        baseline_hash=None
    source_frames,audio_ms=_source_pcm16(src)
    if output_parent.exists():
        raise FileExistsError("Adaptive output root already exists")
    output_parent.mkdir(exist_ok=False)
    baseline_detect=detect_pathologies(text,audio_ms,
                                     failure_or_timeout=baseline_failed,
                                     policy=policy.detection)
    trigger,eligible_count=_new_span_bounds(baseline_detect,audio_ms,policy)
    ordered_strategies, cost_estimates=(
        estimated_strategy_order(trigger,audio_ms,policy)
        if trigger is not None else ((),[])
    )
    report={
        "schema":"whispercpp/adaptive-controller-prototype/v1",
        "original_source_sha256":sha256_file(src),
        "original_srt_sha256":baseline_hash,
        "source_duration_ms":audio_ms,
        "initial_profile":asdict(initial_profile),
        "baseline_failed_or_timed_out":baseline_failed,
        "baseline_detected":[asdict(t) for t in baseline_detect],
        "eligible_triggers":eligible_count,
        "selected_trigger":asdict(trigger) if trigger else None,
        "all_regions_processed":eligible_count <= policy.max_regions,
        "cost_planning":{
            "enabled":policy.cost_aware,
            "assumed_native_rtf":policy.assumed_native_rtf,
            "startup_s":policy.estimated_startup_s,
            "per_join_review_penalty_s":policy.boundary_review_penalty_s,
            "estimates":cost_estimates,
            "ordered_strategies":[s.name for s in ordered_strategies],
            "not_measured_native_latency":True,
        },
        "policy_limits":{"max_attempts":policy.max_attempts,
                         "max_total_decode_ms":policy.max_total_decode_ms,
                         "max_total_jobs":policy.max_total_jobs,
                         "max_wall_s":policy.max_wall_s},
        "attempts":[],
        "total_transcribed_audio_ms":0,
        "native_jobs_executed":0,
        "estimated_actual_wall_s":0.,
        "status":"NO_CONFIRMED_TRIGGER" if trigger is None else "UNRESOLVED",
        "human_audio_accuracy_verified":False,
        "historical_baseline_modified":False,
        "never_promote_or_replace_original":True,
    }
    begin=time.perf_counter()
    best=None
    if trigger is not None:
        for index,step in enumerate(ordered_strategies[:policy.max_attempts],1):
            remaining_ms=policy.max_total_decode_ms-report["total_transcribed_audio_ms"]
            remaining_jobs=policy.max_total_jobs-report["native_jobs_executed"]
            attempt={
                "index":index,"strategy":step.name,"scope":step.scope,
                "parameter_adjustments":dict(step.profile_updates),
                "windows":[],"status":"NOT_EXECUTED",
            }
            report["attempts"].append(attempt)
            window_ms=min(step.window_ms,audio_ms)
            if remaining_ms<window_ms or remaining_jobs<1:
                attempt["status"]="SKIPPED_GLOBAL_BUDGET"
                break
            rp=RetryPolicy(window_ms=window_ms,
                overlap_ms=min(step.overlap_ms,window_ms-1),
                lead_ms=step.lead_ms if step.scope=="trigger" else 0,
                tail_ms=step.tail_ms if step.scope=="trigger" else 0,
                max_jobs=min(step.max_jobs,remaining_jobs),
                max_audio_ms=remaining_ms)
            try:
                candidate=_planning_candidate(trigger,audio_ms,step)
                plan=plan_retries(candidate,audio_ms,rp)
            except ValueError:
                attempt["status"]="SKIPPED_SOURCE_TRIGGER_OUT_OF_BOUNDS"
                continue
            attempt["source_endpoint_clamp_ms"]=max(0,trigger.end_ms-candidate.end_ms) if step.scope=="trigger" else 0
            attempt["planned_full_coverage"]=plan.full_coverage
            attempt["planned_windows"]=len(plan.windows)
            attempt["planned_audio_ms"]=sum(w.duration_ms for w in plan.windows)
            if not plan.full_coverage:
                attempt["status"]="SKIPPED_INCOMPLETE_WINDOW_COVERAGE"
                continue
            work=output_parent/f"attempt_{index:02d}_{step.name}"
            work.mkdir()
            audio_dir=work/"audio";audio_dir.mkdir()
            jobs_dir=work/"jobs";jobs_dir.mkdir()
            profile=replace(initial_profile,**dict(step.profile_updates),native_json_full=True)
            attempt["effective_profile"]=asdict(profile)
            runner=(adapters_by_strategy or {}).get(step.name,adapter)
            good=True
            raw_srts=[]
            for window in plan.windows:
                seconds_remaining=policy.max_wall_s-(time.perf_counter()-begin)
                if seconds_remaining < 1:
                    attempt["status"]="STOP_WALL_TIME_BUDGET"
                    good=False
                    break
                wav=audio_dir/f"w{window.job:03d}.wav"
                clip_hash=_exact_window(src,wav,window)
                target=jobs_dir/f"w{window.job:03d}"
                run_kwargs={"timeout_s":max(1,min(step.timeout_s,int(seconds_remaining)))}
                if policy.watchdog is not None:
                    run_kwargs["watchdog"]=policy.watchdog
                record=runner.run_once(wav,model_name,vad_name,target,profile,
                                        **run_kwargs)
                report["total_transcribed_audio_ms"]+=window.duration_ms
                report["native_jobs_executed"]+=1
                srtpath=target/"transcript.srt"
                pathologies=()
                if srtpath.exists():
                    try:
                        srt=srtpath.read_text(encoding="utf-8-sig",errors="strict")
                        pathologies=detect_pathologies(srt,window.duration_ms,
                            policy=policy.detection)
                    except (UnicodeDecodeError,ValueError):
                        srt=None
                else:
                    srt=None
                json_quality=_safe_native_json_artifact(target,record)
                entry={
                    "job":window.job,
                    "source_start_ms":window.global_start_ms,
                    "source_end_ms":window.global_end_ms,
                    "source_window_sha256":clip_hash,
                    "model_sha256":record.get("hashes",{}).get("model"),
                    "binary_sha256":record.get("hashes",{}).get("binary"),
                    "exit_code":record.get("exit_code"),
                    "native_engine":record.get("engine","external_adapter_unspecified"),
                    "watchdog":record.get("watchdog"),
                    "native_inference_passed":bool(record.get("inference_passed")),
                    "wall_s":record.get("elapsed_seconds"),
                    "json_quality":json_quality,
                    "output_anomalies":[asdict(t) for t in pathologies],
                    "cue_count":len(parse_srt(srt)) if srt else 0,
                    "srt_present":bool(srt),
                    "status":"CHECK_PENDING",
                }
                entry["status"]=(
                    "PROVISIONAL_OUTPUT_OK"
                    if (entry["native_inference_passed"] and
                        json_quality["status"] in ("strict_native_valid","reversible_token_fragment_derived")
                        and entry["cue_count"]>=policy.min_cues_per_non_silent_window and
                        not pathologies)
                    else "OUTPUT_REJECTED_OR_REVIEW_REQUIRED"
                )
                attempt["windows"].append(entry)
                if entry["status"]!="PROVISIONAL_OUTPUT_OK":
                    attempt["status"]="RETRY_REQUIRED_WINDOW_FAILURE"
                    good=False
                    break
                raw_srts.append(srt)
            if not good:
                continue
            try:
                joined=merge_windows(plan.windows,tuple(raw_srts))
                summary=diagnose_srt(joined.text_srt)
            except (ValueError,IndexError) as e:
                attempt["status"]="JOIN_VALIDATION_FAILURE"
                attempt["failure_reason_type"]=type(e).__name__
                continue
            joined_triggers=detect_pathologies(joined.text_srt,audio_ms,
                                               policy=policy.detection)
            conflicted=[c for c in joined.conflicts if
                        c["kind"]!="DUPLICATE_CUE_SUPPRESSED"]
            attempt["joined_cue_count"]=summary.cue_count
            attempt["join_conflicts"]=len(conflicted)
            attempt["joined_anomalies"]=[asdict(t) for t in joined_triggers]
            attempt["mechanical_join_status"]=joined.status
            if step.scope=="full_source":
                # Source-energy/timing and machine agreement are *diagnostics*,
                # never independent speech or linguistic ground truth.
                attempt["energy_coverage_proxy"]=coverage_report(src,joined.text_srt)
                if text is not None:
                    attempt["machine_hypothesis_consistency"]=compare_machine_hypotheses(
                        text,joined.text_srt,baseline_pathological=bool(baseline_detect))
            attempt["status"]=("PATHOLOGY_REMAINS"
                               if joined_triggers else
                               "PATHOLOGY_CLEARED_JOIN_REVIEW" if conflicted
                               else "PATHOLOGY_CLEARED_STRUCTURAL")
            preview=work/"provisional_recovery.srt"
            preview.write_text(joined.text_srt,encoding="utf-8")
            attempt["private_preview_sha256"]=sha256_file(preview)
            attempt["join_review"]=[{"kind":x["kind"],"jobs":x.get("jobs"),
                                     "start_ms":x.get("start_ms"),
                                     "cut_ms":x.get("cut_ms")} for x in conflicted]
            if not joined_triggers:
                if best is None or (len(conflicted),len(plan.windows)) < best[0]:
                    best=((len(conflicted),len(plan.windows)),index,
                          attempt["status"])
                if len(conflicted)<=policy.stop_at_conflicts:
                    break
        report["selected_provisional_attempt"]=best[1] if best else None
        if best is not None:
            report["status"]=(
                "PATHOLOGY_CLEARED_STRUCTURAL_UNVERIFIED"
                if best[0][0]==0 else
                "PATHOLOGY_CLEARED_JOIN_REVIEW_REQUIRED"
            )
            if policy.compose_full_timeline_preview:
                chosen=next(x for x in report["attempts"] if x["index"]==best[1])
                if chosen["scope"]=="full_source":
                    report["full_timeline"]={"status":"NOT_NEEDED_FULL_SOURCE",
                                             "reference_splice_performed":False}
                elif baseline_srt is None or not chosen["windows"]:
                    report["full_timeline"]={"status":"SKIPPED_NO_HISTORICAL_BASELINE",
                                             "reference_splice_performed":False}
                else:
                    work=output_parent/f"attempt_{best[1]:02d}_{chosen['strategy']}"
                    source_window=(min(x["source_start_ms"] for x in chosen["windows"]),
                                   max(x["source_end_ms"] for x in chosen["windows"]))
                    try:
                        source_preview=work/"provisional_recovery.srt"
                        output_preview=work/"provisional_full_timeline.srt"
                        output_ledger=work/"provisional_full_timeline_review.json"
                        combined=write_full_timeline_preview(
                            baseline_srt,source_preview,output_preview,output_ledger,
                            source_audio=src,repaired_audio_window_ms=source_window)
                        remaining=detect_pathologies(
                            output_preview.read_text(encoding="utf-8"),audio_ms,
                            policy=policy.detection)
                        report["full_timeline"]={
                            "status":combined["status"],
                            "cue_count":combined["output_cue_count"],
                            "hazards_count":combined["hazards_count"],
                            "other_unresolved_triggers":[asdict(t) for t in remaining],
                            "provisional_preview_sha256":sha256_file(output_preview),
                            "historical_srt_unchanged":True,
                            "auto_release_authorized":False
                        }
                        if (remaining or combined["hazards_count"] or
                                chosen.get("join_conflicts",0)):
                            report["status"]="PATHOLOGY_CLEARED_FULL_TIMELINE_REVIEW_REQUIRED"
                        else:
                            report["status"]="PATHOLOGY_CLEARED_FULL_TIMELINE_STRUCTURAL_UNVERIFIED"
                    except (ValueError,UnicodeError,FileExistsError) as e:
                        report["full_timeline"]={
                            "status":"FAILED_REVIEW_REQUIRED",
                            "reason_type":type(e).__name__,
                            "source_historical_srt_unchanged":True,
                        }
                        report["status"]="PATHOLOGY_CLEARED_FULL_TIMELINE_ASSEMBLY_FAILED"
        elif all(x["status"].startswith("SKIPPED") for x in report["attempts"]):
            report["status"]="NO_RETRY_BUDGET_FOR_COMPLETE_COVERAGE"
    report["estimated_actual_wall_s"]=round(time.perf_counter()-begin,3)
    audit=output_parent/"adaptive_decision_ledger.json"
    if audit.exists():
        raise FileExistsError("Controller ledger exists")
    audit.write_text(json.dumps(report,indent=2,ensure_ascii=True),encoding="utf-8")
    return report
