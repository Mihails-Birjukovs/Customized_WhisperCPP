"""Conservative, backend-neutral, bounded re-transcription planner.

Planning only; this module neither destroys originals nor declares audio truth.
Actual test executions call the same WhisperCppAdapter used for initial runs.
"""
from __future__ import annotations
from dataclasses import dataclass
from .diagnostics import RepetitionCandidate


@dataclass(frozen=True)
class RetryPolicy:
    window_ms: int = 60_000
    overlap_ms: int = 3_000
    lead_ms: int = 3_000
    tail_ms: int = 3_000
    max_jobs: int = 2
    max_audio_ms: int = 120_000

    def __post_init__(self) -> None:
        if self.window_ms < 1000:
            raise ValueError("Window too short")
        if not 0 <= self.overlap_ms < self.window_ms:
            raise ValueError("Overlap must be nonnegative and less than window")
        if self.lead_ms < 0 or self.tail_ms < 0 or self.max_jobs < 1:
            raise ValueError("Invalid policy")
        if self.max_audio_ms < self.window_ms:
            raise ValueError("Audio budget smaller than one full window")


@dataclass(frozen=True)
class RetryWindow:
    job: int
    global_start_ms: int
    global_end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.global_end_ms - self.global_start_ms

    def to_global(self, relative_ms: int) -> int:
        if not 0 <= relative_ms <= self.duration_ms:
            raise ValueError("Relative timing outside retry window")
        return self.global_start_ms + relative_ms


@dataclass(frozen=True)
class RetryPlan:
    candidate_start_ms: int
    candidate_end_ms: int
    padded_start_ms: int
    padded_end_ms: int
    windows: tuple[RetryWindow, ...]
    full_coverage: bool
    remaining_uncovered_ms: int
    status: str


def plan_retries(candidate: RepetitionCandidate, audio_duration_ms: int,
                 policy: RetryPolicy = RetryPolicy()) -> RetryPlan:
    if audio_duration_ms <= 0:
        raise ValueError("Invalid recording length")
    if (candidate.start_ms < 0 or candidate.end_ms <= candidate.start_ms
            or candidate.end_ms > audio_duration_ms):
        raise ValueError("Candidate outside recording")
    begin = max(0,candidate.start_ms-policy.lead_ms)
    finish = min(audio_duration_ms,candidate.end_ms+policy.tail_ms)
    windows=[]
    spent=0
    pos=begin
    # Strict budget: no recursive retries and no unbounded window count.
    while pos < finish and len(windows) < policy.max_jobs and spent < policy.max_audio_ms:
        span = min(policy.window_ms, finish-pos, policy.max_audio_ms-spent)
        if span <= 0:
            break
        end=pos+span
        windows.append(RetryWindow(len(windows)+1,pos,end))
        spent+=span
        if end >= finish:
            break
        pos=end-policy.overlap_ms
        if span <= policy.overlap_ms:  # Budget exhausted before any forward progress.
            break
    complete=bool(windows) and windows[-1].global_end_ms >= finish
    missing=max(0,finish-(windows[-1].global_end_ms if windows else begin))
    return RetryPlan(candidate.start_ms,candidate.end_ms,begin,finish,
                     tuple(windows),complete,missing,
                     "COVERAGE_COMPLETE_UNVERIFIED" if complete else "PARTIAL_PROBE_ONLY_NO_STITCH")
