"""Build a *review-only* source-clock SRT of fully heard, manually timed words.

This is NOT a transcript publisher or an automatic ASR corrector. Human
audibility/timing evidence is necessary but not sufficient for editorial
approval. Partial/context-completed words are excluded from SRT text and
retained with explicit provenance in a private sidecar.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

from .boundary_review import validate_human_interval
from .diagnostics import parse_srt
from .stitch import fmt


@dataclass(frozen=True)
class EditorialWordFragment:
    word_srt: str
    review: dict[str, Any]


def build_review_only_fragment(
    human_review: dict[str, Any],
    machine_candidates: Mapping[str, str],
    *,
    original_duration_ms: int,
) -> EditorialWordFragment:
    """Derive human-timed fully-audible word cues, with competing cue ledger.

    No inferred word boundaries, no inferred acoustic accuracy of machine
    text, no insertion into a full-clock SRT, and no publication authority.
    """
    validated = validate_human_interval(
        human_review, source_duration_ms=original_duration_ms)
    words = human_review["words"]
    if not isinstance(machine_candidates, Mapping) or not machine_candidates:
        raise ValueError("Must preserve at least one named machine hypothesis")
    if any(not isinstance(name,str) or not name or
           not isinstance(srt,str) for name,srt in machine_candidates.items()):
        raise ValueError("Invalid candidate mapping")
    if len(set(machine_candidates)) != len(machine_candidates):
        raise ValueError("Duplicate candidate keys")

    start_ms,end_ms=validated["start_ms"],validated["end_ms"]
    previous_end = start_ms
    for w in words:
        a,b=w.get("word_start_ms"),w.get("word_end_ms")
        if a is None or b is None:
            if w["audibility"]=="fully_heard":
                raise ValueError("A fully heard word is missing human timing")
            continue
        if w.get("human_timing_verified") is not True:
            raise ValueError("Word-clock bounds were not human-verified")
        if a<previous_end:
            raise ValueError("Overlapping human word intervals are not resolved")
        previous_end=b

    candidate_cues={}
    for name,srt in machine_candidates.items():
        cues=parse_srt(srt)
        if srt.strip() and not cues:
            raise ValueError("Nonempty invalid candidate SRT: "+name)
        if any(c.start_ms<0 or c.end_ms<=c.start_ms
               or c.end_ms>original_duration_ms+50 for c in cues):
            raise ValueError("Candidate cue exceeds original source clock: "+name)
        candidate_cues[name]=cues

    full_words=[]
    partial_words=[]
    word_cue_rows=[]
    for w in words:
        a,b=w.get("word_start_ms"),w.get("word_end_ms")
        fully=w["audibility"]=="fully_heard"
        if fully:
            full_words.append(w)
        else:
            partial_words.append({
                "human_word_index":w["index"],
                "source_interval_ms":[a,b] if a is not None else None,
                "audible_fragment":w["heard_fragment"],
                "context_completed_word":w["transcribed_word"],
                "entire_word_acoustically_confirmed":False,
                "right_censored_by_review_clip":b==end_ms if b is not None else None,
                "human_word_timestamp_verified":w["human_timing_verified"],
                "excluded_from_fully_heard_review_srt":True,
            })
        candidate_links={}
        for label,cues in candidate_cues.items():
            intersections=[
                c for c in cues
                if a is not None and b is not None
                and c.start_ms < b and c.end_ms > a
            ]
            handoffs=sorted({
                c.end_ms for c in intersections if a < c.end_ms < b
            } | {
                c.start_ms for c in intersections if a < c.start_ms < b
            })
            candidate_links[label]={
                "cue_indices":[c.index for c in intersections],
                "cue_source_intervals_ms":[[c.start_ms,c.end_ms] for c in intersections],
                "cue_boundaries_inside_human_word_ms":handoffs,
                "machine_cues_split_human_word":len(intersections)>1 and bool(handoffs),
                "machine_text_verified_by_human":False,
                "manual_word_ownership_decision":None,
            }
        word_cue_rows.append({
            "human_word_index":w["index"],
            "human_word_sha256":hashlib.sha256(
                w["transcribed_word"].encode("utf-8")).hexdigest(),
            "source_interval_ms":[a,b] if a is not None else None,
            "fully_heard":fully,
            "partial_context_word":not fully,
            "candidate_cue_intersections":candidate_links,
            "editorial_word_ownership_confirmed":False,
        })

    srt="".join(
        f"{i}\n{fmt(w['word_start_ms'])} --> {fmt(w['word_end_ms'])}\n"
        f"{w['transcribed_word']}\n\n"
        for i,w in enumerate(full_words,1)
    )
    if len(parse_srt(srt))!=len(full_words):
        raise RuntimeError("Generated human-word preview cannot be parsed")
    if not full_words:
        raise ValueError("No fully audible, human-timed words to review")
    outside={
        label:{
            "candidate_srt_sha256":hashlib.sha256(
                machine_candidates[label].encode("utf-8")).hexdigest(),
            "intersecting_review_cues":[
                {"cue_index":c.index,"source_interval_ms":[c.start_ms,c.end_ms],
                 "crosses_review_begin":c.start_ms<start_ms,
                 "crosses_review_end":c.end_ms>end_ms}
                for c in cues if c.start_ms<end_ms and c.end_ms>start_ms
            ],
            "nearest_preceding_cue_end_ms":max(
                (c.end_ms for c in cues if c.end_ms<=start_ms),default=None),
            "nearest_following_cue_start_ms":min(
                (c.start_ms for c in cues if c.start_ms>=end_ms),default=None),
        } for label,cues in candidate_cues.items()
    }
    ledger={
        "schema":"whispercpp/fully-heard-word-fragment-review/v1",
        "source_clock_ms":[start_ms,end_ms],
        "locked_human_reference_sha256":validated["locked_reference_sha256"],
        "source_audio_sha256":validated["source_audio_sha256"],
        "words_in_human_review":len(words),
        "fully_heard_word_cues":len(full_words),
        "partial_context_completed_words_excluded":len(partial_words),
        "partially_heard_word_evidence":partial_words,
        "human_word_to_machine_cue_ownership":word_cue_rows,
        "source_hypotheses":outside,
        "fully_heard_word_preview_srt_sha256":hashlib.sha256(
            srt.encode("utf-8")).hexdigest(),
        "source_clock_timings_are_human_verified":True,
        "machine_caption_texts_not_used_to_create_review_words":True,
        "word_confidence_from_audio_not_inferred_by_machine":True,
        "full_recording_srt_has_not_been_edited":True,
        "all_editorial_decisions_require_explicit_approval":True,
        "publication_ready":False,
        "auto_release_authorized":False,
    }
    return EditorialWordFragment(word_srt=srt,review=ledger)
