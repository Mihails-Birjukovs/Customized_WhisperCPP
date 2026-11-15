"""Conservative lexical evidence for explicit word-level editorial review.

This intentionally produces *neither* word-timing alignment nor corrected
subtitles. Machine captions crossing review bounds may contain words from
outside the interval; semi-global token matching is only a worklist hint.
Human-verified timing and an editorial decision must precede publication.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

from .boundary_review import validate_human_interval
from .diagnostics import parse_srt


def _word(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def tokens_with_cue_provenance(
    candidate_srt: str, start_ms: int, end_ms: int,
    *, max_tokens: int = 1500,
) -> list[dict[str, Any]]:
    if not 0 <= start_ms < end_ms or max_tokens < 1:
        raise ValueError("Invalid source window or token limit")
    cues = parse_srt(candidate_srt)
    if candidate_srt.strip() and not cues:
        raise ValueError("Malformed nonempty candidate SRT")
    rows = []
    for cue in cues:
        if cue.end_ms <= start_ms or cue.start_ms >= end_ms:
            continue
        crossing = cue.start_ms < start_ms or cue.end_ms > end_ms
        for text in re.findall(r"\w+", cue.text, flags=re.UNICODE):
            rows.append({
                "token": _word(text), "source_cue_index": cue.index,
                "source_cue_start_ms": cue.start_ms,
                "source_cue_end_ms": cue.end_ms,
                "source_cue_crosses_review_interval": crossing,
            })
            if len(rows) > max_tokens:
                raise ValueError("Too many machine tokens in reviewed source window")
    return rows


def semiglobal_lexical_alignment(
    reference: list[str], candidates: list[str],
) -> dict[str, Any]:
    """Align reference to a *substring* of machine cue words.

    Free machine prefix/suffix avoids double-counting words belonging to
    adjacent audio. This is NOT exact-source-window WER and cannot decide
    which part of a crossing cue is audible inside the clip.
    """
    if not reference:
        raise ValueError("Need at least one fully audible reference word")
    if len(reference) > 600 or len(candidates) > 1500:
        raise ValueError("Alignment exceeds bounded memory policy")
    n, m = len(reference), len(candidates)
    cost = [[0]*(m+1) for _ in range(n+1)]
    path = [["start"]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1):
        cost[i][0] = i
        path[i][0] = "human_word_not_in_machine"
    # Row 0 is zero at every candidate prefix, allowing all prefixes
    # to remain unassigned. This is never represented as exact alignment.
    for i in range(1,n+1):
        for j in range(1,m+1):
            same = reference[i-1] == candidates[j-1]
            options = (
                (cost[i-1][j-1]+(not same), 0, "exact" if same else "substituted"),
                (cost[i-1][j]+1, 1, "human_word_not_in_machine"),
                (cost[i][j-1]+1, 2, "machine_word_not_in_human"),
            )
            score, _, operation = min(options)
            cost[i][j] = score
            path[i][j] = operation
    # Prefer the earliest endpoint among equally good competing
    # substrings; this tie-break is deterministic, not acoustic proof.
    stop = min(range(m+1), key=lambda j: (cost[n][j], j))
    i,j=n,stop
    assignments = []
    while i:
        operation = path[i][j]
        if operation in ("exact","substituted"):
            assignments.append({
                "human_index":i,"machine_index":j,
                "relation":operation
            })
            i-=1;j-=1
        elif operation == "human_word_not_in_machine":
            assignments.append({
                "human_index":i,"machine_index":None,
                "relation":"human_word_not_in_machine"
            })
            i-=1
        elif operation == "machine_word_not_in_human":
            j-=1
        else:
            raise RuntimeError("Unexpected alignment backtrace")
    assignments.reverse()
    if len(assignments)!=n or any(x["human_index"]!=ix
                                  for ix,x in enumerate(assignments,1)):
        raise RuntimeError("Incomplete semiglobal human inventory")
    return {
        "reference_word_count":n,
        "candidate_token_count":m,
        "candidate_subsequence_start_index_zero_based":j,
        "candidate_subsequence_end_index_exclusive":stop,
        "minimum_sequence_edit_cost":cost[n][stop],
        "assignments":assignments,
        "not_WER":True,
        "not_acoustic_alignment":True,
        "free_candidate_prefix_and_suffix":True,
    }


def reconciliation_evidence(
    human_review: dict[str, Any], machine_srt: str, *,
    original_duration_ms: int,
) -> dict[str, Any]:
    validated=validate_human_interval(
        human_review, source_duration_ms=original_duration_ms
    )
    start,end=validated["start_ms"],validated["end_ms"]
    human=human_review["words"]
    fully_audible=[(word["index"],_word(word["transcribed_word"]))
                   for word in human if word["audibility"]=="fully_heard"]
    if not fully_audible:
        raise ValueError("No fully audible words available for lexical screening")
    machine=tokens_with_cue_provenance(machine_srt,start,end)
    align=semiglobal_lexical_alignment(
        [word for _,word in fully_audible],
        [row["token"] for row in machine]
    )
    suggestions={}
    for row in align["assignments"]:
        human_ix,token = fully_audible[row["human_index"]-1]
        match = machine[row["machine_index"]-1] if row["machine_index"] else None
        suggestions[human_ix]={
            "human_word_index":human_ix,
            "human_word_sha256":hashlib.sha256(token.encode("utf-8")).hexdigest(),
            "lexical_relation":row["relation"],
            "machine_source_cue_index":match["source_cue_index"] if match else None,
            "machine_token_sha256":hashlib.sha256(
                match["token"].encode("utf-8")).hexdigest() if match else None,
            "machine_source_cue_crosses_review_interval":
                match["source_cue_crosses_review_interval"] if match else None,
            "machine_original_cue_interval_ms":
                [match["source_cue_start_ms"],match["source_cue_end_ms"]]
                if match else None,
            "word_timestamp_assigned_by_this_algorithm":False,
            "acoustic_correctness_certified":False,
        }
    rows=[]
    for word in human:
        ix=word["index"]
        if ix in suggestions:
            rows.append(suggestions[ix])
        else:
            token=_word(word["transcribed_word"])
            rows.append({
                "human_word_index":ix,
                "human_word_sha256":hashlib.sha256(token.encode("utf-8")).hexdigest(),
                "lexical_relation":"context_completed_partial_word_unscored",
                "machine_source_cue_index":None,
                "machine_token_sha256":None,
                "machine_source_cue_crosses_review_interval":None,
                "machine_original_cue_interval_ms":None,
                "word_timestamp_assigned_by_this_algorithm":False,
                "acoustic_correctness_certified":False,
            })
    return {
        "schema":"whispercpp/editorial-lexical-reconciliation-review/v1",
        "interval_ms":[start,end],
        "human_word_count":len(human),
        "fully_heard_reference_words_compared":len(fully_audible),
        "partially_heard_context_completed_words_excluded":len(human)-len(fully_audible),
        "machine_tokens_from_intersecting_cues":len(machine),
        "machine_cue_boundary_crossings_not_resolved":any(
            x["source_cue_crosses_review_interval"] for x in machine),
        "lexical_alignment": {
            key:value for key,value in align.items() if key!="assignments"
        },
        "word_review_rows":rows,
        "human_timestamps_complete":validated["human_timestamps_complete"],
        "editorial_decision":"PENDING_HUMAN_SOURCE_WORD_ALIGNMENT",
        "machine_word_times_inferred":False,
        "cue_source_ownership_not_resolved":True,
        "not_validated_WER":True,
        "no_private_text_in_report":True,
        "automatic_transcript_release_authorized":False,
    }



def write_unapproved_editorial_worklist(
    human_review: dict[str, Any],
    candidate_reports: dict[str, dict[str, Any]],
    destination: "Path",
    *,
    original_duration_ms: int,
) -> dict[str, Any]:
    """Create a PRIVATE CSV of human words and competing cue relationships.

    The output has blank editorial decisions by design. It is a review aid,
    not a machine correction or transcript export. Do not commit the output.
    """
    import csv
    from pathlib import Path

    if destination.exists():
        raise FileExistsError("Refusing to overwrite existing editorial worklist")
    check = validate_human_interval(
        human_review, source_duration_ms=original_duration_ms
    )
    if set(candidate_reports) != {"original", "extended_tail"}:
        raise ValueError("Expected explicitly named original and extended hypotheses")
    words = human_review["words"]
    by_label = {}
    for label, report in candidate_reports.items():
        if report.get("schema") != "whispercpp/editorial-lexical-reconciliation-review/v1":
            raise ValueError("Unrecognized machine reconciliation report")
        if report.get("interval_ms") != [check["start_ms"], check["end_ms"]]:
            raise ValueError("Machine report source interval does not match human review")
        if report.get("human_word_count") != len(words):
            raise ValueError("Machine report word inventory differs")
        rows = report.get("word_review_rows")
        if not isinstance(rows,list) or len(rows)!=len(words):
            raise ValueError("Machine report lacks human word inventory")
        if [r.get("human_word_index") for r in rows]!=list(range(1,len(words)+1)):
            raise ValueError("Machine report word sequence differs")
        for word,row in zip(words,rows):
            digest = hashlib.sha256(
                _word(word["transcribed_word"]).encode("utf-8")).hexdigest()
            if row.get("human_word_sha256")!=digest:
                raise ValueError("Machine report refers to a different human token")
        by_label[label]=rows
    fields=[
        "human_word_index","human_reference_word","human_audibility",
        "heard_fragment","human_verified_start_ms","human_verified_end_ms",
        "original_lexical_relation","original_source_cue",
        "extended_lexical_relation","extended_source_cue",
        "editorial_choice_UNFILLED","manual_audio_reason_UNFILLED",
        "source_word_owner_confirmed_UNFILLED",
    ]
    with destination.open("x",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(fields)
        for i,word in enumerate(words):
            original=by_label["original"][i]
            extended=by_label["extended_tail"][i]
            writer.writerow([
                word["index"],word["transcribed_word"],word["audibility"],
                word.get("heard_fragment") or "",
                (word["word_start_ms"] if word["word_start_ms"] is not None else ""),
                (word["word_end_ms"] if word["word_end_ms"] is not None else ""),
                original["lexical_relation"],
                original["machine_source_cue_index"] or "",
                extended["lexical_relation"],
                extended["machine_source_cue_index"] or "",
                "", "", "",
            ])
    return {
        "schema":"whispercpp/unapproved-word-editorial-worklist/v1",
        "human_word_count":len(words),
        "original_and_alternate_lexical_comparison_present":True,
        "editorial_choices_completed":0,
        "word_timing_verifications_inherited":check["human_timed_word_count"],
        "no_machine_word_timing_inferred":True,
        "contains_private_human_text":True,
        "automatic_approval_or_srt_rewrite":False,
        "auto_release_authorized":False,
    }
