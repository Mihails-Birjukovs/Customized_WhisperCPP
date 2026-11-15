"""Cross-clip human-corroborated, immutable-gap SRT repair *preview*.

A word cut off by a short review clip can be fully heard in other
overlapping, audio-first human references. Preserve the distinction:
word identity may be corroborated while its full acoustic end time is
not measured. This builds a separately versioned review-only full-clock
SRT candidate; no model choices or native source files are modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any, Sequence
import unicodedata

from .boundary_review import validate_human_interval
from .diagnostics import parse_srt
from .stitch import fmt


def canonical_words(text: str) -> list[str]:
    return [unicodedata.normalize("NFKC",x).casefold()
            for x in re.findall(r"\w+",text,flags=re.UNICODE)]


@dataclass(frozen=True)
class CrossClipGapPreview:
    proposed_full_srt: str
    amendment_ledger: dict[str, Any]


def build_crossclip_gap_preview(
    base_srt: str,
    human_word_reference: dict[str, Any],
    complete_word_sources: Sequence[dict[str, Any]],
    *,
    original_duration_ms: int,
    expected_empty_gap_ms: tuple[int,int],
) -> CrossClipGapPreview:
    """Insert one preview-only human phrase into a proven empty SRT gap.

    Conditions:
    - byte-frozen reference input is represented by SHA in provenance;
    - every human word has an explicit source-clock timing and category;
    - last word is partial only in short clip but independently *within
      this single reviewer's other audio-first longer clips* complete;
    - the replacement does not change any existing caption text/time;
    - no automatic publication flag is ever raised.
    """
    check=validate_human_interval(
        human_word_reference,source_duration_ms=original_duration_ms)
    gap_start,gap_end=expected_empty_gap_ms
    if (type(gap_start) is not int or type(gap_end) is not int or
            (gap_start,gap_end)!=(check["start_ms"],check["end_ms"])):
        raise ValueError("Gap source clock must exactly equal reviewed interval")
    words=human_word_reference["words"]
    if (len(words)<2 or any(w.get("word_start_ms") is None or
                            w.get("word_end_ms") is None or
                            w.get("human_timing_verified") is not True for w in words)):
        raise ValueError("Human source-clock timing must be complete")
    if any(w["word_end_ms"]>z["word_start_ms"] for w,z in zip(words,words[1:])):
        raise ValueError("Overlapping human word timing is unresolved")
    partial=[w for w in words if w["audibility"]=="partial_context_completed"]
    if len(partial)!=1 or partial[0]["index"]!=len(words):
        raise ValueError("This preview requires one censored final word")
    last=partial[0]
    if last["word_end_ms"]!=gap_end:
        raise ValueError("Final partial word must be right-censored at short clip cutoff")
    final_word=canonical_words(last["transcribed_word"])
    if len(final_word)!=1:
        raise ValueError("Invalid single final word")
    if not isinstance(complete_word_sources,Sequence) or len(complete_word_sources)<2:
        raise ValueError("At least two earlier audio-first context clips required")
    seen=set()
    provenance=[]
    for source in complete_word_sources:
        if not isinstance(source,dict):
            raise ValueError("Invalid complete-word evidence")
        clip=source.get("clip_id")
        if not isinstance(clip,str) or not clip or clip in seen:
            raise ValueError("Missing or duplicated clip identity")
        seen.add(clip)
        a,b=source.get("source_clock_ms",(None,None))
        if (type(a) is not int or type(b) is not int
                or not (0<=a<=last["word_start_ms"]<gap_end<b<=original_duration_ms)):
            raise ValueError("Independent clip must cover source word and extend after G01")
        if source.get("audio_first_review") is not True:
            raise ValueError("Corroboration must be audio-first")
        if source.get("full_word_audibly_heard") is not True:
            raise ValueError("Full word not attested as audible")
        transcript=source.get("human_transcript")
        if not isinstance(transcript,str) or final_word[0] not in canonical_words(transcript):
            raise ValueError("Longer human transcript does not contain the completed word")
        digest=source.get("locked_csv_sha256")
        if not isinstance(digest,str) or re.fullmatch(r"[0-9a-f]{64}",digest) is None:
            raise ValueError("Missing complete-word reference hash")
        provenance.append({
            "clip_id":clip,
            "source_clock_ms":[a,b],
            "locked_audio_first_csv_sha256":digest,
            "human_complete_word_identity_attested":True,
            "corroboration_same_human_reviewer":True,
            "acoustic_full_word_end_ms_human_verified":False,
            "matching_word_in_context_reference":True,
        })

    cues=parse_srt(base_srt)
    if base_srt.strip() and not cues:
        raise ValueError("Invalid source SRT")
    if any(c.start_ms<0 or c.end_ms<=c.start_ms or c.end_ms>original_duration_ms+50
           for c in cues):
        raise ValueError("Out-of-clock base subtitle cue")
    if any(c.start_ms<gap_end and c.end_ms>gap_start for c in cues):
        raise ValueError("Refusing to replace or overlap original machine subtitle cues")
    insert_start=words[0]["word_start_ms"]
    insert_end=gap_end
    if not (gap_start<=insert_start<insert_end<=original_duration_ms):
        raise ValueError("Invalid proposed phrase interval")
    if any(c.start_ms<insert_end and c.end_ms>insert_start for c in cues):
        raise ValueError("Proposed phrase overlaps existing subtitle")
    if any(c.start_ms>c.end_ms for c in cues):
        raise ValueError("Invalid pre-existing cue")
    phrase=" ".join(w["transcribed_word"] for w in words)
    if not phrase or canonical_words(phrase)!=[
        canonical_words(w["transcribed_word"])[0] for w in words
    ]:
        raise ValueError("Unexpected tokenization in source human words")
    sequence=[(c.start_ms,c.end_ms,c.text) for c in cues]
    proposed=sorted(sequence+[(insert_start,insert_end,phrase)],
                    key=lambda x:(x[0],x[1]))
    if len(proposed)!=len(sequence)+1:
        raise RuntimeError("Unexpected candidate cue count")
    if [row for row in proposed if row!=(insert_start,insert_end,phrase)]!=sequence:
        # A preexisting machine cue may have an unusual ordering that
        # must not be silently changed by a proposed splice.
        raise ValueError("Base caption ordering would change")
    text="".join(
        f"{i}\n{fmt(a)} --> {fmt(b)}\n{caption}\n\n"
        for i,(a,b,caption) in enumerate(proposed,1)
    )
    reparsed=parse_srt(text)
    if (len(reparsed)!=len(cues)+1
            or [(c.start_ms,c.end_ms,c.text) for c in reparsed]!=proposed):
        raise RuntimeError("Preview transcription failed roundtrip")
    return CrossClipGapPreview(
        proposed_full_srt=text,
        amendment_ledger={
            "schema":"whispercpp/human-crossclip-gap-preview/v1",
            "source_clock_gap_ms":[gap_start,gap_end],
            "human_confirmed_phrase_start_ms":insert_start,
            "human_review_clip_end_ms":gap_end,
            "amendment_text_sha256":sha256(phrase.encode("utf-8")).hexdigest(),
            "human_word_count":len(words),
            "fully_audible_within_short_clip":len(words)-1,
            "whole_final_word_identity_corroborated_in_longer_clips":True,
            "final_word_acoustic_end_ms_unmeasured":True,
            "final_word_last_short_clip_timestamp_is_right_censored":True,
            "final_word_right_edge_is_a_display_cutoff_not_full_word_end":True,
            "cross_clip_references":provenance,
            "base_srt_sha256":sha256(base_srt.encode("utf-8")).hexdigest(),
            "preview_srt_sha256":sha256(text.encode("utf-8")).hexdigest(),
            "base_caption_count":len(cues),
            "preview_caption_count":len(reparsed),
            "all_original_cue_text_and_times_preserved":True,
            "new_source_clock_caption_ms":[insert_start,insert_end],
            "other_recording_hazards_not_cleared":True,
            "not_independently_adjudicated":True,
            "human_word_identity_evidence_is_from_one_reviewer":True,
            "automatic_word_end_inference":False,
            "machine_caption_text_used_to_restore_phrase":False,
            "editorial_provisional_status":"CROSSCLIP_HUMAN_WORD_IDENTITY_TIME_RIGHT_CENSORED",
            "complete_full_timeline_transcript_approved":False,
            "auto_release_authorized":False,
        }
    )
