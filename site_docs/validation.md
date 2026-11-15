# Validation

## Stage 20 — frozen structural regression

**Code:** 442aa2a. **Date:** 15 November 2026. Source recordings are not redistributed.

| Property | Value |
|---|---:|
| WAV entries inventoried | 61 |
| WAV/SRT entries with historic repetition | 27 |
| Historical output triggers across affected entries | 66 |
| Actual WAV excerpts reprocessed | 27 (one each) |
| Selected excerpts with no remaining detected repetition | **27/27** |
| Selected excerpts without caption-join conflicts | **25/27** |
| Selected excerpts requiring join review | **2/27** |
| CUDA native jobs executed | **85** |
| Native JSON strictly valid as written | **18/85** |
| Native JSON with separately reversible derived representation | **67/85** |

The two join-review recordings remain privately identified in the source audit. Neither is accepted as an accurate transcript.

## What exactly passed?

The source historical SRTs and original PCM-derived excerpts were rehashed. All native jobs recorded actual GPU selection, and selected previews were rechecked for output repetition. The Stage 18 synthetic suite passed 215/215 on both Windows Python 3.10 and 3.14. The adaptive-framework CI passed at the frozen commit.

**Important qualifications:** only the strongest historical problem region per recording was tested, not all triggers or complete recordings. Two clean machine captions do not establish real human words. 67/85 native JSON files failed strict validation as written; the reversible derived JSON is separate from the raw artifact. WebAssembly CI has an unresolved upstream-derived build error.

## Human linguistic accuracy

**WER** is word substitutions + deletions + insertions divided by reference word count. **CER** uses characters instead. Both require a frozen *audio-first human reference* and documented Unicode, punctuation, case and uncertain-interval normalization. Stage 21 human annotation is pending; no numerical WER/CER result is established.

## Remaining research gates

Real missing-speech precision and false positives; actual acoustic word ends; complete WAV coverage; all historic trigger regions; independent reviewer agreement; verified model/binary provenance and backend breadth.
