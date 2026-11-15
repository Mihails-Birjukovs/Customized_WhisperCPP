# Release and validation status

**Snapshot:** 15 November 2026, frozen experimental source 442aa2a01c1319041354aaabfe82b3dc2ccefb2a.

| Work | Verified state |
|---|---|
| Default master | Phase 0 baseline, **not** Stage 18/20 |
| Draft PR #2 | Windows Vulkan batch package; unmerged |
| Draft PR #3 | Shared backend and bounded recovery prototype; unmerged |
| Draft PR #4 | Stage 8–18 experimental recovery; unmerged |
| Stage 20 | Private source-audio problem-region replay, completed |
| Stage 21 | Blind human-audio reference packet prepared; reference **pending** |
| Stage 22 | Documentation review branch, no merge or deployment |

## Frozen Stage 20 results

Across 61 archived WAV entries, 27 entries with historical SRT repetition patterns were selected (66 detected historical triggers). The replay processed **one strongest problem excerpt per flagged WAV**, not the entire WAV and not all 66 triggers. All 27 chosen outputs had zero remaining output-level repetition. Of these, 25 had no selected-output join conflicts; two cases still require review (identified in the private source audit). There were 85 native CUDA jobs.

Native JSON was strictly valid as written for 18/85 jobs; the other 67 required separate reversible derived representations, with raw native files preserved. These checks do **not** establish correctly transcribed words.

## Test and release gates

At source 442aa2a, local Python synthetic tests passed 215/215 on both Python 3.10 and 3.14 and adaptive-framework CI passed. The upstream-derived WebAssembly CI failed with a std::lock_guard compilation error also visible in the preceding commit. Root cause and cross-platform release acceptance remain unresolved.

No accepted human WER/CER, natural-omission precision/recall, full-WAV coverage, all-trigger review, release-quality JSON or automatic word-recovery approval is claimed. Code PR merge, site publication and native product release are independent decisions.
