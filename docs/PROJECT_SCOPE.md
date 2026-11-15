# Customized WhisperCPP — active scope

**Status:** experimental source snapshot 2026-11-15, commit 442aa2a01c1319041354aaabfe82b3dc2ccefb2a.

The Phase 0 project scope is preserved unchanged in [the historical archive](archive/PROJECT_SCOPE_PRE_STAGE22.md). Its sentence stating that the Python controller does not exist is **historical**.

## Implemented experimental capabilities

Native whisper.cpp remains the inference engine. The separate Python adaptive_whisper package provides bounded native execution, diagnostics and output-level repetition detection, retry-window planning, conflict-aware timeline previews, evidence and hashes, human-audio review tools, and reference-free **suspected** uncaptioned activity checks.

Some components are on dependent draft PRs rather than the default branch. The controller is not a live token-stream watchdog, and machine-only transcript suggestions are not accepted acoustic truth.

## Scientific and operational boundaries

- Keep historical BAT scripts, recording files and machine transcript baselines immutable.
- Distinguish native execution, caption coverage, output repetition and human word accuracy.
- Preserve raw native JSON even when invalid; separately derived lossless JSON does not qualify the native file as valid.
- Fail closed when word ownership, cue joins or source clocks are unresolved.
- Do not auto-release transcripts, promote decoding defaults or merge code as a result of structural loop clearance.

See [release status](RELEASE_STATUS.md), [Stage 22 review gates](STAGE22_REVIEW_AND_PUBLICATION.md) and the full experimental development ledger [ADAPTIVE_FRAMEWORK_PROTOTYPE.md](ADAPTIVE_FRAMEWORK_PROTOTYPE.md).
