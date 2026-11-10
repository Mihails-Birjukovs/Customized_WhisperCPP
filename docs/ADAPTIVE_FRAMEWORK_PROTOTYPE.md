# Error-adaptive whisper.cpp — experimental shared core

This is a **prototype**, not a production transcript repair tool. It is intentionally independent of the CPU, CUDA or Vulkan Windows distribution. The existing archived Windows BAT runners and Phase 0 baseline are not changed.

The `adaptive_whisper` Python package provides:

- `backend.py`: one argument builder, bounded native subprocess runner, hashing/provenance, optional `-ojf` JSON with **strict UTF-8 and JSON syntax integrity gate**, and explicit CPU/CUDA/Vulkan backend-evidence checks.
- `diagnostics.py`: SRT parser, timestamp checks and conservative consecutive-caption repetition candidates.
- `recovery.py`: bounded retry-window planner with global timestamp mapping and explicit incomplete-coverage status.
- `stitch.py`: provisional SRT assembly with source cue ownership and a boundary conflict ledger. An optional joint-caption-gap handoff strategy does not assume such gaps are acoustic silence.

**Safety:** Never treat historical ASR output as human ground truth. Never label a machine-stitched preview as a corrected transcript without reviewing boundary conflicts and acoustic content. Inputs/outputs remain caller-controlled; native executables must be trusted, and ordinary path checks are not an OS sandbox. Do not commit private audio/transcript payloads.

**Python:** 3.10+; standard library only. For synthetic offline tests, run:

```sh
python -m unittest discover -s tests/adaptive_whisper -v
```

CI runs these tests without models or audio. Separate, explicitly authorized GPU hardware validation is required for Vulkan/CUDA. Earlier RTX 3070 hardware tests and scientific-fixture tests were conducted locally with separate private manifests; those fixtures are not included here. PR #2 covers the Windows Vulkan package and is independent of this experimental core.

**Native JSON limitation:** Some historical Windows CUDA Large-v3 runs produced TXT/SRT and a native JSON file but that JSON contains invalid UTF-8. The framework separately records inference success and artifact integrity. It does not silently replace invalid bytes or claim native JSON acceptance merely because the file exists.

- `native_json.py`: opt-in **lossless derived** JSON for native token-level UTF-8 splits. Invalid isolated token text is represented with exact hexadecimal bytes and `text = null`; the source raw JSON is preserved separately. Round-trip verification reconstructs the original JSON structure from the canonical representation. This is not automatic acoustic correction or permission to accept invalid native files silently.

## Experimental adaptive controller

`adaptive_whisper/controller.py` adds bounded output-level anomaly detection (short repeated n-grams, repeated SRT cues, CLI timeout/process failures) and a budgeted sequence of context-reset, reduced-boundary-count, temperature-adjusted, and VAD-adjusted retries. Strategies may optionally use different trusted adapters (for example CUDA followed by Vulkan) without duplicating diagnostic logic.

**Explicit limitations:** This is an offline/post-run or timeout-triggered controller, **not yet a live token-stream watchdog**. It processes at most one selected anomaly interval per current invocation; additional triggers remain unprocessed and are reported. It requires a mono 16-kHz PCM16 source and a trusted locally installed native executable. Each strategy must cover its entire selected source interval. Every completed retry includes raw-output evidence, SHA-256s, JSON integrity or reversible derivation, valid SRT timing and explicit boundary-conflict counts.

A recovery with no detected loop and a disputed join remains `PATHOLOGY_CLEARED_JOIN_REVIEW_REQUIRED`. No automatic publication, historical-file replacement, or word-accuracy claim is made. A parameter/engine change is applied only if previous attempts failed or had unresolved join problems; the best permitted candidate is chosen by explicit structural conflict count rather than by WER.

**Synthetic tests:** `python -m unittest discover -s tests/adaptive_whisper -v` now covers 45 offline cases, including forced timeout recovery, bounded-audio exhaustion, decoder switching via injected adapters and exact preservation of baseline files. Native hardware integration tests and real recordings are kept entirely outside the public repository.

## Opt-in progress watchdog and cost-aware scheduler

`adaptive_whisper/watchdog.py` provides an optional native-process heartbeat supervisor with distinct startup-without-progress, stalled-progress, and hard-wall timeouts. It reads the actual CLI progress callback stream through dedicated stderr/stdout drain threads (without shell execution), preserves byte-exact raw logs and records observed percentage events and the termination reason. **The heartbeat cannot locate the exact stalled word and is not a live transcript-correctness detector.** Use conservative timeout parameters to avoid aborting a legitimate quiet decoder phase; no-progress runs that exit normally are not automatically classified as failed.

`WhisperCppAdapter.run_once(..., watchdog=WatchdogPolicy(...))` opts into this behavior; existing calls without a watchdog use the original native subprocess runner. Controller `AdaptivePolicy(watchdog=...)` passes the same optional policy to native execution.

The experimental `AdaptivePolicy(cost_aware=True)` ranks explicitly configured strategies using a transparent heuristic: planned seconds of audio × assumed native real-time factor + process startup overhead + a configured penalty per potential join. It rejects incomplete-coverage plans. Estimates are **not measured wall times** and must be calibrated per device/engine. `RetryStrategy(scope="full_source")` allows a bounded full-recording intervention such as changing only the VAD threshold, without universally changing the preserved default profile.

Cost-aware scheduling remains **opt-in**. A retry with eliminated repetition but unresolved joins is provisional; a clean-looking transcript can still omit speech. Both options are intended for controlled, non-destructive experiments, not automatic replacement of historical recordings.

Synthetic no-model tests additionally cover progress advancement, startup stalling, idle stalling, hard timeouts, raw-log preservation, cost-ranked fallbacks and incomplete full-source rejection. Run `python -m unittest discover -s tests/adaptive_whisper -v` for the complete portable suite.

## Experimental non-WER coverage and output-consistency diagnostics

`adaptive_whisper/coverage.py` introduces private-file-compatible, backend-neutral sound-energy/SRT coverage screening. It reads mono 16-kHz PCM16 WAV in bounded 100-ms frames, computes RMS activity using pre-registered thresholds, checks whether padded subtitle intervals cover those active frames, and flags long high-energy intervals lacking any nearby caption. A three-threshold sensitivity report helps expose threshold dependence. **Sound energy is not a speech/voice classifier.** Timestamp overlap is not a proof that any word was recognized or retained.

The `compare_machine_hypotheses` function compares two ASR outputs using token-count ratio, token-multiset overlap and optional order similarity, returning only counts/warnings, not verbatim text. It highlights large output changes requiring review even when repetitions disappear. When the baseline itself is pathological, its text cannot be treated as reference truth; the consistency method explicitly disables deletion-style warnings in that case.

These engineering QA tools do **not** calculate WER/CER, infer a transcription ground truth, or certify zero missing speech. Preserve source WAVs, native TXT/SRT/JSON and original machine results. The coverage analysis does not automatically release a provisional recovery.

**Controller integration:** For `scope="full_source"` provisional recoveries, the controller now records the audio-energy coverage proxy and, when a baseline SRT exists, a machine-to-machine output consistency report. Both are metadata-only engineering warnings; neither is a release gate or a claim that all spoken words survived the intervention. For localized source windows, no misleading whole-recording coverage score is emitted.

### Fixed-interval recovery comparison

`region_output_diversity` and `region_energy_coverage` compare baseline and retry on the **same predefined original-audio time interval**, reporting text/caption diversity, longest exact repeated-caption run, and original PCM energy covered by subtitle timestamps at an explicit dBFS threshold. Region endpoints are frame-discretized with their uncertainty recorded. The functions never expose transcript words in the report and never confuse high sound energy with speech truth.

**Important scientific limitation:** A recovered transcript may have high caption/energy overlap while omitting or misrecognizing words. Region diversity and energy coverage are engineering screening proxies, not a replacement for selective listening where speech loss is ambiguous. The absence of a long identical-caption run is necessary but not sufficient for final acoustic acceptance.

### Native subtitle end-time rounding guard

When an SRT-based pathology extends only a few milliseconds beyond the exact PCM duration, local retry planning now permits **at most 100 ms** of end-time rounding and safely clamps the *planned audio window* to the last real sample. The original unmodified trigger timestamp and the explicit adjustment are kept in the decision ledger. Larger out-of-bounds source intervals are rejected/marked infeasible rather than silently shortened. The correction is covered by scheduler and end-to-end fake-adapter unit tests and does not alter the native WAV or historical SRT.

A limited local retry must **not** be labeled a complete reconstructed recording: prefix/suffix continuity and phrase retention remain distinct acceptance gates.

## Opt-in full-timeline reconstruction — experimental

`adaptive_whisper/full_timeline.py` assembles a **provisional** full-audio-clock SRT from an immutable historical baseline and source-time-aligned localized retry output. It selects whole subtitle cues, retaining original healthy prefix/suffix and retry cue provenance (source index, timestamp, and text SHA-256), rather than trimming or silently synthesizing words at the boundary. The output ledger separately identifies baseline/retry caption overlap, disputed competing hypotheses, gaps, source-time rounding and any old captions that straddle the replacement window.

`AdaptivePolicy(compose_full_timeline_preview=True)` opts into this step after a structurally cleared localized recovery; the default behavior is unchanged. A source-boundary conflict yields `PATHOLOGY_CLEARED_FULL_TIMELINE_REVIEW_REQUIRED` and **does not publish or overwrite a final transcript**. Even a structurally clean full-clock timeline is not independently verified speech completeness or WER.

Source-time ownership spanning the complete PCM recording is different from **complete recognized speech**. Optional local audio-energy/caption overlap is explicitly labeled a proxy and should be reviewed across multiple dBFS thresholds. A native cue ending up to 100 ms beyond the PCM end may be retained but is flagged; more severe timing violations and source-path traversal are rejected.

The module and new synthetic tests are backend-neutral. Real WAVs, private SRTs, review decisions and workstation-specific acceptance manifests stay out of GitHub.
