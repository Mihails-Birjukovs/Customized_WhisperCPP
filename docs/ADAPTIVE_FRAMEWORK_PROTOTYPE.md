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

**Explicit limitations:** This is an offline/post-run or timeout-triggered controller, **not yet a live token-stream watchdog**. The legacy `run_adaptive` entry point processes one selected anomaly interval per invocation; additional triggers remain unprocessed and are reported. The separate opt-in `run_adaptive_regions` orchestrator (Stage 8 below) can process multiple distinct intervals with shared budgets. It requires a mono 16-kHz PCM16 source and a trusted locally installed native executable. Each strategy must cover its entire selected source interval. Every completed retry includes raw-output evidence, SHA-256s, JSON integrity or reversible derivation, valid SRT timing and explicit boundary-conflict counts.

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


## Stage 8: bounded multi-region recovery (experimental)

The new `adaptive_whisper/regions.py` accepts several independently
detected output-level pathologies from one mono 16-kHz PCM16 recording.
`plan_regions` sorts them on the source clock and combines intersecting
intervals, including intervals within an explicitly chosen grouping guard.
It retains all member triggers and does not equate anomalies with verified
speech errors. Distinct regions are retried chronologically through the
existing `run_adaptive` implementation, using an explicitly checked
detected-trigger override. Each region writes a separate immutable decision
ledger and provisional SRT; the aggregate ledger documents recovered,
unresolved and deferred regions.

Usage from Python: `run_adaptive_regions(workspace, adapter, source_audio,
baseline_srt, baseline_failed, output_parent, model_name, vad_name,
initial_profile, policy=AdaptivePolicy(max_regions=3, ...))`.
The default `max_regions=1` is unchanged, so processing more than one
region is an explicit opt-in. The source input, output parent and historical
SRT must remain inside the trusted workspace.

**Budget semantics:** max decoded source milliseconds, number of native jobs,
and elapsed wall time are shared across region invocations (not reset for
every anomaly); `max_attempts` remains per region. Work already spent
is never treated as available to a later region. An excluded region is
marked deferred, not recovered. Mode explicitly rejects
`RetryStrategy(scope="full_source")`: full-source VAD/parameter experiments
belong to a separate intervention, rather than repeatedly decoding the
entire recording for multiple local regions. Strategy-specific backend
fallbacks still use the existing adapter injection.

**Output safety:** By default, no cross-region splice occurs. Explicit
`compose_full_timeline_preview=True` invokes experimental multi-region
full-clock assembly **only when every detected region is structurally cleared**,
and a historical SRT exists. Recovery ownership windows must not overlap:
ambiguous padded-window ownership is rejected before creating a composite,
and the refusal is recorded as `BLOCKED_OVERLAPPING_RECOVERY_WINDOWS`.
The input alternatives remain available in their independent region folders.
The composer selects whole cues by source time and preserves historical
captions between the repaired windows. No word is cut or synthetically filled.
Its manifest records SHA-256s, per-cue provenance, original-clock ownership,
source-boundary crossings, cross-origin cue overlaps (including nonadjacent),
caption gaps, optional acoustic **energy** proxies and inherited native-window
join conflicts. A structurally clean preview is never independently verified
speech. All outputs remain private/provisional, non-destructive and prohibited
from automatic release. Existing `run_adaptive` calls remain backward-compatible.

**Boundary analysis:** `full_timeline.py` now inspects all overlapping
cross-source cue pairs (including nonadjacent pairs nested under a long
historical cue), not only adjacent captions. This closes a missed-hazard
case without changing any transcript text or assuming energy implies speech.

Synthetic regressions cover independent regions, grouped intervals, budget
exhaustion, region limits, historical-input immutability, unsafe full-source
scope, unknown trigger override and nonadjacent splice overlap. These tests
are model-free. Real CUDA/Vulkan recordings, June/July/August evidence,
blind listening and acoustically validated WER remain separate follow-up
acceptance gates.


### Stage 8 hash-gated archival regression (2026-11-11)

A separate **private**, immutable-input replay used the existing August 4 and
July 7 PCM16 WAVs, historical SRTs and already-computed recovery SRTs.
Each input SHA-256 was checked before and after; all matched. No native
whisper.cpp decoding was run. The new multi-timeline compositor used one
region per replay case, for compatibility testing; this does **not** validate
two independently decoded regions on the same real recording.

- **August 4 localized source segment:** ten output cues matched the earlier
  Stage 7 combined preview exactly in cue text and timestamps. Five boundary
  hazards were recorded, including the pre-existing 1,620-ms cross-origin
  overlap and native 20-ms timestamp rounding.
- **July 7 long-context localized recovery:** 243 preview cues; two
  cross-boundary hazards. One inherited internal window-join conflict remains
  review-required.
- **July 7 full-source VAD control:** 246 preview cues; zero structural
  boundary hazards. This does not establish linguistic accuracy, complete
  speech transcription, or a human WER.

The private replay script, input locations, private WAV/SRT files and
machine-readable audit remain in the workstation sandbox and are not committed
to GitHub. A future real multi-anomaly native run and blinded boundary
listening must precede any claim of full-clock speech correctness.


## Stage 10: natural-recording, exact-caption-boundary fixture gate

`adaptive_whisper/natural_gate.py` adds a conservative preflight and
non-destructive fixture maker for natural recordings. It specifically avoids
the Stage 9 negative-control problem where cutting an audio excerpt through
a historical caption produced misleading uncaptioned energy.

`plan_natural_prefix(source_srt_bytes, prefix_end_ms, expected_regions_ms)`
first verifies that the source SRT is parseable, the requested cut is
**exactly** at the last retained cue's end, no existing cue crosses the
cut, and the current detector returns **exactly** the explicitly expected,
independently grouped anomaly intervals. The private source SRT byte
prefix is extracted at a validated original SRT cue-header boundary,
without changing text, cue numbering, punctuation, Unicode or CRLF.
Unsupported/unmatched caption formatting fails closed.

`prepare_natural_prefix(workspace, original_wav=..., original_srt=...,
destination=..., prefix_end_ms=..., expected_regions_ms=...)` then checks
that the source WAV is *actually present* and is 16-kHz mono PCM16
with more audio than the chosen prefix. It copies exact original PCM
samples in bounded blocks to a **new** isolated WAV, writes the exact
historical source SRT prefix and a hash/provenance ledger. It refuses to
overwrite outputs or modify either original, and rechecks input hashes
after preparing the derived data.

The original SRT (historical ASR output) is **not human word truth**.
Passing this gate proves only reproducibility and exact source-time
ownership, not that two detector candidates are true acoustic errors.
The subsequent `run_adaptive_regions` native CUDA/Vulkan/CPU run,
boundary listening and independent accuracy testing remain separate
acceptance stages. The synthetic tests deliberately cover source
cue clipping, unexpected detector regions, byte-exact CRLF roundtrip,
WAV layout/length rejection, preservation of both input hashes,
and overwrite protection.

The March 3 natural-meeting candidate was preflighted privately,
with 218 exact historical cues in a 1,291,390-ms prefix and detector
intervals [451,350,459,310] and [1,287,360,1,291,390] ms.
No private source WAV/SRT contents are included in GitHub, and this
example is **SRT-only** until its original 123,664,428-byte WAV can be
staged in the trusted workstation workspace. Do not claim native
natural-recording acceptance until that happens. The audio cannot be
downloaded anonymously from the Drive link; no credentials were
exposed or used outside the connected app.


### Stage 10 signal-specific bounded recovery and actual-window budget correction

A natural March 3, 2026 meeting segment has two distinct output-level detector
candidates from the *same original*, source-contiguous 21:31.390 recording
prefix. The first is a within-caption repeated n-gram (~7:31–7:39) and
the second is consecutive repeated captions (~21:27–21:31). The original
WAV, its SRT, the exact source-clock PCM16 prefix and all first-run outputs
are hash-preserved outside GitHub.

Initial generic short-context recovery structurally cleared the second
region but not the first. A separate predeclared extended-context
investigation found that a **30-second lead and 30-second tail** removed
the first region's trigger (15 seconds on either side did not). The two
successful hypotheses can be combined into a private provisional full-clock
SRT with no residual repetition triggers but **ten boundary hazards**,
including overlapping baseline/recovered cues and a high-energy 3.41-s
caption gap. This cross-run success is *not* success of the first single
controller invocation and does not imply human-word accuracy.

To support explicit heterogeneous treatment, `run_adaptive_regions`
now has an **opt-in** `strategies_by_signal` argument mapping an exact
`Trigger.signal` to a nonempty tuple of local `RetryStrategy` values.
The same shared decode-ms, native-job and wall-time limits still apply
across all regions. The selected policy key and strategy names are
audited per region. Unsupported, mistyped or mixed-signal policies
fail closed **before** writing outputs. There is no change to the
default strategy list or automatic transcript release behavior.

**Discovered budget bug and correction.** In the first signal-aware
single-invocation natural test, Region 1 cleared after 67,960 ms of
redecode, leaving 22,040 ms of a 90,000-ms shared budget. The second
region needed only 7,030 ms, but the code rejected its *nominal*
30,000-ms maximum window. Two incorrect gates caused this:
`RetryPolicy.__post_init__` demanded `max_audio_ms >= window_ms`,
and both the pre-run heuristic and execution loop compared available
budget against nominal `window_ms`, not the actual source-window
plan. The corrected planner allows a longer nominal maximum but
still caps the **sum of actual retry-window durations** by the strict
remaining budget. Requests that genuinely exceed budget remain
incomplete and cannot be emitted as finished transcript regions.
Regression tests cover fitting and non-fitting windows, actual
native dispatch, shared-budget preservation and the independent
signal-specific routing path.

These signal-context choices are **case-derived exploratory strategies**;
they must not be silently promoted to global defaults or represented as
held-out scientific validation. The initial failed run, extended-context
run, composed private preview and first signal-aware budget-failure run
are preserved as independent, hash-gated evidence. Real natural-case
word error, speaker accuracy and splice correctness require a fresh
single-reviewer audio-first reference, not SRT-only diagnostics.


**Single-invocation natural CUDA acceptance after correction (2026-11-13):**
The corrected opt-in signal policy was tested again from scratch in a fresh
output directory. One `run_adaptive_regions` call on the exact March 3
21:31.390 PCM prefix used two native CUDA jobs:

- 67,960 ms for the within-caption candidate, with 30-s audio lead/tail;
- 7,030 ms for the consecutive-caption candidate, with 3-s lead/tail.

This consumed 74,990 ms of an explicitly shared 90,000-ms decode budget
with positive native CUDA backend selection for both calls. The resulting
full-clock provisional transcript has 234 cues, SHA-256
`5af5defa9fc16d835783d8e1d5734794683d422b333badb9f9bfee535b6e3e2a`,
**exactly identical** to the previously verified two-independent-runs
composite. Both regions structurally cleared without residual detector
triggers, but the combined output carries **ten boundary review hazards**.
The controller reports `ALL_REGIONS_FULL_TIMELINE_REVIEW_REQUIRED`
and refuses automatic transcript release.

The private hash-gated acceptance audit and audio-first single-reviewer
packet reside under the workstation `results/CUDA/` tree; *neither the
source audio nor the human reviewer annotations are in GitHub*. This
acceptance establishes actual multi-region native execution on one
original meeting, not human word error accuracy or an independent
held-out evaluation. The signal-specific policy was calibrated with
this same case, so generalization requires other recordings and
acoustic human verification. No default strategy was changed.

An exact shared-budget synthetic regression exercises a 36,000-ms cap
where Region 1 consumes 22,000 ms, leaving 14,000 ms for Region 2:
the second nominal 20,000-ms window must still execute its actual
13,000-ms plan. Both 3.10 and 3.14 synthetic test suites pass.
