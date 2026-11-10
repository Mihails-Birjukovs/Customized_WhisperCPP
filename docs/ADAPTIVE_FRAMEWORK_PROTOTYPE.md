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
