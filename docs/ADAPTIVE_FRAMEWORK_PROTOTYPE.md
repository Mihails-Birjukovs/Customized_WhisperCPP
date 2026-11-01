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
