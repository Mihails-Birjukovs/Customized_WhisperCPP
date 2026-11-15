# Architecture

1. **Native whisper.cpp:** original C/C++ decoder, CPU/CUDA/Vulkan execution and optional Silero VAD.
2. **Backend adapter:** adaptive_whisper/backend.py handles native processes, profiles, source hashes, timeouts, return codes and backend evidence.
3. **Detection and retry:** diagnostics.py, controller.py, recovery.py and regions.py identify suspicious repeated output and plan bounded source-audio retries.
4. **Caption ownership:** stitch.py, full_timeline.py and multi_timeline.py retain original cue provenance, source-clock offsets and explicit join-conflict reports.
5. **Acoustic/human review:** boundary_review.py, alignment_workbench.py, editorial_reconciliation.py and uncaptioned_recovery.py preserve audio-first evidence and review-only hypotheses.

## Evidence categories

| Evidence | Supports | Does not establish |
|---|---|---|
| Repeated machine captions | Suspicious output | Real spoken repetition |
| Native CUDA job completed | Executed backend | Correct words |
| Energy without captions | Candidate activity | Confirmed speech omission |
| Two machine retries agree | Machine hypothesis consistency | Independent word truth |
| Human audio-first reference | Single-reviewer transcription | Inter-reviewer adjudication |
| No join conflicts | Structural consistency | Whole-recording coverage |

An automatic word-faithful repair pipeline, live token-stream watchdog, and public release of machine-only corrections are **not validated**.
