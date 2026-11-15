# Audio Transcription Framework — project-local development scope

- Production engine: native whisper.cpp, initially Windows CPU/CUDA and Silero VAD.
- Python is intended to supervise runs, capture structured diagnostics, detect and recover from local failures and preserve TXT/SRT. This controller does not yet exist in this fork.
- Phase 0 preserves the working-reference assets before implementation; Phase 1 implements provenance and diagnostic capture, Phase 2 calibrates detectors, Phase 3 adds bounded error-adaptive recovery. Optional alignment, diarization, Optuna and LoRA are later modules, not Phase 0 requirements.
- Domain knowledge from documents and experiments is evidence, not an unreviewable threshold mandate. Failed retries must never silently replace correct baseline transcript spans.
- Keep this project context restricted to Audio Transcription Framework materials; external upstream libraries are technical dependencies, not a reason to pull unrelated work into the project.
