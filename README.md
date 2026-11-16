# Customized WhisperCPP

**Experimental error-adaptive transcription framework built around [whisper.cpp](https://github.com/ggml-org/whisper.cpp).**

This fork extends native **CPU/CUDA/Vulkan** inference with a separate Python supervisory layer for source-clock diagnostics, bounded retries, provenance, and **review-only** transcript-recovery candidates.

> **Status, 15 November 2026:** Experimental adaptive source lives on feature branches and in [draft PR #3](https://github.com/Mihails-Birjukovs/Customized_WhisperCPP/pull/3) and [draft PR #4](https://github.com/Mihails-Birjukovs/Customized_WhisperCPP/pull/4). The default master branch is **not** the validated Stage 18/20 implementation. No production recovery policy has been promoted.

[Documentation website](https://mihails-birjukovs.github.io/Customized_WhisperCPP/) | [Documentation source](site_docs/index.md) | [Current status](docs/RELEASE_STATUS.md) | [Validation](site_docs/validation.md) | [Preserved upstream README](UPSTREAM_README.md)

## Why this fork exists

Long recordings may produce repeated captions, apparent stalls, boundary conflicts, incomplete coverage, or malformed native JSON. The experimental controller detects suspicious machine output, retries bounded original-audio intervals, and retains competing evidence **without silently overwriting transcripts**.

Implemented experimental modules provide:

- Source-clock-aware SRT repetition diagnostics and retry planning.
- Native backend invocation and explicit CPU/CUDA/Vulkan evidence capture.
- Multi-region timeline assembly, cue ownership and join-conflict ledgers.
- Separate lossless derived views of invalid native JSON, with original bytes retained.
- Audio-energy versus caption-coverage screening, without claiming energy proves speech.
- Human-audio review tooling and unapproved editorial correction proposals.

The native C/C++ whisper.cpp inference runtime is not replaced.

## Evidence snapshot

The frozen Stage 20 benchmark replayed **one historically problematic original-audio excerpt per 27 WAV/SRT entries**, not entire recordings or every historic trigger. Every selected result had zero remaining **output-level repetition triggers**; **25** had no selected-output join conflict and **two** require join review. There were **66 historical triggers** in the source cohort, and **85 native CUDA jobs** were executed.

This is **not** verified transcription accuracy, WER, CER, complete speech coverage, or success on all historic triggers. Stage 21 blind audio-first word review is not yet complete. See [validation](site_docs/validation.md).

## Developer smoke test

The experimental Python package uses Python 3.10+ and the standard library for its synthetic tests:

~~~console
python -m unittest discover -s tests/adaptive_whisper -v
~~~

Actual native decoding additionally requires a trusted native executable, matching models, optional VAD, supported PCM, and hardware evidence. Private experiments and models are not included in the public fork. See [getting started](site_docs/getting-started.md).

## Upstream and rights

This is a fork of **[ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)**. The entire pre-modernization root README is retained byte-for-byte as [UPSTREAM_README.md](UPSTREAM_README.md) so upstream attribution, build reference, and links are not lost. The upstream [MIT software license](LICENSE) is retained. Research audio, private annotations, unpublished results and institutional marks are **not** automatically licensed as MIT.

## Documentation status

The documentation is approved for publication via a master-only GitHub Actions workflow at https://mihails-birjukovs.github.io/Customized_WhisperCPP/. Live availability must be verified independently after deployment. No official institutional logo is included. The experimental transcription controller remains unmerged and unpromoted.
