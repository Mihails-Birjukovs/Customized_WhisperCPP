# Phase 0 — reproducible baseline inventory (2026-10-17)

Status: **asset inventory complete for named scripts/notebook; runtime execution and transcript equivalence NOT VERIFIED.** This file documents existing evidence, not a claim that recognition software has been run.

## Provenance

- Fork: https://github.com/Mihails-Birjukovs/Customized_WhisperCPP ; initially identical to upstream ggml-org/whisper.cpp at `0c7be696371069e0baa2515ed7aa08d5a5473aa6` (2026-10-12).
- Source of archived scripts/notebook: Google Drive shared drive **Custom WhisperCPP Dev**. Byte-for-byte preserved under `legacy/`; checksums in `legacy/SHA256SUMS.json`.
- CUDA and CPU executables/DLLs are separate *2025-10-15* Windows builds; exact commit/build flags and diagnostic features are **not verified**. Do not assume they match the October 2026 source tree.
- Models observed in Drive for each CPU/CUDA/Vulkan package: `ggml-large-v3.bin` (~3.095 GB), `ggml-large-v3-q5_0.bin` (~1.081 GB) and `ggml-silero-v6.2.0.bin` (~885 kB). SHA256 hashes and actual model loading are pending local verification.
- The Wolfram `Audio_rip_from_video_v4.nb` imports MP4 audio and exports WAV at 16 kHz using Integer16 encoding, plus MP3, into its output directory; not executed here.

## Preserved behavior

The original BAT scripts scan the working directory for `*.flac *.mp3 *.wav *.ogg`, and write numbered `_run###.txt`, `_run###.srt`, and `_run###_config.txt` under `output/`. If `sweep_configs.txt` is absent, they use the defaults recorded in `configs/baseline_2026-10-17.json`. If it exists, each line supplies eleven fields: beam, best-of, temperature increment, no-speech threshold, max context, temperature, disable fallback, VAD threshold, VAD minimum silence, VAD padding, VAD minimum speech. The CUDA and CPU single-run defaults differ in thread count and VAD padding.

## Existing defects — document before changing

- In the CUDA BAT sweep loop, `if "%DISABLE_FALLBACK%"=="1"` is expanded at block parse time instead of referring to `!DISABLE_FALLBACK!`. The CPU variant has the expected delayed expansion. **The archived scripts are intentionally unchanged.**
- Both BAT scripts print `%ERRORLEVEL%` following `whisper-cli.exe` but return `exit /b 0` from the run subroutine, so a supervising caller cannot rely on the exit status. The top-level `goto :done` path also does not set a reliable failure code.
- In the no-VAD branch, neither script emits `-nth`, `-nf`, or VAD-related options. Their actual invoked arguments are thus not fully determined by the saved configuration file when VAD is disabled.
- The `for /f tokens=1-11 delims=, ` loader is positional and does not validate row length, numeric domains, or that a comment begins with `#` as a full line. Do not treat it as a robust configuration parser.
- `-bs 8` enables beam-search in current upstream; `-bo 3` belongs to greedy sampling. Confirm strategy semantics for the **historical binaries** before interpreting sweep results as an independent two-dimensional search.
- The local WAV/FLAC/MP3/OGG codec support, timestamp behavior, native JSON-full fields, dynamic text-context behavior and DTW availability must be probed on actual binaries.

## Next reproducibility gate

1. On a controlled Windows machine, inventory installed CUDA/CPU/Vulkan binaries and dependencies, hashes of all model files, driver/runtime versions, `whisper-cli.exe --help` output and `ffprobe` metadata for test audio. Never commit private audio or large model weights to this repository.
2. Capture one clean, one technical/multilingual, one long-form, and at least one known-loop/instability clip with human-verified reference or explicitly mark reference as missing. Keep held-out controls as well as failure examples.
3. Invoke *original* CUDA and CPU scripts in their native folder layout and retain raw TXT/SRT/configuration/log/exit-code output. Preserve a mapping from the original audio timeline to decoded windows.
4. Run the baseline with the intended new orchestrator **before enabling auto-recovery**, verify transcript equivalence (or document any intentional differences), record RTF, WER/CER/entity metrics when reference is available, and verify that unsuccessful runs are reported as failures.
5. Define and freeze exact Level 0 / Level 1 / Level 2 transcript-cleaning conventions before implementing exporters. The knowledge handoff names those levels but does not contain their full operative definitions.

## Project scope

Phase 0 is baseline preservation and documentation. It performs no speech decoding, retries, native source modification, repository cleanup/deletion, fine-tuning, or calibration of proposed fallbacks.
