# Drive inventory — Custom WhisperCPP Dev

Inspected Drive ID `0AFqAhjDvfBTwUk9PVA` (2026-10-17). This is an inventory, not a bulk transfer or executable validation.

- Root at initial inspection: `Audio_Transcription_Framework_Knowledge_Intake_Handoff_2026-10-17.md`, `Whisper_Audio_Track_Transcription/`, `Audio Track Extractor/`. The later-added `Example Input-Output/` is documented below.
- Transcription: `Whisper_GPU_CUDA/`, `Whisper_GPU_Vulkan/`, `Whisper_CPU/`, system test cases TXT, `Whisper_Parameter_Space_and_Behaviour_Summary.docx`, `Adaptive_Whisper_Transcription_Pipeline_Design_Summary.docx`, `Audio_transcription_rabbithole.pdf`, `whisper-cublas-11.8.0-bin-x64.zip`, `whisper-bin-x64.7z`.
- CUDA: `Whisper_GPU_run_script_batch.bat`, `whisper-cli.exe`, `whisper.dll`, `ggml.dll`, `ggml-cpu.dll`, `ggml-base.dll`, `ggml-cuda.dll`, `models/`, `temp_input_audio_files/`, `output/`.
- CPU: `Whisper_CPU_run_script_batch.bat`, `whisper-cli.exe`, `whisper.dll`, `ggml.dll`, `ggml-cpu.dll`, `ggml-base.dll`, `models/`, `temp_input_audio_files/`.
- Vulkan: `whisper-cli.exe`, `whisper.dll`, `ggml.dll`, `ggml-cpu.dll`, `ggml-base.dll`, `vulkan-1.dll`, `models/`, `diagnose_vulkan.bat`, `READ_ME_Vulkan.txt`, `temp_input_audio_files/`. The README refers to `Whisper_Vulkan_run_script.bat`, not present at inventory time.
- Each CPU/CUDA/Vulkan `models/` contained `ggml-large-v3.bin`, `ggml-large-v3-q5_0.bin`, `ggml-silero-v6.2.0.bin` by name/size. Identical filenames/sizes alone do not prove identical hashes.
- Audio extractor: `Audio_rip_from_video_v4.nb` and empty `temp_input_files/`.
- The listed CUDA output/input and CPU input folders were empty at the first inspection. The separately added example corpus is documented below; human-corrected transcripts and complete historical sweep logs remain unverified.

Model weights, DLLs, EXEs, private recordings and all other bulky artifacts remain in Drive. Only two BAT scripts and one WL notebook are committed as source snapshots under `legacy/`.

## Added example corpus — reviewed 2026-10-25

The `Example Input-Output/` folder was created around **2026-10-16 23:13 UTC**, after draft PR #1 was created around **23:08 UTC**. This expands the original inventory; it does not retroactively change the initial folder observations.

- `Example Set 1 -- Lecture Audio/`: **5 WAV recordings** and **5** complete historical TXT/SRT/config triplets.
- `Example Set 2 -- Team Meeting Audio/`: **36 WAV files** and **35** complete TXT/SRT/config triplets; the extra WAV is the same-named and same-sized `Lecture_3` file already seen in the lecture folder. A bytewise hash comparison is pending.
- **41 physical WAV files**, approximately **7.57 GB decimal** in all, including the suspected duplicate; **40 matched recording/transcript/config records**. Existing transcripts are ASR hypotheses, **not** human-verified ground truth.
- **All 40 historical config snapshots** use the CUDA stable defaults: threads=12, max context=128, beam=8, best-of=3, temperature=0, temperature increment=0.10, no-speech=0.995, Silero VAD threshold=0.80, minimum silence=400 ms, padding=50 ms, minimum speech=350 ms.
- **Eight** SRT recordings have independently confirmed runs of consecutive identical subtitles with advancing timestamps. See `benchmarks/phase0_corpus_manifest.json` for exact counts and intervals. These remain **suspected output-level ASR failures**, not yet verified against acoustics.
- **2026-03-10** meeting TXT was not readable as UTF-8 via the Drive reader; inspect its raw encoding instead of discarding it.
- The meeting transcripts also include `v1_for_redundancy.7z`. Preserve it and existing `whisper-bin-x64.7z` and `whisper-cublas-11.8.0-bin-x64.zip` until archive contents and checksums are examined.
- No complete manual correction corpus, raw sweep-run archive, model SHA-256 comparison, audio listening or end-to-end decoding reproduction is established yet.

The public benchmark manifest deliberately contains only **folder-key-relative filename selectors, dates/indexes, file sizes, and provisional findings**; no private Drive file IDs, direct URLs, transcript excerpts or individually named source files.
