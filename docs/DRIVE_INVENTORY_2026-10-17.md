# Drive inventory — Custom WhisperCPP Dev

Inspected Drive ID `0AFqAhjDvfBTwUk9PVA` (2026-10-17). This is an inventory, not a bulk transfer or executable validation.

- Root: `Audio_Transcription_Framework_Knowledge_Intake_Handoff_2026-10-17.md`, `Whisper_Audio_Track_Transcription/`, `Audio Track Extractor/`.
- Transcription: `Whisper_GPU_CUDA/`, `Whisper_GPU_Vulkan/`, `Whisper_CPU/`, system test cases TXT, `Whisper_Parameter_Space_and_Behaviour_Summary.docx`, `Adaptive_Whisper_Transcription_Pipeline_Design_Summary.docx`, `Audio_transcription_rabbithole.pdf`, `whisper-cublas-11.8.0-bin-x64.zip`, `whisper-bin-x64.7z`.
- CUDA: `Whisper_GPU_run_script_batch.bat`, `whisper-cli.exe`, `whisper.dll`, `ggml.dll`, `ggml-cpu.dll`, `ggml-base.dll`, `ggml-cuda.dll`, `models/`, `temp_input_audio_files/`, `output/`.
- CPU: `Whisper_CPU_run_script_batch.bat`, `whisper-cli.exe`, `whisper.dll`, `ggml.dll`, `ggml-cpu.dll`, `ggml-base.dll`, `models/`, `temp_input_audio_files/`.
- Vulkan: `whisper-cli.exe`, `whisper.dll`, `ggml.dll`, `ggml-cpu.dll`, `ggml-base.dll`, `vulkan-1.dll`, `models/`, `diagnose_vulkan.bat`, `READ_ME_Vulkan.txt`, `temp_input_audio_files/`. The README refers to `Whisper_Vulkan_run_script.bat`, not present at inventory time.
- Each CPU/CUDA/Vulkan `models/` contained `ggml-large-v3.bin`, `ggml-large-v3-q5_0.bin`, `ggml-silero-v6.2.0.bin` by name/size. Identical filenames/sizes alone do not prove identical hashes.
- Audio extractor: `Audio_rip_from_video_v4.nb` and empty `temp_input_files/`.
- Listed CUDA output/input and CPU input did not contain reference samples or baseline output at the time of inventory. Human-verified clips and historical sweep raw results remain missing from the current analysis.

Model weights, DLLs, EXEs, private recordings and all other bulky artifacts remain in Drive. Only two BAT scripts and one WL notebook are committed as source snapshots under `legacy/`.
