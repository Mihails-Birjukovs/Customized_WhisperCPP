# Reproducible Windows x64 Vulkan build and package

This development track provides a **Vulkan GPU backend** edition of the
existing Windows whisper.cpp batch-transcription workflow, intended for
Vulkan-capable AMD and Intel graphics (and other supported Vulkan devices).
It is separate from the ongoing error-adaptive transcription work.

## Building in GitHub Actions

The workflow `.github/workflows/build-windows-vulkan.yml` runs on a Windows
runner, installs a pinned Vulkan SDK, configures a clean Visual Studio x64
Release build and uploads an artifact called `whisper-vulkan-windows-x64`.

The Vulkan build is configured with:

- `GGML_VULKAN=ON`
- `GGML_CUDA=OFF`, `GGML_HIP=OFF`, `GGML_SYCL=OFF`
- `BUILD_SHARED_LIBS=ON`, `GGML_BACKEND_DL=OFF`
- `GGML_NATIVE=OFF`, `GGML_AVX512=OFF`

This is a *GPU-enabled* build: helper work still runs on the CPU, and
`ggml-cpu.dll` is an expected dependency. It is not a CUDA release.
A successful build on a hosted Windows runner does **not** establish
successful GPU inference on an actual AMD/Intel computer.

The packaged archive deliberately includes **no ASR or VAD model weights**.

## Manual Windows build

Prerequisites: Git, CMake, Visual Studio 2022 C++ tools and a compatible
[LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home) including `glslc`.
Build using PowerShell on a Windows development machine:

```powershell
git clone https://github.com/Mihails-Birjukovs/Customized_WhisperCPP.git
cd Customized_WhisperCPP
git switch feature/windows-vulkan-batch
cmake -S . -B build-windows-vulkan -G "Visual Studio 17 2022" -A x64 `
  -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_SYCL=OFF `
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=OFF -DGGML_NATIVE=OFF `
  -DGGML_AVX512=OFF
cmake --build build-windows-vulkan --config Release --target whisper-cli --parallel 4
./scripts/windows/vulkan/package-vulkan.ps1 `
  -BuildRoot build-windows-vulkan -OutputDir dist/Whisper_Vulkan_Windows_x64
```

Do not build into an existing CPU/CUDA build folder or copy libraries across
builds. The packaging script fails if a required Vulkan DLL is missing.

## Expected output package

```text
Whisper_Vulkan_Windows_x64/
  whisper-cli.exe
  whisper.dll
  ggml.dll
  ggml-base.dll
  ggml-cpu.dll
  ggml-vulkan.dll
  Whisper_Vulkan_run_script_batch.bat
  diagnose_vulkan.bat
  README_Vulkan.md
  LICENSE
  BUILD_MANIFEST.txt
  models/
    README_MODELS.txt
    [add your locally authorized .bin models]
  output/
```

The Windows graphics driver normally provides `vulkan-1.dll`; **do not**
copy the one from a Vulkan SDK into a redistributable package. If a GPU
driver/runtime is missing, install the graphics manufacturer's driver.

For the existing launcher's default settings, supply:

```text
models/ggml-large-v3.bin
models/ggml-silero-v6.2.0.bin
```

For low-memory GPUs, try `models/ggml-large-v3-q5_0.bin` and change the
`MODEL` variable at the top of the launcher. The diagnostic script
uses the Q5 model by default.

The launcher was adapted *directly from the existing CUDA .bat* from
the user's `Custom WhisperCPP Dev` Drive and retains its non-GPU-specific
comments, VAD, parameter-sweep and TXT/SRT output conventions.
Changes include `-dev 0`, working-directory safety, Vulkan backend
checks, and a per-sweep delayed-expansion correction.

## Mandatory AMD/Intel checks before release

1. On a machine with Vulkan-capable graphics, update the manufacturer's
   driver and run `diagnose_vulkan.bat "path\to\short.wav"`.
2. Inspect `output\vulkan_smoke.log` for Vulkan initialization, actual
   selected device and model placement on the GPU. Merely succeeding or
   printing CLI help is not proof of GPU use.
3. Compare against the CPU baseline with the same model, short audio,
   decoder parameters, and VAD settings. Record transcript differences.
4. Test Q5 and full Large-v3 (where memory permits), long recordings,
   silent sections, parameter sweeps and Silero VAD.
5. Record model SHA256, GPU/driver, Windows version, build manifest,
   elapsed runtime, sample duration, output TXT/SRT and full logs.
6. If an unsupported device silently runs on CPU, mark that as a Vulkan
   backend failure, not a successful GPU result.

Suggested initial hardware: Intel Arc 130T and AMD Radeon 740M, followed
by additional integrated and discrete GPUs. Performance on an iGPU may be
limited by shared memory and power/thermal limits.

## Build artifact versus hardware validation

A green GitHub Actions build verifies **compilation and packaging** only.
For release-quality assurance we still need a user-supplied Windows
Vulkan machine to confirm GPU inference and transcript correctness.
The workflow never downloads private research recordings.
