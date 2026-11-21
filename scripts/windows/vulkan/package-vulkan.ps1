# Package only binaries from the same fresh Windows x64 Vulkan build.
# This script deliberately refuses missing backend DLLs rather than mixing builds.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$BuildRoot,
    [Parameter(Mandatory=$true)][string]$OutputDir
)

$ErrorActionPreference = 'Stop'
$build = (Resolve-Path $BuildRoot).Path
$dest = [System.IO.Path]::GetFullPath($OutputDir)
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$cmakeCache = Join-Path $build 'CMakeCache.txt'
if (-not (Test-Path $cmakeCache)) {
    throw "Missing build CMakeCache.txt: $cmakeCache"
}
$cache = Get-Content $cmakeCache -Raw
foreach ($assertion in @('(?m)^GGML_VULKAN:BOOL=ON\s*$', '(?m)^BUILD_SHARED_LIBS:BOOL=ON\s*$', '(?m)^GGML_CUDA:BOOL=OFF\s*$')) {
    if ($cache -notmatch $assertion) { throw "Build configuration does not match requirement: $assertion" }
}

$binRoot = Join-Path $build 'bin'
if (-not (Test-Path $binRoot)) { throw "Build bin folder is missing: $binRoot" }
$candidates = @(Get-ChildItem -Path $binRoot -Recurse -File -Filter whisper-cli.exe |
    Where-Object { $_.Directory.Name -eq 'Release' })
if ($candidates.Count -ne 1) {
    throw "Expected exactly one Release whisper-cli.exe; found $($candidates.Count)"
}
$bin = $candidates[0].Directory.FullName

$required = @(
    'whisper-cli.exe', 'whisper.dll', 'ggml.dll',
    'ggml-base.dll', 'ggml-cpu.dll', 'ggml-vulkan.dll'
)
foreach ($name in $required) {
    if (-not (Test-Path (Join-Path $bin $name))) {
        throw "Missing $name in $bin. Do not copy a binary from another build."
    }
}
if (Test-Path (Join-Path $bin 'ggml-cuda.dll')) {
    throw 'Mixed CUDA and Vulkan libraries in the build output'
}

New-Item -ItemType Directory -Path $dest -Force | Out-Null
foreach ($name in $required) {
    Copy-Item (Join-Path $bin $name) (Join-Path $dest $name) -Force
}
foreach ($name in @('Whisper_Vulkan_run_script_batch.bat', 'diagnose_vulkan.bat')) {
    Copy-Item (Join-Path $PSScriptRoot $name) (Join-Path $dest $name) -Force
}
Copy-Item (Join-Path $repo 'docs/WINDOWS_VULKAN_BUILD.md') (Join-Path $dest 'README_Vulkan.md') -Force
Copy-Item (Join-Path $repo 'LICENSE') (Join-Path $dest 'LICENSE') -Force
New-Item -ItemType Directory -Path (Join-Path $dest 'models') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $dest 'output') -Force | Out-Null
@'
Place locally authorized whisper.cpp GGML model files here:
  ggml-large-v3.bin
  ggml-large-v3-q5_0.bin
  ggml-silero-v6.2.0.bin

Models are deliberately NOT included in the GitHub Actions runtime artifact.
The main batch launcher defaults to ggml-large-v3.bin; change MODEL if needed.
'@ | Set-Content (Join-Path $dest 'models/README_MODELS.txt') -Encoding UTF8

$gitSha = (& git -C $repo rev-parse HEAD).Trim()
$manifest = @(
    "Repository: Mihails-Birjukovs/Customized_WhisperCPP"
    "Commit: $gitSha"
    "Backend: Vulkan (GGML_VULKAN=ON)"
    "Build: Windows x64 Release (Visual Studio 2022)"
    "Generated: $((Get-Date).ToUniversalTime().ToString('o'))"
    "Model files: not included"
)
foreach ($name in $required) {
    $hash = (Get-FileHash (Join-Path $dest $name) -Algorithm SHA256).Hash
    $manifest += "$hash  $name"
}
$manifest | Set-Content (Join-Path $dest 'BUILD_MANIFEST.txt') -Encoding UTF8

Write-Host "Validated Vulkan runtime package at: $dest"
Write-Host "All required whisper/ggml DLLs came from: $bin"
Write-Host "The Vulkan loader vulkan-1.dll is supplied by the end-user GPU driver."
