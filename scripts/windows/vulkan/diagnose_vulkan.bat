@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM =====================================================
REM Whisper.cpp Vulkan diagnostic runner
REM - Use a short known-good WAV for actual inference testing
REM - --version alone does NOT prove Vulkan GPU acceleration
REM =====================================================

echo ===== Vulkan package prerequisites =====
for %%N in (whisper-cli.exe whisper.dll ggml.dll ggml-base.dll ggml-cpu.dll ggml-vulkan.dll) do (
  if not exist "%%N" (
    echo ERROR: Missing %%N
    goto :fail
  )
)
if exist "ggml-cuda.dll" echo WARNING: mixed CUDA libraries present
if exist "vulkan-1.dll" echo WARNING: local Vulkan loader; normally use system GPU driver loader

where vulkan-1.dll >nul 2>&1
if errorlevel 1 echo WARNING: Vulkan runtime DLL not found on PATH

where vulkaninfo.exe >nul 2>&1
if not errorlevel 1 (
  vulkaninfo --summary
) else (
  echo INFO: vulkaninfo unavailable; optional for end users
)

echo.
echo ===== CLI startup =====
whisper-cli.exe --version
if errorlevel 1 goto :fail

if "%~1"=="" (
  echo.
  echo Supply a short WAV path to test real Vulkan model inference:
  echo diagnose_vulkan.bat "C:\path\to\sample.wav"
  goto :done
)
if not exist "%~1" (
  echo ERROR: WAV path not found: %~1
  goto :fail
)
if not exist "models\ggml-large-v3-q5_0.bin" (
  echo ERROR: models\ggml-large-v3-q5_0.bin missing
  goto :fail
)
if not exist output mkdir output

echo.
echo ===== 10-second transcription smoke test =====
whisper-cli.exe -m "models\ggml-large-v3-q5_0.bin" -f "%~1" -dev 0 -t 8 -d 10000 -otxt -of "output\vulkan_smoke" > "output\vulkan_smoke.log" 2>&1
set "RC=!ERRORLEVEL!"
type "output\vulkan_smoke.log"
echo.
echo Exit code: !RC!
echo Backend evidence:
findstr /i /c:"ggml_vulkan" /c:"vulkan" "output\vulkan_smoke.log"
echo CPU fallback warnings:
findstr /i /c:"no GPU found" /c:"failed to load" "output\vulkan_smoke.log"
if not "!RC!"=="0" goto :fail

echo.
echo Important: Compare GPU backend registration and model placement in the
echo log, not just the successful transcription. A CPU fallback is not GPU success.
goto :done

:fail
echo FAILURE: Diagnostic prerequisites or inference failed.
:done
pause
endlocal
