@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =====================================================
REM Whisper.cpp Vulkan runner with VAD + param sweep
REM - For each audio file: runs ALL configs, then next file
REM - Outputs: _run###.txt, _run###.srt, _run###_config.txt
REM - Vulkan GPU backend for AMD/Intel and other compatible devices
REM =====================================================

REM ---- Fixed configuration ----
set "MODEL=models\ggml-large-v3.bin"
set "THREADS=12"
set "GPU_DEVICE=0"
set "OUTDIR=output"
set "LANG=auto"
set "DISABLE_FALLBACK=0"
set "TEMP=0.0"

REM ---- VAD model ----
set "VADMODEL=models\ggml-silero-v6.2.0.bin"
set "VAD=1"

REM ---- Sweep file ----
set "SWEEP_FILE=sweep_configs.txt"



REM ---- Use the script directory so it can be launched from anywhere ----
cd /d "%~dp0"

echo Working directory: %CD%
echo.

REM ---- Derived flags ----
set "NF_FLAG="
if "%DISABLE_FALLBACK%"=="1" set "NF_FLAG=-nf"

REM ---- Checks ----
if not exist "whisper-cli.exe" (
  echo ERROR: whisper-cli.exe not found in %CD%
  goto :done
)

REM ---- The Vulkan backend must be present; CPU-only DLLs are insufficient ----
if not exist "ggml-vulkan.dll" (
  echo ERROR: ggml-vulkan.dll is missing. Build and package matching Vulkan binaries.
  goto :done
)

if exist "ggml-cuda.dll" echo WARNING: ggml-cuda.dll detected in the Vulkan package. Remove mixed backends.
if exist "vulkan-1.dll" echo WARNING: Local vulkan-1.dll detected. Prefer driver-supplied Vulkan loader.

if not exist "%MODEL%" (
  echo ERROR: Model not found: %MODEL%
  goto :done
)

if not exist "%VADMODEL%" (
  echo ERROR: VAD model not found: %VADMODEL%
  goto :done
)

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM ---- Find audio files ----
set "FOUND="
for %%F in (*.flac *.mp3 *.wav *.ogg) do (
  if exist "%%F" set "FOUND=1"
)

if not defined FOUND (
  echo No audio files found in %CD%
  echo Supported: *.flac *.mp3 *.wav *.ogg
  goto :done
)



REM ---- Ensure sweep file exists; if not, run a single default config ----
if not exist "%SWEEP_FILE%" (
  echo WARNING: %SWEEP_FILE% not found. Running single default config.
  REM Default values -- current best stable settings
  set "MAX_CONTEXT=128"
  set "NO_SPEECH_THRESHOLD=0.995"
  set "BEAM_SIZE=8"
  set "BEST_OF=3"
  set "TEMP_INC=0.10"
  set "VAD_THRESHOLD=0.80"
  set "VAD_MIN_SILENCE_MS=400"
  set "VAD_PAD_MS=50"
  set "VAD_MIN_SPEECH_MS=350"

  for %%F in (*.flac *.mp3 *.wav *.ogg) do (
    if exist "%%F" call :run_one_file_single "%%F"
  )

  goto :done
)



REM =====================================================
REM Main loop:
REM For each file -> for each config line in sweep_configs.txt
REM =====================================================
for %%F in (*.flac *.mp3 *.wav *.ogg) do (
  if exist "%%F" call :run_sweep_for_file "%%F"
)

goto :done



REM =====================================================
REM Run all configs for ONE file
REM =====================================================
:run_sweep_for_file
set "INFILE=%~1"
set "BASENAME=%~n1"

echo ============================================================
echo File: %INFILE%
echo ============================================================

set /a RUNID=0

for /f "usebackq tokens=1-11 delims=, " %%a in ("%SWEEP_FILE%") do (
  REM Skip comment/empty lines
  if not "%%a"=="" (
    if /i not "%%a"=="#" (
      set /a RUNID+=1

      REM Assign per-run config from sweep line (11 columns)
      set "BEAM_SIZE=%%a"
      set "BEST_OF=%%b"
      set "TEMP_INC=%%c"
      set "NO_SPEECH_THRESHOLD=%%d"
      set "MAX_CONTEXT=%%e"
      set "TEMP=%%f"
      set "DISABLE_FALLBACK=%%g"
      set "VAD_THRESHOLD=%%h"
      set "VAD_MIN_SILENCE_MS=%%i"
      set "VAD_PAD_MS=%%j"
      set "VAD_MIN_SPEECH_MS=%%k"

      REM Recompute derived flag in case DISABLE_FALLBACK changes per run
      set "NF_FLAG="
      if "!DISABLE_FALLBACK!"=="1" set "NF_FLAG=-nf"

      call :run_one_config "%INFILE%" "%BASENAME%" !RUNID!
    )
  )
)

exit /b 0



REM =====================================================
REM Single-config path (when no sweep file)
REM =====================================================
:run_one_file_single
set "INFILE=%~1"
set "BASENAME=%~n1"
set "RUNID=1"
call :run_one_config "%INFILE%" "%BASENAME%" %RUNID%
exit /b 0



REM =====================================================
REM Run one config for one file
REM Args: INFILE, BASENAME, RUNID
REM =====================================================
:run_one_config
set "INFILE=%~1"
set "BASENAME=%~2"
set "RUNID=%~3"

REM Format run id as 3 digits
set "RID=00%RUNID%"
set "RID=!RID:~-3!"

set "OUTBASE=%OUTDIR%\%BASENAME%_run!RID!"
set "CFGFILE=%OUTBASE%_config.txt"

echo ------------------------------------------------------------
echo Run !RID! for %BASENAME%
echo OUTBASE: %OUTBASE%
echo ------------------------------------------------------------

REM ---- Write config dump ----
> "%CFGFILE%" (
  echo REM ---- Configuration ----
  echo set "MODEL=%MODEL%"
  echo set "THREADS=%THREADS%"
  echo set "BACKEND=Vulkan"
  echo set "GPU_DEVICE=%GPU_DEVICE%"
  echo set "OUTDIR=%OUTDIR%"
  echo set "LANG=%LANG%"
  echo.
  echo REM ---- Decoding controls ----
  echo set "NO_SPEECH_THRESHOLD=%NO_SPEECH_THRESHOLD%"
  echo set "BEAM_SIZE=%BEAM_SIZE%"
  echo set "DISABLE_FALLBACK=%DISABLE_FALLBACK%"
  echo set "BEST_OF=%BEST_OF%"
  echo.
  echo REM ---- Context control ----
  echo set "MAX_CONTEXT=%MAX_CONTEXT%"
  echo.
  echo REM ---- Temperature control ----
  echo set "TEMP=%TEMP%"
  echo set "TEMP_INC=%TEMP_INC%"
  echo.
  echo REM ---- VAD basic options ----
  echo set "VADMODEL=%VADMODEL%"
  echo set "VAD=%VAD%"
  echo set "VAD_THRESHOLD=%VAD_THRESHOLD%"
  echo set "VAD_MIN_SILENCE_MS=%VAD_MIN_SILENCE_MS%"
  echo.
  echo REM ---- VAD fine-grain segmentation controls ----
  echo set "VAD_PAD_MS=%VAD_PAD_MS%"
  echo set "VAD_MIN_SPEECH_MS=%VAD_MIN_SPEECH_MS%"
)

REM ---- Run whisper ----
if "%VAD%"=="1" (
  whisper-cli.exe ^
    -m "%MODEL%" ^
    -f "%INFILE%" ^
    -l %LANG% ^
    -t %THREADS% ^
    -dev %GPU_DEVICE% ^
    -mc %MAX_CONTEXT% ^
    -bs %BEAM_SIZE% ^
    -bo %BEST_OF% ^
    -tp %TEMP% ^
    -tpi %TEMP_INC% ^
    -pp ^
    --vad ^
    -vm "%VADMODEL%" ^
    -vt %VAD_THRESHOLD% ^
    -vsd %VAD_MIN_SILENCE_MS% ^
    -vspd %VAD_MIN_SPEECH_MS% ^
    -vp %VAD_PAD_MS% ^
    -nth %NO_SPEECH_THRESHOLD% ^
    %NF_FLAG% ^
    -otxt ^
    -osrt ^
    -of "%OUTBASE%"
) else (
  whisper-cli.exe ^
    -m "%MODEL%" ^
    -f "%INFILE%" ^
    -l %LANG% ^
    -t %THREADS% ^
    -dev %GPU_DEVICE% ^
    -mc %MAX_CONTEXT% ^
    -bs %BEAM_SIZE% ^
    -bo %BEST_OF% ^
    -tp %TEMP% ^
    -tpi %TEMP_INC% ^
    -pp ^
    -otxt ^
    -osrt ^
    -of "%OUTBASE%"
)



echo Exit code: %ERRORLEVEL%
echo.
exit /b 0

:done
echo Finished. Output folder: %CD%\%OUTDIR%
pause
endlocal
