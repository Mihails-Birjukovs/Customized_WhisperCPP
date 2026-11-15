# Getting started

## Source and environment

The adaptive Python package is on experimental source branch 442aa2a, not the fork's default master branch. The core synthetic tests require Python 3.10+ and the standard library:

~~~console
python -m unittest discover -s tests/adaptive_whisper -v
~~~

For native inference, provide a separately trusted whisper-cli executable and matching model files. Historical Windows CUDA experiments used Whisper large-v3 and Silero v6.2.0. The retry-controller input contract uses 16-kHz mono PCM16 WAV. Check backend selection, binary/model integrity and available CUDA hardware before claiming successful execution.

For upstream compilation and base whisper-cli use, consult the [preserved upstream README](https://github.com/Mihails-Birjukovs/Customized_WhisperCPP/blob/docs/stage22-modernization-20261115/UPSTREAM_README.md). These are not automatically the native experimental distribution instructions.

## Safe usage boundary

Keep original WAVs and historical SRTs unchanged; use isolated outputs and hash-verified source-clock excerpts. A recovered SRT that no longer loops is still a **provisional machine hypothesis**, not a final corrected transcript. Inspect join conflicts, verify what was spoken, and require explicit editorial approval before release.
