# Customized WhisperCPP

<div class="inm-hero">
<p class="inm-eyebrow">Experimental transcription framework</p>
<h2>Recover from output loops. Preserve the evidence.</h2>
<p>Source-clock-aware diagnostics, bounded CUDA retries and review-first transcript assembly around the whisper.cpp runtime.</p>
</div>

<div class="inm-identity">
<p><strong>Source:</strong> experimental feature branch, not production master</p>
<p><strong>Evidence:</strong> Stage 20 frozen structural benchmark, November 2026</p>
<p><strong>Release:</strong> review-only correction candidates</p>
</div>

## Demonstrated output-repetition recovery

**27/27** selected historical problem excerpts ended with no detected output repetition; **25/27** had no selected-output caption-join conflict. **85 native CUDA jobs** were evaluated. This is **one representative excerpt per recording**, not complete WAV accuracy. Human WER/CER has not yet been established.

The project addresses output-level loops, incomplete captions, cautious retry planning, source-clock provenance, and machine/human evidence separation. See [Architecture](architecture.md), [Adaptive recovery](adaptive-recovery.md), [Validation](validation.md) and [Roadmap](roadmap.md).

**Attribution:** fork of [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp), original MIT software licensing retained. The site now displays owner-authorized INM header, favicon and footer artwork as part of the institutional documentation theme. The marks remain separately owned and are not licensed as part of whisper.cpp or this fork; their appearance does not imply endorsement of provisional experimental results.
